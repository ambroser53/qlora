import argparse
import json
import os.path

import pandas as pd
from glob import glob
from sklearn.model_selection import KFold
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    HfArgumentParser, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import pipeline
from peft import PeftConfig, PeftModel, PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model, \
    prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional
import re
from datasets import load_dataset, load_from_disk
from qlora import make_data_module
from utils.prompter import Prompter
import json

IGNORE_INDEX = -100
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_PAD_TOKEN = "[PAD]"


def main(args):
    prompter = Prompter(args.prompt_template)

    out_pattern = re.compile(
        '.*(Abstract:\s+(?P<abstract>.+)\s+\\n Objectives:\s+(?P<obj>.+)\s+Selection Criteria:\s+(?P<sel_cri>.*))',
        re.DOTALL)
    reviews = glob(f'{args.data_dir}/*.json')
    if len(reviews) == 0:
        raise ValueError(f'No reviews found in {args.data_dir}')

    tokenizer_kwargs = {'max_length': args.max_token_len, 'truncation': True, 'return_tensors': 'pt', 'model_max_length': args.max_token_len}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    oracle = pipeline(model=args.model_name_or_path, task="zero-shot-classification", tokenizer=tokenizer)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=0)
    y_pred = []
    y_true = []

    for review in tqdm(reviews):
        args.dataset = review

        with open(review, 'r') as f:
            dataset = json.load(f)

        review_y_pred = []
        review_y_true = []

        p_bar = tqdm(total=len(dataset))

        print(dataset)

        for train_index, test_index in kf.split(dataset):

            test_set = [dataset[i] for i in test_index]

            for example in test_set:
                match = out_pattern.match(example['input'])
                if match is None:
                    continue
                if args.prompt_template is None:
                    text = match.groupdict()
                    if 'abstract' in text and 'obj' in text and 'sel_cri' in text:
                        text = text['obj'] + text['sel_cri'] + text['abstract']
                    elif 'abstract' in text and 'obj' in text:
                        text = text['obj'] + text['abstract']
                    elif 'abstract' in text and 'sel_cri' in text:
                        text = text['sel_cri'] + text['abstract']
                    elif 'abstract' in text:
                        text = text['abstract']
                    else:
                        continue

                    if args.max_token_len > 624:
                        text = "Should this abstract be included or excluded from the review with the following objectives and selection criteria? " + text
                else:
                    text = prompter.generate_prompt(example['instruction'], example['input'])

                label = example['label']
                response = oracle(text, candidate_labels=["Included", "Excluded"], **tokenizer_kwargs)

                response = {l: v for l, v in zip(response['labels'], response['scores'])}
                response = sorted(response.items(), key=lambda x: x[1], reverse=True)[0][0]
                print("responses:")
                print(response)
                print("labels:")
                print(label)

                review_y_pred.append(response)
                review_y_true.append(label)
                print(metrics.classification_report(review_y_true, review_y_pred))

                p_bar.update(1)

        y_pred.extend(review_y_pred)
        y_true.extend(review_y_true)

        results_output_dir = f'{review.split(".")[0]}_{args.model_name_or_path.split("/")[1]}_zeroshot_results.txt'

        with open(results_output_dir, 'w+') as f:
            f.write(metrics.classification_report(review_y_true, review_y_pred))
            f.write(str(metrics.confusion_matrix(review_y_true, review_y_pred)))

    complete_results_dir = f'review_{args.model_name_or_path.split("/")[1]}_zeroshot_results_complete.txt'
    complete_results_dir = os.path.join(args.data_dir, complete_results_dir)
    with open(complete_results_dir, 'w+') as f:
        f.write(metrics.classification_report(y_true, y_pred))
        f.write(str(metrics.confusion_matrix(y_true, y_pred)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to lora adapter weights merged into model or model path')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing reviews')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--max_token_len", type=int, default=512)
    parser.add_argument("--prompt_template", type=str, default=None, help="Whether using prompted model then use specified prompt template otherwise use pushing")
    args = parser.parse_args()

    main(args)
