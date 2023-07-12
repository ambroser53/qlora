import argparse
import json
import os.path

import pandas as pd
from glob import glob
from sklearn.model_selection import KFold
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    HfArgumentParser, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
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

IGNORE_INDEX = -100
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={
        "help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={
        "help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={
        "help": 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False,
                                        metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={
        "help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True,
                                         metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={
        "help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10,
                               metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={
        "help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40,
                                  metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})


def main(args):
    def preprocess_function(examples):
        return tokenizer(re.findall('(?<=Abstract: )(.*?)(?=\s*\\n Objectives:)', examples['input'])[0], truncation=True, padding='max_length', max_length=512)

    reviews = glob(f'{args.data_dir}/*.json')
    if len(reviews) == 0:
        raise ValueError(f'No reviews found in {args.data_dir}')

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    if not args.do_train:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            return_dict=True,
            torch_dtype=compute_dtype,
            device_map={'': 0},
            label2id={"Included": 0, "Excluded": 1},
            id2label={0: "Included", 1: "Excluded"},
            num_labels=2,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.model_max_length = 512

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=0)
    y_pred = []
    y_true = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for review in tqdm(reviews):
        args.dataset = review
        data_module = {
            "train_dataset": load_dataset("json", data_files=args.dataset).map(preprocess_function)['train'],
            "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        }
        dataset = data_module['train_dataset']

        review_y_pred = []
        review_y_true = []

        p_bar = tqdm(total=len(data_module['train_dataset']))

        print(data_module['train_dataset'])

        for train_index, test_index in kf.split(data_module['train_dataset']):

            if args.do_train:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name_or_path,
                    return_dict=True,
                    torch_dtype=compute_dtype,
                    device_map={'': 0},
                    label2id={"Included": 0, "Excluded": 1},
                    id2label={0: "Included", 1: "Excluded"},
                    num_labels=2,
                )

                data_module['train_dataset'] = dataset.select(train_index)

                hfparser = HfArgumentParser((
                    TrainingArguments
                ))
                training_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

                model.train()
                training_args.max_steps = (len(
                    data_module['train_dataset']) * args.num_train_epochs) // args.train_batch_size
                training_args.per_device_train_batch_size = args.train_batch_size

                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
                )

                trainer.train()

            print("pre-eval cuda usage: " + str(torch.cuda.mem_get_info()))

            with torch.no_grad():
                model.eval()

                test_set = dataset.select(test_index)

                test_set_labels = test_set.map(
                    lambda x: x,
                    remove_columns=[c for c in test_set.column_names if c != 'label']
                )
                original_columns = test_set.column_names
                test_set = test_set.map(
                    lambda x: tokenizer(
                        x['input'],
                        truncation=True,
                        padding=False),
                    remove_columns=original_columns)

                collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True, max_length=512)
                batch_iter = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
                label_iter = DataLoader(test_set_labels, batch_size=args.eval_batch_size, shuffle=False)

                for batch, labels in zip(batch_iter, label_iter):
                    labels = labels['label']
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

                    predicted_class_ids = [x.item() for x in logits.argmax(dim=-1)]

                    responses = [model.config.id2label[class_id] for class_id in
                                 predicted_class_ids]
                    print("responses:")
                    print(responses)
                    print("labels:")
                    print(labels)

                    #print_sequence_response(model, tokenizer, input_ids, outputs, args.num_beams)

                    review_y_pred.extend(responses)
                    review_y_true.extend([label.split()[0] for label in labels])
                    print(metrics.classification_report(review_y_true, review_y_pred))

                    p_bar.update(args.eval_batch_size)

            if args.do_train:
                del model

        y_pred.extend(review_y_pred)
        y_true.extend(review_y_true)

        results_output_dir = f'{review.split(".")[0]}_{args.model_name_or_path.split("/")[1]}_results.txt' if not args.do_train else f'{review.split(".")[0]}_{args.model_name_or_path.split("/")[1]}_results_train.txt'

        with open(results_output_dir, 'w+') as f:
            f.write(metrics.classification_report(review_y_true, review_y_pred))

    complete_results_dir = f'review_{args.model_name_or_path.split("/")[1]}_results_complete.txt' if not args.do_train else f'review_{args.model_name_or_path.split("/")[1]}_results_complete_train.txt'
    with open(complete_results_dir, 'w+') as f:
        f.write(metrics.classification_report(y_true, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to lora adapter weights merged into model or model path')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing reviews')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument("--prompt_template", type=str, default="alpaca")
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--double_quant", type=bool, default=True)
    parser.add_argument("--quant_type", type=str, default="nf4")
    parser.add_argument("--custom_eval_dir", type=bool, default=False)
    parser.add_argument("--load_from_disk", type=bool, default=False)
    parser.add_argument("--source_max_len", type=int, default=1024)
    parser.add_argument("--target_max_len", type=int, default=384)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--output_file", type=str, default="eval.jsonl")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()

    if args.output_file == "eval.jsonl" and os.path.exists(args.model_name_or_path):
        args.output_file = args.model_name_or_path + "_eval.jsonl"

    args.do_predict = False
    args.do_eval = False
    args.predict_with_generate = False
    args.train_on_source = False
    args.max_train_samples = None
    args.group_by_length = True
    main(args)
