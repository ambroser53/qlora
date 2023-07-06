import argparse
import json
import os.path

import pandas as pd
from glob import glob
from sklearn.model_selection import KFold
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup, \
    HfArgumentParser, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
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


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main(args):
    reviews = glob(f'{args.data_dir}/*.json')
    if len(reviews) == 0:
        raise ValueError(f'No reviews found in {args.data_dir}')

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        return_dict=True,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        torch_dtype=compute_dtype,
        device_map={'': 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

        if hasattr(base_model, "enable_input_require_grads"):
            print("Enabling input require grads")
            base_model.enable_input_require_grads()
        else:
            print("Enabling input require grads via forward hook")

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if tokenizer.bos_token is None:
        tokenizer.bos_token = DEFAULT_BOS_TOKEN
    if tokenizer.eos_token is None:
        tokenizer.eos_token = DEFAULT_EOS_TOKEN,
    if tokenizer.unk_token is None:
        tokenizer.unk_token = DEFAULT_UNK_TOKEN,

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=base_model,
        )

    tokenizer.padding_side = "left"

    if not args.bits == 4 and not args.bits == 8:
        base_model.half()

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # Prompt tuning bits
    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=19,
        prompt_tuning_init_text="Below is an instruction that describes a task, paired with an input that provides further context.",
        tokenizer_name_or_path=args.model_name_or_path,
        base_model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
    )

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=0)
    y_pred = []
    y_true = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_pattern = re.compile(
        '.*(### Instruction:\s+(?P<instruction>.+)\s+### Input:\s+(?P<input>.+)\s+### Response:\s+(?P<response>.*))',
        re.DOTALL)

    for review in tqdm(reviews):
        args.dataset = review
        temp_do_train = args.do_train
        args.do_train = True
        data_module = make_data_module(tokenizer, args)
        args.do_train = temp_do_train
        dataset = data_module['train_dataset']

        review_y_pred = []
        review_y_true = []

        p_bar = tqdm(total=len(data_module['train_dataset']))

        for train_index, test_index in kf.split(data_module['train_dataset']):

            if args.do_train:
                data_module['train_dataset'] = dataset.select(train_index)

                hfparser = HfArgumentParser((
                    TrainingArguments
                ))
                training_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

                print("pre-peft cuda usage: " + str(torch.cuda.mem_get_info()))
                model = get_peft_model(base_model, prompt_config)
                print("New soft prompts:")
                print(model.print_trainable_parameters())
                print("pre-train cuda usage: " + str(torch.cuda.mem_get_info()))

                model.train()
                data_module['data_collator'].eval(False)
                training_args.max_steps = (len(
                    data_module['train_dataset']) * args.num_train_epochs) // args.train_batch_size
                training_args.per_device_train_batch_size = args.train_batch_size

                trainer = Seq2SeqTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
                )

                trainer.train()
            else:
                model = base_model

            print("pre-eval cuda usage: " + str(torch.cuda.mem_get_info()))

            with torch.no_grad():
                model.eval()
                data_module['data_collator'].eval(True)

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

                collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
                batch_iter = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
                label_iter = DataLoader(test_set_labels, batch_size=args.eval_batch_size, shuffle=False)

                for batch, labels in zip(batch_iter, label_iter):
                    labels = labels['label']
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                             max_new_tokens=args.target_max_len,
                                             return_dict_in_generate=True,
                                             output_scores=True,
                                             num_beams=args.num_beams, )

                    decoded_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

                    responses = [out_pattern.match(output).groupdict()["response"].split()[0] for output in
                                 decoded_outputs]
                    print("responses:")
                    print(responses)
                    print("labels:")
                    print(labels)

                    print_sequence_response(model, tokenizer, input_ids, outputs, args.num_beams)

                    review_y_pred.extend(responses)
                    review_y_true.extend([label.split()[0] for label in labels])
                    print(metrics.classification_report(review_y_true, review_y_pred))

                    p_bar.update(args.eval_batch_size)

        y_pred.extend(review_y_pred)
        y_true.extend(review_y_true)

        with open(f'{review.split(".")[0]}_prompt_results.txt', 'w+') as f:
            f.write(metrics.classification_report(review_y_true, review_y_pred))

    with open(f'{review.split(".")[0]}_prompt_results.txt', 'w+') as f:
        f.write(metrics.classification_report(y_true, y_pred))


def print_sequence_response(model, tokenizer, input_ids, outputs, num_beams):
    # compute probability of each generated token
    if num_beams == 1:
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )[0]
    else:
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        ).cpu()

        # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        # Tip: set `normalize_logits=True` to recompute the scores from the normalized logits.

        output_length = input_ids.shape[1] + np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        length_penalty = model.generation_config.length_penalty
        reconstructed_scores = transition_scores.sum(axis=1) / (output_length ** length_penalty)
        print(np.allclose(outputs.sequences_scores.cpu(), reconstructed_scores))
        transition_scores = reconstructed_scores
    input_length = input_ids.shape[1]
    input_toks = input_ids[:, input_length - 2:]
    generated_tokens = outputs.sequences[:, input_length - 2:input_length + 5]
    i = -2
    print(tokenizer.decode(input_ids[0]))
    for tok, score in zip(generated_tokens[0], transition_scores[:7]):
        # | token | token string | probability
        print(
            f"| {i} | {input_toks[i + 2][0] if i < 0 else None} | {tok:5d} | {tokenizer.decode(tok):8s} | {np.exp(score.cpu().numpy()):.2%}")
        i += 1


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
