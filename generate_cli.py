import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional, Any, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer, \
    PreTrainedTokenizerBase
from peft import PeftConfig, PeftModel
from transformers.utils import PaddingStrategy

from utils.prompter import Prompter
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
import re
import numpy as np

DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class DataCollatorForCausalLM:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["input_ids"]))
                attn_remainder = [0] * (max_label_length - len(feature["attention_mask"]))
                if isinstance(feature["input_ids"], list):
                    feature["input_ids"] = (
                        feature["input_ids"] + remainder if padding_side == "right" else remainder + feature["input_ids"]
                    )
                    feature["attention_mask"] = (
                        feature["attention_mask"] + attn_remainder if padding_side == "right" else attn_remainder + feature["attention_mask"]
                    )
                elif padding_side == "right":
                    feature["input_ids"] = np.concatenate([feature["input_ids"], remainder]).astype(np.int64)
                    feature['attention_mask'] = np.concatenate([feature['attention_mask'], np.zeros_like(remainder)]).astype(np.int64)
                else:
                    feature["input_ids"] = np.concatenate([remainder, feature["input_ids"]]).astype(np.int64)
                    feature['attention_mask'] = np.concatenate([np.zeros_like(remainder), feature['attention_mask']]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features


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


def batch_generate(args, dataset, device, generation_config, model, prompter, tokenizer):
    alpaca_pattern = re.compile(
        '.*(### Instruction:\s+(?P<instruction>.+)\s+### Input:\s+(?P<input>.+)\s+### Response:\s+(?P<response>.*))',
        re.DOTALL)
    wizard_pattern = re.compile('.*(USER: (?P<instruction>.*)\n{2})(?P<input>.*)(\s{2} ASSISTANT: (?P<response>.*))', re.DOTALL)

    original_columns = dataset['train'].column_names
    dataset['train'] = dataset['train'].map(
        lambda x: tokenizer(
            prompter.generate_prompt(x['instruction'], x['input']),
            truncation=True,
            padding=False),
        remove_columns=original_columns).select(range(args.start_from, len(dataset['train'])))

    tokenizer.padding_side = "left"
    collator = DataCollatorForCausalLM(tokenizer, return_tensors="pt", padding=True)
    batch_iter = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    for batch in tqdm(batch_iter, total=len(batch_iter)):
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    generation_config=generation_config)

        decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        full_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

        with open(args.output_file, "a+") as f:
            for output, full in zip(decoded_outputs, full_outputs):

                if args.prompt_template == 'alpaca':
                    o = alpaca_pattern.match(output).groupdict()
                    o['full'] = full
                    f.write(json.dumps(o) + '\n')
                elif args.prompt_template == 'wizard13b':
                    o = wizard_pattern.match(output).groupdict()
                    o['full'] = full
                    f.write(json.dumps(o) + '\n')


def main(args):
    dataset = load_dataset("json", data_files=args.dataset)
    prompter = Prompter(args.prompt_template)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=0.5,
        top_k=40,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.dont_sample,
    )

    print(args.lora_weights)
    peft_config = PeftConfig.from_pretrained(args.lora_weights)
    print("peft_config: ", peft_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
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

    if args.llama_specifically:
        tokenizer = LlamaTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.lora_weights)
    print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)
    print("finetune model is_loaded_in_4bit: ", model.is_loaded_in_4bit)
    print(model.hf_device_map)

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
            model=model,
        )

    if not args.bits == 4 and not args.bits == 8:
        model.half()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32" and args.compile:
        model = torch.compile(model)

    batch_generate(args, dataset, device, generation_config, model, prompter, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--lora_weights", type=str, default="tloen/alpaca-lora-7b")
    parser.add_argument("--prompt_template", type=str, default="alpaca")
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--double_quant", type=bool, default=True)
    parser.add_argument("--quant_type", type=str, default="nf4")  # either fp4 or nf4
    parser.add_argument("--output_file", type=str, default="eval.jsonl")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--llama_specifically", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dont_sample", action="store_true")
    args = parser.parse_args()

    if args.output_file == "eval.jsonl":
        args.output_file = args.lora_weights + "_eval.jsonl"

    main(args)