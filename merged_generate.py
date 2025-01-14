import argparse
import json
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
from utils.prompter import Prompter
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
import re

DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_PAD_TOKEN = "[PAD]"


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
    out_pattern = re.compile(
        '.*(### Instruction:\s+(?P<instruction>.+)\s+### Input:\s+(?P<input>.+)\s+### Response:\s+(?P<response>.*))',
        re.DOTALL)

    original_columns = dataset['train'].column_names
    dataset['train'] = dataset['train'].map(
        lambda x: tokenizer(
            prompter.generate_prompt(x['instruction'], x['input']),
            truncation=True,
            padding=False),
        remove_columns=original_columns).select(range(args.start_from, len(dataset['train'])))

    collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
    batch_iter = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    for batch in tqdm(batch_iter, total=len(batch_iter)):
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    generation_config=generation_config)

        decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        with open(args.output_file, "a+") as f:
            for output in decoded_outputs:
                f.write(json.dumps(out_pattern.match(output).groupdict()) + '\n')


def main(args):
    dataset = load_dataset("json", data_files=args.dataset)
    prompter = Prompter(args.prompt_template)
    temperature = 0.6
    top_p = 0.5
    top_k = 40
    num_beams = args.num_beams
    max_new_tokens = args.max_new_tokens

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)
    print("finetune model is_loaded_in_4bit: ", model.is_loaded_in_4bit)
    print(model.hf_device_map)

    if not args.bits == 4 and not args.bits == 8:
        model.half()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32" and args.compile:
        model = torch.compile(model)

    for data in tqdm(dataset['train']):
        instruction = data['instruction']
        input = data['input']
        prompt = prompter.generate_prompt(instruction, input)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(device)
        output_ids = model.generate(input_ids=input_ids, generation_config=generation_config)

        with open(args.output_file, "a+") as f:
            f.write(json.dumps({
                "instruction": instruction,
                "input": input,
                "response": tokenizer.decode(output_ids[0], skip_special_tokens=True),
                "label": data["output"]
            }) + '\n')



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
    args = parser.parse_args()

    if args.output_file == "eval.jsonl":
        args.output_file = args.model_name_or_path + "guanaco65B_eval.jsonl"

    main(args)
