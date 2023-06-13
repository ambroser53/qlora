import argparse
import json
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
from accelerate import infer_auto_device_map
from peft import PeftConfig, PeftModel
from utils.prompter import Prompter
from datasets import load_dataset
from torch.utils.data import DataLoader


def batch_generate(args, dataset, device, generation_config, model, prompter, tokenizer):
    dataset['train'].map(lambda x: {"data": x, "prompt":prompter.generate_prompt(x['instruction'], x['input'])})
    batch_iter = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, num_workers=4)
    for i, batch in tqdm(enumerate(batch_iter)):
        if i < args.start_from:
            continue

        batch_input = [b['prompt'] for b in batch]
        batch_data = [b['data'] for b in batch]

        input_ids = tokenizer.batch_encode(batch_input, return_tensors="pt")

        input_ids = input_ids.to(device)
        output_ids = model.generate(input_ids=input_ids, generation_config=generation_config)

        for b_data, output_id in zip(batch_data, output_ids):
            with open(args.output_file, "a+") as f:
                f.write(json.dumps({
                    "instruction": b_data['instruction'],
                    "input": b_data['input'],
                    "response": tokenizer.decode(output_id[0], skip_special_tokens=True),
                    "label": b_data["output"]
                }) + '\n')


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

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.lora_weights)
    print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)
    print("finetune model is_loaded_in_4bit: ", model.is_loaded_in_4bit)
    print(model.hf_device_map)

    if not args.bits == 4 and not args.bits == 8:
        model.half()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32" and args.compile:
        model = torch.compile(model)

    if args.batch_size > 1:
        batch_generate(args, dataset, device, generation_config, model, prompter, tokenizer)
    else:
        normal_generate(args, dataset, device, generation_config, model, prompter, tokenizer)


def normal_generate(args, dataset, device, generation_config, model, prompter, tokenizer):
    for i, data in tqdm(enumerate(dataset['train'])):
        if i < args.start_from:
            continue
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


def main_one(args):
    instruction = args.instruction
    input = args.input
    temperature = 0.6
    top_p = 0.5
    top_k = 40
    num_beams = 4
    max_new_tokens = args.max_new_tokens

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
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

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.lora_weights)
    print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)
    print("finetune model is_loaded_in_4bit: ", model.is_loaded_in_4bit)
    print(model.hf_device_map)

    if not args.bits == 4 and not args.bits == 8:
        model.half()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32" and args.compile:
        model = torch.compile(model)

    prompter = Prompter(args.prompt_template)
    prompt = prompter.generate_prompt(instruction, input)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(input_ids)
    input_ids = input_ids.to(device)
    output_ids = model.generate(input_ids=input_ids, generation_config=generation_config)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


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
        args.output_file = args.lora_weights + ".jsonl"

    if args.dataset is None:
        main_one(args)
    else:
        main(args)
