import argparse
import json
import os
import sys
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer, pipeline
from peft import PeftConfig, PeftModel
from utils.prompter import Prompter
import re
from sklearn import metrics


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


def generate(args, dataset, oracle, prompter):
    out_pattern = re.compile(
        '.*(Abstract:\s+(?P<abstract>.+)\s+\n Objectives:\s+(?P<obj>.+)\s+Selection Criteria:\s+(?P<sel_cri>.*))',
        re.DOTALL)

    dataset = [dataset[i] for i in range(args.start_from, len(dataset))]

    for x in dataset:
        x['prompt'] = prompter.generate_prompt(x['instruction'], x['input'])

    y_pred = []
    y_true = []

    for example in tqdm(dataset, total=len(dataset)):
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
                text = "Should this abstract be included in the review? " + text
        else:
            text = prompter.generate_prompt(example['instruction'], example['input'])

        response = oracle(text, candidate_labels=["Included", "Excluded"])
        example['response'] = response
        
        with open(args.output_file, "a+") as f:
            f.write(json.dumps(example) + '\n')

        y_pred.append(response)
        y_true.append(example[args.label_field_name])

    results_output_dir = os.path.join(os.path.dirname(args.output_dir), "pipeline_results.txt")
    with open(results_output_dir, 'w+') as f:
        f.write(metrics.classification_report(y_true, y_pred))
        f.write(str(metrics.confusion_matrix(y_true, y_pred)))


def main(args):
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    prompter = Prompter(args.prompt_template)

    if args.lora_weights is not None:
        print(args.lora_weights)
        peft_config = PeftConfig.from_pretrained(args.lora_weights)
        print("peft_config: ", peft_config)
        base_model_name_or_path = peft_config.base_model_name_or_path
    elif args.model_name_or_path is not None:
        base_model_name_or_path = args.model_name_or_path
    else:
        raise ValueError("Either --lora_weights or --model_name_or_path must be specified.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
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

    tokenizer_kwargs = {'max_length': args.max_token_len, 'truncation': True, 'return_tensors': 'pt',
                        'model_max_length': args.max_token_len}
    if args.llama_specifically:
        tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, **tokenizer_kwargs)

    if args.lora_weights is not None:
        model = PeftModel.from_pretrained(base_model, args.lora_weights)
    else:
        model = base_model

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

    oracle = pipeline(model=model, device=device, task="zero-shot-classification", tokenizer=tokenizer)

    generate(args, dataset, oracle, prompter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, default=None)
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
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument('--label_field_name', default='label', type=str, help='Name of label field in dataset')
    args = parser.parse_args()

    if args.wandb_project is not None and args.wandb_entity is not None:
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    if args.output_file == "eval.jsonl" and args.lora_weights is not None:
        args.output_file = args.lora_weights + "_eval.jsonl"
    elif args.output_file == "eval.jsonl" and args.model_name_or_path is not None:
        args.output_file = args.model_name_or_path + "_eval.jsonl"

    main(args)
