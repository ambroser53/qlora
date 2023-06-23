import argparse
import pandas as pd
from glob import glob
from sklearn.model_selection import KFold
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
import torch
from torch.utils.data import DataLoader
import sys
from qlora import make_data_module

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


def process_text(text):
    pass

def main(args):
    reviews = glob(f'{args.data_dir}/*.json')
    if len(reviews) == 0:
        raise ValueError(f'No reviews found in {args.data_dir}')

    peft_config = PeftConfig.from_pretrained(args.lora_weights)

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

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=0)
    y_pred = []
    y_true = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.do_predict = False
    args.do_eval = False
    
    for review in reviews:
        args.dataset = review
        dataset_dict = make_data_module(tokenizer, args)

        review_y_pred = []
        review_y_true = []

        for train_index, test_index in kf.split(dataset_dict['train_dataset']):
            train_loader = DataLoader(dataset_dict['train_dataset'].select(train_index), batch_size=8, shuffle=True)
            test_loader = DataLoader(dataset_dict['train_dataset'].select(test_index), batch_size=8, shuffle=True)

            model.train()
            for batch in train_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                model.optimizer.step()
                model.scheduler.step()
                model.zero_grad()

            model.eval()
            for batch in test_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                review_y_pred.extend([output.split()[0] for output in decoded_outputs])
                review_y_true.extend([label.split()[0] for label in decoded_labels])

        y_pred.extend(review_y_pred)
        y_true.extend(review_y_true)

        with open(f'{review.split(".")[0]}_prompt_results.txt', 'w+') as f:
            f.write(metrics.classification_report(review_y_true, review_y_pred))

    with open(f'{review.split(".")[0]}_prompt_results.txt', 'w+') as f:
        f.write(metrics.classification_report(y_true, y_pred))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_weights', type=str, required=True, help='Path to lora adapter weights')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing reviews')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument("--prompt_template", type=str, default="alpaca")
    args = parser.parse_args()
    main(args)