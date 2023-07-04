import argparse
import pandas as pd
from glob import glob
from sklearn.model_selection import KFold
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import PeftConfig, PeftModel, PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys
from qlora import make_data_module
from tqdm import tqdm
import numpy as np

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

    args.do_predict = False
    args.do_eval = False
    
    for review in tqdm(reviews):
        args.dataset = review
        dataset_dict = make_data_module(tokenizer, args)

        review_y_pred = []
        review_y_true = []

        for train_index, test_index in kf.split(dataset_dict['train_dataset']):
            train_loader = DataLoader(dataset_dict['train_dataset'].select(train_index), batch_size=args.train_batch_size, shuffle=True, collate_fn=dataset_dict['data_collator'])
            test_loader = DataLoader(dataset_dict['train_dataset'].select(test_index), batch_size=args.eval_batch_size, shuffle=True, collate_fn=dataset_dict['data_collator'])


            if args.do_train:
                print("pre-peft cuda usage: " + str(torch.cuda.mem_get_info()))
                peft_model = get_peft_model(base_model, prompt_config)
                print("New soft prompts:")
                print(peft_model.print_trainable_parameters())
                print("pre-train cuda usage: " + str(torch.cuda.mem_get_info()))

                optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.lr)
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=(len(train_loader)),
                )

                peft_model.train()
                dataset_dict['data_collator'].eval(False)
                for batch in train_loader:
                    input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                    if input_ids is None:
                        raise ValueError("input_ids is None")

                    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                model = peft_model
            else:
                model = base_model

            print("pre-eval cuda usage: "+str(torch.cuda.mem_get_info()))

            with torch.no_grad():
                model.eval()
                dataset_dict['data_collator'].eval(True)
                for batch in test_loader:
                    input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels']

                    print(attention_mask)
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.target_max_len,
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_beams=args.num_beams,
                    )

                    # compute probability of each generated token
                    if args.num_beams == 1:
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

                    decoded_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                    decoded_labels = labels #tokenizer.batch_decode([[t for t in l if t != -100] for l in labels], skip_special_tokens=True)
                    review_y_pred.extend([output.split()[0] for output in decoded_outputs])
                    review_y_true.extend([label.split()[0] for label in decoded_labels])

            ## swap out the prompt

        y_pred.extend(review_y_pred)
        y_true.extend(review_y_true)

        with open(f'{review.split(".")[0]}_prompt_results.txt', 'w+') as f:
            f.write(metrics.classification_report(review_y_true, review_y_pred))

    with open(f'{review.split(".")[0]}_prompt_results.txt', 'w+') as f:
        f.write(metrics.classification_report(y_true, y_pred))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to lora adapter weights merged into model or model path')
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
    args = parser.parse_args()
    args.do_predict = False
    args.do_eval = False
    args.do_train = True
    args.predict_with_generate = False
    args.train_on_source = False
    args.max_train_samples = None
    args.group_by_length = True
    main(args)