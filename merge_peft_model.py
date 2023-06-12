from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False,
        device_map='cpu'
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = PeftModel.from_pretrained(model, args.lora_weights)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir + '/merged')
    tokenizer.save_pretrained(args.output_dir + '/merged')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='elinas/llama-13b-hf-transformers-4.29', help='Name or path to base model')
    parser.add_argument('--lora_weights', type=str, default=None, help='Output directory')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    args = parser.parse_args()
    main(args)