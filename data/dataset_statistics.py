import argparse
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from transformers import AutoTokenizer

pandarallel.initialize(progress_bar=True)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    df = pd.read_json(args.data_file)
    df = df.drop_duplicates()

    df = df.drop(columns=['long_methods_only', 'label'])
    df_tokenized = df.fillna('').parallel_applymap(lambda x: tokenizer.encode(x, add_special_tokens=False), axis=0)
    df_lens = df_tokenized.applymap(len)

    total_row_stats = df_lens.sum(axis=1).agg(['mean', 'max', 'min'])
    total_col_stats = df_lens.agg(['mean', 'max', 'min'])

    print('Total row stats:')
    print(total_row_stats)
    print('='*20)
    print('Total column stats:')
    print(total_col_stats)

    grouped = df.notnull().groupby(list(df.columns)).apply(lambda x: x.index)
    for k, v in grouped.items():
        print('='*20)
        print(list(zip(grouped.keys().names, k)))
        split = df_lens.loc[v]
        print(split.sum(axis=1).agg(['mean', 'max', 'min']))
        print('-----')
        print(split.agg(['mean', 'max', 'min']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='elinas/llama-7b-hf-transformers-4.29', help='Model tokenizer to use for statistics (default: elinas/llama-7b-hf-transformers-4.29)')
    parser.add_argument('--data_file', default='relevant_data.json', help='File containing relevant data (default: relevant_data.json)')
    args = parser.parse_args()
    main(args)
