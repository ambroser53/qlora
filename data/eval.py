import pandas as pd
import argparse
import json
from sklearn import metrics
import warnings

def warn(*args, **kwargs):
    pass

def main(args):
    warnings.warn = warn

    try:
        df = pd.read_json(args.data_path, lines=True)
    except:
        with open(args.data_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except:
                    raise ValueError(f'Error on line {i}')
                
        raise ValueError(f'Error reading {args.data_path}')

        
    
    inc_exc = df[df['instruction'].str.contains('should the study be included or excluded?')]

    if args.model_type == 'llama':
        inc_exc['prediction'] = inc_exc['response'].str.split('### Response:\n').str[1].str.strip().str.split('</s>').str[0].str.split().str[0]
    elif args.model_type == 'gpt':
        inc_exc['prediction'] = inc_exc['response'].str.split('# # # Response:').str[1].str.strip().str.split('</s>').str[0].str.split().str[0]
    else:
        raise ValueError(f'Unknown model type {args.model_type}')
    
    label_to_idx = {'Included': 1, 'Excluded': 0}

    inc_exc['label_idx'] = inc_exc['label'].str.split().str[0].apply(lambda x: label_to_idx[x])
    deduped = inc_exc[~inc_exc[['input', 'instruction', 'response']].duplicated(keep='last')]

    print(f'{len(inc_exc) - len(deduped)} duplicates removed')

    filtered_labels = deduped[deduped['prediction'].isin(['Included', 'Excluded'])]

    print(f'{len(inc_exc) - len(filtered_labels)} non-included/excluded labels removed')

    print('='*40)
    print('Full Dataset Results\n')
    print(metrics.classification_report(filtered_labels['label_idx'], filtered_labels['prediction'].apply(lambda x: label_to_idx[x])), end='\n\n')

    print('='*40)
    print('Subset Results\n')
    filtered_labels[['abstract', 'review']] = filtered_labels['input'].apply(lambda x: x.split('Objectives: ')).apply(pd.Series)
    filtered_df = filtered_labels.groupby('review').filter(lambda x: x['instruction'].count() > 100)
    subset = filtered_labels[filtered_labels['input'].isin(filtered_df['input'])]
    print(metrics.classification_report(subset['label_idx'], subset['prediction'].apply(lambda x: label_to_idx[x])), end='\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to eval results')
    parser.add_argument('--model_type', type=str, required=True, choices=['gpt', 'llama'], help='Model type')
    args = parser.parse_args()
    main(args)
