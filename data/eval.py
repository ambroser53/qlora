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
        results = pd.read_json(args.results_path, lines=True)
    except:
        with open(args.data_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except:
                    raise ValueError(f'Error on line {i}')
                
        raise ValueError(f'Error reading {args.data_path}')

    label_mapping = {'Include': 'Included', 'Exclude': 'Excluded', 'Insufficient': 'Included'}
    dataset_inc_exc = pd.read_json(args.dataset_path)
    dataset_inc_exc[args.label_field_name] = dataset_inc_exc[args.label_field_name].str.split().str[0]
    dataset_inc_exc[args.label_field_name] = dataset_inc_exc[args.label_field_name].transform(lambda x: label_mapping[x] if x in label_mapping else x)

    inc_exc = results[results['instruction'].str.contains('should the study be included or excluded?')]
    inc_exc = inc_exc.transform(lambda x: x.str.strip())
    inc_exc['prediction'] = inc_exc['response'].str.split().str[0]

    merged = pd.merge(dataset_inc_exc, inc_exc, on=['instruction', 'input'], how='left')
    
    print(metrics.classification_report(merged[args.label_field_name], merged['prediction'], digits=2))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True, help='Path to eval results')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to eval dataset')
    parser.add_argument('--label_field_name', default='label', type=str, help='Name of label field in dataset')
    args = parser.parse_args()
    main(args)