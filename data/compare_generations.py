import pandas as pd
import argparse
import evaluate
import json
from sklearn import metrics
import warnings
import math
import re
import os


def calculate_bleu_score(generations, references):
    bleu = evaluate.load('bleu')
    return bleu.compute(predictions=generations, references=references)


def calculate_rouge_score(generations, references):
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=generations, references=references)


def warn(*args, **kwargs):
    pass


def main(args):
    warnings.warn = warn

    results = pd.read_json(args.results_path, lines=args.lines)

    dataset_exc_rea = pd.read_json(args.dataset_path)

    print(args.instruction_subset)
    exc_rea = results[results['instruction'].str.contains(args.instruction_subset)]
    exc_rea = exc_rea.transform(lambda x: x.str.strip())

    merged = pd.merge(dataset_exc_rea, exc_rea, on=['instruction', 'input'], how='left')
    merged = merged.dropna(subset=['response'])

    print('evaluating: ', len(merged))
    bleu_score = calculate_bleu_score(merged['response'], merged['output'])
    rouge_score = calculate_rouge_score(merged['response'], merged['output'])
    print("BLEU score: {}".format(bleu_score))
    print("ROUGE score: {}".format(rouge_score))

    if args.instruction_subset == "Provide an explanation for why the abstract was excluded":
        extension = '_exclusion_reasons.json'
    elif args.instruction_subset == "what is the study's":
        extension = '_pio_extraction.json'
    elif args.instruction_subset == "what is the study's I":
        extension = '_intervention_extraction.json'
    elif args.instruction_subset == "what is the study's P":
        extension = '_population_extraction.json'
    elif args.instruction_subset == "what is the study's O":
        extension = '_outcome_extraction.json'
    else:
        raise ValueError("Instruction subset not recognised")

    merged.to_json(os.path.splitext(args.results_path)[0]+extension, indent=4, orient='records')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True, help='Path to eval results')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to eval dataset')
    parser.add_argument('--instruction_subset', type=str, choices=["Provide an explanation for why the abstract was excluded",
                                                                   "what is the study's",
                                                                   "what is the study's I",
                                                                   "what is the study's P",
                                                                   "what is the study's O"],
                        default="Provide an explanation for why the abstract was excluded", help='Instruction phrase to filter on')
    parser.add_argument('--label_field_name', default='label', type=str, help='Name of label field in dataset')
    parser.add_argument('--lines', action='store_true', help='Whether results are stored as lines (jsonl)')
    args = parser.parse_args()
    main(args)
