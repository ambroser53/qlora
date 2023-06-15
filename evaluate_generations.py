import evaluate
import argparse
import pandas as pd
import json


def calculate_bleu_score(generations, references):
    bleu = evaluate.load('bleu')
    return bleu.compute(predictions=generations, references=references)


def calculate_rouge_score(generations, references):
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=generations, references=references)


def main_jsonl(args):
    exc_generations = []
    exc_references = []
    pio_generations = []
    pio_references = []
    wrongful_exclusions = []
    i = 0

    with open(args.dataset) as f:
        for line in f:
            try:
                doc = json.loads(line)
                if doc[args.reference_column].startswith('Excluded'):
                    exc_generations.append(doc[args.generation_column])
                    exc_references.append(doc[args.reference_column])
                elif doc['label'].startswith(('Population', 'Intervention', 'Outcome')):
                    pio_generations.append(doc[args.generation_column])
                    pio_references.append(doc[args.reference_column])

                if args.get_wrong_exclusions:
                    if args.model_type == 'gpt' and \
                            doc[args.reference_column].split("# # # Response:")[1].startswith('Included') and \
                            doc[args.generation_column].startswith('Excluded'):
                        wrongful_exclusions.append(doc)
                    elif args.model_type == 'llama' and \
                            doc[args.reference_column].split("### Response:\n")[1].startswith('Included') and \
                            doc[args.generation_column].startswith('Excluded'):
                        wrongful_exclusions.append(doc)

            except ValueError:
                print('error on line', i)
            i += 1

    if args.model_type == 'gpt':
        exc_generations = [generation.split("# # # Response:")[1] for generation in exc_generations]
        pio_generations = [generation.split("# # # Response:")[1] for generation in pio_generations]
    elif args.model_type == 'llama':
        exc_generations = [generation.split("### Response:\n")[1] for generation in exc_generations]
        pio_generations = [generation.split("### Response:\n")[1] for generation in pio_generations]

    # save wrongful exclusions
    with open(args.dataset + 'wrongful_exclusions.jsonl', 'w') as f:
        for doc in wrongful_exclusions:
            f.write(json.dumps(doc) + '\n')

    print('wrongful exclusions: ', len(wrongful_exclusions))

    print('evaluating exclusion reasons: ', len(exc_generations))
    bleu_score = calculate_bleu_score(exc_generations, exc_references)
    rouge_score = calculate_rouge_score(exc_generations, exc_references)
    print("BLEU score: {}".format(bleu_score))
    print("ROUGE score: {}".format(rouge_score))

    print('evaluating pio reasons: ', len(pio_generations))
    bleu_score = calculate_bleu_score(pio_generations, pio_references)
    rouge_score = calculate_rouge_score(pio_generations, pio_references)
    print("BLEU score: {}".format(bleu_score))
    print("ROUGE score: {}".format(rouge_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, default=None)
    parser.add_argument("--reference_column", type=str, default="label")
    parser.add_argument("--generation_column", type=str, default="response")
    parser.add_argument('--model_type', type=str, required=True, choices=['gpt', 'llama'], help='Model type')
    parser.add_argument('--get_wrong_exclusions', action='store_true', help='Get wrongful exclusions and save them.')
    args = parser.parse_args()
    main_jsonl(args)
