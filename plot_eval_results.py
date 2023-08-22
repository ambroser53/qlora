import matplotlib.pyplot as plt
from glob import glob
from argparse import Namespace
from itertools import groupby

from matplotlib.ticker import MaxNLocator

import data.eval as eval
import re
from collections import defaultdict
import pandas as pd

gold_files = glob('generated_samples/**/Gold/**/*.jsonl', recursive=True)
eval_files = glob('generated_samples/**/Eval/**/*.jsonl', recursive=True)

def get_model_df(grouped_results, all_epochs=False, type='gold'):
    data_dict = {
        'gold': {
            'dataset_path': 'data/instruct_cochrane_gold.json',
            'label_field_name': 'gold_label'
        },
        'eval': {
            'dataset_path': 'data/instruct_cochrane_eval_final_inc_exc_only.json',
            'label_field_name': 'output'
        }
    }

    results_arr = []

    for model, files in grouped_results:
        for file in files:
            epoch = re.match(r'.*_(?P<epoch>\d+)_.*', file).group(1)

            if not all_epochs and int(epoch) > 8 or int(epoch) == 0:
                continue

            args = Namespace(
                results_path=file,
                lines=True,
                rogue_tokens=True,
                response_field_name='response',
                **data_dict[type]
            )

            results_df, _ = eval.main(args)
            results_df['model'] = model
            results_df['epoch'] = int(epoch)
            results_df.drop(index=['weighted avg'], columns='support', inplace=True)

            results_df.set_index(['model', 'epoch', results_df.index], inplace=True)

            results_arr.append(results_df)

    return pd.concat(results_arr).sort_index()

grouped_gold = groupby(sorted(gold_files, key=lambda x: x.split('/')[1]), lambda x: x.split('/')[1])
grouped_eval = groupby(sorted(eval_files, key=lambda x: x.split('/')[1]), lambda x: x.split('/')[1])

gold_df = get_model_df(grouped_gold, type='gold')
eval_df = get_model_df(grouped_eval, type='eval')

grouped_gold = groupby(sorted([g for g in gold_files if 'Revaco7bMultiEx' in g], key=lambda x: x.split('/')[1]), lambda x: x.split('/')[1])
grouped_eval = groupby(sorted([g for g in eval_files if 'Revaco7bMultiEx' in g], key=lambda x: x.split('/')[1]), lambda x: x.split('/')[1])

gold_df_extended = get_model_df(grouped_gold, all_epochs=True, type='gold')
eval_df_extended = get_model_df(grouped_eval, all_epochs=True, type='eval')

def plot_results(results_df, metric='f1-score', label=''):
    models = results_df.index.get_level_values(0).unique()
    models = [model for model in models if '13' not in model]

    if len(models) == 1:
        _, axs = plt.subplots(1, len(models), figsize=(8, 5))
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        results_df.xs(models[0]).unstack(level=1)[metric].plot(ax=axs)
        axs.set_title(models[0].replace('SysRev', '').replace('7b', ' ').replace('Ex', '').replace('aco', 'Guan'))
        axs.set_xlabel('Epoch')
        axs.set_ylabel(metric)

    else:
        _, axs = plt.subplots(1, len(models), sharey=True, figsize=(20, 5))

        for i, model in enumerate(models):
            results_df.xs(model).unstack(level=1)[metric].plot(ax=axs[i])
            axs[i].set_title(model.replace('SysRev', '').replace('7b', ' ').replace('Ex', '').replace('aco', 'Guan')+label)
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(metric)

    plt.show()

#plot_results(gold_df)
#plot_results(eval_df)

plot_results(gold_df_extended, label=' Gold')
plot_results(eval_df_extended, label=' Eval')