import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

path = './Talos_results'
all_files = glob.glob(os.path.join(path , "*.csv"))

li = []; targets = []
# load all CSVs and merge them into on
for filename in all_files:
    tg = filename.split('_')[-1][:-4]
    df = pd.read_csv(filename, index_col=None, header=0, delimiter=';')
    df['target'] = tg # append the name of target to df
    li.append(df); targets.append(tg)
frame = pd.concat(li, axis=0, ignore_index=True)

# print(targets)
# print(frame.loc[frame.query('target == "NR-AR" and fp == "ecfp4"')['crossval_auc'].idxmax(), 'crossval_auc'])

# prepare data for plotting
best_AUCs = {}; AUCs = []
for fp in frame.fp.unique():
    for target in targets:
        try:
            # for every (fingerprint, target), get the best crossval_auc, std and save it
            crossval_auc = frame.loc[frame.query(f'target == "{target}" and fp == "{fp}"')['crossval_auc'].idxmax(), 'crossval_auc']
            std = frame.loc[frame.query(f'target == "{target}" and fp == "{fp}"')['crossval_auc'].idxmax(), 'crossval_auc_std']
            AUCs.append((crossval_auc,std,target))
        except ValueError:
            continue
    best_AUCs[f'{fp}'] = AUCs; AUCs = []

N = 12
ind = np.arange(N)    # the x locations for the groups
width = 0.85
colors = 7*['#F19018'] + 5*['#7DD144']
fps_abbreviations = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aro', 'NR-ER-LBD', 'NR-ER', 'NR-PPARG', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

for key in best_AUCs.keys():
    print(len(best_AUCs[key]), key)
    # if fp has crossval_auc scores for all targets, plot the results
    if len(best_AUCs[key]) == 12:
        auc, err, tg = zip(*best_AUCs[key])
        tg = fps_abbreviations
        auc_label = [ '%.2f' % elem for elem in auc ]

        fig, ax = plt.subplots()
        p1 = ax.bar(ind, auc, width, yerr=err, capsize=3, color=colors)

        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_ylabel('10-fold crossvalidated ROC_AUC')
        ax.set_title('ROC_AUCs for ecfp4')
        ax.set_xticks(ind)
        ax.set_xticklabels(tg, rotation=45)
        ax.set_yticks(np.arange(0, 1.1, step=0.1))
        ax.bar_label(p1, auc_label, label_type='edge')
        plt.xlim([-0.7,N-0.3])
        plt.savefig(f'./figs/talos_{key}.png', dpi=600, bbox_inches="tight")