import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from graspologic.plot import heatmap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('z', help='`true` or `estimated`')
args = parser.parse_args()

participants = pd.read_csv('data/mouse_participants.csv', index_col=0)

fig, axs = plt.subplots(1, 5, figsize=(34, 6))
tests = ['Pearson', 'Gcorr Pooled', 'Gcorr Unpooled', 'DCSBM Gcorr Pooled', 'DCSBM Gcorr Unpooled']
tf = 'binarize'

for i, test in enumerate(tests):
    if test == 'Pearson':
        input_file = 'mouse_pearson_teststats_{}.pkl'.format(tf)
    elif test == 'Gcorr':
        input_file = 'mouse_gcorr_teststats_pooled_Z{}_{}.pkl'.format(args.z, tf)
    elif test == 'Gcorr Unpooled':
        input_file = 'mouse_gcorr_teststats_unpooled_Z{}_{}.pkl'.format(args.z, tf)
    elif test == 'DCSBM Gcorr':
        input_file = 'mouse_gcorrDC_teststats_pooled_Z{}_{}.pkl'.format(args.z, tf)
    elif test == 'DCSBM Gcorr Unpooled':
        input_file = 'mouse_gcorrDC_teststats_unpooled_Z{}_{}.pkl'.format(args.z, tf)

    with open('outputs/{}'.format(input_file), 'rb') as f:
        result = pickle.load(f)
    result += result.T
    axs[i].set_title('{}'.format(test), pad=40, fontsize=15)
    heatmap(result, ax=axs[i], vmax=0.7, center=0.35, inner_hier_labels=participants['genotype'], hier_label_fontsize=15)

height = .95
if args.z == 'true':
    title = 'Test Statistics on Duke Mouse Data (Given Community Assignments)'
else:
    title = 'Test Statistics on Duke Mouse Data'

fig.suptitle(title, fontsize=20, y=height)
plt.savefig('figures/mouse_teststats_Z{}.png'.format(args.z))