import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from graspologic.plot import heatmap

participants = pd.read_csv('data/mouse_participants.csv', index_col=0)

fig, axs = plt.subplots(2, 5, figsize=(40, 20))
tests = ['Pearson', 'Gcorr Pooled', 'Gcorr Unpooled', 'DCSBM Gcorr Pooled', 'DCSBM Gcorr Unpooled']
zs = ['true', 'estimated']
tf = 'binarize'

for i, z in enumerate(zs):
    for j, test in enumerate(tests):
        if test == 'Pearson':
            input_file = 'mouse_pearson_teststats_{}.pkl'.format(tf)
        elif test == 'Gcorr Pooled':
            input_file = 'mouse_gcorr_teststats_pooled_Z{}_{}.pkl'.format(z, tf)
        elif test == 'Gcorr Unpooled':
            input_file = 'mouse_gcorr_teststats_unpooled_Z{}_{}.pkl'.format(z, tf)
        elif test == 'DCSBM Gcorr Pooled':
            input_file = 'mouse_gcorrDC_teststats_pooled_Z{}_{}.pkl'.format(z, tf)
        elif test == 'DCSBM Gcorr Unpooled':
            input_file = 'mouse_gcorrDC_teststats_unpooled_Z{}_{}.pkl'.format(z, tf)

        with open('outputs/{}'.format(input_file), 'rb') as f:
            result = pickle.load(f)
        result += result.T
        heatmap(result, ax=axs[i, j], vmax=0.7, center=0.35, inner_hier_labels=participants['genotype'], hier_label_fontsize=25, cbar=not bool(i + j))


pad = 60
label_size = 40
rows = ['Z Given', 'Z Estimated']
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=label_size, ha='right', va='center')

cols = tests
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=label_size, ha='center', va='baseline')

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# plt.tight_layout()


height = .98
fig.suptitle('Test Statistics on Mouse Connectome Dataset', fontsize=50, y=height)
plt.tight_layout()
plt.savefig('figures/mouse_teststats.png')