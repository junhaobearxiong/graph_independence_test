import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sim', help='`bern` or `gauss`')
args = parser.parse_args()

labels = {
    'pearson_exact': 'Pearson analytical p-values',
    'pearson_vertex_perm': 'Pearson vertex permutation',
    'pearson_block_perm': 'Pearson block permutation',
    'gcorr_vertex_perm': 'Gcorr vertex permutation',
    'gcorr_block_perm': 'Gcorr block permutation'
}

def plot_power_curve(result, ax):
    x = np.linspace(10, 100, 10, dtype=int)
    for test, power in result.items():
        ax.plot(x, power, 'o-', label=labels[test])
    ax.grid()


fig, axs = plt.subplots(3, 4, figsize=(20, 14), sharex=True, sharey=True)
rho = [0.0, 0.1, -0.1]
settings = [
    'er',
    'sbm_diffmarg',
    'sbm_diffblock',
    'sbm_estblock'
]

for i, r in enumerate(rho):
    for j, s in enumerate(settings):
        with open('outputs/sim_power_{}_{}_r{}.pkl'.format(args.sim, s, r), 'rb') as f:
            result = pickle.load(f)
        plot_power_curve(result, axs[i, j])
        axs[i, j].axhline(y=0.05, label='$\\alpha=0.05$', color='black')
        axs[i, j].axhline(y=1, label='power=1', color='m')


pad = 20
label_size = 15
rows = ['$\\rho=0$', '$\\rho=0.1$', '$\\rho=-0.1$']
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=label_size, ha='right', va='center')

cols = ['1.$\\rho$-ER', '2.$\\rho$-SBM', '3.$\\rho$-SBM (Different Block Size)', '4.$\\rho$-SBM (Unknown Community Assignment)']
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=label_size, ha='center', va='baseline')

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# plt.tight_layout()
plt.xlabel('number of vertices', fontsize=15)
plt.ylabel('power', fontsize=15)
axs[0, 0].legend(fontsize=12)

height = .95
if args.sim == 'bern':
    fig.suptitle('Power on Correlated Bernoulli Graphs', fontsize=label_size, y=height)
else:
    fig.suptitle('Power on Correlated Gaussian Graphs', fontsize=label_size, y=height)

plt.savefig('figures/sim_power_{}.png'.format(args.sim))