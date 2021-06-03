import pickle
import matplotlib.pyplot as plt
import numpy as np

labels = {
	'gcorr_block_perm': 'Gcorr block permutation',
    'gcorr_param_bootstrap': 'Gcorr paramteric bootstrap',
    'gcorrDC_block_perm': 'DCSBM Gcorr block permutation',
    'gcorrDC_param_bootstrap': 'DCSBM Gcorr parametric bootstrap'
}

def plot_power_curve(result, ax):
    x = np.linspace(20, 100, 9, dtype=int)
    for test, power in result.items():
        ax.plot(x, power, 'o-', label=labels[test])
    ax.grid()


fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
rho = [0.0, 0.1]
settings = [
	'sbm',
	'dcsbm'
]

for i, r in enumerate(rho):
    for j, s in enumerate(settings):
        with open('outputs/sim_power_dc_{}_rho{}.pkl'.format(s, r), 'rb') as f:
            result = pickle.load(f)
        plot_power_curve(result, axs[i, j])
        axs[i, j].axhline(y=0.05, label='$\\alpha=0.05$', color='black')
        axs[i, j].axhline(y=1, label='power=1', color='m')


pad = 20
label_size = 15
rows = ['$\\rho=0$', '$\\rho=0.1$']
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=label_size, ha='right', va='center')

cols = ['1.$\\rho$-SBM', '2.$\\rho$-DCSBM']
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
fig.suptitle('Power on Correlated Bernoulli DCSBM Graphs', fontsize=label_size, y=height)
plt.savefig('figures/sim_power_dcsbm.png')