import pickle
import matplotlib.pyplot as plt
import numpy as np


labels = {
    'pearson': 'Pearson',
    'gcorr': 'Gcorr',
    'gcorr_unpooled': 'Gcorr unpooled'
}

titles = {
    'er': '1. $\\rho$-ER',
    'sbm_diffmarg': '2.$\\rho$-SBM', 
    'sbm_diffblock': '3.$\\rho$-SBM (Different Block Size)', 
    'sbm_estblock': '4.$\\rho$-SBM (Unknown Community Assignment)'
}

fontsize=15

def plot_test_stats(result, ax):
    x = result['true']
    settings = ['pearson', 'gcorr', 'gcorr_unpooled']
    for i, s in enumerate(settings):
        mean = result[s].mean(axis=1)
        std = result[s].std(axis=1)
        ax.errorbar(x, mean, yerr=2 * std, label=labels[s])
    ax.plot(x, x, 'o-', label='y=x', color='r')
    ax.grid()

fig, axs = plt.subplots(1, 4, figsize=(24, 5))
settings = ['er', 'sbm_diffmarg', 'sbm_diffblock', 'sbm_estblock']
for i, s in enumerate(settings): 
    with open ('outputs/sim_teststat_{}.pkl'.format(s), 'rb') as f:
        result = pickle.load(f)
    plot_test_stats(result, axs[i])
    axs[i].set_title(titles[s], fontsize=fontsize)
    if i == 0:
        axs[i].set_xlabel('true correlation', fontsize=fontsize)
        axs[i].set_ylabel('test statistic', fontsize=fontsize)
        axs[i].legend(fontsize=fontsize)

fig.suptitle('Test Statistics on Correlated Bernoulli Graphs', fontsize=fontsize, y=0.98)

plt.savefig('figures/sim_test_stat.png')