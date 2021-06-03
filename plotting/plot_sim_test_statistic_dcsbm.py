import pickle
import matplotlib.pyplot as plt
import numpy as np


labels = {
    'gcorr_pooled': 'Gcorr Pooled',
    'gcorr_unpooled': 'Gcorr Unpooled',
    'gcorr_dcsbm_pooled': 'DCSBM Gcorr Pooled'
    'gcorr_dcsbm_unpooled': 'DCSBM Gcorr Unpooled'
}

titles = {
    'givenblock': '1.$\\rho$-DCSBM', 
    'diffblock': '2.$\\rho$-DCSBM (Different Block Size)', 
    'estblock': '3.$\\rho$-DCSBM (Unknown Community Assignment)'
}

fontsize=15

def plot_test_stats(result, ax):
    x = result['true']
    settings = ['gcorr_pooled', 'gcorr_unpooled', 'gcorr_dcsbm_pooled', 'gcorr_dcsbm_unpooled']
    for i, s in enumerate(settings):
        mean = result[s].mean(axis=1)
        std = result[s].std(axis=1)
        ax.errorbar(x, mean, yerr=2 * std, label=labels[s])
    ax.plot(x, x, 'o-', label='y=x', color='r')
    ax.grid()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
settings = ['givenblock', 'diffblock', 'estblock']
for i, s in enumerate(settings): 
    with open ('outputs/sim_teststat_dcsbm_{}.pkl'.format(s), 'rb') as f:
        result = pickle.load(f)
    plot_test_stats(result, axs[i])
    axs[i].set_title(titles[s], fontsize=fontsize)
    if i == 0:
        axs[i].set_xlabel('true correlation', fontsize=fontsize)
        axs[i].set_ylabel('test statistic', fontsize=fontsize)
        axs[i].legend(fontsize=fontsize)

fig.suptitle('Test Statistics on Correlated Bernoulli DCSBM', fontsize=fontsize, y=0.98)

plt.savefig('figures/sim_test_stat_dcsbm.png')