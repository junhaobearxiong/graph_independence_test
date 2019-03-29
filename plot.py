import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def double_plot(A, B):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(A, ax=ax[0], xticklabels=False, yticklabels=False, cbar=False, cmap='Blues')
    sns.heatmap(B, ax=ax[1], xticklabels=False, yticklabels=False, cbar=False, cmap='Blues')
    ax[0].set_title('Adjacency matrix A', fontsize=20)
    ax[1].set_title('Adjacency matrix B', fontsize=20)
    plt.show()
    return


def plot_pvalue_ecdf(pvalue_mc, title):
    x = np.sort(pvalue_mc)
    y = np.arange(len(x))/float(len(x))
    plt.xlabel('p-value')
    plt.ylabel('probability')
    plt.title(title)
    plt.plot(x, y, label='ecdf')
    plt.plot([0, 1], [0, 1], label='y=x')
    plt.legend()
    # plt.savefig('plots/rho_SBM_pvalue_{}_rho_{}.png'.format(test.get_name(), rho))
    plt.show()
