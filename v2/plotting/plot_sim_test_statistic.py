import pickle
import matplotlib.pyplot as plt


with open('outputs/sim_test_stat_er.pkl', 'rb') as f:
    er_result = pickle.load(f)
with open('outputs/sim_test_stat_sbm.pkl', 'rb') as f:
    sbm_result = pickle.load(f)

def plot_test_stats(result, ax):
    x = result['true']
    settings = ['pearson', 'gcorr']
    for i, s in enumerate(settings):
        mean = result[s].mean(axis=1)
        std = result[s].std(axis=1)
        ax.errorbar(x, mean, yerr=std, label=settings[i])
    ax.set_xlabel('true correlation')
    ax.set_ylabel('test statistic')
    ax.legend()
    ax.grid()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plot_test_stats(er_result, axs[0])
plot_test_stats(sbm_result, axs[1])
axs[0].set_title('ER')
axs[1].set_title('SBM')

plt.savefig('figures/sim_test_stat.png')