import numpy as np
import matplotlib.pyplot as plt
import pickle

fig, axs = plt.subplots(1, 4, figsize=(28, 7), squeeze=True, sharex=True, sharey=True)
axs = axs.reshape(-1)
xmin = -0.5
xmax = 1
ymin = -0.5
ymax = 1
alpha = 0.5
colors = {
    'true': [0.6350, 0.0780, 0.1840],
    'dcorr': (0, 0.4470, 0.7410), 
    'pearson': 'orange',
}
plt.rcParams['xtick.labelsize']=28
plt.rcParams['ytick.labelsize']=28
plt.rcParams["legend.fancybox"] = True
legend_size = 28
label_size = 28
titles = [
	'ER(0.5)',
	'ER(0.7) & ER(0.2)',
	'SBM(0.7, 0.3)',
	'SBM(0.7, 0.3) & SBM(0.2, 0.5)'
]

with open('results/sim_test_stats_params.pkl', 'rb') as f:
	params = pickle.load(f)
with open('results/sim_test_stats.pkl', 'rb') as f:
	output = pickle.load(f)

for i, (sim_name, results) in enumerate(output.items()):
	rho_arr = params[sim_name]['rho_arr']
	axs[i].plot(rho_arr, rho_arr, label='true correlation', marker='o', linestyle='-', color=colors['true'])
	for test_name, test_stats in results.items():
		axs[i].errorbar(rho_arr, np.mean(test_stats, axis=1), yerr=np.std(test_stats, axis=1), marker='o', linestyle='-', label='{}'.format(test_name), alpha=alpha, color=colors[test_name])
		axs[i].set_title(titles[i], fontsize=label_size)
		axs[i].grid(True)
	if i == 0:
		axs[i].legend(prop={'size': legend_size})

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.xlabel("True correlation", fontsize=label_size)
plt.ylabel("Test statistic", fontsize=label_size)
fig.suptitle('Test Statistics on Correlated Bernoulli Graphs', fontsize=label_size, y=1)

plt.savefig('figures/all_teststats.png', bbox_inches='tight')