import numpy as np
import pickle
import matplotlib.pyplot as plt


# process the output after parallelizing
n_arr = np.linspace(10, 100, 10, dtype=int)
rho_arr = np.array([0, 0.1, -0.1])
graph_names = ['er', 'er_marg', 'sbm', 'sbm_marg']
test_names = ['pearson', 'dcorr', 'mgc']

with open('results/sim_power_gaussian.pkl', 'rb') as f:
    results = pickle.load(f)
count = 0
power = {}
for graph in graph_names:
	power[graph] = {
        'pearson': np.zeros((rho_arr.shape[0], n_arr.shape[0])),
        'dcorr': np.zeros((rho_arr.shape[0], n_arr.shape[0])),
        'mgc': np.zeros((rho_arr.shape[0], n_arr.shape[0]))
	}
	for test in test_names:
		for i, rho in enumerate(rho_arr):
			for j, n in enumerate(n_arr):
			    power[graph][test][i, j] = results[count][4]
			    count += 1
			if count == n_arr.shape[0] * rho_arr.shape[0] - 1:
			    count = 0
with open('results/sim_power_gaussian_processed.pkl', 'wb') as f:
    pickle.dump(power, f)


# plot settings
colors = {
    'mgc': [0.6350, 0.0780, 0.1840],
    'dcorr': (0, 0.4470, 0.7410), 
    'pearson': 'orange',
    'ttest': 'green'
}
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams["legend.fancybox"] = True
legend_size = 20
label_size = 20
linestyle = '-'
marker = 'o'


# plotting 
fig, axs = plt.subplots(4, 3, figsize=(12, 16), squeeze=True, sharex=True, sharey=True)
for i, (graph, result) in enumerate(power.items()):
	for j in range(3):
		for test_name, test_power in result.items():
			axs[i, j].plot(n_arr, test_power[j, :], color=colors[test_name], linestyle=linestyle, 
				marker=marker, label=test_name)
		if j == 0:
			axs[i, j].hlines(y=0.05, xmin=np.amin(n_arr), xmax=np.amax(n_arr), label='alpha')
		else:
			axs[i, j].hlines(y=1, xmin=np.amin(n_arr), xmax=np.amax(n_arr))
		if i == 0 and j == 0:
			axs[i, j].legend(prop={'size': legend_size})


# add labels and titles
pad = 5
rows = ['ER(0)', 'ER(0) \n & ER(2)', 'SBM(2, 0)', 'SBM(2, 0) \n & SBM(4, 2)']
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=label_size, ha='right', va='center')

cols = ['$\\rho=0$', '$\\rho=0.1$', '$\\rho=-0.1$']
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=label_size, ha='center', va='baseline')

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel("Number of vertices", fontsize=label_size)
plt.ylabel("Power", fontsize=label_size)
fig.suptitle('Power on Correlated Gaussian Graphs', fontsize=label_size, y=0.92)

plt.savefig('figures/all_power_gaussian.png', bbox_inches='tight')