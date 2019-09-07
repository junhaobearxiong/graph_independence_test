import matplotlib.pyplot as plt


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
fig, axs = plt.subplots(6, 3, figsize=(12, 24), squeeze=True, sharex=True, sharey=True)

with open ('results/sim_power_bernoulli.pkl', 'rb') as f:
    power = pickle.load(f)
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
rows = ['ER(0.5)', 'ER(0.7) \n & ER(0.2)', 'SBM(0.7, 0.3)', 'SBM(0.7, 0.3) \n & SBM(0.2, 0.5)', 
        'SBM(0.7, 0.3) \n & SBM(0.2, 0.5) \n estimated blocks', 
        'SBM(0.7, 0.3) \n & SBM(0.2, 0.5) \n different block sizes']
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=label_size, ha='right', va='center')

cols = ['$\\rho=0$', '$\\rho=0.1$', '$\\rho=-0.1$']
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=label_size, ha='center', va='baseline')
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel("Number of vertices", fontsize=label_size)
plt.ylabel("Power", fontsize=label_size)
fig.suptitle('Power on Correlated Bernoulli Graphs', fontsize=label_size, y=0.92)
plt.savefig('figures/all_power_bernoulli.png', bbox_inches='tight')