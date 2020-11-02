import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_power_curve(result, ax):
    x = np.linspace(10, 100, 10, dtype=int)
    for test, power in result.items():
        ax.plot(x, power, 'o-', label=test)
    ax.grid()


fig, axs = plt.subplots(3, 4, figsize=(12, 8), sharex=True, sharey=True)
rho = [0.0, 0.1, -0.1]
settings = [
    'er',
    'sbm_diffmarg',
    'sbm_diffblock'

]

for i, r in enumerate(rho):
    for j, s in enumerate(settings):
        with open('outputs/sim_power_bernoulli_{}_r{}.pkl'.format(s, r), 'rb') as f:
            result = pickle.load(f)
            plot_power_curve(result, axs[i, j])

pad = 5
label_size = 15
rows = ['$\\rho=0$', '$\\rho=0.1$', '$\\rho=-0.1$']
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=label_size, ha='right', va='center')

cols = settings
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=label_size, ha='center', va='baseline')

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.tight_layout()
plt.xlabel('number of vertices')
plt.ylabel('power')
axs[0, 0].legend()

plt.savefig('figures/sim_power_bernoulli.png')