import numpy as np
import matplotlib.pyplot as plt
from simulations import dcsbm_corr
from graspologic.plot import heatmap

p = [[0.7, 0.3], [0.3, 0.7]]
r = 0.5
n = [70, 30]
theta1 = np.linspace(100, 1, n[0])
theta2 = np.linspace(100, 1, n[1])
# make sure the weights in each block sums to 1
theta1 /= theta1.sum()
theta2 /= theta2.sum()
theta = np.concatenate([theta1, theta2]) 

g1, g2 = dcsbm_corr(n, p, r, theta)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
heatmap(g1, ax=axs[0])
heatmap(g2, ax=axs[1])
fig.suptitle('$\\rho$-DCSBM w/ $\\rho$={}'.format(r))
plt.savefig('figures/sim_graphs_dcsbm.png')
