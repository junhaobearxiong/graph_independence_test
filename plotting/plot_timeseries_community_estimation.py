import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from graspologic.plot import heatmap


parser = argparse.ArgumentParser()
parser.add_argument('transformation')
args = parser.parse_args()


with open('outputs/timeseries_Zhat_{}.pkl'.format(args.transformation), 'rb') as f:
    Zhat = pickle.load(f)
with open('outputs/timeseries_Zhat_dict_{}.pkl'.format(args.transformation), 'rb') as f:
    Zhat_dict = pickle.load(f)

def flip_diag(a, size=204):
    b = np.zeros((size, size))
    b[np.triu_indices(size)] = a
    b += b.T
    b[np.diag_indices(size)] /= 2
    return b

# number of clusters
num_clusters = np.zeros(Zhat.shape[0])
for i in range(Zhat.shape[0]):
    num_clusters[i] = np.unique(Zhat[i, :]).size
pair_clusters = flip_diag(num_clusters)

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
cluster, count = np.unique(num_clusters, return_counts=True)
axs[0].bar(cluster, count)
axs[0].set_xlabel('number of estimated clusters')
axs[0].set_ylabel('number of pairs of graphs')

heatmap(pair_clusters, 
        ax=axs[1],
        title='number of estimated clusters'
       )
plt.savefig('figures/timeseries_num_clusters_{}'.format(args.transformation))
