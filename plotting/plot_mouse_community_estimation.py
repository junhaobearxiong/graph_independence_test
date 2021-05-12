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


with open('outputs/mouse_Zhat_{}.pkl'.format(args.transformation), 'rb') as f:
    Zhat = pickle.load(f)
with open('outputs/mouse_Zhat_dict_{}.pkl'.format(args.transformation), 'rb') as f:
    Zhat_dict = pickle.load(f)
with open('data/mouse_community_assignments.pkl', 'rb') as f:
    Ztrue = pickle.load(f)

participants = pd.read_csv('data/mouse_participants.csv', index_col=0)

def flip_diag(a, size=32):
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
        inner_hier_labels=participants['genotype'].values,
        ax=axs[1],
        title='number of estimated clusters'
       )
plt.savefig('figures/mouse_num_clusters_{}'.format(args.transformation))


# clustering metrics
num_tests = Zhat.shape[0]
adj_rand = np.zeros(num_tests)
adj_mi = np.zeros(num_tests)

for i in range(num_tests):
    adj_rand[i] = adjusted_rand_score(Ztrue, Zhat[i, :])
    adj_mi[i] = adjusted_mutual_info_score(Ztrue, Zhat[i, :])

adj_rand = flip_diag(adj_rand)
adj_mi = flip_diag(adj_mi)

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
heatmap(adj_rand, 
        inner_hier_labels=participants['genotype'].values,
        ax=axs[0],
        title='Adjusted RAND'
       )
heatmap(adj_mi, 
        inner_hier_labels=participants['genotype'].values,
        ax=axs[1],
        title='Adjusted MI'
       )
plt.savefig('figures/mouse_clustering_metrics_{}'.format(args.transformation))
