import numpy as np
import pickle
import os
from tqdm import tqdm

def symmetrize_or(A):
    A_sym = np.bitwise_or(A.astype(int), np.transpose(A).astype(int))
    return A_sym.astype(float)

dirname = 'data_utils/raw_data/enron/graphs/'
n_vertices = 184

filenames = sorted(os.listdir(dirname), key=lambda x: float(x[:-4]))
graphs_orig = np.zeros(shape=(len(filenames), n_vertices, n_vertices))
# symmetrized graphs
graphs = np.zeros(shape=(len(filenames), n_vertices, n_vertices))

for i, f in tqdm(enumerate(filenames), 'Loading graphs'):
    G = np.loadtxt(dirname + f, skiprows=1)
    graphs_orig[i] = G
    graphs[i] = symmetrize_or(G)

for i in range(graphs.shape[0]):
    g = graphs[i, ...]
    assert(np.all(g == g.transpose()))

with open('data/enron_graphs_original.pkl', 'wb') as f:
	pickle.dump(graphs_orig, f)

with open('data/enron_graphs_untransformed.pkl', 'wb') as f:
    pickle.dump(graphs, f)