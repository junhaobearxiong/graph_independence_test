"""
For every pair of graphs, sweep through 2 <= K <= sqrt(N) = 20 
store the estimated assignment with the K with the best BIC
"""
import numpy as np
import pickle
import argparse
from core import community_estimation

parser = argparse.ArgumentParser()
parser.add_argument('transformation', nargs='?', default='untransformed', help='transformation applied to the graphs')
args = parser.parse_args()

if args.transformation not in [
        'untransformed',
        'binarize'
    ]:
    raise ValueError('{} is not implemented'.format(args.transformation))
else:
    with open('data/mouse_graphs_{}.pkl'.format(args.transformation), 'rb') as f:
        graphs = pickle.load(f)

num_graphs = graphs.shape[0]
num_vertices = graphs.shape[1]
num_tests = int(num_graphs * (num_graphs - 1) / 2) + num_graphs

# each row stores the optimal community assignment for a pair of graphs
# easier for iterating over
Zhat = np.zeros((num_tests, num_vertices))
# key: (i, j), value: optimal assignment for graph i and j
# easier for access
Zhat_dict = {}

count = 0
for i in range(num_graphs):
    for j in range(i, num_graphs):
        print('{}, {}'.format(i, j))
        G1 = graphs[i, ...]
        if i == j:
            assignment = community_estimation(G1, min_components=2, max_components=20)
        else:
            G2 = graphs[j, ...]
            assignment = community_estimation(G1, G2, min_components=2, max_components=20)
        Zhat[count, :] = assignment
        count += 1
        Zhat_dict[(i, j)] = assignment

with open('outputs/mouse_Zhat_{}.pkl'.format(args.transformation), 'wb') as f:
    pickle.dump(Zhat, f)
with open('outputs/mouse_Zhat_dict_{}.pkl'.format(args.transformation), 'wb') as f:
    pickle.dump(Zhat_dict, f)