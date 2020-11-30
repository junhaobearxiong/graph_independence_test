"""
For every pair of graphs, sweep through 2 <= K <= sqrt(N) = 20 
store the estimated assignment with the K with the best BIC
"""
import numpy as np
import pickle
from core import community_estimation

with open('data/mouse_graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)

num_graphs = graphs.shape[0]
num_vertices = graphs.shape[1]
num_tests = int(num_graphs * (num_graphs - 1) / 2) + num_graphs

# each row stores the optimal community assignment for a pair of graphs
assignments = np.zeros((num_tests, num_vertices))

count = 0
for i in range(num_graphs):
    for j in range(i, num_graphs):
        print('{}, {}'.format(i, j))
        G1 = graphs[i, ...]
        if i == j:
            assignments[count, :] = community_estimation(G1, min_components=2, max_components=20)
        else:
            G2 = graphs[j, ...]
            assignments[count, :] = community_estimation(G1, G2, min_components=2, max_components=20)
        count += 1

with open('outputs/mouse_Zhat.pkl', 'wb') as f:
    pickle.dump(assignments, f)