import numpy as np
import math
import pickle
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing as mp
import networkx as nx

from graspy.utils import symmetrize

from mgcpy.independence_tests.mgc.mgc import MGC

from utils import estimate_block_assignment, block_permute, sort_graph, \
    to_distance_mtx, identity, pvalue


def preprocess_csv():
    with open('data/drosophila/weighted_drosophilia_match.pkl', 'rb') as f:
        w_graph_match = pickle.load(f)
    w_graph_match = np.reshape(w_graph_match, (213, 213))
    left = np.loadtxt('data/drosophila/left_adjacency.csv', dtype=int)
    right = np.loadtxt('data/drosophila/right_adjacency.csv', dtype=int)

    # left and right, symmetrized
    left = symmetrize(left)
    right = symmetrize(right)

    right_nx = nx.from_numpy_matrix(right)
    right_nx.remove_nodes_from(np.where(w_graph_match == 1)[1][left.shape[0]:])
    # right (removed nodes, not sorted)
    right_rm = nx.to_numpy_array(right_nx)

    # right (removed nodes and sorted)
    right_sorted = sort_graph(right_rm, np.where(w_graph_match == 1)[1][:left.shape[0]])
    return (left, right_sorted)


def test_stats_parallel(inputs):
    left = inputs[0]
    right_sorted = inputs[1]
    k = inputs[2]
    reps = inputs[3]
    mgc = MGC(compute_distance_matrix=identity)
    test_stats_null_arr = np.zeros(reps)
    for r in tqdm(range(reps)):
        block_assignment = estimate_block_assignment(left, right_sorted,
                                                     k=k, set_k=True, num_repeats=50)
        test_stats_null, _ = mgc.test_statistic(
            to_distance_mtx(block_permute(left, block_assignment)),
            to_distance_mtx(sort_graph(right_sorted, block_assignment)))
        test_stats_null_arr[r] = test_stats_null
    print('finish k={}'.format(k))
    return (k, test_stats_null_arr)


def main(argv):
    reps = int(argv[0])
    left, right_sorted = preprocess_csv()
    k_arr = np.logspace(start=1, stop=7, num=7, base=2, dtype=int)

    inputs = [(left, right_sorted, k, reps) for k in k_arr]

    with mp.Pool(mp.cpu_count() - 1) as p:
        test_stats = p.map(test_stats_parallel, inputs)
    with open('results/drosophila_weighted_teststats_null.pkl', 'wb') as f:
        pickle.dump(test_stats, f)


if __name__ == '__main__':
    main(sys.argv[1:])
