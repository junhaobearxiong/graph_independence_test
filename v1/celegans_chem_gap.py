import numpy as np
import math
import pickle
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing as mp

from graspy.utils import is_symmetric

from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.dcorr import DCorr
from scipy.stats import pearsonr

from utils import estimate_block_assignment, block_permute, sort_graph, \
    to_distance_mtx, identity, triu_no_diag


def preprocess_csv(weighted):
    chem_file = 'data/herm_chem_A_full_undirected.csv'
    gap_file = 'data/herm_gap_A_full_undirected.csv'
    chem_cell_file = 'data/herm_chem_full_cells.csv'
    gap_cell_file = 'data/herm_gap_full_cells.csv'

    chem = pd.read_csv(chem_file, header=None).values
    gap = pd.read_csv(gap_file, header=None).values
    chem_cell = np.squeeze(pd.read_csv(chem_cell_file, header=None).values)
    gap_cell = np.squeeze(pd.read_csv(gap_cell_file, header=None).values)

    # take intersection
    common_cell, chem_idx, gap_idx = np.intersect1d(chem_cell, gap_cell,
                                                    return_indices=True)

    chem = chem[np.ix_(chem_idx, chem_idx)]
    gap = gap[np.ix_(gap_idx, gap_idx)]

    if weighted:
        return chem, gap
    else:
        chem_uw = np.where(chem > 0, 1, 0).astype(float)
        gap_uw = np.where(gap > 0, 1, 0).astype(float)
        return chem_uw, gap_uw

def edge_density(graph):
    num_edges = np.where(graph > 0)[0].size
    total = graph.shape[0] * graph.shape[1]
    return num_edges / total

def test_stats_parallel(inputs):
    chem = inputs[0]
    gap = inputs[1]
    k = inputs[2]
    reps = inputs[3]
    test_num = inputs[4]
    if test_num == 0:
        test = MGC(compute_distance_matrix=identity)
    elif test_num == 1:
        test = DCorr(compute_distance_matrix=identity)
    elif test_num == 2:
        test = RVCorr(which_test='pearson')
    test_stats_null_arr = np.zeros(reps)
    for r in tqdm(range(reps)):
        block_assignment = estimate_block_assignment(chem, gap,
                                                     k=k, set_k=True,
                                                     num_repeats=10)
        if test_num == 0 or test_num == 1:
            test_stats_null, _ = test.test_statistic(
                to_distance_mtx(block_permute(chem, block_assignment)),
                to_distance_mtx(sort_graph(gap, block_assignment)))
        else:
            test_stats_null, _ = test.test_statistic(
                triu_no_diag(block_permute(chem, block_assignment)),
                triu_no_diag(sort_graph(gap, block_assignment)))
        test_stats_null_arr[r] = test_stats_null
    print('finish k={}'.format(k))
    return (k, test_stats_null_arr)


def main(argv):
    reps = int(argv[0])
    weighted = bool(int(argv[1]))
    test_num = int(argv[2])

    chem, gap = preprocess_csv(weighted)
    k_arr = np.logspace(start=1, stop=8, num=8, base=2, dtype=int)

    inputs = [(chem, gap, k, reps, test_num) for k in k_arr]

    with mp.Pool(mp.cpu_count() - 1) as p:
        test_stats = p.map(test_stats_parallel, inputs)

    if weighted:
        file_name = 'results/celegans_chem_gap_weighted_teststats_null_{}.pkl'.format(test_num)
    else:
        file_name = 'results/celegans_chem_gap_unweighted_teststats_null_{}.pkl'.format(test_num)

    with open(file_name, 'wb') as f:
        pickle.dump(test_stats, f)


if __name__ == '__main__':
    main(sys.argv[1:])
