import numpy as np
import math
import pickle
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing as mp

from graspy.utils import get_multigraph_intersect_lcc, is_symmetric

from mgcpy.independence_tests.mgc.mgc import MGC

from utils import estimate_block_assignment, block_permute, sort_graph, \
    to_distance_mtx, identity


def pvalue(A, B, indept_test, transform_func, k=10, set_k=False, null_mc=500,
           block_est_repeats=1):
    test_stat_alternative, _ = indept_test.test_statistic(
        matrix_X=transform_func(A), matrix_Y=transform_func(B))

    block_assignment = estimate_block_assignment(A, B, k=k, set_k=set_k,
                                                 num_repeats=block_est_repeats)
    B_sorted = sort_graph(B, block_assignment)

    test_stat_null_array = np.zeros(null_mc)
    for j in tqdm(range(null_mc)):
        # A_null is the permuted matrix after being sorted by block assignment
        A_null = block_permute(A, block_assignment)
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=transform_func(A_null), matrix_Y=transform_func(B_sorted))
        test_stat_null_array[j] = test_stat_null

    p_value = np.where(test_stat_null_array > test_stat_alternative)[
        0].shape[0] / test_stat_null_array.shape[0]
    return p_value


def preprocess_csv(chem_file, gap_file, chem_cell_file, gap_cell_file):
    chem = pd.read_csv(chem_file, header=None).values
    gap = pd.read_csv(gap_file, header=None).values
    chem_cell = pd.read_csv(chem_cell_file, header=None)
    gap_cell = pd.read_csv(gap_cell_file, header=None)
    chem_cell = np.squeeze(chem_cell.values)
    gap_cell = np.squeeze(gap_cell.values)

    # take intersection
    common_cell, chem_idx, gap_idx = np.intersect1d(chem_cell, gap_cell,
                                                    return_indices=True)
    chem_idx = np.sort(chem_idx)
    gap_idx = np.sort(gap_idx)
    chem = chem[np.ix_(chem_idx, chem_idx)]
    gap = gap[np.ix_(gap_idx, gap_idx)]

    # convert to unweighted
    chem = np.where(chem > 0, 1, 0).astype(float)
    gap = np.where(gap > 0, 1, 0).astype(float)

    return chem, gap


def pvalue_parallel(inputs):
    chem = inputs[0]
    gap = inputs[1]
    k = inputs[2]
    mgc = MGC(compute_distance_matrix=identity)
    pval = pvalue(chem, gap, indept_test=mgc, transform_func=to_distance_mtx,
                  k=k, set_k=True, block_est_repeats=100)

    return (k, pval)


def main(argv):
    chem_file = 'celegans_data/male_chem_A_full_undirected.csv'
    gap_file = 'celegans_data/male_gap_A_full_undirected.csv'
    chem_cell_file = 'celegans_data/male_chem_full_cells.csv'
    gap_cell_file = 'celegans_data/male_gap_full_cells.csv'
    chem, gap = preprocess_csv(chem_file, gap_file, chem_cell_file,
                               gap_cell_file)
    max_k = int(argv[0])
    k_arr = np.linspace(1, max_k, max_k, dtype=int)
    inputs = [(chem, gap, k) for k in k_arr]

    with mp.Pool(mp.cpu_count() - 1) as p:
        pval = p.map(pvalue_parallel, inputs)
    with open('celegans_data/pvalue.pkl', 'wb') as f:
        pickle.dump(pval, f)


if __name__ == '__main__':
    main(sys.argv[1:])
