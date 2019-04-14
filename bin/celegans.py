import numpy as np
import math
import pickle
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing as mp

from graspy.utils import get_multigraph_intersect_lcc, is_symmetric

from mgcpy.independence_tests.mgc.mgc import MGC

from utils import estimate_block_assignment, block_permute, sort_graph,
to_distance_mtx, identity, pvalue


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

    return chem, gap


def test_stats_parallel(inputs):
    chem = inputs[0]
    gap = inputs[1]
    k = inputs[2]
    mgc = MGC(compute_distance_matrix=identity)
    reps = 100
    test_stats_null_arr = np.zeros(reps)
    for r in range(reps):
        block_assignment = estimate_block_assignment(left, right_sorted,
                                                     k=k, set_k=True, num_repeats=50)
        test_stats_null, _ = mgc.test_statistic(
            to_distance_mtx(block_permute(left, block_assignment)),
            to_distance_mtx(sort_graph(right_sorted, block_assignment)))
        test_stats_null_arr[r] = test_stats_null

    return (k, pval)


def main(argv):
    chem_file = 'celegans_data/male_chem_A_full_undirected.csv'
    gap_file = 'celegans_data/male_gap_A_full_undirected.csv'
    chem_cell_file = 'celegans_data/male_chem_full_cells.csv'
    gap_cell_file = 'celegans_data/male_gap_full_cells.csv'
    chem, gap = preprocess_csv(chem_file, gap_file, chem_cell_file,
                               gap_cell_file)
    k_arr = np.logspace(start=1, stop=7, num=7, base=2, dtype=int)

    inputs = [(chem, gap, k) for k in k_arr]

    with mp.Pool(mp.cpu_count() - 1) as p:
        pval = p.map(pvalue_parallel, inputs)
    with open('celegans_data/pvalue.pkl', 'wb') as f:
        pickle.dump(pval, f)


if __name__ == '__main__':
    main(sys.argv[1:])
