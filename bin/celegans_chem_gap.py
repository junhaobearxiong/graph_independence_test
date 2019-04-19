import numpy as np
import math
import pickle
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing as mp

from graspy.utils import is_symmetric

from mgcpy.independence_tests.mgc.mgc import MGC

from utils import estimate_block_assignment, block_permute, sort_graph, \
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
    reps = inputs[3]
    mgc = MGC(compute_distance_matrix=identity)
    test_stats_null_arr = np.zeros(reps)
    for r in tqdm(range(reps)):
        block_assignment = estimate_block_assignment(chem, gap,
                                                     k=k, set_k=True,
                                                     num_repeats=50)
        test_stats_null, _ = mgc.test_statistic(
            to_distance_mtx(block_permute(chem, block_assignment)),
            to_distance_mtx(sort_graph(gap, block_assignment)))
        test_stats_null_arr[r] = test_stats_null
    print('finish k={}'.format(k))
    return (k, test_stats_null_arr)


def main(argv):
    reps = int(argv[0])
    chem_file = 'data/celegans/male_chem_A_full_undirected.csv'
    gap_file = 'data/celegans/male_gap_A_full_undirected.csv'
    chem_cell_file = 'data/celegans/male_chem_full_cells.csv'
    gap_cell_file = 'data/celegans/male_gap_full_cells.csv'
    chem, gap = preprocess_csv(chem_file, gap_file, chem_cell_file,
                               gap_cell_file)
    k_arr = np.logspace(start=1, stop=9, num=9, base=2, dtype=int)

    inputs = [(chem, gap, k, reps) for k in k_arr]

    with mp.Pool(mp.cpu_count() - 1) as p:
        test_stats = p.map(test_stats_parallel, inputs)
    with open('results/celegans_teststats_null.pkl', 'wb') as f:
        pickle.dump(test_stats, f)


if __name__ == '__main__':
    main(sys.argv[1:])
