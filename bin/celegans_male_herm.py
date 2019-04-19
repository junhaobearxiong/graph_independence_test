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


def preprocess_csv(weighted):
    male = pd.read_csv('data/celegans/male_chem_A_full_undirected.csv',
                       header=None).values
    herm = pd.read_csv('data/celegans/herm_chem_A_full_undirected.csv',
                       header=None).values
    male_labels = np.squeeze(
        pd.read_csv('data/celegans/male_chem_full_cells.csv',
                    header=None).values)
    herm_labels = np.squeeze(
        pd.read_csv('data/celegans/herm_chem_full_cells.csv',
                    header=None).values)

    common_labels, male_idx, herm_idx = np.intersect1d(
        male_labels, herm_labels, return_indices=True)

    male_idx = np.sort(male_idx)
    herm_idx = np.sort(herm_idx)

    if weighted:
        male_sorted = male[np.ix_(male_idx, male_idx)]
        herm_sorted = herm[np.ix_(herm_idx, herm_idx)]
        return male_sorted, herm_sorted
    else:
        male_uw = np.where(male > 0, 1, 0).astype(float)
        herm_uw = np.where(herm > 0, 1, 0).astype(float)
        male_uw_sorted = male_uw[np.ix_(male_idx, male_idx)]
        herm_uw_sorted = herm_uw[np.ix_(herm_idx, herm_idx)]
        return male_uw_sorted, herm_uw_sorted


def test_stats_parallel(inputs):
    male = inputs[0]
    herm = inputs[1]
    k = inputs[2]
    reps = inputs[3]
    mgc = MGC(compute_distance_matrix=identity)
    test_stats_null_arr = np.zeros(reps)
    for r in tqdm(range(reps)):
        block_assignment = estimate_block_assignment(male, herm,
                                                     k=k, set_k=True,
                                                     num_repeats=10)
        test_stats_null, _ = mgc.test_statistic(
            to_distance_mtx(block_permute(male, block_assignment)),
            to_distance_mtx(sort_graph(herm, block_assignment)))
        test_stats_null_arr[r] = test_stats_null
    print('finish k={}'.format(k))
    return (k, test_stats_null_arr)


def main(argv):
    reps = int(argv[0])
    weighted = bool(int(argv[1]))
    male, herm = preprocess_csv(weighted)
    k_arr = np.logspace(start=1, stop=8, num=8, base=2, dtype=int)

    inputs = [(male, herm, k, reps) for k in k_arr]

    with mp.Pool(mp.cpu_count() - 1) as p:
        test_stats = p.map(test_stats_parallel, inputs)

    if weighted:
        file_name = 'results/celegans_male_herm_weighted_teststats_null.pkl'
    else:
        file_name = 'results/celegans_male_herm_unweighted_teststats_null.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump(test_stats, f)


if __name__ == '__main__':
    main(sys.argv[1:])
