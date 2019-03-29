import numpy as np
import math
import pickle
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing as mp


from graspy.utils import get_multigraph_intersect_lcc, is_symmetric

from mgcpy.independence_tests.mgc.mgc import MGC

from utils import estimate_block_assignment, permute_edges, block_permute, \
    permute_matrix, triu_no_diag, to_distance_mtx, identity, power


def pvalue(A, B, indept_test, transform_func, k=10, null_mc=500,
           svd='randomized', random_state=None):
    test_stat_alternative, _ = indept_test.test_statistic(
        matrix_X=transform_func(A), matrix_Y=transform_func(B))

    block_assignment = estimate_block_assignment(A, B, k=k, svd=svd,
                                                 random_state=random_state)

    test_stat_null_array = np.zeros(null_mc)
    for j in tqdm(range(null_mc)):
        A_null = block_permute(A, block_assignment)
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=transform_func(A_null), matrix_Y=transform_func(B))
        test_stat_null_array[j] = test_stat_null

    p_value = np.where(test_stat_null_array > test_stat_alternative)[
        0].shape[0] / test_stat_null_array.shape[0]
    return p_value


def preprocess_csv(file1, file2):
    df1 = pd.read_csv(file1, header=None).values
    df2 = pd.read_csv(file2, header=None).values
    df1, df2 = get_multigraph_intersect_lcc([df1, df2])
    df1 = np.where(df1 > 0, 1, 0).astype(float)
    df2 = np.where(df2 > 0, 1, 0).astype(float)
    return df1, df2


def pvalue_parallel(inputs):
    chem_male = inputs[0]
    gap_male = inputs[1]
    mgc = inputs[2]
    return pvalue(A=chem_male, B=gap_male, indept_test=mgc,
                  null_mc=500,
                  transform_func=to_distance_mtx,
                  svd='randomized',
                  random_state=None)


def main(argv):
    num_repeats = int(argv[0])
    file1 = 'celegans_data/male_chem_A_full_undirected.csv'
    file2 = 'celegans_data/male_gap_A_full_undirected.csv'
    chem_male, gap_male = preprocess_csv(file1, file2)
    mgc = MGC(compute_distance_matrix=identity)
    inputs = [(chem_male, gap_male, mgc) for i in range(num_repeats)]

    with mp.Pool(mp.cpu_count() - 1) as p:
        pval = p.map(pvalue_parallel, inputs)
    with open('celegans_data/pvalue.pkl', 'wb') as f:
        pickle.dump(pval, f)


if __name__ == '__main__':
    main(sys.argv[1:])
