from utils import triu_no_diag, to_distance_mtx, identity, estimate_block_assignment, sort_graph, sbm_params
from simulations import rho_sbm_diff_block
from graspy.plot import heatmap
from graspy.utils import is_symmetric, symmetrize
from graspy.embed.jrdpg import JointRDPG
from graspy.simulations import sbm, rdpg
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.rv_corr import RVCorr
from mgcpy.independence_tests.mgc.mgc import MGC
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
from tqdm import tqdm
import sys
import seaborn as sns
from os.path import dirname, join
from itertools import permutations
import multiprocessing as mp


def corr_rdpg(n, null):
    X = np.random.uniform(0, 1, (n, 1))
    A = rdpg(X, rescale=False, loops=False)
    if null:
        Y = np.random.uniform(0, 1, (n, 1))
    else:
        Y = np.square(X)
    B = rdpg(Y, rescale=False, loops=False)
    return A, B


def block_permute_vertex(A, block_assignment):
    A = sort_graph(A, block_assignment)
    block_assignment = np.sort(block_assignment)
    _, block_counts = np.unique(block_assignment, return_counts=True)
    shifted_block_counts = np.cumsum(block_counts)

    permuted_idx = []
    for i, count in enumerate(block_counts):
        idx = np.arange(count)
        np.random.shuffle(idx)
        if i >= 1:
            idx += shifted_block_counts[i-1]
        permuted_idx.append(idx)
    permuted_idx = np.concatenate(permuted_idx)

    permuted_A = sort_graph(A, permuted_idx)
    return permuted_A


def corr_rdpg_power(indept_test, transform_func, n, null, mc=500, alpha=0.05):
    test_stat_null_array = np.zeros(mc)
    test_stat_alt_array = np.zeros(mc)
    khat = np.zeros(mc)
    for i in tqdm(range(mc)):
        A, B = corr_rdpg(n, null)
        test_stat_alt, _ = indept_test.test_statistic(
            matrix_X=transform_func(A), matrix_Y=transform_func(B))
        test_stat_alt_array[i] = test_stat_alt

        # generate the null by permutation
        block_assignment = estimate_block_assignment(A, B, k=10, num_repeats=10)
        khat[i] = np.unique(block_assignment).size

        B_sorted = sort_graph(B, block_assignment)
        A_null = block_permute_vertex(A, block_assignment)
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=transform_func(A_null), matrix_Y=transform_func(B_sorted))
        test_stat_null_array[i] = test_stat_null
    # if pearson, use the absolute value of test statistic then use one-sided
    # rejection region
    if indept_test.get_name() == 'pearson':
        test_stat_null_array = np.absolute(test_stat_null_array)
        test_stat_alt_array = np.absolute(test_stat_alt_array)
    critical_value = np.sort(test_stat_null_array)[math.ceil((1-alpha)*mc)]
    power = np.where(test_stat_alt_array > critical_value)[0].shape[0] / mc
    return power, khat


def power_parallel(inputs):
    name = inputs[0]
    rho = inputs[1]
    n = inputs[2]
    nmc = inputs[3]
    if name == 'pearson':
        test = RVCorr(which_test='pearson')
        test_power, test_khat = corr_rdpg_power(test, triu_no_diag, n=n, null=rho, mc=nmc)
    elif name == 'dcorr':
        test = DCorr(compute_distance_matrix=identity)
        test_power, test_khat = corr_rdpg_power(test, to_distance_mtx, n=n, null=rho, mc=nmc)
    elif name == 'mgc':
        test = MGC(compute_distance_matrix=identity)
        test_power, test_khat = corr_rdpg_power(test, to_distance_mtx, n=n, null=rho, mc=nmc)
    print('finish {} for rho={}, n={}'.format(name, rho, n))
    return (inputs, test_power, test_khat)


def fill_inputs(nmc):
    inputs = []
    n_arr = np.linspace(10, 200, 20, dtype=int)
    rho_arr = np.array([True, False])
    test_names = ['pearson', 'dcorr']
    for name in test_names:
        for i, rho in enumerate(rho_arr):
            for j, n in enumerate(n_arr):
                inputs.append((name, rho, n, nmc))
    return inputs


def main(argv):
    nmc = int(argv[0])
    inputs = fill_inputs(nmc)

    with mp.Pool(mp.cpu_count() - 1) as p:
        results = p.map(power_parallel, inputs)
    with open('results/rdpg_power_parallel.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main(sys.argv[1:])
