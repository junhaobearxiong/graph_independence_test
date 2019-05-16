import numpy as np
import math
import pickle
import pandas as pd
from tqdm import tqdm

from graspy.plot import heatmap
from graspy.utils import symmetrize
from graspy.simulations import sbm

from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.rv_corr import RVCorr
from mgcpy.independence_tests.mgc import MGC

from simulations import sbm_corr, ER_corr
from utils import triu_no_diag, to_distance_mtx, identity, sbm_params, dcorr_power_two_sided, pearson_power_two_sided

import multiprocessing as mp


def power_parallel(inputs):
    name = inputs[0]
    rho = inputs[1]
    n = inputs[2]
    k = inputs[3]
    P1 = inputs[4]
    P2 = inputs[5]
    nmc = inputs[6]
    graph_type = inputs[7]

    block_assignments = np.repeat(np.arange(k), n//k)
    if graph_type == 'er' or graph_type == 'er_marg':
        block_sizes = np.array([n])
    elif graph_type == 'sbm' or graph_type == 'sbm_marg':
        block_sizes = np.array([n//2, n//2])
    elif graph_type == 'sbm_diff_block':
        block_sizes = np.array([int(0.7*n), n-int(0.7*n)])
    if name == 'pearson':
        test = RVCorr(which_test='pearson')
        test_power = pearson_power_two_sided(test, rho_gaussian_sbm,
                                             given_blocks=True, blocks=block_assignments,
                                             rho=rho, AL=P1, BL=P2, k=block_sizes, n=n, mc=nmc)
    elif name == 'dcorr':
        test = DCorr(compute_distance_matrix=identity)
        test_power = dcorr_power_two_sided(test, rho_gaussian_sbm,
                                           given_blocks=True, blocks=block_assignments,
                                           rho=rho, AL=P1, BL=P2, k=block_sizes, n=n, mc=nmc)

    elif name == 'mgc':
        test = MGC(compute_distance_matrix=identity)
        test_power = dcorr_power_two_sided(test, rho_gaussian_sbm,
                                           given_blocks=True, blocks=block_assignments,
                                           rho=rho, AL=P1, BL=P2, k=block_sizes, n=n, mc=nmc)
    print('finish {} for rho={}, n={}'.format(name, rho, n))
    return name, rho, n, test_power


def fill_inputs(nmc, graph_type):
    inputs = []
    n_arr = np.linspace(10, 100, 10, dtype=int)
    rho_arr = np.array([0, 0.1, -0.1])
    test_names = ['pearson', 'dcorr', 'mgc']
    if graph_type == 'er':
        P1 = np.array([[0]])
        P2 = np.array([[0]])
        k = 1
    elif graph_type == 'er_marg':
        P1 = np.array([[0]])
        P2 = np.array([[2]])
        k = 1
    elif graph_type == 'sbm':
        P1 = np.array([[2, 0], [0, 2]])
        P2 = np.array([[2, 0], [0, 2]])
        k = 2
    elif graph_type == 'sbm_marg':
        P1 = np.array([[2, 0], [0, 2]])
        P2 = np.array([[4, 2], [2, 4]])
        k = 2

    for name in test_names:
        for i, rho in enumerate(rho_arr):
            for j, n in enumerate(n_arr):
                inputs.append((name, rho, n, k, P1, P2, nmc, graph_type))
    return inputs


def main(argv):
    nmc = int(argv[0])
    graph_type = argv[1]
    inputs = fill_inputs(nmc, graph_type)

    with mp.Pool(mp.cpu_count() - 1) as p:
        results = p.map(power_parallel, inputs)
    with open('results/rho_gaussain_power_parallel_{}.pkl'.format(graph_type), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main(sys.argv[1:])
