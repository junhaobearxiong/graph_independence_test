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

from simulations import rho_ER_marg, rho_sbm_marg
from utils import triu_no_diag, to_distance_mtx, identity, sbm_params

import multiprocessing as mp
import sys


def dcorr_ttest_power(sim_func, mc=500, alpha=0.05, **kwargs):
    # power for any test that builds on distance matrices
    # can use dcorr / mgc
    pval_array = np.zeros(mc)
    for i in range(mc):
        A, B = sim_func(**kwargs)
        dcorr = DCorr()
        pval, _ = dcorr.p_value(
            matrix_X=triu_no_diag(A), matrix_Y=triu_no_diag(B))
        pval_array[i] = pval
    power = np.where(pval_array < alpha)[0].shape[0] / mc
    return power


def ttest_power(sim_func, mc=500, alpha=0.05, **kwargs):
    # power for any test that builds on distance matrices
    # can use dcorr / mgc
    pval_array = np.zeros(mc)
    for i in range(mc):
        A, B = sim_func(**kwargs)
        test = RVCorr(which_test='pearson')
        pval, _ = test.p_value(
            matrix_X=triu_no_diag(A), matrix_Y=triu_no_diag(B))
        pval_array[i] = pval
    power = np.where(pval_array < alpha)[0].shape[0] / mc
    return power


def power_parallel(inputs):
    rho = inputs[0]
    n = inputs[1]
    P1 = inputs[2]
    P2 = inputs[3]
    nmc = inputs[4]
    graph_type = inputs[5]

    if graph_type == 'er' or graph_type == 'er_marg':
        test_power = ttest_power(rho_ER_marg,
                                 rho=rho, p=P1, q=P2, n=n, mc=nmc)
    else:
        test_power = ttest_power(rho_sbm_marg,
                                 rho=rho, AL=P1, BL=P2, k=2, n=n, mc=nmc)
    print('finish for rho={}, n={}'.format(rho, n))
    return rho, n, test_power


def fill_inputs(nmc, graph_type):
    inputs = []
    n_arr = np.linspace(10, 100, 10, dtype=int)
    rho_arr = np.array([0, 0.1, -0.1])
    if graph_type == 'er':
        P1 = 0.5
        P2 = 0.5
    elif graph_type == 'er_marg':
        P1 = 0.7
        P2 = 0.2
    elif graph_type == 'sbm':
        P1 = sbm_params(a=0.7, b=0.3)
        P2 = sbm_params(a=0.7, b=0.3)
    elif graph_type == 'sbm_marg':
        P1 = sbm_params(a=0.7, b=0.3)
        P2 = sbm_params(a=0.2, b=0.5)

    for i, rho in enumerate(rho_arr):
        for j, n in enumerate(n_arr):
            inputs.append((rho, n, P1, P2, nmc, graph_type))
    return inputs


def main(argv):
    nmc = int(argv[0])
    graph_type = argv[1]
    inputs = fill_inputs(nmc, graph_type)

    with mp.Pool(mp.cpu_count() - 1) as p:
        results = p.map(power_parallel, inputs)
    with open('results/rho_{}_power_parallel_pearson_ttest.pkl'.format(graph_type), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main(sys.argv[1:])
