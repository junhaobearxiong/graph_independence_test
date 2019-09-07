import numpy as np
import multiprocessing as mp
import pickle
import sys

from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.mgc import MGC

from simulations import rho_gaussian_sbm
from utils import dcorr_power, pearson_power


def fill_inputs():
    inputs = []
    n_arr = np.linspace(10, 100, 10, dtype=int)
    rho_arr = np.array([0, 0.1, -0.1])
    test_names = ['pearson', 'dcorr', 'mgc']
    params = {
        'er': {
            'P1': np.array([[0]]),
            'P2': np.array([[0]]),
            'k': [[n] for n in n_arr]
        },
        'er_marg': {
            'P1': np.array([[0]]),
            'P2': np.array([[2]]),
            'k': [[n] for n in n_arr]
        },
        'sbm': {
            'P1': np.array([[2, 0], [0, 2]]),
            'P2': np.array([[2, 0], [0, 2]]),
            'k': [[int(n/2), int(n/2)] for n in n_arr]
        },
        'sbm_marg': {
            'P1': np.array([[2, 0], [0, 2]]),
            'P2': np.array([[4, 2], [2, 4]]),
            'k': [[int(n/2), int(n/2)] for n in n_arr]
        }
    }

    for graph, param in params.items():
        for test in test_names:
            for rho in rho_arr:
                for i, n in enumerate(n_arr):
                    inputs.append((graph, param['P1'], param['P2'], param['k'][i], test, rho, n))
    return inputs


def get_power(inputs):
    graph, P1, P2, k, test_name, rho, n = inputs
    blocks = np.repeat(np.arange(len(k)), n//len(k))

    if test_name == 'pearson':
        test_power = pearson_power(rho_gaussian_sbm, given_blocks=True, blocks=blocks,
            rho=rho, n=n, k=k, AL=P1, BL=P2)
    elif test_name == 'dcorr':
        test = DCorr()
        test_power = dcorr_power(test, rho_gaussian_sbm, given_blocks=True, blocks=blocks,
            rho=rho, n=n, k=k, AL=P1, BL=P2)
    print('finish {}, {} for rho={}, n={}'.format(graph, test_name, rho, n))
    return graph, test_name, rho, n, test_power


inputs = fill_inputs()
with mp.Pool(mp.cpu_count() - 1) as p:
    results = p.map(get_power, inputs)
with open('results/sim_power_gaussian.pkl', 'wb') as f:
    pickle.dump(results, f)