import numpy as np
import math
import pickle
import pandas as pd
from tqdm import tqdm

from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.rv_corr import RVCorr
from mgcpy.independence_tests.mgc import MGC

from simulations import rho_ER_marg, rho_sbm_marg, rho_sbm_diff_block
from utils import sbm_params, dcorr_power, pearson_power, identity


def power_parallel(inputs):
    rho = inputs[0]
    n = inputs[1]
    p = 0.5
    q = 0.5
    nmc = 500
    k = 1
    blocks = np.repeat(np.arange(k), n//k)
    test = MGC(compute_distance_matrix=identity)
    test_power = dcorr_power(test, rho_ER_marg, given_blocks=True, blocks=blocks,
                             rho=rho, p=p, q=q, n=n, mc=nmc)
    print('finish rho={}, n={}'.format(rho, n))
    return (inputs, test_power)


def fill_inputs():
    inputs = []
    rho_arr = np.linspace(0.1, 0.9, 9, dtype=float)
    n_arr = np.array([10, 50, 100, 200, 300, 400, 500])

    for i, rho in enumerate(rho_arr):
        for j, n in enumerate(n_arr):
            inputs.append((rho, n))
    return inputs


def main(argv):
    inputs = fill_inputs()

    with mp.Pool(mp.cpu_count() - 1) as p:
        results = p.map(power_parallel, inputs)
    with open('results/power_er_mgc_debug.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main(sys.argv[1:])
