import numpy as np
import pickle
from simulations import *
from core import (
    gcorr,
    pearson_graph
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sim", help="which simulation to run, either `er` or `sbm`")
args = parser.parse_args()

def run_er():
    n = 100
    p = 0.7
    q = 0.3
    num_corr = 10
    num_rep = 500
    z = np.repeat([0], n)
    delta = 0.1
    corr = np.linspace(get_lowest_r(p, q) + delta, get_highest_r(p, q) - delta, num_corr)
    result = {
        'true': corr,
        'pearson': np.zeros((num_corr, num_rep)),
        'gcorr': np.zeros((num_corr, num_rep))
    }
    for i, r in enumerate(corr):
        for j in range(num_rep):
            g1, g2 = er_corr_diffmarg(n, p, q, r)
            result['pearson'][i, j] = pearson_graph(g1, g2)
            result['gcorr'][i, j] = gcorr(g1, g2, z)

    with open('outputs/sim_test_stat_er.pkl', 'wb') as f:
        pickle.dump(result, f)


def run_sbm():
    n = [100, 50]
    p = [[0.7, 0.3], [0.3, 0.5]]
    q = [[0.2, 0.5], [0.5, 0.4]]
    z = np.repeat([0, 1], n)
    num_corr = 10
    num_rep = 500
    delta = 0.1
    corr = np.linspace(get_lowest_r_sbm(p, q) + delta, get_highest_r_sbm(p, q) - delta, num_corr)
    result = {
        'true': corr,
        'pearson': np.zeros((num_corr, num_rep)),
        'gcorr': np.zeros((num_corr, num_rep))
    }
    for i, r in enumerate(corr):
        for j in range(num_rep):
            g1, g2 = sbm_corr_diffmarg(n, p, q, r)
            result['pearson'][i, j] = pearson_graph(g1, g2)
            result['gcorr'][i, j] = gcorr(g1, g2, z)

    with open('outputs/sim_test_stat_sbm.pkl', 'wb') as f:
        pickle.dump(result, f)


if (args.sim == 'er'):
    run_er()
elif (args.sim == 'sbm'):
    run_sbm()


