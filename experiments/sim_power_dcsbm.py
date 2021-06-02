import numpy as np
import argparse
import pickle
from tqdm import tqdm

from graspologic.simulations import sbm_corr
from graspologic.models import DCSBMEstimator

from core import block_permutation, gcorr, power_estimation, community_estimation, gcorr_dcsbm
from simulations import dcsbm_corr
from utils import off_diag


parser = argparse.ArgumentParser()
parser.add_argument('sim', help='`sbm` or `dcsbm`')
parser.add_argument('rho', type=float)
args = parser.parse_args()

if args.sim not in ['sbm', 'dcsbm']:
    raise ValueError('{} not implemented'.format(args.sim))
if args.rho > 1 or args.rho < -1:
    raise ValueError('rho needs to be >= -1 and <= 1')


# settings
num_vertices = np.linspace(20, 100, 9, dtype=int)
num_reps = 500
alpha = 0.05
p = [[0.7, 0.3], [0.3, 0.7]]
max_comm = 5


# set up structures to save output
tests = [
    'gcorr_block_perm',
    'gcorr_param_bootstrap',
    'gcorrDC_block_perm',
    'gcorrDC_param_bootstrap'
]

power = {}
test_stats_null = {}
test_stats_alt = {}
for t in tests:
    power[t] = np.zeros(num_vertices.size)
    test_stats_null[t] = np.zeros((num_vertices.size, num_reps))
    test_stats_alt[t] = np.zeros((num_vertices.size, num_reps))


# calculate test statistics under null and alternative
for i in range(num_vertices.size):
    print('rep: {}'.format(i))
    for rep in tqdm(range(num_reps)):
        n = [int(num_vertices[i] / 2), int(num_vertices[i] / 2)]

        if args.sim == 'sbm':
            G1, G2 = sbm_corr(n, p, args.rho)
        elif args.sim == 'dcsbm':
            theta = np.linspace(100, 1, n[0])
            theta /= theta.sum()
            theta = np.concatenate([theta, theta])
            G1, G2 = dcsbm_corr(n, p, args.rho, theta)

        # null by block permutation
        Z = community_estimation(G1, G2, min_components=max_comm)
        # Z = np.repeat([0, 1], n)
        G2_block_perm = block_permutation(G2, Z)

        # null by parametric bootstrap
        G1_dcsbm = DCSBMEstimator(directed=False).fit(G1)
        G2_dcsbm = DCSBMEstimator(directed=False).fit(G2)
        G1_bootstrap = G1_dcsbm.sample()[0]
        G2_bootstrap = G2_dcsbm.sample()[0]

        test_stats_alt['gcorr_block_perm'][i, rep] = gcorr(G1, G2, Z)
        test_stats_null['gcorr_block_perm'][i, rep] = gcorr(G1, G2_block_perm, Z)
        test_stats_alt['gcorr_param_bootstrap'][i, rep] = gcorr(G1, G2, Z)
        test_stats_null['gcorr_param_bootstrap'][i, rep] = gcorr(G1_bootstrap, G2_bootstrap, Z)
        test_stats_alt['gcorrDC_param_bootstrap'][i, rep] = gcorr_dcsbm(G1, G2, max_comm)
        test_stats_null['gcorrDC_param_bootstrap'][i, rep] = gcorr_dcsbm(G1_bootstrap, G2_bootstrap, max_comm)
        test_stats_alt['gcorrDC_block_perm'][i, rep] = gcorr_dcsbm(G1, G2, max_comm)
        test_stats_null['gcorrDC_block_perm'][i, rep] = gcorr_dcsbm(G1, G2_block_perm, max_comm)


# compute power
for i in range(num_vertices.size):
    for t in tests:
        power[t][i] = power_estimation(test_stats_null[t][i, :], test_stats_alt[t][i, :], alpha)


with open('outputs/sim_power_dc_{}_rho{}.pkl'.format(args.sim, args.rho), 'wb') as f:
    pickle.dump(power, f)
