import numpy as np
import pickle
import argparse

from graspologic.simulations.simulations_corr import er_corr, sbm_corr

from simulations import er_corr_diffmarg, sbm_corr_diffmarg
from core import gcorr, block_permutation, community_estimation
from utils import pearson_graph, vertex_permutation, power_estimation, pearson_exact_pvalue

parser = argparse.ArgumentParser()
parser.add_argument("setting", help="which setting to run: `er`, `sbm_diffmarg`, `sbm_diffblock` or `sbm_estblock`")
parser.add_argument("r", type=float, help="correlation")
parser.add_argument("num_reps", type=int, nargs="?", default=50)
args = parser.parse_args()

num_reps = args.num_reps
r = args.r
setting = args.setting
num_vertices = np.linspace(10, 100, 10, dtype=int)
alpha = 0.05

tests = [
    'pearson_vertex_perm',
    'pearson_block_perm',
    'gcorr_block_perm' 
]

power = {}
test_stats_null = {}
test_stats_alt = {}
for t in tests:
    power[t] = np.zeros(num_vertices.size)
    test_stats_null[t] = np.zeros((num_vertices.size, num_reps))
    test_stats_alt[t] = np.zeros((num_vertices.size, num_reps))

power['pearson_exact'] = np.zeros(num_vertices.size)
pearson_exact_pvals = np.zeros((num_vertices.size, num_reps))


if 'sbm' in setting:
    p = [[0.7, 0.3], [0.3, 0.7]]
    q = [[0.2, 0.5], [0.5, 0.2]]
else:
    p = 0.7
    q = 0.3

for i in range(num_vertices.size):
    for rep in range(num_reps):
        if setting == 'sbm_diffmarg':
            n = [int(num_vertices[i] / 2), int(num_vertices[i] / 2)]
            Z = np.repeat([0, 1], n) 
        elif setting == 'sbm_diffblock':
            n = [int(num_vertices[i] * 0.7), int(num_vertices[i] * 0.3)]
            Z = np.repeat([0, 1], n) 
        elif setting == 'sbm_estblock':
            n = [int(num_vertices[i] * 0.7), int(num_vertices[i] * 0.3)]
        else:
            n = num_vertices[i]
            Z = np.repeat([0], n)

        # TODO: could add correlated gaussian graphs here
        if 'sbm' in setting:
            G1, G2 = sbm_corr_diffmarg(n, p, q, r)
        else:
            G1, G2 = er_corr_diffmarg(n, p, q, r)

        if setting == 'sbm_estblock':
            Z = community_estimation(G1, G2, min_components=5)

        G2_vertex_perm = vertex_permutation(G2)
        G2_block_perm = block_permutation(G2, Z)
        test_stats_null['pearson_vertex_perm'][i, rep] = pearson_graph(G1, G2_vertex_perm)
        test_stats_alt['pearson_vertex_perm'][i, rep] = pearson_graph(G1, G2)
        test_stats_null['pearson_block_perm'][i, rep] = pearson_graph(G1, G2_block_perm)
        test_stats_alt['pearson_block_perm'][i, rep] = pearson_graph(G1, G2)
        test_stats_null['gcorr_block_perm'][i, rep] = gcorr(G1, G2_block_perm, Z)
        test_stats_alt['gcorr_block_perm'][i, rep] = gcorr(G1, G2, Z)
        pearson_exact_pvals[i, rep] = pearson_exact_pvalue(G1, G2)

for i in range(num_vertices.size):
    for t in tests:
        power[t][i] = power_estimation(test_stats_null[t][i, :], test_stats_alt[t][i, :], alpha)
    power['pearson_exact'][i] = np.where(pearson_exact_pvals[i, :] < alpha)[0].size / num_reps


with open('outputs/sim_power_bernoulli_{}_r{}.pkl'.format(args.setting, r), 'wb') as f:
    pickle.dump(power, f)