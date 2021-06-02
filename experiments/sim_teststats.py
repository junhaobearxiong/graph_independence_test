import numpy as np
import pickle
from simulations import *
from core import (
    gcorr,
    pearson_graph,
    community_estimation
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sim", help="which setting to run: `er`, `sbm_diffmarg`, `sbm_diffblock` or `sbm_estblock`")
args = parser.parse_args()

num_corr = 10
num_reps = 500
delta = 0.1

# initialize simulation parameters
if 'sbm' in args.sim:
    p = [[0.7, 0.3], [0.3, 0.7]]
    q = [[0.2, 0.5], [0.5, 0.2]]
    corr = np.linspace(get_lowest_r_sbm(p, q) + delta, get_highest_r_sbm(p, q) - delta, num_corr)
else:
    p = 0.7
    q = 0.3
    corr = np.linspace(get_lowest_r(p, q) + delta, get_highest_r(p, q) - delta, num_corr)

result = {
    'true': corr,
    'pearson': np.zeros((num_corr, num_reps)),
    'gcorr': np.zeros((num_corr, num_reps)),
    'gcorr_unpooled': np.zeros((num_corr, num_reps))
}

for i, r in enumerate(corr):
    print('{} in {}'.format(i + 1, num_corr))
    for j in range(num_reps):
        if args.sim == 'sbm_diffmarg':
            n = [50, 50]
            Z = np.repeat([0, 1], n) 
        elif args.sim == 'sbm_diffblock':
            n = [70, 30]
            Z = np.repeat([0, 1], n) 
        elif args.sim == 'sbm_estblock':
            n = [70, 30]
        elif args.sim == 'er':
            n = 100
            Z = np.repeat([0], n)

        if 'sbm' in args.sim:
            G1, G2 = sbm_corr_diffmarg(n, p, q, r)
        else:
            G1, G2 = er_corr_diffmarg(n, p, q, r)

        if args.sim == 'sbm_estblock':
            Z = community_estimation(G1, G2, min_components=5)

        result['pearson'][i, j] = pearson_graph(G1, G2)
        result['gcorr'][i, j] = gcorr(G1, G2, Z, pooled_variance=True)
        result['gcorr_unpooled'][i, j] = gcorr(G1, G2, Z, pooled_variance=False)


with open('outputs/sim_teststat_{}.pkl'.format(args.sim), 'wb') as f:
    pickle.dump(result, f)

