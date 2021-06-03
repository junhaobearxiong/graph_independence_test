import numpy as np
import pickle
from tqdm import tqdm
import argparse
 
from simulations import dcsbm_corr
from core import gcorr_dcsbm, gcorr, community_estimation

parser = argparse.ArgumentParser()
parser.add_argument("sim", help="which setting to run: `givenblock`, `diffblock`, `estblock`")
args = parser.parse_args()

p = [[0.7, 0.3], [0.3, 0.7]]
num_corr = 10
num_rep = 500
corr = np.linspace(0, 0.9, num_corr)

if args.sim == 'givenblock':
    n = [50, 50]
elif args.sim == 'diffblock' or args.sim == 'estblock':
    n = [70, 30]
else:
    raise ValueError('{} is not implemented'.format(args.sim))

ztrue = np.repeat([0, 1], n)

maxcomm = 5
theta1 = np.linspace(100, 1, n[0])
theta2 = np.linspace(100, 1, n[1])
# make sure the weights in each block sums to 1
theta1 /= theta1.sum()
theta2 /= theta2.sum()
theta = np.concatenate([theta1, theta2])

result = {
    'true': corr,
    'gcorr_pooled': np.zeros((num_corr, num_rep)),
    'gcorr_unpooled': np.zeros((num_corr, num_rep)),
    'gcorr_dcsbm_pooled': np.zeros((num_corr, num_rep)),
    'gcorr_dcsbm_unpooled': np.zeros((num_corr, num_rep))
}

for i, r in enumerate(corr):
    print('iteration {} in {}: r = {}'.format(i + 1, num_corr, r))
    for j in tqdm(range(num_rep)):
        g1, g2 = dcsbm_corr(n, p, r, theta)
        if args.sim == 'estblock':
            # note `community_estimation` uses MASE
            # whereas `graspologic.DCSBM` uses LSE
            zest = community_estimation(g1, g2, maxcomm)
            result['gcorr_pooled'][i, j] = gcorr(g1, g2, zest, pooled_variance=True)
            result['gcorr_unpooled'][i, j] = gcorr(g1, g2, zest, pooled_variance=False)
            result['gcorr_dcsbm_pooled'][i, j] = gcorr_dcsbm(g1, g2, maxcomm, pooled_variance=True, Z=None)
            result['gcorr_dcsbm_unpooled'][i, j] = gcorr_dcsbm(g1, g2, maxcomm, pooled_variance=False, Z=None)
        else:
            result['gcorr_pooled'][i, j] = gcorr(g1, g2, ztrue, pooled_variance=True)
            result['gcorr_unpooled'][i, j] = gcorr(g1, g2, ztrue, pooled_variance=False)
            result['gcorr_dcsbm_pooled'][i, j] = gcorr_dcsbm(g1, g2, maxcomm, pooled_variance=True, Z=ztrue)
            result['gcorr_dcsbm_unpooled'][i, j] = gcorr_dcsbm(g1, g2, maxcomm, pooled_variance=False, Z=ztrue)

with open('outputs/sim_teststat_dcsbm_{}.pkl'.format(args.sim), 'wb') as f:
    pickle.dump(result, f)