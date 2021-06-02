import numpy as np
import pickle
from tqdm import tqdm
 
from simulations import dcsbm_corr
from core import gcorr_dcsbm, gcorr


n = [50, 50]
p = [[0.7, 0.3], [0.3, 0.7]]
z = np.repeat([0, 1], n)
num_corr = 10
num_rep = 500
corr = np.linspace(0, 0.9, num_corr)

maxcomm = 5
theta = np.linspace(100, 1, n[0])
theta /= theta.sum()
theta = np.concatenate([theta, theta])


result = {
    'true': corr,
    'gcorr': np.zeros((num_corr, num_rep)),
    'gcorr_dcsbm_pooled': np.zeros((num_corr, num_rep)),
    'gcorr_dcsbm': np.zeros((num_corr, num_rep))
}

for i, r in enumerate(corr):
    print('iteration {} in {}: r = {}'.format(i + 1, num_corr, r))
    for j in tqdm(range(num_rep)):
        g1, g2 = dcsbm_corr(n, p, r, theta)
        result['gcorr'][i, j] = gcorr(g1, g2, z)
        result['gcorr_dcsbm_pooled'][i, j] = gcorr_dcsbm(g1, g2, maxcomm, pooled_variance=True)
        result['gcorr_dcsbm'][i, j] = gcorr_dcsbm(g1, g2, maxcomm, pooled_variance=False)

with open('outputs/sim_teststat_dcsbm.pkl', 'wb') as f:
    pickle.dump(result, f)