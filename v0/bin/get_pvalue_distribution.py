from utils import *
from simulations import *
from mgcpy.independence_tests.mgc.mgc import MGC
from mgcpy.independence_tests.dcorr import DCorr
import pickle
import time

num_vertices = 50
pval_mc = 100
# test = MGC(compute_distance_matrix=identity)
# test = DCorr(compute_distance_matrix=identity)

start_time = time.time()

'''
result = pvalue_distribution(null_rdpg, test, to_distance_mtx,
                             pval_mc=pval_mc,
                             n=num_vertices,
                             nonlinear_transform=linear)
with open('data/rdpg_pvalue_null_{}.pkl'.format(test.get_name()), 'wb') as f:
    pickle.dump(result, f)


result = pvalue_distribution(nonlinear_rdpg, test, to_distance_mtx,
                             pval_mc=pval_mc,
                             n=num_vertices,
                             nonlinear_transform=linear)
with open('data/rdpg_pvalue_linear_{}.pkl'.format(test.get_name()), 'wb') as f:
    pickle.dump(result, f)

result = pvalue_distribution(nonlinear_rdpg, test, to_distance_mtx,
                             pval_mc=pval_mc,
                             n=num_vertices,
                             nonlinear_transform=diff_square)
with open('data/rdpg_pvalue_diff_square_{}.pkl'.format(test.get_name()), 'wb') as f:
    pickle.dump(result, f)

result = pvalue_distribution(nonlinear_rdpg, test, to_distance_mtx,
                             pval_mc=pval_mc,
                             n=num_vertices,
                             nonlinear_transform=exponential)
with open('data/rdpg_pvalue_exp_{}.pkl'.format(test.get_name()), 'wb') as f:
    pickle.dump(result, f)
'''

test = MGC()

rho_ER_null = pvalue_distribution(rho_ER, test, triu_no_diag, is_null=True,
                                  pval_mc=pval_mc,
                                  rho=0, p=0.5, n=20)
with open('data/rho_ER_pvalue_mgc_null.pkl', 'wb') as f:
    pickle.dump(rho_ER_null, f)


rho_ER_alt = pvalue_distribution(rho_ER, test, triu_no_diag,
                                 pval_mc=pval_mc,
                                 rho=0.7, p=0.5, n=20)
with open('data/rho_ER_pvalue_mgc_alt.pkl', 'wb') as f:
    pickle.dump(rho_ER_alt, f)

test = MGC(compute_distance_matrix=identity)

rho_SBM_null = pvalue_distribution(rho_sbm, test, to_distance_mtx, is_null=True,
                                   pval_mc=pval_mc,
                                   rho=0, k=2, L=sbm_params(), n=50)
with open('data/rho_SBM_pvalue_mgc_null.pkl', 'wb') as f:
    pickle.dump(rho_SBM_null, f)

rho_SBM_alt = pvalue_distribution(rho_sbm, test, to_distance_mtx,
                                  pval_mc=pval_mc,
                                  rho=0.5, k=2, L=sbm_params(), n=50)
with open('data/rho_SBM_pvalue_mgc_alt.pkl', 'wb') as f:
    pickle.dump(rho_SBM_alt, f)

print('took {} minutes'.format((time.time() - start_time) / 60))
