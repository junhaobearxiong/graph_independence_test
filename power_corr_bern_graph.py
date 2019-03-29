from utils import *
from simulations import *
from mgcpy.independence_tests.mgc.mgc import MGC
from mgcpy.independence_tests.rv_corr import RVCorr
from mgcpy.independence_tests.dcorr import DCorr
import pickle
import multiprocessing as mp
import time


def power_corr_bern_graph_parallel(params):
    rho = params['rho']
    n = params['n']
    setting = params['setting']
    mc = 500

    # test = MGC(compute_distance_matrix=identity)
    # test = DCorr(compute_distance_matrix=identity)
    # transform_func = to_distance_mtx

    test = RVCorr(which_test='pearson')
    transform_func = triu_no_diag

    print('rho={}, n={}, sbm={} started using {}'.format(rho, n,
                                                         setting,
                                                         test.get_name()))
    if rho == -1:
        result = power(test, rho_sbm, transform_func=transform_func,
                       is_null=True, mc=mc,
                       rho=0, k=2, L=sbm_params(setting), n=n)
    else:
        result = power(test, rho_sbm, transform_func=transform_func, mc=mc,
                       rho=rho, k=2, L=sbm_params(setting), n=n)
    print('rho={}, n={}, sbm={} completed'.format(rho, n, setting))
    return (rho, n, result, setting)


params_list = []
num_samples = [10, 15, 20, 25, 30]
rho_arr = [-1, 0.2, 0.5, 0.8]
sbm_setting = ['homog_balanced']
for setting in sbm_setting:
    for i, rho in enumerate(rho_arr):
        for j, num_vertices in enumerate(num_samples):
            params = {'rho': rho, 'n': num_vertices, 'setting': setting}
            params_list.append(params)

start_time = time.time()
with mp.Pool(mp.cpu_count() - 1) as p:
    outputs = p.map(power_corr_bern_graph_parallel, params_list)
with open('data/rho_SBM_power_pearson.pkl', 'wb') as f:
    pickle.dump(outputs, f)
print('took {} minutes'.format((time.time() - start_time) / 60))
