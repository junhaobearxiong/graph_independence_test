from utils import *
from simulations import *
from mgcpy.independence_tests.mgc.mgc import MGC
from mgcpy.independence_tests.rv_corr import RVCorr
from mgcpy.independence_tests.dcorr import DCorr
import pickle
import multiprocessing as mp
import time


def power_rdpg_parallel(params):
    n = params['n']
    transformation = params['transformation']
    mc = 500

    #test = MGC(compute_distance_matrix=identity)
    test = DCorr(compute_distance_matrix=identity)
    transform_func = to_distance_mtx

    #test = RVCorr(which_test='pearson')
    #transform_func = triu_no_diag

    print('n={}, transformation={} started using {}'.format(n,
                                                            transformation,
                                                            test.get_name()))
    if transformation == 'null':
        result = power(test, null_rdpg, transform_func=transform_func,
                       is_null=True, mc=mc,
                       n=n, nonlinear_transform=linear)
    elif transformation == 'linear':
        result = power(test, nonlinear_rdpg, transform_func=transform_func,
                       mc=mc, n=n, nonlinear_transform=linear)
    elif transformation == 'diff_square':
        result = power(test, nonlinear_rdpg, transform_func=transform_func,
                       mc=mc, n=n, nonlinear_transform=diff_square)
    elif transformation == 'exp':
        result = power(test, nonlinear_rdpg, transform_func=transform_func,
                       mc=mc, n=n, nonlinear_transform=exponential)
    elif transformation == 'sine':
        result = power(test, nonlinear_rdpg, transform_func=transform_func,
                       mc=mc, n=n, nonlinear_transform=sine)
    elif transformation == 'mild_diff_square':
        result = power(test, nonlinear_rdpg, transform_func=transform_func,
                       mc=mc, n=n, nonlinear_transform=mild_diff_square)
    print('n={}, transformation={} completed'.format(n, transformation))
    return (n, result, transformation)


params_list = []
num_samples = [10, 50, 100, 500]
transformation = ['null', 'linear', 'diff_square', 'exp', 'sine',
                  'mild_diff_square']
for t in transformation:
    for n in num_samples:
        params = {'n': n, 'transformation': t}
        params_list.append(params)

start_time = time.time()
with mp.Pool(mp.cpu_count() - 1) as p:
    outputs = p.map(power_rdpg_parallel, params_list)
with open('data/rdpg_power_dcorr.pkl', 'wb') as f:
    pickle.dump(outputs, f)
print('took {} minutes'.format((time.time() - start_time) / 60))
