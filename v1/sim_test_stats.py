import numpy as np
import pickle

from simulations import rho_sbm
from utils import triu_no_diag, to_distance_mtx

from scipy.stats import pearsonr
from mgcpy.independence_tests.dcorr import DCorr

params = {
	'er': {
		'n': 100,
		'k': [100],
		'P1': np.array([[0.5]]),
		'P2': np.array([[0.5]]),
		'nmc': 500,
		'rho_arr': np.around(np.linspace(-0.4, 0.9, 14), 1)
	},
	'er_marg': {
		'n': 100,
		'k': [100],
		'P1': np.array([[0.7]]),
		'P2': np.array([[0.2]]),
		'nmc': 500,
		'rho_arr': np.around(np.linspace(-0.4, 0.3, 8), 1)
	},
	'sbm': {
		'n': 100,
		'k': [50, 50],
		'P1': np.array([[0.7, 0.3], [0.3, 0.7]]),
		'P2': np.array([[0.7, 0.3], [0.3, 0.7]]),
		'nmc': 500,
		'rho_arr': np.around(np.linspace(-0.3, 0.9, 14), 1)
	},
	'sbm_marg': {
		'n': 100,
		'k': [50, 50],
		'P1': np.array([[0.7, 0.3], [0.3, 0.7]]),
		'P2': np.array([[0.2, 0.5], [0.5, 0.2]]),
		'nmc': 500,
		'rho_arr': np.around(np.linspace(-0.6, 0.3, 10), 1)
	}
}

with open('results/sim_test_stats_params.pkl', 'wb') as f:
	pickle.dump(params, f)

def get_test_stats(param):
	result = {
		'pearson': np.zeros((param['rho_arr'].size, param['nmc'])),
		'dcorr': np.zeros((param['rho_arr'].size, param['nmc']))
	}
	
	for i, rho in enumerate(param['rho_arr']):
		for j in range(param['nmc']):
			A, B = rho_sbm(rho, param['k'], param['P1'], param['P2'], param['n']) 
			
			test_stats, _ = pearsonr(triu_no_diag(A), triu_no_diag(B))
			result['pearson'][i, j] = test_stats

			test = DCorr()
			test_stats, _ = test.test_statistic(to_distance_mtx(A), to_distance_mtx(B))
			result['dcorr'][i, j] = test_stats

	return result

output = {}
for sim_name, param in params.items():
	output[sim_name] = get_test_stats(param)

with open('results/sim_test_stats.pkl', 'wb') as f:
	pickle.dump(output, f)