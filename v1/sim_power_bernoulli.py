import numpy as np
import multiprocessing as mp
import pickle
import sys

from simulations import rho_sbm
from utils import pearson_power, dcorr_power, ttest_power

from scipy.stats import pearsonr

from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.mgc import MGC


n_arr = np.linspace(10, 100, 10, dtype=int)
rho_arr = np.array([0, 0.1, -0.1])

params = {
	'er': {
		'given_blocks': True,
		'k': [[n] for n in n_arr],
		'P1': np.array([[0.5]]),
		'P2': np.array([[0.5]])
	},
	'er_marg': {
		'given_blocks': True,
		'k': [[n] for n in n_arr],
		'P1': np.array([[0.7]]),
		'P2': np.array([[0.2]])
	},
	'sbm': {
		'given_blocks': True,
		'k': [[int(n/2), int(n/2)] for n in n_arr],
		'P1': np.array([[0.7, 0.3], [0.3, 0.7]]),
		'P2': np.array([[0.7, 0.3], [0.3, 0.7]])
	},
	'sbm_marg': {
		'given_blocks': True,
		'k': [[int(n/2), int(n/2)] for n in n_arr],
		'P1': np.array([[0.7, 0.3], [0.3, 0.7]]),
		'P2': np.array([[0.2, 0.5], [0.5, 0.2]])
	},
	'sbm_marg_est': {
		'given_blocks': False,
		'k': [[int(n/2), int(n/2)] for n in n_arr],
		'P1': np.array([[0.7, 0.3], [0.3, 0.7]]),
		'P2': np.array([[0.2, 0.5], [0.5, 0.2]])
	},
	'sbm_diff_block': {
		'given_blocks': False,
		'k': [[int(0.7*n), n-int(0.7*n)] for n in n_arr],
		'P1': np.array([[0.7, 0.3], [0.3, 0.7]]),
		'P2': np.array([[0.2, 0.5], [0.5, 0.2]])
	}
}


def get_power(sim_name, param):
	result = {
	    'pearson': np.zeros((rho_arr.shape[0], n_arr.shape[0])),
    	'dcorr': np.zeros((rho_arr.shape[0], n_arr.shape[0])),
    	'mgc': np.zeros((rho_arr.shape[0], n_arr.shape[0]))
	}
	if sim_name not in ['sbm_marg_est', 'sbm_diff_block']: 
		 result['pearson_exact'] = np.zeros((rho_arr.shape[0], n_arr.shape[0]))

	for i, rho in enumerate(rho_arr):
		for j, n in enumerate(n_arr):
			k = param['k'][j]
			if param['given_blocks']:
				blocks = np.repeat(np.arange(len(k)), n//len(k))
			else:
				blocks = None
			if sim_name not in ['sbm_marg_est', 'sbm_diff_block']:
				result['pearson_exact'][i, j] = ttest_power(rho_sbm, rho=rho, n=n, k=k, AL=param['P1'], BL=param['P2'])
			result['pearson'][i, j] = pearson_power(rho_sbm, given_blocks=param['given_blocks'], blocks=blocks, 
				rho=rho, n=n, k=k, AL=param['P1'], BL=param['P2'])
			test = DCorr()
			result['dcorr'][i, j] = dcorr_power(test, rho_sbm, given_blocks=param['given_blocks'], blocks=blocks,
				rho=rho, n=n, k=k, AL=param['P1'], BL=param['P2'])
			test = MGC()
			result['mgc'][i, j] = dcorr_power(test, rho_sbm, given_blocks=param['given_blocks'], blocks=blocks,
				rho=rho, n=n, k=k, AL=param['P1'], BL=param['P2'])
			print('finish rho={}, n={}'.format(rho, n))

	return result


output = {}
for sim_name, param in params.items():
	output[sim_name] = get_power(sim_name, param)
	print('finish {}'.format(sim_name))

with open('results/sim_power_bernoulli.pkl', 'wb') as f:
	pickle.dump(output, f)