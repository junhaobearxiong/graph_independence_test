import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing as mp
import sys

from mgcpy.independence_tests.mgc.mgc import MGC

from utils import to_distance_mtx, pvalue, identity

from load_data import read_dwi, read_fmri

subject_list = ['00254{}'.format(i) for i in np.arange(27, 57)]
session_list = [i for i in range(1, 11)]


def fill_inputs(case):
    inputs = []
    if case == 1:
        session_num = 1
        for subject_id in subject_list:
            inputs.append((read_dwi(subject_id, session_num),
                           read_fmri(subject_id, session_num)))


def pvalue_parallel(param):
    A = param[0]
    B = param[1]
    mgc = MGC(compute_distance_matrix=identity)
    pval = pvalue(A, B, mgc, to_distance_mtx, block_est_repeats=10)
    return pval


def main(argv):
    case = int(argv[0])
    inputs = fill_inputs(case)

    with mp.Pool(mp.cpu_count() - 1) as p:
        pvalues = p.map(pvalue_parallel, inputs)

    with open('results/mri_pvalues_case_{}.pkl'.format(case), 'wb') as f:
        pickle.dump(pvalues, f)


if __name__ == '__main__':
    main(sys.argv[1:])
