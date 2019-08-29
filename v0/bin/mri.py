import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing as mp
import sys

from mgcpy.independence_tests.mgc.mgc import MGC

from utils import to_distance_mtx, pvalue, identity

from load_data import read_dwi, read_fmri, pairs_from_list

subject_list = ['00254{}'.format(i) for i in np.arange(27, 57)]
session_list = [i for i in range(1, 11)]


def fill_inputs(case):
    inputs = []
    session_num = 1
    if case == 1:
        # dwi vs. fmri for a single session
        for subject_id in subject_list:
            inputs.append((read_dwi(subject_id, session_num),
                           read_fmri(subject_id, session_num)))
    elif case == 2:
        # dwi subject-wise comparison for a single session
        for subject1, subject2 in pairs_from_list(subject_list):
            inputs.append((read_dwi(subject1, session_num),
                           read_dwi(subject2, session_num)))
    elif case == 3:
        # dwi or fmri subject-wise comparison for a single session
        for subject1, subject2 in pairs_from_list(subject_list):
            # determine which subject is dwi and which is fmri
            choice = np.random.binomial(n=1, p=0.5, size=1)[0]
            if choice:
                inputs.append((read_dwi(subject1, session_num),
                               read_fmri(subject2, session_num)))
            else:
                inputs.append((read_fmri(subject1, session_num),
                               read_dwi(subject2, session_num)))
    elif case == 4:
        # dwi vs. fmri for all sessions
        for num in session_num:
            for subject_id in subject_list:
                inputs.append((read_dwi(subject_id, num),
                               read_fmri(subject_id, num)))
    else:
        raise ValueError('case {} has not been implemented'.format(case))
    return inputs


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
