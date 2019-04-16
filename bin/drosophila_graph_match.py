# - standard imports
from utils import sort_graph
import numpy as np
import os
import pickle

# - graspy datasets and utils
# from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.utils import symmetrize

#- rpy2
from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


# - get data
C_left = np.loadtxt('data/drosophila/left_adjacency.csv', dtype=int)
C_right = np.loadtxt('data/drosophila/right_adjacency.csv', dtype=int)
C_left = symmetrize(C_left)
C_right = symmetrize(C_right)
A_left = (C_left > 0)
A_right = (C_right > 0)

# - Define sgm functions
r_source = ro.r['source']
r_source("bin/sgm.interface.R")  # path)

cmd = """
        fn <- function(g1_matrix, g2_matrix, seeds, reps) {
            sgm.interface(g1_matrix, g2_matrix, seeds, reps)
        }
        """
sgm_fn = ro.r(cmd)

# - Set number of graph matching iterations
reps = 1
ro.r.assign("reps", reps)

# - No seeds
seeds = ro.rinterface.NULL
ro.r.assign("seeds", seeds)
'''
#- Do unseeded, weighted graph matching
nr, nc = C_left.shape
C_left_R = ro.r.matrix(C_left, nrow=nr, ncol=nc)
ro.r.assign("C_left_R", C_left_R)

nr, nc = C_right.shape
C_right_R = ro.r.matrix(C_right, nrow=nr, ncol=nc)
ro.r.assign("C_right_R", C_right_R)

weighted_P = sgm_fn(C_left, C_right, seeds, reps)

#- Do unseeded, unweighted graph matching
nr, nc = A_left.shape
A_left_R = ro.r.matrix(A_left, nrow=nr, ncol=nc)
ro.r.assign("A_left_R", A_left_R)

nr, nc = A_right.shape
A_right_R = ro.r.matrix(A_right, nrow=nr, ncol=nc)
ro.r.assign("A_right_R", A_right_R)

unweighted_P = sgm_fn(A_left, A_right, seeds, reps)

#- Dump unseeded permutation matrices
import _pickle as pickle
pickle.dump(weighted_P, open('weighted_P.pkl', 'wb'))
pickle.dump(unweighted_P, open('unweighted_P.pkl', 'wb'))
'''
# - Process labels
left_labels = np.loadtxt('data/drosophila/left_cell_labels.csv', dtype=str)
n_L = len(left_labels)
right_labels = np.loadtxt('data/drosophila/right_cell_labels.csv', dtype=str)
n_R = len(right_labels)
unique_L, counts_L = np.unique(left_labels, return_counts=True)
unique_R, counts_R = np.unique(right_labels, return_counts=True)
'''
# - Permute adjacencies so that seeds are in first n_seeds
P_left = np.eye(len(left_labels))
seeds_left = counts_L[0]
for i in range(counts_L[2]):
    e_nonseed = np.zeros(n_L)
    e_nonseed[seeds_left] = 1

    e_seed = np.zeros(n_L)
    e_seed[np.sum(counts_L[:2]) + i] = 1

    P_left[seeds_left, :] = e_seed
    P_left[np.sum(counts_L[:2]) + i, :] = e_nonseed
    seeds_left += 1

P_right = np.eye(len(right_labels))
seeds_right = counts_R[0]
for i in range(counts_R[2]):
    e_nonseed = np.zeros(n_R)
    e_nonseed[seeds_right] = 1

    e_seed = np.zeros(n_R)
    e_seed[np.sum(counts_R[:2]) + i] = 1

    P_right[seeds_right, :] = e_seed
    P_right[np.sum(counts_R[:2]) + i, :] = e_nonseed
    seeds_right += 1

# - Save new matrices
C_left_seeds = P_left @ C_left @ P_left.T
C_right_seeds = P_right @ C_right @ P_right.T
'''

left_seeds = np.zeros(left_labels.size)
left_seeds[:counts_L[0]] = np.where(left_labels == 'I')[0]
left_seeds[counts_L[0]:counts_L[0]+counts_L[2]] = np.where(left_labels == 'O')[0]
left_seeds[counts_L[0]+counts_L[2]:counts_L[0]+counts_L[2] +
           counts_L[1]] = np.where(left_labels == 'K')[0]
left_seeds[counts_L[0]+counts_L[2]+counts_L[1]:] = np.where(left_labels == 'P')[0]

right_seeds = np.zeros(right_labels.size)
right_seeds[:counts_R[0]] = np.where(right_labels == 'I')[0]
right_seeds[counts_R[0]:counts_R[0]+counts_R[2]] = np.where(right_labels == 'O')[0]
right_seeds[counts_R[0]+counts_R[2]:counts_R[0]+counts_R[2] +
            counts_R[1]] = np.where(right_labels == 'K')[0]
right_seeds[counts_R[0]+counts_R[2]+counts_R[1]:] = np.where(right_labels == 'P')[0]

C_left_seeds = sort_graph(C_left, left_seeds)
C_right_seeds = sort_graph(C_right, right_seeds)

pickle.dump(C_left_seeds, open('data/drosophila/weighted_left_permuted.pkl', 'wb'))
pickle.dump(C_right_seeds, open('data/drosophila/weighted_right_permuted.pkl', 'wb'))

A_left_seeds = sort_graph(A_left, left_seeds)
A_right_seeds = sort_graph(A_right, right_seeds)

pickle.dump(A_left_seeds, open('data/drosophila/unweighted_left_permuted.pkl', 'wb'))
pickle.dump(A_right_seeds, open('data/drosophila/unweighted_right_permuted.pkl', 'wb'))

seeds1 = np.zeros((np.sum(counts_L[counts_L == counts_R]), 2))
seeds1[:, 0] = np.arange(1, np.sum(counts_L[counts_L == counts_R]) + 1)
seeds1[:, 1] = np.arange(1, np.sum(counts_L[counts_L == counts_R]) + 1)
ro.r.assign("seeds1", seeds1)

# - Do seeded, weighted graph matching
nr, nc = C_left_seeds.shape
C_left_seeds_R = ro.r.matrix(C_left_seeds, nrow=nr, ncol=nc)
ro.r.assign("C_left_seeds_R", C_left_seeds_R)

nr, nc = C_right_seeds.shape
C_right_seeds_R = ro.r.matrix(C_right_seeds, nrow=nr, ncol=nc)
ro.r.assign("C_right_seeds_R", C_right_seeds_R)

seeds_weighted_P = sgm_fn(C_left_seeds, C_right_seeds, seeds1, reps)

# - Do seeded, unweighted graph matching
nr, nc = A_left_seeds.shape
A_left_seeds_R = ro.r.matrix(A_left_seeds, nrow=nr, ncol=nc)
ro.r.assign("A_left_seeds_R", A_left_seeds_R)

nr, nc = A_right.shape
A_right_seeds_R = ro.r.matrix(A_right_seeds, nrow=nr, ncol=nc)
ro.r.assign("A_right_seeds_R", A_right_seeds_R)

seeds_unweighted_P = sgm_fn(A_left_seeds, A_right_seeds, seeds1, reps)

# - Dump seeded permutation matrices
pickle.dump(seeds_weighted_P, open('data/drosophila/seeds_weighted_P.pkl', 'wb'))
pickle.dump(seeds_unweighted_P, open('data/drosophila/seeds_unweighted_P.pkl', 'wb'))
