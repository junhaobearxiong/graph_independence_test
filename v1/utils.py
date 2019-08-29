import numpy as np
import math
from tqdm import tqdm
import pandas as pd

from sklearn.mixture import GaussianMixture

from graspy.embed import MultipleASE
from graspy.cluster import GaussianCluster, KMeansCluster
from graspy.utils import symmetrize

# from mgcpy.independence_tests.mgc import MGC


def dcorr_power_two_sided(indept_test, sim_func, mc=500, alpha=0.05, given_blocks=False, blocks=None, **kwargs):
    # power for any test that builds on distance matrices
    # can use dcorr / mgc
    test_stat_null_array = np.zeros(mc)
    test_stat_alt_array = np.zeros(mc)
    for i in range(mc):
        A, B = sim_func(**kwargs)
        if given_blocks:
            block_assignment = blocks
        else:
            block_assignment = estimate_block_assignment(A, B)
        A_null = block_permute(A, block_assignment)
        B_sorted = sort_graph(B, block_assignment)

        test_stat_alt, _ = indept_test.test_statistic(
            matrix_X=to_distance_mtx(A), matrix_Y=to_distance_mtx(B))
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=to_distance_mtx(A_null), matrix_Y=to_distance_mtx(B_sorted))

        test_stat_alt_array[i] = test_stat_alt
        test_stat_null_array[i] = test_stat_null
    c1 = np.sort(test_stat_null_array)[math.floor(alpha/2*mc)]
    c2 = np.sort(test_stat_null_array)[math.ceil((1-alpha/2)*mc)]
    power = np.where((test_stat_alt_array > c2) | (test_stat_alt_array < c1))[0].shape[0] / mc
    return power


def pearson_power_two_sided(indept_test, sim_func, mc=500, alpha=0.05,
                            given_blocks=False, blocks=None, **kwargs):
    # power for any test that uses vectorized matrix as samples
    test_stat_null_array = np.zeros(mc)
    test_stat_alt_array = np.zeros(mc)
    for i in range(mc):
        A, B = sim_func(**kwargs)
        if given_blocks:
            block_assignment = blocks
        else:
            block_assignment = estimate_block_assignment(A, B)
        A_null = block_permute(A, block_assignment)
        B_sorted = sort_graph(B, block_assignment)

        test_stat_alt, _ = indept_test.test_statistic(
            matrix_X=triu_no_diag(A), matrix_Y=triu_no_diag(B))
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=triu_no_diag(A_null), matrix_Y=triu_no_diag(B_sorted))

        test_stat_alt_array[i] = test_stat_alt
        test_stat_null_array[i] = test_stat_null
    c1 = np.sort(test_stat_null_array)[math.floor(alpha/2*mc)]
    c2 = np.sort(test_stat_null_array)[math.ceil((1-alpha/2)*mc)]
    power = np.where((test_stat_alt_array > c2) | (test_stat_alt_array < c1))[0].shape[0] / mc
    return power


def to_undirected(A):
    return np.where(A > 0, 1, 0).astype(float)


def get_null_test_stats(A, B, k_arr, reps):
    mgc = MGC(compute_distance_matrix=identity)
    test_stats_null_arr = np.zeros((k_arr.size, reps))
    for i, k in enumerate(k_arr):
        for r in tqdm(range(reps)):
            block_assignment = estimate_block_assignment(A, B, k=k,
                                                         set_k=True,
                                                         num_repeats=10)
            test_stats_null, _ = mgc.test_statistic(
                to_distance_mtx(block_permute(A, block_assignment)),
                to_distance_mtx(sort_graph(B, block_assignment)))
            test_stats_null_arr[i, r] = test_stats_null
    return test_stats_null_arr


def estimate_block_assignment(A, B, k=10, set_k=False, num_repeats=1,
                              svd='randomized', random_state=None):

    mase = MultipleASE(algorithm='randomized')
    mase.fit([A, B])
    Vhat = mase.latent_left_

    bics = []
    models = []

    # use set number of clusters
    if set_k:
        for rep in range(num_repeats):
            model = GaussianMixture(n_components=k)
            models.append(model.fit(Vhat))
            bics.append(model.bic(Vhat))
    else:
        for rep in range(num_repeats):
            gmm = GaussianCluster(max_components=k, random_state=None)
            models.append(gmm.fit(Vhat))
            bics.append(min(gmm.bic_))
    best_model = models[np.argmin(bics)]
    return best_model.predict(Vhat)


def permute_off_diag(A):
    return np.random.permutation(A.flatten()).reshape(A.shape)


def permute_on_diag(A):
    triu = np.random.permutation(triu_no_diag(A).flatten())
    A_perm = np.zeros_like(A)
    A_perm[np.triu_indices(A.shape[0], 1)] = triu
    A_perm = symmetrize(A_perm)
    return A_perm


def block_permute(A, block_assignment):
    A = sort_graph(A, block_assignment)
    block_assignment = np.sort(block_assignment)
    permuted_A = np.zeros_like(A)
    num_blocks = np.unique(block_assignment).size
    # get the index of the blocks in the upper triangular
    row_idx, col_idx = np.triu_indices(num_blocks)
    for t in range(row_idx.size):
        i = row_idx[t]
        j = col_idx[t]
        block_i_idx = np.where(block_assignment == i)[0]
        block_j_idx = np.where(block_assignment == j)[0]
        block = A[np.ix_(block_i_idx, block_j_idx)]
        # permute only the upper triangular if the block is on the diagonal
        if i == j:
            permuted_block = permute_on_diag(block)
        else:
            permuted_block = permute_off_diag(block)
        permuted_A[np.ix_(block_i_idx, block_j_idx)] = permuted_block
    permuted_A = symmetrize(permuted_A)
    return permuted_A


def _sort_inds(inner_labels, outer_labels):
    sort_df = pd.DataFrame(columns=("inner_labels", "outer_labels"))
    sort_df["inner_labels"] = inner_labels
    if outer_labels is not None:
        sort_df["outer_labels"] = outer_labels
        sort_df.sort_values(
            by=["outer_labels", "inner_labels"], kind="mergesort", inplace=True
        )
        outer_labels = sort_df["outer_labels"]
    inner_labels = sort_df["inner_labels"]
    sorted_inds = sort_df.index.values
    return sorted_inds


def sort_graph(graph, inner_labels):
    inds = _sort_inds(inner_labels, np.ones_like(inner_labels))
    graph = graph[inds, :][:, inds]
    return graph



def non_diagonal(A):
    '''
    Get all the off-diagonal entries of a square matrix
    '''
    return A[np.where(~np.eye(A.shape[0], dtype=bool))]


def to_distance_mtx(A):
    '''
    convert graph to distance matrix
    '''
    distance_mtx_A = 1 - (A / np.max(A))
    # the graph assumes no self loop, so a node is disconnected from itself
    # but in the distance matrices the diagonal entries should always be 0
    # instead of 1
    np.fill_diagonal(distance_mtx_A, 0)
    return distance_mtx_A


def triu_no_diag(A):
    '''
    Get the entries in the upper triangular part of the adjacency matrix (not
    including the diagonal)

    Returns
    --------
    2d array:
        The vectorized upper triangular part of graph A
    '''
    n = A.shape[0]
    iu1 = np.triu_indices(n, 1)
    triu_vec = A[iu1]
    # return triu_vec[:, np.newaxis]
    return triu_vec


def sbm_params(setting='homog_balanced', a=0.7, b=0.5, c=0.3, k=2):
    if setting == 'homog_balanced':
        L = np.array([[a, b], [b, a]])
    elif setting == 'core_peri':
        L = np.array([[a, b], [b, b]])
    elif setting == 'rank_one':
        L = np.array([[np.square(a), a*b], [a*b, np.square(b)]])
    elif setting == 'full_rank':
        L = np.array([[a, b], [b, c]])
    elif setting == 'k_block':
        L = np.zeros((k, k))
        L[np.diag_indices(k)] = a
        L[np.triu_indices(k, 1)] = b
        L = symmetrize(L)
    else:
        raise ValueError('setting {} is not defined'.format(setting))
    return L


def identity(x): return x
