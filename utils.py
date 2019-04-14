import numpy as np
import math
from tqdm import tqdm_notebook as tqdm
import pandas as pd

from sklearn.mixture import GaussianMixture

from graspy.embed import JointRDPG
from graspy.cluster import GaussianCluster, KMeansCluster
from graspy.utils import symmetrize


def binarize(A):
    return np.where(A > 0, 1, 0).astype(float)


def estimate_block_assignment(A, B, k=10, set_k=False, num_repeats=1,
                              svd='randomized', random_state=None):

    jrdpg = JointRDPG(algorithm='randomized')
    jrdpg.fit([A, B])
    Vhat = jrdpg.latent_left_

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


def pvalue(A, B, indept_test, transform_func, k=10, set_k=False, null_mc=500,
           block_est_repeats=1):
    test_stat_alternative, _ = indept_test.test_statistic(
        matrix_X=transform_func(A), matrix_Y=transform_func(B))

    block_assignment = estimate_block_assignment(A, B, k=k, set_k=set_k,
                                                 num_repeats=block_est_repeats)
    B_sorted = sort_graph(B, block_assignment)

    test_stat_null_array = np.zeros(null_mc)
    for j in tqdm(range(null_mc)):
        # A_null is the permuted matrix after being sorted by block assignment
        A_null = block_permute(A, block_assignment)
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=transform_func(A_null), matrix_Y=transform_func(B_sorted))
        test_stat_null_array[j] = test_stat_null

    p_value = np.where(test_stat_null_array > test_stat_alternative)[
        0].shape[0] / test_stat_null_array.shape[0]
    return p_value


'''
def pvalue(A, B, indept_test, transform_func, null_mc=1000):
    test_stat_alternative, _ = indept_test.test_statistic(
        matrix_X=transform_func(A), matrix_Y=transform_func(B))
    test_stat_null_array = np.zeros(null_mc)
    for j in range(null_mc):
        # permutation test to generate the null
        A_null = permute_matrix(A)
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=transform_func(A_null), matrix_Y=transform_func(B))
        test_stat_null_array[j] = test_stat_null

    p_value = np.where(test_stat_null_array > test_stat_alternative)[
        0].shape[0] / test_stat_null_array.shape[0]
    return p_value
'''


def pvalue_distribution(sample_func, indept_test, transform_func, null_mc=1000,
                        pval_mc=500, is_null=False, **kwargs):
    p_value_mc = np.zeros(pval_mc)
    for i in tqdm(range(pval_mc)):
        A, B = sample_func(**kwargs)
        if is_null:
            A = permute_matrix(A)
        p_value_mc[i] = pvalue(A, B, indept_test, transform_func, null_mc)
    return p_value_mc


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
    return triu_vec[:, np.newaxis]


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
    return L


def diff_square(x): return 4 * np.square(x-0.5)


def linear(x): return 2*x


def square(x): return np.square(x)


def exponential(x): return math.exp(x) / math.exp(1)


def sine(x): return math.sin(2*x)


def mild_diff_square(x): return np.square(x-0.5) + 0.5


def identity(x): return x
