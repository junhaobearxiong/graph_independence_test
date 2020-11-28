import numpy as np
import math
from scipy.stats import pearsonr
from core import block_permutation, gcorr


def pearson_graph(G1, G2):
    """
    Naively apply pearson correlation on a pair of graphs 
    by measuring the correlation between the vectorized adjacency matrices
    """
    # ignore the diagonal entries and calculate the correlation between the upper triangulars
    triu_idx = np.triu_indices(G1.shape[0], k=1)
    return np.corrcoef(G1[triu_idx], G2[triu_idx])[0, 1]


def pearson_exact_pvalue(G1, G2):
    """
    Naively compute the p-value of a correlation test on a pair of graphs
    by an exact t-test
    """
    triu_idx = np.triu_indices(G1.shape[0], k=1)
    _, pval = pearsonr(G1[triu_idx], G2[triu_idx])
    return pval

def vertex_permutation(G):
    """
    Permute the vertices of random graph G
    """
    vertex_ind = np.arange(G.shape[0])
    vertex_ind_perm = np.random.permutation(vertex_ind)
    G_perm = G[vertex_ind_perm, :][:, vertex_ind_perm]
    return G_perm


def power_estimation(test_stats_null, test_stats_alt, alpha):
    """
    Estimate power based on the test statistics generated from simulated experiments
    note the test is two-sided
    test statistics under the null are used to estimate the rejection region
    """
    nmc = test_stats_null.size
    c1 = np.sort(test_stats_null)[math.floor(alpha / 2 * nmc)]
    c2 = np.sort(test_stats_null)[math.ceil((1 - alpha / 2) * nmc)]
    power = np.where((test_stats_alt > c2) | (test_stats_alt < c1))[0].size / nmc
    return power


def permutation_pvalue(G1, G2, Z, num_perm):
    """
    Estimate p-value via a block permutation test
    """
    obs_test_stat = gcorr(G1, G2, Z)
    null_test_stats = np.zeros(num_perm)
    for i in range(num_perm):
        G2_perm = block_permutation(G2, Z)
        null_test_stats[i] = gcorr(G1, G2_perm, Z)
    num_extreme = np.where(null_test_stats >= obs_test_stat)[0].size
    if num_extreme < num_perm / 2:
        # P(T > t | H0) is smaller 
        return 2 * num_extreme / num_perm
    else:
        # P(T < t | H0) is smaller
        return 2 * (num_perm - num_extreme) / num_perm

