import numpy as np
import math
from scipy.stats import pearsonr
# from graspologic.cluster.gclust import GaussianCluster
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from graspologic.utils import symmetrize
from graspologic.embed import AdjacencySpectralEmbed, MultipleASE
from graspologic.models import DCSBMEstimator

from utils import off_diag


def gcorr(G1, G2, Z, pooled_variance=True, epsilon1=1e-3, epsilon2=1e-3):
    """
    Compute the `gcorr` test statistic between a pair of graphs G1, G2
    given the community assignment vector Z
    The test stastic estimate the correlation between G1 and G2
    conditioned on the community assignment Z
    Parameters
    ----------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix representing the first random graph.
    G2: ndarray (n_vertices, n_vertices)
        Adjacency matrix representing the second random graph.
    Z: ndarray (n_vertices)
        Vector representing the community assignments of each vertex (zero-indexed)
        Example: if Z[i] == 2, then the ith vertex belongs to the 3rd community
        The number of communities equals the number of unique values in Z
    Returns
    -------
    T: float
        Value of the `gcorr` test statistic 
    """
    if G1.shape[0] != G2.shape[0] or G1.shape[1] != G2.shape[1]:
        raise ValueError("G1, G2 must have the same shape")
    if Z.size != G1.shape[0]:
        raise ValueError("Z must have the same number of elements as the number of vertices in G1/G2")
    Phat = np.zeros_like(G1)
    Qhat = np.zeros_like(G2)
    communities, num_nodes_in_communities = np.unique(Z, return_counts=True)

    # iterate over all combinations of communities (instead of just the lower/upper triangular)
    # because the community assignments can be in arbitrary order
    # the resulting matrix should still be symmetric because whenever you swap i and j you get the same answer
    for i in communities:
        for j in communities:
            if (i == j):
                num_nodes = num_nodes_in_communities[i] * (num_nodes_in_communities[i] - 1)
            else:
                num_nodes = num_nodes_in_communities[i] * num_nodes_in_communities[j]
            block_idx = np.ix_(Z == i, Z == j)
            if num_nodes == 0:
                # some block might only have one node, as a result of the calculation above, num_nodes would be 0
                # since we are ignoring the diagonal entries in the correlation anyway
                # we can set those probabilities arbitrarily
                Phat[block_idx] = 0
                Qhat[block_idx] = 0
            else:
                p = np.sum(G1[block_idx]) / num_nodes
                q = np.sum(G2[block_idx]) / num_nodes
                Phat[block_idx] = p
                Qhat[block_idx] = q

    # since the diagonal entries are forced to be zeros in graphs with no loops
    # we should ignore them in the calculation of correlation 
    g1 = off_diag(G1)
    g2 = off_diag(G2)
    phat = off_diag(Phat)
    qhat = off_diag(Qhat)

    # calculate the test statistic
    if pooled_variance:
        T = np.sum((g1 - phat) * (g2 - qhat)) / np.sqrt(np.sum(np.square(g1 - phat)) * np.sum(np.square(g2 - qhat)))
    else:
        # trim the estimated probability matrix
        phat[phat < epsilon1] = epsilon1
        phat[phat > 1 - epsilon2] = 1 - epsilon2
        qhat[qhat < epsilon1] = epsilon1
        qhat[qhat > 1 - epsilon2] = 1 - epsilon2
        num_vertices = G1.shape[0]
        T = np.sum((g1 - phat) * (g2 - qhat) / np.sqrt(phat * (1 - phat) * qhat * (1 - qhat))) / (num_vertices * (num_vertices - 1))
    return T


def block_permutation(G, Z):
    """
    Permute the vertices of G within each community given the community assignment vector Z
    Parameters
    -----------
    G: ndarray (n_vertices, n_vertices)
        Adjacency matrix representing a random graph
    Z: ndarray (n_vertices)
        Vector representing the community assignments of each vertex (zero-indexed)
        Example: if Z[i] == 2, then the ith vertex belongs to the 3rd community
        The number of communities equals the number of unique values in Z
    Returns
    --------
    G_perm: ndarray (n_vertices, n_vertices)
        The block permuted version of G
    """
    vertex_ind_perm = np.zeros(G.shape[0], dtype=int)
    for b in np.unique(Z):
        v_orig = np.where(Z==b)[0]
        v_perm = np.random.choice(v_orig, size=v_orig.size, replace=False)
        vertex_ind_perm[v_orig] = v_perm
    G_perm = G[vertex_ind_perm, :][:, vertex_ind_perm]
    return G_perm


def community_estimation(G1, G2=None, min_components=2, max_components=None):
    """
    Estimate the community assignments of the vertices of a single random graph or a pair
    when estimate for pair of graphs, assuming the two graphs have the same community structure
    First jointly embed G1 and G2, then cluster the embedding by GMM
    Parameters
    ----------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix representing the first random graph.
    G2: ndarray (n_vertices, n_vertices), default=None
        Adjacency matrix representing the second random graph.
    min_components : int, default=2.
        The minimum number of mixture components to consider (unless
        ``max_components=None``, in which case this is the maximum number of
        components to consider). If ``max_componens`` is not None, ``min_components``
        must be less than or equal to ``max_components``.
    max_components : int or None, default=None.
        The maximum number of mixture components to consider. Must be greater
        than or equal to ``min_components``.
    Returns
    --------
    Zhat: ndarray (n_vertices)
        Vector representing the estimated community assignments of each vertex (zero-indexed)
        Example: if Zhat[i] == 2, then the ith vertex is estimated to belong to the 3rd community
    """
    if G2 is None:
        Vhat = AdjacencySpectralEmbed().fit_transform(G1)
    else:
        Vhat = MultipleASE().fit_transform([G1, G2])
    # TODO: use graspologic.cluster.gclust after the bug is fixed
    # for now, manual iterate over GaussianMixture
    # Deal with number of clusters
    if max_components is None:
        lower_ncomponents = 1
        upper_ncomponents = min_components
    else:
        lower_ncomponents = min_components
        upper_ncomponents = max_components

    # the number of components we need to iterate through
    n_mixture_components = upper_ncomponents - lower_ncomponents + 1

    models = []
    bics = []
    for i in range(n_mixture_components):
        model = GaussianMixture(n_components=i + lower_ncomponents).fit(Vhat)
        models.append(model)
        bics.append(model.bic(Vhat))
    best_model = models[np.argmin(bics)]
    Zhat = best_model.predict(Vhat)
    return Zhat


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


def block_permutation_pvalue(G1, G2, test, num_perm, Z=None):
    """
    Estimate p-value via a block permutation test
    """
    if test not in ['gcorr', 'pearson']:
        raise ValueError('`test` needs to be `gcorr` or `pearson`, not {}'.format(test))
    
    if test == 'gcorr' and Z is None:
        raise ValueError('please provide `Z` for `gcorr`')

    if test == 'gcorr':
        obs_test_stat = gcorr(G1, G2, Z)
    elif test == 'pearson':
        obs_test_stat = pearson_graph(G1, G2)

    null_test_stats = np.zeros(num_perm)
    for i in range(num_perm):
        G2_perm = block_permutation(G2, Z)
        if test == 'gcorr':
            null_test_stats[i] = gcorr(G1, G2_perm, Z)
        elif test == 'pearson':
            null_test_stats[i] = pearson_graph(G1, G2_perm)

    num_extreme = np.where(null_test_stats >= obs_test_stat)[0].size
    if num_extreme < num_perm / 2:
        # P(T > t | H0) is smaller 
        return (2 * num_extreme + 1) / (num_perm + 1)
    else:
        # P(T < t | H0) is smaller
        return (2 * (num_perm - num_extreme) + 1) / (num_perm + 1)


def gcorr_dcsbm(G1, G2, max_comm, pooled_variance=True, min_comm=1, epsilon1=1e-3, epsilon2=1e-3, Z1=None, Z2=None, return_fit=False, seed=None):
    """
    Compute a test statistic based on DC-SBM fit
    Note this test statistic doesn't require the vertex assignment
    optionally give fitted DC-SBM to save computation time
    Note: if `G1_dcsbm` or `G2_dcsbm` is given, the estimated P matrices are extracted from these model fits
    otherwise, they are extracted from the model fitted on `G1`, `G2`
    """
    # if we are fixing the number of communities, we should also fix the number of latent dimensions of the embedding
    # otherwise (when we let the algorithm to automatically choose the number of communities)
    # we also let it choose the number of latent dimensions
    if min_comm == max_comm:
        K = min_comm
    else:
        K = None
    G1_dcsbm = DCSBMEstimator(directed=False, min_comm=min_comm, max_comm=max_comm, n_components=K, 
        cluster_kws={'random_state': seed}).fit(G1, y=Z1)
    G2_dcsbm = DCSBMEstimator(directed=False, min_comm=min_comm, max_comm=max_comm, n_components=K,
        cluster_kws={'random_state': seed}).fit(G2, y=Z2)
    # since the diagonal entries are forced to be zeros in graphs with no loops
    # we should ignore them in the calculation of correlation 
    g1 = off_diag(G1)
    g2 = off_diag(G2)
    phat = off_diag(G1_dcsbm.p_mat_)
    qhat = off_diag(G2_dcsbm.p_mat_)
    # trim the estimated probability matrix
    phat[phat < epsilon1] = epsilon1
    phat[phat > 1 - epsilon2] = 1 - epsilon2
    qhat[qhat < epsilon1] = epsilon1
    qhat[qhat > 1 - epsilon2] = 1 - epsilon2

    # calculate the test statistic
    if pooled_variance:
        T = np.sum((g1 - phat) * (g2 - qhat)) / np.sqrt(np.sum(np.square(g1 - phat)) * np.sum(np.square(g2 - qhat)))
    else:
        num_vertices = G1.shape[0]
        T = np.sum((g1 - phat) * (g2 - qhat) / np.sqrt(phat * (1 - phat) * qhat * (1 - qhat))) / (num_vertices * (num_vertices - 1))
    
    if return_fit:
        dcsbm_fit = {'G1': G1_dcsbm, 'G2': G2_dcsbm}
        return T, dcsbm_fit
    else:
        return T


def dcsbm_pvalue(G1, G2, max_comm, num_perm, pooled_variance=True, min_comm=1, epsilon1=1e-3, epsilon2=1e-3, Z1=None, Z2=None):
    """
    Estimate p-value via parametric bootstrap, i.e. fit a DC-SBM
    """
    # if we are fixing the number of communities, we should also fix the number of latent dimensions of the embedding
    # otherwise (when we let the algorithm to automatically choose the number of communities)
    # we also let it choose the number of latent dimensions
    if min_comm == max_comm:
        K = min_comm
    else:
        K = None
    obs_test_stat = gcorr_dcsbm(G1, G2, min_comm=min_comm, max_comm=max_comm,
        pooled_variance=pooled_variance, epsilon1=epsilon1, epsilon2=epsilon2)
    G1_dcsbm = DCSBMEstimator(directed=False, min_comm=min_comm, max_comm=max_comm, n_components=K).fit(G1, y=Z1)
    G2_dcsbm = DCSBMEstimator(directed=False, min_comm=min_comm, max_comm=max_comm, n_components=K).fit(G2, y=Z2)
    # create bootstrap samples
    G1_bootstrap = G1_dcsbm.sample(n_samples=num_perm)
    G2_bootstrap = G2_dcsbm.sample(n_samples=num_perm)
    null_test_stats = np.zeros(num_perm)
    for i in tqdm(range(num_perm)):
        null_test_stats[i] = gcorr_dcsbm(G1_bootstrap[i], G2_bootstrap[i], min_comm=min_comm, max_comm=max_comm,
            pooled_variance=pooled_variance, epsilon1=epsilon1, epsilon2=epsilon2)
    num_extreme = np.where(null_test_stats >= obs_test_stat)[0].size
    if num_extreme < num_perm / 2:
        # P(T > t | H0) is smaller 
        return (2 * num_extreme + 1) / (num_perm + 1)
    else:
        # P(T < t | H0) is smaller
        return (2 * (num_perm - num_extreme) + 1) / (num_perm + 1)
