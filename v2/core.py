import numpy as np
from graspologic.utils import symmetrize
from graspologic.embed.mase import MultipleASE
# from graspologic.cluster.gclust import GaussianCluster
from sklearn.mixture import GaussianMixture


def gcorr(G1, G2, Z):
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
            p = np.sum(G1[block_idx]) / num_nodes
            q = np.sum(G2[block_idx]) / num_nodes
            Phat[block_idx] = p
            Qhat[block_idx] = q
    # since the diagonal entries are forced to be zeros in graphs with no loops
    # we should ignore them in the calculation of correlation 
    # we do this by setting them to be the same as the diagonals of the`Phat` and `Qhat` matrix
    diag_idx = np.diag_indices(G1.shape[0])
    G1[diag_idx] = Phat[diag_idx]
    G2[diag_idx] = Qhat[diag_idx]

    # calculate the test statistic
    T = np.sum((G1 - Phat) * (G2 - Qhat)) / np.sqrt(np.sum(np.square(G1 - Phat)) * np.sum(np.square(G2 - Qhat)))
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


def community_estimation(G1, G2, min_components=2, max_components=None):
    """
    Estimate the community assignments of the vertices of random graphs G1 and G2
    assuming the two graphs have the same community structure
    First jointly embed G1 and G2, then cluster the embedding by GMM
    Parameters
    ----------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix representing the first random graph.
    G2: ndarray (n_vertices, n_vertices)
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
