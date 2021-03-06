import numpy as np
from graspologic.simulations import sample_edges, sample_edges_corr, sbm
from graspologic.simulations.simulations_corr import check_dirloop, check_r
from graspologic.utils import symmetrize
from graspologic.models import DCSBMEstimator


def get_lowest_r(p, q):
    return max(-np.sqrt(p * q / (1 - p) / (1 - q)), -np.sqrt((1 - p) * (1 - q) / p / q))


def get_highest_r(p, q):    
    return min(np.sqrt(p * (1 - q) / q / (1 - p)), np.sqrt(q * (1 - p) / p / (1 - q)))


def check_r_er(p, q, r):
    low_r = get_lowest_r(p, q)
    high_r = get_highest_r(p, q)
    if r < low_r:
        raise ValueError("r is lower than the lowest r allowed by p and q: {:.2f}".format(low_r))
    if r > high_r:
        raise ValueError("r is higher than the highest r allowed by p and q: {:.2f}".format(high_r))


def get_lowest_r_sbm(p, q):
    r = -1
    for i in range(np.array(p).shape[0]):
        for j in range(np.array(p).shape[1]):
            low_r = get_lowest_r(p[i][j], q[i][j])
            if r < low_r:
                r = low_r
    return r


def get_highest_r_sbm(p, q):
    r = 1
    for i in range(np.array(p).shape[0]):
        for j in range(np.array(p).shape[1]):
            high_r = get_highest_r(p[i][j], q[i][j])
            if r > high_r:
                r = high_r
    return r


def check_r_sbm(p, q, r):
    low_r = get_lowest_r_sbm(p, q)
    high_r = get_highest_r_sbm(p, q)
    if r < low_r:
        raise ValueError("r is lower than the lowest r allowed by p and q: {:.2f}".format(low_r))
    if r > high_r:
        raise ValueError("r is higher than the highest r allowed by p and q: {:.2f}".format(high_r))
                

def sample_edges_corr_diffmarg(P, Q, R, directed=False, loops=False):
    """
    Generate a pair of correlated graphs with Bernoulli distribution.
    Both G1 and G2 are binary matrices.
    Allows for different marginal distributions
    Parameters
    ----------
    P: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of probabilities (between 0 and 1) for the first random graph.
    Q: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of probabilities (between 0 and 1) for the second random graph.
    R: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of correlation (between 0 and 1) between graph pairs.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    Returns
    -------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph.
    G2: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph.
    """
    # test input
    # check P
    if type(P) is not np.ndarray:
        raise TypeError("P must be numpy.ndarray")
    if len(P.shape) != 2:
        raise ValueError("P must have dimension 2 (n_vertices, n_vertices)")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")

    # check Q
    if type(Q) is not np.ndarray:
        raise TypeError("Q must be numpy.ndarray")
    if len(Q.shape) != 2:
        raise ValueError("Q must have dimension 2 (n_vertices, n_vertices)")
    if Q.shape[0] != P.shape[0] or Q.shape[1] != P.shape[1]:
        raise ValueError("Q must have the same shape as P")

    # check R
    if type(R) is not np.ndarray:
        raise TypeError("R must be numpy.ndarray")
    if len(R.shape) != 2:
        raise ValueError("R must have dimension 2 (n_vertices, n_vertices)")
    if R.shape[0] != P.shape[0] or R.shape[1] != P.shape[1]:
        raise ValueError("R must have the same shape as P")

    # check directed and loops
    check_dirloop(directed, loops)

    G1 = sample_edges(P, directed=directed, loops=loops)
    P2 = G1.copy()
    P2 = np.where(P2 == 1,
            Q + R * np.sqrt((1 - P) * Q * (1 - Q) / P),
            Q - R * np.sqrt(P * Q * (1 - Q) / (1 - P))
        )
    G2 = sample_edges(P2, directed=directed, loops=loops)
    return G1, G2


def er_corr_diffmarg(n, p, q, r, directed=False, loops=False):
    """
    Generate a pair of correlated graphs with specified edge probability
    Both G1 and G2 are binary matrices.
    Parameters
    ----------
    n: int
       Number of vertices
    p: float
        Probability of an edge existing between two vertices in the first graph, between 0 and 1.
    q: float
        Probability of an edge existing between two vertices in the second graph, between 0 and 1.
    r: float
        The value of the correlation between the same vertices in two graphs.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    Returns
    -------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph.
    G2: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph.
    """
    # test input
    # check n
    if not np.issubdtype(type(n), np.integer):
        raise TypeError("n is not of type int.")
    elif n <= 0:
        msg = "n must be > 0."
        raise ValueError(msg)

    # check p
    if not np.issubdtype(type(p), np.floating):
        raise TypeError("p is not of type float.")
    elif p < 0 or p > 1:
        msg = "p must between 0 and 1."
        raise ValueError(msg)

    # check q
    if not np.issubdtype(type(q), np.floating):
        raise TypeError("q is not of type float.")
    elif q < 0 or q > 1:
        msg = "q must between 0 and 1."
        raise ValueError(msg)

    # check r
    check_r(r)

    # check the relation between r, p and q
    check_r_er(p, q, r)

    # check directed and loops
    check_dirloop(directed, loops)

    P = p * np.ones((n, n))
    Q = q * np.ones((n, n))
    R = r * np.ones((n, n))
    G1, G2 = sample_edges_corr_diffmarg(P, Q, R, directed=directed, loops=loops)
    return G1, G2


def sbm_corr_diffmarg(n, p, q, r, directed=False, loops=False):
    """
    Generate a pair of correlated graphs with specified edge probability
    Both G1 and G2 are binary matrices.
    Parameters
    ----------
    n: list of int, shape (n_communities)
        Number of vertices in each community. Communities are assigned n[0], n[1], ...
    p: array-like, shape (n_communities, n_communities)
        Probability of an edge between each of the communities in the first graph, where p[i, j] indicates
        the probability of a connection between edges in communities [i, j].
        0 < p[i, j] < 1 for all i, j.
    q: array-like, shape (n_communities, n_communities)
        same as p, but for the second graph
    r: float
        Probability of the correlation between the same vertices in two graphs.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    Returns
    -------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph.
    G2: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph.
    """
    # test input
    # Check n
    if not isinstance(n, (list, np.ndarray)):
        msg = "n must be a list or np.array, not {}.".format(type(n))
        raise TypeError(msg)
    else:
        n = np.array(n)
        if not np.issubdtype(n.dtype, np.integer):
            msg = "There are non-integer elements in n"
            raise ValueError(msg)

    # Check p
    if not isinstance(p, (list, np.ndarray)):
        msg = "p must be a list or np.array, not {}.".format(type(p))
        raise TypeError(msg)
    else:
        p = np.array(p)
        if not np.issubdtype(p.dtype, np.number):
            msg = "There are non-numeric elements in p"
            raise ValueError(msg)
        elif p.shape != (n.size, n.size):
            msg = "p is must have shape len(n) x len(n), not {}".format(p.shape)
            raise ValueError(msg)
        elif np.any(p < 0) or np.any(p > 1):
            msg = "Values in p must be in between 0 and 1."
            raise ValueError(msg)

    # Check q
    if not isinstance(q, (list, np.ndarray)):
        msg = "q must be a list or np.array, not {}.".format(type(q))
        raise TypeError(msg)
    else:
        q = np.array(q)
        if not np.issubdtype(q.dtype, np.number):
            msg = "There are non-numeric elements in q"
            raise ValueError(msg)
        elif q.shape != (n.size, n.size):
            msg = "q is must have shape len(n) x len(n), not {}".format(q.shape)
            raise ValueError(msg)
        elif np.any(q < 0) or np.any(q > 1):
            msg = "Values in q must be in between 0 and 1."
            raise ValueError(msg)

    # check r
    check_r(r)

    # check the relation between r, p and q
    check_r_sbm(p, q, r)

    # check directed and loops
    check_dirloop(directed, loops)

    P = np.zeros((np.sum(n), np.sum(n)))
    Q = np.zeros((np.sum(n), np.sum(n)))
    block_indices = np.insert(np.cumsum(np.array(n)), 0, 0)
    for i in range(np.array(p).shape[0]):  # for each row
        for j in range(np.array(p).shape[1]):  # for each column
            P[
                block_indices[i] : block_indices[i + 1],
                block_indices[j] : block_indices[j + 1],
            ] = p[i][j]
            Q[
                block_indices[i] : block_indices[i + 1],
                block_indices[j] : block_indices[j + 1],                
            ] = q[i][j]
    R = r * np.ones((np.sum(n), np.sum(n)))
    G1, G2 = sample_edges_corr_diffmarg(P, Q, R, directed=directed, loops=loops)
    return G1, G2


def sample_edges_corr_weighted(shape, mu1, mu2, Sigma):
    """
    Generate a pair of correlated matrices with the bivariate normal distribution.
    Both G1 and G2 are non-binary matrices.
    Every pair of entries is distributed as a bivariate normal, with mean = [mu1, mu2]
    and covariance matrix Sigma
    The correlation between G1 and G2 is Sigma12 / sqrt(Sigma11 * Sigma22)
    Parameters
    ----------
    shape: tuple
        The shape of the output matrices: shape[0] denotes the number of rows, shape[1] the columns
    mu1: float
        The mean of the edge weights of G1 (analogous the marginal probability p in correlated Bernoulli graph)
    mu2: float
        The mean of the edge weights of G2 (analogous the marginal probability q in correlated Bernoulli graph)
    Sigma: list or ndarray (2, 2)
        The covariance matrix encoding the variances of the edge weights of G1, G2
        and the covariance beteween them
    Returns
    -------
    G1: ndarray (shape)
    G2: ndarray (shape)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")

    if not np.issubdtype(type(mu1), np.floating) and not np.issubdtype(type(mu1), np.integer):
        raise ValueError("mu1 is not of type int or float")

    if not np.issubdtype(type(mu2), np.floating) and not np.issubdtype(type(mu2), np.integer):
        raise ValueError("mu2 is not of type int or float")

    if not isinstance(Sigma, (list, np.ndarray)):
        raise ValueError("Sigma must be list or np.ndarray")
    if np.array(Sigma).shape != (2, 2):
        raise ValueError("Sigma must have shape (2, 2)")

    sample = np.random.multivariate_normal([mu1, mu2], Sigma, size=shape)
    G1 = sample[..., 0]
    G2 = sample[..., 1]
    return G1, G2


def er_corr_weighted(n, mu1, mu2, Sigma, directed=False, loops=False):
    """
    Generate a pair of correlated graphs with the bivariate normal distribution.
    Both G1 and G2 are non-binary matrices.
    Every pair of edges is distributed as a bivariate normal, with mean = [mu1, mu2]
    and covariance matrix Sigma
    The correlation between G1 and G2 is Sigma12 / sqrt(Sigma11 * Sigma22)
    Parameters
    ----------
    n: int
       Number of vertices
    mu1: float
        The mean of the edge weights of G1 (analogous the marginal probability p in correlated Bernoulli graph)
    mu2: float
        The mean of the edge weights of G2 (analogous the marginal probability q in correlated Bernoulli graph)
    Sigma: list or ndarray (2, 2)
        The covariance matrix encoding the variances of the edge weights of G1, G2
        and the covariance beteween them
    Returns
    -------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix representing a random graph.
    G2: ndarray (n_vertices, n_vertices)
        Adjacency matrix representing a random graph.
    """
    G1, G2 = sample_edges_corr_weighted((n, n), mu1, mu2, Sigma)
    if not directed:
        G1 = symmetrize(G1, method="triu")
        G2 = symmetrize(G2, method="triu")
    if not loops:
        G1 = G1 - np.diag(np.diag(G1))
        G2 = G2 - np.diag(np.diag(G2))
    return G1, G2


def sbm_corr_weighted(n, mu1, mu2, Sigma, directed=False, loops=False):
    """
    Parameters
    ----------
    n: list of int, shape (n_communities)
        Number of vertices in each community. Communities are assigned n[0], n[1], ...
    mu1: array-like, shape (n_communities, n_communities)
        Mean of the edge weight between each of the communities in the first graph, where mu1[i, j] indicates
        the mean of the edge weights of edges in communities [i, j].
    mu2: array-like, shape (n_communities, n_communities)
        same as mu1, but for the second graph
    Sigma: list or ndarray (2, 2)
        The covariance matrix encoding the variances of the edge weights of G1, G2
        and the covariance beteween them. 
        Right now we are forcing the entire graph to have the same variance and covariance
    """
    n = np.array(n)
    G1 = np.zeros((np.sum(n), np.sum(n)))
    G2 = np.zeros((np.sum(n), np.sum(n)))
    block_indices = np.insert(np.cumsum(np.array(n)), 0, 0)
    for i in range(n.size):  # for each row
        for j in range(n.size):  # for each column
            g1, g2 = sample_edges_corr_weighted((n[i], n[j]), mu1[i][j], mu2[i][j], Sigma)
            G1[
                block_indices[i] : block_indices[i + 1],
                block_indices[j] : block_indices[j + 1],
            ] = g1
            G2[
                block_indices[i] : block_indices[i + 1],
                block_indices[j] : block_indices[j + 1],                
            ] = g2
    if not directed:
        G1 = symmetrize(G1, method="triu")
        G2 = symmetrize(G2, method="triu")
    if not loops:
        G1 = G1 - np.diag(np.diag(G1))
        G2 = G2 - np.diag(np.diag(G2))
    return G1, G2


def dcsbm_corr(n, p, r, theta, epsilon1=1e-3, epsilon2=1e-3, directed=False, loops=False):
    '''
    Sample a pair of DC-SBM with the same marginal probabilities
    '''
    Z = np.repeat(np.arange(0, np.array(n).size), n)
    R = r * np.ones((np.sum(n), np.sum(n)))
    # sample a DC-SBM w/ block prob p
    G = sbm(n, p, dc=theta)
    # fit DC-SBM to G1 to estimate P
    G_dcsbm = DCSBMEstimator(directed=False).fit(G, y=Z)
    p_mat = G_dcsbm.p_mat_
    # P could be out of range
    p_mat[p_mat < epsilon1] = epsilon1
    p_mat[p_mat > 1 - epsilon2] = 1 - epsilon2
    # sample correlated graphs based on P
    G1, G2 = sample_edges_corr(p_mat, R, directed, loops)
    return G1, G2
