import numpy as np
from graspologic.simulations import sample_edges
from graspologic.simulations.simulations_corr import check_dirloop, check_r


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

