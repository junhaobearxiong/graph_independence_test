import numpy as np
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize, is_symmetric
from utils import non_diagonal


def rho_sbm_diff_block(rho, k, AL, BL, n=100):
    if sum(k) != n:
        raise ValueError('the total number of vertices in each community \
        should equal n')

    if np.any(AL == 0) or np.any(AL == 1) \
            or np.any(BL == 0) or np.any(BL == 1):
        raise ValueError('block probabilities AL and BL cannot have 0 or 1')

    largest_rho = np.minimum(np.sqrt(AL*(1-BL)/((1-AL)*BL)),
                             np.sqrt((1-AL)*BL/(AL*(1-BL))))
    if np.any(rho > largest_rho):
        raise ValueError('the largest valid rho for the specified AL and BL is \
        {}. Please specify a rho that is smaller than the largest valid rho.'.format(
            np.amin(largest_rho)))

    AL = symmetrize(AL)
    BL = symmetrize(BL)
    A = sbm(k, AL)

    AL_new = np.zeros_like(A)
    BL_new = np.zeros_like(A)
    block_indices = np.insert(np.cumsum(np.array(k)), 0, 0)
    for i in range(AL.shape[0]):
        for j in range(AL.shape[1]):
            AL_new[block_indices[i]:block_indices[i+1],
                   block_indices[j]:block_indices[j+1]] = AL[i, j]
            BL_new[block_indices[i]:block_indices[i+1],
                   block_indices[j]:block_indices[j+1]] = BL[i, j]

    prob = BL_new + A*rho*np.sqrt((1-AL_new)*BL_new*(1-BL_new)/AL_new) - \
        (1-A)*rho*np.sqrt(AL_new*BL_new*(1-BL_new)/(1-AL_new))
    B = np.random.binomial(1, prob)
    B = B.astype(np.float64)
    B = symmetrize(B)
    np.fill_diagonal(B, 0)
    return A, B


def rho_rdpg(rho, n=50):
    X = np.random.uniform(0, 1, (n, 1))
    P = np.dot(X, X.T)
    A = rdpg(X, rescale=False, loops=False)
    B = np.random.binomial(1, (1-rho)*P + rho*A)
    B = B.astype(np.float64)
    B = symmetrize(B)
    np.fill_diagonal(B, 0)
    return A, B


def nonlinear_rdpg(n, nonlinear_transform):
    X = np.random.uniform(size=(n, 1))
    A = rdpg(X, loops=False)
    Y = np.vectorize(nonlinear_transform)(X)
    B = rdpg(Y, loops=False)
    return A, B


def null_rdpg(n, nonlinear_transform):
    _, B = nonlinear_rdpg(n, nonlinear_transform)
    Z = np.random.uniform(size=(n, 1))
    C = rdpg(Z, loops=False)
    return B, C


def rho_ER(rho, p, n=100):
    """
    rho-ER for the same marginal, assuming undirected

    Parameters
    ----------
    rho : float, int
        edge correlation between ER random graphs
    p : float, int
        probability of edge within graph
    n : int, optional
        number of vertices in graph

    Returns
    -------
    A : array-like
        The adjacency matrix of graph G1
    B : array-like
        The adjacency matrix of graph G2
    """
    nvec = [n]
    pvec = np.array([[p]])
    L = np.repeat(np.repeat(pvec, n, 0), n, 1)

    A = sbm(nvec, pvec)
    B = np.random.binomial(1, (1-rho)*L + rho*A)
    B = B.astype(np.float64)
    # force B to be symmetric, since A is assumed to be undirected
    B = symmetrize(B)
    np.fill_diagonal(B, 0)

    return A, B


def rho_ER_marg(rho, p, q, n=100):
    """
    rho-ER for different marginals

    Parameters
    ----------
    rho : float, int
        edge correlation between ER random graphs
    p : float, int
        probability of edge within the first graph
    q : float, int
        probability of edge within the second graph
    n : int, optional
        number of vertices in each of the graphs

    Returns
    -------
    A : array-like
        The adjacency matrix of graph G1
    B : array-like
        The adjacency matrix of graph G2
    """
    if p == 0 or p == 1 or q == 0 or q == 1:
        raise ValueError('p or q cannot be 0 or 1')
    '''
    if rho is too large, the probability for B in order to get the correlation
    to be rho exceeds one
    '''
    largest_rho = min(np.sqrt(p*(1-q)/((1-p)*q)), np.sqrt((1-p)*q/(p*(1-q))))
    if rho > largest_rho:
        raise ValueError('the largest valid rho for p={} and q={} is {}. \
        Please specify a rho that is \
        smaller than the largest valid rho.'.format(p, q, largest_rho))

    nvec = [n]
    pvec = np.array([[p]])
    L = np.repeat(np.repeat(pvec, n, 0), n, 1)

    A = sbm(nvec, pvec)
    prob = q + A*rho*np.sqrt((1-p)*q*(1-q)/p) - \
        (1-A)*rho*np.sqrt(p*q*(1-q)/(1-p))
    B = np.random.binomial(1, prob)
    B = B.astype(np.float64)
    B = symmetrize(B)
    np.fill_diagonal(B, 0)
    return A, B


def rho_sbm(rho, k, L, n=100):
    """
    Generates 2 adjacency matrices A,B of graphs G1, G2.
    (G1,G2) are sampled from a rho-SBM(k,b,L) by the method described in [1].
    The block membership function assumes the first n//k belong to block 1, and
    so on.

    Parameters
    ----------
    rho : float, int
        correlation between graphs.
        0.0 < rho < 1.0
    k : int
        number of blocks of the rho-SBM.
        k >= 1
    L : array-like
        edge probability matrix, each entry a float between 0 and 1.
        L should be of size k by k.
    n : int, optional, default = 100
        number of vertices for each graph

    Returns
    -------
    A : array-like
        The adjacency matrix of graph G1
    B : array-like
        The adjacency matrix of graph G2

    References
    ----------
    .. [1] Fishkind et. al (2012).
        Seeded Graph Matching
        arXiv:1209.0367v4
    """
    if np.any(L == 0) or np.any(L == 1):
        raise ValueError('block probabilities L cannot have 0 or 1')
    L = symmetrize(L)
    A = sbm([int(n/k)]*k, L)
    BL = np.repeat(np.repeat(L, n//k, 0), n//k, 1)
    B = np.random.binomial(1, (1-rho)*BL + rho*A)
    # important for mgc to run directly on the distance matrix
    B = B.astype(np.float64)
    B = symmetrize(B)
    np.fill_diagonal(B, 0)
    return A, B


def rho_sbm_marg(rho, k, AL, BL, n=100):
    '''
    Parameters
    ----------
    rho : float, int
        correlation between graphs.
        0.0 < rho < 1.0
    k : int
        number of blocks of the rho-SBM.
        k >= 1
    AL : array-like
        edge probability matrix for the first graph A, each entry a float
        between 0 and 1.
        AL should be of size k by k.
    BL : array-like
        edge probability matrix for the second graph B, each entry a float
        between 0 and 1.
        BL should be of size k by k.
    n : int, optional, default = 100
        number of vertices for each graph
    '''
    if np.any(AL == 0) or np.any(AL == 1) \
            or np.any(BL == 0) or np.any(BL == 1):
        raise ValueError('block probabilities AL and BL cannot have 0 or 1')

    largest_rho = np.minimum(np.sqrt(AL*(1-BL)/((1-AL)*BL)), np.sqrt((1-AL)*BL/(AL*(1-BL))))
    if np.any(rho > largest_rho):
        raise ValueError('the largest valid rho for the specified AL and BL is {}. Please specify a rho that is smaller than the largest valid rho.'.format(
            np.amin(largest_rho)))

    AL = symmetrize(AL)
    BL = symmetrize(BL)
    A = sbm([int(n/k)]*k, AL, loops=True)
    AL = np.repeat(np.repeat(AL, n//k, 0), n//k, 1)
    BL = np.repeat(np.repeat(BL, n//k, 0), n//k, 1)
    prob = BL + A*rho*np.sqrt((1-AL)*BL*(1-BL)/AL) - (1-A)*rho*np.sqrt(AL*BL*(1-BL)/(1-AL))
    B = np.random.binomial(1, prob)
    B = B.astype(np.float64)
    B = symmetrize(B)
    np.fill_diagonal(B, 0)
    return A, B


def ER_corr(A, B):
    '''
    Calculate the correlation between two ER-graphs (of the same dimensions)
    by treating each graph as one variable with the same number of examples as the number of vertices of the graph
    '''
    # A2 = triu_no_diag(A)
    # B2 = triu_no_diag(B)
    # diagonal entries are ignored since they are always set to zero when the graphs are sampled
    A2 = non_diagonal(A)
    B2 = non_diagonal(B)
    x = np.vstack((A2, B2))
    return np.corrcoef(x)[0, 1]


def sbm_corr(A, B, k):
    '''
    Calculate the correlation between two SBMs by treating each block as an ER graph and
    use ER_corr to calculate their correlations

    Note: the blocks off diagonal are technically not ERs since they are not symmetric
    but the edges within such block are still sampled i.i.d from the same Bernoulli
    hence we can use sample correlation to estimate the true correlation.
    By using ER_corr to calculate the correlation of these off-diagonal blocks,
    we technically lose the data points on the diagonals, but we did so for the sake of simplicity of the code
    '''
    corr_sum = 0
    n = A.shape[0]
    for i in range(1, k+1):
        for j in range(1, k+1):
            block1 = A[(i-1)*(n//k):i*(n//k), (j-1)*(n//k):j*(n//k)]
            block2 = B[(i-1)*(n//k):i*(n//k), (j-1)*(n//k):j*(n//k)]
            corr_sum += ER_corr(block1, block2)
    return corr_sum / np.square(k)


def ER_corr_diff(p_n=11, q_n=11):
    """
    For ERs with different marginals:
    Calculate the difference between expected correlation between graphs and actual edge correlation
    Varying p and q

    Parameters
    ----------
    func:
        the correlation function
    rho_n : int, optional, default = 11
        the number of values of rho between 0 and 1
    p_n : int, optional, default = 11
        the number of values of p between 0 and 1
    """
    rho = 0.2
    x = np.linspace(0, 1, p_n)
    y = np.linspace(0, 1, q_n)
    z = np.zeros((p_n, q_n))
    for i, p in enumerate(x):
        for j, q in enumerate(y):
            try:
                A, B = rho_ER_marg(rho, p, q)
            except ValueError:
                continue
            else:
                z[i, j] = rho - ER_corr(A, B)
    #fig = plotly_plot(x,y,z)
    # return fig
    return z


def sbm_marg_corr_diff(p_n=11, q_n=11, rho_n=11, rho=0.3, fixed_p=0.5,
                       fixed_q=0.2, n=100):
    """
    For SBMs with different marginals:
    Calculate the difference between expected correlation between graphs and actual edge correlation
    Varying one parameter of AL and one parameter of BL

    Parameters
    ----------
    rho_n : int, optional, default = 11
        the number of values of rho between 0 and 1
    p_n : int, optional, default = 11
        the number of values of p between 0 and 1
    """
    rho = rho
    x = np.linspace(0, 1, p_n)
    # y = np.linspace(0,1,rho_n)
    y = np.linspace(0, 1, q_n)
    z = np.zeros((p_n, q_n))
    zerr = np.zeros((p_n, rho_n))
    for i, p in enumerate(x):
        for j, q in enumerate(y):
            monte = []
            for _ in range(10):
                AL = np.array([[p, fixed_p], [fixed_p, p]])
                BL = np.array([[q, fixed_q], [fixed_q, q]])
                k = AL.shape[0]
                try:
                    A, B = rho_sbm_marg(rho, k, AL, BL, n)
                except ValueError:
                    continue
                else:
                    monte.append(sbm_corr(A, B, k))
            if monte:
                z[i, j] = rho - np.mean(monte)
                zerr[i, j] = np.std(monte)
    return z, zerr


def sbm_corr_diff(p_n=11, rho_n=11, n=100):
    """
    For SBMs with the same marginal:
    Calculate the difference between expected correlation between graphs and actual edge correlation
    Varying one parameter in L and rho

    Parameters
    ----------
    rho_n : int, optional, default = 11
        the number of values of rho between 0 and 1
    p_n : int, optional, default = 11
        the number of values of p between 0 and 1
    """
    x = np.linspace(0, 1, p_n)
    y = np.linspace(0, 1, rho_n)
    z = np.zeros((p_n, rho_n))
    zerr = np.zeros((p_n, rho_n))
    for i, p in enumerate(x):
        for j, rho in enumerate(y):
            monte = []
            for _ in range(10):
                L = np.array([[p, 0.1], [0.1, p]])
                k = L.shape[0]
                try:
                    A, B = rho_sbm(rho, k, L, n)
                except ValueError:
                    continue
                else:
                    monte.append(sbm_corr(A, B, k))
            if monte:
                z[i, j] = rho - np.mean(monte)
                zerr[i, j] = np.std(monte)
    return z, zerr
