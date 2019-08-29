import numpy as np
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize, is_symmetric
from utils import non_diagonal


def rho_gaussian_sbm(rho, k, AL, BL, n, var_x=1, var_y=1):
    if sum(k) != n:
        raise ValueError('the total number of vertices in each community \
        should equal n')
    sigma = np.array([[var_x, rho], [rho, var_y]])
    AL = symmetrize(AL)
    BL = symmetrize(BL)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    block_indices = np.insert(np.cumsum(k), 0, 0)

    for i in range(AL.shape[0]):
        for j in range(AL.shape[1]):
            mu_x = AL[i, j]
            mu_y = BL[i, j]
            sample = np.random.multivariate_normal([mu_x, mu_y], sigma,
                                                   size=(block_indices[i+1]-block_indices[i],
                                                         block_indices[j+1]-block_indices[j]))
            A[block_indices[i]:block_indices[i+1],
              block_indices[j]:block_indices[j+1]] = sample[:, :, 0]
            B[block_indices[i]:block_indices[i+1],
              block_indices[j]:block_indices[j+1]] = sample[:, :, 1]
    A = symmetrize(A, method='triu')
    B = symmetrize(B, method='triu')
    return A, B


def rho_sbm(rho, k, AL, BL, n=100):
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
    A = sbm(k, AL, loops=True)

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
    B = symmetrize(B, method='triu')
    return A, B
