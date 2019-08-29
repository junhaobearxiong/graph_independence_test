import numpy as np

from simulations import rho_sbm

def er_corr(A, B):
    x = np.vstack((A.reshape(-1), B.reshape(-1)))
    return np.corrcoef(x)[0, 1]