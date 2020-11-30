import numpy as np


def off_diag(G):
    """
    Get the off-diagonal elements of a graph as a flatten array
    """
    return G[~np.eye(G.shape[0], dtype=bool)]
