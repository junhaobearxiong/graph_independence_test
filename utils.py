import numpy as np


def off_diag(G):
    """
    Get the off-diagonal elements of a graph as a flatten array
    """
    return G[~np.eye(G.shape[0], dtype=bool)]


def binarize(G):
	"""
	Binarize input graph such that any edge weight greater than 0 becomes 1.
	"""
	G[np.where(G > 0)] = 1
	return G