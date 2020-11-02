import pytest
import numpy as np
from core import (
	gcorr
)
from simulations import (
	sbm_corr_diffmarg
)

def test_gcorr():
    n = [300, 200]
    p = [[0.7, 0.3], [0.3, 0.5]]
    q = [[0.2, 0.5], [0.8, 0.2]]
    r = 0.2
    g1, g2 = sbm_corr_diffmarg(n, p, q, r)
    with pytest.raises(ValueError):
    	z = np.repeat([0, 1], 200)
    	gcorr(g1, g2, z)
    z = np.repeat([0, 1], n)
    T = gcorr(g1, g2, z)
    assert(np.isclose(T, r, atol=0.01))