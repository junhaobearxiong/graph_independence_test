import pytest
import numpy as np
from simulations import (
    er_corr_diffmarg,
    sbm_corr_diffmarg,
    er_corr_weighted,
)

def test_er_corr():
    n = 500
    p = 0.7
    q = 0.2
    # check r
    with pytest.raises(ValueError):
        r = 0.4
        er_corr_diffmarg(n, p, q, r)
    with pytest.raises(ValueError):
        r = -0.8
        er_corr_diffmarg(n, p, q, r)
    with pytest.raises(ValueError):
        r = 1.5
        er_corr_diffmarg(n, p, q, r)
    with pytest.raises(ValueError):
        r = -1.5
        er_corr_diffmarg(n, p, q, r)
    
    # check marginals
    r = 0.2
    g1, g2 = er_corr_diffmarg(n, p, q, r)
    assert np.isclose(p, g1.sum() / (n * (n - 1)), atol=0.05)
    assert np.isclose(q, g2.sum() / (n * (n - 1)), atol=0.05)

    # check rho
    k1 = g1.copy()
    k2 = g2.copy()
    k1 = k1[np.where(~np.eye(k1.shape[0], dtype=bool))]
    k2 = k2[np.where(~np.eye(k2.shape[0], dtype=bool))]
    output_r = np.corrcoef(k1, k2)[0, 1]
    assert np.isclose(r, output_r, atol=0.06)

    # check shape
    assert g1.shape == (n, n)
    assert g2.shape == (n, n)


def test_sbm_corr():
    n = [200, 200]
    p = [[0.7, 0.3], [0.3, 0.7]]
    q = [[0.2, 0.5], [0.5, 0.2]]
    r = 0.2

    # check marginals 
    g1, g2 = sbm_corr_diffmarg(n, p, q, r)
    a1, a2 = g1[0 : n[0], 0 : n[0]], g1[0 : n[0], n[0] :]
    b1, b2 = g2[0 : n[0], 0 : n[0]], g2[0 : n[0], n[0] :]
    pa1, pa2 = (
        a1.sum() / (n[0] * (n[0] - 1)),
        a2.sum() / (n[0] * n[1]),
    )
    pb1, pb2 = (
        b1.sum() / (n[0] * (n[0] - 1)),
        b2.sum() / (n[0] * n[1]),
    )
    # check the 1 probability of the output binary matrix
    # should be near to the input P and Q matrix
    assert np.isclose(p[0][0], pa1, atol=0.05)
    assert np.isclose(p[0][1], pa2, atol=0.05)
    assert np.isclose(q[0][0], pb1, atol=0.05)
    assert np.isclose(q[0][1], pb2, atol=0.05)

    # check rho
    a1 = a1[np.where(~np.eye(a1.shape[0], dtype=bool))]
    b1 = b1[np.where(~np.eye(a2.shape[0], dtype=bool))]
    a2 = a2.flatten()
    b2 = b2.flatten()
    m1 = np.corrcoef(a1, b1)[0, 1]
    m2 = np.corrcoef(a2, b2)[0, 1]
    avr = (m1 + m2) / 2
    assert np.isclose(avr, r, atol=0.05)

    # check shape
    assert g1.shape == (np.sum(n), np.sum(n))
    assert g2.shape == (np.sum(n), np.sum(n))


def test_er_corr_weighted():
    n = 200
    mu1 = 2
    mu2 = 0
    Sigma = [[1, 0.2], [0.2, 1]]
    with pytest.raises(ValueError):
        er_corr_weighted(n, [2], [0], Sigma)
    with pytest.raises(ValueError):
        er_corr_weighted(n, mu1, mu2, [[1, 0.2, 0], [0, 0.2, 1]])

    g1, g2 = er_corr_weighted(n, mu1, mu2, Sigma)
    assert np.allclose(g1, g1.T)
    assert np.allclose(g2, g2.T)
    assert np.allclose(np.diag(g1), 0)
    assert np.allclose(np.diag(g2), 0)
    assert np.isclose(g1.sum() / n / (n - 1), mu1, atol=0.02)
    assert np.isclose(g2.sum() / n / (n - 1), mu2, atol=0.02)
    Sigmahat = np.corrcoef(g1.flatten(), g2.flatten())
    assert np.isclose(Sigmahat[0][0], Sigma[0][0], atol=0.02)
    assert np.isclose(Sigmahat[0][1], Sigma[0][1], atol=0.02)
    assert np.isclose(Sigmahat[1][1], Sigma[1][1], atol=0.02)

def main():
    test_er_corr_weighted()
    test_er_corr()
    test_sbm_corr()


if __name__ == "__main__":
    main()

