import numpy as np

from finguard.monitor import psi


def test_psi_identique_proche_de_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 0.01


def test_psi_detecte_un_decalage():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(2, 1, size=5000)) > 0.25
