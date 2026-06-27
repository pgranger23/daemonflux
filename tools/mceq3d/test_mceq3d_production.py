"""Offline tests for the production engine's angular-kernel logic.

The full MCEq-per-multipole solve is exercised in ``mceq3d_production.main``
(the l=0 == plain-MCEq validation prints 0.0e0); it needs MCEq and is slow, so
here we test only the kernel function used inside ``set_mod_pprod`` -- offline.
"""

import numpy as np

import mceq3d_production as mp


def _setup():
    e = np.logspace(-0.5, 5, 60)
    mp._EREF = e
    mp._TH1 = 0.3 / e  # theta1 ~ 0.3/E [rad]
    # xmat[i,j] = E_i/E_j (secondary/primary), lower triangle zeroed
    xmat = np.tril(e[:, None] / e[None, :])  # actually want upper; build simply
    xmat = np.where(e[:, None] <= e[None, :], e[:, None] / e[None, :], 0.0)
    return e, xmat


def test_l0_kernel_is_unity():
    e, xmat = _setup()
    k = mp._kernel_func(xmat, e, 0, 1.0)
    assert np.allclose(k, 1.0)  # l=0 -> no modification -> MCEq unchanged


def test_higher_l_suppresses_and_orders():
    e, xmat = _setup()
    k2 = mp._kernel_func(xmat, e, 2, 1.0)
    k6 = mp._kernel_func(xmat, e, 6, 1.0)
    # within (0,1], and higher l suppresses more
    assert np.all(k2 <= 1.0 + 1e-12) and np.all(k6 <= k2 + 1e-12)
    assert np.all(k2 >= 0.0)


def test_dummy_arg_ignored():
    # MCEq may pass a rescaled second arg (K0 isospin); result must not depend.
    e, xmat = _setup()
    a = mp._kernel_func(xmat, e, 4, 1.0)
    b = mp._kernel_func(xmat, e, 4, 0.5)
    assert np.allclose(a, b)
