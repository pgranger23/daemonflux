"""Tests of ``assemble3d.py`` (cascade3d C2).

Run: ``taskset -c 0-3 $PY -m pytest tools/cascade3d/test_assemble3d.py -q``
"""

import os

import numpy as np
import pytest

import assemble3d as A3
import boundary as B
import grid
import kernelset as K
import rays

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KS = os.path.join(N3, "kernels", "mc_close", "ks_mc_pool2000_L32_losson.npz")
SITE = os.path.join(N3, "c2", "solver", "sky", "sky_h0_x+00.0_y+00.0.npz")


@pytest.fixture(scope="module")
def setup():
    if not (os.path.exists(KS) and os.path.exists(SITE)):
        pytest.skip("kernel set or site table not available")
    atm = grid.CorsikaAtmosphere(r_e=6371e5)
    lev = grid.make_levels(atm, spec=((50.0, 5.0), (70.0, 10.0), (np.inf, 20.0)))
    fam = grid.LineFamily(np.array([1.0, 0.6, 0.3, 0.12, 0.05]), lev[:-1][::2],
                          r_e=6371e5)
    r = rays.Rays(atm, lev, fam, n_sub=2, dX_max=50.0)
    ks = K.KernelSet.load(KS)
    d = dict(ks.d)
    d["phi_top"] = np.where(ks.e_grid[None] <= 6.0, ks.phi_top, 0.0)
    ks = K.KernelSet(d)
    tab = B.SkyTable.load(SITE)
    gw = B.GroupWeights(ks.e_bins)
    cw = B.chain_weights(r, tab.nodes)
    Dc = A3.location_deficit(tab, gw, ks, cw)
    return {"r": r, "ks": ks, "tab": tab, "gw": gw, "cw": cw, "Dc": Dc}


def _mock_solve(tangents, nE=52, nSk=4, n=7):
    """A linear stand-in for the tangent solver: dI = fixed random linear map of the
    top tangent (bookkeeping tests only)."""
    rng = np.random.default_rng(0)
    out = []
    for m, part, dt in tangents:
        x = dt.sum(axis=(0,))                                  # (nE, Nl)
        P = np.random.default_rng(m).random((x.shape[1], nSk * n))
        out.append((x @ P).reshape(nE, nSk, n) * (1 + 0.1 * m))
    del rng
    return np.array(out)


def test_zero_deficit_is_c1_bit_for_bit(setup):
    ks = setup["ks"]
    rng = np.random.default_rng(1)
    mu0 = np.linspace(0, 1, 9)
    I0 = rng.random((ks.nE, 4, mu0.size)) + 0.5
    tl = A3.order_tangents(np.zeros_like(setup["Dc"]), 4, ks)
    dI = np.zeros((len(tl), ks.nE, 4, mu0.size))
    Iaz = A3.azimuth_table(I0, dI, [(m, p) for m, p, _ in tl])
    cells = A3.azimuth_cells(mu0, Iaz, ks)
    phi0 = A3.phi0_cells(mu0, I0, ks)
    for b in range(12):
        assert np.array_equal(cells[..., b], phi0)


def test_orders_reconstruct_the_sectors(setup):
    """The top tangents of orders 0..18 synthesise back to -D on the 36 sectors."""
    Dc, ks = setup["Dc"], setup["ks"]
    tl = A3.order_tangents(Dc, 18, ks)
    (ip, iN), _ = A3.b0_rows(ks)
    c = np.zeros((Dc.shape[0], 19, ks.nE, 2))
    s = np.zeros_like(c)
    for m, part, dt in tl:
        arr = c if part == "c" else s
        arr[:, m, :, 0] = -dt[ip].T
        arr[:, m, :, 1] = -dt[iN].T
    back = B.synth_sectors(c, s, 36, axis=1)
    assert np.max(np.abs(back - Dc)) < 1e-12 * np.abs(Dc).max()
    assert len(tl) == 36


def test_azimuth_table_bin_means():
    """A single order: the bin value is I0 + dI times the numerical bin mean."""
    I0 = np.ones((3, 2, 4))
    dI = np.full((1, 3, 2, 4), 0.25)
    for m, part in ((1, "c"), (3, "s"), (5, "c")):
        Iaz = A3.azimuth_table(I0, dI, [(m, part)])
        f = np.cos if part == "c" else np.sin
        for b in range(12):
            ph = np.radians(np.linspace(30 * b, 30 * b + 30, 20001))
            mean = np.trapezoid(f(m * ph), ph) / np.radians(30)
            assert abs(Iaz[b, 0, 0, 0] - (1 + 0.25 * mean)) < 1e-9


def test_rotation_by_30_degrees_moves_cells_one_bin(setup):
    """The sky rotated by 30 deg about the vertical (three sectors) gives the cells
    moved by one azimuth bin (G2.5 b bookkeeping, with a linear stand-in solver)."""
    ks, Dc = setup["ks"], setup["Dc"]
    mu0 = np.linspace(0, 1, 7)
    I0 = np.random.default_rng(2).random((ks.nE, 4, mu0.size)) + 5.0
    res = []
    for D in (Dc, np.roll(Dc, 3, axis=1)):
        tl = A3.order_tangents(D, 12, ks)
        dI = _mock_solve(tl) * 1e2
        Iaz = A3.azimuth_table(I0, dI, [(m, p) for m, p, _ in tl])
        res.append(A3.azimuth_cells(mu0, Iaz, ks))
    assert np.max(np.abs(res[1] - np.roll(res[0], 1, axis=-1))) < 1e-11 * np.abs(
        res[0]).max()


def test_cap_deficit_linear(setup):
    t, gw, ks, cw = setup["tab"], setup["gw"], setup["ks"], setup["cw"]
    D1 = A3.cap_deficit({0: t, 1: t}, [0.25, 0.75], gw, ks, cw)
    assert np.max(np.abs(D1 - setup["Dc"])) < 1e-15 * np.abs(D1).max() + 1e-300
    assert np.array_equal(A3.cap_deficit({0: t}, [1.0], gw, ks, cw), setup["Dc"])


def test_run_at_M_is_partial_sum_of_run_at_2M(setup):
    """GATES.md, entry on G2.5(a): on a reduced geometry a run at M equals the
    partial sum of a run at 2M to 1e-8 (orders independent; here M = 2, 2M = 4,
    the real guarded tangent solver, batches of different composition)."""
    import solver3d as S3

    r, ks, Dc = setup["r"], setup["ks"], setup["Dc"]
    ts = S3.TangentSolver(r, ks, guard=S3.GUARD_PROTOTYPE)
    t2 = A3.order_tangents(Dc, 2, ks)
    t4 = A3.order_tangents(Dc, 4, ks)
    r2 = A3.run_tangents(ts, t2, batch=5, log=lambda s: None)
    r4 = A3.run_tangents(ts, t4, batch=3, log=lambda s: None)
    c2 = A3.azimuth_cells(r2["mu0"], A3.azimuth_table(r2["I"], r2["dI"],
                                                      r2["tangents"]), ks)
    c4 = A3.azimuth_cells(r4["mu0"], A3.azimuth_table(r4["I"], r4["dI"],
                                                      r4["tangents"], M=2), ks)
    rel = np.abs(c4 - c2) / np.abs(c2)
    assert np.nanmax(rel) < 1e-8, np.nanmax(rel)
    full = A3.azimuth_cells(r4["mu0"], A3.azimuth_table(r4["I"], r4["dI"],
                                                        r4["tangents"]), ks)
    assert np.nanmax(np.abs(full - c2) / np.abs(c2)) > 1e-8  # orders 3, 4 matter
