"""The opt-in ``converged`` muon decay clock (``shower.MU_CLOCKS``, 2026-11-19).

The legacy clock ages a slowing muon with ``beta gamma`` of the start of each step
and so lets it decay late and at too low an energy.  The tests pin the legacy
transport (numbers produced by the code before the switch existed, commit af178e2),
check the converged clock against a fine integration, and the vectorised transport
against the scalar one.
"""
import math

import numpy as np
import pytest

import atmosphere as atm
import bend
import shower
import sky3d
from constants import CTAU_CM, M_MU, R_EARTH_CM

# pdg, total energy [GeV], altitude [cm], cos(zenith), decay time [lifetimes]
CASES = [(13, 0.25, 15e5, 1.0, 1.5), (-13, 0.40, 15e5, 0.3, 1.0),
         (13, 0.70, 8e5, 1.0, 0.8), (-13, 1.30, 25e5, 0.6, 2.0),
         (13, 5.00, 20e5, 0.1, 0.5), (-13, 0.30, 4e5, 1.0, 3.0)]

# kind, path [cm], final energy [GeV] of the scalar and of the vectorised legacy
# transport, from tools/mc3d at commit af178e2
LEGACY = [
    ("decay", 175506.39112651348, 0.14926826480140243,
     175506.3928567171, 0.14926826752472971),
    ("decay", 225000.47216907702, 0.3079425766002468,
     225000.47222449633, 0.30794257683936216),
    ("decay", 274145.75440859795, 0.37878163784139657,
     274143.0509530306, 0.3787752375721251),
    ("decay", 1520145.7629706115, 1.048587207667562,
     1520145.8433265078, 1.0485873942923327),
    ("decay", 1520796.8577786465, 4.690374614749216,
     1520796.8578178675, 4.690374614999585),
    ("stop", 106272.66091990471, 0.1056583745,
     106272.66091990471, 0.1056583745),
]
CODE = {"decay": bend.MU_DECAY, "ground": bend.MU_GROUND,
        "escape": bend.MU_ESCAPE, "stop": bend.MU_STOP}


class _Rng:
    """One fixed uniform variate; counts the draws."""

    def __init__(self, v):
        self.v, self.n = v, 0

    def random(self):
        self.n += 1
        return self.v


def _state(h, cz):
    r = np.array([0.0, 0.0, R_EARTH_CM + h])
    u = np.array([math.sqrt(1.0 - cz * cz), 0.0, -cz])
    return r, u


def _scalar(case, **kw):
    pdg, e0, h, cz, tt = case
    r, u = _state(h, cz)
    rng = _Rng(math.exp(-tt))
    kind, r1, _, e1 = shower.transport_muon(
        rng, pdg, e0, r, u, shower.Config(collinear=True, bfield=None, **kw))
    return kind, float(np.linalg.norm(r1 - r)), float(e1), rng.n


def _vector(case, **kw):
    pdg, e0, h, cz, tt = case
    r, u = _state(h, cz)
    k, rv, _, ev = bend.transport_muons(np.array([pdg]), np.array([e0]), r[None],
                                        u[None], np.array([tt]), None, **kw)
    return int(k[0]), float(np.linalg.norm(rv[0] - r)), float(ev[0])


def _fine(case, dp=1e-3):
    """Decay path [cm] and energy by mid-point steps losing ``dp`` of the momentum."""
    _, e, h, cz, tt = case
    m, ctau = M_MU, CTAU_CM[13]
    r, u = _state(h, cz)

    def rho(s):
        return float(atm.density(np.linalg.norm(r + s * u) - R_EARTH_CM))

    s, t = 0.0, 0.0
    while e > m * 1.0005:
        p2 = e * e - m * m
        de = shower.dedx_air(e) * rho(s)
        ds = min(dp * p2 / e / de, 2.0e4)
        em = e - 0.5 * ds * de
        dem = shower.dedx_air(em) * rho(s + 0.5 * ds)
        dtau = ds * m / (math.sqrt(em * em - m * m) * ctau)
        if t + dtau >= tt:
            f = (tt - t) / dtau
            return s + f * ds, e - f * ds * dem
        t, s, e = t + dtau, s + ds, e - ds * dem
    return s, m


def test_defaults_are_legacy_and_names_are_checked():
    assert shower.MU_CLOCKS == ("legacy", "converged")
    assert shower.Config().mu_clock == "legacy"
    assert sky3d.Options().muon_clock == "legacy"
    assert bend._MU_CLOCK_DP == shower._MU_CLOCK_DP
    assert bend._MU_CLOCK_IT == shower._MU_CLOCK_IT
    assert bend._MU_CLOCK_GUESS == shower._MU_CLOCK_GUESS
    with pytest.raises(ValueError):
        shower.Config(mu_clock="exact")
    with pytest.raises(ValueError):
        sky3d.Options(muon_clock="exact")
    with pytest.raises(ValueError):
        _vector(CASES[0], clock="exact")


def test_run_signature_unchanged_unless_switched():
    leg = sky3d.Options().signature()
    assert np.array_equal(leg, sky3d.Options(muon_clock="legacy").signature())
    conv = sky3d.Options(muon_clock="converged").signature()
    assert np.array_equal(conv[:len(leg)], leg)
    assert list(conv[len(leg):]) == [-4.0, 1.0]
    both = sky3d.Options(thresholds="chain", muon_clock="converged").signature()
    assert list(both[len(leg):]) == [-3.0, 1.0, -4.0, 1.0]


@pytest.mark.parametrize("case,want", list(zip(CASES, LEGACY)))
def test_legacy_clock_is_what_it_was(case, want):
    kind, path, e1, ndraw = _scalar(case)
    assert kind == want[0] and ndraw == 1
    assert path == pytest.approx(want[1], rel=1e-12)
    assert e1 == pytest.approx(want[2], rel=1e-12)
    k, pathv, ev = _vector(case)
    assert k == CODE[want[0]]
    assert pathv == pytest.approx(want[3], rel=1e-12)
    assert ev == pytest.approx(want[4], rel=1e-12)
    # the default spelled out is the same code path
    assert _scalar(case, mu_clock="legacy")[:3] == (kind, path, e1)
    assert _vector(case, clock="legacy") == (k, pathv, ev)


@pytest.mark.parametrize("case", CASES[:5])
def test_converged_clock_matches_fine_integration(case):
    s_x, e_x = _fine(case)
    kind, path, e1, ndraw = _scalar(case, mu_clock="converged")
    assert kind == "decay" and ndraw == 1
    assert path == pytest.approx(s_x, rel=2e-3)
    assert e1 - M_MU == pytest.approx(e_x - M_MU, rel=2e-3)
    k, pathv, ev = _vector(case, clock="converged")
    assert k == bend.MU_DECAY
    assert pathv == pytest.approx(s_x, rel=2e-3)
    assert ev - M_MU == pytest.approx(e_x - M_MU, rel=2e-3)


def test_legacy_clock_decays_late_and_slow():
    """The defect the switch removes: a 0.14 GeV (kinetic) muon made at 15 km and
    decaying after 1.5 lifetimes travels 8% too far and keeps 15% too little
    kinetic energy."""
    s_x, e_x = _fine(CASES[0])
    _, path, e1, _ = _scalar(CASES[0])
    assert path / s_x - 1.0 == pytest.approx(0.085, abs=0.01)
    assert (e1 - M_MU) / (e_x - M_MU) - 1.0 == pytest.approx(-0.153, abs=0.015)


def test_stopping_muon_stops_under_both_clocks():
    s_x, _ = _fine(CASES[5])
    kind, path, e1, _ = _scalar(CASES[5], mu_clock="converged")
    assert kind == "stop" and e1 == M_MU
    assert path == pytest.approx(s_x, rel=2e-3)
    k, pathv, ev = _vector(CASES[5], clock="converged")
    assert k == bend.MU_STOP and ev <= M_MU * 1.0005
    assert pathv == pytest.approx(s_x, rel=2e-3)


def test_clocks_agree_without_energy_loss():
    for case in CASES:
        assert (_scalar(case, energy_loss=False, mu_clock="converged")
                == _scalar(case, energy_loss=False))
        assert (_vector(case, energy_loss=False, clock="converged")
                == _vector(case, energy_loss=False))


def test_vectorised_converged_clock_matches_scalar():
    rng = np.random.default_rng(11)
    n = 300
    h = rng.uniform(5e5, 40e5, n)
    cz = np.cos(np.radians(rng.uniform(0.0, 89.0, n)))
    e = np.exp(rng.uniform(np.log(0.15), np.log(20.0), n))
    tt = -np.log(rng.random(n))
    pdg = np.where(rng.random(n) < 0.5, 13, -13)
    ru = [_state(h[i], cz[i]) for i in range(n)]
    r = np.array([x[0] for x in ru])
    u = np.array([x[1] for x in ru])
    k, rv, _, ev = bend.transport_muons(pdg, e, r, u, tt, None, clock="converged")
    cfg = shower.Config(collinear=False, bfield=None, mu_clock="converged")
    n_dec = 0
    for i in range(n):
        kind, rs, _, es = shower.transport_muon(_Rng(math.exp(-tt[i])), int(pdg[i]),
                                                e[i], r[i], u[i], cfg)
        assert CODE[kind] == k[i]
        if kind == "decay":
            n_dec += 1
            # the two transports choose different steps; same bound as the
            # legacy comparison in test_m3b (measured: 48 m, 4e-4)
            assert np.linalg.norm(rs - rv[i]) < 1.0e4                 # 100 m
            assert abs(es - ev[i]) / (es - M_MU) < 2e-3
    assert n_dec > 100
