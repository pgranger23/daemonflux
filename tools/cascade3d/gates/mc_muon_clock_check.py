"""Review check: does the Monte Carlo's muon transport age slowing muons correctly?

``tools/mc3d/shower.transport_muon`` advances the decay clock of a step with the
``beta * gamma`` of the start of the step (steps of 30 g/cm2 of air at most).  This
script measures what that does to the number of decay neutrinos above a threshold,
with the Monte Carlo's **own function** on one side and a fine integration of the same
continuous slowing-down on the other (same stopping power, same atmosphere).

- Monte Carlo side: ``transport_muon`` is called with a random generator replaced by a
  stratified sequence u = (i + 1/2) / N, so the single exponential variate it draws
  scans the decay-time distribution deterministically (N-point midpoint rule).
- Exact side: ``dE/ds = -dedx(E) rho(h(s))`` and ``d tau/ds = m / (p c tau)`` by
  midpoint steps that change the momentum by 0.5% at most.
- Neutrinos: unpolarised three-body spectra in the muon rest frame (numu-like
  ``2 x^2 (3 - 2 x)``, nue-like ``12 x^2 (1 - x)``), boosted isotropically; the fraction
  above ``E_b`` in the laboratory is a one-dimensional quadrature.  The polarisation is
  left out on both sides.

Usage::

    source $N3/env.sh
    taskset -c 6 $PY tools/cascade3d/gates/mc_muon_clock_check.py [N [clock]]

``clock`` is passed to ``shower.Config(mu_clock=...)`` unless it is ``legacy`` (the
default, which calls ``Config`` as every run before 2026-11-19 did).  Output
``$N3/c1/review/mc_muon_clock.md`` (legacy) or ``mc_muon_clock_<clock>.md``; the last
line printed is the largest deviation of the table.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mc3d"))
import atmosphere as atm  # noqa: E402
import shower  # noqa: E402
from constants import CTAU_CM, M_MU, R_EARTH_CM  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
E_B = (0.1, 0.2, 0.4)
T0 = (0.15, 0.3, 0.6, 1.2, 2.5, 5.0)
# altitude km, cos theta
START = ((15.0, 1.0), (15.0, 0.3), (25.0, 1.0), (8.0, 1.0), (15.0, 0.05))
_X, _W = np.polynomial.legendre.leggauss(96)
_X, _W = 0.5 * (_X + 1.0), 0.5 * _W
SPEC = {"numu": 2.0 * _X**2 * (3.0 - 2.0 * _X), "nue": 12.0 * _X**2 * (1.0 - _X)}


class Stratified:
    """Stands in for the generator: ``random()`` returns one fixed number."""

    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def n_above(e_mu, e_b):
    """Neutrinos above ``e_b`` per decay of a muon of total energy ``e_mu``."""
    p = np.sqrt(max(e_mu * e_mu - M_MU * M_MU, 0.0))
    if e_mu + p <= 2.0 * e_b:
        return {k: 0.0 for k in SPEC}
    beta, gamma = p / e_mu, e_mu / M_MU
    c_min = (2.0 * e_b / (gamma * M_MU * _X) - 1.0) / beta
    frac = np.clip(0.5 * (1.0 - c_min), 0.0, 1.0)
    return {k: float(np.sum(_W * f * frac)) for k, f in SPEC.items()}


def monte_carlo(e0, r0, u0, n, clock="legacy"):
    """Neutrinos above each threshold per muon, through ``transport_muon``."""
    kw = {} if clock == "legacy" else {"mu_clock": clock}
    cfg = shower.Config(collinear=True, bfield=None, energy_loss=True, **kw)
    out = {(k, eb): 0.0 for k in SPEC for eb in E_B}
    n_dec = 0
    for i in range(n):
        kind, _, _, e = shower.transport_muon(
            Stratified((i + 0.5) / n), 13, e0, r0, u0, cfg
        )
        if kind != "decay":
            continue
        n_dec += 1
        for eb in E_B:
            for k, v in n_above(e, eb).items():
                out[(k, eb)] += v / n
    return out, n_dec / n


def exact(e0, r0, u0):
    """The same quantity for continuous slowing-down, by fine midpoint steps."""
    s_end, _ = atm.path_to_exit(r0, u0)
    r0n = float(np.linalg.norm(r0))
    r0u = float(np.dot(r0, u0))
    ctau = CTAU_CM[13]

    def rho(s):
        return atm.density(np.sqrt(r0n * r0n + 2.0 * s * r0u + s * s) - R_EARTH_CM)

    def rates(e, s):
        p = np.sqrt(max(e * e - M_MU * M_MU, 1e-12))
        return shower.dedx_air(e) * rho(s), M_MU / (p * ctau), p

    out = {(k, eb): 0.0 for k in SPEC for eb in E_B}
    e, s, tau = float(e0), 0.0, 0.0
    while s < s_end:
        de, dt, p = rates(e, s)
        if e + p <= 2.0 * min(E_B):
            break
        ds = min(5.0e3, s_end - s)
        if de > 0.0:
            ds = min(ds, 0.005 * p * p / (e * de))
        e_mid = e - 0.5 * ds * de
        de_m, dt_m, _ = rates(e_mid, s + 0.5 * ds)
        w = np.exp(-tau) * -np.expm1(-dt_m * ds)
        for eb in E_B:
            for k, v in n_above(e_mid, eb).items():
                out[(k, eb)] += w * v
        e, s, tau = e - ds * de_m, s + ds, tau + dt_m * ds
    return out, 1.0 - np.exp(-tau)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    clock = sys.argv[2] if len(sys.argv) > 2 else "legacy"
    worst = 0.0
    lines = ["# Monte Carlo muon decay clock against exact slowing-down", "",
             f"Clock: `{clock}`. "
             f"`shower.transport_muon` with a stratified decay variate (N = {n}) "
             "against a fine integration of the same stopping power and atmosphere. "
             "Entries: neutrinos above E_b per muon, Monte Carlo / exact - 1, in %. "
             "T0 is the muon kinetic energy at the start.", ""]
    for h_km, c in START:
        r0 = np.array([0.0, 0.0, R_EARTH_CM + h_km * 1e5])
        u0 = np.array([np.sqrt(1.0 - c * c), 0.0, -c])
        lines += [f"## start at {h_km:g} km, cos theta = {c:g}", "",
                  "| T0 [GeV] | " + " | ".join(
                      f"{k} > {eb:g}" for eb in E_B for k in SPEC)
                  + " | exact per muon (numu > 0.1) | decays in flight MC |",
                  "|---|" + "---|" * (len(E_B) * len(SPEC) + 2)]
        for t0 in T0:
            mc, f_mc = monte_carlo(t0 + M_MU, r0, u0, n, clock)
            ex, _ = exact(t0 + M_MU, r0, u0)
            cells = []
            for eb in E_B:
                for k in SPEC:
                    if ex[(k, eb)] > 1e-6:
                        dev = 100 * (mc[(k, eb)] / ex[(k, eb)] - 1)
                        worst = max(worst, abs(dev))
                        cells.append(f"{dev:+.2f}")
                    else:
                        cells.append("-")
            lines.append(f"| {t0:g} | " + " | ".join(cells)
                         + f" | {ex[('numu', 0.1)]:.4f} | {f_mc:.3f} |")
            print(lines[-1], flush=True)
        lines.append("")
    lines += [f"Largest deviation of the table: {worst:.2f}%.", ""]
    name = "mc_muon_clock.md" if clock == "legacy" else f"mc_muon_clock_{clock}.md"
    path = os.path.join(N3, "c1", "review", name)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(path)
    print(f"largest deviation {worst:.2f}%")


if __name__ == "__main__":
    main()
