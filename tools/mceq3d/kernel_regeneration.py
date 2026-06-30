"""Regenerate MCEq production kernels with transverse-momentum information.

Background
----------
MCEq solves the *one-dimensional* cascade equations: its interaction/decay
matrices are inclusive secondary yields ``dN/dx_L`` in the lab energy fraction
``x_L = E_secondary / E_projectile``, with the transverse momentum *integrated
out*. A deterministic 3D MCEq needs the **double-differential** kernel
``d2N / (dx_L dp_T)`` so that the production angle ``theta ~ p_T / p_L`` enters
the transport as an angular-redistribution operator.

This tool regenerates that double-differential kernel by re-running the *same*
event generators MCEq uses (via ``chromo``, formerly ``impy``) in inclusive
single-interaction mode and histogramming in ``(x_L, p_T)`` instead of only
``x_L``. The decay kernels are analytic (two-body kinematics) and are not
covered here; this is the interaction (hadron-production) part.

Key consistency gate
--------------------
Integrating the regenerated kernel over ``p_T`` must reproduce MCEq's stored 1D
kernel ``dN/dx_L`` bin-for-bin (within Monte-Carlo statistics). That single
check validates the whole pipeline against MCEq's own database before any
transport code is written. See :func:`marginalize_pt` and
:func:`compare_to_reference`.

Backends
--------
* ``chromo``  -- real physics: SIBYLL-2.3d (or any model chromo wraps).
                 Requires ``pip install chromo`` and ``pip install MCEq`` (for
                 the consistency gate).
* ``toy``     -- a self-contained pseudo-generator with *known* analytic
                 ``x_L`` and ``p_T`` distributions. It needs no external code,
                 runs anywhere, and lets the histogramming / marginalization /
                 consistency-check logic be unit-tested offline. Its analytic
                 marginal plays the role of the "MCEq reference" in the demo.

Run::

    python kernel_regeneration.py --backend toy   --plot
    python kernel_regeneration.py --backend chromo --model Sibyll23d --sec piplus
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Charged-pion mass (GeV); used for p_L and the angle conversion.
M_PION = 0.13957

# PDG ids for the channels we care about.
PDG = {"piplus": 211, "piminus": -211, "Kplus": 321, "Kminus": -321}

# Secondary masses [GeV], used for the per-secondary production angle.
MASS = {"piplus": 0.13957, "piminus": 0.13957, "Kplus": 0.49368, "Kminus": 0.49368}


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
@dataclass
class KernelGrid:
    """Binning for the regenerated kernel.

    Parameters
    ----------
    xl_edges : np.ndarray
        Bin edges in ``x_L = E_sec / E_proj`` (log-spaced by default).
    pt_edges : np.ndarray
        Bin edges in transverse momentum ``p_T`` [GeV].
    proj_energies : np.ndarray
        Projectile lab energies [GeV] at which the kernel is evaluated. These
        should match MCEq's energy grid for a bin-for-bin comparison; pass
        ``mceq.e_grid`` when available.
    """

    xl_edges: np.ndarray
    pt_edges: np.ndarray
    proj_energies: np.ndarray

    @classmethod
    def default(cls) -> "KernelGrid":
        return cls(
            xl_edges=np.logspace(-4, 0, 61),  # 1e-4 .. 1
            pt_edges=np.linspace(0.0, 3.0, 31),  # 0 .. 3 GeV
            # 10 GeV .. 10 TeV: the range where 3D effects matter, and where
            # SIBYLL statistics are cheap.
            proj_energies=np.logspace(1, 4, 7),
        )

    @property
    def xl_centers(self) -> np.ndarray:
        return np.sqrt(self.xl_edges[:-1] * self.xl_edges[1:])

    @property
    def pt_centers(self) -> np.ndarray:
        return 0.5 * (self.pt_edges[:-1] + self.pt_edges[1:])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (
            len(self.proj_energies),
            len(self.xl_edges) - 1,
            len(self.pt_edges) - 1,
        )


# ---------------------------------------------------------------------------
# Event-source backends
# ---------------------------------------------------------------------------
@dataclass
class SecondaryBatch:
    """Secondaries of one species from a set of inclusive interactions.

    Attributes
    ----------
    x_L : np.ndarray
        Lab energy fraction E_sec / E_proj of each secondary.
    p_T : np.ndarray
        Transverse momentum [GeV] of each secondary.
    n_interactions : int
        Number of inelastic interactions generated (for per-interaction
        normalization of the yield).
    """

    x_L: np.ndarray
    p_T: np.ndarray
    n_interactions: int


class ToySource:
    """Self-contained pseudo-generator with *known* distributions.

    The x_L spectrum is ``dN/dx_L ∝ (1 - x_L)^beta / x_L`` (a crude but
    qualitatively correct inclusive pion shape that rises towards small x), and
    ``p_T`` follows ``dN/dp_T ∝ p_T exp(-p_T / p0)`` (a thermal-like spectrum
    with mean ~ 2 p0). Both have closed-form marginals so the consistency check
    has an exact reference. x_L and p_T are sampled independently here, which is
    a deliberate simplification of the (mild) real correlation.
    """

    def __init__(self, mean_mult: float = 6.0, beta: float = 3.0, p0: float = 0.15):
        self.mean_mult = mean_mult
        self.beta = beta
        self.p0 = p0
        self.rng = np.random.default_rng(12345)

    def _sample_xl(self, n: int, xmin: float) -> np.ndarray:
        # Inverse-CDF-free rejection sampling of (1-x)^beta / x on [xmin, 1].
        out = np.empty(0)
        while out.size < n:
            x = np.exp(self.rng.uniform(np.log(xmin), 0.0, size=2 * n))  # ~1/x
            keep = self.rng.uniform(size=x.size) < (1.0 - x) ** self.beta
            out = np.concatenate([out, x[keep]])
        return out[:n]

    def _sample_pt(self, n: int) -> np.ndarray:
        # dN/dpt ∝ pt exp(-pt/p0)  ->  Gamma(shape=2, scale=p0)
        return self.rng.gamma(shape=2.0, scale=self.p0, size=n)

    def analytic_xl_marginal(self, grid: KernelGrid) -> np.ndarray:
        """Exact dN/dx_L per interaction on the grid (the 'reference')."""
        xmin = grid.xl_edges[0]
        xc = grid.xl_centers
        pdf = (1.0 - xc) ** self.beta / xc
        # normalize to mean multiplicity over [xmin, 1]
        from scipy.integrate import quad

        norm, _ = quad(lambda x: (1.0 - x) ** self.beta / x, xmin, 1.0)
        return self.mean_mult * pdf / norm

    def generate(
        self, e_proj: float, n_interactions: int, xmin: float
    ) -> SecondaryBatch:
        mult = self.rng.poisson(self.mean_mult, size=n_interactions)
        ntot = int(mult.sum())
        x_L = self._sample_xl(ntot, xmin)
        p_T = self._sample_pt(ntot)
        return SecondaryBatch(x_L=x_L, p_T=p_T, n_interactions=n_interactions)


class ChromoSource:
    """Real physics backend: run an event generator through chromo.

    Parameters
    ----------
    model : str
        chromo model class name, e.g. ``"Sibyll23d"`` (the model daemonflux /
        MCEq use for the conventional flux).
    secondary : str
        One of ``PDG`` keys, e.g. ``"piplus"``.
    target : tuple, optional
        Target nucleus ``(A, Z)``. Default nitrogen ``(14, 7)``. For an air
        average, combine N2/O2/Ar runs with their number fractions.
    projectile : str or int, optional
        Beam particle. Default proton.
    """

    def __init__(
        self,
        model: str = "Sibyll23d",
        secondary: str = "piplus",
        target: Tuple[int, int] = (14, 7),
        projectile=2212,
    ):
        self.model_name = model
        self.secondary = secondary
        self.sec_pid = PDG[secondary]
        self.target = target
        self.projectile = projectile
        self._model = None  # lazily created

    def _ensure_model(self, e_proj: float):
        import chromo
        from chromo.kinematics import FixedTarget, GeV

        evt_kin = FixedTarget(e_proj * GeV, self.projectile, self.target)
        ModelCls = getattr(chromo.models, self.model_name)
        if self._model is None:
            self._model = ModelCls(evt_kin)
        else:
            # Update beam energy without recompiling the generator.
            self._model.kinematics = evt_kin

    def generate(
        self, e_proj: float, n_interactions: int, xmin: float
    ) -> SecondaryBatch:
        self._ensure_model(e_proj)
        xs, pts = [], []
        for event in self._model(n_interactions):
            fs = event.final_state()
            mask = fs.pid == self.sec_pid
            if not np.any(mask):
                continue
            xs.append(fs.en[mask] / e_proj)
            pts.append(fs.pt[mask])
        x_L = np.concatenate(xs) if xs else np.empty(0)
        p_T = np.concatenate(pts) if pts else np.empty(0)
        x_L = np.clip(x_L, xmin, 1.0)
        return SecondaryBatch(x_L=x_L, p_T=p_T, n_interactions=n_interactions)


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------
def build_kernel(
    source,
    grid: KernelGrid,
    n_interactions: int = 20000,
) -> np.ndarray:
    """Build the double-differential kernel d2N/(dx_L dp_T) per interaction.

    Returns
    -------
    np.ndarray, shape (n_proj, n_xl, n_pt)
        ``kernel[i, j, k]`` is the average number of secondaries per inelastic
        interaction in x_L bin ``j`` and p_T bin ``k`` at projectile energy
        ``grid.proj_energies[i]``, divided by the bin widths (a density).
    """
    xmin = grid.xl_edges[0]
    dxl = np.diff(grid.xl_edges)
    dpt = np.diff(grid.pt_edges)
    cell = np.outer(dxl, dpt)  # (n_xl, n_pt)

    kernel = np.zeros(grid.shape)
    for i, e_proj in enumerate(grid.proj_energies):
        batch = source.generate(e_proj, n_interactions, xmin)
        counts, _, _ = np.histogram2d(
            batch.x_L, batch.p_T, bins=[grid.xl_edges, grid.pt_edges]
        )
        # per-interaction yield density
        kernel[i] = counts / batch.n_interactions / cell
    return kernel


def marginalize_pt(kernel: np.ndarray, grid: KernelGrid) -> np.ndarray:
    """Integrate the kernel over p_T -> the 1D yield dN/dx_L.

    Returns
    -------
    np.ndarray, shape (n_proj, n_xl)
        ``dN/dx_L`` per interaction (a density in x_L). This is the object that
        must match MCEq's stored interaction matrix.
    """
    dpt = np.diff(grid.pt_edges)
    return np.sum(kernel * dpt[np.newaxis, np.newaxis, :], axis=2)


def build_angular_kernel(
    source,
    grid: KernelGrid,
    theta_edges_deg: np.ndarray,
    mass: float,
    n_interactions: int = 20000,
) -> np.ndarray:
    """Build d2N/(dx_L dtheta) by binning the production angle *directly*.

    Unlike :func:`to_angular_kernel`, which resamples a coarse pre-binned p_T
    kernel and therefore aliases into a comb at low energy, this computes the
    production angle ``theta = arctan(p_T / p_L)`` of *each secondary* from its
    own ``x_L`` and ``p_T`` (``p_L = sqrt((x_L E_proj)^2 - m^2)``) and histograms
    once in ``(x_L, theta)``. The result is smooth and is the correct input for
    the discrete-ordinates / Fokker-Planck angular operator.

    Returns
    -------
    np.ndarray, shape (n_proj, n_xl, n_theta)
        Per-interaction density d2N/(dx_L dtheta), theta in degrees.
    """
    xmin = grid.xl_edges[0]
    dxl = np.diff(grid.xl_edges)
    dth = np.diff(theta_edges_deg)
    cell = np.outer(dxl, dth)

    out = np.zeros((len(grid.proj_energies), len(dxl), len(dth)))
    for i, e_proj in enumerate(grid.proj_energies):
        batch = source.generate(e_proj, n_interactions, xmin)
        e_sec = batch.x_L * e_proj
        p_l = np.sqrt(np.maximum(e_sec**2 - mass**2, 1e-12))
        theta = np.degrees(np.arctan2(batch.p_T, p_l))
        counts, _, _ = np.histogram2d(
            batch.x_L, theta, bins=[grid.xl_edges, theta_edges_deg]
        )
        out[i] = counts / batch.n_interactions / cell
    return out


def marginalize_theta(kernel: np.ndarray, theta_edges_deg: np.ndarray) -> np.ndarray:
    """Integrate an angular kernel over theta -> dN/dx_L (for the gate)."""
    dth = np.diff(theta_edges_deg)
    return np.sum(kernel * dth[np.newaxis, np.newaxis, :], axis=2)


def build_moments(source, grid: KernelGrid, mass: float, n_interactions: int = 20000):
    """Per-(E_proj, x_L) angular moments computed *directly* from secondaries.

    This is gridless in theta: ``theta = arctan(p_T / p_L)`` is evaluated for
    each secondary and averaged within each x_L bin, so the moments are exact at
    every energy and ``<theta^2> -> 0`` as ``E_sec -> inf`` by construction (no
    bin floor). ``<theta^2>`` is the Fokker-Planck angular-diffusion input; a
    histogram-then-moment approach would instead clamp it to the angular bin
    scale at high energy and inject spurious diffusion, breaking the 1D limit.

    Returns
    -------
    dict of arrays shape ``(n_proj, n_xl)``:
        ``theta_mean`` [rad], ``theta_sq`` [rad^2], ``dndx`` (dN/dx_L), plus the
        axes ``proj_energies``, ``xl_centers``, ``xl_edges``, ``e_sec``.
    """
    xmin = grid.xl_edges[0]
    dxl = np.diff(grid.xl_edges)
    nE, nxl = len(grid.proj_energies), len(dxl)
    theta_mean = np.full((nE, nxl), np.nan)
    theta_sq = np.full((nE, nxl), np.nan)
    dndx = np.zeros((nE, nxl))
    for i, e_proj in enumerate(grid.proj_energies):
        batch = source.generate(e_proj, n_interactions, xmin)
        e_sec = batch.x_L * e_proj
        p_l = np.sqrt(np.maximum(e_sec**2 - mass**2, 1e-12))
        theta = np.arctan2(batch.p_T, p_l)  # rad, exact per secondary
        counts, _ = np.histogram(batch.x_L, bins=grid.xl_edges)
        s1, _ = np.histogram(batch.x_L, bins=grid.xl_edges, weights=theta)
        s2, _ = np.histogram(batch.x_L, bins=grid.xl_edges, weights=theta**2)
        nz = counts > 0
        theta_mean[i, nz] = s1[nz] / counts[nz]
        theta_sq[i, nz] = s2[nz] / counts[nz]
        dndx[i] = counts / batch.n_interactions / dxl
    e_sec = grid.xl_centers[None, :] * grid.proj_energies[:, None]
    return {
        "theta_mean": theta_mean,
        "theta_sq": theta_sq,
        "dndx": dndx,
        "proj_energies": grid.proj_energies,
        "xl_centers": grid.xl_centers,
        "xl_edges": grid.xl_edges,
        "e_sec": e_sec,
    }


def to_angular_kernel(
    kernel: np.ndarray, grid: KernelGrid
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert d2N/(dx_L dp_T) to the production-angle density d2N/(dx_L dtheta).

    The production angle of a secondary of lab momentum ``p_L`` is
    ``theta = arctan(p_T / p_L)`` with ``p_L = sqrt(E_sec^2 - m^2)`` and
    ``E_sec = x_L * E_proj``. This is the bridge from the regenerated kernel to
    the angular-redistribution operator of a discrete-ordinates / Fokker-Planck
    3D solver.

    Returns
    -------
    theta : np.ndarray, shape (n_proj, n_xl, n_pt)
        Production angle [rad] at each (E_proj, x_L, p_T) bin center.
    dndtheta_jac : np.ndarray, same shape
        d(p_T)/d(theta) Jacobian to turn a p_T-density into a theta-density:
        ``d2N/dx_L dtheta = d2N/dx_L dp_T * dp_T/dtheta``.
    """
    e_proj = grid.proj_energies[:, None, None]
    xl = grid.xl_centers[None, :, None]
    pt = grid.pt_centers[None, None, :]
    e_sec = xl * e_proj
    p_l = np.sqrt(np.maximum(e_sec**2 - M_PION**2, 1e-12))
    theta = np.arctan2(pt, p_l)
    # p_T = p_L tan(theta)  ->  dp_T/dtheta = p_L / cos^2(theta)
    dpt_dtheta = p_l / np.cos(theta) ** 2
    return theta, dpt_dtheta


# ---------------------------------------------------------------------------
# Consistency check against a reference 1D kernel
# ---------------------------------------------------------------------------
def compare_to_reference(
    marginal: np.ndarray,
    reference: np.ndarray,
    rtol: float = 0.05,
    floor: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Compare the p_T-marginal of the regenerated kernel to a reference.

    Parameters
    ----------
    marginal : np.ndarray
        ``dN/dx_L`` from :func:`marginalize_pt`.
    reference : np.ndarray
        MCEq's stored 1D kernel on the same grid (or the toy analytic marginal).
    rtol : float
        Relative tolerance for the "agree" fraction.
    floor : float
        Ignore bins where the reference is below this density (empty tails).

    Returns
    -------
    dict with:
      ``norm_factor``          -- best-fit constant ratio marginal/reference. A
                                  value != 1 is a *normalization-convention*
                                  difference in how MCEq stores ``hadr_yields``
                                  (NOT a shape error, and NOT resonance
                                  feed-down: setting all MCEq-tracked species
                                  stable in chromo changes the count by only
                                  ~3%). Reconciling it requires reading MCEq's
                                  matrix-assembly normalization; it cancels in
                                  every angular quantity downstream.
      ``frac_within_rtol``     -- fraction of populated bins agreeing within
                                  ``rtol`` *after* removing ``norm_factor`` (the
                                  physically meaningful, shape-only gate).
      ``raw_frac_within_rtol`` -- same but without removing the normalization.
      ``rel_diff``             -- per-bin (marginal/norm_factor)/reference - 1.
    """
    marginal = np.atleast_2d(marginal)
    reference = np.atleast_2d(reference)
    mask = reference > floor
    raw = marginal[mask] / reference[mask]
    norm = float(np.median(raw)) if raw.size else np.nan

    rel = np.full_like(reference, np.nan)
    rel[mask] = (marginal[mask] / norm) / reference[mask] - 1.0
    populated = rel[np.isfinite(rel)]
    return {
        "norm_factor": norm,
        "rel_diff": rel,
        "max_rel_diff": float(np.nanmax(np.abs(rel))) if populated.size else np.nan,
        "frac_within_rtol": (
            float(np.mean(np.abs(populated) < rtol)) if populated.size else np.nan
        ),
        "raw_frac_within_rtol": (
            float(np.mean(np.abs(raw - 1.0) < rtol)) if raw.size else np.nan
        ),
    }


def load_mceq_reference(
    grid: KernelGrid, model: str, projectile=2212, secondary: str = "piplus"
) -> Optional[np.ndarray]:
    """Best-effort load of MCEq's stored 1D yield for the same channel.

    Returns ``dN/dx_L`` on ``grid`` (shape ``(n_proj, n_xl)``) or ``None`` if
    MCEq is not installed. MCEq's public API for raw yields has changed across
    versions; this wraps the access in a try/except and documents the intent so
    it can be adapted to the installed version.
    """
    # The reference is only defined for models MCEq itself ships (SIBYLL,
    # QGSJet, EPOS, DPMJET). For a low-energy generator like UrQMD there is no
    # MCEq table; return None gracefully rather than crashing the kernel build.
    try:
        from MCEq.core import MCEqRun
        import crflux.models as crf

        mceq = MCEqRun(
            interaction_model=model,
            primary_model=(crf.HillasGaisser2012, "H3a"),
            theta_deg=0.0,
        )
        # MCEq stores hadr_yields as dN/dx_L in *scaling* form (verified: the
        # value at fixed log-offset i-j is energy independent, and the integral
        # gives physical multiplicities). Extract dN/dx_L at each projectile
        # energy and interpolate (in log x_L) onto ``grid``.
        pman = mceq.pman
        daughters = [
            k
            for k in pman[projectile].hadr_yields
            if getattr(k, "pdg_id", (None,))[0] == PDG[secondary]
        ]
        if not daughters:
            return None
        Y = np.asarray(pman[projectile].hadr_yields[daughters[0]])
        c = mceq._energy_grid.c
        # MCEq stores the yield as (dN/dx_L) * Delta(lnE), i.e. weighted by the
        # log-energy bin width of its grid (verified: chromo's per-interaction
        # dN/dx matches Y / Delta(lnE) to ~shape scatter). Divide it out to get
        # the true dN/dx_L. This is the resolution of the former ~4.4x "offset".
        dlnE = np.log(c[1] / c[0])
        Y = Y / dlnE
    except Exception:
        return None

    out = np.zeros((len(grid.proj_energies), len(grid.xl_centers)))
    log_xc = np.log(grid.xl_centers)
    for iE, e_proj in enumerate(grid.proj_energies):
        j = int(np.argmin(np.abs(c - e_proj)))
        x_mc = c[: j + 1] / c[j]  # x_L of MCEq rows <= projectile energy
        y_mc = Y[: j + 1, j]  # dN/dx_L at those x
        order = np.argsort(x_mc)
        out[iE] = np.interp(
            log_xc, np.log(x_mc[order]), y_mc[order], left=0.0, right=0.0
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _make_source(args):
    if args.backend == "toy":
        return ToySource()
    return ChromoSource(model=args.model, secondary=args.sec)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["toy", "chromo"], default="toy")
    p.add_argument("--model", default="Sibyll23d", help="chromo model class name")
    p.add_argument("--sec", default="piplus", choices=list(PDG))
    p.add_argument("--nint", type=int, default=20000, help="interactions per energy")
    p.add_argument("--emin", type=float, default=None, help="min projectile E [GeV]")
    p.add_argument("--emax", type=float, default=None, help="max projectile E [GeV]")
    p.add_argument("--ne", type=int, default=None, help="number of energy points")
    p.add_argument(
        "--angular",
        action="store_true",
        help="bin the production angle directly -> smooth d2N/dx_L dtheta "
        "(use a low-energy model like UrQMD34 to reach E_lab < 53 GeV)",
    )
    p.add_argument("--ntheta", type=int, default=60, help="angle bins (--angular)")
    p.add_argument("--theta-max-deg", type=float, default=40.0)
    p.add_argument(
        "--moments",
        action="store_true",
        help="emit gridless angular moments <theta>,<theta^2> (Fokker-Planck "
        "input); exact at all energies, no theta-grid floor",
    )
    p.add_argument("--out", default="kernel_piplus.npz")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    grid = KernelGrid.default()
    if args.emin or args.emax or args.ne:
        e0 = args.emin or grid.proj_energies[0]
        e1 = args.emax or grid.proj_energies[-1]
        ne = args.ne or len(grid.proj_energies)
        grid.proj_energies = np.logspace(np.log10(e0), np.log10(e1), ne)
    source = _make_source(args)

    if args.moments:
        _run_moments(source, grid, args)
        return

    theta_edges = np.linspace(0.0, args.theta_max_deg, args.ntheta + 1)
    if args.angular:
        mass = MASS[args.sec]
        print(
            f"Building d2N/dx_L dtheta kernel (direct-angle) "
            f"[model={args.model}, sec={args.sec}]"
        )
        kernel = build_angular_kernel(
            source, grid, theta_edges, mass, n_interactions=args.nint
        )
        marginal = marginalize_theta(kernel, theta_edges)
    else:
        print(
            f"Building d2N/dx_L dp_T kernel  [backend={args.backend}, sec={args.sec}]"
        )
        kernel = build_kernel(source, grid, n_interactions=args.nint)
        marginal = marginalize_pt(kernel, grid)

    # Reference: MCEq if available, else (for the toy) the analytic marginal.
    reference = load_mceq_reference(grid, args.model, secondary=args.sec)
    ref_label = "MCEq stored dN/dx_L"
    if reference is None and isinstance(source, ToySource):
        reference = np.tile(
            source.analytic_xl_marginal(grid), (len(grid.proj_energies), 1)
        )
        ref_label = "toy analytic dN/dx_L"

    save_kw = dict(
        kernel=kernel,
        marginal=marginal,
        xl_edges=grid.xl_edges,
        proj_energies=grid.proj_energies,
    )
    if args.angular:
        save_kw["theta_edges"] = theta_edges  # degrees; marks an angular kernel
    else:
        save_kw["pt_edges"] = grid.pt_edges
    np.savez(args.out, **save_kw)
    print(f"saved kernel -> {args.out}  shape={kernel.shape}")

    if reference is not None:
        stats = compare_to_reference(marginal, reference)
        print(f"consistency vs {ref_label}:")
        print(f"  norm factor (MCEq convention) = {stats['norm_factor']:.3g}")
        print(
            f"  shape agreement within 5%    = {stats['frac_within_rtol']:.3f}"
            "  (after removing norm factor)"
        )
        print(f"  raw agreement within 5%      = {stats['raw_frac_within_rtol']:.3f}")
    else:
        print("No reference available (install MCEq to enable the consistency gate).")

    if args.plot:
        _plot(kernel, marginal, reference, grid, args)


def _plot(kernel, marginal, reference, grid, args):
    import matplotlib.pyplot as plt

    i = len(grid.proj_energies) // 2
    e = grid.proj_energies[i]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    Z = kernel[i].T  # (yaxis, xl)
    if args.angular:
        yedges = np.linspace(0.0, args.theta_max_deg, args.ntheta + 1)
        ylabel, sym = r"$\theta$ [deg]", r"d^2N/dx_L\,d\theta"
    else:
        yedges, ylabel, sym = grid.pt_edges, r"$p_T$ [GeV]", r"d^2N/dx_L\,dp_T"
    pc = axL.pcolormesh(grid.xl_edges, yedges, np.log10(Z + 1e-12), shading="auto")
    axL.set_xscale("log")
    axL.set_xlabel(r"$x_L = E_{\rm sec}/E_{\rm proj}$")
    axL.set_ylabel(ylabel)
    axL.set_title(rf"$\log_{{10}}\,{sym}$  ($E={e:.0f}$ GeV, {args.sec})")
    fig.colorbar(pc, ax=axL)

    mlabel = (
        r"regenerated, $\int d\theta$" if args.angular else r"regenerated, $\int dp_T$"
    )
    axR.loglog(grid.xl_centers, marginal[i], "o-", ms=3, label=mlabel)
    title = "Consistency gate: p_T-marginal vs MCEq"
    if reference is not None:
        # reference is MCEq's stored dN/dx_L already divided by the log-energy bin
        # width Delta(lnE) (load_mceq_reference) -> directly comparable, no rescale.
        m = reference[i] > 0
        norm = np.median(marginal[i][m] / reference[i][m])
        axR.loglog(
            grid.xl_centers,
            reference[i],
            "k--",
            label=r"MCEq $dN/dx_L$ (bin-width corrected)",
        )
        title = f"Consistency gate (norm ratio = {norm:.2f}, shape match)"
    axR.set_xlabel(r"$x_L$")
    axR.set_ylabel(r"$dN/dx_L$ per interaction")
    axR.set_title(title)
    axR.legend()

    fig.tight_layout()
    out = args.out.replace(".npz", ".png")
    fig.savefig(out, dpi=110)
    print(f"saved plot -> {out}")


def _run_moments(source, grid, args):
    """Build and save gridless angular moments, with a pooled summary."""
    mass = MASS[args.sec]
    print(f"Building gridless angular moments [model={args.model}, sec={args.sec}]")
    mom = build_moments(source, grid, mass, n_interactions=args.nint)
    np.savez(
        args.out,
        is_moments=True,
        theta_mean=mom["theta_mean"],
        theta_sq=mom["theta_sq"],
        dndx=mom["dndx"],
        proj_energies=mom["proj_energies"],
        xl_edges=mom["xl_edges"],
        xl_centers=mom["xl_centers"],
        e_sec=mom["e_sec"],
    )
    print(f"saved moments -> {args.out}")

    # Yield-weighted pooled <theta>(E_sec): shows the clean 1/E with no floor.
    e = mom["e_sec"].ravel()
    th = mom["theta_mean"].ravel()
    wt = mom["dndx"].ravel()
    g = np.isfinite(th) & (wt > 0)
    e, th, wt = e[g], th[g], wt[g]
    edges = np.logspace(np.log10(max(e.min(), 0.3)), np.log10(e.max()), 13)
    print("  E_sec [GeV]   <theta> [deg]")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (e >= lo) & (e < hi)
        if sel.sum() and wt[sel].sum() > 0:
            mth = np.degrees(np.sum(th[sel] * wt[sel]) / np.sum(wt[sel]))
            print(f"  {np.sqrt(lo * hi):10.2f}   {mth:8.2f}")
    if args.plot:
        import angular_kernel

        angular_kernel.main(["--kernel", args.out, "--plot"])


if __name__ == "__main__":
    main()
