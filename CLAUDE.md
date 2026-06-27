# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

daemonflux is a Python package providing a data-driven, muon-calibrated atmospheric neutrino and muon flux model. It uses pre-computed scipy UnivariateSpline objects (stored as pickled files) to evaluate fluxes and propagate uncertainties via Jacobian-based parameter corrections with covariance tracking.

## Commands

```bash
# Install in editable mode
pip install -e .

# Install with test dependencies
pip install -e .[test]

# Run all tests
pytest -vv

# Run a single test
pytest tests/test_daemonflux.py::test_name -vv
```

## Code Quality

- **Formatter**: Black (default settings, line length 88; pre-commit hook psf/black 24.4.2)
- **Linter**: Flake8 (config in setup.cfg, ignore E203)
- **Docstrings**: NumPy convention
- Pre-commit hooks configured in `.pre-commit-config.yaml`

## Architecture

The package source lives in `src/daemonflux/`. There are three core classes in `flux.py`:

- **`Flux`** — Public API. Manages multiple locations/experiments, handles lazy loading of spline and calibration data files. Users call `flux()`, `error()`, and `chi2()`.
- **`_FluxEntry`** — Internal per-location flux evaluator. Evaluates splines, interpolates across zenith angles, applies parameter corrections via Jacobian matrices.
- **`Parameters`** — Manages model parameters, their covariance/inverse-covariance matrices, and supports iteration. Provides `errors`, `invcov`, and `chi2` properties.

`utils.py` contains helpers for file downloading/caching, angle formatting, covariance matrix operations, and the list of supported flux quantities.

`geomagnetic.py` provides the optional 3D / geomagnetic extension. `GeomagneticModel`
computes a directional admittance factor `G(E, zenith, azimuth) ∈ [0, 1]` (Störmer
cutoff + penumbral admittance) that `_FluxEntry` multiplies onto the 1D flux/error.
When no model is attached (default), behaviour is identical to the published 1D model.
The model is the single plug-point for replacing the analytic cutoff with admittance
ratios from a full 3D Monte-Carlo. See README "3D / geomagnetic corrections".

Data files (`src/daemonflux/data/`) are pickled scipy spline objects for different detector locations (generic, Kamioka) and calibration sets (default, with_deis). They are downloaded from GitHub on first use and cached locally.

## Key Conventions

- Flux units: neutrinos use E³-weighted (GeV²/(cm² s sr)), muons use p³-weighted ((GeV/c)²/(cm² s sr))
- Supported quantities include: `muflux`, `nueflux`, `numuflux`, individual charge/flavor components, ratios, and `total_`-prefixed versions (conventional + prompt)
- The `flux()` method accepts scalar or array energy arguments
- Python 3.8+ required; scipy >= 1.8.0 required
