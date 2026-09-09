# daemonflux: DAta-drivEn and MuOn-calibrated Neutrino flux

Daemonflux is a tabulated/splined version of the an atmospheric flux model calibrated on muon spectrometer data. For the details about how daemonflux is built and calibrated to muon data [the following publication](https://inspirehep.net/literature/2637710).

```
@article{Yanez:2023lsy,
    author = "Ya\~nez, Juan Pablo and Fedynitch, Anatoli",
    title = "{daemonflux: DAta-drivEn MuOn-calibrated Neutrino Flux}",
    eprint = "2303.00022",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "2",
    year = "2023"
}
```

## Requirements
 * `Python > 3.7`, `numpy`, `scipy`
 * `matplotlib` for examples

## Installation
a) From PyPi: 
    
    pip install daemonflux
    
b) From source in editable mode, so the package gets updated after each `git pull`:
```bash
$ git clone https://github.com/mceq-project/daemonflux
$ cd daemonflux
$ python3 -m pip install -e .
```

## Quickstart

To see more features and a detailed example, refer to the [example](examples/example.ipynb) notebook. In summary, the process of calculating calibrated fluxes from the provided tables is as follows:

    from daemonflux import Flux
    import numpy as np
    import matplotlib.pyplot as plt

    daemonflux = Flux(location='generic')
    egrid = np.logspace(0,5) # Energy in GeV

    fl = daemonflux.flux(egrid, '15', 'numuflux')
    err = daemonflux.error(egrid, '15', 'numuflux')
    plt.loglog(egrid, fl, color='k')
    plt.fill_between(egrid, fl + err, fl - err,
        color='r', alpha=.3, label=r'1$\sigma$ error')
    ...

Resulting in the following figure:

![Muon Neutrino Flux plot](flux_example.png "Muon neutrino flux scaled by $E^3$ for clarity.")


## Explanation of quantities and units

For **neutrinos**, the methods `Flux.flux` and `Flux.error` return values in the units of $(E/\text{GeV})^3/(\text{GeV }\text{s }\text{sr }\text{cm}^2)$, i.e. multiplied by $E^3$. For **muon quantities** are reported as a function of total momentum instead of energy, i.e. the units are  $(p/\text{(GeV/c)})^3/(\text{(GeV/c) } \text{s }\text{sr }\text{cm}^2)$. Natural units $\hbar=c=1$ are used everywhere.

The quantities are: 

- muons: `muflux`, `muratio`, `mu+`, `mu-`,
- muon neutrinos: `numuflux`, `numuratio`, `numu`, `antinumu`, `flavorratio`
- electron neutrinos: `nueflux`, `nueratio`, `nue`, `antinue`, `flavorratio`

Those titled XXXflux are the sum of particle and antiparticle fluxes `numuflux = numu + antinumu`, the ratio is `numuratio = numu/antinumu`, and the is defined as `flavorratio = (numu + antinumu)/(nue + antinue)`.

The `total_` quantities, such as `total_muflux`, represent the total flux, which includes both conventional and prompt atmospheric fluxes. However, unlike the conventional flux, the prompt flux is not calibrated using the daemonflux method, as surface muons are not sensitive to prompt fluxes. As a result, the prompt component does not include correction parameters or errors. It is important to note, however, that the conventional part of the flux remains calibrated, so the total_ flux is simply the sum of the calibrated conventional and uncalibrated prompt fluxes.

## 3D / geomagnetic corrections for low energies (experimental)

The tabulated daemonflux fluxes are computed with MCEq, which solves the
**one-dimensional** cascade equations. This is accurate above a few GeV but
misses geomagnetic effects that matter below ~2 GeV — most importantly the
direction-dependent **rigidity cutoff** and the resulting **East–West
asymmetry**. The paper deliberately avoids this regime (muon data were cut at
5 GeV "to avoid complications introduced by geomagnetic effects, not yet
included in MCEq").

daemonflux now provides an optional, fast analytic geomagnetic correction that
multiplies the 1D flux by a directional admittance factor `G(E, zenith, azimuth)
∈ [0, 1]`:

```python
from daemonflux import Flux
import numpy as np

# Enable the geomagnetic model for a detector site
flux = Flux(location="generic", geomag_location="kamioka")
# (equivalently: flux.set_geomagnetic_model("kamioka"))

egrid = np.logspace(-0.5, 2, 50)   # 0.3 – 100 GeV
fE = flux.flux(egrid, 70.0, "numuflux", azimuth_deg=90.0)    # from the East
fW = flux.flux(egrid, 70.0, "numuflux", azimuth_deg=270.0)   # from the West
# fW / fE > 1 at low energy is the East–West effect; -> 1 above the cutoff.
```

Azimuth is geographic (N=0, E=90, S=180, W=270). If `azimuth_deg` is omitted, the
azimuth-averaged cutoff is applied (still captures the latitude / low-energy
suppression). Ratio quantities (`numuratio`, …) and the high-energy flux are left
unchanged, so **existing 1D usage is completely unaffected** when no
`geomag_location` is set.

Known sites: `kamioka`, `southpole`, `ino`, `gransasso`, `snolab`. A custom
location is supplied via `daemonflux.geomagnetic.GeomagneticSite`. See
[`examples/geomagnetic_example.py`](examples/geomagnetic_example.py).

> **Scope.** This is a first-cut analytic model (Störmer dipole cutoff + a
> penumbral admittance, with a single effective primary→lepton inelasticity). It
> reproduces the latitude dependence, the low-energy cutoff, and the East–West
> asymmetry, and is designed as a drop-in plug point: replacing
> `GeomagneticModel.admittance` with admittance ratios from a full 3D
> Monte-Carlo (or a 3D-enabled MCEq) upgrades the accuracy without changing the
> rest of the package. See `src/daemonflux/geomagnetic.py` for the documented
> assumptions.

## Using parameter correlations represented by the covariance matrix

The parameters of the model are correlated. These correlations are drdetermined from the data we have used for the fit. The errors are already computed taking the covariance matrix into account when using the `error` method. If daemonflux is used in a fit with free floating parameters, one can include these correlations by adding the chi2 as additional penalty term. The chi2 for the current combination of parameters can be obtained by calling `flux.chi2({dictionary of modified parameters})`.

## LICENSE

[BSD 3-Clause License](LICENSE)
