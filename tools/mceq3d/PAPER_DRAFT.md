# A fast, data-anchored, directional extension of the daemonflux atmospheric-neutrino flux to low energy

**P. Granger$^{1}$**

$^{1}$*[affiliation]*

*Draft — for expert review. Correspondence: pgranger23@gmail.com*

---

## Abstract

We present a deterministic framework that extends the muon-calibrated, one-dimensional `daemonflux` atmospheric-lepton model into a fully directional, absolute, all-flavour flux $\Phi(E,\cos\theta_z,\varphi)$ covering $0.1$–$100\,\mathrm{GeV}$ — the energy range relevant for atmospheric-neutrino oscillation experiments. The flux is written as a trusted one-dimensional base spectrum multiplied by a cascade-correct geomagnetic suppression factor evaluated with a first-principles, full-IGRF rigidity cutoff obtained by trajectory back-tracing. The base spectrum is selectable between a self-contained MCEq calculation and the data-anchored `daemonflux` flux; the latter restores agreement with the Honda HKKM2014 calculation to $\approx 10\%$ above $1\,\mathrm{GeV}$. Three-dimensional effects beyond the geomagnetic cutoff — the finite production angle of secondaries, the curved-atmosphere geometry of the up-going hemisphere, and the geomagnetic bending of muons before decay — are included and validated, and are shown to be sub-leading ($\lesssim$ few percent) for the conventional flux. Against the Honda tables at Kamioka the model reproduces the absolute $\nu_\mu$ normalisation, the up/down and East–West asymmetries (E–W amplitude $2.4$ versus Honda's $2.1$ at $1\,\mathrm{GeV}$), the $\sec\theta$ horizon enhancement ($2.17$ versus $2.19$ at $100\,\mathrm{GeV}$), and the $\nu_e/\nu_\mu$ flavour ratio to better than $4\%$. The production kinematics are validated against NA61/SHINE $\pi^\pm$ and $K^\pm$ data. The dominant residual uncertainty is the genuine model spread of the sub-GeV flux, which we quantify and deliver as an explicit, energy-dependent systematic band ($\pm 31\%$ at $0.5\,\mathrm{GeV}$, falling to $\pm 4\%$ at $100\,\mathrm{GeV}$). The full geomagnetic cutoff is precomputed and cached, making the first-principles directional flux fast enough for routine use.

---

## 1. Introduction

Atmospheric neutrinos are a cornerstone source for oscillation physics, from the original discovery of oscillations [1] to current and next-generation measurements of the mass ordering and mixing parameters at Super-Kamiokande, IceCube/DeepCore, Hyper-Kamiokande, DUNE, JUNO and KM3NeT/ORCA [1–5]. The interpretation of these data rests on a prediction of the atmospheric-neutrino flux as a function of energy and arrival direction. Below a few GeV — the region that drives the sub-dominant oscillation signatures — the flux prediction must be genuinely three-dimensional and must include geomagnetic effects, and the residual flux uncertainty in this region is among the leading systematics of the measurements.

State-of-the-art flux calculations (Honda *et al.* / HKKM [6,23], Bartol [7], FLUKA [8]) are full three-dimensional Monte-Carlo cascade simulations through the geomagnetic field and a realistic atmosphere. They are authoritative but computationally heavy, are tabulated for a fixed set of sites, and carry the hadronic-model uncertainties of their underlying interaction codes. A complementary approach, `daemonflux` [10], parametrises MCEq [11] cascade calculations with splines and calibrates them against a global set of muon measurements, yielding a fast, data-anchored flux *with a rigorous nuisance-parameter covariance*. By construction, however, `daemonflux` is one-dimensional: the neutrino is taken collinear with the primary, and the flux is azimuthally symmetric and free of any geomagnetic cutoff.

This work bridges the two: it retains the calibrated normalisation and uncertainty model of the 1D approach while adding the directional and geomagnetic structure of the 3D calculations, in a fast, deterministic engine that can be evaluated at an arbitrary site. We deliberately separate the physics into a factorised, trusted-ingredient construction (Section 3), validate each ingredient against data or against the authoritative 3D tables (Section 4), and quantify the irreducible low-energy uncertainty as an explicit systematic (Section 5). Section 6 discusses scope and limitations; Section 7 concludes.

---

## 2. The one-dimensional baseline

The starting point is the inclusive 1D flux $\Phi^{1D}_s(E,\theta_z)$ for each neutrino species $s \in \{\nu_\mu,\bar\nu_\mu,\nu_e,\bar\nu_e\}$. We support two choices. The first is a direct MCEq [11] solution with the SIBYLL-2.3d hadronic interaction model [12] and the Hillas–Gaisser H3a primary-flux parametrisation [14], solved per zenith in the realistic CORSIKA US-Standard layered atmosphere. The second, and our recommended choice for the absolute scale, is the muon-calibrated `daemonflux` flux [10]: it provides the $E^3$-weighted species sums and the $\nu/\bar\nu$ ratios, from which we reconstruct the four species (Section 3.2). The MCEq base is fully self-contained but carries the SIBYLL/H3a normalisation, which we find lies $\approx 25$–$30\%$ below Honda in the sub-GeV–few-GeV band (Section 4.1); the `daemonflux` base removes this deficit by anchoring to muon data.

**Outputs, conventions and uncertainties.** The engine returns the absolute differential flux $\Phi_s$ for the four species in $\mathrm{m^{-2}\,s^{-1}\,sr^{-1}\,GeV^{-1}}$ on a logarithmic energy grid ($\approx 10$ bins per decade) spanning $0.1$–$100\,\mathrm{GeV}$, evaluable at arbitrary $(E,\cos\theta_z,\varphi)$ by trilinear interpolation (log-$E$, azimuth-periodic, $\cos\theta_z$). The same 1D base also carries the inclusive **muon** flux and charge ratio (charge-separated $\mu^\pm$), so a directional muon flux follows by the identical construction; we do not pursue it here. On uncertainties, two scopes should be distinguished: (i) the **model/normalisation** spread, which we quantify and deliver as an explicit systematic (Section 5), and (ii) daemonflux's own **nuisance-parameter covariance**, its principal feature — the latter is *not yet propagated* through the directional factor (we use the central flux). Because the geomagnetic factor is a multiplicative, largely parameter-independent envelope, propagating it is a straightforward extension (apply $G$ to each parameter gradient), noted in Section 6.

---

## 3. Method

### 3.1 Factorised construction

The absolute directional flux is constructed as the 1D base modulated by a geomagnetic suppression factor,

$$
\Phi_s(E,\cos\theta_z,\varphi) \;=\; \Phi^{1D}_s\!\left(E,\,|\cos\theta_z|\right)\; G_s\!\left(E,\,R_c(\cos\theta_z,\varphi)\right),
\tag{1}
$$

where $G_s \in [0,1]$ is the cascade-correct response of species $s$ to the removal of primaries below the directional rigidity cutoff $R_c$. The factorisation is exact in the limit that the geomagnetic cutoff acts only on the primary spectrum (which it does) and that the suppression ratio is independent of the small directional smearing of the secondaries (verified at the $\lesssim 2\%$ level, Section 3.4). The absolute value of $|\cos\theta_z|$ appears in the base because atmospheric production is up/down symmetric; the up/down asymmetry of the *observed* flux arises entirely through the direction dependence of $R_c$ (Section 3.5).

### 3.2 Species reconstruction from the data-anchored base

`daemonflux` reports the $E^3$-weighted flux *sums* and the species ratios. Writing $\Sigma_\mu \equiv E^3(\Phi_{\nu_\mu}+\Phi_{\bar\nu_\mu})$ and $r_\mu \equiv \Phi_{\nu_\mu}/\Phi_{\bar\nu_\mu}$ (and analogously for $e$), the individual species follow from

$$
\Phi_{\nu_\mu} = \frac{\Sigma_\mu}{E^3}\,\frac{r_\mu}{1+r_\mu},\qquad
\Phi_{\bar\nu_\mu} = \frac{\Sigma_\mu}{E^3}\,\frac{1}{1+r_\mu},
\tag{2}
$$

with the cm$^{-2}\to$m$^{-2}$ unit conversion applied. The reconstruction is valid over the calibrated range; below $\sim 0.3\,\mathrm{GeV}$ the splines extrapolate beyond the muon data, an effect we treat as a systematic (Section 5).

### 3.3 Geomagnetic cutoff by trajectory back-tracing

The directional rigidity cutoff $R_c(\cos\theta_z,\varphi)$ is computed from first principles rather than from the analytic Størmer formula [17]. A cosmic ray of rigidity $R = pc/Ze$ moving with unit tangent $\mathbf{u}=d\mathbf{r}/ds$ obeys

$$
\frac{d\mathbf{u}}{ds} = \frac{c}{R}\,\big(\mathbf{u}\times\mathbf{B}(\mathbf{r})\big),
\tag{3}
$$

i.e. a gyration of radius $r_g = R/(cB_\perp)$. To test whether a positive primary of rigidity $R$ can reach the detector from a direction $\mathbf{d}$, we integrate the *reversed* trajectory (reversed velocity and charge) outward from the detector with a fourth-order Runge–Kutta scheme in arc length: if it escapes to large radius the rigidity is **allowed**; if it returns to the Earth it is **forbidden** [16]. Scanning $R$ from high to low locates the effective cutoff at the allowed/forbidden transition. The field $\mathbf{B}$ is the full IGRF-13 model [15] (degree 13) near the Earth, smoothly matched to the tilted-dipole term beyond a few Earth radii where the higher multipoles are negligible.

As a closure test, replacing the field by a centred, aligned dipole reproduces the Størmer cutoff: $14.9\,\mathrm{GV}$ vertical at the geomagnetic equator, the $\cos^4(\lambda)$ latitude dependence, and the correct sign of the East–West asymmetry (positive primaries arrive preferentially from the West). For the full IGRF field at Kamioka we obtain a vertical cutoff of $11.31\,\mathrm{GV}$, consistent with the literature value ($\approx 11.3\,\mathrm{GV}$). Figure 1 shows the resulting cutoff sky map.

Because back-tracing each direction is expensive, the full $(\theta_z,\varphi)$ cutoff map is precomputed once per site and cached; subsequent evaluations are bilinear interpolations, making the first-principles cutoff fast enough for routine use. The analytic Størmer cutoff remains available as a dependency-free fallback.

![**Figure 1.** Back-traced full-IGRF rigidity-cutoff sky map at Kamioka.](geomag_cutoff_map.png)

### 3.4 Cascade-correct suppression and per-nucleus rigidity

The geomagnetic cut acts on the primary nucleons inside the developing shower, not on the neutrino energy. We therefore define the suppression factor as the ratio of two full cascade solutions,

$$
G_s(E,R_c) \;=\; \frac{\Phi_s\big[\text{primaries cut at } R_c\big]}{\Phi_s\big[\text{all primaries}\big]},
\tag{4}
$$

which automatically folds in the spread of primary energies feeding a given neutrino energy and avoids the common approximation of applying an effective-rigidity factor directly to the secondary spectrum.

The cut is on rigidity, $R = (A/Z)\,E_{\rm nucleon}$, so free protons ($A/Z=1$) and nucleons bound in heavier nuclei ($A/Z\simeq 2$) are removed at different nucleon energies. We split the primary nucleon flux accordingly, using MCEq's own proton and neutron spectra: the free-proton fraction is the proton–neutron excess,

$$
f_{\rm free}(E) = \frac{\Phi_p(E)-\Phi_n(E)}{\Phi_p(E)},
\tag{5}
$$

and the bound nucleons carry a composition-weighted $\langle A/Z\rangle$ from the H3a abundances ($\approx 2.005$, rising slowly with energy as the iron fraction grows). The per-nucleon transmission combines the two channels,

$$
T(E;R_c) = f_{\rm free}\,T_1(E;R_c) + (1-f_{\rm free})\,T_{\langle A/Z\rangle}(E;R_c),
\tag{6}
$$

with a smooth rigidity step (penumbra width $\sigma$ in $\ln R$),

$$
T_a(E;R_c) = \tfrac12\left[1 + \operatorname{erf}\!\left(\frac{\ln(aE)-\ln R_c}{\sqrt{2}\,\sigma}\right)\right],\qquad a \equiv A/Z .
\tag{7}
$$

Neglecting the free/bound split over-suppresses the bound component near the sub-GeV horizon.

A potential concern is that $G_s$ is precomputed on the vertical column, whereas at large zenith the slant atmosphere is much thicker and the shower develops differently (more complete meson decay, more muon energy loss). We tested this directly by recomputing Eq. (4) in MCEq from vertical to $\cos\theta_z = 0.1$ ($\approx 84^\circ$): $G_s$ is zenith-independent to $\le 2.1\%$ at the worst point ($0.3\,\mathrm{GeV}$, $R_c=11\,\mathrm{GV}$) and to $\le 0.4\%$ above $1\,\mathrm{GeV}$ (Figure 2). The slant-depth effects act on numerator and denominator of Eq. (4) alike and largely cancel. An option to recompute $G_s$ per zenith is provided for studies that wish to remove even this residual.

![**Figure 2.** The suppression ratio $G_s(E,R_c)$ is zenith-independent to $\le 2\%$: the slant-depth shower effects cancel in the cut/full ratio.](geomag_zenith_check.png)

### 3.5 The up-going hemisphere

For up-going directions the neutrino is produced on the far side of the Earth and traverses the planet essentially undeflected, so atmospheric production is identical to the down-going case at $|\cos\theta_z|$ (hence the absolute value in Eq. 1), but the geomagnetic cutoff must be evaluated at the **far-side production point**, not at the detector. Given the detector position and arrival direction we compute the line-of-sight intersection with the production shell at altitude $h\simeq 20\,\mathrm{km}$ on the opposite limb, and evaluate $R_c$ there with the primary direction set equal to the (straight-line) neutrino direction. This makes the up-going flux a genuine global-geomagnetic prediction. The leading approximation is the use of a single representative production point per direction; the finite extent of the production region is a sub-leading geometric correction.

### 3.6 Production angle and angular transport

The finite production angle that 1D MCEq integrates away is recovered by regenerating the relevant hadronic kernels with event generators (UrQMD-3.4 [13] in the low-energy regime, SIBYLL-2.3d [12] above its threshold, via the `chromo` interface [20]) and propagating the transverse-momentum distribution through the decay chain. The production angle of a secondary is

$$
\theta_{\rm prod} = \arctan\!\left(\frac{p_T}{p_\parallel}\right),
\tag{8}
$$

and the cumulative angular spread over the production-and-decay chain is treated both with a small-angle Fokker–Planck operator, $\sigma_\theta^2 = N\,\theta_1^2$ for $N$ kick-giving generations, and with a full multipole ($P_N$) transport on the sphere. The two agree where both are valid, and a variance-only treatment is shown to be sufficient for the conventional flux. The net effect of the production angle on the conventional flux is small — at the $1$–$2\%$ level at multi-GeV energies, where $\theta_{\rm prod} \lesssim 0.5^\circ$ — so it is a correction rather than a leading term, but it is included and validated against data (Section 4.2) rather than assumed.

![**Figure 3.** Production-and-decay angular spread of the conventional flux versus neutrino energy, from the regenerated kernels. The Fokker–Planck (variance) and full $P_N$ transports agree, and the spread falls below a degree above a few GeV, confirming that the production-angle 3D effect on the conventional flux is a $\sim 1$–$2\%$ correction.](fokker_planck_3d.png)

### 3.7 Muon bending

Muons produced in the cascade bend in the geomagnetic field before they decay, so the decay neutrinos inherit a deflected direction. Over the in-flight decay the bending angle is

$$
\Delta\phi = \omega_c\,t_{\rm lab} = \frac{qB}{\gamma m_\mu}\cdot \gamma\tau_\mu = \frac{qB\,\tau_\mu}{m_\mu},
\tag{9}
$$

i.e. **energy-independent** (the longer flight of an energetic muon is exactly offset by its larger gyroradius), and numerically $\approx 3^\circ$ for the local field. What depends on energy is whether the muon decays in flight at all, governed by the decay-in-flight fraction

$$
f_{\rm dec}(E_\mu,\theta_z) = 1 - \exp\!\left(-\,\frac{L(\theta_z)}{\gamma\beta c\tau_\mu}\right),
\tag{10}
$$

with $L(\theta_z)$ the curved-atmosphere slant path from the production altitude. We weight the bending by $f_{\rm dec}/(1+f_{\rm dec})$ and restrict it to the muon-decay $\nu_\mu$ channel so it does not contaminate the direct $\nu_\mu$. The coherent, charge-dependent part is computed with the full local field vector (IGRF-13) and the actual muon direction: $\mu^+$ and $\mu^-$ deflect oppositely, giving a $\approx 3^\circ$ charge-separated East–West split and a small ($\approx 0.3^\circ$) net shift for the summed flux given the near-unity muon charge ratio (Figure 4).

![**Figure 4.** Muon-bending angular spread and coherent East–West shift, restricted to the muon-decay channel.](muon_bending.png)

---

## 4. Validation

### 4.1 Absolute and directional flux versus Honda HKKM2014

We compare the absolute $\nu_\mu$ flux at Kamioka against the Honda HKKM2014 tables [6] on a dense energy grid from $0.1$ to $100\,\mathrm{GeV}$. Both predictions are log–log interpolated to common energies; a naïve nearest-grid-point comparison mismatches MCEq's $0.89\,\mathrm{GeV}$ node against Honda's $1.0\,\mathrm{GeV}$ bin and is misleading on a steep spectrum.

The comparison (Table 1, Figure 5) reveals the role of the 1D base. The raw-MCEq base lies $\approx 25$–$30\%$ below Honda across $0.3$–$10\,\mathrm{GeV}$, a genuine SIBYLL/H3a normalisation deficit. The `daemonflux`-anchored base agrees with Honda to $\approx 10\%$ for $E\gtrsim 1\,\mathrm{GeV}$ up to $100\,\mathrm{GeV}$, but over-predicts toward $0.1\,\mathrm{GeV}$ (up to $\sim\!2\times$), where it extrapolates beyond its muon calibration. The two bases **bracket Honda below $\sim 1\,\mathrm{GeV}$** (daemonflux above, MCEq below); above $\sim 1\,\mathrm{GeV}$ both lie below Honda, with the data-anchored daemonflux base the closer of the two. We therefore adopt the daemonflux base as the central estimate, and use the inter-base spread to define the systematic (Section 5).

**Table 1.** Ratio of the absolute vertical $\nu_\mu$ flux at Kamioka to Honda HKKM2014.

| $E$ (GeV) | MCEq base / Honda | `daemonflux` base / Honda |
|---|---|---|
| 0.10 | 0.94 | 1.98 |
| 0.30 | 0.72 | 1.34 |
| 0.50 | 0.70 | 1.30 |
| 1.0 | 0.73 | 1.11 |
| 3.0 | 0.73 | 0.96 |
| 10–100 | 0.74–0.85 | 0.91 |

![**Figure 5.** Absolute $\nu_\mu$ spectrum and zenith dependence versus Honda HKKM2014 (Kamioka).](mceq3d_flux.png)

The directional observables are ratios and are therefore independent of the absolute base. The engine recovers the up/down asymmetry (up-going exceeds down-going at fixed energy, as in Honda), the East–West amplitude at Kamioka ($2.4$ here versus $2.1$ in Honda at $1\,\mathrm{GeV}$, with the correct sub-GeV peak and high-energy vanishing), and the $\sec\theta$ horizon enhancement ($2.17$ versus $2.19$ at $100\,\mathrm{GeV}$) (Figure 6). The $\nu_e/\nu_\mu$ flavour ratio — the sharpest, most model-independent test, set by the $\pi\to\mu\to e$ chain — agrees with Honda to $0.7$–$4\%$ across the band.

![**Figure 6.** East–West and $\sec\theta$ directional cross-checks against the Honda 3D tables.](validate_honda.png)

### 4.2 Hadronic production versus NA61/SHINE

The production kinematics that set the angular content are validated against fixed-target data, since MCEq's 1D kernels carry no angular information. We compare the mean transverse momentum $\langle p_T\rangle(p)$ of charged pions and kaons in $p+\mathrm{C}$ at $31\,\mathrm{GeV}/c$ against the NA61/SHINE measurements [18,19] (HEPData [22] records ins886780 and ins1397003, fetched via `hepdata-cli`), applying the experiment's angular acceptance. For pions the agreement is $\approx 10\%$ (Figure 7); for kaons UrQMD reproduces the $\langle p_T\rangle$ scale ($0.2$–$0.6\,\mathrm{GeV}$) and its rise with momentum, running $10$–$20\%$ harder than the data (Figure 8). Both map to a small over-estimate of the production angle in the relevant decay channel, sub-percent on the directional flux at the multi-GeV energies where kaons dominate.

![**Figure 7.** Pion $\langle p_T\rangle(p)$ versus NA61/SHINE.](validate_na61_pt.png)

![**Figure 8.** Kaon $\langle p_T\rangle(p)$ versus NA61/SHINE.](validate_na61_kaon_pt.png)

### 4.3 Behaviour across magnetic environments

As a stability check we evaluate the central flux and its systematic from a high-cutoff equatorial site to the polar limit (Figure 9). The vertical cutoff falls monotonically and smoothly ($17.2 \to 9.0 \to 1.8 \to 0.8\,\mathrm{GV}$ from equator to pole), the flux rises correspondingly with no discontinuity at the no-cutoff polar limit, and the fractional systematic is site-robust (the cutoff cancels in the base ratio). The composition-weighted $\langle A/Z\rangle$ is site-independent by construction.

![**Figure 9.** Central flux and systematic band across magnetic environments.](latitude_check.png)

---

## 5. Systematic uncertainties

The dominant uncertainty below a few GeV is physical, not numerical: independent, credible flux models disagree at the $20$–$40\%$ level there [6,7,8,24], and no method choice removes this. We quantify it from the spread of our two bases. Denoting the MCEq- and `daemonflux`-based predictions $\Phi_{\rm mc}$ and $\Phi_{\rm df}$, the half log-spread defines a one-sigma model systematic, with the data-anchored daemonflux base as the central value (and the geometric mean as a convenience central where the bases bracket Honda),

$$
\sigma_{\rm sys}(E) = \tfrac12\left|\ln\!\frac{\Phi_{\rm df}}{\Phi_{\rm mc}}\right| ,\qquad
\Phi_{\rm central} \equiv \Phi_{\rm df}\ (E\gtrsim 0.5\,\mathrm{GeV}),\quad
\sqrt{\Phi_{\rm mc}\Phi_{\rm df}}\ (\mathrm{sub\text{-}GeV}) .
\tag{11}
$$

The band is $\pm 37\%$ at $0.1\,\mathrm{GeV}$, $\pm 31\%$ at $0.5\,\mathrm{GeV}$, $\pm 21\%$ at $1\,\mathrm{GeV}$, $\pm 10\%$ at $10\,\mathrm{GeV}$ and $\pm 4\%$ at $100\,\mathrm{GeV}$ (Figure 10), to be propagated through an oscillation fit as a correlated normalisation-versus-energy systematic. Two honest caveats apply. First, the bases bracket Honda only **below $\sim 1\,\mathrm{GeV}$** (daemonflux above, MCEq below), where the geometric-mean central reproduces Honda to $\approx 5\%$; **above $\sim 1\,\mathrm{GeV}$ both bases lie below Honda** (daemonflux at $\approx 0.91$, the better central; the geometric mean is biased $\approx 10$–$18\%$ low and should not be used there). Second, being a two-model spread it is an *intra-framework* systematic — a floor on the flux uncertainty — and does not by itself span other calculations (Honda lies $\approx 10$–$18\%$ above the geometric mean at multi-GeV); a complete analysis should fold in the inter-calculation (Honda/Bartol/FLUKA) envelope. Sub-dominant, well-bounded effects include the production-angle treatment ($\sim 1$–$2\%$), the zenith-independence of $G_s$ ($\le 2\%$, Section 3.4), the composition-averaged bound $\langle A/Z\rangle$, and the muon-bending kinematic constants, which matter only for fine-grained charge-resolved studies.

![**Figure 10.** The two 1D bases versus Honda, with their geometric-mean central and log-spread systematic band over $0.1$–$100\,\mathrm{GeV}$. The bases bracket Honda only below $\sim 1\,\mathrm{GeV}$; above, both lie below Honda (daemonflux closer).](base_comparison.png)

**Decomposing the spread: interaction model versus calibration.** To separate the hadronic-interaction contribution from the overall normalisation, we run the same base setup (H3a primary, US-Standard atmosphere) through the four hadronic models MCEq carries — SIBYLL-2.3d, EPOS-LHC, DPMJET-III-19.3, QGSJET-II-04 — a proxy for the inter-calculation (Bartol/FLUKA) spread whose dominant driver is the hadronic model. The **interaction-model spread is only $\approx 4$–$7\%$ sub-GeV, rising to $\approx 12$–$13\%$ at multi-GeV** (Figure 11). This is much smaller than the daemonflux$\leftrightarrow$MCEq spread below a few GeV, which establishes an important point: the **sub-GeV flux uncertainty is dominated by the overall normalisation/muon-calibration, not by the interaction model** — precisely why anchoring to muon data matters. At multi-GeV the two contributions become comparable ($\sim 10\%$ each). We caution that this MCEq-internal spread is *not* a substitute for a literal Bartol/FLUKA comparison — those codes differ also in primary flux, atmosphere and 3D treatment — but it bounds the hadronic-model part; a direct table comparison is left for when those tables are ingested through the same validation path.

![**Figure 11.** Hadronic interaction-model spread of the vertical $\nu_\mu$ base (four MCEq models, same primary/atmosphere): $E^3\Phi$ (left) and ratio to the geometric mean with the inter-model envelope (right). The spread is $\approx 4$–$7\%$ sub-GeV, rising to $\approx 12$–$13\%$ at multi-GeV — smaller than the calibration-driven spread of Figure 10 below a few GeV.](hadronic_spread.png)

---

## 6. Discussion and limitations

We are explicit about the trust boundary. The absolute directional engine described in Section 3 — a calibrated 1D base modulated by the cascade-correct geomagnetic factor with a back-traced cutoff — is the component we regard as analysis-grade: it is validated absolutely against Honda and against NA61 production data. The associated deterministic-3D machinery developed alongside it (a coupled angular cascade, a spherical-streaming transport operator, parametrised angular engines) is research-grade: it establishes the architecture and bounds the size of the genuinely-3D corrections, but its absolute normalisation is meaningful only where backed by the MCEq/`daemonflux` base.

Known approximations, each bounded above, are: (i) the up-going hemisphere uses a single representative far-side production point per direction; (ii) the high-statistics SIBYLL kernels degrade near the generator's soft-pion centre-of-mass threshold (irrelevant to the angular moments, but not usable for standalone absolute yields); (iii) meson re-interactions are neglected (a sub-GeV-negligible effect, since low-energy mesons decay before re-interacting); and (iv) a few muon-bending kinematic constants ($E_\mu\approx 3E_\nu$, the muon charge ratio) are fixed rather than derived. The largest single uncertainty remains the physical sub-GeV model spread of Section 5, which is delivered as a systematic rather than hidden.

**Performance.** On a single core, one MCEq cascade solve takes $\approx 1.5\,\mathrm{s}$. A full directional solve over a $6\times 8 = 48$-direction sky grid takes $\approx 6\,\mathrm{min}$ — roughly $240\times$ one MCEq solve, or $\approx 14\times$ the equivalent 1D MCEq flux at the same zeniths. The cost is dominated ($\sim 85\%$) by the geomagnetic-cutoff trajectory back-tracing ($\sim 5\,\mathrm{s}$ per direction); the 1D base contributes $\approx 26\,\mathrm{s}$ (MCEq base) or $\approx 10\,\mathrm{ms}$ (daemonflux spline base), and the suppression response $G_s$ adds $\approx 21\,\mathrm{s}$. Both heavy ingredients are reusable and are memoised to disk (`solve(use_cache=True)`): the cutoff map is a one-time per-site precompute and $G_s$ is site- and zenith-independent, so a warm re-evaluation loads from disk in milliseconds (measured: a warm solve is $\approx 5\,\mathrm{ms}$ versus $\approx 60\,\mathrm{s}$ cold, a $>10^3$ speed-up, reproducing the direct result to $\approx 0.1\%$). The framework is therefore a one-time per-site precompute followed by near-instant evaluation, in contrast to the CPU-weeks of a full 3D Monte-Carlo.

Two capabilities are deliberately out of scope. First, the full daemonflux **nuisance-parameter covariance** is not yet propagated through the directional factor (Section 2); folding it in is the most valuable next step, since it would give the directional flux the same rigorous, correlated error model that distinguishes the 1D model. Second, the **correlated muon** accompanying a down-going neutrino — the atmospheric self-veto passing fraction relevant to neutrino telescopes — is an event-level quantity that an inclusive flux model (this work, MCEq, or daemonflux) cannot provide; it requires a dedicated calculation such as nuVeto [25].

The seasonal and site dependence of the atmosphere is available through the NRLMSISE-00 profiles exposed by MCEq, but has not been studied here. Anchoring the sub-$0.3\,\mathrm{GeV}$ region — where the present central value relies on bracketing two extrapolating models — to a dedicated low-energy dataset is the other high-value extension.

---

## 7. Conclusions

We have presented a fast, deterministic, data-anchored framework that extends the 1D `daemonflux` model into a fully directional, absolute, all-flavour atmospheric-neutrino flux over $0.1$–$100\,\mathrm{GeV}$. Each ingredient is either taken from a trusted source or validated against data: the absolute scale is anchored to muon-calibrated splines and agrees with Honda to $\approx 10\%$ above $1\,\mathrm{GeV}$; the geomagnetic cutoff is a first-principles full-IGRF back-trace, cached for speed; the directional observables match the authoritative Honda 3D tables; and the production kinematics match NA61/SHINE. The irreducible sub-GeV uncertainty is quantified as an explicit systematic band. The framework is suitable for oscillation analyses that require a directional flux at an arbitrary site with a transparent, propagatable error model, and provides a fast complement to full 3D Monte-Carlo calculations.

We particularly invite expert feedback on the choice and sub-GeV extrapolation of the 1D base; the cascade-correct definition of the geomagnetic suppression and the per-nucleus rigidity split; the far-side treatment of the up-going hemisphere; and whether the model-spread systematic is an appropriate way to present the irreducible low-energy uncertainty to an oscillation analysis.

---

## Acknowledgements

This work builds directly on the MCEq, `daemonflux`, `crflux`, `chromo` and `ppigrf` software, on the Honda HKKM2014 flux tables, and on the NA61/SHINE measurements made available through HEPData.

---

## References

[1] Y. Fukuda *et al.* (Super-Kamiokande Collaboration), *Evidence for oscillation of atmospheric neutrinos*, Phys. Rev. Lett. **81**, 1562 (1998).

[2] M. G. Aartsen *et al.* (IceCube Collaboration), *Measurement of atmospheric neutrino oscillations with three years of IceCube DeepCore data*, Phys. Rev. Lett. **120**, 071801 (2018).

[3] K. Abe *et al.* (Hyper-Kamiokande Collaboration), *Hyper-Kamiokande Design Report*, arXiv:1805.04163 (2018).

[4] B. Abi *et al.* (DUNE Collaboration), *Deep Underground Neutrino Experiment (DUNE) — Volume II: DUNE Physics*, arXiv:2002.03005 (2020).

[5] F. An *et al.* (JUNO Collaboration), *Neutrino physics with JUNO*, J. Phys. G **43**, 030401 (2016).

[6] M. Honda, M. Sajjad Athar, T. Kajita, K. Kasahara, S. Midorikawa, *Atmospheric neutrino flux calculation using the NRLMSISE-00 atmospheric model*, Phys. Rev. D **92**, 023004 (2015).

[7] G. D. Barr, T. K. Gaisser, P. Lipari, S. Robbins, T. Stanev, *Three-dimensional calculation of atmospheric neutrinos*, Phys. Rev. D **70**, 023006 (2004).

[8] G. Battistoni, A. Ferrari, T. Montaruli, P. R. Sala, *The atmospheric neutrino flux below 100 MeV: the FLUKA results*, Astropart. Phys. **23**, 526 (2005); G. Battistoni *et al.*, Astropart. Phys. **19**, 269 (2003).

[9] P. Lipari, *The geometry of atmospheric neutrino production*, Astropart. Phys. **14**, 153 (2000).

[10] J. P. Yáñez and A. Fedynitch, *daemonflux: a data-driven, muon-calibrated atmospheric lepton flux model*, arXiv:2303.00022 (2023).

[11] A. Fedynitch, R. Engel, T. K. Gaisser, F. Riehn, T. Stanev, *Calculation of conventional and prompt lepton fluxes at very high energy*, EPJ Web Conf. **99**, 08001 (2015); A. Fedynitch *et al.*, Phys. Rev. D **100**, 103018 (2019).

[12] F. Riehn, R. Engel, A. Fedynitch, T. K. Gaisser, T. Stanev, *Hadronic interaction model SIBYLL 2.3d and extensive air showers*, Phys. Rev. D **102**, 063002 (2020).

[13] S. A. Bass *et al.*, *Microscopic models for ultrarelativistic heavy ion collisions*, Prog. Part. Nucl. Phys. **41**, 255 (1998); M. Bleicher *et al.*, J. Phys. G **25**, 1859 (1999).

[14] T. K. Gaisser, *Spectrum of cosmic-ray nucleons, kaon production, and the atmospheric muon charge ratio*, Astropart. Phys. **35**, 801 (2012).

[15] P. Alken *et al.*, *International Geomagnetic Reference Field: the thirteenth generation*, Earth, Planets and Space **73**, 49 (2021).

[16] D. F. Smart, M. A. Shea, E. O. Flückiger, *Magnetospheric models and trajectory computations*, Space Sci. Rev. **93**, 305 (2000).

[17] C. Störmer, *The Polar Aurora* (Oxford University Press, 1955).

[18] N. Abgrall *et al.* (NA61/SHINE Collaboration), *Measurements of cross sections and charged pion spectra in proton–carbon interactions at 31 GeV/c*, Phys. Rev. C **84**, 034604 (2011).

[19] N. Abgrall *et al.* (NA61/SHINE Collaboration), *Measurements of $\pi^\pm$, $K^\pm$, $K^0_S$, $\Lambda$ and proton production in proton–carbon interactions at 31 GeV/c*, Eur. Phys. J. C **76**, 84 (2016).

[20] A. Fedynitch *et al.*, *chromo — Cosmic ray and HadROnic interactiOn MOnte-carlo frontend*, software, https://github.com/impy-project/chromo .

[21] *ppigrf — pure-Python IGRF*, software, https://github.com/IAGA-VMOD/ppigrf .

[22] E. Maguire, L. Heinrich, G. Watt, *HEPData: a repository for high energy physics data*, J. Phys. Conf. Ser. **898**, 102006 (2017).

[23] M. Honda, T. Kajita, K. Kasahara, S. Midorikawa, T. Sanuki, *Calculation of atmospheric neutrino flux using the interaction model calibrated with atmospheric muon data*, Phys. Rev. D **75**, 043006 (2007).

[24] G. D. Barr, T. K. Gaisser, S. Robbins, T. Stanev, *Uncertainties in atmospheric neutrino fluxes*, Phys. Rev. D **74**, 094009 (2006).

[25] C. A. Argüelles, S. Palomares-Ruiz, A. Schneider, L. Wille, T. Yuan, *Unified atmospheric neutrino passing fractions for large-scale neutrino telescopes*, JCAP **07**, 047 (2018).

---

## Appendix A: numerical methods

The cascade solutions use MCEq with its default 1D matrix solver on a logarithmic energy grid of $\approx 10$ bins per decade from $0.1\,\mathrm{GeV}$. The trajectory back-tracing (Eq. 3) integrates with a fixed-step RK4 in arc length; the escape radius and step length are chosen so that the recovered Størmer cutoff is stable to better than $1\%$. The cutoff sky map is precomputed on a $(\theta_z,\varphi)$ grid and cached as a compressed array; production evaluations bilinearly interpolate it. The systematic band (Eq. 11) is evaluated pointwise in energy from the two base predictions.

## Appendix B: code and data availability

All code, unit tests, and figure-generating scripts are available on the fork `github.com/pgranger23/daemonflux`, branch `3d-extension`, under `tools/mceq3d/`. The absolute engine is `mceq3d_flux.py`; the geomagnetic back-tracer and cached cutoff are in `geomag_backtrace.py`; the validation scripts are `validate_honda.py`, `validate_na61.py`, `validate_na61_kaon.py`, `base_comparison.py`, `hadronic_spread.py`, `geomag_zenith_check.py` and `latitude_check.py`; performance and caching are exercised by `profile_3d.py` and `verify_cache.py`. A detailed development report with the complete module inventory and an exhaustive list of approximations is provided in `IMPLEMENTATION_REPORT.md`.
