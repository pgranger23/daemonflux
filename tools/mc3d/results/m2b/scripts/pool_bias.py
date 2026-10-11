import numpy as np
R = "/afs/cern.ch/work/p/pigrange/daemonflux/tools/mceq3d/"
for f in ("m_spliced_v2.npz", "m_piminus_v2.npz"):
    d = dict(np.load(R + f))
    pe, xe = d["proj_energies"], d["xl_edges"]
    dx = np.diff(xe)
    es, t2, w = d["e_sec"], d["theta_sq"], d["dndx"]
    print(f"{f}: projectile rows {pe.min():.1f} .. {pe.max():.1f} GeV ({len(pe)} rows)")
    for lo, hi in ((0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)):
        m = (es >= lo) & (es < hi) & np.isfinite(t2) & (w > 0)
        a = np.degrees(np.sqrt(np.sum(t2[m] * w[m]) / np.sum(w[m])))      # engine
        n = w * dx[None, :]                                                # pions/cell
        b = np.degrees(np.sqrt(np.sum((t2 * n)[m]) / np.sum(n[m])))       # + dx
        sw = (pe ** -1.7)[:, None] * np.ones_like(w)                       # interactions per lnE
        c = np.degrees(np.sqrt(np.sum((t2 * n * sw)[m]) / np.sum((n * sw)[m])))
        rows = np.unique(np.where(m)[0])
        top = pe[rows][np.argsort(-(w[m]).reshape(-1)[:0].sum() if False else 0)] if False else None
        share = np.array([w[i][m[i]].sum() for i in range(len(pe))]) / w[m].sum()
        print(f"  E_pi [{lo},{hi}): engine pooling (dndx) {a:6.2f} deg | dndx*dx {b:6.2f} | "
              f"dndx*dx*E_proj^-1.7 {c:6.2f}  -> ratio {c/a:5.3f};  engine weight share of rows > 1 TeV: "
              f"{share[pe > 1000].sum():.2f}")
