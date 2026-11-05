"""PM check (2026-11-05): Honda HKKM2014 Kamioka solmin -> solmax shift of the 12 pooled cells."""
import gzip, os, re, sys
import numpy as np
R = "/afs/cern.ch/work/p/pigrange/daemonflux"
sys.path.insert(0, R + "/tools/mc3d"); sys.path.insert(0, R + "/tools/mceq3d")
import analyse_m3b as A
H = "/home/pigrange/mc3d_work/runs/m3f/honda"

def parse(gz, out):
    raw = gzip.decompress(open(gz, "rb").read()).decode()
    blocks, cz, az, rows = {}, None, None, []
    for ln in raw.splitlines():
        m = re.match(r"average flux in \[cosZ =\s*([\-0-9.]+) --.*phi_Az =\s*([0-9]+) --", ln)
        if m:
            if cz is not None and rows:
                blocks[(cz, az)] = np.array(rows)
            cz, az, rows = float(m.group(1)), int(m.group(2)), []
            continue
        p = ln.split()
        if len(p) == 5 and p[0][0].isdigit():
            rows.append([float(x) for x in p])
    blocks[(cz, az)] = np.array(rows)
    czs = sorted({k[0] for k in blocks}); azs = sorted({k[1] for k in blocks})
    E = blocks[(czs[0], azs[0])][:, 0]
    d = dict(E=E, czlo=np.array(czs), azlo=np.array(azs))
    for j, k in enumerate(("numu", "numubar", "nue", "nuebar")):
        d[k] = np.array([[blocks[(c, a)][:, 1 + j] for a in azs] for c in czs])
    np.savez(out, **d)
    return d

smax = parse(H + "/kam-ally-20-12-solmax.d.gz", H + "/honda_kam_solmax.npz")
smin = dict(np.load(R + "/tools/mceq3d/honda_kam.npz"))
assert np.allclose(smax["E"], smin["E"]) and np.allclose(smax["czlo"], smin["czlo"])
ax = A.ew_axis()
omin = A.table_obs(A.honda_patterns(R + "/tools/mceq3d/honda_kam.npz"), ax)
omax = A.table_obs(A.honda_patterns(H + "/honda_kam_solmax.npz"), ax)
pmin = A.honda_patterns(R + "/tools/mceq3d/honda_kam.npz"); pmax = A.honda_patterns(H + "/honda_kam_solmax.npz")
print("obs:", A.OBS)
for i, o in enumerate(A.OBS):
    print(f"{o:8s} solmin mean {np.nanmean(omin[..., i]):+.4f}  solmax {np.nanmean(omax[..., i]):+.4f}  "
          f"shift {np.nanmean(omax[..., i] - omin[..., i]):+.4f}")
print("per species a1/a0 shift (mean over E band, cz):",
      [f"{np.mean(omax[k,...,0]-omin[k,...,0]):+.4f}" for k in range(4)], A.SP_NAMES)
print("per E band a1/a0 shift:", [f"{np.mean(omax[:,ib,:,0]-omin[:,ib,:,0]):+.4f}" for ib in range(3)], list(A.E_BANDS))
print("flux ratio solmax/solmin (az-mean, all species) per E band:",
      [f"{pmax[:,ib].sum()/pmin[:,ib].sum():.4f}" for ib in range(3)])
# vertical / all-sky flux ratio vs E from the raw tables (numu)
for e in (0.3, 0.5, 1.0, 3.0, 10.0):
    i = int(np.argmin(abs(smin["E"] - e)))
    up = smin["czlo"] >= 0.9
    print(f"E {smin['E'][i]:.3g}: numu solmax/solmin vertical {smax['numu'][up,:,i].mean()/smin['numu'][up,:,i].mean():.4f}, "
          f"horizon(0-0.3) {smax['numu'][(smin['czlo']>=0)&(smin['czlo']<0.3),:,i].mean()/smin['numu'][(smin['czlo']>=0)&(smin['czlo']<0.3),:,i].mean():.4f}")
amin = np.nanmean(omin[..., 0]); amax = np.nanmean(omax[..., 0])
print(f"Honda d<a1/a0> solmin->solmax = {amax-amin:+.4f}; at our slope -0.054/GV this is dphi = {(amax-amin)/-0.054:+.2f} GV")

# ---- pooled cos Z 0-0.3 cells, as analyse_m3e (sum over cz bands), vs pilot-2 ----
IDX = (0, 1, 5, 6, 7); NAMES = ("a1/a0", "dphi", "fN", "fS", "N/S")
def pooled(p):
    p = p.sum(2)
    return np.array([[A.observables(p[k, ib], ax)[list(IDX)] for ib in range(3)] for k in range(4)])
hmin = pooled(pmin); hmax = pooled(pmax)
t = np.load("/home/pigrange/mc3d_work/runs/m3e/reports/m3e_table.npz")
est, err = t["est"], t["err"]
print("\nsanity: saved Honda (solmin) vs recomputed, max |diff|:", np.nanmax(abs(t["honda"] - hmin)))
for lab, h in (("solmin", hmin), ("solmax", hmax)):
    d = est - h; d[..., 1] = (d[..., 1] + 180) % 360 - 180
    z = d / np.where(err > 0, err, np.nan)
    print(f"pilot-2 vs Honda {lab}: <a1/a0> MC {est[...,0].mean():.4f} Honda {h[...,0].mean():.4f} diff {d[...,0].mean():+.4f}; chi2/12 "
          + "  ".join(f"{nm} {np.nansum(z[..., j]**2)/12:.2f}" for j, nm in enumerate(NAMES)))
    print("   fan-out antinumu-numu Honda", " ".join(f"{x:+.3f}" for x in h[3, :, 0] - h[2, :, 0]),
          " nue-antinue", " ".join(f"{x:+.3f}" for x in h[0, :, 0] - h[1, :, 0]))
    print("   per species a1/a0 (mean of 3 bands): MC", " ".join(f"{est[k,:,0].mean():.3f}" for k in range(4)),
          " Honda", " ".join(f"{h[k,:,0].mean():.3f}" for k in range(4)))
