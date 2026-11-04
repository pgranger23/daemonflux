"""One-line-per-model summary of ``ew_models.npz`` (mean over the 12 pooled
cells of each observable, chi2/12 vs Honda per observable, mean paired shift
of a1/a0 vs the first entry) as a markdown table."""
import os, sys
import numpy as np
M = os.environ["M"]
z = np.load(sys.argv[1] if len(sys.argv) > 1 else f"{M}/reports/ew_models.npz")
labs = [k[:-4] for k in z.files if k.endswith("_est")]
hon = z["honda"]
NAMES = ("a1/a0", "dphi", "fN", "fS", "N/S")
def wrap(d):
    d = d.copy(); d[..., 1] = (d[..., 1] + 180) % 360 - 180; return d
L = ["| model | <a1/a0> | <dphi> [deg] | <fN> | <fS> | <N/S> | chi2/12: a1/a0 | dphi | fN | fS | N/S |", "|" + "---|" * 11]
L.append("| Honda | %.4f | %.2f | %.3f | %.3f | %.3f | | | | | |" % tuple([hon[..., 0].mean(), hon[..., 1].mean()] + [hon[..., j].mean() for j in (2, 3, 4)]))
for lab in labs:
    e, s = z[lab + "_est"], z[lab + "_err"]
    zz = wrap(e - hon) / np.where(s > 0, s, np.nan)
    chi = [np.nansum(zz[..., j] ** 2) / 12 for j in range(5)]
    L.append(f"| {lab} | {e[..., 0].mean():.4f} | {e[..., 1].mean():.2f} | {e[..., 2].mean():.3f} | {e[..., 3].mean():.3f} | {e[..., 4].mean():.3f} | " + " | ".join(f"{c:.2f}" for c in chi) + " |")
print("\n".join(L))
print()
print("a1/a0 per cell (rows nue antinue numu antinumu; columns E 0.3 0.5 1.0 GeV); Honda first")
def row(a):
    return "; ".join(" ".join(f"{x:.3f}" for x in a[k, :, 0]) for k in range(4))
print("Honda:", row(hon))
for lab in labs:
    print(f"{lab}:", row(z[lab + "_est"]))
