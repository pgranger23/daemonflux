"""All-nucleon flux of each crflux model in its native and non-native energy
variable over GSF (total energy), vs kinetic T per nucleon; plus GSFG checks."""
import numpy as np, primaries as prim
T = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100.])
gsf = prim.primary_model("GSF")
ref = gsf.nucleon_flux(T, "total")[0]
L = ["# all-nucleon flux / GSF(total), vs kinetic T/n [GeV]: " + " ".join(f"{t:g}" for t in T)]
def allnuc(model, conv):
    return sum(prim.nucleon_intensity(model, g, T, a, conv) for _n, a, z, g in prim.SPECIES)
for nm in ("H3A", "GST", "GST4", "GH"):
    m = prim.primary_model(nm)
    for conv in ("total", "kinetic"):
        tag = " (native)" if conv == prim.native_convention(m) else ""
        L.append(f"{nm:5s} {conv:8s}{tag:9s}" + " ".join(f"{x:.3f}" for x in allnuc(m, conv) / ref))
for ref_name in ("H3A", "GST", "GH"):
    g = prim.primary_model("GSFG:" + ref_name)
    jp = sum(g.group_split(gid, T)[0] for _n, a, z, gid in prim.SPECIES)
    jn = sum(g.group_split(gid, T)[1] for _n, a, z, gid in prim.SPECIES)
    _f, pf, nf = gsf._m.p_and_n_flux(T + prim.M_NUCLEON)
    L.append(f"GSFG:{ref_name} sum_j(Jp)/GSF_p {np.max(abs(jp/(pf*1e-4)-1)):.2e}  sum_j(Jn)/GSF_n {np.max(abs(jn/(nf*1e-4)-1)):.2e}")
    L.append(f"   group nucleon shares at T=" + " ".join(f"{t:g}" for t in (3, 10, 30)))
    for _n, a, z, gid in prim.SPECIES:
        tt = np.array([3., 10., 30.])
        p, n = g.group_split(gid, tt)
        _f, pf, nf = gsf._m.p_and_n_flux(tt + prim.M_NUCLEON)
        L.append(f"     {_n:5s} " + " ".join(f"{x:.3f}" for x in (p + n) / ((pf + nf) * 1e-4)))
open(__import__("os").environ["M"] + "/reports/native_table.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
