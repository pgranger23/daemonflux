"""Isolate the 0.5-1 GeV overshoot: how much is the base sec-theta vs E_off's cone?

delivered H/V = base_HV(E) x [E_off(horizon)/E_off(vertical)].
The reference wants delivered H/V = Honda_HV. So:
  * if base_HV already exceeds Honda_HV, the base sec-theta overshoots on its own
    (E_off, being >1 at the horizon, can only make it worse) -> a BASE problem;
  * the cone that WOULD match the reference is  E_off_target = Honda_HV / base_HV;
    compare to the delivered E_off_actual = E_off(hor)/E_off(vert). If actual >
    target, E_off's cone is too strong -> an E_OFF problem.

This attributes the overshoot per energy, using the SAME (mceq) base the delivered
engine uses, with no tuning. Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import numpy as np


def main():
    cz = np.round(np.arange(0.05, 1.0, 0.1), 2)
    ih, iv = 0, int(np.argmax(cz))

    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="generic")
    e = eng.e
    base = eng.base(cz)["total_numu"]                 # (n_cz, nE)
    Eoff = eng.offaxis_factor(cz)["total_numu"]       # (n_cz, nE)

    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    import validate_bartol as vb
    Eb, czb, gb = vb.load_bartol("num", "fmin")

    def hHV(E):
        a = int(np.argmin(np.abs(Hcz - (cz[ih] - 0.05))))
        b = int(np.argmin(np.abs(Hcz - (cz[iv] - 0.05))))
        return np.interp(E, He, nm[a].mean(0)) / np.interp(E, He, nm[b].mean(0))

    def bHV(E):
        a = int(np.argmin(np.abs(czb - cz[ih])))
        b = int(np.argmin(np.abs(czb - cz[iv])))
        return np.interp(E, Eb, gb[a]) / np.interp(E, Eb, gb[b])

    print("numu horizon/vertical decomposition (mceq base):")
    print(f"{'E':>5} {'base_HV':>8} {'Eoff_act':>9} {'deliv_HV':>9} | "
          f"{'Honda':>7} {'Bartol':>7} | {'Eoff_tgtH':>9} {'Eoff_tgtB':>9} | "
          f"{'base/Hon':>8} {'act/tgtH':>8}")
    for E in (0.3, 0.5, 1.0, 2.0, 3.0):
        je = int(np.argmin(np.abs(e - E)))
        base_hv = base[ih, je] / base[iv, je]
        eoff_act = Eoff[ih, je] / Eoff[iv, je]
        deliv = base_hv * eoff_act
        H, B = hHV(E), bHV(E)
        tgtH, tgtB = H / base_hv, B / base_hv
        print(f"{E:5.1f} {base_hv:8.3f} {eoff_act:9.3f} {deliv:9.3f} | "
              f"{H:7.3f} {B:7.3f} | {tgtH:9.3f} {tgtB:9.3f} | "
              f"{base_hv/H:8.3f} {eoff_act/tgtH:8.3f}")
    print("\nbase/Hon > 1 => base sec-theta overshoots on its own (base problem).")
    print("act/tgtH  > 1 => E_off cone too strong vs what the ref wants (E_off problem).")
    print("DIAG_ISOLATE_DONE")


if __name__ == "__main__":
    main()
