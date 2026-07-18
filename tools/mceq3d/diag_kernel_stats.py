"""Is the sampled pion-cone tail at 0.5-1 GeV real physics or thin-statistics noise?

diag_cone_shape.py showed a Gaussian cone of the SAME variance removes the
0.5-1 GeV E_off overshoot, so the excess is driven by the sampled kernel's
large-angle TAIL. This decides whether that tail is trustworthy:

  * per E_nu row: how many Monte-Carlo samples land in it (vs the floor=800 below
    which the code already falls back to Gaussian), and
  * a tail metric: p95/RMS of the sampled angle (Gaussian-on-sphere ~ 1.7-1.9;
    a much larger value = heavy, and if it rides on few samples = noisy).

If the 0.5-1 GeV rows are thin / the tail metric is noisy, using the Gaussian
where statistics are inadequate (raise the fallback floor) is a principled fix,
not tuning. If they are well-populated and robustly heavy-tailed, the tail is
real generator physics and the overshoot is a factorisation effect.

Run from tools/mceq3d (PYTHONPATH=$PWD). No MCEq needed.
"""
import numpy as np

from kinematic_kernel import pion_alpha_pdf, channel_shapes


def main():
    e_grid = np.geomspace(0.15, 20.0, 22)
    alpha = np.linspace(0.5, 89.0, 60)
    # big sample so the tail is well-measured where physics allows
    W = pion_alpha_pdf(e_grid, alpha, n=8_000_000, floor=1)  # floor=1: never zero
    sig_pi = channel_shapes(e_grid)["pi"]

    print("Sampled pion cone per E_nu: statistics and tail shape")
    print(f"{'E':>6} {'sig_pi':>7} {'Nsamp':>9} {'RMS':>6} {'p95':>6} "
          f"{'p95/RMS':>7} {'p99':>6}")
    for k, E in enumerate(e_grid):
        w = W[k]
        N = w.sum()
        if N < 1:
            print(f"{E:6.2f} {sig_pi[k]:7.1f} {int(N):9d}   (empty)")
            continue
        c = np.cumsum(w) / N
        rms = np.sqrt(np.average(alpha**2, weights=w))
        p95 = np.interp(0.95, c, alpha)
        p99 = np.interp(0.99, c, alpha)
        flag = ""
        if N < 800:
            flag = "  <- below floor (Gaussian fallback in delivered)"
        if E >= 0.4 and E <= 1.2:
            flag += "  [overshoot band]"
        print(f"{E:6.2f} {sig_pi[k]:7.1f} {int(N):9d} {rms:6.1f} {p95:6.1f} "
              f"{p95/max(rms,1e-9):7.2f} {p99:6.1f}{flag}")
    print("\nGaussian-on-sphere reference p95/RMS ~ 1.7-1.9, p99/RMS ~ 2.1.")
    print("Heavy p95/RMS on FEW samples in the overshoot band => noisy tail")
    print("(-> raise floor is principled). Heavy on MANY samples => real physics.")
    print("DIAG_KERNEL_STATS_DONE")


if __name__ == "__main__":
    main()
