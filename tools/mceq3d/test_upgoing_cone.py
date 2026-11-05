"""``solve(upgoing_cone='mirror')``: the up-going rows get the cone correction F_cone/F_single of
the mirrored down-going direction (logic only; the physics is validated in INTEGRATION.md 11)."""
import importlib.util  # noqa: F401
import numpy as np
import pytest

import mceq3d_flux as mf
from mceq3d_flux import MCEq3DFlux, SPECIES


class Fake(MCEq3DFlux):
    def __init__(self):
        self.calls = []

        def solve(lat, lon, cos_zeniths, azimuths, **kw):
            self.calls.append(dict(kw, cz=np.asarray(cos_zeniths, float)))
            cz = np.asarray(cos_zeniths, float)
            if kw.get("cone_cutoff", True) is False:
                val = 2.0
            elif kw.get("cutoff_anchor") == "detector":
                val = 3.0 * (1.0 + np.abs(cz))[:, None, None]  # cone: depends on |cos Z|
            else:
                val = 1.0 + 0.0 * cz[:, None, None]
            fl = {s: np.broadcast_to(val * np.ones((len(cz), len(azimuths), 4)),
                                     (len(cz), len(azimuths), 4)).copy() for s in SPECIES}
            return {"flux": fl, "cos_zeniths": cz, "azimuths": np.asarray(azimuths)}

        self.solve = solve


def _call(**kw):
    f = Fake()
    cz = np.array([-0.95, -0.25, -0.05, 0.05, 0.95])
    res = MCEq3DFlux.solve(f, 36.43, 137.31, cz, np.array([10.0, 100.0]), **kw)
    return f, cz, res


def test_mirror_multiplies_only_the_near_horizon_up_going_rows():
    f, cz, res = _call(upgoing_cone="mirror", upgoing_cone_czmax=0.35)
    s = SPECIES[0]
    base = 1.0
    # up rows -0.25, -0.05: ratio F_cone/F_single of the mirrored |cos Z|
    for i, c in ((1, 0.25), (2, 0.05)):
        npt = 3.0 * (1.0 + c) / 2.0
        assert np.allclose(res["flux"][s][i], base * npt)
        assert np.allclose(res["upgoing_cone_ratio"][s][list(np.abs(cz[[1, 2]])).index(c)], npt)
    for i in (0, 3, 4):  # -0.95 beyond czmax, and the down-going rows: untouched
        assert np.allclose(res["flux"][s][i], base)
    # the two extra solves are at the mirrored directions, detector-anchored, cone vs single
    mirror_calls = [c for c in f.calls if c.get("cutoff_anchor") == "detector"]
    assert len(mirror_calls) == 2
    assert np.allclose(sorted(mirror_calls[0]["cz"]), [0.05, 0.25])
    assert mirror_calls[1]["cone_cutoff"] is False and mirror_calls[1]["joint_cone"] is False
    assert all(c["upgoing_cone"] == "none" for c in f.calls)




def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="upgoing_cone"):
        MCEq3DFlux.solve(Fake(), 36.43, 137.31, np.array([-0.5]), np.array([0.0]),
                         upgoing_cone="bogus")


def test_default_is_the_legacy_behaviour():
    assert mf.UPGOING_CONE == "none"
