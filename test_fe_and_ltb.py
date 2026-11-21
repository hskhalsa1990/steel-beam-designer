"""Finite-element (indeterminate beams) and lateral-torsional buckling verification tests."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_fe import deflection_fe, shear_moment_fe  # noqa: E402
from beam_solver import BeamModel, PointLoad, PointMoment, Support, UDL, shear_moment  # noqa: E402
from design import DesignSpec, run_design, support_layout  # noqa: E402
from steel_sections import SECTIONS, SteelSection, get_section, lateral_torsional  # noqa: E402

E, I, L, W = 2e11, 1e-4, 6.0, 5e3
EI = E * I


def _rel(a, b):
    return abs(a - b) / abs(b)


# ---------- indeterminate beams (finite-element solver) ----------

def test_fe_matches_determinate_solver_for_simple_span():
    m = BeamModel(L=L, E=E, I=I, supports=[Support(0, "pinned"), Support(L, "roller")], udls=[UDL(0, L, W)])
    fe, det = shear_moment_fe(m, 601), shear_moment(m, 601)
    assert np.max(np.abs(fe["M"] - det["M"])) < 10.0  # N·m on a 22.5 kN·m peak
    assert _rel(fe["reactions"][0.0], W * L / 2) < 1e-6


def test_fixed_fixed_udl_closed_form():
    m = BeamModel(L=L, E=E, I=I, supports=[Support(0, "fixed"), Support(L, "fixed")], udls=[UDL(0, L, W)])
    sm = shear_moment_fe(m, 601)
    # moments from Hermite elements converge at O(h^2); 120 elements gives < 0.1 %
    assert _rel(sm["M"][0], -W * L**2 / 12) < 1e-3          # hogging at ends
    assert _rel(sm["M"][300], W * L**2 / 24) < 1e-3         # sagging at midspan
    assert _rel(sm["reactions"][0.0], W * L / 2) < 1e-6
    assert _rel(deflection_fe(m, 601)["y_max"], W * L**4 / (384 * EI)) < 1e-4


def test_propped_cantilever_reactions_and_moment():
    m = BeamModel(L=L, E=E, I=I, supports=[Support(0, "fixed"), Support(L, "roller")], udls=[UDL(0, L, W)])
    sm = shear_moment_fe(m, 601)
    assert _rel(sm["reactions"][0.0], 5 * W * L / 8) < 1e-6  # fixed end carries 5wL/8
    assert _rel(sm["reactions"][L], 3 * W * L / 8) < 1e-6    # prop carries 3wL/8
    assert _rel(sm["M"][0], -W * L**2 / 8) < 1e-3            # fixed-end moment wL²/8


def test_propped_cantilever_deflection_matches_exact_curve():
    # Exact curve from integrating the moment equation (not the rounded wL^4/185EI).
    x = np.linspace(0, L, 200001)
    v = (W / EI) * (L**2 * x**2 / 16 - 5 * L * x**3 / 48 + x**4 / 24)
    assert _rel(deflection_fe(BeamModel(L=L, E=E, I=I, supports=[Support(0, "fixed"), Support(L, "roller")],
                                        udls=[UDL(0, L, W)]), 4001)["y_max"], v.max()) < 1e-4


def test_couple_sign_agrees_with_determinate_solver():
    m = BeamModel(L=L, E=E, I=I, supports=[Support(0, "pinned"), Support(L, "roller")],
                  moments=[PointMoment(2.0, 10e3)])
    fe, det = shear_moment_fe(m, 601), shear_moment(m, 601)
    assert np.max(np.abs(fe["M"] - det["M"])) < 1.0


def test_point_load_propped_cantilever():
    # Fixed-roller beam, midspan point load: M_fixed = 3PL/16, R_fixed = 11P/16, R_roller = 5P/16
    P = 20e3
    m = BeamModel(L=L, E=E, I=I, supports=[Support(0, "fixed"), Support(L, "roller")],
                  point_loads=[PointLoad(L / 2, P)])
    sm = shear_moment_fe(m, 601)
    assert _rel(abs(sm["M"][0]), 3 * P * L / 16) < 1e-6
    assert _rel(sm["reactions"][0.0], 11 * P / 16) < 1e-6
    assert _rel(sm["reactions"][L], 5 * P / 16) < 1e-6


def test_design_layer_runs_on_propped_and_fixed_fixed():
    for kind in ("propped", "fixed_fixed"):
        spec = DesignSpec(L=L, supports=support_layout(kind, L), loads=[UDL(0, L, W, "L")],
                          section_name="ISMB 300", grade="S275", auto_select=True)
        res = run_design(spec)
        assert res["status"] in ("safe", "none_safe", "manual")
        assert res["M_env_kNm"] > 0


# ---------- lateral-torsional buckling: independent limits ----------

class _NoWarping(SteelSection):
    @property
    def Iw_m6(self) -> float:  # override: warping constant set to zero
        return 0.0


def test_mcr_warping_free_limit_matches_torsion_formula():
    """With Iw = 0, Mcr must reduce to (pi/L) * sqrt(E Iy G It)  (independent closed form)."""
    base = get_section("ISMB 300")
    sec = _NoWarping(base.name, base.h, base.b, base.tw, base.tf, base.Ixx, base.Zxx, base.A,
                     base.weight, base.rxx)
    Lcr = 4.0
    G = 2.1e11 / (2 * 1.3)
    expected = np.pi / Lcr * np.sqrt(2.1e11 * sec.Iy_m4 * G * sec.It_m4)
    got = lateral_torsional(sec, "S275", Lcr)["Mcr_kNm"] * 1e3
    assert _rel(got, expected) < 1e-9


def test_mcr_is_torsion_dominated_for_catalog_section_over_10_to_20_m():
    """ISMB 300 over 10–20 m is torsion-dominated, so Mcr ~ 1/L (ratio ~2), not 1/L²."""
    sec = get_section("ISMB 300")
    m1 = lateral_torsional(sec, "S275", 10.0)["Mcr_kNm"]
    m2 = lateral_torsional(sec, "S275", 20.0)["Mcr_kNm"]
    assert 1.9 < m1 / m2 < 2.2


def test_lateral_torsional_is_monotonic_across_catalog():
    for name, sec in SECTIONS.items():
        chis = [lateral_torsional(sec, "S275", L_)["chi_LT"] for L_ in (1, 3, 6, 12)]
        assert all(a >= b - 1e-12 for a, b in zip(chis, chis[1:])), name


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
