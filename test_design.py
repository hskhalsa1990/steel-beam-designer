"""Tests for load cases, combinations, lateral-torsional buckling, design layer and report."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_solver import BeamModel, PointLoad, Support, UDL, analyse, scale_model  # noqa: E402
from design import DesignSpec, LoadCombo, run_design, support_layout  # noqa: E402
from report import build_report  # noqa: E402
from steel_sections import get_section, lateral_torsional, stress_check  # noqa: E402


def test_scale_model_applies_case_factors():
    m = BeamModel(L=6.0, E=2e11, I=1e-4, supports=[Support(0, "pinned"), Support(6, "roller")],
                  udls=[UDL(0, 6, 5e3, "D"), UDL(0, 6, 10e3, "L")])
    r = analyse(scale_model(m, {"D": 1.5, "L": 1.5}), 600)
    assert abs(r["M_max"] - 1.5 * 15e3 * 36 / 8) / (1.5 * 15e3 * 36 / 8) < 0.01


def test_scale_model_missing_case_is_zero():
    m = BeamModel(L=6.0, E=2e11, I=1e-4, supports=[Support(0, "pinned"), Support(6, "roller")],
                  udls=[UDL(0, 6, 5e3, "D"), UDL(0, 6, 10e3, "L")])
    r = analyse(scale_model(m, {"D": 1.0}), 600)  # live load not applied
    assert abs(r["M_max"] - 5e3 * 36 / 8) / (5e3 * 36 / 8) < 0.01


def test_ltb_no_reduction_for_short_unbraced_length():
    res = lateral_torsional(get_section("ISMB 300"), "S275", 0.5)
    assert res["chi_LT"] == 1.0


def test_ltb_reduces_with_unbraced_length():
    sec = get_section("ISMB 300")
    chis = [lateral_torsional(sec, "S275", L)["chi_LT"] for L in (2, 4, 6, 10)]
    assert all(a >= b for a, b in zip(chis, chis[1:]))
    assert chis[-1] < 0.5


def test_ltb_flows_into_stress_check():
    sec = get_section("ISMB 300")
    unbraced = stress_check(60e3, 20e3, sec, "S275", Lcr_m=10.0)
    braced = stress_check(60e3, 20e3, sec, "S275", Lcr_m=0.5)
    assert unbraced["util_bending"] > braced["util_bending"]
    assert unbraced["ltb"]["chi_LT"] < 1.0


def test_section_properties_are_plausible():
    sec = get_section("ISMB 300")
    # Catalog weak-axis Iyy for ISMB 300 is ~453 cm4; approximation should be same order.
    assert 350 < sec.Iy_m4 * 1e8 < 550
    assert sec.It_m4 > 0 and sec.Iw_m6 > 0


def _spec(**kw):
    base = dict(L=6.0, supports=support_layout("simple", 6.0),
                loads=[UDL(0, 6.0, 5e3, "L"), PointLoad(3.0, 20e3, "L")],
                section_name="ISMB 300", grade="S275")
    base.update(kw)
    return DesignSpec(**base)


def test_governing_combo_is_the_factored_one():
    res = run_design(_spec(auto_select=False))
    names = {r["combo"]: r["M_kNm"] for r in res["per_combo"]}
    assert res["governing_moment_combo"] == "1.5(D+L)"
    assert names["1.5(D+L)"] > names["1.35D"]


def test_self_weight_adds_dead_load():
    with_sw = run_design(_spec(auto_select=False, self_weight=True))["M_env_kNm"]
    without = run_design(_spec(auto_select=False, self_weight=False))["M_env_kNm"]
    assert with_sw > without


def test_auto_select_passes_all_checks_and_is_lightest():
    res = run_design(_spec(auto_select=True))
    assert res["status"] == "safe"
    assert res["ok"] is True
    assert res["checks"]["safe"] and res["serviceability"]["ok"]
    # no lighter catalog section also passes
    from steel_sections import SECTIONS
    lighter = [n for n, s in SECTIONS.items() if s.weight < res["weight_kg_m"]]
    for n in lighter:
        assert not run_design(_spec(auto_select=False, section_name=n))["ok"]


def test_serviceability_uses_unfactored_loads():
    """Deflection must equal the unfactored D+L model (self-weight included), not factored loads."""
    spec = _spec(auto_select=False)
    res = run_design(spec)
    sec = get_section("ISMB 300")
    w_sw = sec.weight * 9.81
    m = BeamModel(L=6.0, E=2.1e11, I=sec.I_m4, supports=support_layout("simple", 6.0),  # S275 E
                  udls=[UDL(0, 6.0, 5e3, "L"), UDL(0, 6.0, w_sw, "D")],
                  point_loads=[PointLoad(3.0, 20e3, "L")])
    direct = analyse(m, 500)
    assert abs(res["y_max_mm"] - direct["y_max_mm"]) / direct["y_max_mm"] < 1e-6


def test_user_custom_combo_changes_envelope():
    lo = run_design(_spec(auto_select=False, combos=[LoadCombo("1.0(D+L)", {"D": 1, "L": 1})]))
    hi = run_design(_spec(auto_select=False, combos=[LoadCombo("1.5(D+L)", {"D": 1.5, "L": 1.5})]))
    assert hi["M_env_kNm"] > lo["M_env_kNm"]


def test_report_is_a_pdf():
    spec = _spec(auto_select=True)
    pdf = build_report(run_design(spec), spec)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
