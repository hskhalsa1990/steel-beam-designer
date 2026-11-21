"""Tests for continuous beams, lateral-restraint sensitivity, LTB-aware comparison and load-case report."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_fe import shear_moment_fe  # noqa: E402
from beam_solver import BeamModel, PointLoad, Support, UDL  # noqa: E402
from design import DesignSpec, describe_loads, ltb_sensitivity, run_design, support_layout  # noqa: E402
from report import build_report  # noqa: E402
from steel_sections import compare_sections, get_section, lateral_torsional  # noqa: E402

E, I, L, W = 2e11, 1e-4, 6.0, 5e3


def _rel(a, b):
    return abs(a - b) / abs(b)


# ---- continuous beam (more than two supports) ----

def test_two_span_continuous_reactions():
    m = BeamModel(L=2 * L, E=E, I=I, supports=[Support(0, "pinned"), Support(L, "roller"),
                                               Support(2 * L, "roller")], udls=[UDL(0, 2 * L, W)])
    sm = shear_moment_fe(m, 1201)
    assert _rel(sm["reactions"][0.0], 3 * W * L / 8) < 1e-3
    assert _rel(sm["reactions"][L], 5 * W * L / 4) < 1e-3
    assert _rel(sm["reactions"][2 * L], 3 * W * L / 8) < 1e-3


def test_two_span_centre_moment_and_span_moment():
    m = BeamModel(L=2 * L, E=E, I=I, supports=[Support(0, "pinned"), Support(L, "roller"),
                                               Support(2 * L, "roller")], udls=[UDL(0, 2 * L, W)])
    sm = shear_moment_fe(m, 1201)
    assert _rel(sm["M"][np.argmin(np.abs(sm["x"] - L))], -W * L**2 / 8) < 1e-3
    assert _rel(sm["M"].max(), 9 * W * L**2 / 128) < 1e-3


def test_two_span_layout_and_design_run():
    spec = DesignSpec(L=2 * L, supports=support_layout("two_span", 2 * L), loads=[UDL(0, 2 * L, W, "L")],
                      section_name="ISMB 300", grade="S275", auto_select=True)
    res = run_design(spec)
    assert res["status"] in ("safe", "none_safe", "manual")
    assert len(res["reactions"]) == 3


# ---- lateral restraint sensitivity ----

def _spec():
    return DesignSpec(L=6.0, supports=support_layout("simple", 6.0), loads=[UDL(0, 6, 5e3, "L")],
                      section_name="ISMB 300", grade="S275")


def test_ltb_sweep_matches_single_length_check():
    spec = _spec()
    rows = ltb_sensitivity(spec, "ISMB 300", [6.0])
    from design import _check_section
    M_env = _check_section(spec, "ISMB 300")["M_env_kNm"] * 1e3
    ltb = lateral_torsional(get_section("ISMB 300"), "S275", 6.0)
    assert _rel(rows[0]["util_LTB"], M_env / 1e3 / ltb["Mdb_kNm"]) < 1e-9


def test_ltb_sweep_is_monotonic_in_length():
    rows = ltb_sensitivity(_spec(), "ISMB 300", np.linspace(0.5, 15, 25))
    utils = [r["util_LTB"] for r in rows]
    chis = [r["chi_LT"] for r in rows]
    assert all(a <= b + 1e-12 for a, b in zip(utils, utils[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(chis, chis[1:]))


# ---- LTB-aware steel comparison ----

def test_compare_sections_includes_ltb_and_is_stricter():
    plain = {r["section"]: r["util_b"] for r in compare_sections(120e3, 60e3, "S275")}
    ltb = {r["section"]: r for r in compare_sections(120e3, 60e3, "S275", Lcr_m=10.0)}
    for name, r in ltb.items():
        assert r["util_b"] >= plain[name] - 1e-9
        assert r["chi_LT"] is not None and r["util_LTB"] is not None


def test_compare_sections_without_lcr_has_no_ltb_columns_filled():
    rows = compare_sections(120e3, 60e3, "S275")
    assert all(r["chi_LT"] is None and r["util_LTB"] is None for r in rows)


# ---- load-case report section ----

def test_describe_loads_lists_every_case_and_self_weight():
    spec = DesignSpec(L=6.0, supports=support_layout("simple", 6.0),
                      loads=[UDL(0, 6, 5e3, "L"), PointLoad(3, 20e3, "D")], section_name="ISMB 300")
    rows = describe_loads(spec)
    kinds = [r["type"] for r in rows]
    assert kinds == ["UDL", "Point load", "Self-weight"]
    assert rows[1]["case"] == "D"


def test_report_with_load_cases_and_sensitivity_is_pdf():
    spec = DesignSpec(L=12.0, supports=support_layout("two_span", 12.0),
                      loads=[UDL(0, 12, 8e3, "L"), PointLoad(3, 30e3, "L")], grade="S275", lateral_m=4.0)
    pdf = build_report(run_design(spec), spec)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
