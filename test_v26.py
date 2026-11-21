"""Continuous beams with point loads, equilibrium, moment redistribution and unit conversion."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_fe import equilibrium_error, redistribute_moments, shear_moment_fe  # noqa: E402
from beam_solver import BeamModel, PointLoad, Support, UDL, UVL  # noqa: E402
from design import DesignSpec, run_design, support_layout  # noqa: E402
from units import Units  # noqa: E402

E, I, L, W = 2e11, 1e-4, 6.0, 5e3


def _two_span(loads):
    return BeamModel(L=2 * L, E=E, I=I, supports=[Support(0, "pinned"), Support(L, "roller"),
                                                  Support(2 * L, "roller")], **loads)


def _rel(a, b):
    return abs(a - b) / abs(b)


# ---- continuous beam with a point load ----

def test_two_span_point_load_reactions_sum_to_load():
    P = 30e3
    m = _two_span({"point_loads": [PointLoad(3.0, P)]})
    sm = shear_moment_fe(m, 1201)
    assert abs(sum(sm["reactions"].values()) - P) < 1e-3 * P
    assert abs(equilibrium_error(sm, m)) < 1e-3 * P


def test_two_span_point_load_matches_three_moment_results():
    # Midspan point load P on span 1 of two equal spans (classic three-moment result):
    #   support moment M_B = -3PL/32, centre reaction R_B = 11P/16, far reaction R_C = -3P/32 (uplift)
    P = 20e3
    m = _two_span({"point_loads": [PointLoad(L / 2, P)]})
    sm = shear_moment_fe(m, 1201)
    assert _rel(sm["reactions"][L], 11 * P / 16) < 1e-3
    assert _rel(sm["reactions"][2 * L], -3 * P / 32) < 1e-3
    assert _rel(sm["M"][np.argmin(np.abs(sm["x"] - L))], -3 * P * L / 32) < 1e-3


def test_two_span_mixed_loads_equilibrium():
    m = _two_span({"udls": [UDL(0, 2 * L, W)], "point_loads": [PointLoad(4.0, 15e3)],
                   "uvls": [UVL(7.0, 11.0, 0, 6e3)]})
    sm = shear_moment_fe(m, 1201)
    assert abs(equilibrium_error(sm, m)) < 1e-2  # N, on loads of ~100 kN


# ---- moment redistribution ----

def test_redistribution_reduces_support_moment_by_fraction():
    m = _two_span({"udls": [UDL(0, 2 * L, W)]})
    el = shear_moment_fe(m, 1201)
    rd = redistribute_moments(el, m, 0.2)
    i = np.argmin(np.abs(el["x"] - L))
    assert _rel(rd["M"][i], -0.8 * W * L**2 / 8) < 1e-3


def test_redistribution_raises_span_moment_by_closed_form():
    m = _two_span({"udls": [UDL(0, 2 * L, W)]})
    rd = redistribute_moments(shear_moment_fe(m, 1201), m, 0.2)
    span_mid = rd["M"][np.argmin(np.abs(rd["x"] - L / 2))]
    assert _rel(span_mid, 0.075 * W * L**2) < 1e-3   # 0.0625 + 0.0125 wL²


def test_redistribution_preserves_equilibrium_and_total_reaction():
    m = _two_span({"udls": [UDL(0, 2 * L, W)], "point_loads": [PointLoad(3.0, 20e3)]})
    el = shear_moment_fe(m, 1201)
    rd = redistribute_moments(el, m, 0.25)
    assert abs(sum(rd["reactions"].values()) - sum(el["reactions"].values())) < 1e-6
    assert abs(equilibrium_error(rd, m)) < 1e-2


def test_redistribution_zero_is_identity_and_determinate_is_untouched():
    m = _two_span({"udls": [UDL(0, 2 * L, W)]})
    el = shear_moment_fe(m, 1201)
    assert redistribute_moments(el, m, 0.0) is el
    simple = BeamModel(L=L, E=E, I=I, supports=[Support(0, "pinned"), Support(L, "roller")], udls=[UDL(0, L, W)])
    from beam_solver import shear_moment
    sm = shear_moment(simple, 601)
    assert redistribute_moments(sm, simple, 0.2) is sm


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def test_redistribution_rejects_out_of_range_beta():
    m = _two_span({"udls": [UDL(0, 2 * L, W)]})
    el = shear_moment_fe(m, 601)
    assert _raises(ValueError, lambda: redistribute_moments(el, m, 1.0))


def test_design_applies_redistribution_and_reports_equilibrium():
    spec = DesignSpec(L=2 * L, supports=support_layout("two_span", 2 * L),
                      loads=[UDL(0, 2 * L, W, "L"), PointLoad(3.0, 20e3, "L")],
                      section_name="ISMB 300", grade="S275", auto_select=False, redistribution=0.2)
    res = run_design(spec)
    assert res["redistribution"] == 0.2
    assert res["max_equilibrium_error_N"] < 1e-2
    base = run_design(DesignSpec(**{**spec.__dict__, "redistribution": 0.0}))
    assert res["M_env_kNm"] != base["M_env_kNm"]


def test_design_rejects_redistribution_above_30_percent():
    spec = DesignSpec(L=2 * L, supports=support_layout("two_span", 2 * L), loads=[UDL(0, 2 * L, W, "L")],
                      section_name="ISMB 300", redistribution=0.5)
    assert _raises(ValueError, lambda: run_design(spec))


# ---- units ----

def test_same_load_in_kn_m_and_n_mm_gives_same_si_values():
    # 5 kN/m is the same load as 5 N/mm; 6 m is the same span as 6000 mm; 10 kN·m = 1e7 N·mm
    kn = Units("kN", "m")
    nmm = Units("N", "mm")
    assert abs(kn.intensity_to_Npm(5.0) - 5000.0) < 1e-9
    assert abs(nmm.intensity_to_Npm(5.0) - 5000.0) < 1e-9
    assert abs(kn.length_to_m(6.0) - nmm.length_to_m(6000.0)) < 1e-12
    assert abs(kn.moment_to_Nm(10.0) - nmm.moment_to_Nm(1e7)) < 1e-6


def test_unit_round_trips_are_exact():
    for fu in ("kN", "N"):
        for lu in ("m", "mm"):
            u = Units(fu, lu)
            assert abs(u.Nm_to_moment(u.moment_to_Nm(123.4)) - 123.4) < 1e-9
            assert abs(u.N_to_force(u.force_to_N(77.0)) - 77.0) < 1e-12
            assert abs(u.m_to_length(u.length_to_m(4.5)) - 4.5) < 1e-12
            assert abs(u.Npm_to_intensity(u.intensity_to_Npm(9.0)) - 9.0) < 1e-12


def test_units_known_values():
    assert Units("kN", "m").intensity_to_Npm(5) == 5000.0     # 5 kN/m
    assert Units("N", "mm").intensity_to_Npm(5) == 5000.0     # 5 N/mm = 5000 N/m
    assert Units("kN", "m").moment_to_Nm(10) == 10000.0       # 10 kN·m
    assert abs(Units("N", "mm").moment_to_Nm(10) - 0.01) < 1e-12  # 10 N·mm = 0.01 N·m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
