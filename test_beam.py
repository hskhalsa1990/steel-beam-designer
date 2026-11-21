import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_solver import BeamModel, Support, PointLoad, UDL, PointMoment, analyse, serviceability
from steel_sections import get_section, recommend_section

def test_simple_udl():
    L, w = 6.0, 5000.0
    sec = get_section("ISMB 300")
    m = BeamModel(L=L, E=2e11, I=sec.I_m4,
                  supports=[Support(0,"pinned"), Support(L,"roller")],
                  udls=[UDL(0, L, w)])
    r = analyse(m, 600)
    assert abs(r["M_max"] - w*L**2/8)/(w*L**2/8) < 0.03
    assert abs(r["V_max"] - w*L/2)/(w*L/2) < 0.03

def test_cantilever_tip():
    L, P = 4.0, 10000.0
    sec = get_section("ISMB 250")
    m = BeamModel(L=L, E=2e11, I=sec.I_m4,
                  supports=[Support(0,"fixed")],
                  point_loads=[PointLoad(L, P)])
    r = analyse(m, 400)
    assert abs(r["M_max"] - P*L)/(P*L) < 0.05
    assert abs(r["V_max"] - P)/P < 0.05

def test_serviceability():
    s = serviceability(10.0, 5.0, 250)
    assert s["allow_mm"] == 20.0
    assert s["ok"] is True
    assert serviceability(25.0, 5.0, 250)["ok"] is False

def test_recommend():
    rec = recommend_section(50e3, 20e3)
    assert "recommended" in rec

def test_overhang_runs():
    sec = get_section("ISMB 300")
    m = BeamModel(L=8.0, E=2e11, I=sec.I_m4,
                  supports=[Support(1.0,"pinned"), Support(6.5,"roller")],
                  udls=[UDL(0, 8, 4000)])
    r = analyse(m, 400)
    assert r["M_max"] > 0 and r["V_max"] > 0

def test_overhang_roller_moment():
    """Regression: moment over the roller must be -wa^2/2 (was wrongly forced to 0)."""
    from beam_solver import shear_moment
    E, I = 2e11, 1e-4
    m = BeamModel(L=8.0, E=E, I=I, supports=[Support(2.0, "pinned"), Support(6.0, "roller")],
                  udls=[UDL(0, 8.0, 10e3)])
    sm = shear_moment(m, 801)
    i_roller = int(np.argmin(np.abs(sm["x"] - 6.0)))
    assert abs(sm["M"][i_roller] - (-20e3)) / 20e3 < 0.01
    assert abs(sm["M"][-1]) < 1e-6  # free tip

def test_ss_point_load_deflection():
    E, I = 2e11, 1e-4
    m = BeamModel(L=6.0, E=E, I=I, supports=[Support(0, "pinned"), Support(6, "roller")],
                  point_loads=[PointLoad(3.0, 30e3)])
    y = analyse(m, 600)["y_max"]
    exact = 30e3 * 6.0 ** 3 / (48 * E * I)
    assert abs(y - exact) / exact < 0.01

def test_cantilever_udl_tip_deflection():
    E, I = 2e11, 1e-4
    m = BeamModel(L=4.0, E=E, I=I, supports=[Support(0, "fixed")], udls=[UDL(0, 4.0, 5e3)])
    y = analyse(m, 600)["y_max"]
    exact = 5e3 * 4.0 ** 4 / (8 * E * I)
    assert abs(y - exact) / exact < 0.01

def test_point_moment_reactions():
    """Applied couple on a simple span: reactions equal and opposite, M_max = M0/2 ideally."""
    from beam_solver import shear_moment
    m = BeamModel(L=6.0, E=2e11, I=1e-4, supports=[Support(0, "pinned"), Support(6, "roller")],
                  moments=[PointMoment(3.0, 10e3)])
    sm = shear_moment(m, 601)
    assert abs(sm["reactions"][0.0] + sm["reactions"][6.0]) < 1e-6
    assert abs(sm["reactions"][0.0] - 10e3 / 6.0) < 1.0

def test_recommend_respects_deflection():
    """Auto-select must not return a section that fails the deflection check."""
    rec_strength = recommend_section(50e3, 20e3)
    rec_both = recommend_section(50e3, 20e3, defl_ok=lambda s: s.Ixx > 5000)
    assert rec_both["status"] == "safe"
    assert get_section(rec_both["recommended"]).Ixx > 5000
    assert rec_both["weight_kg_m"] >= rec_strength["weight_kg_m"]

def test_parametric_overhang_matches_full_model():
    """Parametric sweep must use the same support layout as the main model."""
    from beam_solver import parametric_udl
    sup = [Support(2.0, "pinned"), Support(6.0, "roller")]
    row = parametric_udl(8.0, 2e11, 1e-4, [10.0], supports=sup)[0]
    direct = analyse(BeamModel(L=8.0, E=2e11, I=1e-4, supports=sup, udls=[UDL(0, 8.0, 10e3)]), 300)
    assert abs(row["M_max_kNm"] - direct["M_max"] / 1e3) < 1e-9
    assert abs(row["M_max_kNm"] - 20.0) < 0.5  # |M| peaks at the supports (wa^2/2)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
