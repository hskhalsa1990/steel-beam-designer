"""
Design layer: load combinations, strength/serviceability checks and automatic
section selection. No UI code here, so it can be tested directly.

Load cases: every load carries a case tag ("D" dead, "L" live). Strength uses the
envelope of the factored combinations; serviceability uses the unfactored D + L.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from beam_fe import equilibrium_error, redistribute_moments
from beam_solver import (
    BeamModel, PointLoad, PointMoment, Support, UDL, UVL,
    analyse, scale_model, shear_moment,
)
from steel_sections import SECTIONS, STEEL_GRADES, get_section, stress_check

G_ACC = 9.81  # m/s²


@dataclass(frozen=True)
class LoadCombo:
    name: str
    factors: Dict[str, float]


# Simplified gravity combinations (modelled on IS 875-style 1.5 / 1.35 factors).
DEFAULT_COMBOS: tuple = (
    LoadCombo("1.5(D+L)", {"D": 1.5, "L": 1.5}),
    LoadCombo("1.35D", {"D": 1.35}),
)
SERVICE_COMBO = LoadCombo("Service D+L", {"D": 1.0, "L": 1.0})


@dataclass
class DesignSpec:
    L: float
    supports: List[Support]
    loads: Sequence = ()                       # PointLoad / UDL / UVL / PointMoment with .case
    section_name: str = "ISMB 300"
    grade: str = "S275"
    combos: Sequence[LoadCombo] = DEFAULT_COMBOS
    defl_ratio: float = 250.0
    lateral_m: Optional[float] = None          # unrestrained length for LTB; None -> L
    self_weight: bool = True                   # adds a D UDL from the section mass
    auto_select: bool = True
    redistribution: float = 0.0                # fraction of interior hogging moment redistributed (0–0.3)


def _base_model(spec: DesignSpec, sec) -> BeamModel:
    """Unfactored model for a given section, including self-weight (case D)."""
    E = STEEL_GRADES[spec.grade]["E"] * 1e6
    pl = [p for p in spec.loads if isinstance(p, PointLoad)]
    ud = [u for u in spec.loads if isinstance(u, UDL)]
    uv = [v for v in spec.loads if isinstance(v, UVL)]
    mo = [m for m in spec.loads if isinstance(m, PointMoment)]
    if spec.self_weight:
        ud = ud + [UDL(0.0, spec.L, sec.weight * G_ACC, "D")]
    return BeamModel(L=spec.L, E=E, I=sec.I_m4, supports=list(spec.supports),
                     point_loads=pl, udls=ud, uvls=uv, moments=mo)


def _check_section(spec: DesignSpec, sec_name: str) -> dict:
    sec = get_section(sec_name)
    base = _base_model(spec, sec)
    Lcr = spec.lateral_m if spec.lateral_m else spec.L

    if not 0.0 <= spec.redistribution <= 0.3:
        raise ValueError("redistribution must be between 0 and 0.3 (30 %)")
    per_combo = []
    env_M, env_V = 0.0, 0.0
    gov_M_name, gov_M_res = None, None
    gov_V_name = None
    max_eq_err = 0.0
    for c in spec.combos:
        m_c = scale_model(base, c.factors)
        r = shear_moment(m_c, n=500)
        if spec.redistribution > 0:
            r = redistribute_moments(r, m_c, spec.redistribution)
        max_eq_err = max(max_eq_err, abs(equilibrium_error(r, m_c)))
        per_combo.append({"combo": c.name, "M_kNm": r["M_max"] / 1e3, "V_kN": r["V_max"] / 1e3,
                          "eq_err_N": equilibrium_error(r, m_c)})
        if r["M_max"] > env_M:
            env_M, gov_M_name, gov_M_res = r["M_max"], c.name, r
        if r["V_max"] > env_V:
            env_V, gov_V_name = r["V_max"], c.name

    chk = stress_check(env_M, env_V, sec, spec.grade, Lcr_m=Lcr)

    service = analyse(scale_model(base, SERVICE_COMBO.factors), n=500,
                      defl_limit_ratio=spec.defl_ratio)
    if gov_M_res is None:  # no loads at all
        gov_M_res = shear_moment(base, n=500)

    return {
        "section": sec_name,
        "weight_kg_m": sec.weight,
        "checks": chk,
        "strength_ok": chk["safe"],
        "defl_ok": service["serviceability"]["ok"],
        "ok": chk["safe"] and service["serviceability"]["ok"],
        "per_combo": per_combo,
        "governing_moment_combo": gov_M_name,
        "governing_shear_combo": gov_V_name,
        "M_env_kNm": env_M / 1e3,
        "V_env_kN": env_V / 1e3,
        "serviceability": service["serviceability"],
        "x": gov_M_res["x"],
        "V": gov_M_res["V"],
        "M": gov_M_res["M"],
        "reactions": gov_M_res["reactions"],
        "y": service["y"],
        "y_max_mm": service["y_max_mm"],
        "lateral_m": Lcr,
        "redistribution": spec.redistribution,
        "max_equilibrium_error_N": max_eq_err,
    }


def run_design(spec: DesignSpec) -> dict:
    """
    Run all checks. With auto_select, the lightest section passing strength
    (incl. lateral-torsional buckling) AND the deflection limit is chosen.
    Returns a dict with the chosen section's results and a `status`:
      "manual"      auto-select off; the chosen section is used
      "safe"        auto-select picked a passing section
      "none_safe"   no catalog section passes; the manual section is shown
    """
    manual = _check_section(spec, spec.section_name)
    if not spec.auto_select:
        return {**manual, "status": "manual", "requested": spec.section_name}

    candidates = sorted(SECTIONS.keys(), key=lambda n: SECTIONS[n].weight)
    for name in candidates:
        res = _check_section(spec, name)
        if res["ok"]:
            return {**res, "status": "safe", "requested": spec.section_name}
    return {**manual, "status": "none_safe", "requested": spec.section_name}


def describe_loads(spec: DesignSpec) -> List[dict]:
    """Human-readable list of every load case in the spec (self-weight included)."""
    rows = []
    for p in spec.loads:
        if isinstance(p, PointLoad):
            rows.append({"case": p.case, "type": "Point load", "where": f"x = {p.x:.2f} m",
                         "magnitude": f"{p.P/1e3:.2f} kN"})
        elif isinstance(p, UDL):
            rows.append({"case": p.case, "type": "UDL", "where": f"{p.x1:.2f}–{p.x2:.2f} m",
                         "magnitude": f"{p.w/1e3:.2f} kN/m"})
        elif isinstance(p, UVL):
            rows.append({"case": p.case, "type": "UVL", "where": f"{p.x1:.2f}–{p.x2:.2f} m",
                         "magnitude": f"{p.w1/1e3:.2f} → {p.w2/1e3:.2f} kN/m"})
        elif isinstance(p, PointMoment):
            rows.append({"case": p.case, "type": "Point moment", "where": f"x = {p.x:.2f} m",
                         "magnitude": f"{p.M/1e3:.2f} kN·m"})
    if spec.self_weight:
        sec = get_section(spec.section_name)
        rows.append({"case": "D", "type": "Self-weight", "where": f"0–{spec.L:.2f} m",
                     "magnitude": f"{sec.weight * G_ACC / 1e3:.3f} kN/m"})
    return rows


def ltb_sensitivity(spec: DesignSpec, section_name: str, lengths_m: Sequence[float]) -> List[dict]:
    """
    Strength check of one section against the envelope bending moment for a range of
    unbraced lengths. Only the buckling check changes with L_cr, so the envelope is
    computed once.
    """
    from steel_sections import lateral_torsional
    sec = get_section(section_name)
    M_env = _check_section(spec, section_name)["M_env_kNm"] * 1e3  # N·m
    rows = []
    for L_cr in lengths_m:
        ltb = lateral_torsional(sec, spec.grade, L_cr)
        util = abs(M_env) / 1e3 / ltb["Mdb_kNm"] if ltb["Mdb_kNm"] > 0 else float("inf")
        rows.append({"Lcr_m": float(L_cr), "Mcr_kNm": ltb["Mcr_kNm"], "lambda_LT": ltb["lambda_LT"],
                     "chi_LT": ltb["chi_LT"], "Mdb_kNm": ltb["Mdb_kNm"], "M_env_kNm": M_env / 1e3,
                     "util_LTB": util, "ok": util <= 1.0})
    return rows


def support_layout(kind: str, L: float) -> List[Support]:
    """Standard support layouts used by the dashboard."""
    if kind == "simple":
        return [Support(0.0, "pinned"), Support(L, "roller")]
    if kind == "cantilever":
        return [Support(0.0, "fixed")]
    if kind == "overhang":
        return [Support(0.2 * L, "pinned"), Support(0.85 * L, "roller")]
    if kind == "propped":
        return [Support(0.0, "fixed"), Support(L, "roller")]
    if kind == "fixed_fixed":
        return [Support(0.0, "fixed"), Support(L, "fixed")]
    if kind == "two_span":
        # two equal spans of length L/2 each, pin at left, rollers at mid and right
        return [Support(0.0, "pinned"), Support(L / 2, "roller"), Support(L, "roller")]
    raise ValueError(f"unknown support layout: {kind}")
