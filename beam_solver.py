"""
Beam analysis v2 — simply supported, cantilever, overhang.
Loads: point, UDL, UVL (trapezoid), point moment.
Outputs: reactions, V(x), M(x), deflection, serviceability helpers.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, Dict

SupportType = Literal["pinned", "roller", "fixed", "free"]

@dataclass
class Support:
    x: float
    kind: SupportType

@dataclass
class PointLoad:
    x: float
    P: float  # N, downward +
    case: str = "L"  # load case tag, e.g. "D" (dead) or "L" (live)

@dataclass
class UDL:
    x1: float
    x2: float
    w: float  # N/m
    case: str = "L"

@dataclass
class UVL:
    x1: float
    x2: float
    w1: float
    w2: float
    case: str = "L"

@dataclass
class PointMoment:
    x: float
    M: float  # N·m
    case: str = "L"

@dataclass
class BeamModel:
    L: float
    E: float
    I: float
    supports: List[Support] = field(default_factory=list)
    point_loads: List[PointLoad] = field(default_factory=list)
    udls: List[UDL] = field(default_factory=list)
    uvls: List[UVL] = field(default_factory=list)
    moments: List[PointMoment] = field(default_factory=list)

def scale_model(model: "BeamModel", factors: Dict[str, float]) -> "BeamModel":
    """Return a copy of `model` with each load multiplied by factors[load.case].
    Cases missing from `factors` are treated as zero (not applied)."""
    f = lambda case: factors.get(case, 0.0)
    return BeamModel(
        L=model.L, E=model.E, I=model.I, supports=list(model.supports),
        point_loads=[PointLoad(p.x, p.P * f(p.case), p.case) for p in model.point_loads],
        udls=[UDL(u.x1, u.x2, u.w * f(u.case), u.case) for u in model.udls],
        uvls=[UVL(v.x1, v.x2, v.w1 * f(v.case), v.w2 * f(v.case), v.case) for v in model.uvls],
        moments=[PointMoment(m.x, m.M * f(m.case), m.case) for m in model.moments],
    )

def _is_indeterminate(model: BeamModel) -> bool:
    restraints = sum(2 if s.kind == "fixed" else 1
                     for s in model.supports if s.kind in ("pinned", "roller", "fixed"))
    return restraints > 2


def _vertical_supports(model: BeamModel) -> List[float]:
    return sorted(s.x for s in model.supports if s.kind in ("pinned", "roller", "fixed"))

def _total_load_and_moment_about(model: BeamModel, a: float) -> Tuple[float, float]:
    """Sum vertical loads and moment of loads about x=a (downward load +)."""
    total_F = 0.0
    Ma = 0.0
    for pl in model.point_loads:
        total_F += pl.P
        Ma += pl.P * (pl.x - a)
    for u in model.udls:
        length = u.x2 - u.x1
        F = u.w * length
        xc = 0.5 * (u.x1 + u.x2)
        total_F += F
        Ma += F * (xc - a)
    for v in model.uvls:
        length = v.x2 - v.x1
        F = 0.5 * (v.w1 + v.w2) * length
        if abs(v.w1 + v.w2) < 1e-12:
            xc = 0.5 * (v.x1 + v.x2)
        else:
            rel = length * (2 * v.w2 + v.w1) / (3 * (v.w1 + v.w2))
            xc = v.x1 + rel
        total_F += F
        Ma += F * (xc - a)
    for m in model.moments:
        Ma -= m.M
    return total_F, Ma

def _reactions(model: BeamModel) -> Dict[str, float]:
    """
    Returns dict with keys like R0, RL, M_fixed (if cantilever).
    Supports:
      - 1 fixed at 0: cantilever
      - 2 vertical supports: determinate beam (simple or overhang)
    """
    verts = _vertical_supports(model)
    out: Dict[str, float] = {}
    if len(verts) == 1 and model.supports[0].kind == "fixed":
        a = verts[0]
        F, Ma = _total_load_and_moment_about(model, a)
        out[f"R@{a}"] = F
        out["M_fixed"] = Ma
        return out
    if len(verts) < 2:
        # fallback ends
        verts = [0.0, model.L]
    a, b = verts[0], verts[-1]
    span = b - a
    F, Ma = _total_load_and_moment_about(model, a)
    Rb = Ma / span if span > 0 else 0.0
    Ra = F - Rb
    out[f"R@{a}"] = Ra
    out[f"R@{b}"] = Rb
    return out

def shear_moment(model: BeamModel, n: int = 500) -> dict:
    if _is_indeterminate(model):
        from beam_fe import shear_moment_fe  # local import: beam_fe depends on this module
        return shear_moment_fe(model, n)
    L = model.L
    x = np.linspace(0, L, n)
    V = np.zeros(n)
    M = np.zeros(n)
    reactions = _reactions(model)
    verts = _vertical_supports(model)
    is_cantilever = len(verts) == 1 and model.supports[0].kind == "fixed"
    fixed_M0 = reactions.get("M_fixed", 0.0) if is_cantilever else 0.0

    # reaction vertical map
    Rmap = {float(k.split("@")[1]): v for k, v in reactions.items() if k.startswith("R@")}

    for i, xi in enumerate(x):
        v = 0.0
        for xs, R in Rmap.items():
            if xi + 1e-12 >= xs:
                v += R
        for pl in model.point_loads:
            if xi + 1e-12 >= pl.x:
                v -= pl.P
        for u in model.udls:
            if xi > u.x1:
                seg = min(xi, u.x2) - u.x1
                if seg > 0:
                    v -= u.w * seg
        for uv in model.uvls:
            if xi > uv.x1:
                x2c = min(xi, uv.x2)
                seg = x2c - uv.x1
                if seg > 0 and uv.x2 > uv.x1:
                    t = seg / (uv.x2 - uv.x1)
                    w_local = uv.w1 + t * (uv.w2 - uv.w1)
                    v -= 0.5 * (uv.w1 + w_local) * seg
        V[i] = v

    # Bending moment in closed form at every station (no integration error at the
    # discontinuities caused by point loads, reactions and point moments).
    M0 = -fixed_M0 if is_cantilever else 0.0
    gx, gw = np.polynomial.legendre.leggauss(3)  # exact for the quadratic UVL integrand
    for i, xi in enumerate(x):
        m = M0
        for xs, R in Rmap.items():
            if xi + 1e-12 >= xs:
                m += R * (xi - xs)
        for pl in model.point_loads:
            if xi + 1e-12 >= pl.x:
                m -= pl.P * (xi - pl.x)
        for u in model.udls:
            c = min(xi, u.x2)
            if c > u.x1:
                m -= u.w * ((xi - u.x1) ** 2 - (xi - c) ** 2) / 2.0
        for uv in model.uvls:
            c = min(xi, uv.x2)
            if c > uv.x1 and uv.x2 > uv.x1:
                half, mid = 0.5 * (c - uv.x1), 0.5 * (c + uv.x1)
                t = mid + half * gx
                wt = uv.w1 + (t - uv.x1) * (uv.w2 - uv.w1) / (uv.x2 - uv.x1)
                m -= half * np.sum(gw * wt * (xi - t))
        for pm in model.moments:
            if xi + 1e-12 >= pm.x:
                m -= pm.M
        M[i] = m

    return {
        "x": x,
        "V": V,
        "M": M,
        "reactions": Rmap,
        "reactions_raw": reactions,
        "V_max": float(np.max(np.abs(V))),
        "M_max": float(np.max(np.abs(M))),
        "is_cantilever": is_cantilever,
    }

def deflection(model: BeamModel, sm: Optional[dict] = None, n: int = 500) -> dict:
    if _is_indeterminate(model):
        from beam_fe import deflection_fe
        return deflection_fe(model, n)
    if sm is None:
        sm = shear_moment(model, n)
    x, M = sm["x"], sm["M"]
    EI = model.E * model.I
    if EI <= 0:
        return {"x": x, "y": np.zeros_like(x), "y_max": 0.0, "y_max_mm": 0.0}
    dx = x[1] - x[0]
    theta = np.zeros_like(x)
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        theta[i] = theta[i - 1] + 0.5 * (M[i - 1] + M[i]) / EI * dx
        y[i] = y[i - 1] + 0.5 * (theta[i - 1] + theta[i]) * dx

    verts = _vertical_supports(model)
    if sm.get("is_cantilever"):
        pass  # y(0)=theta(0)=0 already
    elif len(verts) >= 2:
        i0 = int(np.argmin(np.abs(x - verts[0])))
        i1 = int(np.argmin(np.abs(x - verts[-1])))
        y0, y1 = y[i0], y[i1]
        x0, x1 = x[i0], x[i1]
        if abs(x1 - x0) > 1e-12:
            y = y - y0 - (y1 - y0) / (x1 - x0) * (x - x0)
    else:
        y = y - y[0]

    ymax = float(np.max(np.abs(y)))
    return {"x": x, "y": y, "y_max": ymax, "y_max_mm": ymax * 1e3}

def serviceability(y_max_mm: float, L: float, limit_ratio: float = 250.0) -> dict:
    """Check deflection against L/limit_ratio (default L/250)."""
    allow_mm = (L * 1000.0) / limit_ratio
    return {
        "y_max_mm": y_max_mm,
        "allow_mm": allow_mm,
        "ratio": (L * 1000.0) / y_max_mm if y_max_mm > 1e-9 else 1e9,
        "limit_ratio": limit_ratio,
        "ok": y_max_mm <= allow_mm + 1e-9,
    }

def analyse(model: BeamModel, n: int = 500, defl_limit_ratio: float = 250.0) -> dict:
    sm = shear_moment(model, n)
    df = deflection(model, sm, n)
    svc = serviceability(df["y_max_mm"], model.L, defl_limit_ratio)
    return {**sm, **df, "serviceability": svc}

def parametric_udl(L: float, E: float, I: float, w_list_kNpm: List[float],
                   support: str = "simple", supports: Optional[List[Support]] = None) -> list:
    """
    UDL sweep. Pass `supports` to use an explicit layout (e.g. overhang);
    otherwise `support` selects "simple" (pin-roller at the ends) or "cantilever".
    """
    if supports is None:
        if support == "cantilever":
            supports = [Support(0, "fixed")]
        else:
            supports = [Support(0, "pinned"), Support(L, "roller")]
    rows = []
    for w in w_list_kNpm:
        m = BeamModel(L=L, E=E, I=I, supports=list(supports), udls=[UDL(0, L, w * 1e3)])
        r = analyse(m, 300)
        rows.append({
            "w_kNpm": w,
            "M_max_kNm": r["M_max"] / 1e3,
            "V_max_kN": r["V_max"] / 1e3,
            "y_max_mm": r["y_max_mm"],
        })
    return rows
