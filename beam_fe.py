"""
Euler–Bernoulli finite-element solver for statically indeterminate beams
(e.g. propped cantilever, fixed–fixed, fixed–roller–roller).

Hermite cubic elements, 2 DOF per node (deflection v, rotation θ). Loads are
converted to consistent nodal loads, so UDL / UVL results are exact for the
element shape functions; point loads and couples are placed on nodes.

Sign convention (matches beam_solver):
  * V(x) is positive for a positive left reaction (V = dM/dx)
  * M(x) is positive for sagging
  * y(x) is positive upward (y'' = M/EI)
Internally v is positive downward, so v = -y.
"""
from __future__ import annotations

import numpy as np

from beam_solver import BeamModel

_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)


def is_indeterminate(model: BeamModel) -> bool:
    """True when the restraints exceed the 3 equilibrium equations of a 2D beam."""
    restraints = sum(2 if s.kind == "fixed" else 1
                     for s in model.supports if s.kind in ("pinned", "roller", "fixed"))
    return restraints > 2


def _mesh(model: BeamModel, n_base: int) -> np.ndarray:
    pts = set(np.linspace(0.0, model.L, n_base + 1).tolist())
    pts.add(0.0)
    pts.add(model.L)
    for s in model.supports:
        pts.add(float(s.x))
    for p in model.point_loads:
        pts.add(float(p.x))
    for m in model.moments:
        pts.add(float(m.x))
    for u in model.udls:
        pts.update([float(u.x1), float(u.x2)])
    for v in model.uvls:
        pts.update([float(v.x1), float(v.x2)])
    xs = np.array(sorted(p for p in pts if 0.0 <= p <= model.L))
    # drop near-duplicates created by float round-off
    keep = [xs[0]]
    for x in xs[1:]:
        if x - keep[-1] > 1e-9:
            keep.append(x)
    return np.array(keep)


def _q_at(model: BeamModel, x: float) -> float:
    """Distributed load intensity (N/m, downward positive) at position x."""
    q = 0.0
    for u in model.udls:
        if u.x1 - 1e-12 <= x <= u.x2 + 1e-12:
            q += u.w
    for v in model.uvls:
        if v.x1 - 1e-12 <= x <= v.x2 + 1e-12 and v.x2 > v.x1:
            t = (x - v.x1) / (v.x2 - v.x1)
            q += v.w1 + t * (v.w2 - v.w1)
    return q


def _hermite(l: float, s: float):
    """Shape functions N and their 2nd and 3rd derivatives wrt x (local)."""
    N = np.array([1 - 3 * s**2 + 2 * s**3,
                  l * (s - 2 * s**2 + s**3),
                  3 * s**2 - 2 * s**3,
                  l * (-s**2 + s**3)])
    d2 = np.array([(-6 + 12 * s) / l**2,
                   (-4 + 6 * s) / l,
                   (6 - 12 * s) / l**2,
                   (-2 + 6 * s) / l])
    d3 = np.array([12 / l**3, 6 / l**2, -12 / l**3, 6 / l**2])
    return N, d2, d3


def _solve(model: BeamModel, n_base: int = 60):
    xs = _mesh(model, n_base)
    nn = len(xs)
    ndof = 2 * nn
    EI = model.E * model.I
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    elems = []

    for e in range(nn - 1):
        x1, x2 = xs[e], xs[e + 1]
        l = x2 - x1
        k = EI / l**3 * np.array([
            [12, 6 * l, -12, 6 * l],
            [6 * l, 4 * l * l, -6 * l, 2 * l * l],
            [-12, -6 * l, 12, -6 * l],
            [6 * l, 2 * l * l, -6 * l, 4 * l * l]])
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(dofs, dofs)] += k
        # consistent nodal loads from distributed load (Gauss, exact for these shapes)
        fe = np.zeros(4)
        for gx, gw in zip(_GAUSS_X, _GAUSS_W):
            x = x1 + 0.5 * l * (gx + 1)
            s = (x - x1) / l
            N, _, _ = _hermite(l, s)
            fe += 0.5 * l * gw * N * _q_at(model, x)
        F[dofs] += fe
        elems.append((x1, l, dofs))

    # point loads (downward positive) and couples on nodes
    node_of = {round(float(x), 9): i for i, x in enumerate(xs)}
    for p in model.point_loads:
        i = node_of[round(float(p.x), 9)]
        F[2 * i] += p.P
    for m in model.moments:
        i = node_of[round(float(m.x), 9)]
        # couple M (sagging positive) acts on rotation DOF with sign so that M jumps by -M
        F[2 * i + 1] -= m.M  # sign chosen so the M jump is -M, as in beam_solver

    # restraints
    fixed_dofs = []
    for s in model.supports:
        i = node_of[round(float(s.x), 9)]
        if s.kind in ("pinned", "roller", "fixed"):
            fixed_dofs.append(2 * i)
        if s.kind == "fixed":
            fixed_dofs.append(2 * i + 1)
    fixed_dofs = sorted(set(fixed_dofs))
    free = [d for d in range(ndof) if d not in fixed_dofs]

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    r = K @ u - F  # reaction forces in displacement (downward) sense
    return xs, elems, u, r, fixed_dofs, node_of


def _interp(xs, elems, u, x: float, order: int):
    """order 0: v, 1: θ, 2: v'', 3: v''' at x (down-positive v)."""
    e = min(max(np.searchsorted(xs, x, side="right") - 1, 0), len(xs) - 2)
    x1, l, dofs = elems[e]
    s = (x - x1) / l
    N, d2, d3 = _hermite(l, s)
    ue = u[dofs]
    if order == 0:
        return N @ ue
    if order == 1:
        dN = np.array([(-6 * s + 6 * s**2) / l,
                       1 - 4 * s + 3 * s**2,
                       (6 * s - 6 * s**2) / l,
                       -2 * s + 3 * s**2])
        return dN @ ue
    if order == 2:
        return d2 @ ue
    return d3 @ ue


def shear_moment_fe(model: BeamModel, n: int = 500, n_base: int = 120) -> dict:
    xs, elems, u, r, fixed_dofs, node_of = _solve(model, n_base)
    EI = model.E * model.I
    x = np.linspace(0.0, model.L, n)
    V = np.zeros(n)
    M = np.zeros(n)
    for i, xi in enumerate(x):
        v2 = _interp(xs, elems, u, xi, 2)
        v3 = _interp(xs, elems, u, xi, 3)
        M[i] = -EI * v2   # sagging positive
        V[i] = -EI * v3   # = dM/dx

    reactions = {}
    for d in fixed_dofs:
        if d % 2 == 0:
            node_x = float(xs[d // 2])
            # r is in the downward-displacement sense; the upward reaction is -r
            reactions[node_x] = reactions.get(node_x, 0.0) - r[d]
    return {
        "x": x,
        "V": V,
        "M": M,
        "reactions": reactions,
        "reactions_raw": {f"R@{k}": v for k, v in reactions.items()},
        "V_max": float(np.max(np.abs(V))),
        "M_max": float(np.max(np.abs(M))),
        "is_cantilever": False,
    }


def deflection_fe(model: BeamModel, n: int = 500, n_base: int = 120) -> dict:
    xs, elems, u, _, _, _ = _solve(model, n_base)
    x = np.linspace(0.0, model.L, n)
    y = np.array([-_interp(xs, elems, u, xi, 0) for xi in x])  # up-positive
    ymax = float(np.max(np.abs(y)))
    return {"x": x, "y": y, "y_max": ymax, "y_max_mm": ymax * 1e3}


def redistribute_moments(sm: dict, model: BeamModel, beta: float) -> dict:
    """
    Moment redistribution for continuous beams (pin/roller interior supports).

    Each interior support's hogging moment is reduced by a fraction `beta` of its
    elastic value. The correction is a linear moment diagram in every span, equal to the
    change in end moments at the supports. Such a diagram carries no net load, so
    equilibrium is preserved: the reactions are updated consistently and their sum is unchanged.

    Only the moment, shear and reaction values change; deflection is not redistributed.
    Returns a new result dict; the input is not modified.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    vert = sorted(s.x for s in model.supports if s.kind in ("pinned", "roller"))
    interior = [s for s in vert[1:-1]]
    if beta == 0.0 or not interior:
        return sm
    x = sm["x"].copy()
    M = sm["M"].copy()
    V = sm["V"].copy()
    reactions = dict(sm["reactions"])

    # end-moment change at each interior support (positive reduces hogging)
    dM = {}
    for xs in interior:
        M_el = float(np.interp(xs, x, M))
        dM[xs] = -beta * M_el if M_el < 0 else 0.0  # adds sagging to relieve hogging

    # span boundaries: 0-points for the linear correction (zero at the end supports)
    nodes = vert
    dmom = {xs: dM.get(xs, 0.0) for xs in nodes}
    dV_span = {}
    corr = np.zeros_like(M)
    spans = list(zip(nodes[:-1], nodes[1:]))
    for k, (a, b) in enumerate(spans):
        l = b - a
        dV_span[(a, b)] = (dmom[b] - dmom[a]) / l
        last = k == len(spans) - 1
        sel = (x >= a - 1e-12) & ((x <= b + 1e-12) if last else (x < b - 1e-12))
        t = (x[sel] - a) / l
        corr[sel] += dmom[a] * (1 - t) + dmom[b] * t
    M_new = M + corr

    # shear: derivative of the correction
    dVx = np.zeros_like(V)
    for k, ((a, b), dv) in enumerate(dV_span.items()):
        last = k == len(spans) - 1
        sel = (x >= a - 1e-12) & ((x <= b + 1e-12) if last else (x < b - 1e-12))
        dVx[sel] = dv
    V_new = V + dVx

    # reactions: jump in the shear correction across each support
    new_reactions = {}
    for i, xs in enumerate(nodes):
        R = reactions.get(xs, 0.0)
        left = dV_span.get((nodes[i - 1], xs), 0.0) if i > 0 else 0.0
        right = dV_span.get((xs, nodes[i + 1]), 0.0) if i < len(nodes) - 1 else 0.0
        new_reactions[xs] = R + (right - left)

    return {**sm, "M": M_new, "V": V_new, "reactions": new_reactions,
            "V_max": float(np.max(np.abs(V_new))), "M_max": float(np.max(np.abs(M_new)))}


def equilibrium_error(sm: dict, model: BeamModel) -> float:
    """Sum of reactions minus total applied load, N (should be ~0 for any valid result)."""
    total = sum(p.P for p in model.point_loads)
    total += sum(u.w * (u.x2 - u.x1) for u in model.udls)
    total += sum(0.5 * (v.w1 + v.w2) * (v.x2 - v.x1) for v in model.uvls)
    return float(sum(sm["reactions"].values()) - total)
