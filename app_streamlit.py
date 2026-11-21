"""Beam Designer v2.6 — Streamlit dashboard. Analysis lives in design.py; units in units.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beam_solver import PointLoad, PointMoment, UDL, UVL, parametric_udl  # noqa: E402
from design import (  # noqa: E402
    DEFAULT_COMBOS, DesignSpec, LoadCombo, ltb_sensitivity, run_design, support_layout,
)
from report import build_report  # noqa: E402
from steel_sections import STEEL_GRADES, compare_sections, get_section, list_section_names  # noqa: E402
from units import Units  # noqa: E402

st.set_page_config(page_title="Beam Designer v2.6", layout="wide", page_icon="📐")
st.title("Beam Designer v2.6")
st.caption("SFD · BMD · Deflection · Load combinations · LTB · Redistribution · Auto-select · 3D · PDF")

CASES = ["L", "D"]

with st.sidebar:
    st.header("Units")
    force_u = st.selectbox("Force unit", ["kN", "N"], index=0)
    length_u = st.selectbox("Length unit", ["m", "mm"], index=0)
    U = Units(force_u, length_u)
    st.caption("Deflections are shown in mm and stresses in MPa.")
    ku = f"{force_u}_{length_u}"   # widget keys change with units so stale values are not reused

    st.header("Geometry & steel")
    L_disp = st.number_input(f"Span L ({U.L})", U.m_to_length(0.5), U.m_to_length(40.0),
                             U.m_to_length(6.0), U.m_to_length(0.5), key=f"L_{ku}")
    L = U.length_to_m(L_disp)
    support_cfg = st.selectbox("Supports", [
        "Simply supported (pin–roller)",
        "Cantilever (fixed left)",
        "Overhang (pin at 0.2L, roller at 0.85L)",
        "Propped cantilever (fixed left, roller right)",
        "Fixed–fixed",
        "Two-span continuous (pin, roller, roller)",
    ])
    kind = {
        "Simply": "simple", "Cantilever": "cantilever", "Propped": "propped",
        "Fixed–fixed": "fixed_fixed", "Two-span": "two_span",
    }.get(support_cfg.split(" ")[0], "overhang")
    supports = support_layout(kind, L)
    grade = st.selectbox("Steel grade", list(STEEL_GRADES.keys()), index=1)
    sec_name = st.selectbox("Section (manual)", list_section_names(), index=4)
    sec = get_section(sec_name)
    st.caption(f"Ixx={sec.Ixx:.0f} cm⁴ · Zxx={sec.Zxx:.1f} cm³ · {sec.weight} kg/m")
    defl_limit = st.selectbox("Deflection limit", ["L/250", "L/300", "L/360", "L/500"], index=0)
    defl_ratio = float(defl_limit.split("/")[1])
    lateral_disp = st.number_input(f"Unrestrained length for LTB ({U.L})", U.m_to_length(0.1),
                                   U.m_to_length(40.0), U.m_to_length(L), U.m_to_length(0.5),
                                   key=f"lat_{ku}", help="Distance between lateral restraints of the "
                                   "compression flange. Default = span (conservative).")
    lateral_m = U.length_to_m(lateral_disp)
    self_weight = st.checkbox("Include self-weight (D)", True)
    auto_sec = st.checkbox("Auto-pick lightest section passing all checks", True)
    redis_pct = st.slider("Moment redistribution at interior supports (%)", 0, 30, 0,
                          help="Continuous beams only. Reduces hogging moment at interior supports and "
                               "increases span moment. Requires sections capable of rotation (Class 1/2); "
                               "not verified here.")
    redistribution = redis_pct / 100.0

    st.header("Loads (↓ positive)")
    st.caption("Tag each load as Live (L) or Dead (D). Combinations are set in the Load combinations tab.")
    loads = []
    n_pt = int(st.number_input("Point loads", 0, 6, 1))
    for i in range(n_pt):
        a, b, c = st.columns([2, 2, 1.3])
        px_d = a.number_input(f"P{i+1} x ({U.L})", 0.0, L_disp, min(L_disp * 0.5, L_disp),
                              U.m_to_length(0.1), key=f"px{i}_{ku}")
        pv_d = b.number_input(f"P{i+1} ({U.F})", 0.0, U.N_to_force(500e3), U.N_to_force(20e3),
                              U.N_to_force(1e3), key=f"pv{i}_{ku}")
        pc = c.selectbox("Case", CASES, key=f"pc{i}_{ku}")
        loads.append(PointLoad(U.length_to_m(px_d), U.force_to_N(pv_d), pc))

    if st.checkbox("UDL", True):
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1])
        x1 = c1.number_input(f"x1 ({U.L})", 0.0, L_disp, 0.0, U.m_to_length(0.1), key=f"ux1_{ku}")
        x2 = c2.number_input(f"x2 ({U.L})", 0.0, L_disp, L_disp, U.m_to_length(0.1), key=f"ux2_{ku}")
        w = c3.number_input(f"w ({U.intensity})", 0.0, U.Npm_to_intensity(100e3), U.Npm_to_intensity(5e3),
                            U.Npm_to_intensity(500.0), key=f"uw_{ku}")
        uc = c4.selectbox("Case", CASES, key=f"uc_{ku}")
        if x2 > x1:
            loads.append(UDL(U.length_to_m(x1), U.length_to_m(x2), U.intensity_to_Npm(w), uc))

    if st.checkbox("UVL / trapezoid"):
        c1, c2, c3 = st.columns([1, 1, 1])
        vx1 = c1.number_input(f"UVL x1 ({U.L})", 0.0, L_disp, 0.0, key=f"vx1_{ku}")
        vx2 = c2.number_input(f"UVL x2 ({U.L})", 0.0, L_disp, L_disp, key=f"vx2_{ku}")
        vc = c3.selectbox("Case", CASES, key=f"vc_{ku}")
        w1 = st.number_input(f"w1 ({U.intensity})", 0.0, U.Npm_to_intensity(100e3), 0.0, key=f"w1_{ku}")
        w2 = st.number_input(f"w2 ({U.intensity})", 0.0, U.Npm_to_intensity(100e3),
                             U.Npm_to_intensity(8e3), key=f"w2_{ku}")
        if vx2 > vx1:
            loads.append(UVL(U.length_to_m(vx1), U.length_to_m(vx2),
                             U.intensity_to_Npm(w1), U.intensity_to_Npm(w2), vc))

    if st.checkbox("Point moment"):
        mx = st.number_input(f"M at x ({U.L})", 0.0, L_disp, L_disp / 2, key=f"mx_{ku}")
        mv = st.number_input(f"M ({U.moment})", U.Nm_to_moment(-500e3), U.Nm_to_moment(500e3),
                             U.Nm_to_moment(10e3), key=f"mv_{ku}")
        mc = st.selectbox("Case", CASES, key=f"mc_{ku}")
        loads.append(PointMoment(U.length_to_m(mx), U.moment_to_Nm(mv), mc))

combo_df = pd.DataFrame(
    [{"Combination": c.name, "D factor": c.factors.get("D", 0.0), "L factor": c.factors.get("L", 0.0)}
     for c in DEFAULT_COMBOS]
)
combo_df = st.session_state.get("combo_df", combo_df)


def _combos_from(df: pd.DataFrame) -> list:
    out = []
    for _, row in df.iterrows():
        name = str(row["Combination"]).strip()
        if name:
            out.append(LoadCombo(name, {"D": float(row["D factor"]), "L": float(row["L factor"])}))
    return out or list(DEFAULT_COMBOS)


spec = DesignSpec(
    L=L, supports=supports, loads=loads, section_name=sec_name, grade=grade,
    combos=_combos_from(combo_df), defl_ratio=defl_ratio, lateral_m=lateral_m,
    self_weight=self_weight, auto_select=auto_sec, redistribution=redistribution,
)
res = run_design(spec)
sec = get_section(res["section"])
chk = res["checks"]
svc = res["serviceability"]
ltb = chk.get("ltb") or {}

if res["status"] == "safe" and res["section"] != sec_name:
    st.info(f"Auto-selected **{res['section']}** (lightest section passing strength, LTB and deflection).")
elif res["status"] == "none_safe":
    st.error("No catalog section passes all checks for these loads. Showing the manual section.")

# KPIs (internal SI -> display units)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"Max |V| ({U.F})", f"{U.N_to_force(res['V_env_kN'] * 1e3):.2f}")
k2.metric(f"Max |M| ({U.moment})", f"{U.Nm_to_moment(res['M_env_kNm'] * 1e3):.2f}")
k3.metric("Max |δ| (mm)", f"{res['y_max_mm']:.2f}")
k4.metric("δ limit (mm)", f"{svc['allow_mm']:.1f}", delta="OK" if svc["ok"] else "FAIL",
          delta_color="normal" if svc["ok"] else "inverse")
k5.metric("σ/LTB util", f"{chk['util_bending']:.0%}", delta="OK" if chk["safe_bending"] else "FAIL",
          delta_color="normal" if chk["safe_bending"] else "inverse")
k6.metric("τ util", f"{chk['util_shear']:.0%}", delta="OK" if chk["safe_shear"] else "FAIL",
          delta_color="normal" if chk["safe_shear"] else "inverse")

if res["ok"]:
    st.success(f"**{res['section']}** · strength OK (incl. LTB) · serviceability OK ({defl_limit})")
elif chk["safe"] and not svc["ok"]:
    st.warning(f"**{res['section']}** strength OK but deflection exceeds {defl_limit}.")
else:
    st.error(f"**{res['section']}** fails the strength check — increase section, reduce load or restrain the beam.")

tabs = st.tabs(["SFD / BMD / δ", "3D Views", "Steel comparison", "Parametric",
                "Load combinations", "Summary, CSV & PDF"])
x_m = res["x"]
x_disp = np.array([U.m_to_length(v) for v in x_m])
V_disp = np.array([U.N_to_force(v) for v in res["V"]])
M_disp = np.array([U.Nm_to_moment(v) for v in res["M"]])
y_mm = res["y"] * 1e3

with tabs[0]:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Shear V ({U.F})", f"Moment M ({U.moment})", "Deflection δ (mm)"),
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=x_disp, y=V_disp, fill="tozeroy", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_disp, y=M_disp, fill="tozeroy", line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_disp, y=y_mm, fill="tozeroy", line=dict(color="#2ca02c")), row=3, col=1)
    fig.update_xaxes(title_text=f"x ({U.L})", row=3, col=1)
    fig.update_layout(height=800, showlegend=False, template="plotly_white",
                      title=f"SFD · BMD · Deflection — governing: {res['governing_moment_combo'] or '—'}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shear and moment show the governing factored combination; deflection uses service loads (D + L).")

    st.caption(f"Equilibrium check: sum of reactions − applied loads = "
               f"{res['max_equilibrium_error_N']:.3g} N (should be ~0).")
    if redistribution > 0:
        st.info(f"Moment redistribution {redis_pct}% applied at interior supports. Span moments rise as "
                "support moments fall; the envelope above includes both.")

with tabs[1]:
    scale = st.slider("Deflection scale", 50.0, 3000.0, 500.0)
    y_def = res["y"] * scale
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter3d(x=x_disp, y=np.zeros_like(x_disp), z=np.zeros_like(x_disp), mode="lines",
                                line=dict(color="gray", width=5), name="Undeformed"))
    fig3.add_trace(go.Scatter3d(x=x_disp, y=np.zeros_like(x_disp), z=y_def, mode="lines",
                                line=dict(color="crimson", width=7), name="Deformed"))
    for s_ in supports:
        fig3.add_trace(go.Scatter3d(x=[U.m_to_length(s_.x)], y=[0], z=[0], mode="markers",
                                    marker=dict(size=5, color="green"), name=s_.kind))
    fig3.update_layout(scene=dict(xaxis_title=f"x ({U.L})", yaxis_title="y", zaxis_title="scaled δ",
                                  aspectmode="manual", aspectratio=dict(x=2.2, y=0.25, z=0.45)),
                       height=500, title="3D axis + scaled deflection", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter3d(x=x_disp, y=V_disp, z=np.zeros_like(x_disp), mode="lines",
                                line=dict(width=5, color="#1f77b4"), name="V"))
    fig4.add_trace(go.Scatter3d(x=x_disp, y=np.zeros_like(x_disp), z=M_disp, mode="lines",
                                line=dict(width=5, color="#d62728"), name="M"))
    fig4.update_layout(scene=dict(xaxis_title=f"x ({U.L})", yaxis_title=f"V ({U.F})", zaxis_title=f"M ({U.moment})"),
                       height=460, title="3D Shear–Moment path", template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    h, b, tw, tf = sec.h / 1000, sec.b / 1000, sec.tw / 1000, sec.tf / 1000
    xs = np.linspace(0, L, 30)
    outline = np.array([
        [-b/2, -h/2], [b/2, -h/2], [b/2, -h/2+tf], [tw/2, -h/2+tf],
        [tw/2, h/2-tf], [b/2, h/2-tf], [b/2, h/2], [-b/2, h/2],
        [-b/2, h/2-tf], [-tw/2, h/2-tf], [-tw/2, -h/2+tf], [-b/2, -h/2+tf], [-b/2, -h/2]])
    fig5 = go.Figure()
    for i, xi in enumerate(xs[::3]):
        fig5.add_trace(go.Scatter3d(x=np.full(len(outline), xi), y=outline[:, 0], z=outline[:, 1],
                                    mode="lines", line=dict(color="#1f77b4", width=2),
                                    showlegend=i == 0, name=sec.name))
    fig5.update_layout(scene=dict(xaxis_title="x (m)", yaxis_title="width (m)", zaxis_title="depth (m)",
                                  aspectmode="data"),
                       height=480, title=f"3D I-section — {sec.name} (section drawn in m)",
                       template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

with tabs[2]:
    rows = compare_sections(res["M_env_kNm"] * 1e3, res["V_env_kN"] * 1e3, grade, Lcr_m=res["lateral_m"])
    st.dataframe(rows, use_container_width=True)
    st.write(f"Auto-select result: **{res['section']}** ({res['status']}). "
             f"Bending utilisation includes lateral-torsional buckling at L_cr = "
             f"{U.m_to_length(res['lateral_m']):.2f} {U.L}.")
    names = [r["section"] for r in rows]
    util = [r["util_b"] for r in rows]
    chi = [r["chi_LT"] for r in rows]
    weights = [r["weight_kg_m"] for r in rows]
    colors = ["#2ca02c" if r["safe"] else "#d62728" for r in rows]
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(go.Bar(x=names, y=util, marker_color=colors, text=[f"{u:.0%}" for u in util],
                               textposition="outside"))
        fig.add_hline(y=1.0, line_dash="dash")
        fig.update_layout(title="Bending utilisation incl. LTB", xaxis_tickangle=-35, height=400,
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = go.Figure(go.Bar(x=names, y=chi, marker_color="#17becf"))
        fig.update_layout(title="Lateral-torsional reduction χ_LT", xaxis_tickangle=-35, height=400,
                          yaxis=dict(range=[0, 1.05]), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure(go.Scatter(x=weights, y=util, mode="markers+text", text=names, textposition="top center",
                               marker=dict(size=11, color=colors)))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="Utilisation (incl. LTB) vs weight", xaxis_title="kg/m", yaxis_title="util",
                      height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Lateral restraint sensitivity — {res['section']}")
    st.caption("How the buckling check changes as the compression flange is restrained at larger spacings. "
               "The envelope moment is fixed; only L_cr changes.")
    max_len_d = st.slider(f"Max unbraced length to sweep ({U.L})", U.m_to_length(0.5),
                          U.m_to_length(max(4.0, 3.0 * L)), U.m_to_length(max(2.0 * L, 1.0)),
                          U.m_to_length(0.5), key=f"sweep_{ku}")
    max_len = U.length_to_m(max_len_d)
    lengths = np.linspace(max(0.25, max_len / 40), max_len, 40)
    sdf = pd.DataFrame(ltb_sensitivity(spec, res["section"], lengths))
    sdf["Lcr_disp"] = [U.m_to_length(v) for v in sdf["Lcr_m"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sdf["Lcr_disp"], y=sdf["util_LTB"], mode="lines+markers", name="util (LTB)",
                             line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=sdf["Lcr_disp"], y=sdf["chi_LT"], mode="lines", name="χ_LT",
                             yaxis="y2", line=dict(color="#17becf", dash="dot")))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="LTB utilisation and χ_LT vs unbraced length", xaxis_title=f"L_cr ({U.L})",
                      yaxis=dict(title="utilisation"),
                      yaxis2=dict(title="χ_LT", overlaying="y", side="right", range=[0, 1.05]),
                      height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    limit_rows = sdf[~sdf["ok"]]
    if len(limit_rows):
        st.warning(f"LTB fails for L_cr ≥ about {limit_rows['Lcr_disp'].iloc[0]:.2f} {U.L} with this section "
                   "and envelope moment.")
    else:
        st.success("LTB passes across the whole swept range for this section.")
    st.dataframe(sdf.drop(columns=["Lcr_m"]).round(3), use_container_width=True)

with tabs[3]:
    st.subheader("Parametric study — vary UDL intensity")
    w_max_d = st.slider(f"Max UDL ({U.intensity}) for sweep", U.Npm_to_intensity(2e3),
                        U.Npm_to_intensity(30e3), U.Npm_to_intensity(15e3), U.Npm_to_intensity(500.0),
                        key=f"wmax_{ku}")
    w_list_disp = list(np.linspace(U.Npm_to_intensity(1e3), w_max_d, 10))
    w_list_kNpm = [U.intensity_to_Npm(w) / 1e3 for w in w_list_disp]  # parametric_udl takes kN/m
    E_pa = STEEL_GRADES[grade]["E"] * 1e6
    prow = parametric_udl(L, E_pa, sec.I_m4, w_list_kNpm, supports=supports)
    pdf_rows = pd.DataFrame({
        f"w ({U.intensity})": [U.Npm_to_intensity(r["w_kNpm"] * 1e3) for r in prow],
        f"Mmax ({U.moment})": [U.Nm_to_moment(r["M_max_kNm"] * 1e3) for r in prow],
        f"Vmax ({U.F})": [U.N_to_force(r["V_max_kN"] * 1e3) for r in prow],
        "δmax (mm)": [r["y_max_mm"] for r in prow],
    })
    st.dataframe(pdf_rows, use_container_width=True)
    wcol = pdf_rows.columns[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pdf_rows[wcol], y=pdf_rows.iloc[:, 1], mode="lines+markers", name="Mmax"))
    fig.add_trace(go.Scatter(x=pdf_rows[wcol], y=pdf_rows.iloc[:, 2], mode="lines+markers", name="Vmax"))
    fig.add_trace(go.Scatter(x=pdf_rows[wcol], y=pdf_rows["δmax (mm)"], mode="lines+markers",
                             name="δmax mm", yaxis="y2"))
    fig.update_layout(title=f"Response vs UDL — {res['section']}", xaxis_title=f"w ({U.intensity})",
                      yaxis=dict(title="M / V"), yaxis2=dict(title="δ mm", overlaying="y", side="right"),
                      height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Load combinations")
    st.caption("Strength uses the envelope of the rows below (factors on each load case). "
               "Deflection always uses unfactored D + L. Edit, add or delete rows.")
    edited = st.data_editor(combo_df, num_rows="dynamic", key="combo_editor", use_container_width=True)
    st.session_state["combo_df"] = edited
    pc = pd.DataFrame(res["per_combo"]).copy()
    pc["Max |M| (" + U.moment + ")"] = [U.Nm_to_moment(v * 1e3) for v in pc["M_kNm"]]
    pc["Max |V| (" + U.F + ")"] = [U.N_to_force(v * 1e3) for v in pc["V_kN"]]
    st.dataframe(pc[["combo", "Max |M| (" + U.moment + ")", "Max |V| (" + U.F + ")", "eq_err_N"]],
                 use_container_width=True)
    st.write(f"Governing moment: **{res['governing_moment_combo'] or '—'}** · "
             f"governing shear: **{res['governing_shear_combo'] or '—'}**")

with tabs[5]:
    st.subheader("Reactions")
    for xs_, R in res["reactions"].items():
        st.write(f"· Support at x = {U.m_to_length(xs_):.3f} {U.L} → "
                 f"**R = {U.N_to_force(R):.3f} {U.F}** (governing combination)")

    st.subheader("Design summary")
    summary = {
        "span": f"{L_disp:g} {U.L}", "force_unit": U.F, "length_unit": U.L,
        "support": support_cfg, "grade": grade,
        "section": res["section"], "status": res["status"],
        "governing_combo": res["governing_moment_combo"],
        f"M_env_{U.moment}": round(U.Nm_to_moment(res["M_env_kNm"] * 1e3), 3),
        f"V_env_{U.F}": round(U.N_to_force(res["V_env_kN"] * 1e3), 3),
        "y_max_mm": round(res["y_max_mm"], 3), "defl_limit": defl_limit, "defl_ok": svc["ok"],
        "redistribution_pct": redis_pct,
        "lateral_restraint": f"{lateral_disp:g} {U.L}", "chi_LT": round(ltb.get("chi_LT", 1.0), 4),
        "sigma_MPa": round(chk["sigma_MPa"], 2), "tau_MPa": round(chk["tau_MPa"], 2),
        "util_b": round(chk["util_bending"], 3), "util_s": round(chk["util_shear"], 3),
        "equilibrium_error_N": float(f"{res['max_equilibrium_error_N']:.3g}"),
        "overall_ok": res["ok"],
    }
    st.json(summary)

    df_x = pd.DataFrame({
        f"x_{U.L}": x_disp, f"V_{U.F}": V_disp, f"M_{U.moment}": M_disp, "y_mm": y_mm,
    })
    st.download_button("Download V/M/δ CSV", df_x.to_csv(index=False),
                       file_name="beam_VM_deflection.csv", mime="text/csv")
    st.download_button("Download summary JSON", pd.Series(summary).to_json(),
                       file_name="beam_summary.json", mime="application/json")
    st.download_button("Download PDF design report", build_report(res, spec),
                       file_name="beam_design_report.pdf", mime="application/pdf")
    st.caption("The PDF report uses SI units (kN, kN·m, m).")
