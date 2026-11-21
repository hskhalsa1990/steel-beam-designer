"""Regenerate every image in this folder from the current solver (v2.3)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT
OUT.mkdir(exist_ok=True)

from beam_solver import BeamModel, PointLoad, UDL, UVL, analyse, shear_moment  # noqa: E402
from design import DesignSpec, run_design, support_layout  # noqa: E402
from steel_sections import SECTIONS, STEEL_GRADES, compare_sections, lateral_torsional, get_section  # noqa: E402

E_S275 = STEEL_GRADES["S275"]["E"] * 1e6
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def base_model(kind: str, L: float, loads, I: float, E: float = E_S275):
    return BeamModel(L=L, E=E, I=I, supports=support_layout(kind, L), point_loads=[
        p for p in loads if isinstance(p, PointLoad)], udls=[u for u in loads if isinstance(u, UDL)],
        uvls=[v for v in loads if isinstance(v, UVL)])


def sfd_bmd_defl(name, kind, L, loads, I, title):
    m = base_model(kind, L, loads, I)
    sm = shear_moment(m, 800)
    df = analyse(m, 800)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    x = sm["x"]
    ax[0].fill_between(x, sm["V"] / 1e3, color=BLUE, alpha=0.35); ax[0].plot(x, sm["V"] / 1e3, color=BLUE)
    ax[0].set_ylabel("V (kN)")
    ax[1].fill_between(x, sm["M"] / 1e3, color=RED, alpha=0.35); ax[1].plot(x, sm["M"] / 1e3, color=RED)
    ax[1].set_ylabel("M (kN·m)")
    ax[2].fill_between(x, df["y"] * 1e3, color=GREEN, alpha=0.35); ax[2].plot(x, df["y"] * 1e3, color=GREEN)
    ax[2].set_ylabel("δ (mm)"); ax[2].set_xlabel("x (m)")
    for a in ax:
        a.axhline(0, color=GREY, lw=0.6); a.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    save(fig, name)


def support_comparison():
    L, I, w = 6.0, get_section("ISMB 300").I_m4, 10e3
    kinds = ["simple", "propped", "fixed_fixed", "overhang"]
    labels = ["Simply supported", "Propped cantilever", "Fixed–fixed", "Overhang"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey="row")
    for j, (k, lab) in enumerate(zip(kinds, labels)):
        m = BeamModel(L=L, E=E_S275, I=I, supports=support_layout(k, L), udls=[UDL(0, L, w)])
        sm = shear_moment(m, 600)
        df = analyse(m, 600)
        axes[0, j].plot(sm["x"], sm["M"] / 1e3, color=RED); axes[0, j].set_title(lab)
        axes[0, j].fill_between(sm["x"], sm["M"] / 1e3, color=RED, alpha=0.25)
        axes[1, j].plot(df["x"], df["y"] * 1e3, color=GREEN)
        axes[1, j].fill_between(df["x"], df["y"] * 1e3, color=GREEN, alpha=0.25)
        axes[1, j].set_xlabel("x (m)")
        axes[0, j].set_ylabel("M (kN·m)" if j == 0 else "")
        axes[1, j].set_ylabel("δ (mm)" if j == 0 else "")
        for r in range(2):
            axes[r, j].axhline(0, color=GREY, lw=0.6); axes[r, j].grid(alpha=0.3)
    fig.suptitle("UDL 10 kN/m on 6 m, ISMB 300: moment (top) and deflection (bottom)")
    fig.tight_layout()
    save(fig, "support_type_comparison.png")


def load_case_gallery():
    L = 6.0
    cases = {
        "Point load": [PointLoad(3.0, 20e3)],
        "UDL": [UDL(0, L, 5e3)],
        "Trapezoid UVL": [UVL(0, L, 0, 8e3)],
        "Point moment": None,
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for j, (lab, loads) in enumerate(cases.items()):
        if loads is None:
            from beam_solver import PointMoment
            m = BeamModel(L=L, E=E_S275, I=1e-4, supports=support_layout("simple", L),
                          moments=[PointMoment(3.0, 10e3)])
        else:
            m = base_model("simple", L, loads, 1e-4)
        sm = shear_moment(m, 600)
        axes[0, j].plot(sm["x"], sm["V"] / 1e3, color=BLUE); axes[0, j].set_title(lab)
        axes[1, j].plot(sm["x"], sm["M"] / 1e3, color=RED)
        axes[1, j].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("V (kN)"); axes[1, 0].set_ylabel("M (kN·m)")
    for a in axes.flat:
        a.axhline(0, color=GREY, lw=0.6); a.grid(alpha=0.3)
    fig.suptitle("Load case gallery — simply supported 6 m (shear top, moment bottom)")
    fig.tight_layout()
    save(fig, "load_case_gallery.png")


def overhang_plot():
    L = 8.0
    sfd_bmd_defl("overhang_sfd_bmd.png", "overhang", L, [UDL(0, L, 10e3)], 1e-4, "Overhang, UDL 10 kN/m on 8 m")


def section_comparison():
    rows = compare_sections(120e3, 60e3, "S275")
    names = [r["section"] for r in rows]
    w = [r["weight_kg_m"] for r in rows]
    util = [r["util_b"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GREEN if r["safe"] else RED for r in rows]
    ax.bar(names, w, color=colors)
    ax.set_ylabel("Weight (kg/m)")
    ax.set_title("Sections for M = 120 kN·m, V = 60 kN (green = strength OK)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout(); save(fig, "section_comparison.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, util, color=colors); ax.axhline(1.0, ls="--", color="k")
    ax.set_ylabel("Bending utilisation"); ax.set_title("Bending utilisation by section (strength only)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout(); save(fig, "utilisation_bars.png")


def util_vs_weight():
    rows = compare_sections(120e3, 60e3, "S275")
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        ax.scatter(r["weight_kg_m"], r["util_b"], color=GREEN if r["safe"] else RED, s=50)
        ax.annotate(r["section"], (r["weight_kg_m"], r["util_b"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(1.0, ls="--", color="k"); ax.set_xlabel("Weight (kg/m)"); ax.set_ylabel("Bending utilisation")
    ax.set_title("Utilisation vs weight"); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "util_vs_weight.png")


def deflection_by_section():
    L = 6.0
    names = ["ISMB 150", "ISMB 250", "ISMB 300", "ISMB 400"]
    fig, ax = plt.subplots(figsize=(8, 5))
    w = np.linspace(1e3, 20e3, 15)
    for n in names:
        s = get_section(n)
        d = [analyse(BeamModel(L=L, E=E_S275, I=s.I_m4, supports=support_layout("simple", L),
                               udls=[UDL(0, L, q)]), 300)["y_max_mm"] for q in w]
        ax.plot(w / 1e3, d, marker="o", ms=3, label=n)
    ax.plot(w / 1e3, [L * 1e3 / 250] * len(w), "k--", label="L/250")
    ax.set_xlabel("UDL (kN/m)"); ax.set_ylabel("Max deflection (mm)")
    ax.set_title("Deflection by section, simply supported 6 m"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "deflection_by_section.png")


def response_vs_udl():
    L = 6.0
    sec = get_section("ISMB 300")
    w = np.linspace(1e3, 20e3, 25)
    M, V, D = [], [], []
    for q in w:
        r = analyse(BeamModel(L=L, E=E_S275, I=sec.I_m4, supports=support_layout("simple", L),
                              udls=[UDL(0, L, q)]), 300)
        M.append(r["M_max"] / 1e3); V.append(r["V_max"] / 1e3); D.append(r["y_max_mm"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(w / 1e3, M, color=RED, label="M_max (kN·m)")
    ax.plot(w / 1e3, V, color=BLUE, label="V_max (kN)")
    ax2 = ax.twinx(); ax2.plot(w / 1e3, D, color=GREEN, label="δ_max (mm)"); ax2.set_ylabel("δ (mm)")
    ax.set_xlabel("UDL (kN/m)"); ax.set_ylabel("M / V"); ax.grid(alpha=0.3)
    ax.legend(loc="upper left"); ax2.legend(loc="lower right")
    ax.set_title("Response vs UDL, ISMB 300, 6 m simply supported")
    fig.tight_layout(); save(fig, "response_vs_udl.png")


def serviceability_check():
    L = 6.0
    names = list(SECTIONS.keys())
    lim = L * 1e3 / 250
    d = [analyse(BeamModel(L=L, E=E_S275, I=SECTIONS[n].I_m4, supports=support_layout("simple", L),
                           udls=[UDL(0, L, 8e3)]), 300)["y_max_mm"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, d, color=[GREEN if v <= lim else RED for v in d])
    ax.axhline(lim, ls="--", color="k", label=f"L/250 = {lim:.1f} mm")
    ax.set_ylabel("δ max (mm)"); ax.set_title("Serviceability, UDL 8 kN/m on 6 m")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right"); ax.legend()
    fig.tight_layout(); save(fig, "serviceability_check.png")


def moment_heatmap():
    Ls = np.linspace(3, 10, 25)
    ws = np.linspace(2e3, 20e3, 25)
    Z = np.zeros((len(ws), len(Ls)))
    for i, q in enumerate(ws):
        for j, L in enumerate(Ls):
            Z[i, j] = analyse(BeamModel(L=L, E=E_S275, I=1e-4, supports=support_layout("simple", L),
                                        udls=[UDL(0, L, q)]), 120)["M_max"] / 1e3
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=[Ls[0], Ls[-1], ws[0] / 1e3, ws[-1] / 1e3],
                   cmap="viridis")
    fig.colorbar(im, label="M_max (kN·m)")
    ax.set_xlabel("Span L (m)"); ax.set_ylabel("UDL (kN/m)"); ax.set_title("Moment heatmap, simply supported")
    fig.tight_layout(); save(fig, "moment_heatmap.png")


def ltb_curve():
    sec = get_section("ISMB 300")
    Ls = np.linspace(1, 12, 60)
    chi = [lateral_torsional(sec, "S275", L)["chi_LT"] for L in Ls]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ls, chi, color=BLUE)
    ax.set_xlabel("Unrestrained length L_cr (m)"); ax.set_ylabel("χ_LT")
    ax.set_title("Lateral-torsional buckling reduction, ISMB 300, S275 (simplified)")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "ltb_curve.png")


def auto_recommend_summary():
    loads = [UDL(0, 6.0, 5e3, "L"), PointLoad(3.0, 20e3, "L")]
    spec = DesignSpec(L=6.0, supports=support_layout("simple", 6.0), loads=loads, grade="S275")
    res = run_design(spec)
    manual = run_design(DesignSpec(**{**spec.__dict__, "auto_select": False, "section_name": "ISMB 150"}))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["ISMB 150 (manual)", f"{res['section']} (auto)"]
    vals = [manual["checks"]["util_bending"], res["checks"]["util_bending"]]
    ax.barh(labels, vals, color=[RED if v > 1 else GREEN for v in vals])
    ax.axvline(1.0, ls="--", color="k")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center")
    ax.set_xlabel("Bending utilisation (incl. LTB)"); ax.set_title("Manual choice vs automatic selection")
    fig.tight_layout(); save(fig, "auto_recommend_summary.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["Manual ISMB 150", f"Auto {res['section']}"], [manual["y_max_mm"], res["y_max_mm"]],
           color=[RED if not manual["ok"] else GREEN, GREEN])
    ax.axhline(6000 / 250, ls="--", color="k", label="L/250")
    ax.set_ylabel("δ max (mm)"); ax.legend(); ax.set_title("Before / after auto-select")
    fig.tight_layout(); save(fig, "before_after_autoselect.png")


def section_capacity():
    names = list(SECTIONS.keys())
    Mdb = [lateral_torsional(SECTIONS[n], "S275", 6.0)["Mdb_kNm"] for n in names]
    Mp = [SECTIONS[n].Zxx * 1e-6 * 275e6 / 1.1 / 1e3 for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, Mp, 0.4, label="Plastic-style M_d (no LTB)", color=GREY)
    ax.bar(x + 0.2, Mdb, 0.4, label="M_db with LTB, L = 6 m", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Moment capacity (kN·m)"); ax.legend(); ax.set_title("Section moment capacity, S275")
    fig.tight_layout(); save(fig, "section_capacity_chart.png")


def workflow():
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    steps = ["Geometry &\nsupports", "Loads\n(D / L tags)", "Combinations\n(ULS envelope)",
             "Section check\n(σ, τ, LTB)", "Deflection\n(D + L)", "Auto-select\n& report"]
    for i, s in enumerate(steps):
        ax.add_patch(plt.Rectangle((i * 1.85, 0.3), 1.6, 1.2, fc="#e8eef5", ec="#4a6a8a"))
        ax.text(i * 1.85 + 0.8, 0.9, s, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=((i + 1) * 1.85, 0.9), xytext=(i * 1.85 + 1.6, 0.9),
                        arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.2, 11); ax.set_ylim(0, 1.8)
    ax.set_title("Design workflow")
    save(fig, "design_workflow.png")


def schematic_3d():
    L = 6.0
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot([0, L], [0, 0], [0, 0], color=GREY, lw=5, label="beam axis")
    ax.scatter([0, L], [0, 0], [0, 0], color=GREEN, s=60, label="supports")
    ax.plot([3, 3], [0, 0], [0, -1.0], color="orange", lw=3, label="point load")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("3D schematic: beam, supports and load"); ax.legend(fontsize=7)
    save(fig, "beam_3d_schematic.png")


def load_schematic():
    L = 6.0
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot([0, L], [0, 0], color="k", lw=6)
    ax.add_patch(plt.Polygon([[-0.2, -0.35], [0.2, -0.35], [0, 0]], color=GREEN))
    ax.add_patch(plt.Polygon([[L - 0.2, -0.35], [L + 0.2, -0.35], [L, 0]], color=GREEN))
    for x in np.linspace(0.5, L - 0.5, 6):
        ax.annotate("", xy=(x, 0.02), xytext=(x, 0.6), arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.plot([0, L], [0.75, 0.75], color=BLUE, lw=1)
    ax.plot([3, 3], [0, 0.9], color="orange", lw=0)
    ax.annotate("", xy=(3, 0.0), xytext=(3, 1.1), arrowprops=dict(arrowstyle="->", color="orange", lw=2))
    ax.text(3.1, 1.15, "P", color="orange")
    ax.set_xlim(-0.6, L + 0.6); ax.set_ylim(-0.6, 1.4); ax.axis("off")
    ax.set_title("Load and support schematic (UDL on top, point load at midspan)")
    save(fig, "load_support_schematic.png")


def main():
    sfd_bmd_defl("sfd_bmd_deflection.png", "simple", 6.0,
                 [UDL(0, 6.0, 5e3), PointLoad(3.0, 20e3)], get_section("ISMB 300").I_m4,
                 "Simply supported 6 m, UDL 5 kN/m + 20 kN midspan")
    overhang_plot()
    support_comparison()
    load_case_gallery()
    section_comparison()
    util_vs_weight()
    deflection_by_section()
    response_vs_udl()
    serviceability_check()
    moment_heatmap()
    ltb_curve()
    auto_recommend_summary()
    section_capacity()
    workflow()
    schematic_3d()
    load_schematic()


if __name__ == "__main__":
    main()
