"""PDF design report built from the dict returned by design.run_design()."""
from __future__ import annotations

import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from design import describe_loads, ltb_sensitivity  # noqa: E402


def _diagram_png(res: dict) -> bytes:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6.5))
    x = res["x"]
    axes[0].plot(x, res["V"] / 1e3, color="#1f77b4"); axes[0].set_ylabel("V (kN)")
    axes[1].plot(x, res["M"] / 1e3, color="#d62728"); axes[1].set_ylabel("M (kN·m)")
    axes[2].plot(x, res["y"] * 1e3, color="#2ca02c"); axes[2].set_ylabel("δ (mm)")
    axes[2].set_xlabel("x (m)")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.6); ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build_report(res: dict, spec, title: str = "Beam design report") -> bytes:
    """Return the PDF as bytes. `spec` is the DesignSpec used to produce `res`."""
    styles = getSampleStyleSheet()
    chk = res["checks"]
    svc = res["serviceability"]
    verdict = "PASS" if res["ok"] else "FAIL"
    ltb = chk.get("ltb") or {}

    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Date: {date.today().isoformat()} · Section: <b>{res['section']}</b> "
                  f"({res['status']}) · Overall: <b>{verdict}</b>", styles["Normal"]),
        Paragraph("Simplified checks for educational use. Not a full IS 800 / AISC design.",
                  styles["Italic"]),
        Spacer(1, 6 * mm),
        Paragraph("Geometry and material", styles["Heading2"]),
        _table([
            ["Span L", f"{spec.L:.3f} m"],
            ["Supports", ", ".join(f"{s.kind}@{s.x:.2f} m" for s in spec.supports)],
            ["Steel grade", spec.grade],
            ["Unrestrained length (LTB)", f"{res['lateral_m']:.3f} m"],
            ["Deflection limit", f"L/{spec.defl_ratio:g} = {svc['allow_mm']:.1f} mm"],
        ], col_widths=[55 * mm, 110 * mm]),
        Spacer(1, 5 * mm),
    ]
    story += [Paragraph("Load cases", styles["Heading2"])]
    load_rows = [["Case", "Type", "Location", "Magnitude"]]
    for row in describe_loads(spec):
        load_rows.append([row["case"], row["type"], row["where"], row["magnitude"]])
    story += [
        _table(load_rows, col_widths=[18 * mm, 40 * mm, 55 * mm, 52 * mm]) if len(load_rows) > 1
        else Paragraph("No loads defined.", styles["Normal"]),
        Spacer(1, 5 * mm),
        Paragraph("Load combinations", styles["Heading2"]),
    ]
    combo_rows = [["Combination", "Max |M| (kN·m)", "Max |V| (kN)"]]
    for row in res["per_combo"]:
        combo_rows.append([row["combo"], f"{row['M_kNm']:.2f}", f"{row['V_kN']:.2f}"])
    sens_rows = [["Lcr (m)", "Mcr (kN·m)", "χ_LT", "Mdb (kN·m)", "util (LTB)", "Result"]]
    Lcr0 = res["lateral_m"]
    for s_ in ltb_sensitivity(spec, res["section"], [0.5 * Lcr0, Lcr0, 2 * Lcr0]):
        sens_rows.append([f"{s_['Lcr_m']:.2f}", f"{s_['Mcr_kNm']:.1f}", f"{s_['chi_LT']:.3f}",
                          f"{s_['Mdb_kNm']:.1f}", f"{s_['util_LTB']:.0%}", "OK" if s_["ok"] else "FAIL"])
    story += [
        _table(combo_rows, col_widths=[70 * mm, 45 * mm, 45 * mm]),
        Paragraph(f"Governing moment: {res['governing_moment_combo'] or '—'}", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph("Lateral restraint sensitivity (selected section, envelope moment)", styles["Heading3"]),
        _table(sens_rows, col_widths=[24 * mm, 30 * mm, 22 * mm, 30 * mm, 28 * mm, 22 * mm]),
        Paragraph("Shown at half, equal to, and twice the design unbraced length; the dashboard sweep shows the full range.",
                  styles["Italic"]),
        Spacer(1, 5 * mm),
        Paragraph("Checks", styles["Heading2"]),
        _table([
            ["Check", "Value", "Limit", "Utilisation", "Result"],
            ["Bending σ (MPa)", f"{chk['sigma_MPa']:.1f}", f"{chk['sigma_all_MPa']:.1f}",
             f"{chk['util_bending']:.0%}", "OK" if chk["safe_bending"] else "FAIL"],
            ["Shear τ (MPa)", f"{chk['tau_MPa']:.1f}", f"{chk['tau_all_MPa']:.1f}",
             f"{chk['util_shear']:.0%}", "OK" if chk["safe_shear"] else "FAIL"],
            ["Lateral-torsional", f"M_cr={ltb.get('Mcr_kNm', 0):.1f} kN·m",
             f"M_db={ltb.get('Mdb_kNm', 0):.1f} kN·m", f"χ_LT={ltb.get('chi_LT', 1):.3f}",
             "OK" if chk["safe_bending"] else "FAIL"],
            ["Deflection δmax (mm)", f"{res['y_max_mm']:.2f}", f"{svc['allow_mm']:.1f}",
             f"{res['y_max_mm'] / svc['allow_mm']:.0%}", "OK" if svc["ok"] else "FAIL"],
        ], col_widths=[42 * mm, 36 * mm, 36 * mm, 28 * mm, 22 * mm]),
        Spacer(1, 5 * mm),
        Paragraph("Reactions", styles["Heading2"]),
        _table([["Support x (m)", "Reaction (kN)"]] +
               [[f"{xs:.3f}", f"{R / 1e3:.2f}"] for xs, R in res["reactions"].items()],
               col_widths=[55 * mm, 55 * mm]),
        Spacer(1, 5 * mm),
        PageBreak(),
        Paragraph("Diagrams (governing combination; deflection under service loads)",
                  styles["Heading2"]),
        Image(io.BytesIO(_diagram_png(res)), width=170 * mm, height=158 * mm),
    ]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=title)
    doc.build(story)
    return buf.getvalue()
