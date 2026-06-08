"""
examples/examples.py
====================
Worked examples from the paper, reproducing all cases A–E of
Theorem 1 (edit distance to full flat-foldability).

Run with:
    python examples/examples.py

Outputs are printed to stdout.  If matplotlib is installed, crease-pattern
diagrams are saved to outputs/images/.
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

from cdsvcp import SVCP

from cdmsvcp import (
    LSVCP,
    M,
    V,
    edit_distance,
    is_fully_flat_foldable,
    maekawa_deficit,
    signed_delta,
    repair,
    lsvcp_insert,
    lsvcp_flip,
    kawasaki_deficit,
    kawasaki_residual,
    is_flat_foldable,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def deg(x) -> Fraction:
    return Fraction(x, 180)


def fmt(f: Fraction) -> str:
    return f"{float(f)*180:.4g}°"


def fmt_labels(labels) -> str:
    return "(" + ", ".join(labels) + ")"


def show(lc: LSVCP, label: str = ""):
    tag = f"  [{label}]" if label else ""
    angles_str = ", ".join(fmt(a) for a in lc.angles)
    m = len(lc)
    kaw = kawasaki_residual(lc.svcp)
    foldable = "✓ fully flat-foldable" if is_fully_flat_foldable(lc) else "✗"
    kap_str = ""
    nu_str = ""
    if m % 2 == 0:
        kap = kawasaki_deficit(lc.svcp)
        nu  = maekawa_deficit(lc)
        kap_str = f"  κ={fmt(kap)}"
        nu_str  = f"  ν={nu}"
    delta_str = f"  δ={signed_delta(lc)}"
    print(f"  C = ({angles_str}){tag}")
    print(f"  μ = {fmt_labels(lc.labels)}   m={m}{kap_str}{delta_str}{nu_str}   {foldable}")


def show_ops(ops):
    if not ops:
        print("  (no operations)")
        return
    for i, op in enumerate(ops, 1):
        if op.kind == 'insert':
            print(
                f"  Op {i}: insert at pos {op.pos}, "
                f"s={op.param}π={float(op.param)*180:.4g}°, label={op.label}"
            )
        else:
            print(f"  Op {i}: flip at pos {op.pos}")


def section(title: str):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def subsection(title: str):
    print()
    print(f"  --- {title} ---")


def lsvcp(degrees, labels) -> LSVCP:
    return LSVCP(SVCP(tuple(deg(d) for d in degrees)), tuple(labels))


# ---------------------------------------------------------------------------
# Case A
# ---------------------------------------------------------------------------

def example_A():
    section("Case A:  m even, κ = 0, ν = 0  →  distance = 0")
    lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
    show(lc, "input")
    d = edit_distance(lc)
    print(f"\n  d_MV = {d}  (already fully flat-foldable)")


# ---------------------------------------------------------------------------
# Case B
# ---------------------------------------------------------------------------

def example_B():
    section("Case B:  m even, κ = 0, ν ≠ 0  →  distance = |ν|/2")

    subsection("ν = 2  (δ = 4, all mountains)")
    lc = lsvcp([90, 90, 90, 90], [M, M, M, M])
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}")
    show_ops(ops)
    print()
    show(lc_star, "after")

    subsection("ν = −2  (δ = 0, equal M and V)")
    lc = lsvcp([90, 90, 90, 90], [M, M, V, V])
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}")
    show_ops(ops)
    print()
    show(lc_star, "after")

    subsection("ν = 4  (ten-sector pattern)")
    a = deg(36)
    lc = LSVCP(SVCP((a,) * 10), (M,) * 8 + (V,) * 2)
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}")
    show_ops(ops)
    print()
    show(lc_star, "after")


# ---------------------------------------------------------------------------
# Case C
# ---------------------------------------------------------------------------

def example_C():
    section("Case C:  m even, κ ≠ 0, |ν| ≤ 2  →  distance = 2")

    subsection("ν = 0  (δ = 2)")
    lc = lsvcp([120, 80, 100, 60], [M, M, M, V])
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}")
    show_ops(ops)
    print()
    show(lc_star, "after")

    subsection("ν = 2  (δ = 4, all mountains)")
    lc = lsvcp([120, 80, 100, 60], [M, M, M, M])
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}")
    show_ops(ops)
    print()
    show(lc_star, "after")

    subsection("ν = −2  (δ = 0, alternating M/V)")
    lc = lsvcp([120, 80, 100, 60], [M, V, M, V])
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}")
    show_ops(ops)
    print()
    show(lc_star, "after")


# ---------------------------------------------------------------------------
# Case D
# ---------------------------------------------------------------------------

def example_D():
    section("Case D:  m even, κ ≠ 0, ν ≥ 4  →  distance = ν/2 + 1")

    subsection("ν = 4  (six-sector pattern)")
    lc = lsvcp([80, 60, 70, 50, 60, 40], [M]*6)
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}  (= ν/2 + 1 = 4/2 + 1 = 3)")
    show_ops(ops)
    print()
    show(lc_star, "after")

    subsection("ν = 6  (eight-sector pattern)")
    lc = lsvcp([60, 40, 50, 40, 60, 40, 50, 20], [M]*8)
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}  (= ν/2 + 1 = 6/2 + 1 = 4)")
    show_ops(ops)
    print()
    show(lc_star, "after")


# ---------------------------------------------------------------------------
# Case E
# ---------------------------------------------------------------------------

def example_E():
    section("Case E:  m odd  →  distance = max(1, (|δ|−1)/2)")

    subsection("|δ| = 1  (three-sector, one insertion)")
    lc = lsvcp([120, 120, 120], [M, M, V])
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}  (= max(1, 0) = 1)")
    show_ops(ops)
    print()
    show(lc_star, "after")

    subsection("|δ| = 3  (three-sector, all mountains)")
    lc = lsvcp([120, 120, 120], [M, M, M])
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}  (= max(1, 1) = 1)")
    show_ops(ops)
    print()
    show(lc_star, "after")

    subsection("|δ| = 5  (five-sector, all mountains)")
    lc = lsvcp([72, 72, 72, 72, 72], [M]*5)
    show(lc, "before")
    d = edit_distance(lc)
    lc_star, ops = repair(lc)
    print(f"\n  d_MV = {d}  (= max(1, 2) = 2)")
    show_ops(ops)
    print()
    show(lc_star, "after")


# ---------------------------------------------------------------------------
# Optional: matplotlib diagrams
# ---------------------------------------------------------------------------

def _save_diagram(lc: LSVCP, lc_star: LSVCP, filename: str, title: str):
    """
    Save a before/after crease-pattern diagram to outputs/images/.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), subplot_kw={"aspect": "equal"})
    fig.suptitle(title, fontsize=11)

    for ax, lc_plot, subtitle in [
        (axes[0], lc,      "Before"),
        (axes[1], lc_star, "After  ✓"),
    ]:
        ax.set_title(subtitle, fontsize=9)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")

        cum = 0.0
        for a, lbl in zip(lc_plot.angles, lc_plot.labels):
            angle_deg = float(a) * 180
            mid = cum + angle_deg / 2
            end = cum + angle_deg
            mid_rad = math.radians(mid)
            end_rad = math.radians(end)
            color = "#b03030" if lbl == M else "#3060b0"
            ax.plot(
                [0, math.cos(end_rad)],
                [0, math.sin(end_rad)],
                color=color, linewidth=1.8,
            )
            ax.text(
                1.25 * math.cos(end_rad),
                1.25 * math.sin(end_rad),
                lbl, ha="center", va="center", fontsize=7, color=color,
            )
            ax.text(
                0.65 * math.cos(mid_rad),
                0.65 * math.sin(mid_rad),
                f"{angle_deg:.3g}°",
                ha="center", va="center", fontsize=6.5, color="#444",
            )
            cum = end

        if len(lc_plot) % 2 == 0:
            kap = kawasaki_deficit(lc_plot.svcp)
            nu  = maekawa_deficit(lc_plot)
            info = f"κ={float(kap)*180:.3g}°  ν={nu}"
        else:
            info = f"m={len(lc_plot)} (odd)  δ={signed_delta(lc_plot)}"
        ax.text(0, -1.45, info, ha="center", va="center", fontsize=8)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "images")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [diagram saved to {path}]")


def diagrams():
    cases = [
        (lsvcp([90, 90, 90, 90], [M, M, M, M]),          "case_B_nu2.png",   "Case B (ν=2): one flip"),
        (lsvcp([120, 80, 100, 60], [M, M, M, V]),         "case_C_nu0.png",   "Case C (ν=0): two insertions"),
        (lsvcp([80, 60, 70, 50, 60, 40], [M]*6),          "case_D_nu4.png",   "Case D (ν=4): two insertions + one flip"),
        (lsvcp([72, 72, 72, 72, 72], [M]*5),              "case_E_delta5.png","Case E (|δ|=5): one insertion + one flip"),
    ]
    for lc_in, fname, title in cases:
        lc_out, _ = repair(lc_in)
        _save_diagram(lc_in, lc_out, fname, title)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_A()
    example_B()
    example_C()
    example_D()
    example_E()
    diagrams()
    print()
    print("All examples completed.")
