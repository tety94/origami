"""
examples/examples.py
====================
Worked examples from the paper, reproducing Examples 2.1–2.4.

Run with:
    python examples/examples.py

Outputs are printed to stdout.  If matplotlib is installed, simple
crease-pattern diagrams are saved to outputs/images/.
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cdsvcp import (
    SVCP,
    edit_distance,
    is_flat_foldable,
    kawasaki_deficit,
    kawasaki_residual,
    repair,
    single_insert,
    two_insert,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def deg(x) -> Fraction:
    return Fraction(x, 180)


def fmt(f: Fraction) -> str:
    return f"{float(f)*180:.4g}°"


def show(C: SVCP, label: str = ""):
    tag = f"  [{label}]" if label else ""
    angles_str = ", ".join(fmt(a) for a in C.angles)
    kaw = kawasaki_residual(C)
    foldable = "✓ flat-foldable" if is_flat_foldable(C) else f"✗ κ/K={fmt(kaw)}"
    print(f"  C = ({angles_str}){tag}   |C|={len(C)}   {foldable}")


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Example 2.1 — dC = 1 (odd crease count)
# ---------------------------------------------------------------------------

def example_d1():
    section("Example 2.1  —  dC = 1  (odd crease count)")
    C = SVCP((deg(120), deg(100), deg(140)))
    print(f"Input:")
    show(C, "before")
    print(f"  edit_distance = {edit_distance(C)}")

    C_star, op = single_insert(C)
    print(f"\nOperation: INSERT at position {op.pos}, s = {fmt(op.param)}")
    show(C_star, "C* ∈ F")
    assert is_flat_foldable(C_star)
    print("  ✓ Verified flat-foldable.")


# ---------------------------------------------------------------------------
# Example 2.2 — dC = 2, Case 2  (n ≥ 2, M > κ)
# ---------------------------------------------------------------------------

def example_case2():
    section("Example 2.2  —  dC = 2, Case 2  (n≥2, M>κ)")
    C = SVCP((deg(120), deg(80), deg(100), deg(60)))
    print("Input:")
    show(C, "before")
    kap = kawasaki_deficit(C)
    M = max(C.angles[i] for i in range(0, len(C), 2))
    print(f"  κ = {fmt(kap)},  M = {fmt(M)},  M > κ → Case 2")

    C_star, ops = two_insert(C)
    print(f"\nOperations:")
    for k, op in enumerate(ops, 1):
        print(f"  {k}. {op.kind.upper()} at pos {op.pos}"
              + (f", s={fmt(op.param)}" if op.param else ""))
    show(C_star, "C* ∈ F")
    assert is_flat_foldable(C_star)
    print("  ✓ Verified flat-foldable.")


# ---------------------------------------------------------------------------
# Example 2.3 — dC = 2, Case 3  (n ≥ 3, M = κ)
# ---------------------------------------------------------------------------

def example_case3():
    section("Example 2.3  —  dC = 2, Case 3  (n≥3, M=κ)")
    C = SVCP((deg(90), deg(40), deg(90), deg(30), deg(90), deg(20)))
    print("Input:")
    show(C, "before")
    kap = kawasaki_deficit(C)
    M = max(C.angles[i] for i in range(0, len(C), 2))
    print(f"  κ = {fmt(kap)},  M = {fmt(M)},  M = κ → Case 3")

    C_star, ops = two_insert(C)
    print(f"\nOperations:")
    for k, op in enumerate(ops, 1):
        print(f"  {k}. {op.kind.upper()} at pos {op.pos}"
              + (f", s={fmt(op.param)}" if op.param else ""))
    show(C_star, "C* ∈ F")
    assert is_flat_foldable(C_star)
    print("  ✓ Verified flat-foldable.")


# ---------------------------------------------------------------------------
# Example 2.4 — dC = 2, Case 2b  (n ≥ 3, M < κ)
# ---------------------------------------------------------------------------

def example_case2b():
    section("Example 2.4  —  dC = 2, Case 2b  (n≥3, M<κ)")
    C = SVCP((deg(100), deg(30), deg(100), deg(20), deg(100), deg(10)))
    print("Input:")
    show(C, "before")
    kap = kawasaki_deficit(C)
    M = max(C.angles[i] for i in range(0, len(C), 2))
    print(f"  κ = {fmt(kap)},  M = {fmt(M)},  M < κ → Case 2b")

    C_star, ops = two_insert(C)
    print(f"\nOperations:")
    for k, op in enumerate(ops, 1):
        print(f"  {k}. {op.kind.upper()} at pos {op.pos}"
              + (f", s={fmt(op.param)}" if op.param else ""))
    show(C_star, "C* ∈ F")
    assert is_flat_foldable(C_star)
    print("  ✓ Verified flat-foldable.")


# ---------------------------------------------------------------------------
# Case 1 (n = 1) — not in the main examples but worth showing
# ---------------------------------------------------------------------------

def example_case1():
    section("Bonus: Case 1  (n=1)")
    C = SVCP((deg(210), deg(150)))
    print("Input:")
    show(C, "before")
    kap = kawasaki_deficit(C)
    print(f"  κ = {fmt(kap)},  n=1 → Case 1")

    C_star, ops = two_insert(C)
    print(f"\nOperations:")
    for k, op in enumerate(ops, 1):
        print(f"  {k}. {op.kind.upper()} at pos {op.pos}"
              + (f", s={fmt(op.param)}" if op.param else ""))
    show(C_star, "C* ∈ F")
    assert is_flat_foldable(C_star)
    print("  ✓ Verified flat-foldable.")


# ---------------------------------------------------------------------------
# Optional: matplotlib visualisation
# ---------------------------------------------------------------------------

def _try_plot(C_before: SVCP, C_after: SVCP, title: str, filename: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return   # matplotlib not available, skip silently

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5),
                              subplot_kw=dict(projection='polar'))

    def draw_svcp(ax, C: SVCP, label: str):
        angles_rad = [float(a) * math.pi for a in C.angles]
        cumulative = [0.0]
        for a in angles_rad:
            cumulative.append(cumulative[-1] + a)
        colors = ['#5b9bd5', '#ed7d31']   # blue/orange alternating
        for i, (start, end) in enumerate(zip(cumulative, cumulative[1:])):
            theta = np.linspace(start, end, 200)
            ax.fill_between(theta, 0, 1,
                            color=colors[i % 2], alpha=0.55)
        for angle in cumulative[:-1]:
            ax.plot([angle, angle], [0, 1], 'k-', lw=1.2)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(label, pad=12, fontsize=10)

    draw_svcp(axes[0], C_before, f"Before  (|C|={len(C_before)})")
    draw_svcp(axes[1], C_after,  f"C* ∈ F  (|C*|={len(C_after)})")
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "images")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  (diagram saved to {path})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_d1()
    _try_plot(
        SVCP((Fraction(120,180), Fraction(100,180), Fraction(140,180))),
        repair(SVCP((Fraction(120,180), Fraction(100,180), Fraction(140,180))))[0],
        "Example 2.1 — dC = 1", "example_d1.png",
    )

    example_case2()
    _try_plot(
        SVCP((deg(120), deg(80), deg(100), deg(60))),
        repair(SVCP((deg(120), deg(80), deg(100), deg(60))))[0],
        "Example 2.2 — Case 2 (dC=2)", "example_case2.png",
    )

    example_case3()
    _try_plot(
        SVCP((deg(90), deg(40), deg(90), deg(30), deg(90), deg(20))),
        repair(SVCP((deg(90), deg(40), deg(90), deg(30), deg(90), deg(20))))[0],
        "Example 2.3 — Case 3 (dC=2)", "example_case3.png",
    )

    example_case2b()
    _try_plot(
        SVCP((deg(100), deg(30), deg(100), deg(20), deg(100), deg(10))),
        repair(SVCP((deg(100), deg(30), deg(100), deg(20), deg(100), deg(10))))[0],
        "Example 2.4 — Case 2b (dC=2)", "example_case2b.png",
    )

    example_case1()

    print()
    print("All examples completed successfully.")
