"""
example_n3.py – Paper Examples 3.2 and 5.1 (n = 3).

Two sub-cases:
  A) C = (70°, 50°, 80°, 60°, 90°, 10°)  –  L² feasible (paper Example 3.2).
  B) C = (150°, 10°, 100°, 60°, 30°, 10°) –  clipping needed (paper Section 6.2).

Run:  python examples/example_n3.py
"""

from __future__ import annotations

import numpy as np
from gdsvcp.core import compute_kappa, validate_svcp, PositivityViolation
from gdsvcp.l2 import l2_minimiser, dG_l2
from gdsvcp.clipping import clipped_project
from gdsvcp.plotting import plot_angle_shift, plot_crease_pattern, plot_norm_comparison


def run_case(label: str, angles_deg: list[float], mv: list[str],
             bar_out: str, circle_out: str, norms_out: str) -> None:
    alpha = np.radians(angles_deg)
    validate_svcp(alpha)
    kappa = compute_kappa(alpha)
    n     = len(alpha) // 2

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  Angles : {angles_deg} °")
    print(f"  κ      : {np.degrees(kappa):.4f}°")

    try:
        delta  = l2_minimiser(alpha)
        method = "L² (unconstrained)"
    except PositivityViolation:
        delta  = clipped_project(alpha)
        method = "L² (clipped)"

    alpha_star = alpha + delta
    dg         = float(np.linalg.norm(delta))

    print(f"  Method : {method}")
    print(f"  δ*     : {np.degrees(delta).round(4).tolist()} °")
    print(f"  α+δ*   : {np.degrees(alpha_star).round(4).tolist()} °")
    print(f"  d_G L² : {np.degrees(dg):.6f}°")
    print(f"  d_G L¹ : {np.degrees(2*abs(kappa)):.4f}°")
    print(f"  d_G L∞ : {np.degrees(abs(kappa)/n):.4f}°")
    print(f"  κ after: {np.degrees(compute_kappa(alpha_star)):.2e}°")

    p1 = plot_angle_shift(alpha, alpha_star, title=label, outpath=bar_out)
    print(f"  Bar    → {p1}")

    p2 = plot_crease_pattern(alpha, alpha_star, mountain_valley=mv,
                             title=label, kappa=kappa, outpath=circle_out)
    print(f"  Circle → {p2}")

    p3 = plot_norm_comparison(alpha, title=f"{label} – norms", outpath=norms_out)
    print(f"  Norms  → {p3}")


print("=" * 55)
print("  Example n=3  (paper Examples 3.2 & 6.2)")
print("=" * 55)

# Case A – feasible L²
run_case(
    label      = "n=3 Case A – L² feasible (paper Ex. 3.2)",
    angles_deg = [70., 50., 80., 60., 90., 10.],
    mv         = ["M", "V", "M", "V", "M", "V"],
    bar_out    = "outputs/images/example_n3a_bar.png",
    circle_out = "outputs/images/example_n3a_circle.png",
    norms_out  = "outputs/images/example_n3a_norms.png",
)

# Case B – clipping needed
run_case(
    label      = "n=3 Case B – clipping needed (paper §6.2)",
    angles_deg = [150., 10., 100., 60., 30., 10.],
    mv         = ["M", "V", "M", "V", "M", "V"],
    bar_out    = "outputs/images/example_n3b_bar.png",
    circle_out = "outputs/images/example_n3b_circle.png",
    norms_out  = "outputs/images/example_n3b_norms.png",
)
print()