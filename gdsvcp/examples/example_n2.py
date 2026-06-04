"""
example_n2.py – Paper Example 3.1 (n = 2).

C = (100°, 80°, 100°, 80°)  →  κ = 20°  →  δ* = (−10°, +10°, −10°, +10°)
Expected d_G = 20° = π/9 rad.

Run:  python examples/example_n2.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from gdsvcp.core import compute_kappa, validate_svcp
from gdsvcp.l2 import l2_minimiser, dG_l2
from gdsvcp.clipping import clipped_project
from gdsvcp.core import PositivityViolation
from gdsvcp.plotting import plot_angle_shift, plot_crease_pattern, plot_norm_comparison

# ── input ─────────────────────────────────────────────────────────────────
angles_deg = [100.0, 80.0, 100.0, 80.0]
alpha      = np.radians(angles_deg)

print("=" * 55)
print("  Example n=2  (paper Example 3.1)")
print("=" * 55)
print(f"  Angles : {angles_deg} °")

validate_svcp(alpha)

# ── κ ─────────────────────────────────────────────────────────────────────
kappa = compute_kappa(alpha)
print(f"  κ      : {np.degrees(kappa):.4f}°  ({kappa:.6f} rad)")

# ── L² minimiser ──────────────────────────────────────────────────────────
try:
    delta = l2_minimiser(alpha)
    method = "L² (unconstrained)"
except PositivityViolation:
    delta  = clipped_project(alpha)
    method = "L² (clipped)"

alpha_star = alpha + delta
dg         = float(np.linalg.norm(delta))

print(f"  Method : {method}")
print(f"  δ*     : {np.degrees(delta).round(4).tolist()} °")
print(f"  α+δ*   : {np.degrees(alpha_star).round(4).tolist()} °")
print(f"  d_G    : {np.degrees(dg):.6f}°  ({dg:.6f} rad)")
print(f"  κ after: {np.degrees(compute_kappa(alpha_star)):.2e}°")

# ── L¹ / L∞ for reference ─────────────────────────────────────────────────
n = len(alpha) // 2
print(f"  d_G L¹ : {np.degrees(2*abs(kappa)):.4f}°")
print(f"  d_G L∞ : {np.degrees(abs(kappa)/n):.4f}°")

# ── plots ─────────────────────────────────────────────────────────────────
p1 = plot_angle_shift(
    alpha, alpha_star,
    title="n=2 paper example – L² minimiser",
    outpath="outputs/images/example_n2_bar.png",
)
print(f"\n  Bar chart → {p1}")

p2 = plot_crease_pattern(
    alpha, alpha_star,
    mountain_valley=["M", "V", "M", "V"],
    title="n=2 paper example – crease pattern",
    kappa=kappa,
    outpath="outputs/images/example_n2_circle.png",
)
print(f"  Circle    → {p2}")

p3 = plot_norm_comparison(
    alpha,
    title="n=2 paper example – norm comparison",
    outpath="outputs/images/example_n2_norms.png",
)
print(f"  Norms     → {p3}")
print()