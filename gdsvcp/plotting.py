"""
plotting.py – Visualisation utilities for SVCPs.

Three plot types
----------------
plot_angle_shift          Side-by-side bar chart before/after perturbation.
plot_crease_pattern       Circular SVCP diagram with sector wedges and fold lines.
plot_norm_comparison      Bar chart comparing d_G under L¹, L², L∞.

All saved images go to outputs/images/ (created if missing).
Angles are always passed as **radians**; labels are displayed in degrees.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

__all__ = [
    "plot_angle_shift",
    "plot_crease_pattern",
    "plot_norm_comparison",
]

_OUT_DIR = Path("outputs") / "images"

_C_ORIG  = "#4c8ef0"
_C_PERT  = "#52c47e"
_C_MTN   = "#e05a4e"   # mountain fold – red
_C_VAL   = "#4c8ef0"   # valley fold  – blue
_C_L1    = "#e8a838"   # L¹  – amber
_C_L2    = "#4c8ef0"   # L²  – blue
_C_LINF  = "#7c6ee0"   # L∞  – purple


# ── helpers ───────────────────────────────────────────────────────────────

def _ensure_output_dir(path: Optional[str] = None) -> Path:
    d = Path(path).parent if path else _OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60]


def _auto_path(prefix: str, title: str, outpath: Optional[str]) -> str:
    _ensure_output_dir(outpath)
    if outpath is None:
        outpath = str(_OUT_DIR / f"{prefix}_{_slugify(title)}.png")
    return outpath


# ── 1. Bar chart ──────────────────────────────────────────────────────────

def plot_angle_shift(
    alpha: np.ndarray,
    alpha_perturbed: np.ndarray,
    title: str,
    outpath: Optional[str] = None,
) -> str:
    """Side-by-side bar chart: original (blue) vs perturbed (green) angles.

    Parameters
    ----------
    alpha, alpha_perturbed : np.ndarray  –  sector angles in **radians**.
    title : str  –  plot title (also used for the filename slug).
    outpath : str or None  –  explicit PNG path; auto-generated if None.

    Returns
    -------
    str  –  absolute path of the saved PNG.
    """
    alpha   = np.asarray(alpha,           dtype=np.float64)
    alpha_p = np.asarray(alpha_perturbed, dtype=np.float64)
    deg_o   = np.degrees(alpha)
    deg_p   = np.degrees(alpha_p)
    N       = len(deg_o)

    outpath = _auto_path("bar", title, outpath)
    fig, ax = plt.subplots(figsize=(max(8, N * 1.4), 4.8))
    x, bw   = np.arange(N), 0.38

    b_o = ax.bar(x - bw/2, deg_o, bw, label="Original $C$",
                 color=_C_ORIG, edgecolor="#2a5fbb", linewidth=0.7)
    b_p = ax.bar(x + bw/2, deg_p, bw, label=r"Perturbed $C+\delta^*$",
                 color=_C_PERT, edgecolor="#2a8c4e", linewidth=0.7)

    for bar, v in zip(b_o, deg_o):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.1f}°", ha="center", va="bottom", fontsize=8, color="#1c4a99")
    for bar, v in zip(b_p, deg_p):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.1f}°", ha="center", va="bottom", fontsize=8, color="#1a6636")

    for i, (vo, vp) in enumerate(zip(deg_o, deg_p)):
        d = vp - vo
        if abs(d) > 0.05:
            ax.annotate(
                f"{'+'if d>0 else ''}{d:.1f}°",
                xy=(i, max(vo, vp) + 4.5),
                ha="center", va="bottom", fontsize=7,
                color="#888", style="italic",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\alpha_{{{i+1}}}$" for i in range(N)], fontsize=10)
    ax.set_ylabel("Angle (degrees)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(deg_o.max(), deg_p.max()) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(outpath)


# ── 2. Circular crease-pattern diagram ───────────────────────────────────

def plot_crease_pattern(
    alpha: np.ndarray,
    alpha_perturbed: Optional[np.ndarray] = None,
    mountain_valley: Optional[list[str]] = None,
    title: str = "Crease Pattern",
    outpath: Optional[str] = None,
    kappa: Optional[float] = None,
) -> str:
    """Circular crease-pattern diagram.

    Draws the SVCP as a unit disk divided into 2n sector wedges.
    Crease lines are coloured red (mountain) or blue (valley).
    When *alpha_perturbed* is given, two panels are shown side by side:
    the original pattern and the perturbed one, with the original crease
    lines drawn as a faint dashed ghost on the second panel so the angular
    shift is immediately visible.

    Parameters
    ----------
    alpha : np.ndarray
        Original sector angles in **radians** (must sum to 2π).
    alpha_perturbed : np.ndarray or None
        Perturbed angles; triggers a two-panel figure when given.
    mountain_valley : list[str] or None
        'M' / 'V' assignment per crease (length 2n).
        Defaults to alternating M, V, M, V, … starting with M.
    title : str
        Overall figure title.
    outpath : str or None
        Explicit PNG path; auto-generated if None.
    kappa : float or None
        Kawasaki deficit in radians; printed at the disk centre if given.

    Returns
    -------
    str  –  absolute path of the saved PNG.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    N, n  = len(alpha), len(alpha) // 2

    if mountain_valley is None:
        mountain_valley = ["M" if i % 2 == 0 else "V" for i in range(N)]

    outpath = _auto_path("circle", title, outpath)
    ncols   = 2 if alpha_perturbed is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.8))
    if ncols == 1:
        axes = [axes]

    def _draw_disk(ax, angles_rad, mv, subtitle, ghost=None):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.42, 1.42)

        R, R_lbl = 1.0, 1.22
        offset   = np.pi / 2   # first crease at 12 o'clock

        cum = np.concatenate([[0.0], np.cumsum(angles_rad)])

        # sector wedge fills
        for i in range(N):
            a0  = np.degrees(offset - cum[i])
            a1  = np.degrees(offset - cum[i + 1])
            col = "#fde8e6" if i % 2 == 0 else "#e8f0fd"
            ax.add_patch(mpatches.Wedge(
                (0, 0), R, a1, a0,
                facecolor=col, edgecolor="none", alpha=0.65, zorder=1,
            ))

        # ghost (original) crease lines
        if ghost is not None:
            cum_g = np.concatenate([[0.0], np.cumsum(ghost)])
            for i in range(N):
                ang = offset - cum_g[i]
                ax.plot([0, R * np.cos(ang)], [0, R * np.sin(ang)],
                        color="#bbb", lw=1.0, ls="--", zorder=2, alpha=0.6)

        # disk outline
        ax.add_patch(plt.Circle((0, 0), R, color="#555",
                                fill=False, lw=1.2, zorder=3))

        # crease lines + index markers
        for i in range(N):
            ang = offset - cum[i]
            col = _C_MTN if mv[i] == "M" else _C_VAL
            ax.plot([0, R * np.cos(ang)], [0, R * np.sin(ang)],
                    color=col, lw=2.2, solid_capstyle="round", zorder=4)
            # crease index at rim
            rx, ry = 1.10 * np.cos(ang), 1.10 * np.sin(ang)
            ax.text(rx, ry, str(i + 1),
                    ha="center", va="center", fontsize=7.5, color="#555")

        # sector angle labels
        for i in range(N):
            mid = offset - (cum[i] + cum[i + 1]) / 2
            lx, ly = R_lbl * np.cos(mid), R_lbl * np.sin(mid)
            ax.text(lx, ly, f"{np.degrees(angles_rad[i]):.1f}°",
                    ha="center", va="center", fontsize=8.5, color="#333")

        # centre annotation
        lines = [f"n = {n}"]
        if kappa is not None:
            lines.append(f"κ = {np.degrees(kappa):.2f}°")
        ax.text(0, 0, "\n".join(lines),
                ha="center", va="center", fontsize=9,
                color="#444", multialignment="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

        # legend
        handles = [
            mpatches.Patch(color=_C_MTN, label="Mountain (M)"),
            mpatches.Patch(color=_C_VAL, label="Valley (V)"),
        ]
        if ghost is not None:
            handles.append(mpatches.Patch(color="#bbb", label="Original (ghost)"))
        ax.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.13), ncol=len(handles),
                  fontsize=8, frameon=False)
        ax.set_title(subtitle, fontsize=11, pad=10)

    if alpha_perturbed is not None:
        alpha_p = np.asarray(alpha_perturbed, dtype=np.float64)
        _draw_disk(axes[0], alpha,   mountain_valley, "Original $C$")
        _draw_disk(axes[1], alpha_p, mountain_valley,
                   r"Perturbed $C+\delta^*$", ghost=alpha)
    else:
        _draw_disk(axes[0], alpha, mountain_valley, "")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(outpath)


# ── 3. Norm comparison ────────────────────────────────────────────────────

def plot_norm_comparison(
    alpha: np.ndarray,
    title: str = "Norm comparison",
    outpath: Optional[str] = None,
) -> str:
    """Bar chart: d_G under L¹ = 2|κ|,  L² = |κ|√(2/n),  L∞ = |κ|/n.

    Displays the three theoretical minimum distances (paper Section 5
    summary table).  The analytical formula is shown inside each bar and
    the exact degree/radian value on top.

    Parameters
    ----------
    alpha : np.ndarray  –  sector angles in **radians**.
    title : str  –  plot title.
    outpath : str or None  –  explicit PNG path; auto-generated if None.

    Returns
    -------
    str  –  absolute path of the saved PNG.
    """
    from gdsvcp.core import compute_kappa

    alpha  = np.asarray(alpha, dtype=np.float64)
    n      = len(alpha) // 2
    kappa  = compute_kappa(alpha)
    abs_k  = abs(kappa)

    d_l1   = 2.0 * abs_k
    d_l2   = abs_k * np.sqrt(2.0 / n)
    d_linf = abs_k / n

    vals_deg = [np.degrees(d_l1), np.degrees(d_l2), np.degrees(d_linf)]
    vals_rad = [d_l1, d_l2, d_linf]
    labels   = ["$L^1$", "$L^2$", "$L^\\infty$"]
    colors   = [_C_L1,  _C_L2,   _C_LINF]
    edges    = ["#a06010", "#2a5fbb", "#4a3aaa"]
    formulas = [r"$2|\kappa|$", r"$|\kappa|\sqrt{2/n}$", r"$|\kappa|/n$"]

    outpath = _auto_path("norms", title, outpath)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x       = np.arange(3)

    bars = ax.bar(x, vals_deg, width=0.52,
                  color=colors, edgecolor=edges, linewidth=0.8)

    max_v = max(vals_deg)
    for bar, vd, vr, fml in zip(bars, vals_deg, vals_rad, formulas):
        # value on top
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max_v * 0.016,
                f"{vd:.3f}°\n({vr:.4f} rad)",
                ha="center", va="bottom", fontsize=9,
                color="#333", multialignment="center")
        # formula inside bar (only if bar tall enough)
        if bar.get_height() > max_v * 0.12:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 0.45,
                    fml,
                    ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")

    # ratio annotations between bars
    if d_l2 > 0 and d_l1 > 0:
        for src, dst, xpos in [(0, 1, 0.97), (1, 2, 1.97)]:
            ratio = vals_deg[dst] / vals_deg[src]
            ax.annotate(
                f"×{ratio:.3f}",
                xy=(xpos, (vals_deg[src] + vals_deg[dst]) / 2),
                ha="center", va="center", fontsize=8, color="#666",
                style="italic",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("$d_G$  (degrees)", fontsize=11)
    ax.set_title(
        f"{title}\n"
        rf"$|\kappa|={np.degrees(abs_k):.2f}°$,  $n={n}$",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, max_v * 1.40)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(outpath)