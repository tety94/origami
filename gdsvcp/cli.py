"""
cli.py – Command-line interface for gdsvcp.

Entrypoint: ``gdsvcp``  (see pyproject.toml console_scripts).

Subcommands
-----------
compute-kappa   Print κ and d_G.
validate        Validate an SVCP and print a full report.
l2              Unconstrained L² minimiser.
clipped         Clipping projection (constrained L²).
compare         Compare d_G under L¹, L², L∞ (+ optional plot).
plot            Bar chart of sector angles (before / after).
circle          Circular crease-pattern diagram.

Global flags
------------
--output-json   Print results as JSON instead of human-readable text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gdsvcp.core import compute_kappa, validate_svcp, PositivityViolation
from gdsvcp.l2 import l2_minimiser, dG_l2
from gdsvcp.clipping import clipped_project
from gdsvcp.norms import l1_minimisers, linfty_minimiser
from gdsvcp.plotting import plot_angle_shift, plot_crease_pattern, plot_norm_comparison
from gdsvcp.io import read_angles_csv


# ── helpers ───────────────────────────────────────────────────────────────

def _load_angles(args) -> np.ndarray:
    """Load angles (degrees from CLI/file) and return as radians."""
    if getattr(args, "file", None):
        deg = read_angles_csv(args.file)
    else:
        deg = np.array([float(a) for a in args.angles], dtype=np.float64)
    return np.radians(deg)


def _out(data: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        for k, v in data.items():
            print(f"  {k:<22} {v}")


# ── subcommand handlers ───────────────────────────────────────────────────

def cmd_compute_kappa(args) -> int:
    try:
        alpha = _load_angles(args)
        validate_svcp(alpha)
        kappa = compute_kappa(alpha)
        dg    = dG_l2(alpha)
        data  = {
            "kappa_deg":  round(np.degrees(kappa), 8),
            "kappa_rad":  round(float(kappa), 10),
            "dG_L2_deg":  round(np.degrees(dg), 8),
            "dG_L2_rad":  round(float(dg), 10),
            "n":          len(alpha) // 2,
        }
        _out(data, args.output_json)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_validate(args) -> int:
    """Full SVCP validation report."""
    try:
        alpha = _load_angles(args)
        n = len(alpha) // 2

        errors: list[str] = []
        warnings: list[str] = []

        # positivity
        if np.any(alpha <= 0):
            errors.append("one or more angles ≤ 0")

        # sum
        total = float(np.sum(alpha))
        if not np.isclose(total, 2 * np.pi, atol=1e-9):
            errors.append(f"sum of angles = {np.degrees(total):.4f}° ≠ 360°")

        kappa = compute_kappa(alpha)
        flat_foldable = np.isclose(kappa, 0.0, atol=1e-12)

        # positivity of L2 shift
        if not flat_foldable:
            shift = abs(kappa) / n
            min_odd = min(alpha[i] for i in range(0, 2*n, 2))
            min_even = min(alpha[i] for i in range(1, 2*n, 2))
            if kappa > 0 and min_odd <= shift:
                warnings.append("L² minimiser violates positivity – clipping needed")
            elif kappa < 0 and min_even <= shift:
                warnings.append("L² minimiser violates positivity – clipping needed")

        data = {
            "n":              n,
            "angles_deg":     [round(np.degrees(a), 4) for a in alpha],
            "sum_deg":        round(np.degrees(total), 6),
            "flat_foldable":  flat_foldable,
            "kappa_deg":      round(np.degrees(kappa), 6),
            "min_angle_deg":  round(np.degrees(float(alpha.min())), 4),
            "errors":         errors,
            "warnings":       warnings,
            "status":         "ERROR" if errors else ("WARN" if warnings else "OK"),
        }

        if args.output_json:
            print(json.dumps(data, indent=2))
        else:
            status_sym = {"OK": "✓", "WARN": "⚠", "ERROR": "✗"}[data["status"]]
            print(f"\n  {status_sym}  Status: {data['status']}")
            print(f"  n               = {n}")
            print(f"  sum             = {data['sum_deg']}°")
            print(f"  flat-foldable   = {flat_foldable}")
            print(f"  κ               = {data['kappa_deg']}°")
            print(f"  min angle       = {data['min_angle_deg']}°")
            for e in errors:
                print(f"  ✗ {e}")
            for w in warnings:
                print(f"  ⚠ {w}")
            print()

        return 0 if not errors else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_l2(args) -> int:
    try:
        alpha = _load_angles(args)
        validate_svcp(alpha)
        try:
            delta = l2_minimiser(alpha)
        except PositivityViolation as pv:
            print(f"Positivity violated: {pv}", file=sys.stderr)
            print("  → use the 'clipped' subcommand.", file=sys.stderr)
            return 2
        dg = float(np.linalg.norm(delta))
        data = {
            "delta_deg":  [round(np.degrees(d), 6) for d in delta],
            "dG_L2_deg":  round(np.degrees(dg), 8),
            "dG_L2_rad":  round(dg, 10),
        }
        _out(data, args.output_json)
        if getattr(args, "plot", False):
            p = plot_angle_shift(alpha, alpha + delta, "L² minimiser",
                                 outpath=getattr(args, "outpath", None))
            print(f"  bar chart  → {p}")
        if getattr(args, "circle", False):
            p = plot_crease_pattern(alpha, alpha + delta,
                                    title="L² minimiser – crease pattern",
                                    kappa=compute_kappa(alpha),
                                    outpath=getattr(args, "outpath_circle", None))
            print(f"  circle     → {p}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_clipped(args) -> int:
    try:
        alpha = _load_angles(args)
        validate_svcp(alpha)
        delta = clipped_project(alpha, verbose=getattr(args, "verbose", False))
        dg    = float(np.linalg.norm(delta))
        data  = {
            "delta_deg": [round(np.degrees(d), 6) for d in delta],
            "dG_deg":    round(np.degrees(dg), 8),
            "dG_rad":    round(dg, 10),
        }
        _out(data, args.output_json)
        p = plot_angle_shift(alpha, alpha + delta,
                             "Clipped L² projection",
                             outpath=getattr(args, "outpath", None))
        print(f"  bar chart  → {p}")
        if getattr(args, "circle", False):
            p2 = plot_crease_pattern(alpha, alpha + delta,
                                     title="Clipped projection – crease pattern",
                                     kappa=compute_kappa(alpha),
                                     outpath=getattr(args, "outpath_circle", None))
            print(f"  circle     → {p2}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_compare(args) -> int:
    """Compare d_G under L¹, L², L∞."""
    try:
        alpha  = _load_angles(args)
        validate_svcp(alpha)
        n      = len(alpha) // 2
        kappa  = compute_kappa(alpha)
        abs_k  = abs(kappa)
        d_l1   = 2.0 * abs_k
        d_l2   = abs_k * np.sqrt(2.0 / n)
        d_linf = abs_k / n
        data   = {
            "kappa_deg":    round(np.degrees(kappa), 6),
            "n":            n,
            "dG_L1_deg":   round(np.degrees(d_l1),   8),
            "dG_L2_deg":   round(np.degrees(d_l2),   8),
            "dG_Linf_deg": round(np.degrees(d_linf), 8),
            "dG_L1_rad":   round(d_l1,   10),
            "dG_L2_rad":   round(d_l2,   10),
            "dG_Linf_rad": round(d_linf, 10),
        }
        _out(data, args.output_json)
        if getattr(args, "plot", False):
            p = plot_norm_comparison(alpha, title="Norm comparison",
                                     outpath=getattr(args, "outpath", None))
            print(f"  norm plot  → {p}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_plot(args) -> int:
    try:
        alpha = _load_angles(args)
        if getattr(args, "perturbed", None):
            alpha_p = np.radians(np.array([float(a) for a in args.perturbed],
                                          dtype=np.float64))
        else:
            alpha_p = alpha.copy()
        title = getattr(args, "title", None) or "SVCP Angles"
        p = plot_angle_shift(alpha, alpha_p, title,
                             outpath=getattr(args, "outpath", None))
        print(f"  bar chart  → {p}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_circle(args) -> int:
    try:
        alpha = _load_angles(args)
        if getattr(args, "perturbed", None):
            alpha_p = np.radians(np.array([float(a) for a in args.perturbed],
                                          dtype=np.float64))
        else:
            alpha_p = None
        mv    = getattr(args, "mv", None)
        title = getattr(args, "title", None) or "Crease Pattern"
        kappa = compute_kappa(alpha)
        p = plot_crease_pattern(
            alpha, alpha_p,
            mountain_valley=mv,
            title=title,
            outpath=getattr(args, "outpath", None),
            kappa=kappa,
        )
        print(f"  circle     → {p}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ── argument parser ───────────────────────────────────────────────────────

def _add_angle_args(p: argparse.ArgumentParser) -> None:
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--angles", nargs="+", metavar="A",
                     help="Sector angles in degrees.")
    grp.add_argument("--file", metavar="FILE",
                     help="CSV file with angles in degrees.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gdsvcp",
        description="Geometric Edit Distance for Single-Vertex Crease Patterns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  gdsvcp validate --angles 100 80 100 80
  gdsvcp compute-kappa --angles 70 50 80 60 90 10
  gdsvcp l2 --angles 100 80 100 80 --plot
  gdsvcp clipped --file angles.csv --circle
  gdsvcp compare --angles 70 50 80 60 90 10 --plot
  gdsvcp circle --angles 100 80 100 80 --perturbed 90 90 90 90
""",
    )
    p.add_argument("--output-json", action="store_true",
                   help="Print results as JSON.")
    sub = p.add_subparsers(dest="command", required=True)

    # compute-kappa
    pk = sub.add_parser("compute-kappa", help="Print κ and d_G (L²).")
    _add_angle_args(pk)

    # validate
    pv = sub.add_parser("validate", help="Validate SVCP and print report.")
    _add_angle_args(pv)

    # l2
    pl = sub.add_parser("l2", help="Unconstrained L² minimiser.")
    _add_angle_args(pl)
    pl.add_argument("--plot",   action="store_true", help="Save bar chart.")
    pl.add_argument("--circle", action="store_true", help="Save circle plot.")
    pl.add_argument("--outpath", default=None)
    pl.add_argument("--outpath-circle", default=None, dest="outpath_circle")

    # clipped
    pc = sub.add_parser("clipped", help="Clipping projection (constrained L²).")
    _add_angle_args(pc)
    pc.add_argument("--circle", action="store_true", help="Save circle plot.")
    pc.add_argument("--outpath", default=None)
    pc.add_argument("--outpath-circle", default=None, dest="outpath_circle")
    pc.add_argument("--verbose", action="store_true")

    # compare
    pco = sub.add_parser("compare", help="Compare d_G under L¹, L², L∞.")
    _add_angle_args(pco)
    pco.add_argument("--plot",    action="store_true", help="Save norm-comparison chart.")
    pco.add_argument("--outpath", default=None)

    # plot (bar chart)
    pp = sub.add_parser("plot", help="Bar chart of sector angles.")
    _add_angle_args(pp)
    pp.add_argument("--perturbed", nargs="+", metavar="A",
                    help="Perturbed angles in degrees.")
    pp.add_argument("--title",   default=None)
    pp.add_argument("--outpath", default=None)

    # circle
    pci = sub.add_parser("circle", help="Circular crease-pattern diagram.")
    _add_angle_args(pci)
    pci.add_argument("--perturbed", nargs="+", metavar="A",
                     help="Perturbed angles in degrees (side-by-side panel).")
    pci.add_argument("--mv", nargs="+", metavar="MV",
                     help="Mountain/valley assignment: e.g. M V M V ...")
    pci.add_argument("--title",   default=None)
    pci.add_argument("--outpath", default=None)

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    # propagate global flag into args namespace
    handlers = {
        "compute-kappa": cmd_compute_kappa,
        "validate":      cmd_validate,
        "l2":            cmd_l2,
        "clipped":       cmd_clipped,
        "compare":       cmd_compare,
        "plot":          cmd_plot,
        "circle":        cmd_circle,
    }
    rc = handlers[args.command](args)
    sys.exit(rc)


if __name__ == "__main__":
    main()