"""
cdmsvcp.cli
===========
Command-line interface for the cdmsvcp package.

Usage
-----
    cdmsvcp <angles> -- <labels>
    cdmsvcp --help

Angles are given as rational multiples of π, e.g.:

    # (90°,90°,90°,90°) with labels MMMV → already fully flat-foldable
    cdmsvcp 1/2 1/2 1/2 1/2 -- M M M V

    # (90°,90°,90°,90°) with labels MMMM → ν = 2, distance = 1 (one flip)
    cdmsvcp 1/2 1/2 1/2 1/2 -- M M M M

    # (120°,80°,100°,60°) MMMV → κ≠0, ν=0, distance = 2
    cdmsvcp 2/3 4/9 5/9 1/3 -- M M M V

All angle values are in units of π:  1 = 180°,  1/2 = 90°,  etc.
Their sum must equal 2 (i.e. 2π).
Labels are 'M' (mountain) or 'V' (valley), one per crease.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction

from . import (
    SVCP,
    LSVCP,
    M,
    V,
    edit_distance,
    is_fully_flat_foldable,
    kawasaki_deficit,
    kawasaki_residual,
    maekawa_deficit,
    signed_delta,
    repair,
)


def _fmt_angle(f: Fraction) -> str:
    deg = float(f) * 180
    return f"{deg:.6g}°  ({f}π)"


def _fmt_op(op) -> str:
    if op.kind == 'insert':
        deg = float(op.param) * 180
        return (
            f"  insert at position {op.pos}: "
            f"s = {op.param}π = {deg:.4g}°, label = {op.label}"
        )
    else:
        return f"  flip at position {op.pos}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cdmsvcp",
        description=(
            "Compute the labelled edit distance d_MV((C,μ), F_MV) for a "
            "single-vertex crease pattern and optionally repair it.\n\n"
            "Angles are given as rational multiples of π (e.g. '1/2' for 90°).\n"
            "Labels follow '--': 'M' for mountain, 'V' for valley."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  cdmsvcp 1/2 1/2 1/2 1/2 -- M M M V\n"
            "  cdmsvcp 2/3 4/9 5/9 1/3 -- M M M M --repair\n"
            "  cdmsvcp 2/3 4/9 5/9 1/3 -- M V M V --verbose\n"
        ),
    )
    parser.add_argument(
        "angles",
        metavar="ANGLE",
        nargs="+",
        help="Sector angles as rational multiples of π (before '--').",
    )
    parser.add_argument(
        "labels",
        metavar="LABEL",
        nargs="*",
        help="Mountain/valley labels after '--' (M or V, one per crease).",
    )
    parser.add_argument(
        "--repair", "-r",
        action="store_true",
        help="Print the repaired pattern and the operations performed.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra information (S_odd, κ, δ, ν).",
    )

    # Split on '--' manually (argparse doesn't handle this well)
    if "--" in (argv or sys.argv[1:]):
        argv_list = list(argv or sys.argv[1:])
        sep = argv_list.index("--")
        angle_args = argv_list[:sep]
        rest = argv_list[sep + 1:]
        # Separate labels from flags in the post-'--' part
        flags_after = []
        label_args = []
        for a in rest:
            if a in ("--repair", "-r", "--verbose", "-v"):
                flags_after.append(a)
            else:
                label_args.append(a)
        # Also remove any flags mixed into angle_args
        flags_before = []
        clean_angle_args = []
        for a in angle_args:
            if a in ("--repair", "-r", "--verbose", "-v"):
                flags_before.append(a)
            else:
                clean_angle_args.append(a)
        args = parser.parse_args(clean_angle_args + flags_before + flags_after)
        args.labels = label_args
    else:
        args = parser.parse_args(argv)

    # ---- parse angles ----
    try:
        raw_angles = [Fraction(a) for a in args.angles]
    except ValueError as e:
        print(f"Error parsing angles: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        svcp = SVCP(tuple(raw_angles))
    except ValueError as e:
        print(f"Invalid SVCP: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- parse labels ----
    raw_labels = [lbl.upper() for lbl in args.labels]
    if not raw_labels:
        print(
            "Error: labels required (provide them after '--').",
            file=sys.stderr,
        )
        print("Example:  cdmsvcp 1/2 1/2 1/2 1/2 -- M M M V", file=sys.stderr)
        sys.exit(1)

    if len(raw_labels) != len(svcp):
        print(
            f"Error: {len(raw_labels)} labels given but pattern has "
            f"{len(svcp)} creases.",
            file=sys.stderr,
        )
        sys.exit(1)

    bad = [lbl for lbl in raw_labels if lbl not in (M, V)]
    if bad:
        print(f"Error: invalid labels {bad}; must be 'M' or 'V'.", file=sys.stderr)
        sys.exit(1)

    try:
        lc = LSVCP(svcp, tuple(raw_labels))
    except ValueError as e:
        print(f"Invalid LSVCP: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- display input ----
    m = len(lc)
    print()
    print("Input LSVCP:")
    print(f"  m = {m}  ({'even' if m % 2 == 0 else 'odd'})")
    print(f"  Angles: {', '.join(_fmt_angle(a) for a in lc.angles)}")
    print(f"  Labels: {' '.join(lc.labels)}")

    if args.verbose:
        Kres = kawasaki_residual(svcp)
        print(f"  S_odd  = {float(Kres + 1)*180:.6g}°  ({Kres + 1}π)")
        if m % 2 == 0:
            kap = kawasaki_deficit(svcp)
            print(f"  κ      = {float(kap)*180:.6g}°  ({kap}π)")
            nu = maekawa_deficit(lc)
            print(f"  δ      = {signed_delta(lc)}")
            print(f"  ν      = {nu}")
        else:
            print(f"  K(C)   = {float(Kres)*180:.6g}°  ({Kres}π)  [m odd: κ undefined]")
            print(f"  δ      = {signed_delta(lc)}")

    print()
    d = edit_distance(lc)
    foldable_str = "YES ✓" if is_fully_flat_foldable(lc) else "NO"
    print(f"  Fully flat-foldable: {foldable_str}")
    print(f"  d_MV = {d}")

    if args.repair:
        if d == 0:
            print("\n  (already fully flat-foldable — no operations needed)")
        else:
            lc_star, ops = repair(lc)
            print(f"\n  Operations ({len(ops)}):")
            for op in ops:
                print(_fmt_op(op))
            print()
            print("  Repaired LSVCP:")
            print(f"    Angles: {', '.join(_fmt_angle(a) for a in lc_star.angles)}")
            print(f"    Labels: {' '.join(lc_star.labels)}")
            if args.verbose:
                from cdsvcp import kawasaki_deficit as kd
                kap_star = kd(lc_star.svcp)
                nu_star  = maekawa_deficit(lc_star)
                print(f"    κ* = {float(kap_star)*180:.6g}°,  ν* = {nu_star}")
            print(f"    Fully flat-foldable: YES ✓")
    print()


if __name__ == "__main__":
    main()
