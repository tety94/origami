"""
cdsvcp.cli
==========
Command-line interface for the cdsvcp package.

Usage
-----
    cdsvcp <angles>...
    cdsvcp --help

Angles are given as rational multiples of π, e.g.:

    # Pattern (90°, 90°, 90°, 90°)  →  already flat-foldable
    cdsvcp 1/2 1/2 1/2 1/2

    # Pattern (120°, 100°, 140°)  →  dC = 1
    cdsvcp 2/3 5/9 7/9

    # Pattern (120°, 80°, 100°, 60°)  →  dC = 2
    cdsvcp 2/3 4/9 5/9 1/3

All values are in units of π:  1 = 180°,  1/2 = 90°,  etc.
Their sum must equal 2 (i.e. 2π).
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction

from . import (
    SVCP,
    edit_distance,
    is_flat_foldable,
    kawasaki_deficit,
    kawasaki_residual,
    repair,
)


def _fmt_angle(f: Fraction) -> str:
    """Format a fraction-of-π as degrees (float) and as a fraction of π."""
    deg = float(f) * 180
    return f"{deg:.6g}°  ({f}π)"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cdsvcp",
        description=(
            "Compute the combinatorial edit distance dC(C, F) for a "
            "single-vertex crease pattern and optionally repair it.\n\n"
            "Angles are given as rational multiples of π (e.g. '1/2' for 90°)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "angles",
        metavar="ANGLE",
        nargs="+",
        help=(
            "Sector angles as rational multiples of π "
            "(e.g. '1/2' '1/2' '1/2' '1/2' for four 90° sectors)."
        ),
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Print the repaired pattern and the operations performed.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra information (S_odd, S_even, κ or K).",
    )

    args = parser.parse_args(argv)

    # ---- parse angles ----
    try:
        parsed = [Fraction(a) for a in args.angles]
    except ValueError as exc:
        parser.error(f"Could not parse angle: {exc}")

    # ---- build SVCP ----
    try:
        C = SVCP(tuple(parsed))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    m = len(C)
    print(f"Input pattern  (m = {m} creases):")
    for i, a in enumerate(C.angles, 1):
        print(f"  α_{i} = {_fmt_angle(a)}")

    # ---- Kawasaki info ----
    if args.verbose:
        from . import sodd, seven
        print(f"\n  S_odd  = {_fmt_angle(sodd(C))}")
        print(f"  S_even = {_fmt_angle(seven(C))}")
        if m % 2 == 0:
            kap = kawasaki_deficit(C)
            print(f"  κ(C)   = {float(kap)*180:.6g}°  ({kap}π)")
        else:
            Kres = kawasaki_residual(C)
            print(f"  K(C)   = {float(Kres)*180:.6g}°  ({Kres}π)")

    # ---- edit distance ----
    d = edit_distance(C)
    status = "already flat-foldable" if d == 0 else f"NOT flat-foldable"
    print(f"\n  dC(C, F) = {d}   [{status}]")

    # ---- repair ----
    if args.repair and d > 0:
        C_star, ops = repair(C)
        print(f"\nRepaired pattern  (m = {len(C_star)} creases):")
        for i, a in enumerate(C_star.angles, 1):
            print(f"  α_{i} = {_fmt_angle(a)}")
        print(f"\nOperations applied ({len(ops)}):")
        for k, op in enumerate(ops, 1):
            if op.kind == 'insert':
                print(
                    f"  {k}. INSERT at position {op.pos}, "
                    f"s = {_fmt_angle(op.param)}"
                )
            else:
                print(f"  {k}. DELETE at position {op.pos}")
        if not is_flat_foldable(C_star):  # sanity
            print("  [WARNING] repaired pattern is not flat-foldable!", file=sys.stderr)
    elif args.repair and d == 0:
        print("  (Pattern is already flat-foldable; no repair needed.)")


if __name__ == "__main__":
    main()
