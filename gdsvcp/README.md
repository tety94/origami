# gdsvcp — Geometric Edit Distance for Single-Vertex Crease Patterns

Implementation of the **ClippedProject** algorithm and closed-form distance
formulae from:

> Callegaro, S. *"How Far Is a Single-Vertex Crease Pattern from Flat
> Foldability? Minimal Angular Corrections under Three Norms."* (2026).

---

## Overview

A **single-vertex crease pattern** (SVCP) is a vector of 2n positive sector
angles summing to 2π.  It is *locally flat-foldable* if and only if the
**Kawasaki deficit** κ(C) = Σ_{i odd} α_i − π equals zero
(Kawasaki–Justin theorem).

This package answers the question: *given a non-foldable SVCP, what is the
smallest angular perturbation that restores flat foldability, and which
creases should bear the correction?*

The answer depends on the cost metric chosen:

| Norm | Physical meaning | d_G(C, F) | Minimiser |
|---|---|---|---|
| L² | distributed re-creasing (total energy) | \|κ\| √(2/n) | −(κ/n) **w** |
| L∞ | per-crease tolerance control (worst case) | \|κ\| / n | −(κ/n) **w** (same) |
| L¹ | sparse repair (fewest creases touched) | 2\|κ\| | n² vertex choices |

All angles are stored and returned as **radians** throughout the package.
The CLI accepts degrees for human convenience and converts on entry.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10 and NumPy ≥ 1.24.

---

## Quick start

```python
import numpy as np
from gdsvcp import compute_kappa, dG_l2, l2_minimiser, clipped_project

# Pattern from the paper: (70°, 50°, 80°, 60°, 90°, 10°), n = 3
alpha = np.radians([70., 50., 80., 60., 90., 10.])

kappa = compute_kappa(alpha)          # κ = π/3 rad  (60°)
print(f"κ = {np.degrees(kappa):.4g}°")   # → 60.0°

dg = dG_l2(alpha)                     # |κ| √(2/n)
print(f"d_G(L²) = {np.degrees(dg):.4g}°")  # ≈ 48.99°

# Unconstrained L² minimiser (feasible here: min odd angle = 70° > 20° = κ/n)
delta = l2_minimiser(alpha)
corrected = alpha + delta
print(np.degrees(corrected))          # all six corrected angles

# When positivity might be violated, use the certified clipping algorithm:
delta_safe = clipped_project(alpha, etol=1e-9)
```

---

## Command-line interface

Angles are passed in **degrees** (human-readable); all internal computations
use radians.

```bash
# Compute κ and d_G for a pattern
gdsvcp compute-kappa 70 50 80 60 90 10

# Full validation report
gdsvcp validate 70 50 80 60 90 10

# Unconstrained L² minimiser
gdsvcp l2 70 50 80 60 90 10

# Constrained minimiser (clipping algorithm, recommended for production use)
gdsvcp clipped 70 50 80 60 90 10

# Compare d_G under all three norms
gdsvcp compare 70 50 80 60 90 10

# With JSON output
gdsvcp compare 70 50 80 60 90 10 --output-json

# Load angles from a CSV file (values in degrees)
gdsvcp compare --file my_pattern.csv

# Plot bar chart of sector angles before/after correction
gdsvcp plot 70 50 80 60 90 10

# Circular crease-pattern diagram
gdsvcp circle 70 50 80 60 90 10
```

---

## Package structure

```
gdsvcp/
├── gdsvcp/
│   ├── __init__.py       public API
│   ├── core.py           weight_vector, compute_kappa, validate_svcp,
│   │                     is_flat_foldable, cyclic_shift
│   ├── l2.py             l2_minimiser, dG_l2, positivity_report
│   ├── clipping.py       clipped_project, free_solve  (Algorithm 4.1–4.2)
│   ├── norms.py          linfty_minimiser, l1_minimisers,
│   │                     l1_feasible_minimisers
│   ├── plotting.py       plot_angle_shift, plot_crease_pattern,
│   │                     plot_norm_comparison
│   ├── io.py             read_angles_csv, write_angles_csv
│   └── cli.py            command-line interface (subcommands above)
├── tests/
│   ├── fixtures/         pre-computed JSON reference values
│   ├── test_core.py      unit tests for core, L², clipping, norms, plots
│   └── test_clipping.py  targeted tests for the clipping algorithm
├── examples/
│   ├── example_n2.py     n = 2 worked example from the paper
│   └── example_n3.py     n = 3 clipping example from the paper
├── outputs/
│   └── images/           figures saved by examples and CLI plot commands
├── pyproject.toml
└── README.md
```

---

## Algorithm: ClippedProject

When the unconstrained minimiser δ\* = −(κ/n) **w** violates the positivity
constraint α_i + δ_i > 0, `clipped_project` runs the certified two-phase
active-set method (Algorithm 4.1):

| Phase | What it does | Termination bound |
|---|---|---|
| 1 — primal | clips the most-violated index, re-runs FreeSolve | ≤ n − 1 iterations |
| 2 — dual | releases active indices with negative KKT multiplier | ≤ n − 1 iterations |

**Structural guarantees** (Lemma 4.4, Theorem 4.6):

- Phase 1 clips *only* paper-odd-indexed angles when κ > 0 (only-odd-clipping invariant).
- The last free paper-odd index is *never* clipped.
- FreeSolve is non-degenerate throughout (det M = 4 n_F⁺ n_F⁻ > 0).
- The output satisfies all four KKT conditions (Proposition 4.7).

---

## The three norms

### L² — distributed re-creasing

```python
from gdsvcp import l2_minimiser, dG_l2, clipped_project

alpha  = np.radians([70., 50., 80., 60., 90., 10.])
delta  = l2_minimiser(alpha)   # raises PositivityViolation if not feasible
# or, always safe:
delta  = clipped_project(alpha, etol=1e-9)
dg     = np.linalg.norm(delta)   # true d_G after clipping
```

### L∞ — per-crease tolerance control

```python
from gdsvcp import linfty_minimiser

delta, value = linfty_minimiser(alpha)
# value = |κ|/n  (worst-case single-crease correction)
# delta is identical to the L² minimiser (Proposition 5.2)
```

### L¹ — sparse repair

```python
from gdsvcp import l1_feasible_minimisers

feasible = l1_feasible_minimisers(alpha)
# Each entry touches exactly two creases (one odd, one even) by ±|κ|
# Total cost = 2|κ| for every choice  (Proposition 5.3)
delta = feasible[0]   # pick any; all are equally optimal
```

---

## Positivity and the conservativeness gap

```python
from gdsvcp import positivity_report

report = positivity_report(alpha)
# report["exact_ok"]        → True iff unconstrained δ* is feasible
# report["sufficient_ok"]   → True iff conservative Thm 3.1(iii) holds
# report["gamma_rad"]       → conservativeness gap Γ (Remark 3.3)
# report["clipping_needed"] → True iff clipped_project should be used
```

The *exact* positivity condition (Corollary 3.2) checks only the
paper-odd-indexed angles for κ > 0:

    min_{i paper-odd} α_i  >  κ/n

The *sufficient* condition (Theorem 3.1(iii)) conservatively checks all
angles.  When the smallest angle is even-indexed, the gap Γ > 0 can be
substantial and the sufficient condition may fail even when no clipping is
actually needed.

---

## Running the tests

```bash
pip install -e .
pytest tests/ -v
```

---

## License

See `LICENSE`.