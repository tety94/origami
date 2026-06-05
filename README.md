# origami — Computational Origami: Distance to Flat Foldability

A Python library for quantifying and repairing **single-vertex crease patterns** (SVCPs)
that fail the Kawasaki–Justin flat-foldability condition.

Two companion packages address the same repair problem from complementary angles
(pun intended):

| Package | Distance | Question answered |
|---|---|---|
| [`gdsvcp`](./gdsvcp/) | **geometric** (L¹ / L² / L∞) | *By how much* must the sector angles change? |
| [`cdsvcp`](./cdsvcp/) | **combinatorial** (count) | *How many* crease insertions/deletions are needed? |

Both are based on original work by Stefano Callegaro:

> Callegaro, S. *"How Far Is a Single-Vertex Crease Pattern from Flat Foldability?
> Minimal Angular Corrections under Three Norms."* (2026).

> Callegaro, S. *"Two Creases Suffice: Edit Distance to Flat Foldability
> at a Single Vertex."* (2026).

---

## Background

A **single-vertex crease pattern** is a cyclic sequence of positive sector angles
α₁, …, α_{2n} summing to 2π around a single interior vertex of a flat sheet of paper.
The Kawasaki–Justin theorem states that the vertex can be folded flat if and only if
the **Kawasaki deficit** κ = Σ_{i odd} α_i − π vanishes.

---

## Repository structure

```
origami/
├── gdsvcp/                  geometric edit distance (L¹ / L² / L∞)
│   ├── __init__.py
│   ├── core.py          weight_vector, compute_kappa, validate_svcp
│   ├── l2.py            l2_minimiser, dG_l2, positivity_report
│   ├── clipping.py      clipped_project, free_solve
│   ├── norms.py         linfty_minimiser, l1_minimisers
│   ├── plotting.py      plot_angle_shift, plot_crease_pattern
│   ├── io.py            read_angles_csv, write_angles_csv
│   ├── cli.py           gdsvcp <subcommand> ...
│   ├── tests/
│   ├── examples/
│   ├── outputs/images/
│   ├── README.md
│   └── pyproject.toml
│
├── cdsvcp/                  combinatorial edit distance (crease count)
│   ├── cdsvcp/
│   │   ├── __init__.py
│   │   ├── svcp.py          SVCP, insert, delete, TwoInsert, repair
│   │   └── cli.py           cdsvcp <angles> [--repair] [--verbose]
│   ├── tests/
│   ├── examples/
│   ├── outputs/images/
│   ├── README.md
│   └── pyproject.toml
│
│── LICENSE
└── README.md                ← you are here
```

---

## Installation

Each package is independent and can be installed on its own:

```bash
# geometric distance
pip install -e gdsvcp/

# combinatorial distance
pip install -e cdsvcp/

# or both at once
pip install -e gdsvcp/ -e cdsvcp/
```

Requirements: Python ≥ 3.10.  `gdsvcp` additionally requires NumPy ≥ 1.24
and Matplotlib ≥ 3.7; `cdsvcp` uses only the standard library.

---

## Quick comparison

Both packages work on the same pattern — the degree-6 vertex
C = (120°, 80°, 100°, 60°) with Kawasaki deficit κ = 40°:

```python
import numpy as np
from fractions import Fraction

# ── gdsvcp: how much to change the angles ──────────────────────────────
from gdsvcp import compute_kappa, dG_l2, clipped_project

alpha = np.radians([120., 80., 100., 60.])
print(f"κ  = {np.degrees(compute_kappa(alpha)):.4g}°")   # 40°
print(f"d_G(L²) = {np.degrees(dG_l2(alpha)):.4g}°")     # 40° (n=2)
delta = clipped_project(alpha)
print(np.degrees(alpha + delta))   # corrected angles

# ── cdsvcp: how many operations to change the topology ─────────────────
from cdsvcp import SVCP, edit_distance, repair

C = SVCP((Fraction(2,3), Fraction(4,9), Fraction(5,9), Fraction(1,3)))
print(edit_distance(C))            # 2
C_star, ops = repair(C)
print(C_star)                      # flat-foldable pattern
```

---

## The two distances at a glance

### Geometric distance (`gdsvcp`)

Fixes the **angular geometry** while keeping the crease graph topology fixed
(same number and arrangement of folds).  Perturbs each sector angle by δ_i,
minimising ‖δ‖_p for p ∈ {1, 2, ∞}.

| Norm | d_G(C, F) | Minimiser | Uniqueness |
|---|---|---|---|
| L² | \|κ\| √(2/n) | −(κ/n) **w** | unique |
| L∞ | \|κ\| / n | −(κ/n) **w** (same) | unique |
| L¹ | 2\|κ\| | any odd–even pair ±κ | n² choices |

When the unconstrained minimiser would drive a sector angle to zero, the
certified `ClippedProject` algorithm (≤ 2n − 2 active-set iterations)
recovers the true constrained minimiser.

### Combinatorial distance (`cdsvcp`)

Fixes the **combinatorial topology** by inserting or deleting creases, keeping
only the *count* of operations as the cost.

**Main theorem:** dC(C, F) ∈ {0, 1, 2} for every SVCP, with a sharp
three-way characterisation:

| Crease count | Condition | dC |
|---|---|---|
| even (m = 2n) | κ = 0 | 0 |
| odd (m ≥ 3) | always | 1 |
| even (m = 2n) | κ ≠ 0 | 2 |

The O(m) algorithm `TwoInsert` constructs an optimal flat-foldable pattern
for every input in exactly two Mod-C operations.

---

## When to use which package

Use **`gdsvcp`** when:
- the crease graph is fixed (e.g. a stamped or laser-cut design) and only
  angular tolerances need correction;
- you need a continuous, norm-aware measure of how far a vertex is from
  flat foldability;
- you are doing manufacturing tolerance analysis or stability studies.

Use **`cdsvcp`** when:
- you want to know whether the *structure* of the pattern needs to change
  (adding or removing fold lines);
- you need exact arithmetic guarantees (all computations use
  `fractions.Fraction`);
- you are studying the combinatorial geometry of crease patterns.

---

## Running all tests

```bash
pip install -e gdsvcp/ -e cdsvcp/
pytest gdsvcp/tests/ cdsvcp/tests/ -v
```

---

## License

See `LICENSE`.
