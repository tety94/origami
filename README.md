# origami — Computational Origami: Distance to Flat Foldability

A Python monorepo for quantifying and repairing **single-vertex crease patterns**
(SVCPs) that fail the flat-foldability conditions.  Two packages address two
complementary repair problems:

| Package | Question answered | Cost metric | Main result |
|---|---|---|---|
| [`cdsvcp`](./cdsvcp/) | *How many* crease insertions/deletions to satisfy Kawasaki? | count of operations | dC ∈ {0, 1, 2} |
| [`cdmsvcp`](./cdmsvcp/) | *How many* insertions + label flips to satisfy Kawasaki + Maekawa simultaneously? | count of operations | exact formula, 5 cases |
| [`gdsvcp`](./gdsvcp/) | *By how much* must the sector angles change to satisfy Kawasaki? | angular displacement (L¹ / L² / L∞) | dG = \|κ\| · c(n, p) |

All packages are based on original work by Stefano Callegaro:

> Callegaro, S. *"Two Creases Suffice: Edit Distance to Flat Foldability
> at a Single Vertex."* (2026).

> Callegaro, S. *"Crease Modifications to Full Flat-Foldability at a Single Vertex:
> Kawasaki and Maekawa Simultaneously."* (2026).

> Callegaro, S. *"How Far Is a Single-Vertex Crease Pattern from Flat Foldability?
> Minimal Angular Corrections under Three Norms."* (2026).

---

## Background

A **single-vertex crease pattern** (SVCP) is a cyclic sequence of positive sector
angles α₁, …, αₘ summing to 2π around a single interior vertex.  Full flat-foldability
requires two independent conditions to hold simultaneously:

- **Kawasaki–Justin:** the alternating angle sum S_odd = Σ αᵢ (i odd) equals π.
  This is a purely geometric condition on the sector angles; it is independent
  of which creases are mountains and which are valleys.
- **Maekawa:** the mountain/valley count satisfies |M − V| = 2.  This is a purely
  combinatorial condition on the labelling; it is independent of the specific angle
  values.

Neither condition implies the other.  Correcting one can disturb the other, which
is what makes the joint repair problem non-trivial.

---

## Repository structure

```
origami/
│
├── cdsvcp/                      combinatorial edit distance (Kawasaki only)
│   ├── cdsvcp/
│   │   ├── __init__.py
│   │   ├── svcp.py          SVCP, insert, delete, TwoInsert, repair
│   │   └── cli.py           cdsvcp <angles> [--repair] [--verbose]
│   ├── tests/
│   │   └── test_cdsvcp.py
│   ├── examples/
│   │   └── examples.py
│   ├── outputs/images/
│   ├── README.md
│   └── pyproject.toml
│
├── cdmsvcp/                     labelled edit distance (Kawasaki + Maekawa)
│   ├── cdmsvcp/
│   │   ├── __init__.py
│   │   ├── lsvcp.py         LSVCP, lsvcp_insert, lsvcp_flip, FullRepair
│   │   └── cli.py           cdmsvcp <angles> -- <labels> [--repair] [--verbose]
│   ├── tests/
│   │   └── test_cdmsvcp.py
│   ├── examples/
│   │   └── examples.py
│   ├── outputs/images/
│   ├── README.md
│   └── pyproject.toml
│
├── gdsvcp/                      geometric edit distance (L¹ / L² / L∞)
│   ├── ...
│   ├── README.md
│   └── pyproject.toml
│
├── LICENSE
└── README.md                    ← you are here
```

---

## Installation

`cdmsvcp` depends on `cdsvcp` for its geometric layer; install `cdsvcp` first:

```bash
pip install -e cdsvcp/
pip install -e cdmsvcp/
```

`gdsvcp` is independent:

```bash
pip install -e gdsvcp/
```

Requirements: Python ≥ 3.10.  `cdsvcp` and `cdmsvcp` use only the standard library
(all arithmetic is exact, via `fractions.Fraction`).  `gdsvcp` additionally requires
NumPy ≥ 1.24 and Matplotlib ≥ 3.7.

---

## Quick start

Both `cdsvcp` and `cdmsvcp` work with exact rational arithmetic.  Angles are given
as rational multiples of π: `Fraction(1, 2)` represents 90°, `Fraction(1)`
represents 180°, and so on.

### cdsvcp — geometry-only repair

```python
from fractions import Fraction
from cdsvcp import SVCP, edit_distance, repair

# C = (120°, 80°, 100°, 60°),  κ = 40°
C = SVCP((Fraction(2,3), Fraction(4,9), Fraction(5,9), Fraction(1,3)))

print(edit_distance(C))   # → 2

C_star, ops = repair(C)
print(C_star)             # flat-foldable SVCP
for op in ops:
    print(op)             # two ModCOp(kind='insert', ...)
```

### cdmsvcp — geometry + labelling repair

```python
from fractions import Fraction
from cdsvcp import SVCP
from cdmsvcp import LSVCP, M, V, edit_distance, repair

# Same angles, all mountains:  κ = 40°, δ = 4, ν = 2  →  d_MV = 2
C = SVCP((Fraction(2,3), Fraction(4,9), Fraction(5,9), Fraction(1,3)))
lc = LSVCP(C, (M, M, M, M))

print(edit_distance(lc))  # → 2

lc_star, ops = repair(lc)
for op in ops:
    print(op)  # LabelledOp(kind='insert', pos=..., param=..., label='V')
               # LabelledOp(kind='insert', pos=..., param=..., label='V')
```

---

## The two distances compared

### cdsvcp — combinatorial edit distance dC

Repairs the **geometry** (Kawasaki condition) by inserting or deleting
crease lines.  The mountain/valley assignment is ignored entirely.

The main theorem gives a complete three-way characterisation:

| m (crease count) | Condition | dC |
|---|---|---|
| even | κ = 0 | 0 |
| odd | — | 1 |
| even | κ ≠ 0 | 2 |

Algorithm `TwoInsert` achieves the minimum in O(m) time for every input.
The permitted operations are crease **insertion** and crease **deletion**.

### cdmsvcp — labelled edit distance d_MV

Repairs both the **geometry** (Kawasaki) and the **labelling** (Maekawa)
simultaneously.  The permitted operations are:

- **Labelled crease insertion**: split a sector into two and assign a
  freely chosen mountain/valley label to the new crease.
- **Label flip**: toggle an existing crease between mountain and valley
  without altering any angle.

Writing κ for the Kawasaki deficit, δ = M − V for the signed Maekawa count,
and ν = |δ| − 2 for the Maekawa deficit, the exact formula for even m is:

| κ | ν | d_MV |
|---|---|---|
| κ = 0 | ν = 0 | 0 |
| κ = 0 | ν ≠ 0 | \|ν\|/2 |
| κ ≠ 0 | \|ν\| ≤ 2 | 2 |
| κ ≠ 0 | \|ν\| ≥ 4 | \|ν\|/2 + 1 |

For odd m: d_MV = max(1, (|δ| − 1) / 2).

Algorithm `FullRepair` achieves the minimum in O(m + |ν|) time.
It calls `TwoInsert` from `cdsvcp` internally to find the geometrically
correct insertion positions, then selects the labels to simultaneously
satisfy Maekawa.

**Key structural difference from `cdsvcp`:** deletion is excluded in `cdmsvcp`.
A crease already scored on paper cannot be physically unscored; the model
reflects only the operations available to a folder correcting a pre-creased sheet.

---

## Command-line interfaces

```bash
# cdsvcp: angles as fractions of π, sum must equal 2
cdsvcp 1/2 1/2 1/2 1/2             # → dC = 0
cdsvcp 2/3 5/9 7/9                  # → dC = 1  (odd m)
cdsvcp 2/3 4/9 5/9 1/3 --repair    # → dC = 2, shows repaired pattern

# cdmsvcp: angles before '--', labels after '--'
cdmsvcp 1/2 1/2 1/2 1/2 -- M M M V             # → d_MV = 0
cdmsvcp 1/2 1/2 1/2 1/2 -- M M M M             # → d_MV = 1  (one flip)
cdmsvcp 2/3 4/9 5/9 1/3 -- M M M V --repair    # → d_MV = 2, shows repair
cdmsvcp 2/3 4/9 5/9 1/3 -- M M M M --repair    # → d_MV = 2, labels V V
cdmsvcp 2/3 2/3 2/3 -- M M M --repair          # → d_MV = 1  (odd m)
```

---

## Running all tests

```bash
pip install -e cdsvcp/ -e cdmsvcp/
pytest cdsvcp/tests/ cdmsvcp/tests/ -v
```

---

## gdsvcp — geometric edit distance

A third package, `gdsvcp`, addresses a different question: by how much must the
sector angles *themselves* change (without altering the crease count) to reach
flat-foldability?  It computes the geometric distance dG(C, F) under the L¹, L²,
and L∞ norms, and provides the `ClippedProject` algorithm for the constrained case.

> Callegaro, S. *"How Far Is a Single-Vertex Crease Pattern from Flat Foldability?
> Minimal Angular Corrections under Three Norms."* (2026).

See [`gdsvcp/README.md`](./gdsvcp/README.md) for details.

---

## License

See `LICENSE`.