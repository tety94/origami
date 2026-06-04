"""
cdsvcp.svcp
===========
Combinatorial edit distance for single-vertex crease patterns (SVCPs)
under the Mod-C operations (crease insertion and deletion).

Implements:
  - SVCP         : data class for a single-vertex crease pattern
  - ModCOp       : named-tuple representing one Mod-C operation
  - kawasaki_deficit  : computes κ(C) = S_odd - π  (even m only)
  - kawasaki_residual : computes K(C) = S_odd - π  (all m ≥ 2)
  - is_flat_foldable  : Kawasaki–Justin criterion
  - edit_distance     : dC(C, F) ∈ {0, 1, 2}
  - two_insert        : Algorithm TwoInsert — O(m) optimal repair

All angle arithmetic uses fractions.Fraction (exact rational multiples
of π) to match the exact-arithmetic model of the paper.

References
----------
Callegaro, S. "Two Creases Suffice: Edit Distance to Flat Foldability
at a Single Vertex." (2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal constant: π represented as Fraction(1) in units of π.
# All angles are stored as Fraction multiples of π.
# e.g. 90° = π/2  →  Fraction(1, 2)
#      180° = π   →  Fraction(1, 1)
# ---------------------------------------------------------------------------
_PI = Fraction(1)   # sentinel: "one π"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SVCP:
    """
    Single-Vertex Crease Pattern.

    Parameters
    ----------
    angles : tuple of Fraction
        Sector angles in cyclic order, each a positive rational multiple
        of π.  Their sum must equal 2π (i.e. sum == Fraction(2)).

    Examples
    --------
    >>> from fractions import Fraction
    >>> C = SVCP((Fraction(1,2), Fraction(1,2), Fraction(1,2), Fraction(1,2)))
    >>> is_flat_foldable(C)
    True
    """
    angles: Tuple[Fraction, ...]

    def __post_init__(self):
        if len(self.angles) < 2:
            raise ValueError("An SVCP must have at least 2 sector angles.")
        if any(a <= 0 for a in self.angles):
            raise ValueError("All sector angles must be strictly positive.")
        total = sum(self.angles)
        if total != Fraction(2):
            raise ValueError(
                f"Sector angles must sum to 2π (Fraction(2)); got {total}."
            )

    def __len__(self) -> int:
        return len(self.angles)

    def __repr__(self) -> str:
        degs = [f"{float(a)*180:.4g}°" for a in self.angles]
        return f"SVCP([{', '.join(degs)}])"


@dataclass(frozen=True)
class ModCOp:
    """
    One Mod-C operation.

    kind   : 'insert' or 'delete'
    pos    : 1-based position in the pattern (before the operation)
    param  : insertion parameter s ∈ (0, α_pos)  [None for deletion]
    """
    kind:  str          # 'insert' | 'delete'
    pos:   int          # 1-based
    param: Optional[Fraction] = None

    def __post_init__(self):
        if self.kind not in ('insert', 'delete'):
            raise ValueError("kind must be 'insert' or 'delete'.")
        if self.kind == 'insert' and self.param is None:
            raise ValueError("Insertion requires a param value.")


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def sodd(C: SVCP) -> Fraction:
    """Sum of odd-indexed sector angles (1-based)."""
    return sum(C.angles[i] for i in range(0, len(C), 2))   # indices 0,2,4,…


def seven(C: SVCP) -> Fraction:
    """Sum of even-indexed sector angles (1-based)."""
    return sum(C.angles[i] for i in range(1, len(C), 2))   # indices 1,3,5,…


def kawasaki_residual(C: SVCP) -> Fraction:
    """K(C) = S_odd(C) − π, defined for all m ≥ 2."""
    return sodd(C) - _PI


def kawasaki_deficit(C: SVCP) -> Fraction:
    """
    κ(C) = S_odd(C) − π.

    Only meaningful when |C| is even (Kawasaki–Justin requires even m).
    Raises ValueError for odd m.
    """
    if len(C) % 2 != 0:
        raise ValueError(
            "kawasaki_deficit is defined only for even crease count."
        )
    return kawasaki_residual(C)


def is_flat_foldable(C: SVCP) -> bool:
    """
    Kawasaki–Justin criterion: C ∈ F  iff  |C| is even and κ(C) = 0.
    """
    return len(C) % 2 == 0 and kawasaki_residual(C) == 0


# ---------------------------------------------------------------------------
# Mod-C primitive operations
# ---------------------------------------------------------------------------

def insert(C: SVCP, pos: int, s: Fraction) -> SVCP:
    """
    Crease insertion at 1-based position *pos* with parameter *s*.

    Splits α_pos → (α_pos − s,  s).
    Returns a new SVCP of length |C| + 1.
    """
    m = len(C)
    if not (1 <= pos <= m):
        raise IndexError(f"pos={pos} out of range [1, {m}].")
    a = C.angles[pos - 1]
    if not (Fraction(0) < s < a):
        raise ValueError(
            f"Insertion parameter s={s} not in (0, {a})."
        )
    new_angles = (
        C.angles[:pos - 1]
        + (a - s, s)
        + C.angles[pos:]
    )
    return SVCP(new_angles)


def delete(C: SVCP, pos: int) -> SVCP:
    """
    Crease deletion at 1-based position *pos*.

    Merges α_pos and its cyclic successor α_{[pos+1]_m} into one sector.
    Requires |C| ≥ 3.
    Returns a new SVCP of length |C| − 1.
    """
    m = len(C)
    if m < 3:
        raise ValueError(
            "Deletion is prohibited when |C| = 2 (result would have < 2 angles)."
        )
    if not (1 <= pos <= m):
        raise IndexError(f"pos={pos} out of range [1, {m}].")
    # cyclic successor index (0-based)
    succ = pos % m          # = (pos+1-1) mod m  in 0-based
    merged = C.angles[pos - 1] + C.angles[succ]
    if pos < m:
        new_angles = C.angles[:pos - 1] + (merged,) + C.angles[pos + 1:]
    else:
        # wrap-around: merge last with first; merged goes to position m-1
        new_angles = C.angles[1:m - 1] + (merged,)
    return SVCP(new_angles)


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def edit_distance(C: SVCP) -> int:
    """
    Return dC(C, F) ∈ {0, 1, 2}.

    Theorem (Callegaro 2025):
      • dC = 0  iff C ∈ F
      • dC = 1  iff |C| is odd
      • dC = 2  iff |C| is even and κ(C) ≠ 0
    """
    if is_flat_foldable(C):
        return 0
    if len(C) % 2 == 1:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Insertion-parameter recurrence for odd-length patterns
# (Lemma 3.4 / Alternation Lemma in the paper)
# ---------------------------------------------------------------------------

def _compute_sj(C: SVCP) -> List[Fraction]:
    """
    Compute the insertion-parameter sequence s_1, …, s_{m'} for an
    odd-length SVCP C' (m' = |C| must be odd, ≥ 3).

    Uses the recurrence s_{j+1} = β_{j+1} − s_j anchored at
    s_1 = β_1 − K(C').
    """
    m = len(C)
    assert m % 2 == 1 and m >= 3, "_compute_sj requires odd m ≥ 3"
    betas = C.angles
    Kres = kawasaki_residual(C)
    s = [Fraction(0)] * (m + 1)   # 1-indexed
    s[1] = betas[0] - Kres
    for j in range(1, m):
        s[j + 1] = betas[j] - s[j]
    return s[1:]   # return s[1..m] as 0-indexed list


def _find_feasible_j(C: SVCP) -> int:
    """
    Find the smallest feasible index j* in {1,…,|C|} (or {2,…,|C|−1}
    if K=0) for crease insertion in an odd-length pattern.

    Returns j* (1-based).
    """
    m = len(C)
    Kres = kawasaki_residual(C)
    sj = _compute_sj(C)      # 0-indexed: sj[0] = s_1, …, sj[m-1] = s_m

    if Kres != 0:
        search = range(m)            # j* in {1,…,m}
    else:
        search = range(1, m - 1)     # j* in {2,…,m-1}

    for idx in search:
        j = idx + 1          # 1-based
        s = sj[idx]
        beta = C.angles[idx]
        if Fraction(0) < s < beta:
            return j

    # Should never reach here given the existence theorem
    raise RuntimeError(  # pragma: no cover
        f"No feasible insertion index found for {C}. "
        "This indicates a bug."
    )


# ---------------------------------------------------------------------------
# Algorithm TwoInsert  (O(m), returns C* ∈ F in exactly 2 Mod-C ops)
# ---------------------------------------------------------------------------

def two_insert(C: SVCP) -> Tuple[SVCP, List[ModCOp]]:
    """
    Algorithm TwoInsert (Callegaro 2025, Algorithm 1).

    Given an SVCP C with even |C| = 2n ≥ 2 and κ(C) ≠ 0, returns
    (C*, ops) where C* ∈ F and ops is the list of exactly two Mod-C
    operations applied.

    The function first normalises κ > 0 via cyclic relabelling
    (which does not change dC).

    Parameters
    ----------
    C : SVCP
        Must have even crease count and κ(C) ≠ 0.

    Returns
    -------
    C_star : SVCP
        A flat-foldable pattern reachable in 2 Mod-C operations.
    ops : list of ModCOp
        The two operations in order.

    Raises
    ------
    ValueError
        If |C| is odd (use single_insert for odd inputs).
    ValueError
        If C is already flat-foldable (dC = 0).
    """
    m = len(C)
    if m % 2 != 0:
        raise ValueError(
            "two_insert requires even crease count. "
            "For odd inputs, use single_insert."
        )
    kap = kawasaki_deficit(C)
    if kap == 0:
        raise ValueError("C is already flat-foldable (dC = 0).")

    # ---- normalise: ensure κ > 0 via cyclic relabelling ----
    relabelled = False
    if kap < 0:
        C = SVCP(C.angles[1:] + C.angles[:1])
        kap = -kap
        relabelled = True

    n = m // 2
    angles = C.angles

    # ------------------------------------------------------------------
    # Case 1: n = 1
    # C = (π + κ, π − κ).  Output: ((π+κ)/2, (π+κ)/2, (π−κ)/2, (π−κ)/2)
    # ------------------------------------------------------------------
    if n == 1:
        half_p = (kap + _PI) / 2     # (π + κ)/2
        half_m = (_PI - kap) / 2     # (π − κ)/2
        # Two insertions: first split α_1, then split new α_3
        C1 = insert(C, 1, half_p)    # (half_p, half_p, π−κ)  — wait, let's be precise
        # After inserting s=(π+κ)/2 into α_1=(π+κ):
        #   new pattern = ((π+κ)/2, (π+κ)/2, π−κ)  [length 3, odd]
        # Then insert s=(π−κ)/2 into α_3 = (π−κ):
        C_star = insert(C1, 3, half_m)
        ops = [
            ModCOp('insert', 1, half_p),
            ModCOp('insert', 3, half_m),
        ]
        return C_star, ops

    # ------------------------------------------------------------------
    # Case 2: n ≥ 2, scan for odd p with α_p > κ
    # ------------------------------------------------------------------
    p = None
    for i in range(0, m, 2):      # 0-based odd positions: 0,2,4,…
        if angles[i] > kap:
            p = i + 1              # 1-based
            break

    if p is not None:
        q = p + 1                  # even position (1-based), ≤ m by Lemma 3.1
        alpha_p = angles[p - 1]
        alpha_q = angles[q - 1]
        Delta = kap + alpha_q

        # Primary choice: s* = (κ + α_p)/2
        s = (kap + alpha_p) / 2
        t = Delta - s

        if t <= 0:
            # Fallback: s_fb = κ + α_q/2,  t_fb = α_q/2
            s = kap + alpha_q / 2
            t = alpha_q / 2

        # Apply: first split α_p, then split α_q (now at position q+1)
        C1 = insert(C, p, s)       # α_p → (α_p−s, s) at positions p, p+1
        # After inserting at p, the former α_q was at position q = p+1;
        # it has shifted to position q+1 = p+2.
        C_star = insert(C1, q + 1, t)
        ops = [
            ModCOp('insert', p, s),
            ModCOp('insert', q + 1, t),
        ]
        return C_star, ops

    # ------------------------------------------------------------------
    # Case 3: n ≥ 3, M = κ  (all odd angles equal κ = π/(n−1))
    # One deletion at position 1, then one insertion.
    # ------------------------------------------------------------------
    M = max(angles[i] for i in range(0, m, 2))
    if n >= 3 and M == kap:
        C_del = delete(C, 1)       # merge α_1 and α_2
        j_star = _find_feasible_j(C_del)
        sj = _compute_sj(C_del)
        s_star = sj[j_star - 1]
        C_star = insert(C_del, j_star, s_star)
        ops = [
            ModCOp('delete', 1),
            ModCOp('insert', j_star, s_star),
        ]
        return C_star, ops

    # ------------------------------------------------------------------
    # Case 2b: n ≥ 3, M < κ
    # Two insertions: first at any even position j0 (avoid forbidden r),
    # then find j** in the resulting odd-length pattern.
    # ------------------------------------------------------------------
    # Find j0 ∈ I_even (1-based even index with α_{j0} > 0, always exists)
    j0 = 2   # simplest choice: first even position is always valid

    alpha_j0 = angles[j0 - 1]
    # Forbidden parameter: r such that K(C_1) = 0
    # From Lemma SoddChange(ii) with j0 even:
    #   K(C_1) = kap + r + E   where E = Σ_{i>j0, even} - Σ_{i>j0, odd}
    E = sum(angles[i] for i in range(j0 + 1, m, 2)) \
      - sum(angles[i] for i in range(j0,     m, 2))
    r_bad = -kap - E

    r0 = alpha_j0 / 2
    if r_bad == r0:
        r0 = alpha_j0 / 3

    C1 = insert(C, j0, r0)        # length m+1 (odd)
    j_star = _find_feasible_j(C1)
    sj = _compute_sj(C1)
    s_star = sj[j_star - 1]
    C_star = insert(C1, j_star, s_star)
    ops = [
        ModCOp('insert', j0, r0),
        ModCOp('insert', j_star, s_star),
    ]
    return C_star, ops


# ---------------------------------------------------------------------------
# Single insertion for odd-length inputs  (dC = 1)
# ---------------------------------------------------------------------------

def single_insert(C: SVCP) -> Tuple[SVCP, ModCOp]:
    """
    Repair an odd-crease SVCP by one insertion (dC = 1).

    Parameters
    ----------
    C : SVCP
        Must have odd crease count m ≥ 3.

    Returns
    -------
    C_star : SVCP
    op     : ModCOp
    """
    if len(C) % 2 == 0:
        raise ValueError(
            "single_insert requires odd crease count. "
            "For even inputs, use two_insert."
        )
    j_star = _find_feasible_j(C)
    sj = _compute_sj(C)
    s_star = sj[j_star - 1]
    C_star = insert(C, j_star, s_star)
    op = ModCOp('insert', j_star, s_star)
    return C_star, op


# ---------------------------------------------------------------------------
# Convenience: repair any SVCP optimally
# ---------------------------------------------------------------------------

def repair(C: SVCP) -> Tuple[SVCP, List[ModCOp]]:
    """
    Repair C to the nearest flat-foldable pattern using the minimum
    number of Mod-C operations.

    Returns (C*, ops) where ops has length dC(C, F).
    """
    if is_flat_foldable(C):
        return C, []
    if len(C) % 2 == 1:
        C_star, op = single_insert(C)
        return C_star, [op]
    C_star, ops = two_insert(C)
    return C_star, ops
