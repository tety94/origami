"""
gdsvcp.core
===========
Fundamental definitions for single-vertex crease patterns (SVCPs).

All angles are in **radians** throughout the entire package.
The CLI accepts degrees for human convenience and converts on entry;
all internal computations and all public-API return values are radians.

Paper reference
---------------
Callegaro, S. "How Far Is a Single-Vertex Crease Pattern from Flat
Foldability? Minimal Angular Corrections under Three Norms." (2025).
  - Definition 2.1   : SVCP and angle simplex Δ_{2n}
  - Definition 2.2   : weight vector w, Kawasaki deficit κ
  - Lemma 2.3        : κ = ½ wᵀα,  κ ∈ (−π, π)
  - Theorem 2.4      : Kawasaki–Justin flat-foldability criterion
  - Remark 2.4 (WLOG): cyclic shift maps κ → −κ

Index convention
----------------
The paper uses 1-based indices; code uses 0-based.
    paper index i  →  code index k = i − 1
    paper odd  (i odd)  →  code even (k even): w[k] = +1
    paper even (i even) →  code odd  (k odd):  w[k] = −1
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PositivityViolation",
    "ClippingInfeasible",
    "weight_vector",
    "compute_kappa",
    "validate_svcp",
    "is_flat_foldable",
    "cyclic_shift",
]

_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PositivityViolation(Exception):
    """Raised when a perturbation drives some α_i + δ_i to zero or below.

    The unconstrained L² minimiser can violate positivity when the Kawasaki
    deficit is large relative to the smallest sector angle (Theorem 3.1(iii)).
    Callers should then use ``clipped_project``.
    """


class ClippingInfeasible(Exception):
    """Raised when the active-set clipping algorithm cannot converge.

    Under the hypotheses of Theorem 4.6 (valid SVCP, n·ε_tol < π) this
    should never occur; it signals either an invalid input or a numerical
    degeneracy.
    """


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def weight_vector(n: int) -> np.ndarray:
    """Return the Kawasaki weight vector **w** of length 2n (Definition 2.2).

    The paper uses 1-based indices: w_i = +1 for i odd, −1 for i even.
    The returned array is 0-based: w[k] = +1 when k is even (paper odd)
    and w[k] = −1 when k is odd (paper even).

    Parameters
    ----------
    n : int
        Half-degree; must be ≥ 1.

    Returns
    -------
    np.ndarray
        Float64 array of length 2n with entries ∈ {+1, −1}.

    Examples
    --------
    >>> weight_vector(3)
    array([ 1., -1.,  1., -1.,  1., -1.])
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}.")
    w = np.empty(2 * n, dtype=np.float64)
    w[0::2] = +1.0   # 0-based even → paper odd  → +1
    w[1::2] = -1.0   # 0-based odd  → paper even → −1
    return w


def compute_kappa(alpha: np.ndarray) -> float:
    """Compute the Kawasaki deficit κ(C) = Σ_{i odd} α_i − π (Lemma 2.3).

    Equivalently, κ = ½ wᵀα where w is the weight vector.  For any valid
    SVCP κ ∈ (−π, π) strictly (Lemma 2.3).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in **radians**, 0-based, length 2n (n ≥ 1).

    Returns
    -------
    float
        Kawasaki deficit in radians.

    Raises
    ------
    ValueError
        If ``alpha`` has odd length or length < 2.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1 or len(alpha) < 2 or len(alpha) % 2 != 0:
        raise ValueError(
            f"alpha must be a 1-D array of even length ≥ 2; got shape {alpha.shape}."
        )
    n = len(alpha) // 2
    # Sum of 0-based even indices = sum of paper odd-indexed angles
    return float(alpha[0::2].sum() - np.pi)


def validate_svcp(alpha: np.ndarray, tol: float = 1e-12) -> None:
    """Validate that *alpha* is a well-formed SVCP (Definition 2.1).

    Checks:
      1. 1-D array of even length ≥ 2.
      2. All α_i > 0 (strictly, within *tol*).
      3. Σ α_i = 2π (within *tol*).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians.
    tol : float
        Numerical tolerance for checks (2) and (3).

    Raises
    ------
    ValueError
        If the length or sum condition fails.
    PositivityViolation
        If any α_i ≤ tol.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1 or len(alpha) < 2 or len(alpha) % 2 != 0:
        raise ValueError(
            f"alpha must be a 1-D array of even length ≥ 2; got shape {alpha.shape}."
        )
    bad = np.where(alpha <= tol)[0]
    if bad.size:
        raise PositivityViolation(
            f"All α_i must be strictly positive; violations at 0-based indices "
            f"{bad.tolist()} (values {alpha[bad].tolist()})."
        )
    total = float(alpha.sum())
    if not np.isclose(total, _TWO_PI, rtol=0.0, atol=tol):
        raise ValueError(
            f"Σ α_i must equal 2π ≈ {_TWO_PI:.10f}; "
            f"got {total:.10f} (difference {abs(total - _TWO_PI):.2e})."
        )


def is_flat_foldable(alpha: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True iff C is locally flat-foldable (Theorem 2.4).

    Equivalent to |κ(C)| < *tol*.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.
    tol : float
        Numerical tolerance for the κ = 0 check.
    """
    return abs(compute_kappa(alpha)) < tol


def cyclic_shift(alpha: np.ndarray) -> np.ndarray:
    """Apply the WLOG cyclic shift α_i ← α_{i+1 mod 2n} (Remark 2.4).

    This maps the Kawasaki weight vector w → −w and hence κ → −κ, so it
    converts a pattern with κ < 0 into one with κ > 0 while preserving
    the total angle sum and all L^p distances.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.

    Returns
    -------
    np.ndarray
        Cyclically shifted copy (``np.roll(alpha, -1)``).
    """
    return np.roll(np.asarray(alpha, dtype=np.float64), -1)