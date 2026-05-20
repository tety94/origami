"""
core.py – Fundamental definitions for single-vertex crease patterns (SVCPs).

All angles are in **radians** unless explicitly noted.

Paper reference:
  "Geometric Edit Distance for Single-Vertex Crease Patterns under L^2"
  Sections 2–3.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PositivityViolation",
    "ClippingInfeasible",
    "weight_vector",
    "compute_kappa",
    "validate_svcp",
]

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PositivityViolation(Exception):
    """Raised when the unconstrained L² minimiser violates α_i + δ_i > 0."""


class ClippingInfeasible(Exception):
    """Raised when the active-set clipping algorithm cannot find a feasible solution."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * np.pi


def weight_vector(n: int) -> np.ndarray:
    """Return the weight vector **w** of length 2n (paper Definition 2.2).

    Uses **1-based** paper indexing: w_i = +1 for i odd, −1 for i even.
    In 0-based code (returned array) index k corresponds to paper index k+1,
    so w[k] = +1 when k is even (paper odd) and −1 when k is odd (paper even).

    Parameters
    ----------
    n : int
        Half-degree (number of mountain/valley pairs); must be ≥ 1.

    Returns
    -------
    np.ndarray
        Float64 array of length 2n with entries ±1.

    Examples
    --------
    >>> weight_vector(2)
    array([ 1., -1.,  1., -1.])
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}.")
    w = np.empty(2 * n, dtype=np.float64)
    # 0-based: even index → paper odd index (+1), odd index → paper even index (−1)
    w[0::2] = 1.0   # paper indices 1, 3, 5, …
    w[1::2] = -1.0  # paper indices 2, 4, 6, …
    return w


def compute_kappa(alpha: np.ndarray, etol: float = 1e-12) -> float:
    """Compute the Kawasaki deficit κ(C) = ½ w^T α (paper Lemma 2.3).

    Equivalently, κ = Σ_{i odd} α_i − π.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in **radians**, 0-based, length 2n (n ≥ 1).
    etol : float
        Numerical tolerance (unused here but kept for API consistency).

    Returns
    -------
    float
        Kawasaki deficit in radians; κ ∈ (−π, π) for any valid SVCP.

    Raises
    ------
    ValueError
        If len(alpha) is odd or < 2.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1 or len(alpha) < 2 or len(alpha) % 2 != 0:
        raise ValueError(
            f"alpha must be a 1-D array of even length ≥ 2, got shape {alpha.shape}."
        )
    w = weight_vector(len(alpha) // 2)
    return float(0.5 * w @ alpha)


def validate_svcp(alpha: np.ndarray, tol: float = 1e-12) -> None:
    """Validate that *alpha* is a well-formed SVCP (paper Definition 2.1).

    Checks:
      1. Even length ≥ 2.
      2. All α_i > 0 (within *tol*).
      3. Σ α_i ≈ 2π (within *tol*).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians.
    tol : float
        Numerical tolerance for positivity and sum checks.

    Raises
    ------
    ValueError
        On any structural violation.
    PositivityViolation
        If any α_i ≤ 0.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1 or len(alpha) < 2 or len(alpha) % 2 != 0:
        raise ValueError(
            f"alpha must be a 1-D array of even length ≥ 2, got shape {alpha.shape}."
        )
    if np.any(alpha <= tol):
        bad = np.where(alpha <= tol)[0]
        raise PositivityViolation(
            f"α_i must be strictly positive; violations at indices {bad.tolist()} "
            f"(values {alpha[bad].tolist()})."
        )
    total = float(np.sum(alpha))
    if not np.isclose(total, _TWO_PI, rtol=0.0, atol=tol):
        raise ValueError(
            f"Σ α_i must equal 2π ≈ {_TWO_PI:.6f}; got {total:.6f} "
            f"(difference {abs(total - _TWO_PI):.2e})."
        )