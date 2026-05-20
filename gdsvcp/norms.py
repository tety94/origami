"""
norms.py – L¹ and L∞ minimisers for the geometric edit distance.

Paper reference:
  "Geometric Edit Distance for Single-Vertex Crease Patterns under L^2"
  Section 5 (Other Norms).

All angles in **radians**.
"""

from __future__ import annotations

import numpy as np
from itertools import product as iproduct

from gdsvcp.core import compute_kappa, weight_vector

__all__ = ["l1_minimisers", "enumerate_l1_minimiser_polytope", "linfty_minimiser"]


def l1_minimisers(alpha: np.ndarray) -> list[np.ndarray]:
    """Return all extreme-point L¹ minimisers (Proposition 5.2).

    Each extreme point has exactly two non-zero entries:
      δ_p = −κ  (p is an odd-indexed position, 0-based even)
      δ_q = +κ  (q is an even-indexed position, 0-based odd)
      δ_i = 0   for i ∉ {p, q}

    There are n² extreme points in total.

    The minimum L¹ value is 2|κ| (Proposition 5.1):
      lower bound: |w^T δ| = 2|κ| ≤ ‖w‖_∞ ‖δ‖₁ = ‖δ‖₁
      upper bound: attained by any (p, q) pair.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.

    Returns
    -------
    list of np.ndarray
        All n² extreme L¹ minimisers, each of length 2n.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)

    # 0-based odd indices → paper odd (w=+1); 0-based even → paper even (w=−1)
    odd_indices = list(range(0, 2 * n, 2))   # paper odd: w = +1
    even_indices = list(range(1, 2 * n, 2))  # paper even: w = −1

    minimisers = []
    for p, q in iproduct(odd_indices, even_indices):
        delta = np.zeros(2 * n, dtype=np.float64)
        delta[p] = -kappa
        delta[q] = +kappa
        minimisers.append(delta)
    return minimisers


def enumerate_l1_minimiser_polytope(alpha: np.ndarray):
    """Generator over all extreme L¹ minimisers (same as l1_minimisers, lazy).

    Yields
    ------
    np.ndarray
        Each extreme point δ^{(p,q)}.
    """
    yield from l1_minimisers(alpha)


def linfty_minimiser(alpha: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the unique L∞ minimiser (Propositions 5.3 and 5.4).

    The unique minimiser over H_κ under ‖·‖_∞ is δ* = −(κ/n) w,
    identical to the L² minimiser, with value |κ|/n.

    Proof sketch (Prop 5.4):
      Lower bound: 2|κ| = |w^T δ| ≤ Σ |w_i||δ_i| ≤ 2n ‖δ‖_∞  ⟹  ‖δ‖_∞ ≥ |κ|/n.
      Uniqueness: summing odd and even constraints forces δ_i = ∓κ/n for all i.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.

    Returns
    -------
    (delta, value) : tuple[np.ndarray, float]
        delta : the unique L∞ minimiser of length 2n.
        value : ‖delta‖_∞ = |κ|/n.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    w = weight_vector(n)
    delta = -(kappa / n) * w
    value = abs(kappa) / n
    return delta, value