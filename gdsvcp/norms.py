"""
norms.py – L¹ and L∞ minimisers for the geometric edit distance.

Paper reference:
  "How Far Is a Single-Vertex Crease Pattern from Flat Foldability?
   Minimal Angular Corrections under Three Norms"
  Section 5 (Other Norms):
    Proposition 5.1  – L∞ minimum |κ|/n and uniqueness
    Proposition 5.2  – L²–L∞ minimiser coincidence
    Proposition 5.3  – L¹ minimum 2|κ| and minimiser polytope P
    Remark 5.4       – L¹ feasibility condition α_p > |κ|

All angles in **radians**.
"""

from __future__ import annotations

import numpy as np
from itertools import product as iproduct

from gdsvcp.core import compute_kappa, weight_vector

__all__ = [
    "linfty_minimiser",
    "l1_minimisers",
    "l1_feasible_minimisers",
    "enumerate_l1_minimiser_polytope",
]


def linfty_minimiser(alpha: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the unique L∞ minimiser (Propositions 5.1–5.2).

    The unique minimiser of ‖δ‖_∞ over H_κ is δ* = −(κ/n) w,
    identical to the L² minimiser, with ‖δ*‖_∞ = |κ|/n.

    Proof sketch (Prop. 5.1):
      Lower bound: 2|κ| = |wᵀδ| ≤ ‖w‖₁ ‖δ‖_∞ = 2n ‖δ‖_∞  ⟹  ‖δ‖_∞ ≥ |κ|/n.
      Equality in Hölder iff |δ_i| = ‖δ‖_∞ for all i and w_iδ_i ≤ 0 for all i,
      which uniquely determines δ_i = −(κ/n)w_i  (Prop. 5.2).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.

    Returns
    -------
    (delta, value)
        delta : unique L∞ minimiser, length 2n.
        value : ‖delta‖_∞ = |κ|/n.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    w = weight_vector(n)
    delta = -(kappa / n) * w
    value = float(abs(kappa) / n)
    return delta, value


def l1_minimisers(alpha: np.ndarray) -> list[np.ndarray]:
    """Return all n² extreme-point L¹ minimisers (Proposition 5.3).

    Each extreme point δ^{(p,q)} has exactly two non-zero entries:
        δ_p = −κ   (p is a 0-based even index, i.e. paper odd, w_p = +1)
        δ_q = +κ   (q is a 0-based odd  index, i.e. paper even, w_q = −1)
        δ_i = 0    for i ∉ {p, q}

    The minimum L¹ value is 2|κ| (Proposition 5.3):
        lower bound: |wᵀδ| = 2|κ| ≤ ‖w‖_∞ ‖δ‖₁ = ‖δ‖₁
        upper bound: attained by any (p, q) pair above.

    All n² pairs are returned; note that some may be positivity-infeasible
    (α_p + δ_p = α_p − κ ≤ 0).  Use `l1_feasible_minimisers` to filter.

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

    # 0-based even indices → paper odd (w = +1)
    # 0-based odd  indices → paper even (w = −1)
    paper_odd_indices  = list(range(0, 2 * n, 2))
    paper_even_indices = list(range(1, 2 * n, 2))

    minimisers = []
    for p, q in iproduct(paper_odd_indices, paper_even_indices):
        delta = np.zeros(2 * n, dtype=np.float64)
        delta[p] = -kappa
        delta[q] = +kappa
        minimisers.append(delta)
    return minimisers


def l1_feasible_minimisers(alpha: np.ndarray, etol: float = 0.0) -> list[np.ndarray]:
    """Return the positivity-feasible extreme-point L¹ minimisers (Remark 5.4).

    A vertex minimiser δ^{(p,q)} is feasible iff α_p + δ_p = α_p − κ > etol.
    The even index q is always feasible since α_q + κ > α_q > 0.

    At least one feasible pair always exists for any valid SVCP, except in
    the degenerate edge case described in Remark 5.4 (all odd α_i ≤ κ, which
    requires (n−1)κ ≥ π, i.e. κ close to π/(n−1)).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.
    etol : float
        Positivity lower bound; default 0.0 (strict positivity).

    Returns
    -------
    list of np.ndarray
        Feasible extreme L¹ minimisers.  Empty only in the degenerate edge case.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    kappa = compute_kappa(alpha)
    all_mins = l1_minimisers(alpha)
    return [
        d for d in all_mins
        if np.all(alpha + d > etol)
    ]


def enumerate_l1_minimiser_polytope(alpha: np.ndarray):
    """Generator over all n² extreme L¹ minimisers (lazy version of l1_minimisers).

    Yields
    ------
    np.ndarray
        Each extreme point δ^{(p,q)}, length 2n.
    """
    yield from l1_minimisers(alpha)