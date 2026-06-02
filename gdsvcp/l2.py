"""
l2.py – L² minimiser for the geometric edit distance.

Paper reference:
  "How Far Is a Single-Vertex Crease Pattern from Flat Foldability?
   Minimal Angular Corrections under Three Norms"
  Theorem 3.1 (main theorem), Corollary 3.2 (exact positivity condition),
  Remark 3.3 (conservativeness gap).

All angles in **radians**.
"""

from __future__ import annotations

import numpy as np

from gdsvcp.core import compute_kappa, weight_vector, PositivityViolation

__all__ = ["l2_minimiser", "dG_l2", "positivity_check"]


def l2_minimiser(alpha: np.ndarray, etol: float = 1e-12) -> np.ndarray:
    """Compute the unconstrained L² minimiser δ* (Theorem 3.1(i)).

    The unique minimiser of ‖δ‖₂ over the affine feasible subspace

        H_κ = { δ ∈ ℝ^{2n} : 1ᵀδ = 0,  wᵀδ = −2κ }

    is  δ_i* = −κ · w_i / n,  i = 1, …, 2n  (0-based: i = 0, …, 2n−1).

    Raises PositivityViolation if the exact positivity condition fails
    (Corollary 3.2): for κ > 0, this means min_{i odd} α_i ≤ κ/n;
    for κ < 0, min_{i even} α_i ≤ |κ|/n.
    Callers should then use `clipped_project` (Theorem 3.1(iv)).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n (n ≥ 1).
    etol : float
        Positivity tolerance (passed through to the violation check).

    Returns
    -------
    np.ndarray
        Float64 perturbation vector δ* of length 2n.

    Raises
    ------
    PositivityViolation
        If α_i + δ_i* ≤ etol for some index i.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    w = weight_vector(n)
    delta_star = -(kappa / n) * w

    violated = alpha + delta_star <= etol
    if np.any(violated):
        raise PositivityViolation(
            f"Unconstrained L² minimiser violates positivity at indices "
            f"{np.where(violated)[0].tolist()}. "
            "Use clipped_project() to obtain the constrained minimiser "
            "(Theorem 3.1(iv))."
        )
    return delta_star


def dG_l2(alpha: np.ndarray) -> float:
    """Compute the L² geometric edit distance d_G(C, F) = |κ| √(2/n).

    This is the ‖δ*‖₂ of the unconstrained minimiser (Theorem 3.1(ii)).
    When the exact positivity condition holds (Corollary 3.2), this equals
    the true geometric edit distance.  When clipping is required, the true
    d_G is ‖δ_clipped‖₂ ≥ |κ| √(2/n), and this formula gives a lower bound.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.

    Returns
    -------
    float
        d_G value (or lower bound) in radians.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    return float(abs(kappa) * np.sqrt(2.0 / n))


def positivity_check(alpha: np.ndarray) -> dict:
    """Check whether the unconstrained minimiser is positivity-feasible.

    Implements Corollary 3.2 (exact condition) and Theorem 3.1(iii)
    (sufficient condition), and computes the conservativeness gap Γ
    (Remark 3.3).

    Returns
    -------
    dict with keys:
        kappa           : Kawasaki deficit (radians)
        shift_per_crease: |κ|/n (the per-crease correction)
        min_odd_deg     : min_{i odd} α_i in degrees  (risk direction for κ>0)
        min_even_deg    : min_{i even} α_i in degrees (risk direction for κ<0)
        exact_ok        : True if the exact positivity condition holds
        sufficient_ok   : True if the conservative sufficient condition holds
        gamma_deg       : conservativeness gap Γ in degrees
        clipping_needed : True if clipping is required
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    shift = abs(kappa) / n

    odd_angles  = alpha[0::2]   # 0-based even index = paper odd
    even_angles = alpha[1::2]   # 0-based odd index  = paper even

    min_odd  = float(odd_angles.min())
    min_even = float(even_angles.min())
    alpha_min = float(alpha.min())

    if kappa > 0:
        # Odd sectors are corrected downward: risk at odd (0-based even indices)
        exact_ok      = min_odd > shift
        gamma         = min_odd - alpha_min          # Remark 3.3
    elif kappa < 0:
        # Even sectors are corrected downward: risk at even (0-based odd indices)
        exact_ok      = min_even > shift
        gamma         = min_even - alpha_min
    else:
        exact_ok      = True
        gamma         = 0.0

    sufficient_ok = alpha_min > shift   # Theorem 3.1(iii): checks all indices

    return {
        "kappa":            kappa,
        "shift_per_crease": shift,
        "min_odd_deg":      np.degrees(min_odd),
        "min_even_deg":     np.degrees(min_even),
        "exact_ok":         bool(exact_ok),
        "sufficient_ok":    bool(sufficient_ok),
        "gamma_deg":        float(np.degrees(gamma)),
        "clipping_needed":  not bool(exact_ok),
    }