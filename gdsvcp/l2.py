"""
l2.py – L² minimiser for the geometric edit distance.

Paper reference:
  "Geometric Edit Distance for Single-Vertex Crease Patterns under L^2"
  Theorem 3.1 (main theorem), Section 3.

All angles in **radians**.
"""

from __future__ import annotations

import numpy as np

from gdsvcp.core import compute_kappa, weight_vector, PositivityViolation

__all__ = ["l2_minimiser", "dG_l2"]


def l2_minimiser(alpha: np.ndarray, etol: float = 1e-12) -> np.ndarray:
    """Compute the unconstrained L² minimiser δ* (Theorem 3.1 part (i)).

    The unique minimiser of ‖δ‖₂ over the affine feasible subspace

        H_κ = { δ ∈ ℝ^{2n} : 1^T δ = 0,  w^T δ = −2κ }

    is  δ_i* = −κ · w_i / n,  i = 1, …, 2n.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n (n ≥ 1).
    etol : float
        Positivity tolerance; raises PositivityViolation if any
        α_i + δ_i* ≤ etol.

    Returns
    -------
    np.ndarray
        Float64 perturbation vector δ* of length 2n.

    Raises
    ------
    PositivityViolation
        If the unconstrained minimiser violates α_i + δ_i* > etol for some i.
        Callers should then use `clipped_project` to obtain the constrained
        minimiser (Theorem 3.1 part (iv)).
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
            "Use clipped_project() to obtain the constrained minimiser."
        )
    return delta_star


def dG_l2(alpha: np.ndarray) -> float:
    """Compute the L² geometric edit distance d_G(C, F) = |κ| √(2/n).

    This is the norm of the unconstrained minimiser δ* (Theorem 3.1 part (ii)).
    Note: this formula is always valid as a lower bound; the true d_G equals
    this value when positivity holds (Theorem 3.1 part (iii)).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.

    Returns
    -------
    float
        d_G value in radians.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    return abs(kappa) * np.sqrt(2.0 / n)