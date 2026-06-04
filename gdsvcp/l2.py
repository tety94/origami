"""
gdsvcp.l2
=========
L² minimiser for the geometric edit distance.

Paper reference
---------------
Callegaro, S. "How Far Is a Single-Vertex Crease Pattern from Flat
Foldability? Minimal Angular Corrections under Three Norms." (2025).
  - Theorem 3.1     : unique L² minimiser δ* = −(κ/n) w,  d_G = |κ|√(2/n)
  - Corollary 3.2   : exact positivity condition  min_{i odd} α_i > κ/n
  - Remark 3.3      : conservativeness gap Γ

All angles in **radians**.
"""

from __future__ import annotations

import numpy as np

from gdsvcp.core import compute_kappa, weight_vector, PositivityViolation

__all__ = [
    "l2_minimiser",
    "dG_l2",
    "positivity_report",
]


def l2_minimiser(alpha: np.ndarray, etol: float = 1e-12) -> np.ndarray:
    """Compute the unconstrained L² minimiser δ* (Theorem 3.1(i)).

    The unique minimiser of ‖δ‖₂ over the affine constraint set

        H_κ = { δ ∈ ℝ^{2n} : 1ᵀδ = 0,  wᵀδ = −2κ }

    is  δ_i* = −(κ/n) w_i  for each index i  (Theorem 3.1(i)).

    If the exact positivity condition (Corollary 3.2) fails, the minimiser
    would drive at least one sector angle to zero or below.  In that case
    ``PositivityViolation`` is raised and the caller should use
    ``clipped_project`` instead (Theorem 3.1(iv)).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in **radians**, length 2n (n ≥ 1).
    etol : float
        Positivity lower bound; positivity is enforced as α_i + δ_i > etol.

    Returns
    -------
    np.ndarray
        Float64 perturbation vector δ* of length 2n.

    Raises
    ------
    PositivityViolation
        If α_i + δ_i* ≤ etol for any index i.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)

    if abs(kappa) < etol:
        return np.zeros(len(alpha), dtype=np.float64)

    w = weight_vector(n)
    delta_star = -(kappa / n) * w

    violated = np.where(alpha + delta_star <= etol)[0]
    if violated.size:
        raise PositivityViolation(
            f"Unconstrained L² minimiser violates positivity at 0-based indices "
            f"{violated.tolist()} (κ = {kappa:.6f} rad, κ/n = {kappa/n:.6f} rad). "
            "Use clipped_project() to obtain the constrained minimiser "
            "(Theorem 3.1(iv))."
        )
    return delta_star


def dG_l2(alpha: np.ndarray) -> float:
    """Compute the unconstrained L² geometric edit distance |κ|√(2/n) (Theorem 3.1(ii)).

    When the exact positivity condition (Corollary 3.2) holds, this equals
    the true d_G(C, F).  When clipping is required, this value is a lower
    bound; the true distance is ‖δ_clipped‖₂ ≥ |κ|√(2/n).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in **radians**, length 2n.

    Returns
    -------
    float
        d_G value (or lower bound) in **radians**.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    return float(abs(kappa) * np.sqrt(2.0 / n))


def positivity_report(alpha: np.ndarray) -> dict:
    """Return a structured report on the positivity feasibility of δ*.

    Implements Corollary 3.2 (exact condition), Theorem 3.1(iii) (sufficient
    condition), and the conservativeness gap Γ of Remark 3.3.

    The exact condition checks only the angles that δ* reduces:
      - κ > 0: only odd-indexed (0-based even) angles are decreased by κ/n;
               exact condition is  min_{i paper-odd} α_i > κ/n.
      - κ < 0: only even-indexed (0-based odd) angles are decreased by |κ|/n.
    The sufficient condition (Theorem 3.1(iii)) conservatively checks
    α_min over *all* indices.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in **radians**, length 2n.

    Returns
    -------
    dict
        ``kappa_rad``         : Kawasaki deficit κ (radians)
        ``shift_rad``         : per-crease correction |κ|/n (radians)
        ``min_odd_rad``       : min of paper-odd-indexed angles (radians)
        ``min_even_rad``      : min of paper-even-indexed angles (radians)
        ``exact_ok``          : True iff exact positivity condition holds
        ``sufficient_ok``     : True iff sufficient condition holds
        ``gamma_rad``         : conservativeness gap Γ (radians)
        ``clipping_needed``   : True iff clipping is required
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = len(alpha) // 2
    kappa = compute_kappa(alpha)
    shift = abs(kappa) / n

    # 0-based even indices → paper odd (+1); 0-based odd → paper even (−1)
    paper_odd_vals  = alpha[0::2]
    paper_even_vals = alpha[1::2]

    min_odd  = float(paper_odd_vals.min())
    min_even = float(paper_even_vals.min())
    alpha_min = float(alpha.min())

    if kappa > 0:
        exact_ok = min_odd > shift
        gamma = min_odd - alpha_min          # Remark 3.3
    elif kappa < 0:
        exact_ok = min_even > shift
        gamma = min_even - alpha_min
    else:
        exact_ok = True
        gamma = 0.0

    sufficient_ok = alpha_min > shift

    return {
        "kappa_rad":        kappa,
        "shift_rad":        shift,
        "min_odd_rad":      min_odd,
        "min_even_rad":     min_even,
        "exact_ok":         bool(exact_ok),
        "sufficient_ok":    bool(sufficient_ok),
        "gamma_rad":        float(gamma),
        "clipping_needed":  not bool(exact_ok),
    }