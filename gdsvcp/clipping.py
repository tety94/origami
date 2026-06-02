"""
clipping.py – Active-set clipping projection (Algorithm 4.1 / Algorithm 4.2).

When the unconstrained L² minimiser violates α_i + δ_i ≥ ε_tol, this module
computes the constrained minimiser over K_{ε_tol} via a two-phase active-set
method that exactly implements the paper algorithm.

Phase 1 (primal feasibility):
  Iteratively clip the most-violated index (argmin), re-solve FreeSolve on the
  remaining free variables.  Terminates in ≤ n-1 steps (Lemma 4.3 / joint
  induction).

Phase 2 (dual feasibility):
  Check KKT multipliers ν_i = 2δ_i + λ + μ w_i for all active indices.
  Release any index with ν_i < 0 and re-solve.  Terminates in ≤ n-1 steps.

Paper reference:
  "How Far Is a Single-Vertex Crease Pattern from Flat Foldability?
   Minimal Angular Corrections under Three Norms"
  Algorithm 4.1 (ClippedProject), Algorithm 4.2 (FreeSolve),
  Proposition 4.3 (free-variable KKT), Lemma 4.4 (joint induction),
  Lemma 4.5 (no-monoparity), Theorem 4.6 (termination),
  Proposition 4.7 (KKT optimality).

All angles in **radians**.
"""

from __future__ import annotations

import logging
import numpy as np

from gdsvcp.core import compute_kappa, weight_vector, ClippingInfeasible

__all__ = ["clipped_project"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FreeSolve (Algorithm 4.2 / Proposition 4.3)
# ---------------------------------------------------------------------------

def _free_solve(
    alpha: np.ndarray,
    w: np.ndarray,
    active: set[int],
    kappa: float,
    etol: float,
) -> tuple[np.ndarray, float, float]:
    """Solve the free-variable KKT sub-problem (Algorithm 4.2).

    Given active set A with clipped values δ_i = ε_tol − α_i for i ∈ A,
    find delta_i for i in F (free set) minimising sum_{F} delta_i^2 subject to:

        Σ_{F} δ_i       = R₁   (total-sum residual)
        Σ_{F} w_i δ_i   = R₂   (Kawasaki residual)

    Residuals (eq. Proposition 4.3):
        R₁ = Σ_{i ∈ A} (α_i − ε_tol)
        R₂ = −2κ − Σ_{i ∈ A} w_i (ε_tol − α_i)

    Returns
    -------
    (delta, lambda_, mu)
        delta   : full perturbation array of length 2n (active entries set to
                  ε_tol − α_i, free entries set to the KKT solution).
        lambda_ : Lagrange multiplier for the sum constraint.
        mu      : Lagrange multiplier for the Kawasaki constraint.

    Raises
    ------
    ClippingInfeasible
        If nF_plus = 0 or nF_minus = 0, violating the no-monoparity lemma.
        This should never occur for a valid SVCP with n·ε_tol < π.
    """
    N = len(alpha)
    delta = np.empty(N, dtype=np.float64)

    # Set clipped values
    for i in active:
        delta[i] = etol - alpha[i]

    free = [i for i in range(N) if i not in active]
    nF = len(free)

    # Compute residuals (Proposition 4.3)
    R1 = sum(alpha[i] - etol for i in active)            # > 0 when A non-empty
    R2 = -2.0 * kappa - sum(w[i] * (etol - alpha[i]) for i in active)

    nF_plus  = sum(1 for i in free if w[i] > 0)   # |{i ∈ F : w_i = +1}|
    nF_minus = sum(1 for i in free if w[i] < 0)   # |{i ∈ F : w_i = −1}|

    # No-monoparity lemma: nF_plus ≥ 1 and nF_minus ≥ 1 must hold.
    if nF_plus == 0 or nF_minus == 0:
        raise ClippingInfeasible(
            f"No-monoparity violated: nF_plus={nF_plus}, nF_minus={nF_minus}. "
            "This indicates n·ε_tol ≥ π or an invalid SVCP."
        )

    # KKT solution (eq. 4 / Proposition 4.3):
    #   δ_i = (R₁ + R₂) / (2 n_F⁺)   for i ∈ F, w_i = +1  (odd)
    #   δ_i = (R₁ − R₂) / (2 n_F⁻)   for i ∈ F, w_i = −1  (even)
    a = (R1 + R2) / (2.0 * nF_plus)   # common value for free odd indices
    b = (R1 - R2) / (2.0 * nF_minus)  # common value for free even indices

    for i in free:
        delta[i] = a if w[i] > 0 else b

    # Lagrange multipliers (eq. 5 / Proposition 4.3):
    #   λ + μ = −(R₁ + R₂) / n_F⁺  →  from stationarity at odd free index
    #   λ − μ = −(R₁ − R₂) / n_F⁻  →  from stationarity at even free index
    lam_plus_mu  = -(R1 + R2) / nF_plus
    lam_minus_mu = -(R1 - R2) / nF_minus
    lambda_ = 0.5 * (lam_plus_mu + lam_minus_mu)
    mu      = 0.5 * (lam_plus_mu - lam_minus_mu)

    return delta, lambda_, mu


# ---------------------------------------------------------------------------
# ClippedProject (Algorithm 4.1)
# ---------------------------------------------------------------------------

def clipped_project(
    alpha: np.ndarray,
    etol: float = 1e-12,
    max_iter: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute the constrained L² minimiser via the two-phase active-set algorithm.

    Solves:
        min  ‖δ‖₂
        s.t. 1ᵀδ = 0
             wᵀδ = −2κ
             α_i + δ_i ≥ ε_tol  for all i

    The true geometric edit distance d_G(C, F) is recovered as ε_tol → 0⁺
    (Lemma 3.8 / Definition 3.7).

    Algorithm (Algorithm 4.1):
      0. Compute κ; if κ = 0 return 0.  If κ < 0, apply cyclic shift.
      1. Initialise δ = −(κ/n) w  (unconstrained minimiser).
      2. Phase 1 – primal feasibility: while any α_i + δ_i < ε_tol,
           clip the most-violated index j = argmin(α_i + δ_i),
           add j to A, re-run FreeSolve.
         Terminates in ≤ n−1 iterations (Lemma 4.4 / Theorem 4.6).
      3. Phase 2 – dual feasibility: while any active i has ν_i < 0,
           release all such indices from A, re-run FreeSolve.
         Terminates in ≤ n−1 iterations (Theorem 4.6).
      4. Undo cyclic shift if applied; verify KKT conditions; return δ.

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n (n ≥ 1).
    etol : float
        Lower bound ε_tol > 0 with n·ε_tol < π; default 1e-12.
    max_iter : int or None
        Safety cap on total iterations (default: 4n).
    verbose : bool
        If True, emit DEBUG log messages.

    Returns
    -------
    np.ndarray
        Perturbation vector δ of length 2n satisfying all KKT conditions.

    Raises
    ------
    ClippingInfeasible
        If the algorithm cannot find a feasible solution (should not occur
        for valid SVCPs with n·ε_tol < π).
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    alpha = np.asarray(alpha, dtype=np.float64)
    N = len(alpha)
    n = N // 2
    if max_iter is None:
        max_iter = 4 * n  # safe upper bound: Phase 1 ≤ n−1, Phase 2 ≤ n−1

    kappa = compute_kappa(alpha)

    if abs(kappa) < etol:
        return np.zeros(N, dtype=np.float64)

    # --- Step 0: WLOG κ > 0 via cyclic shift (Remark 2.4) ---
    # Cyclic shift: α'_i = α_{i+1 mod 2n}  ↔  numpy roll by -1
    # This maps w_i → −w_i, κ → −κ.  We record and undo at the end.
    shifted = False
    if kappa < 0:
        alpha = np.roll(alpha, -1)
        kappa = -kappa
        shifted = True
        logger.debug("κ < 0: applied cyclic shift; new κ = %.6f rad", kappa)

    w = weight_vector(n)

    # --- Step 1: unconstrained minimiser ---
    delta = -(kappa / n) * w

    # Fast exit if already feasible
    if np.all(alpha + delta >= etol):
        logger.debug("Unconstrained minimiser is feasible; no clipping needed.")
        if shifted:
            delta = np.roll(delta, 1)
        return delta

    # --- Phase 1: primal feasibility ---
    # Add indices one at a time (argmin violated value), re-solve FreeSolve.
    active: set[int] = set()

    for p1_iter in range(max_iter):
        # Find most-violated index not yet in active set
        corrected = alpha + delta
        candidates = [
            (corrected[i], i)
            for i in range(N)
            if i not in active and corrected[i] < etol
        ]
        if not candidates:
            break  # Phase 1 complete

        _, j = min(candidates)  # argmin α_i + δ_i (most violated)
        active.add(j)
        logger.debug("Phase 1 iter %d: clipping index %d (α+δ=%.4e).",
                     p1_iter, j, alpha[j] + delta[j])

        delta, lambda_, mu = _free_solve(alpha, w, active, kappa, etol)
    else:
        raise ClippingInfeasible(
            f"Phase 1 did not converge in {max_iter} iterations."
        )

    # --- Phase 2: dual feasibility ---
    # Compute ν_i = 2δ_i + λ + μ w_i for each active index.
    # Release any with ν_i < 0 and re-solve until all ν_i ≥ 0.
    for p2_iter in range(max_iter):
        if not active:
            break

        delta, lambda_, mu = _free_solve(alpha, w, active, kappa, etol)

        # KKT multipliers for bound constraints (stationarity residual)
        neg_dual = {
            i for i in active
            if 2.0 * delta[i] + lambda_ + mu * w[i] < -1e-14
        }
        if not neg_dual:
            break  # Phase 2 complete: all ν_i ≥ 0

        active -= neg_dual
        logger.debug("Phase 2 iter %d: releasing %d indices %s.",
                     p2_iter, len(neg_dual), sorted(neg_dual))
    else:
        raise ClippingInfeasible(
            f"Phase 2 did not converge in {max_iter} iterations."
        )

    # Final solve with converged active set
    delta, lambda_, mu = _free_solve(alpha, w, active, kappa, etol)

    # --- PostCheck (Algorithm 4.1 postcondition) ---
    residual_sum = abs(float(np.sum(delta)))
    residual_kaw = abs(0.5 * float(w @ delta) + kappa)
    positivity_ok = bool(np.all(alpha + delta >= etol - 1e-14))

    logger.debug(
        "PostCheck: |Σδ|=%.2e, |κ residual|=%.2e, positivity=%s, |A|=%d",
        residual_sum, residual_kaw, positivity_ok, len(active),
    )

    if not positivity_ok:
        raise ClippingInfeasible("PostCheck: positivity constraint violated.")
    if residual_sum > 1e-9 or residual_kaw > 1e-9:
        raise ClippingInfeasible(
            f"PostCheck: linear constraints not satisfied "
            f"(|Σδ|={residual_sum:.2e}, |κ res|={residual_kaw:.2e})."
        )

    # --- Undo cyclic shift (Remark 2.4) ---
    # Inverse of roll(-1) is roll(+1).
    if shifted:
        delta = np.roll(delta, 1)

    return delta