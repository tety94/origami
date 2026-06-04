"""
gdsvcp.clipping
===============
Two-phase active-set clipping projection (Algorithms 4.1 and 4.2).

When the unconstrained L² minimiser violates the positivity constraint
α_i + δ_i ≥ ε_tol, this module computes the constrained minimiser over

    K_{ε_tol} = { δ ∈ H_κ : α_i + δ_i ≥ ε_tol  for all i }

via the certified active-set method of the paper.  The true geometric edit
distance d_G(C, F) is recovered as ε_tol → 0⁺ (Lemma 3.8).

Algorithm summary
-----------------
Phase 0  Compute κ; if κ = 0 return 0.  If κ < 0 apply cyclic shift.
Phase 1  Primal feasibility: clip the most-violated index, re-run
         FreeSolve.  Terminates in ≤ n − 1 iterations (Theorem 4.6).
Phase 2  Dual feasibility: release active indices with ν_i < 0, re-run
         FreeSolve.  Terminates in ≤ n − 1 iterations (Theorem 4.6).
PostChk  Verify KKT conditions; undo cyclic shift; return δ.

Paper reference
---------------
Callegaro, S. "How Far Is a Single-Vertex Crease Pattern from Flat
Foldability? Minimal Angular Corrections under Three Norms." (2025).
  - Algorithm 4.1   : ClippedProject
  - Algorithm 4.2   : FreeSolve
  - Proposition 4.3 : free-variable KKT solution (eq. 4–5)
  - Lemma 4.4       : joint induction (no-monoparity, only-odd clipping,
                       termination in ≤ n−1 steps)
  - Lemma 4.5       : no-monoparity at the global minimiser
  - Theorem 4.6     : exact-arithmetic termination
  - Proposition 4.7 : KKT optimality of the output

All angles in **radians**.
"""

from __future__ import annotations

import logging

import numpy as np

from gdsvcp.core import compute_kappa, weight_vector, ClippingInfeasible

__all__ = ["clipped_project", "free_solve"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FreeSolve  (Algorithm 4.2)
# ---------------------------------------------------------------------------

def free_solve(
    alpha: np.ndarray,
    w: np.ndarray,
    active: set[int],
    kappa: float,
    etol: float,
) -> tuple[np.ndarray, float, float]:
    """Solve the free-variable KKT sub-problem (Algorithm 4.2).

    Given the active set A with clipped values δ_i = ε_tol − α_i for i ∈ A,
    find δ_i for i ∈ F = {0,…,2n−1} \\ A that minimise

        Σ_{F} δ_i²

    subject to the two residual constraints (Proposition 4.3):

        Σ_{F} δ_i       = R₁  (total-angle residual)
        Σ_{F} w_i δ_i   = R₂  (Kawasaki residual)

    where R₁ = Σ_{A} (α_i − ε_tol) > 0 and
          R₂ = −2κ − Σ_{A} w_i (ε_tol − α_i).

    The KKT solution is (Proposition 4.3, eq. 4):
        δ_i = (R₁ + R₂) / (2 n_F⁺)   for i ∈ F, w_i = +1  (paper-odd)
        δ_i = (R₁ − R₂) / (2 n_F⁻)   for i ∈ F, w_i = −1  (paper-even)

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.
    w : np.ndarray
        Weight vector of length 2n (from ``weight_vector``).
    active : set[int]
        0-based indices in the active set A.
    kappa : float
        Kawasaki deficit κ > 0 (cyclic shift has already been applied if κ < 0).
    etol : float
        Lower bound ε_tol ≥ 0.

    Returns
    -------
    (delta, lambda_, mu) : tuple[np.ndarray, float, float]
        delta   : perturbation vector of length 2n; active entries set to
                  ε_tol − α_i, free entries set to the KKT values.
        lambda_ : Lagrange multiplier for the sum constraint 1ᵀδ = 0.
        mu      : Lagrange multiplier for the Kawasaki constraint wᵀδ = −2κ.

    Raises
    ------
    ClippingInfeasible
        If n_F⁺ = 0 or n_F⁻ = 0 (no-monoparity violated).  Under the
        hypotheses of Lemma 4.4 this cannot happen.
    """
    N = len(alpha)
    delta = np.empty(N, dtype=np.float64)

    # Clipped values
    for i in active:
        delta[i] = etol - alpha[i]

    free = [i for i in range(N) if i not in active]

    # Residuals (Proposition 4.3)
    R1 = sum(alpha[i] - etol for i in active)
    R2 = -2.0 * kappa - sum(w[i] * (etol - alpha[i]) for i in active)

    nF_plus  = sum(1 for i in free if w[i] > 0)
    nF_minus = sum(1 for i in free if w[i] < 0)

    if nF_plus == 0 or nF_minus == 0:
        raise ClippingInfeasible(
            f"No-monoparity violated: n_F⁺={nF_plus}, n_F⁻={nF_minus}. "
            "This indicates n·ε_tol ≥ π or an invalid SVCP (Lemma 4.5)."
        )

    # KKT values (Proposition 4.3, eq. 4)
    a = (R1 + R2) / (2.0 * nF_plus)    # common value for free paper-odd indices
    b = (R1 - R2) / (2.0 * nF_minus)   # common value for free paper-even indices

    for i in free:
        delta[i] = a if w[i] > 0 else b

    # Lagrange multipliers (Proposition 4.3, eq. 5)
    # λ + μ = −(R₁ + R₂)/n_F⁺  and  λ − μ = −(R₁ − R₂)/n_F⁻
    lam_p = -(R1 + R2) / nF_plus
    lam_m = -(R1 - R2) / nF_minus
    lambda_ = 0.5 * (lam_p + lam_m)
    mu      = 0.5 * (lam_p - lam_m)

    return delta, lambda_, mu


# ---------------------------------------------------------------------------
# ClippedProject  (Algorithm 4.1)
# ---------------------------------------------------------------------------

def clipped_project(
    alpha: np.ndarray,
    etol: float = 1e-12,
    max_iter: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute the constrained L² minimiser via the two-phase active-set method.

    Solves the QP (Definition 3.7):

        min  ½ ‖δ‖₂²
        s.t. 1ᵀδ = 0
             wᵀδ = −2κ
             α_i + δ_i ≥ ε_tol  for all i

    The true geometric edit distance d_G(C, F) is recovered in the limit
    ε_tol → 0⁺ (Lemma 3.8).

    When the unconstrained minimiser δ* = −(κ/n) w is already feasible,
    this function returns δ* directly (no iteration needed).

    Structural guarantees (Lemma 4.4, Theorem 4.6)
    -----------------------------------------------
    - Phase 1 clips *only* paper-odd-indexed (0-based even) angles when κ > 0.
    - The last free paper-odd index is *never* clipped (Lemma 4.4(d)).
    - Phase 1 terminates in ≤ n − 1 iterations.
    - Phase 2 terminates in ≤ n − 1 iterations.
    - FreeSolve is non-degenerate throughout (det M = 4 n_F⁺ n_F⁻ > 0).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in **radians**, length 2n (n ≥ 1).
    etol : float
        Lower bound ε_tol > 0 satisfying n·ε_tol < π; default 1e-12.
        Set to a small positive value (e.g. 1e-9) for numerical stability.
    max_iter : int or None
        Safety cap on total loop iterations; default 4n.
    verbose : bool
        If True, emit DEBUG-level log messages at each iteration.

    Returns
    -------
    np.ndarray
        Perturbation vector δ of length 2n satisfying all KKT conditions
        (Proposition 4.7).

    Raises
    ------
    ClippingInfeasible
        If convergence fails (should not occur for valid inputs with n·ε_tol < π).
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    alpha = np.asarray(alpha, dtype=np.float64)
    N = len(alpha)
    n = N // 2
    if max_iter is None:
        max_iter = 4 * n   # Phase 1 ≤ n−1, Phase 2 ≤ n−1; headroom for safety

    kappa = compute_kappa(alpha)

    # κ = 0 → already flat-foldable
    if abs(kappa) < etol:
        return np.zeros(N, dtype=np.float64)

    # Phase 0: WLOG κ > 0 via cyclic shift (Remark 2.4)
    # np.roll(alpha, -1) implements α_i ← α_{i+1 mod 2n}
    shifted = False
    if kappa < 0:
        alpha  = np.roll(alpha, -1)
        kappa  = -kappa
        shifted = True
        logger.debug("κ < 0 → cyclic shift applied; working with κ = %.8f rad.", kappa)

    w = weight_vector(n)

    # Phase 0: unconstrained minimiser (Theorem 3.1(i))
    delta = -(kappa / n) * w

    # Fast path: no clipping needed
    if np.all(alpha + delta >= etol):
        logger.debug("Unconstrained minimiser is feasible; skipping clipping.")
        if shifted:
            delta = np.roll(delta, 1)
        return delta

    # ------------------------------------------------------------------
    # Phase 1 — primal feasibility
    # Iteratively clip the most-violated index (argmin corrected value),
    # add to active set A, re-run FreeSolve.
    # By Lemma 4.4(b)(c): only paper-odd (0-based even) indices are ever
    # clipped; terminates in ≤ n − 1 steps (Lemma 4.4(d)).
    # ------------------------------------------------------------------
    active: set[int] = set()
    lambda_: float = 0.0
    mu: float = 0.0

    for p1_iter in range(max_iter):
        corrected = alpha + delta
        # Find all indices outside A that violate the lower bound
        violated = [
            (corrected[i], i)
            for i in range(N)
            if i not in active and corrected[i] < etol
        ]
        if not violated:
            break
        _, j = min(violated)   # most-violated index
        active.add(j)
        logger.debug(
            "Phase 1 iter %d: clipping index %d (α+δ = %.4e rad, paper-%s).",
            p1_iter, j, corrected[j], "odd" if j % 2 == 0 else "even",
        )
        delta, lambda_, mu = free_solve(alpha, w, active, kappa, etol)
    else:
        raise ClippingInfeasible(
            f"Phase 1 did not converge in {max_iter} iterations."
        )

    # ------------------------------------------------------------------
    # Phase 2 — dual feasibility
    # Check KKT multipliers ν_i = 2δ_i + λ + μ w_i for active indices.
    # Release any with ν_i < 0 (they should not be active at the optimum).
    # Terminates in ≤ n − 1 steps (Theorem 4.6).
    # ------------------------------------------------------------------
    for p2_iter in range(max_iter):
        if not active:
            break
        delta, lambda_, mu = free_solve(alpha, w, active, kappa, etol)
        neg_dual = {
            i for i in active
            if 2.0 * delta[i] + lambda_ + mu * w[i] < -1e-14
        }
        if not neg_dual:
            break
        active -= neg_dual
        logger.debug(
            "Phase 2 iter %d: releasing %d index/indices %s.",
            p2_iter, len(neg_dual), sorted(neg_dual),
        )
    else:
        raise ClippingInfeasible(
            f"Phase 2 did not converge in {max_iter} iterations."
        )

    # Final solve with the converged active set
    delta, lambda_, mu = free_solve(alpha, w, active, kappa, etol)

    # ------------------------------------------------------------------
    # PostCheck (Algorithm 4.1): verify KKT conditions
    # ------------------------------------------------------------------
    res_sum = abs(float(delta.sum()))
    res_kaw = abs(0.5 * float(w @ delta) + kappa)
    pos_ok  = bool(np.all(alpha + delta >= etol - 1e-14))

    logger.debug(
        "PostCheck: |Σδ| = %.2e rad, |κ-residual| = %.2e rad, "
        "positivity = %s, |A| = %d.",
        res_sum, res_kaw, pos_ok, len(active),
    )

    if not pos_ok:
        raise ClippingInfeasible("PostCheck failed: positivity constraint violated.")
    if res_sum > 1e-9 or res_kaw > 1e-9:
        raise ClippingInfeasible(
            f"PostCheck failed: linear constraints not satisfied "
            f"(|Σδ| = {res_sum:.2e} rad, |κ-residual| = {res_kaw:.2e} rad)."
        )

    # Undo cyclic shift (inverse of roll(−1) is roll(+1))
    if shifted:
        delta = np.roll(delta, 1)

    return delta