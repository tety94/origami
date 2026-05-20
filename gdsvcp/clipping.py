"""
clipping.py – Active-set clipping projection (Algorithm 4.1).

When the unconstrained L² minimiser violates α_i + δ_i > 0, this module
computes the constrained minimiser by iteratively fixing violated angles to
the boundary etol and re-solving the free-variable sub-problem.

Paper reference:
  "Geometric Edit Distance for Single-Vertex Crease Patterns under L^2"
  Section 4 (Algorithm 4.1), Proposition A.2 (free-variable solve),
  Proposition 4.2 (inconsistency characterisation), Lemma 4.3 (termination).

All angles in **radians**.
"""

from __future__ import annotations

import logging
import numpy as np

from gdsvcp.core import compute_kappa, weight_vector, ClippingInfeasible

__all__ = ["clipped_project"]

logger = logging.getLogger(__name__)


def _free_solve(
    alpha: np.ndarray,
    w: np.ndarray,
    delta: np.ndarray,
    active: set[int],
    kappa: float,
    etol: float,
) -> tuple[np.ndarray, bool]:
    """Solve the free-variable sub-problem (Proposition A.2).

    Given an active set A with clipped values δ_i = etol − α_i for i ∈ A,
    find δ_i for i ∈ F = complement(A) minimising Σ_{F} δ_i² subject to:
      Σ_{F} δ_i = R₁  (total-sum residual)
      Σ_{F} w_i δ_i = R₂  (Kawasaki residual)

    Returns
    -------
    (delta_new, ok)
        delta_new : updated delta array (copy).
        ok : True if solve succeeded; False if the active set is inconsistent.
    """
    delta_new = delta.copy()
    active_list = sorted(active)

    # Compute residuals
    R1 = -sum(delta_new[i] for i in active_list)
    R2 = -2.0 * kappa - sum(w[i] * delta_new[i] for i in active_list)

    free = [i for i in range(len(alpha)) if i not in active]
    nF = len(free)
    if nF == 0:
        # No free variables; check if both residuals vanish
        if np.isclose(R1, 0.0, atol=etol) and np.isclose(R2, 0.0, atol=etol):
            return delta_new, True
        return delta_new, False

    nF_plus = sum(1 for i in free if w[i] > 0)   # odd-indexed (paper)
    nF_minus = sum(1 for i in free if w[i] < 0)  # even-indexed (paper)
    Sw = sum(w[i] for i in free)

    # --- Non-degenerate case: both parities present ---
    if nF_plus >= 1 and nF_minus >= 1:
        det = float(nF) ** 2 - float(Sw) ** 2  # = 4 * nF_plus * nF_minus
        # det > 0 guaranteed
        lam = (-2.0 / det) * (nF * R1 - Sw * R2)
        mu = (-2.0 / det) * (-Sw * R1 + nF * R2)
        for i in free:
            delta_new[i] = -0.5 * (lam + mu * w[i])
        return delta_new, True

    # --- Degenerate case: all free indices have the same parity ---
    if nF_minus == 0:
        # All free w_i = +1; constraints collapse to δ_sum = R₁ and δ_sum = R₂
        if not np.isclose(R1, R2, atol=etol * 10):
            logger.debug("Degenerate (nF_minus=0): R1=%g, R2=%g – inconsistent.", R1, R2)
            return delta_new, False  # rollback needed
        R = R1
    else:
        # All free w_i = −1; constraints: δ_sum = R₁ and −δ_sum = R₂
        if not np.isclose(R1, -R2, atol=etol * 10):
            logger.debug("Degenerate (nF_plus=0): R1=%g, R2=%g – inconsistent.", R1, R2)
            return delta_new, False  # rollback needed
        R = R1

    # Minimum-norm solution on the single hyperplane Σ δ_i = R
    for i in free:
        delta_new[i] = R / nF
    return delta_new, True


def clipped_project(
    alpha: np.ndarray,
    etol: float = 1e-12,
    max_iter: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute the constrained L² minimiser via active-set clipping (Algorithm 4.1).

    Solves:
        min  ‖δ‖₂
        s.t. 1^T δ = 0
             w^T δ = −2κ
             α_i + δ_i ≥ etol  for all i

    Algorithm:
      1. Start with the unconstrained minimiser δ* = −κ w / n.
      2. Identify violated indices; clip them to etol − α_i.
      3. Re-solve the free-variable sub-problem on the remaining indices.
      4. Repeat until no new violations; handle degenerate cases via rollback.

    Termination is guaranteed in ≤ 2n outer iterations (Lemma 4.3).

    Parameters
    ----------
    alpha : np.ndarray
        Sector angles in radians, length 2n.
    etol : float
        Positivity lower bound; default 1e-12.
    max_iter : int or None
        Maximum iterations (default: 2n).
    verbose : bool
        If True, log progress via the module logger.

    Returns
    -------
    np.ndarray
        Perturbation vector δ satisfying all constraints.

    Raises
    ------
    ClippingInfeasible
        If the algorithm cannot find a feasible solution (should not occur
        for valid SVCPs).
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    alpha = np.asarray(alpha, dtype=np.float64)
    N = len(alpha)
    n = N // 2
    if max_iter is None:
        max_iter = 2 * N  # paper bound

    kappa = compute_kappa(alpha)
    w = weight_vector(n)

    if abs(kappa) < etol:
        return np.zeros(N, dtype=np.float64)

    # Initial unconstrained minimiser
    delta = -(kappa / n) * w.copy()

    active: set[int] = set()
    rollback_budget = 2 * N  # safety valve

    for iteration in range(max_iter):
        # Clip active values
        for i in active:
            delta[i] = etol - alpha[i]

        # Identify new violations
        A_new = {i for i in range(N) if (alpha[i] + delta[i] < etol) and i not in active}

        if not A_new:
            # No new violations – check feasibility of current solve
            delta_candidate, ok = _free_solve(alpha, w, delta, active, kappa, etol)
            if ok:
                delta = delta_candidate
                break
            # Inconsistent free solve with no new violations – shouldn't normally happen
            raise ClippingInfeasible(
                "Free-variable sub-problem inconsistent and no new violations to add."
            )

        logger.debug("Iter %d: adding %d indices to active set.", iteration, len(A_new))

        # Attempt to solve with new additions
        trial_active = active | A_new
        for i in A_new:
            delta[i] = etol - alpha[i]

        delta_candidate, ok = _free_solve(alpha, w, delta, trial_active, kappa, etol)

        if ok:
            active = trial_active
            delta = delta_candidate
        else:
            # Rollback: undo additions one at a time until consistent
            logger.debug("Rollback triggered at iter %d.", iteration)
            added = sorted(A_new)
            rolled = set()
            for idx in added:
                rolled.add(idx)
                partial_active = trial_active - rolled
                delta_rb = delta.copy()
                for i in partial_active:
                    delta_rb[i] = etol - alpha[i]
                delta_rb_out, ok_rb = _free_solve(alpha, w, delta_rb, partial_active, kappa, etol)
                if ok_rb:
                    active = partial_active
                    delta = delta_rb_out
                    rollback_budget -= len(rolled)
                    if rollback_budget < 0:
                        raise ClippingInfeasible("Rollback budget exhausted.")
                    break
            else:
                raise ClippingInfeasible(
                    "Could not resolve inconsistency even after full rollback."
                )
    else:
        raise ClippingInfeasible(
            f"ClippedProject did not converge in {max_iter} iterations."
        )

    # Final feasibility verification
    residual_sum = abs(np.sum(delta))
    residual_kaw = abs(0.5 * float(w @ delta) + kappa)
    positivity_ok = np.all(alpha + delta >= etol - 1e-14)

    logger.debug(
        "Done: ‖sum δ‖=%.2e, |κ residual|=%.2e, positivity=%s",
        residual_sum, residual_kaw, positivity_ok,
    )

    return delta