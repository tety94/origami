"""
tests/test_core.py – Unit tests for core, L², clipping, and norms.

Test cases driven by tests/fixtures/*_expected.json so results are always
consistent with the pre-computed reference values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gdsvcp.core import compute_kappa, validate_svcp, weight_vector, PositivityViolation
from gdsvcp.l2 import l2_minimiser, dG_l2
from gdsvcp.clipping import clipped_project
from gdsvcp.norms import l1_minimisers, linfty_minimiser

FIXTURES = Path(__file__).parent / "fixtures"
TOL = 1e-8


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}_expected.json").read_text())


# ── weight_vector ─────────────────────────────────────────────────────────

def test_weight_vector_n2():
    w = weight_vector(2)
    np.testing.assert_array_equal(w, [1., -1., 1., -1.])


def test_weight_vector_sum_zero():
    for n in [1, 2, 3, 6, 10]:
        assert np.sum(weight_vector(n)) == 0.0


def test_weight_vector_invalid():
    with pytest.raises(ValueError):
        weight_vector(0)


# ── compute_kappa ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", [
    "n2_paper", "n3_paper", "n4_symmetric", "n3_needs_clipping",
    "n2_small_kappa", "n3_flat_foldable",
])
def test_compute_kappa_fixture(fixture):
    ref   = _load(fixture)
    alpha = np.radians(ref["angles_deg"])
    kappa = compute_kappa(alpha)
    assert abs(np.degrees(kappa) - ref["kappa_deg"]) < TOL, \
        f"{fixture}: κ={np.degrees(kappa):.6f}° expected {ref['kappa_deg']:.6f}°"


def test_compute_kappa_uniform():
    # uniform n=3: all angles = 60°, κ = 0
    alpha = np.radians([60.] * 6)
    assert abs(compute_kappa(alpha)) < 1e-12


def test_compute_kappa_odd_length_raises():
    with pytest.raises(ValueError):
        compute_kappa(np.array([1., 2., 3.]))


# ── validate_svcp ─────────────────────────────────────────────────────────

def test_validate_ok():
    validate_svcp(np.radians([100., 80., 100., 80.]))  # should not raise


def test_validate_zero_angle_raises():
    with pytest.raises(PositivityViolation):
        validate_svcp(np.radians([0., 90., 180., 90.]))


def test_validate_wrong_sum_raises():
    with pytest.raises(ValueError):
        validate_svcp(np.radians([100., 80., 100., 70.]))  # sum ≠ 360°


# ── l2_minimiser ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", ["n2_paper", "n3_paper", "n4_symmetric"])
def test_l2_minimiser_feasible(fixture):
    ref   = _load(fixture)
    if ref["clipping_needed"]:
        pytest.skip("clipping case")
    alpha = np.radians(ref["angles_deg"])
    delta = l2_minimiser(alpha)
    assert np.allclose(np.degrees(delta), ref["delta_deg"], atol=TOL)


def test_l2_minimiser_positivity_violation():
    alpha = np.radians([150., 10., 100., 60., 30., 10.])
    with pytest.raises(PositivityViolation):
        l2_minimiser(alpha)


def test_dG_l2_formula():
    ref   = _load("n2_paper")
    alpha = np.radians(ref["angles_deg"])
    n     = ref["n"]
    kappa = np.radians(ref["kappa_deg"])
    expected = abs(kappa) * np.sqrt(2.0 / n)
    assert abs(dG_l2(alpha) - expected) < 1e-12


# ── clipped_project ───────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", ["n3_needs_clipping"])
def test_clipping_feasible(fixture):
    ref   = _load(fixture)
    alpha = np.radians(ref["angles_deg"])
    delta = clipped_project(alpha)
    # positivity
    assert np.all(alpha + delta >= -1e-10)
    # Kawasaki restored
    assert abs(compute_kappa(alpha + delta)) < 1e-10
    # sum preserved
    assert abs(np.sum(delta)) < 1e-10


def test_clipping_constraint_satisfaction():
    """KKT-style: check constraints hold after clipping."""
    alpha = np.radians([150., 10., 100., 60., 30., 10.])
    delta = clipped_project(alpha)
    # 1. Kawasaki
    assert abs(compute_kappa(alpha + delta)) < 1e-10
    # 2. Total-angle conservation
    assert abs(np.sum(delta)) < 1e-10
    # 3. Positivity
    assert np.all(alpha + delta > -1e-10)


def test_clipping_on_feasible_input_matches_l2():
    """When l2 is already feasible, clipped_project returns same delta."""
    alpha = np.radians([100., 80., 100., 80.])
    d_l2  = l2_minimiser(alpha)
    d_cp  = clipped_project(alpha)
    np.testing.assert_allclose(d_cp, d_l2, atol=1e-10)


def test_rollback_degenerate_free_set():
    """
    Construct a case where all but one free index share the same parity
    to trigger the degenerate free-variable path (Proposition A.2).
    The algorithm must still return a feasible solution.
    """
    # Very small odd angles → many get clipped, leaving mostly even free
    etol = 1e-12
    alpha = np.radians([2., 88., 2., 88., 2., 88., 2., 88.])  # n=4
    delta = clipped_project(alpha, etol=etol)
    assert np.all(alpha + delta >= etol - 1e-11)
    assert abs(compute_kappa(alpha + delta)) < 1e-9
    assert abs(np.sum(delta)) < 1e-9


# ── norms ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", [
    "n2_paper", "n3_paper", "n4_symmetric",
])
def test_l1_minimisers_value(fixture):
    ref   = _load(fixture)
    alpha = np.radians(ref["angles_deg"])
    kappa = compute_kappa(alpha)
    mins  = l1_minimisers(alpha)
    n     = ref["n"]
    assert len(mins) == n * n, f"expected {n**2} extreme points"
    for delta in mins:
        assert abs(np.linalg.norm(delta, 1) - 2 * abs(kappa)) < 1e-12
        assert abs(np.sum(delta)) < 1e-12


@pytest.mark.parametrize("fixture", ["n2_paper", "n3_paper"])
def test_linfty_minimiser(fixture):
    ref   = _load(fixture)
    alpha = np.radians(ref["angles_deg"])
    n     = ref["n"]
    kappa = compute_kappa(alpha)
    delta, val = linfty_minimiser(alpha)
    assert abs(val - abs(kappa) / n) < 1e-12
    assert abs(np.linalg.norm(delta, np.inf) - val) < 1e-12
    # uniqueness: must equal -kappa/n * w
    from gdsvcp.core import weight_vector
    w = weight_vector(n)
    np.testing.assert_allclose(delta, -(kappa / n) * w, atol=1e-12)


# ── plotting (smoke tests – no visual assertions) ─────────────────────────

def test_plot_angle_shift_runs(tmp_path):
    from gdsvcp.plotting import plot_angle_shift
    alpha = np.radians([100., 80., 100., 80.])
    delta = l2_minimiser(alpha)
    p = plot_angle_shift(alpha, alpha + delta, "test",
                         outpath=str(tmp_path / "bar.png"))
    assert Path(p).exists()


def test_plot_crease_pattern_runs(tmp_path):
    from gdsvcp.plotting import plot_crease_pattern
    alpha = np.radians([100., 80., 100., 80.])
    delta = l2_minimiser(alpha)
    p = plot_crease_pattern(alpha, alpha + delta, title="test",
                            outpath=str(tmp_path / "circle.png"))
    assert Path(p).exists()


def test_plot_norm_comparison_runs(tmp_path):
    from gdsvcp.plotting import plot_norm_comparison
    alpha = np.radians([100., 80., 100., 80.])
    p = plot_norm_comparison(alpha, outpath=str(tmp_path / "norms.png"))
    assert Path(p).exists()