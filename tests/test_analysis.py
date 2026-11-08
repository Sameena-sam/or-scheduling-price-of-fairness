"""Tests for indicators and statistical tests."""
import numpy as np
import pytest

from cio.analysis import (
    hypervolume, igd_plus, spread, non_dominated, union_non_dominated,
    friedman, vargha_delaney, effect_size_magnitude,
)


def test_non_dominated_simple():
    F = np.array([[1, 1], [2, 2], [3, 3]])
    mask = non_dominated(F)
    # [1,1] dominates the rest
    assert mask.tolist() == [True, False, False]


def test_non_dominated_pareto():
    # A 2D Pareto front: each point dominates none of the others.
    F = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
    mask = non_dominated(F)
    assert mask.tolist() == [True, True, True, True, True]


def test_union_non_dominated_combines_correctly():
    F1 = np.array([[1.0, 4.0], [3.0, 2.0]])
    F2 = np.array([[2.0, 3.0], [5.0, 5.0]])   # last point dominated
    out = union_non_dominated(F1, F2)
    assert len(out) == 3   # the dominated [5,5] should be removed


def test_hypervolume_basic():
    F = np.array([[1.0, 4.0], [3.0, 2.0]])
    hv = hypervolume(F, ref_point=[5.0, 5.0])
    assert hv > 0


def test_igd_plus_zero_when_same():
    """IGD+ of a set against itself should be 0."""
    ref = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]])
    assert igd_plus(ref, ref) == pytest.approx(0.0, abs=1e-9)


def test_spread_positive_for_diverse_front():
    F = np.array([[1.0, 4.0], [3.0, 2.0]])
    assert spread(F) > 0


def test_friedman_smoke():
    """Friedman should detect a difference between configs that obviously
    differ."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, 30)
    b = rng.normal(0.0, 0.1, 30)
    c = rng.normal(5.0, 0.1, 30)
    data = np.column_stack([a, b, c])
    result = friedman(data)
    assert result["p_value"] < 0.05


def test_vargha_delaney_when_a_dominates():
    a = np.array([10.0] * 30)
    b = np.array([1.0] * 30)
    A = vargha_delaney(a, b)
    assert A == 1.0
    assert effect_size_magnitude(A) == "large"


def test_vargha_delaney_when_equal():
    a = np.array([5.0] * 30)
    b = np.array([5.0] * 30)
    A = vargha_delaney(a, b)
    # All ties -> 0.5 by VD convention
    assert A == pytest.approx(0.5, abs=1e-9)
    assert effect_size_magnitude(A) == "negligible"
