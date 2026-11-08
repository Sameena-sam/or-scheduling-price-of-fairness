"""Tests for variation operators.

Validates that crossover and mutation operators produce valid permutations
(every integer in 0..n_var-1 appears exactly once).
"""
import numpy as np
import pytest

from cio.operators import (
    PMXCrossover, OrderCrossover,
    SwapMutation, InsertionMutation, InversionMutation,
)


def _is_permutation(arr: np.ndarray, n: int) -> bool:
    return sorted(arr.tolist()) == list(range(n))


@pytest.fixture
def two_parents():
    rng = np.random.default_rng(0)
    return rng.permutation(20), rng.permutation(20)


@pytest.mark.parametrize("cls", [PMXCrossover, OrderCrossover])
def test_crossover_preserves_permutation(cls, two_parents):
    rng = np.random.default_rng(1)
    op = cls()
    parent_a, parent_b = two_parents
    for _ in range(50):
        c1, c2 = op(parent_a, parent_b, rng)
        assert _is_permutation(c1, 20), f"{cls.__name__} produced non-perm: {c1}"
        assert _is_permutation(c2, 20)


@pytest.mark.parametrize("cls", [SwapMutation, InsertionMutation, InversionMutation])
def test_mutation_preserves_permutation(cls):
    rng = np.random.default_rng(2)
    op = cls(p_per_individual=1.0)
    base = rng.permutation(20)
    for _ in range(50):
        m = op(base, rng)
        assert _is_permutation(m, 20)


def test_swap_actually_swaps():
    """Sanity: swap changes exactly two positions (or none)."""
    rng = np.random.default_rng(3)
    op = SwapMutation(p_per_individual=1.0)
    base = np.arange(20)
    for _ in range(20):
        m = op(base, rng)
        diff = sum(1 for i in range(20) if m[i] != base[i])
        assert diff in (0, 2)


def test_inversion_reverses_subsequence():
    """Inversion: there must be a contiguous span that is reversed."""
    rng = np.random.default_rng(4)
    op = InversionMutation(p_per_individual=1.0)
    base = np.arange(20)
    for _ in range(20):
        m = op(base, rng)
        # Find first and last index where m differs from base
        diffs = [i for i in range(20) if m[i] != base[i]]
        if not diffs:
            continue
        a, b = min(diffs), max(diffs)
        # m[a:b+1] should be the reverse of base[a:b+1]
        assert list(m[a:b + 1]) == list(base[a:b + 1][::-1])
