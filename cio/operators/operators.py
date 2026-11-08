"""Operators and named hooks.

This module exposes baseline implementations of the variation, local-search,
constraint-handling, and replacement operators that students will design and
substitute. Students should subclass or instantiate alternative versions of
these classes and pass them to the algorithm builders in `cio.algorithms`.

Hooks exposed:
    Initialiser        - generates initial permutations
    Crossover          - combines two parent permutations
    Mutation           - perturbs a single permutation
    LocalSearch        - improves a solution via neighbourhood moves
    ConstraintHandler  - currently a no-op (decoder enforces feasibility);
                         students may design alternative decoders or repair ops

Note: pymoo has its own Operator class hierarchy. The wrappers below convert
our hook classes into pymoo operators so students can pass our high-level
hooks and have them work with pymoo's NSGA-II / MOEA/D.

For students:
    To plug in your own crossover operator, write a class with a __call__
    method matching the signature in `BaseCrossover`. Then pass it via the
    `crossover=` argument to the algorithm builder.
"""
from __future__ import annotations

import numpy as np
from typing import Protocol

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


# ---------------------------------------------------------------------------
# Initialiser (a.k.a. sampling)
# ---------------------------------------------------------------------------

class BaseInitialiser(Protocol):
    def sample(self, n_var: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        ...


class RandomPermutationInitialiser:
    """Baseline initialiser: uniform random permutations."""

    def sample(self, n_var: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((n_samples, n_var), dtype=int)
        for i in range(n_samples):
            out[i] = rng.permutation(n_var)
        return out


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class BaseCrossover(Protocol):
    def __call__(self, parent_a: np.ndarray, parent_b: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        ...


class PMXCrossover:
    """Partially-mapped crossover (PMX). Standard permutation crossover."""

    def __call__(self, parent_a, parent_b, rng):
        n = len(parent_a)
        if n < 2:
            return parent_a.copy(), parent_b.copy()
        cx_pts = sorted(rng.choice(n, size=2, replace=False))
        a, b = cx_pts
        child1 = self._pmx_one(parent_a, parent_b, a, b)
        child2 = self._pmx_one(parent_b, parent_a, a, b)
        return child1, child2

    @staticmethod
    def _pmx_one(p1, p2, a, b):
        n = len(p1)
        child = np.full(n, -1, dtype=int)
        child[a:b + 1] = p1[a:b + 1]
        mapping = {int(p1[i]): int(p2[i]) for i in range(a, b + 1)}
        for i in list(range(0, a)) + list(range(b + 1, n)):
            val = int(p2[i])
            while val in child[a:b + 1]:
                val = mapping.get(val, val)
                # Safety: prevent infinite loop
                if val == -1:
                    break
            child[i] = val
        # Fill any remaining -1 (shouldn't happen for valid PMX)
        missing = [v for v in range(n) if v not in child]
        for i in range(n):
            if child[i] == -1 and missing:
                child[i] = missing.pop(0)
        return child


class OrderCrossover:
    """Order crossover (OX). Often competitive with PMX on scheduling."""

    def __call__(self, parent_a, parent_b, rng):
        n = len(parent_a)
        if n < 2:
            return parent_a.copy(), parent_b.copy()
        cx_pts = sorted(rng.choice(n, size=2, replace=False))
        a, b = cx_pts
        child1 = self._ox_one(parent_a, parent_b, a, b)
        child2 = self._ox_one(parent_b, parent_a, a, b)
        return child1, child2

    @staticmethod
    def _ox_one(p1, p2, a, b):
        n = len(p1)
        child = np.full(n, -1, dtype=int)
        child[a:b + 1] = p1[a:b + 1]
        used = set(int(v) for v in p1[a:b + 1])
        remaining = [int(v) for v in p2 if int(v) not in used]
        idx = 0
        for i in list(range(b + 1, n)) + list(range(0, a)):
            child[i] = remaining[idx]
            idx += 1
        return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

class BaseMutation(Protocol):
    def __call__(self, perm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ...


class SwapMutation:
    """Swap two randomly chosen positions."""

    def __init__(self, p_per_individual: float = 1.0):
        self.p = p_per_individual

    def __call__(self, perm, rng):
        if rng.random() > self.p:
            return perm.copy()
        n = len(perm)
        if n < 2:
            return perm.copy()
        i, j = rng.choice(n, size=2, replace=False)
        out = perm.copy()
        out[i], out[j] = out[j], out[i]
        return out


class InsertionMutation:
    """Pick a random element and re-insert it at a random position."""

    def __init__(self, p_per_individual: float = 1.0):
        self.p = p_per_individual

    def __call__(self, perm, rng):
        if rng.random() > self.p:
            return perm.copy()
        n = len(perm)
        if n < 2:
            return perm.copy()
        i, j = rng.choice(n, size=2, replace=False)
        out = perm.copy().tolist()
        val = out.pop(int(i))
        out.insert(int(j), val)
        return np.asarray(out, dtype=int)


class InversionMutation:
    """Reverse a random sub-sequence."""

    def __init__(self, p_per_individual: float = 1.0):
        self.p = p_per_individual

    def __call__(self, perm, rng):
        if rng.random() > self.p:
            return perm.copy()
        n = len(perm)
        if n < 2:
            return perm.copy()
        a, b = sorted(rng.choice(n, size=2, replace=False))
        out = perm.copy()
        out[a:b + 1] = out[a:b + 1][::-1]
        return out


# ---------------------------------------------------------------------------
# pymoo wrappers
# ---------------------------------------------------------------------------

class _SamplingWrapper(Sampling):
    def __init__(self, initialiser: BaseInitialiser, seed: int | None = None):
        super().__init__()
        self.initialiser = initialiser
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, **kwargs):
        return self.initialiser.sample(problem.n_var, n_samples, self.rng)


class _CrossoverWrapper(Crossover):
    def __init__(self, crossover_op: BaseCrossover, prob: float = 0.9, seed: int | None = None):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.crossover_op = crossover_op
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        # X has shape (n_parents=2, n_matings, n_var)
        _, n_matings, n_var = X.shape
        out = np.empty((self.n_offsprings, n_matings, n_var), dtype=int)
        for k in range(n_matings):
            c1, c2 = self.crossover_op(X[0, k], X[1, k], self.rng)
            out[0, k] = c1
            out[1, k] = c2
        return out


class _MutationWrapper(Mutation):
    def __init__(self, mutation_op: BaseMutation, prob: float = 1.0, seed: int | None = None):
        super().__init__(prob=prob)
        self.mutation_op = mutation_op
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            out[i] = self.mutation_op(X[i], self.rng)
        return out


def to_pymoo_sampling(initialiser: BaseInitialiser, seed: int | None = None) -> Sampling:
    return _SamplingWrapper(initialiser, seed=seed)


def to_pymoo_crossover(crossover_op: BaseCrossover, prob: float = 0.9, seed: int | None = None) -> Crossover:
    return _CrossoverWrapper(crossover_op, prob=prob, seed=seed)


def to_pymoo_mutation(mutation_op: BaseMutation, prob: float = 1.0, seed: int | None = None) -> Mutation:
    return _MutationWrapper(mutation_op, prob=prob, seed=seed)
