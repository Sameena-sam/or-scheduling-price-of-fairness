"""Pareto Local Search (PLS).

A simple multi-objective local search that explores neighbours of a single
solution and keeps any non-dominated improvements. Used both as a stand-alone
baseline and as a local-search component in memetic configurations.

Students may design alternative local search operators (problem-tailored
neighbourhoods, adaptive step size, restart strategies) and pass them in.
"""
from __future__ import annotations

import numpy as np
from typing import Protocol

from ..problem import Instance, OTSchedulingProblem
from ..problem.evaluator import evaluate as _evaluate
from ..problem.decoder import GreedyDecoder


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Strict Pareto dominance: a dominates b iff a <= b component-wise
    AND a < b in at least one component."""
    return bool(np.all(a <= b) and np.any(a < b))


class BaseLocalSearch(Protocol):
    def __call__(self, problem: OTSchedulingProblem, perm: np.ndarray, budget: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Returns (improved_perm, improved_F) after up to `budget` evaluations."""
        ...


class SwapNeighbourhoodLocalSearch:
    """Explores swap-neighbours of the current solution.

    Stops on first improvement found in each pass (first-improvement strategy).
    Repeats passes until no improvement is found or budget is exhausted.
    """

    def __call__(self, problem, perm, budget, rng):
        decoder = GreedyDecoder()
        inst = problem.instance
        current = perm.copy()
        sched = decoder(inst, current)
        current_f = np.array(_evaluate(inst, sched))
        evals = 1
        improved = True
        while improved and evals < budget:
            improved = False
            n = len(current)
            pair_order = rng.permutation(n * (n - 1) // 2)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for k in pair_order:
                if evals >= budget:
                    break
                i, j = pairs[int(k)]
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]
                sched = decoder(inst, cand)
                cand_f = np.array(_evaluate(inst, sched))
                evals += 1
                if dominates(cand_f, current_f):
                    current = cand
                    current_f = cand_f
                    improved = True
                    break
        return current, current_f
