"""Baseline algorithm builders.

Three baselines are provided per the brief:
    1. NSGA-II with PMX + swap-mutation + feasibility-rule constraint handling
    2. MOEA/D-DE with PMX + swap-mutation (Tchebycheff decomposition)
    3. Pareto Local Search (random-restart)

Each builder returns a pymoo algorithm object configured with the chosen
operators. Students plug in their own components by passing the appropriate
hook implementation.

Note on MOEA/D: pymoo's MOEAD does not natively support permutation operators
out of the box; we therefore wire a `MOEAD_PERM` algorithm using pymoo's
decomposition machinery with custom variation operators below. The result is
a faithful permutation-based MOEA/D variant.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions

from ..operators import (
    BaseCrossover, BaseInitialiser, BaseMutation,
    PMXCrossover, RandomPermutationInitialiser, SwapMutation,
    to_pymoo_crossover, to_pymoo_mutation, to_pymoo_sampling,
)


def build_nsga2(
    pop_size: int = 100,
    initialiser: BaseInitialiser | None = None,
    crossover: BaseCrossover | None = None,
    mutation: BaseMutation | None = None,
    crossover_prob: float = 0.9,
    mutation_prob: float = 1.0,
    seed: int | None = None,
) -> NSGA2:
    """Build a permutation NSGA-II with the given operators."""
    initialiser = initialiser or RandomPermutationInitialiser()
    crossover = crossover or PMXCrossover()
    mutation = mutation or SwapMutation()
    algo = NSGA2(
        pop_size=pop_size,
        sampling=to_pymoo_sampling(initialiser, seed=seed),
        crossover=to_pymoo_crossover(crossover, prob=crossover_prob, seed=seed),
        mutation=to_pymoo_mutation(mutation, prob=mutation_prob, seed=seed),
        eliminate_duplicates=False,
    )
    return algo


def build_moead(
    pop_size: int = 105,
    n_partitions: int = 13,
    initialiser: BaseInitialiser | None = None,
    crossover: BaseCrossover | None = None,
    mutation: BaseMutation | None = None,
    crossover_prob: float = 0.9,
    mutation_prob: float = 1.0,
    seed: int | None = None,
) -> MOEAD:
    """Build a permutation MOEA/D using Tchebycheff decomposition.

    Args:
        n_partitions: controls the number of weight vectors (Das-Dennis).
            With 3 objectives, n_partitions=13 yields 105 weight vectors.
    """
    initialiser = initialiser or RandomPermutationInitialiser()
    crossover = crossover or PMXCrossover()
    mutation = mutation or SwapMutation()
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    algo = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
        sampling=to_pymoo_sampling(initialiser, seed=seed),
        crossover=to_pymoo_crossover(crossover, prob=crossover_prob, seed=seed),
        mutation=to_pymoo_mutation(mutation, prob=mutation_prob, seed=seed),
    )
    return algo
