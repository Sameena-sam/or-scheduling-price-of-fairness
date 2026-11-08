"""Pareto Local Search baseline (third required baseline per the brief).

This is a stand-alone search procedure that maintains an archive of
non-dominated solutions and iteratively explores neighbourhoods. Implemented
outside pymoo's algorithm class for clarity.

The PLS algorithm here is the "random-restart" variant:
    1. Sample a random permutation and evaluate
    2. Explore swap-neighbourhood; add any non-dominated neighbour to the
       archive; remove any newly-dominated archive members
    3. If no improvement, restart from a random permutation
    4. Repeat until evaluation budget exhausted
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..problem import Instance, OTSchedulingProblem
from ..problem.evaluator import evaluate
from ..problem.decoder import GreedyDecoder
from ..operators.local_search import dominates


@dataclass
class PLSResult:
    archive_X: np.ndarray   # shape (n_archive, n_var)
    archive_F: np.ndarray   # shape (n_archive, 3)
    evaluations_used: int
    history: list[dict] = field(default_factory=list)


def pareto_local_search(
    problem: OTSchedulingProblem,
    n_evaluations: int = 10_000,
    seed: int | None = None,
    initial_archive_size: int = 20,
) -> PLSResult:
    """Run Pareto local search with random restart.

    Args:
        problem: the OT scheduling problem
        n_evaluations: budget of objective evaluations
        seed: random seed
        initial_archive_size: number of random starts seeding the archive
    """
    rng = np.random.default_rng(seed)
    inst = problem.instance
    decoder = GreedyDecoder()
    n_var = inst.n_surgeries

    archive_perms: list[np.ndarray] = []
    archive_Fs: list[np.ndarray] = []
    evals = 0

    def add_to_archive(perm: np.ndarray, F: np.ndarray) -> bool:
        # Reject if dominated by any archive member.
        for i, archF in enumerate(archive_Fs):
            if dominates(archF, F) or np.array_equal(archF, F):
                return False
        # Remove members dominated by this new point.
        keep = [i for i, archF in enumerate(archive_Fs) if not dominates(F, archF)]
        new_perms = [archive_perms[i] for i in keep] + [perm.copy()]
        new_Fs = [archive_Fs[i] for i in keep] + [F.copy()]
        archive_perms.clear(); archive_Fs.clear()
        archive_perms.extend(new_perms); archive_Fs.extend(new_Fs)
        return True

    # Seed archive
    for _ in range(initial_archive_size):
        if evals >= n_evaluations:
            break
        p = rng.permutation(n_var)
        sched = decoder(inst, p)
        F = np.array(evaluate(inst, sched))
        evals += 1
        add_to_archive(p, F)

    # Local search loop: pick a random unexplored archive member, explore its
    # swap-neighbourhood for one pass, then move on.
    explored_set: set[int] = set()
    while evals < n_evaluations:
        if len(archive_perms) == 0:
            break
        # Find an unexplored member; if none, restart
        candidates = [i for i in range(len(archive_perms)) if i not in explored_set]
        if not candidates:
            # Restart: random restart added to archive
            explored_set.clear()
            p = rng.permutation(n_var)
            sched = decoder(inst, p)
            F = np.array(evaluate(inst, sched))
            evals += 1
            add_to_archive(p, F)
            continue
        idx = candidates[rng.integers(len(candidates))]
        base = archive_perms[idx]
        # Single pass over swap-neighbourhood (sampled, not exhaustive)
        n_samples = min(len(base) * 3, n_evaluations - evals)
        for _ in range(n_samples):
            if evals >= n_evaluations:
                break
            i, j = rng.choice(len(base), size=2, replace=False)
            cand = base.copy()
            cand[i], cand[j] = cand[j], cand[i]
            sched = decoder(inst, cand)
            F = np.array(evaluate(inst, sched))
            evals += 1
            add_to_archive(cand, F)
        explored_set.add(idx)

    archive_X = np.stack(archive_perms) if archive_perms else np.empty((0, n_var), dtype=int)
    archive_F = np.stack(archive_Fs) if archive_Fs else np.empty((0, 3))
    return PLSResult(archive_X=archive_X, archive_F=archive_F, evaluations_used=evals)
