"""Conflict-graph-guided Pareto local search.

Design (student-authored): instead of exploring swap-neighbours drawn
uniformly at random from all C(n,2) surgery pairs (as the shipped
SwapNeighbourhoodLocalSearch baseline does), this local search restricts
the neighbourhood to pairs of surgeries that genuinely *compete* for the
same scheduling slot. Swapping the relative decode order of two surgeries
that can never contend for the same resource cannot change either one's
placement, so evaluating that swap wastes budget.

Two conflict definitions are provided so the restriction can be measured
rather than assumed:

  mode="ot"      surgeries conflict iff they share >=1 eligible OT.
  mode="strict"  surgeries conflict iff they share >=1 eligible OT
                 AND >=1 eligible team AND their admissible day-windows
                 [earliest_day, earliest_day + max_wait_days] overlap.

"strict" is the definition that matches the theatre-day-team slot-conflict
structure; "ot" is the weaker relation. Measure both with
`conflict_graph_density()` and report the numbers -- on the shipped
instances the "ot" relation is 100% dense on small and medium (i.e. the
restriction is vacuous there), which is exactly the sort of thing that
should be reported rather than assumed away.
"""
from __future__ import annotations

import numpy as np

from cio.problem import Instance, OTSchedulingProblem
from cio.problem.evaluator import evaluate as _evaluate
from cio.problem.decoder import GreedyDecoder


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Strict Pareto dominance: a dominates b iff a <= b component-wise
    AND a < b in at least one component."""
    return bool(np.all(a <= b) and np.any(a < b))


def build_conflict_graph(instance: Instance, mode: str = "strict") -> dict[int, set[int]]:
    """Static adjacency over permutation *positions* (0-indexed surgery
    array index, matching the permutation encoding -- not surgery.id).

    mode="ot":     share >=1 eligible OT
    mode="strict": share >=1 eligible OT AND >=1 eligible team AND
                   overlapping admissible day-windows
    """
    if mode not in ("ot", "strict"):
        raise ValueError(f"unknown conflict mode: {mode!r}")

    n = instance.n_surgeries
    ot_sets, team_sets, windows = [], [], []
    for idx in range(n):
        s = instance.surgery_by_index(idx)
        ot_sets.append(set(instance.eligible_ots_for_surgery[s.id]))
        team_sets.append(set(instance.eligible_teams_for_surgery[s.id]))
        lo = s.earliest_day
        hi = min(instance.horizon_days, s.earliest_day + s.max_wait_days)
        windows.append((lo, hi))

    graph: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if not (ot_sets[i] & ot_sets[j]):
                continue
            if mode == "strict":
                if not (team_sets[i] & team_sets[j]):
                    continue
                lo_i, hi_i = windows[i]
                lo_j, hi_j = windows[j]
                if hi_i < lo_j or hi_j < lo_i:      # windows disjoint
                    continue
            graph[i].add(j)
            graph[j].add(i)
    return graph


def conflict_graph_density(instance: Instance, mode: str = "strict") -> dict:
    """Report edges / max_edges for a conflict definition. Use this to
    justify (or honestly qualify) the neighbourhood restriction in the paper."""
    g = build_conflict_graph(instance, mode=mode)
    n = instance.n_surgeries
    edges = sum(len(v) for v in g.values()) // 2
    max_edges = n * (n - 1) // 2
    return {
        "instance": instance.name, "mode": mode, "n": n,
        "edges": edges, "max_edges": max_edges,
        "density": edges / max_edges if max_edges else 0.0,
    }


class ConflictGraphLocalSearch:
    """Pareto local search restricted to the conflict-graph neighbourhood.

    Args:
        decoder: decoder used to evaluate candidates. Pass the same decoder
            the surrounding configuration uses, so local search and global
            search share constraint-handling behaviour.
        mode: conflict definition, "strict" (default) or "ot".

    After each call, `self.evals_used` holds the number of decoder
    evaluations actually consumed -- so the caller can do honest budget
    accounting instead of estimating.
    """

    def __init__(self, decoder=None, mode: str = "strict"):
        self.decoder = decoder or GreedyDecoder()
        self.mode = mode
        self.evals_used = 0
        self._edges: list[tuple[int, int]] = []
        self._graph_instance_name: str | None = None

    def _edges_for(self, instance: Instance) -> list[tuple[int, int]]:
        """Cache the edge list per instance (built once, not once per pass)."""
        if self._graph_instance_name != instance.name:
            graph = build_conflict_graph(instance, mode=self.mode)
            edges = []
            for u, neighbours in graph.items():
                for v in neighbours:
                    if u < v:
                        edges.append((u, v))
            self._edges = edges
            self._graph_instance_name = instance.name
        return self._edges

    def __call__(self, problem: OTSchedulingProblem, perm: np.ndarray, budget: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inst = problem.instance
        edges = self._edges_for(inst)

        current = perm.copy()
        sched = self.decoder(inst, current)
        current_f = np.array(_evaluate(inst, sched))
        evals = 1

        if not edges:                      # degenerate: no conflicts at all
            self.evals_used = evals
            return current, current_f

        improved = True
        while improved and evals < budget:
            improved = False
            # Sample only as many candidate edges as remaining budget allows,
            # instead of materialising+shuffling the whole edge list each pass.
            for i, j in self._sample_edges(edges, budget - evals, rng):
                if evals >= budget:
                    break
                pos = {int(v): p for p, v in enumerate(current)}
                a, b = pos[i], pos[j]
                cand = current.copy()
                cand[a], cand[b] = cand[b], cand[a]
                sched = self.decoder(inst, cand)
                cand_f = np.array(_evaluate(inst, sched))
                evals += 1
                if dominates(cand_f, current_f):
                    current = cand
                    current_f = cand_f
                    improved = True
                    break

        self.evals_used = evals
        return current, current_f

    def _sample_edges(self, edges: list[tuple[int, int]], k: int,
                      rng: np.random.Generator) -> list[tuple[int, int]]:
        """Draw up to k distinct conflict edges. O(k), not O(|E|)."""
        k = max(0, int(k))
        if k == 0:
            return []
        n_edges = len(edges)
        if k >= n_edges:
            order = rng.permutation(n_edges)
            return [edges[int(t)] for t in order]
        picked, seen = [], set()
        # Oversample slightly to absorb duplicate draws, then trim.
        for t in rng.integers(0, n_edges, size=min(n_edges, int(k * 1.3) + 8)):
            t = int(t)
            if t in seen:
                continue
            seen.add(t)
            picked.append(edges[t])
            if len(picked) == k:
                break
        return picked
