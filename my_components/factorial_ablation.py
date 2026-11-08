"""Confirmatory crossed factorial: eviction rule x reinsertion policy.

WHY THIS EXISTS. The headline comparison (repair vs greedy) was run at full
protocol -- 3 instances, 30 seeded runs, matched evaluation budget. The
mechanism *attribution* (how much of the deficit belongs to the eviction rule
versus to the allow-then-repair mechanism itself) was initially explored with
one-factor-at-a-time changes on single decodes of random permutations. That is
weaker evidence than the claim requires, for two reasons:

  1. varying one factor at a time cannot detect interaction between factors;
  2. decode-level behaviour need not survive contact with the evolutionary
     search -- and in a pilot it did not (reinsertion policy had a negligible
     decode-level effect on f3 but a visible search-level effect on front
     cardinality).

This script therefore runs the full 3 x 2 crossed design under the same
protocol as the main experiments, so the attribution claim rests on evidence
of the same standard as the headline result.

DESIGN
    Factor A -- eviction rule:        priority | permutation | random
    Factor B -- reinsertion policy:   dayplus1 | earliest
    Reference cell:                   greedy decoder (no repair; neither
                                      factor is defined, so it sits outside
                                      the factorial as a control)

NOTE ON SEPARABILITY. Permissive placement and the repair pass are not
independent factors: repair presupposes permissive placement, and enforcing
capacity during placement *is* the greedy decoder. No design can separate
them, so no experiment here attributes effects between those two.

USAGE (from the cio_skeleton/ root)

    python my_components/factorial_ablation.py --instance instances/ot_medium.json \\
        --out results/factorial_medium/ --n-runs 30 --n-generations 100

    # quick check first (~1 min)
    python my_components/factorial_ablation.py --instance instances/ot_small.json \\
        --out results/factorial_small_quick/ --n-runs 3 --n-generations 20

Writes the same JSON-per-run schema as the main harness, so
`my_components/analyse.py` can read the output directly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymoo.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cio.problem import Instance, OTSchedulingProblem, GreedyDecoder
from cio.algorithms import build_nsga2
from my_components.decoder import RepairBasedDecoder

EVICTION_RULES = ["priority", "permutation", "random"]
REINSERTION_POLICIES = ["dayplus1", "earliest"]


class AblationDecoder(RepairBasedDecoder):
    """RepairBasedDecoder with both mechanisms exposed as factors.

    DIAGNOSTIC ONLY. The submitted algorithm is `RepairBasedDecoder`, which is
    exactly (evict="priority", reinsert="dayplus1"). This subclass exists to
    measure the contribution of each mechanism, not to replace it.
    """

    def __init__(self, evict: str = "priority", reinsert: str = "dayplus1",
                 rule_seed: int = 0, **kw):
        super().__init__(**kw)
        if evict not in EVICTION_RULES:
            raise ValueError(f"unknown eviction rule {evict!r}")
        if reinsert not in REINSERTION_POLICIES:
            raise ValueError(f"unknown reinsertion policy {reinsert!r}")
        self.evict, self.reinsert = evict, reinsert
        self.rng = np.random.default_rng(rule_seed)

    def __call__(self, instance, permutation):
        # Permutation rank is needed by the "permutation" eviction rule; it is
        # what makes that rule genotype-dependent rather than a fixed function
        # of instance properties.
        self._rank = {int(v): i for i, v in enumerate(permutation)}
        self._idx_of_id = {instance.surgery_by_index(i).id: i
                           for i in range(instance.n_surgeries)}
        return super().__call__(instance, permutation)

    def _select_surgery_to_evict(self, instance, sched, candidates, day, resource):
        if self.evict == "priority":
            return min(candidates, key=lambda s: s.priority).id
        if self.evict == "permutation":
            return max(candidates, key=lambda s: self._rank[self._idx_of_id[s.id]]).id
        return candidates[int(self.rng.integers(len(candidates)))].id

    def _repair(self, instance, sched):
        if self.reinsert == "dayplus1":
            return super()._repair(instance, sched)

        # "earliest": reinsertion may search from the surgery's earliest_day.
        # Anti-cycling is then explicit -- a (surgery, day) pair is allowed at
        # most once, after which the strictly-later search is used as fallback.
        max_passes = 4 * instance.horizon_days + 2 * len(sched.assignments) + 16
        S = {s.id: s for s in instance.surgeries}
        O = {o.id: o for o in instance.ots}
        T = {t.id: t for t in instance.teams}
        self.last_repair_passes = self.last_evictions = self.last_evictions_unplaceable = 0
        seen: set[tuple[int, int]] = set()

        for _ in range(max_passes):
            self.last_repair_passes += 1
            overflow = self._find_first_overflow(instance, sched)
            if overflow is None:
                break
            day, resource = overflow
            candidates = self._occupants(sched, S, day, resource)
            if not candidates:
                break
            evict_id = self._select_surgery_to_evict(instance, sched, candidates, day, resource)
            surgery = S[evict_id]
            self._evict(instance, sched, surgery)
            self.last_evictions += 1

            placement = None
            if (evict_id, day) not in seen:
                seen.add((evict_id, day))
                placement = self._try_place(instance, surgery, sched, O, T,
                                            enforce_downstream_capacity=True,
                                            earliest_search_day=None)
            if placement is None:
                placement = self._try_place(instance, surgery, sched, O, T,
                                            enforce_downstream_capacity=True,
                                            earliest_search_day=day + 1)
            if placement is not None:
                self._apply_assignment(instance, placement, surgery, sched)
            else:
                sched.unscheduled_surgery_ids.append(evict_id)
                self.last_evictions_unplaceable += 1


def make_seeds(n_runs: int, base_seed: int = 0) -> list[int]:
    """Same deterministic sequence the main harness uses, so the factorial
    cells are seed-comparable with the headline configurations."""
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(n_runs)]


def run_cell(instance_path, out_dir, decoder_factory, cell_name,
             n_runs, n_generations, pop_size, base_seed, verbose):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    instance = Instance.from_json(instance_path)
    seeds = make_seeds(n_runs, base_seed)
    results = []
    for i, seed in enumerate(seeds, start=1):
        decoder = decoder_factory()
        problem = OTSchedulingProblem(instance, decoder=decoder)
        t0 = time.time()
        res = minimize(problem, build_nsga2(pop_size=pop_size, seed=seed),
                       ("n_gen", n_generations), seed=seed, verbose=False)
        F = res.F if res.F is not None else np.empty((0, 3))
        X = res.X if res.X is not None else np.empty((0, instance.n_surgeries), dtype=int)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        record = {
            "config_name": cell_name,
            "instance_name": instance.name,
            "seed": int(seed),
            "F": F.tolist(),
            "X": X.tolist(),
            "evaluations": int(res.algorithm.evaluator.n_eval) if hasattr(res, "algorithm") else 0,
            "eval_budget": int(pop_size * n_generations),
            "n_generations_effective": int(n_generations),
            "runtime_seconds": float(time.time() - t0),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_dir / f"run_{i:03d}_seed_{seed}.json", "w") as f:
            json.dump(record, f, indent=2)
        results.append(record)
        if verbose:
            print(f"    run {i}/{n_runs}", end="\r", flush=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"config": {"name": cell_name, "pop_size": pop_size,
                              "n_generations": n_generations, "n_runs": n_runs},
                   "instance_path": str(instance_path), "n_runs": len(results),
                   "eval_budget": int(pop_size * n_generations),
                   "completed_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--n-generations", type=int, default=100)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--rule-seed", type=int, default=17,
                    help="seed for the stochastic 'random' eviction rule")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    inst_path = args.instance if args.instance.is_absolute() else root / args.instance
    out_root = args.out if args.out.is_absolute() else root / args.out
    total = time.time()

    print(f"Crossed factorial on {inst_path.name}: "
          f"{len(EVICTION_RULES)}x{len(REINSERTION_POLICIES)} cells + greedy control, "
          f"{args.n_runs} runs, {args.n_generations} gens")

    t0 = time.time()
    print("  [control] greedy (no repair) ...", flush=True)
    run_cell(inst_path, out_root / "control_greedy", lambda: GreedyDecoder(),
             "control_greedy", args.n_runs, args.n_generations, args.pop_size,
             args.base_seed, not args.quiet)
    print(f"  [control] greedy done in {time.time()-t0:.0f}s", flush=True)

    for evict, reinsert in itertools.product(EVICTION_RULES, REINSERTION_POLICIES):
        cell = f"{evict}_{reinsert}"
        t0 = time.time()
        print(f"  [{cell}] ...", flush=True)
        run_cell(inst_path, out_root / cell,
                 lambda e=evict, r=reinsert: AblationDecoder(evict=e, reinsert=r,
                                                             rule_seed=args.rule_seed),
                 cell, args.n_runs, args.n_generations, args.pop_size,
                 args.base_seed, not args.quiet)
        print(f"  [{cell}] done in {time.time()-t0:.0f}s", flush=True)

    print(f"\nFactorial complete in {time.time()-total:.0f}s -> {out_root}")
    print("Analyse with:")
    print(f"  python my_components/analyse_factorial.py --results {args.out}")


if __name__ == "__main__":
    main()
