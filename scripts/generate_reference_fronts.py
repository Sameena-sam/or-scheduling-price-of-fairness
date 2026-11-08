"""Generate reference Pareto fronts for the design instances.

The reference front for each instance is the union of non-dominated solutions
found by running all three baselines repeatedly with generous budgets, then
keeping only non-dominated points across all runs and algorithms.

Reference fronts are saved as JSON arrays of objective values, and are used
downstream for computing IGD+.

IMPORTANT: The default budgets here are PRELIMINARY (small enough to fit a
few minutes on a typical workstation). For a properly competitive reference
front against which student work will be compared, the budgets should be
increased substantially (10-20x the values below) and the script re-run
overnight or on a more powerful machine. Larger budgets in the GENEROUS_BUDGETS
dict below correspond to the standards expected for a published artefact.

Usage:
    # Preliminary (default), fast (~5-15 minutes):
    python scripts/generate_reference_fronts.py --instances instances/

    # Generous (slow, run overnight):
    python scripts/generate_reference_fronts.py --instances instances/ --generous
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pymoo.optimize import minimize

from cio.problem import Instance, OTSchedulingProblem
from cio.algorithms import build_nsga2, build_moead, pareto_local_search
from cio.analysis.indicators import union_non_dominated


# Budgets per instance size. These are GENEROUS so reference fronts are good
# benchmarks for student work. Tune down for faster generation if needed.
# Preliminary budgets: fast generation (~5-15 minutes). Use for development
# and initial testing. For the production reference front against which
# student work is graded, use --generous to switch to GENEROUS_BUDGETS.
BUDGETS = {
    "small": {
        "n_runs_per_algo": 5,
        "nsga2_gens": 200,
        "nsga2_pop": 100,
        "moead_gens": 100,
        "moead_pop": 105,
        "pls_evaluations": 15_000,
    },
    "medium": {
        "n_runs_per_algo": 2,
        "nsga2_gens": 100,
        "nsga2_pop": 100,
        "moead_gens": 50,
        "moead_pop": 105,
        "pls_evaluations": 15_000,
    },
    "large": {
        "n_runs_per_algo": 1,
        "nsga2_gens": 40,
        "nsga2_pop": 80,
        "moead_gens": 20,
        "moead_pop": 105,
        "pls_evaluations": 8_000,
    },
}

# Generous budgets for the production reference front (10-20x preliminary).
# Expected runtime: several hours on a workstation; run overnight.
GENEROUS_BUDGETS = {
    "small": {
        "n_runs_per_algo": 15,
        "nsga2_gens": 400,
        "nsga2_pop": 150,
        "moead_gens": 300,
        "moead_pop": 105,
        "pls_evaluations": 50_000,
    },
    "medium": {
        "n_runs_per_algo": 10,
        "nsga2_gens": 400,
        "nsga2_pop": 200,
        "moead_gens": 300,
        "moead_pop": 105,
        "pls_evaluations": 80_000,
    },
    "large": {
        "n_runs_per_algo": 6,
        "nsga2_gens": 400,
        "nsga2_pop": 250,
        "moead_gens": 300,
        "moead_pop": 105,
        "pls_evaluations": 120_000,
    },
}


def run_nsga2_runs(problem, gens, pop, n_runs):
    fronts = []
    for seed in range(n_runs):
        algo = build_nsga2(pop_size=pop, seed=seed)
        res = minimize(problem, algo, ("n_gen", gens), seed=seed, verbose=False)
        if res.F is not None:
            F = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)
            fronts.append(F)
    return fronts


def run_moead_runs(problem, gens, pop, n_runs):
    # n_partitions=13 -> 105 weight vectors for 3 objectives
    fronts = []
    for seed in range(n_runs):
        algo = build_moead(pop_size=pop, n_partitions=13, seed=seed)
        res = minimize(problem, algo, ("n_gen", gens), seed=seed, verbose=False)
        if res.F is not None:
            F = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)
            fronts.append(F)
    return fronts


def run_pls_runs(problem, evals, n_runs):
    fronts = []
    for seed in range(n_runs):
        result = pareto_local_search(problem, n_evaluations=evals, seed=seed)
        if result.archive_F.size > 0:
            fronts.append(result.archive_F)
    return fronts


def generate_reference_front(instance_path: Path, generous: bool = False) -> np.ndarray:
    instance = Instance.from_json(instance_path)
    problem = OTSchedulingProblem(instance)
    budgets = GENEROUS_BUDGETS[instance.name] if generous else BUDGETS[instance.name]
    label = "GENEROUS" if generous else "PRELIMINARY"

    print(f"\n[{instance.name}] generating reference front  ({label} budgets)")
    t0 = time.time()

    print(f"  NSGA-II: {budgets['n_runs_per_algo']} runs, pop={budgets['nsga2_pop']}, gens={budgets['nsga2_gens']}")
    nsga2_fronts = run_nsga2_runs(problem, budgets["nsga2_gens"],
                                  budgets["nsga2_pop"], budgets["n_runs_per_algo"])

    print(f"  MOEA/D: {budgets['n_runs_per_algo']} runs, pop={budgets['moead_pop']}, gens={budgets['moead_gens']}")
    moead_fronts = run_moead_runs(problem, budgets["moead_gens"],
                                  budgets["moead_pop"], budgets["n_runs_per_algo"])

    print(f"  PLS: {budgets['n_runs_per_algo']} runs, evals={budgets['pls_evaluations']}")
    pls_fronts = run_pls_runs(problem, budgets["pls_evaluations"],
                              budgets["n_runs_per_algo"])

    all_fronts = nsga2_fronts + moead_fronts + pls_fronts
    if not all_fronts:
        raise RuntimeError(f"No fronts produced for {instance.name}")
    reference = union_non_dominated(*all_fronts)
    elapsed = time.time() - t0
    print(f"  reference front size: {len(reference)} non-dominated points")
    print(f"  elapsed: {elapsed:.1f}s")
    return reference


def save_reference(reference: np.ndarray, output_path: Path, generous: bool = False) -> None:
    payload = {
        "n_objectives": int(reference.shape[1]) if reference.size > 0 else 3,
        "n_points": int(len(reference)),
        "budget_profile": "generous" if generous else "preliminary",
        "points": reference.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", type=Path, default=Path("instances"),
                        help="Directory containing instance files")
    parser.add_argument("--only", type=str, default=None,
                        choices=["small", "medium", "large"],
                        help="Generate reference for only one instance size")
    parser.add_argument("--generous", action="store_true",
                        help="Use generous (slow, overnight) budgets for "
                             "production-quality reference fronts")
    args = parser.parse_args()

    sizes = [args.only] if args.only else ["small", "medium", "large"]
    for size in sizes:
        instance_path = args.instances / f"ot_{size}.json"
        if not instance_path.exists():
            print(f"WARNING: {instance_path} not found, skipping")
            continue
        reference = generate_reference_front(instance_path, generous=args.generous)
        save_reference(reference, args.instances / f"ot_{size}_reference_front.json",
                       generous=args.generous)


if __name__ == "__main__":
    main()
