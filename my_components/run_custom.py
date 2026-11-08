"""Run experiments with your custom decoder / local search plugged in.

The shipped `scripts/run.py` + `cio.harness.run_experiment` only support the
baseline decoder (hardcoded `OTSchedulingProblem(instance)` with no decoder
arg) and the baseline crossover/mutation registry -- there's no config knob
for a custom decoder or local search. This script is a thin, mechanical
extension of that harness (same JSON log schema, same seeding approach) so
your existing/expected downstream analysis (indicators, stats) reads its
output identically to the shipped harness's output.

Usage examples:

    # Constraint-handling-only config: your decoder, baseline operators, NSGA-II
    python my_components/run_custom.py \\
        --algorithm nsga2 --decoder repair \\
        --instance instances/ot_small.json \\
        --out results/nsga2_repair_small/ \\
        --n-runs 30 --n-generations 100

    # Memetic config: your decoder + your local search applied post-convergence
    python my_components/run_custom.py \\
        --algorithm nsga2 --decoder repair --local-search conflict_graph \\
        --ls-budget 200 \\
        --instance instances/ot_small.json \\
        --out results/nsga2_repair_ls_small/ \\
        --n-runs 30 --n-generations 100

    # Baseline decoder for comparison (greedy, same as shipped baseline)
    python my_components/run_custom.py \\
        --algorithm nsga2 --decoder greedy \\
        --instance instances/ot_small.json \\
        --out results/nsga2_greedy_small/ \\
        --n-runs 30 --n-generations 100

Run `PYTHONPATH=.` from the cio_skeleton/ root (same convention as the
rest of the skeleton).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymoo.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cio.problem import Instance, OTSchedulingProblem, GreedyDecoder
from cio.algorithms import build_nsga2, build_moead
from cio.harness.runner import ExperimentConfig, CROSSOVER_REGISTRY, MUTATION_REGISTRY, INITIALISER_REGISTRY
from cio.analysis import non_dominated

from my_components.decoder import RepairBasedDecoder
from my_components.local_search import ConflictGraphLocalSearch

DECODER_REGISTRY = {
    "greedy": GreedyDecoder,
    "repair": RepairBasedDecoder,
}

LOCAL_SEARCH_REGISTRY = {
    "conflict_graph": ConflictGraphLocalSearch,
}


def _build_algorithm(config: ExperimentConfig, seed: int):
    initialiser = INITIALISER_REGISTRY[config.initialiser]()
    crossover = CROSSOVER_REGISTRY[config.crossover]()
    mutation = MUTATION_REGISTRY[config.mutation]()
    if config.algorithm == "nsga2":
        return build_nsga2(
            pop_size=config.pop_size, initialiser=initialiser,
            crossover=crossover, mutation=mutation,
            crossover_prob=config.crossover_prob, mutation_prob=config.mutation_prob,
            seed=seed,
        )
    if config.algorithm == "moead":
        return build_moead(
            pop_size=config.pop_size, n_partitions=config.n_partitions,
            initialiser=initialiser, crossover=crossover, mutation=mutation,
            crossover_prob=config.crossover_prob, mutation_prob=config.mutation_prob,
            seed=seed,
        )
    raise ValueError(f"Unknown algorithm: {config.algorithm}")


def run_one(
    config: ExperimentConfig,
    instance_path: str | Path,
    seed: int,
    decoder_name: str,
    local_search_name: str | None,
    ls_budget: int | None = None,
    eval_budget: int | None = None,
    ls_share: float = 0.5,
    conflict_mode: str = "strict",
) -> dict:
    """Run one seeded configuration.

    Budget accounting (brief 5.1 requires a termination criterion fixed
    across compared configurations):

        eval_budget      total decoder evaluations allowed for the run.
                         Defaults to pop_size * n_generations, i.e. exactly
                         what the non-memetic configurations consume, so
                         every DoE cell is compared at the same budget.
        ls_share         fraction of eval_budget reserved for local search
                         in memetic configurations. The EA phase then runs
                         floor((1-ls_share)*eval_budget / pop_size)
                         generations, and the reserved remainder is split
                         evenly across the individuals to be improved.

    Non-memetic configurations spend the whole budget in the EA phase.
    `ls_budget` (per-individual) is honoured only if given explicitly, which
    reproduces the older un-matched behaviour; leave it None for a matched
    comparison.
    """
    instance = Instance.from_json(instance_path)
    decoder = DECODER_REGISTRY[decoder_name]()
    problem = OTSchedulingProblem(instance, decoder=decoder)

    total_budget = eval_budget or (config.pop_size * config.n_generations)

    if local_search_name is not None and ls_budget is None:
        ea_budget = int(round(total_budget * (1.0 - ls_share)))
        ls_reserve = total_budget - ea_budget
        n_gen = max(1, ea_budget // max(1, config.pop_size))
    else:
        ls_reserve = 0
        n_gen = config.n_generations

    start = time.time()
    algo = _build_algorithm(config, seed)
    res = minimize(problem, algo, ("n_gen", n_gen),
                    seed=seed, verbose=False, save_history=False)
    F = res.F if res.F is not None else np.empty((0, 3))
    X = res.X if res.X is not None else np.empty((0, instance.n_surgeries), dtype=int)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    ea_evals = res.algorithm.evaluator.n_eval if hasattr(res, "algorithm") else 0
    evals = int(ea_evals)
    ls_evals = 0

    if local_search_name is not None and len(X) > 0:
        # Memetic phase. Local search is applied to the final non-dominated
        # set; each individual gets an equal slice of the reserved budget so
        # the *total* run cost still matches the non-memetic cells.
        ls = LOCAL_SEARCH_REGISTRY[local_search_name](decoder=decoder, mode=conflict_mode)
        rng = np.random.default_rng(seed)
        if ls_budget is not None:
            per_individual = int(ls_budget)
        else:
            per_individual = max(2, ls_reserve // len(X))
        improved_X, improved_F = [], []
        for row in X:
            perm = np.asarray(row, dtype=int)
            new_perm, new_f = ls(problem, perm, budget=per_individual, rng=rng)
            ls_evals += int(ls.evals_used)      # measured, not estimated
            improved_X.append(np.asarray(new_perm, dtype=int))
            improved_F.append(np.asarray(new_f, dtype=float))
        improved_X = np.asarray(improved_X, dtype=int)
        improved_F = np.asarray(improved_F, dtype=float)
        mask = non_dominated(improved_F)
        X = improved_X[mask]
        F = improved_F[mask]
        evals += ls_evals

    runtime = time.time() - start
    return {
        "config_name": config.name,
        "instance_name": instance.name,
        "seed": int(seed),
        "F": F.tolist(),
        "X": X.tolist(),
        "evaluations": int(evals),
        "ea_evaluations": int(ea_evals),
        "ls_evaluations": int(ls_evals),
        "eval_budget": int(total_budget),
        "n_generations_effective": int(n_gen),
        "conflict_mode": conflict_mode if local_search_name else None,
        "runtime_seconds": float(runtime),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_experiment(
    config: ExperimentConfig,
    instance_path: str | Path,
    output_dir: str | Path,
    decoder_name: str,
    local_search_name: str | None,
    ls_budget: int | None = None,
    eval_budget: int | None = None,
    ls_share: float = 0.5,
    conflict_mode: str = "strict",
    base_seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = config.make_seeds(base_seed=base_seed)
    results = []
    for i, seed in enumerate(seeds, start=1):
        if verbose:
            print(f"[{config.name}] run {i}/{len(seeds)}  seed={seed}")
        result = run_one(
            config, instance_path, seed, decoder_name, local_search_name,
            ls_budget=ls_budget, eval_budget=eval_budget,
            ls_share=ls_share, conflict_mode=conflict_mode,
        )
        log_path = output_dir / f"run_{i:03d}_seed_{seed}.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        results.append(result)
    summary = {
        "config": config.to_dict(),
        "decoder": decoder_name,
        "local_search": local_search_name,
        "conflict_mode": conflict_mode if local_search_name else None,
        "eval_budget": int(eval_budget or (config.pop_size * config.n_generations)),
        "ls_share": ls_share if local_search_name else None,
        "ls_budget_per_individual": ls_budget,
        "instance_path": str(instance_path),
        "n_runs": len(results),
        "evaluations_mean": float(np.mean([r["evaluations"] for r in results])) if results else 0.0,
        "evaluations_max": int(max((r["evaluations"] for r in results), default=0)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=["nsga2", "moead"], default="nsga2")
    parser.add_argument("--decoder", choices=list(DECODER_REGISTRY), default="greedy")
    parser.add_argument("--local-search", choices=list(LOCAL_SEARCH_REGISTRY), default=None)
    parser.add_argument("--ls-budget", type=int, default=None,
                        help="Per-individual LS evaluations. Leave unset for a "
                             "budget-MATCHED run (LS drawn from the same total "
                             "eval budget as the non-memetic cells). Setting it "
                             "explicitly gives LS extra budget on top -- only do "
                             "that deliberately, and disclose it.")
    parser.add_argument("--eval-budget", type=int, default=None,
                        help="Total decoder evaluations per run. Default pop_size*n_generations.")
    parser.add_argument("--ls-share", type=float, default=0.5,
                        help="Fraction of the total budget reserved for local search.")
    parser.add_argument("--conflict-mode", choices=["strict", "ot"], default="strict",
                        help="Conflict-graph definition for the local search.")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--crossover", type=str, default="pmx", choices=["pmx", "ox"])
    parser.add_argument("--mutation", type=str, default="swap", choices=["swap", "insertion", "inversion"])
    parser.add_argument("--crossover-prob", type=float, default=0.9)
    parser.add_argument("--mutation-prob", type=float, default=1.0)
    parser.add_argument("--n-partitions", type=int, default=13)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig(
        name=args.name or f"{args.algorithm}_{args.decoder}" + (f"_{args.local_search}" if args.local_search else ""),
        algorithm=args.algorithm,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        crossover=args.crossover,
        crossover_prob=args.crossover_prob,
        mutation=args.mutation,
        mutation_prob=args.mutation_prob,
        n_partitions=args.n_partitions,
        n_runs=args.n_runs,
    )

    run_experiment(
        config=config,
        instance_path=args.instance,
        output_dir=args.out,
        decoder_name=args.decoder,
        local_search_name=args.local_search,
        ls_budget=args.ls_budget,
        eval_budget=args.eval_budget,
        ls_share=args.ls_share,
        conflict_mode=args.conflict_mode,
        base_seed=args.base_seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
