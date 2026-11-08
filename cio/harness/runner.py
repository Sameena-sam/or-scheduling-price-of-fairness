"""Experiment harness.

A simple, reproducible harness for running algorithm configurations on
instances with controlled seeds and recording results.

Typical usage:

    from cio.harness import ExperimentConfig, run_experiment

    config = ExperimentConfig.from_yaml("configs/nsga2_default.yaml")
    results = run_experiment(config, instance_path="instances/ot_small.json",
                             output_dir="results/nsga2_small/")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import numpy as np
import yaml
from pymoo.optimize import minimize

from ..problem import Instance, OTSchedulingProblem
from ..algorithms import build_nsga2, build_moead, pareto_local_search
from ..operators import (
    PMXCrossover, OrderCrossover,
    SwapMutation, InsertionMutation, InversionMutation,
    RandomPermutationInitialiser,
)


# Registry: algorithm name -> builder factory
ALGORITHM_REGISTRY = {
    "nsga2": "nsga2",
    "moead": "moead",
    "pls": "pls",
}

CROSSOVER_REGISTRY = {
    "pmx": PMXCrossover,
    "ox": OrderCrossover,
}

MUTATION_REGISTRY = {
    "swap": SwapMutation,
    "insertion": InsertionMutation,
    "inversion": InversionMutation,
}

INITIALISER_REGISTRY = {
    "random_permutation": RandomPermutationInitialiser,
}


@dataclass
class ExperimentConfig:
    name: str
    algorithm: str                       # "nsga2", "moead", or "pls"
    pop_size: int = 100
    n_generations: int = 100             # only for EA-based algorithms
    n_evaluations: int = 10_000          # used by PLS (and as alternative termination)
    crossover: str = "pmx"
    crossover_prob: float = 0.9
    mutation: str = "swap"
    mutation_prob: float = 1.0
    initialiser: str = "random_permutation"
    n_partitions: int = 13               # for MOEA/D
    n_runs: int = 30
    seeds: list[int] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def make_seeds(self, base_seed: int = 0) -> list[int]:
        if self.seeds:
            return list(self.seeds)
        # Deterministic seed sequence
        rng = np.random.default_rng(base_seed)
        return [int(rng.integers(0, 2**31 - 1)) for _ in range(self.n_runs)]


def _build_algorithm(config: ExperimentConfig, seed: int):
    initialiser = INITIALISER_REGISTRY[config.initialiser]()
    crossover = CROSSOVER_REGISTRY[config.crossover]()
    mutation = MUTATION_REGISTRY[config.mutation]()
    if config.algorithm == "nsga2":
        return build_nsga2(
            pop_size=config.pop_size,
            initialiser=initialiser,
            crossover=crossover,
            mutation=mutation,
            crossover_prob=config.crossover_prob,
            mutation_prob=config.mutation_prob,
            seed=seed,
        )
    if config.algorithm == "moead":
        return build_moead(
            pop_size=config.pop_size,
            n_partitions=config.n_partitions,
            initialiser=initialiser,
            crossover=crossover,
            mutation=mutation,
            crossover_prob=config.crossover_prob,
            mutation_prob=config.mutation_prob,
            seed=seed,
        )
    raise ValueError(f"Unknown algorithm: {config.algorithm}")


def run_one(
    config: ExperimentConfig,
    instance_path: str | Path,
    seed: int,
) -> dict:
    """Run a single (algorithm, instance, seed) experiment."""
    instance = Instance.from_json(instance_path)
    problem = OTSchedulingProblem(instance)
    start = time.time()
    if config.algorithm == "pls":
        pls_res = pareto_local_search(
            problem,
            n_evaluations=config.n_evaluations,
            seed=seed,
        )
        F = pls_res.archive_F
        X = pls_res.archive_X
        evals = pls_res.evaluations_used
    else:
        algo = _build_algorithm(config, seed)
        # Use n_gen termination by default, but allow n_eval as alternative
        res = minimize(problem, algo, ("n_gen", config.n_generations),
                       seed=seed, verbose=False, save_history=False)
        F = res.F if res.F is not None else np.empty((0, 3))
        X = res.X if res.X is not None else np.empty((0, instance.n_surgeries), dtype=int)
        # If F is a single solution (1D), wrap it
        if F.ndim == 1:
            F = F.reshape(1, -1)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        evals = res.algorithm.evaluator.n_eval if hasattr(res, "algorithm") else 0
    runtime = time.time() - start
    return {
        "config_name": config.name,
        "instance_name": instance.name,
        "seed": int(seed),
        "F": F.tolist(),
        "X": X.tolist(),
        "evaluations": int(evals),
        "runtime_seconds": float(runtime),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_experiment(
    config: ExperimentConfig,
    instance_path: str | Path,
    output_dir: str | Path,
    base_seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """Run all configured runs and save logs to output_dir.

    Returns the list of per-run result dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = config.make_seeds(base_seed=base_seed)
    results = []
    for i, seed in enumerate(seeds, start=1):
        if verbose:
            print(f"[{config.name}] run {i}/{len(seeds)}  seed={seed}")
        result = run_one(config, instance_path, seed)
        # Write per-run JSON log
        log_path = output_dir / f"run_{i:03d}_seed_{seed}.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        results.append(result)
    # Aggregate config info
    summary = {
        "config": config.to_dict(),
        "instance_path": str(instance_path),
        "n_runs": len(results),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return results
