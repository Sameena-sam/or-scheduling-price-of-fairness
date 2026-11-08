"""Run an experiment from a YAML config (with optional CLI overrides).

Usage:
    python scripts/run.py --config configs/nsga2_baseline.yaml \\
                          --instance instances/ot_small.json \\
                          --out results/nsga2_small/

    # Override config values from the command line:
    python scripts/run.py --config configs/nsga2_baseline.yaml \\
                          --instance instances/ot_small.json \\
                          --out results/nsga2_small_30g/ \\
                          --n-generations 30 --n-runs 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the cio package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cio.harness import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config file")
    parser.add_argument("--instance", type=Path, required=True,
                        help="Instance JSON file")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for run logs")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Base seed for the seed sequence")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-run progress output")

    # CLI overrides for common parameters (subset; extend as needed)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-generations", type=int, default=None)
    parser.add_argument("--n-evaluations", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--crossover", type=str, default=None,
                        choices=["pmx", "ox"])
    parser.add_argument("--mutation", type=str, default=None,
                        choices=["swap", "insertion", "inversion"])
    parser.add_argument("--crossover-prob", type=float, default=None)
    parser.add_argument("--mutation-prob", type=float, default=None)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    # Apply CLI overrides
    overrides = {
        "n_runs": args.n_runs,
        "n_generations": args.n_generations,
        "n_evaluations": args.n_evaluations,
        "pop_size": args.pop_size,
        "crossover": args.crossover,
        "mutation": args.mutation,
        "crossover_prob": args.crossover_prob,
        "mutation_prob": args.mutation_prob,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)

    run_experiment(
        config=config,
        instance_path=args.instance,
        output_dir=args.out,
        base_seed=args.base_seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
