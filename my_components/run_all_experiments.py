"""Cross-platform driver: runs the 3 required baselines + your 2x2
factorial DoE, on all three instances, and writes standard JSON logs to
results/. Pure Python (no bash needed) so it works the same on Windows,
macOS, and Linux.

Run from the cio_skeleton/ root:

    python my_components/run_all_experiments.py

Quick test first (recommended -- takes ~1 minute, confirms everything
wired correctly before committing to the multi-hour full run):

    python my_components/run_all_experiments.py --quick

Full run in the background so it survives closing the terminal:

    Windows (PowerShell):
        Start-Process python -ArgumentList "my_components/run_all_experiments.py" `
            -RedirectStandardOutput results/run_all.log -RedirectStandardError results/run_all_err.log -NoNewWindow
        # check progress any time with:
        Get-Content results/run_all.log -Wait -Tail 20

    macOS / Linux / WSL:
        nohup python3 my_components/run_all_experiments.py > results/run_all.log 2>&1 &
        tail -f results/run_all.log

RESUME: safe to interrupt (Ctrl+C) and rerun any time -- any (instance,
config) pair whose summary.json already reports >= the requested n_runs is
skipped automatically, so you never lose already-completed work.

If large is taking too long against your deadline, cut its budget
specifically (small/medium keep full 30 runs):

    python my_components/run_all_experiments.py --large-n-runs 10 --large-n-generations 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cio.harness.runner import ExperimentConfig, run_experiment as run_baseline_experiment
from my_components.run_custom import run_experiment as run_custom_experiment

INSTANCES = ["ot_small.json", "ot_medium.json", "ot_large.json"]


def _already_done(output_dir: Path, n_runs: int) -> bool:
    """True if this config already has a summary.json reporting >= n_runs
    completed runs -- lets you safely interrupt and rerun without redoing
    finished work."""
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return int(summary.get("n_runs", 0)) >= n_runs
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: 2 runs, 10 generations, small instance only")
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--large-n-runs", type=int, default=None,
                        help="Override n-runs specifically for ot_large.json")
    parser.add_argument("--large-n-generations", type=int, default=None,
                        help="Override n-generations specifically for ot_large.json")
    parser.add_argument("--ls-budget", type=int, default=None,
                        help="Per-individual LS evaluations. Leave unset (default) "
                             "for a budget-matched DoE.")
    parser.add_argument("--ls-share", type=float, default=0.5,
                        help="Fraction of the total eval budget reserved for local search.")
    parser.add_argument("--conflict-mode", choices=["strict", "ot"], default="strict",
                        help="Conflict-graph definition used by the local search.")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="Subset of instances to run, e.g. --instances ot_small.json")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing results and rerun everything from scratch")
    args = parser.parse_args()

    if args.quick:
        default_n_runs, default_n_gen = 2, 10
        instances = ["ot_small.json"]
        print(">>> QUICK MODE: 2 runs, 10 generations, small instance only <<<")
    else:
        default_n_runs, default_n_gen = args.n_runs, args.n_generations
        instances = args.instances or INSTANCES

    root = Path(__file__).resolve().parent.parent
    total_start = time.time()

    for instance_file in instances:
        inst_name = instance_file.replace("ot_", "").replace(".json", "")
        instance_path = root / "instances" / instance_file

        # Per-instance overrides (currently only "large" has one, but this
        # generalises if you need to tune others later).
        if inst_name == "large" and not args.quick:
            n_runs = args.large_n_runs or default_n_runs
            n_gen = args.large_n_generations or default_n_gen
        else:
            n_runs = default_n_runs
            n_gen = default_n_gen

        print(f"\n========== [{inst_name}] starting ({n_runs} runs, {n_gen} gens) ==========", flush=True)

        # --- Required baselines ---
        # NOTE: baseline_nsga2 IS the (greedy decoder, no local search) cell of
        # the DoE -- identical configuration, identical seeds, identical output.
        # It is therefore run once and reused as `doe_greedy_noLS` rather than
        # recomputed under a second name.
        baseline_configs = [
            ("baseline_nsga2", "configs/nsga2_baseline.yaml", n_gen),
            ("baseline_moead", "configs/moead_baseline.yaml", n_gen),
            ("baseline_pls", "configs/pls_baseline.yaml", None),
        ]
        for out_name, config_path, gens in baseline_configs:
            out_dir = root / "results" / inst_name / out_name
            if not args.no_resume and _already_done(out_dir, n_runs):
                print(f"  [{inst_name}] {out_name} SKIPPED (already have {n_runs}+ runs)", flush=True)
                continue
            t0 = time.time()
            print(f"  [{inst_name}] {out_name} ...", flush=True)
            config = ExperimentConfig.from_yaml(root / config_path)
            config.n_runs = n_runs
            if gens is not None:
                config.n_generations = gens
            run_baseline_experiment(
                config=config,
                instance_path=instance_path,
                output_dir=out_dir,
                verbose=False,
            )
            print(f"  [{inst_name}] {out_name} done in {time.time()-t0:.0f}s", flush=True)

        # --- 2x2 factorial DoE: decoder (greedy/repair) x local search (off/on) ---
        # All four cells are compared at the SAME total evaluation budget
        # (pop_size * n_generations). Memetic cells split that budget between
        # the EA phase and local search rather than adding LS on top -- brief
        # 5.1 requires the termination criterion to be fixed across compared
        # configurations.
        doe_cells = [
            ("doe_repair_noLS", "repair", None),
            ("doe_greedy_LS", "greedy", "conflict_graph"),
            ("doe_repair_LS", "repair", "conflict_graph"),
        ]
        for out_name, decoder_name, ls_name in doe_cells:
            out_dir = root / "results" / inst_name / out_name
            if not args.no_resume and _already_done(out_dir, n_runs):
                print(f"  [{inst_name}] {out_name} SKIPPED (already have {n_runs}+ runs)", flush=True)
                continue
            t0 = time.time()
            print(f"  [{inst_name}] {out_name} ...", flush=True)
            config = ExperimentConfig(
                name=out_name, algorithm="nsga2",
                n_runs=n_runs, n_generations=n_gen,
            )
            run_custom_experiment(
                config=config,
                instance_path=instance_path,
                output_dir=out_dir,
                decoder_name=decoder_name,
                local_search_name=ls_name,
                ls_budget=args.ls_budget,          # None => budget-matched
                ls_share=args.ls_share,
                conflict_mode=args.conflict_mode,
                verbose=False,
            )
            print(f"  [{inst_name}] {out_name} done in {time.time()-t0:.0f}s", flush=True)

        # doe_greedy_noLS is the same configuration as baseline_nsga2; record
        # the alias so the analysis scripts can find it without recomputation.
        alias = root / "results" / inst_name / "doe_greedy_noLS_ALIAS.json"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(json.dumps({
            "note": "DoE cell (greedy decoder, no local search) == baseline_nsga2; "
                    "identical configuration and seeds, not recomputed.",
            "use_results_from": f"results/{inst_name}/baseline_nsga2/",
        }, indent=2))

        print(f"========== [{inst_name}] complete ==========", flush=True)

    print(f"\nALL DONE in {time.time()-total_start:.0f}s. Results under results/<instance>/<config>/.")


if __name__ == "__main__":
    main()
