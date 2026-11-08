"""Single entry point: runs the complete submission pipeline in order.

    python my_components/run_everything.py

Stages, in dependency order:

  0. verify the environment (skeleton tests + component tests)
  1. OT experiments   -- 3 baselines + 2^2 factorial DoE, 3 instances, 30 runs
  2. crossed factorial -- eviction rule x reinsertion policy, medium, 30 runs
  3. external benchmark -- MMRCPSP, published optima comparison
  4. analysis          -- indicators, statistics, DoE effects, price of fairness
  5. factorial analysis -- two-way effect decomposition
  6. solution outputs  -- brief 8.1 schedules + feasibility certificates
  7. summary           -- what was produced and where

Every stage is resumable: completed configurations are skipped, so re-running
after an interruption picks up where it stopped. Use --force to recompute.

Useful flags:
    --skip-experiments   analysis only, when the runs already exist
    --quick              tiny budgets end to end, for verifying the pipeline
    --no-external        skip the MMRCPSP benchmark
    --external-limit N   number of benchmark instances (default 10)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, args: list[str], optional: bool = False) -> bool:
    print(f"\n{'='*72}\n>> {label}\n{'='*72}", flush=True)
    t0 = time.time()
    env_cmd = [sys.executable] + args
    result = subprocess.run(env_cmd, cwd=ROOT)
    ok = result.returncode == 0
    status = "done" if ok else ("skipped (optional)" if optional else "FAILED")
    print(f"-- {label}: {status} in {time.time()-t0:.0f}s", flush=True)
    if not ok and not optional:
        print(f"\nStopping: '{label}' failed. Fix the error above and re-run; "
              f"completed stages will be skipped.", flush=True)
        sys.exit(1)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-experiments", action="store_true",
                    help="re-run analysis only, using existing run logs")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budgets end to end, to verify the pipeline works")
    ap.add_argument("--force", action="store_true",
                    help="recompute configurations that already have results")
    ap.add_argument("--external-limit", type=int, default=10)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--n-generations", type=int, default=100)
    args = ap.parse_args()

    n_runs = 3 if args.quick else args.n_runs
    n_gen = 15 if args.quick else args.n_generations
    ext_limit = 1 if args.quick else args.external_limit
    total = time.time()

    print(f"CIO submission pipeline")
    print(f"  runs per configuration : {n_runs}")
    print(f"  generations per run    : {n_gen}")
    print(f"  benchmark instances    : {'skipped' if args.no_external else ext_limit}")
    print(f"  mode                   : {'QUICK (verification only)' if args.quick else 'FULL'}")

    # --- 0. environment -----------------------------------------------------
    if not args.skip_tests:
        run("0a. Skeleton tests (must be untouched)",
            ["-m", "pytest", "tests/", "-q"])
        run("0b. Component tests",
            ["-m", "pytest", "my_components/tests/", "-q"])

    # --- 1-3. experiments ---------------------------------------------------
    if not args.skip_experiments:
        cmd = ["my_components/run_all_experiments.py",
               "--n-runs", str(n_runs), "--n-generations", str(n_gen)]
        if args.quick:
            cmd += ["--instances", "ot_small.json"]
        if args.force:
            cmd.append("--no-resume")
        run("1. OT experiments (baselines + 2^2 factorial DoE)", cmd)

        fact_out = ROOT / "results" / "factorial_medium"
        if args.force or not (fact_out / "random_earliest" / "summary.json").exists():
            run("2. Crossed factorial (eviction rule x reinsertion policy)",
                ["my_components/factorial_ablation.py",
                 "--instance", "instances/ot_small.json" if args.quick else "instances/ot_medium.json",
                 "--out", "results/factorial_medium/",
                 "--n-runs", str(n_runs), "--n-generations", str(n_gen)])
        else:
            print("\n>> 2. Crossed factorial: already complete, skipping "
                  "(use --force to recompute)")

        if not args.no_external:
            inst_dir = ROOT / "external_benchmark" / "instances"
            sol_file = ROOT / "external_benchmark" / "j10opt.mm"
            if not inst_dir.exists() or not any(inst_dir.glob("*.mm")):
                print(f"\n>> 3. External benchmark: no instances found in "
                      f"{inst_dir.relative_to(ROOT)}, skipping.")
            else:
                cmd = ["my_components/external/run_external.py",
                       "--instances", "external_benchmark/instances/",
                       "--limit", str(ext_limit),
                       "--n-runs", str(n_runs), "--n-generations", str(n_gen)]
                if sol_file.exists():
                    cmd += ["--solutions-file", "external_benchmark/j10opt.mm"]
                run("3. External benchmark (MMRCPSP)", cmd)

    # --- 4-6. analysis ------------------------------------------------------
    instances = ["small"] if args.quick else ["small", "medium", "large"]
    run("4. Analysis: indicators, statistics, DoE, price of fairness",
        ["my_components/analyse.py", "--instances"] + instances)

    if (ROOT / "results" / "factorial_medium").exists():
        ref = ("instances/ot_small_reference_front.json" if args.quick
               else "instances/ot_medium_reference_front.json")
        run("5. Factorial analysis: two-way effect decomposition",
            ["my_components/analyse_factorial.py",
             "--results", "results/factorial_medium/",
             "--reference-front", ref], optional=True)

    run("6. Solution outputs (brief 8.1: schedules + feasibility certificates)",
        ["my_components/export_solutions.py", "--instances"] + instances)

    # --- 7. summary ---------------------------------------------------------
    print(f"\n{'='*72}\n>> PIPELINE COMPLETE in {time.time()-total:.0f}s\n{'='*72}")
    for folder, what in [
        ("results", "OT run logs (one JSON per run, per configuration)"),
        ("results_external", "benchmark logs, statistics, best-known comparison"),
        ("analysis", "indicator tables, Friedman/Holm/effect sizes, DoE, figures"),
        ("solutions", "schedules, objectives and feasibility certificates"),
    ]:
        p = ROOT / folder
        if p.exists():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {folder + '/':20s} {n:5d} files   {what}")
    key = [
        "analysis/indicators_summary.csv",
        "analysis/effect_sizes.csv",
        "analysis/doe_effects.csv",
        "analysis/price_of_fairness.csv",
        "analysis/friedman.txt",
        "results/factorial_medium/analysis/two_way_effects.csv",
        "results_external/best_known_comparison.csv",
        "solutions/large/certificate_summary.txt",
    ]
    print("\n  Key files for the paper:")
    for k in key:
        print(f"    {'OK ' if (ROOT / k).exists() else '-- '} {k}")


if __name__ == "__main__":
    main()
