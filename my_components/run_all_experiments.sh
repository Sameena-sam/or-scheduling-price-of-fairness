#!/usr/bin/env bash
# Runs the three required baselines + your 2x2 factorial DoE, on all three
# instances, 30 runs each, and writes standard JSON logs to results/.
#
# Usage: run from the cio_skeleton/ root:
#   bash my_components/run_all_experiments.sh 2>&1 | tee results/run_all.log
#
# Recommended: run this in the background so it survives you closing the
# terminal, and check progress with `tail -f results/run_all.log`:
#   nohup bash my_components/run_all_experiments.sh > results/run_all.log 2>&1 &
#
# Tune N_GENERATIONS / N_RUNS down first with a timed dry run on medium/large
# before committing to the full 30-run batch -- decode cost scales with
# instance size, so a run that takes 3s on small can take much longer on
# large. Time one run manually first:
#   time PYTHONPATH=. python my_components/run_custom.py --algorithm nsga2 \
#     --decoder repair --instance instances/ot_large.json \
#     --out /tmp/timing_test/ --n-runs 1 --n-generations 100

set -euo pipefail
export PYTHONPATH=.

N_RUNS=30
N_GEN=100
LS_BUDGET=200

INSTANCES=("instances/ot_small.json" "instances/ot_medium.json" "instances/ot_large.json")
NAMES=("small" "medium" "large")

for i in "${!INSTANCES[@]}"; do
  INST="${INSTANCES[$i]}"
  NAME="${NAMES[$i]}"

  echo "=== [$NAME] Required baselines ==="
  python scripts/run.py --config configs/nsga2_baseline.yaml \
    --instance "$INST" --out "results/${NAME}/baseline_nsga2/" \
    --n-runs "$N_RUNS" --n-generations "$N_GEN"

  python scripts/run.py --config configs/moead_baseline.yaml \
    --instance "$INST" --out "results/${NAME}/baseline_moead/" \
    --n-runs "$N_RUNS" --n-generations "$N_GEN"

  python scripts/run.py --config configs/pls_baseline.yaml \
    --instance "$INST" --out "results/${NAME}/baseline_pls/" \
    --n-runs "$N_RUNS"

  echo "=== [$NAME] 2x2 factorial DoE (decoder x local search) ==="
  # Cell 1: greedy decoder, no LS  (== baseline_nsga2 above, kept separate
  # here for DoE bookkeeping clarity; skip if you want to just reuse
  # baseline_nsga2's logs for this cell)
  python my_components/run_custom.py --algorithm nsga2 \
    --decoder greedy --instance "$INST" \
    --out "results/${NAME}/doe_greedy_noLS/" \
    --n-runs "$N_RUNS" --n-generations "$N_GEN"

  # Cell 2: repair decoder, no LS (isolates the constraint-handling component)
  python my_components/run_custom.py --algorithm nsga2 \
    --decoder repair --instance "$INST" \
    --out "results/${NAME}/doe_repair_noLS/" \
    --n-runs "$N_RUNS" --n-generations "$N_GEN"

  # Cell 3: greedy decoder, with LS (isolates the local-search component)
  python my_components/run_custom.py --algorithm nsga2 \
    --decoder greedy --local-search conflict_graph --ls-budget "$LS_BUDGET" \
    --instance "$INST" \
    --out "results/${NAME}/doe_greedy_LS/" \
    --n-runs "$N_RUNS" --n-generations "$N_GEN"

  # Cell 4: repair decoder + LS (full configuration -- both components on)
  python my_components/run_custom.py --algorithm nsga2 \
    --decoder repair --local-search conflict_graph --ls-budget "$LS_BUDGET" \
    --instance "$INST" \
    --out "results/${NAME}/doe_repair_LS/" \
    --n-runs "$N_RUNS" --n-generations "$N_GEN"

  echo "=== [$NAME] done ==="
done

echo "All experiments complete. Results under results/<instance>/<config>/."
