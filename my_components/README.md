# Designed components and reproducibility package

CT099-3-M Computational Intelligence and Optimisation — individual assignment
Author: Sameena Siddiqua

This directory contains the two designed components, the experiment driver, and
the instructions needed to reproduce every table and figure in the paper.

---

## 1. Contents

| File | What it is |
|---|---|
| `decoder.py` | **Designed component 1** — `RepairBasedDecoder`, a repair-based constraint-handling mechanism (problem-tailored) |
| `local_search.py` | **Designed component 2** — `ConflictGraphLocalSearch`, a Pareto local search restricted to a conflict-graph neighbourhood |
| `run_custom.py` | Experiment runner that plugs the above into the skeleton harness (the shipped `scripts/run.py` has no hook for a custom decoder or local search) |
| `run_all_experiments.py` | Cross-platform driver: 3 required baselines + the 2×2 factorial DoE, all instances |
| `factorial_ablation.py` | Confirmatory crossed factorial (3 eviction rules × 2 reinsertion policies) isolating the mechanisms inside the repair decoder |
| `analyse.py` | Indicators (HV/IGD+/spread), Friedman, Holm post-hoc, Vargha-Delaney, DoE effects, price-of-fairness, figures |
| `analyse_factorial.py` | Two-way effect decomposition and statistics for the crossed factorial |
| `export_solutions.py` | Brief §8.1 deliverables: schedules, objectives and feasibility certificates for the pooled non-dominated set |
| `external/mmrcpsp.py` | PSPLIB parser, project-scheduling decoders and objectives for the external benchmark |
| `external/run_external.py` | External benchmark runner, statistics and best-known comparison |
| `../tests/test_my_components.py` | 32 correctness tests for the two components |
| `../tests/test_external.py` | 26 tests for the external-benchmark parser and decoders |

The problem definition in `cio/problem/` is unmodified, as required.

---

## 2. Install

Python 3.10+.

```bash
pip install -r requirements.txt
```

Then pin the exact versions actually used, which is what brief §8.3 requires:

```bash
pip freeze > requirements-frozen.txt
```

Submit `requirements-frozen.txt` alongside `requirements.txt`.

---

## 3. Verify the install

```bash
PYTHONPATH=. pytest tests/ -q
```

Expected: **107 passed** (49 skeleton + 32 component + 26 external-benchmark tests).

---

## 4. Reproduce the experiments

Smoke test first (~30 s):

```bash
python my_components/run_all_experiments.py --quick
```

Full protocol — 30 runs per configuration per instance:

```bash
python my_components/run_all_experiments.py
```

Results are written to `results/<instance>/<config>/run_XXX_seed_YYY.json`,
one JSON per run, plus a `summary.json` per configuration.

Safe to interrupt with Ctrl+C and rerun: any configuration whose `summary.json`
already records at least the requested number of runs is skipped. Use
`--no-resume` to force a clean rerun.

Reduce the large-instance budget if compute time is short (**and disclose the
reduction in the paper**):

```bash
python my_components/run_all_experiments.py --large-n-runs 10 --large-n-generations 60
```

Run a single configuration directly:

```bash
python my_components/run_custom.py \
    --algorithm nsga2 --decoder repair --local-search conflict_graph \
    --instance instances/ot_medium.json \
    --out results/medium/doe_repair_LS/ \
    --n-runs 30 --n-generations 100
```

---

## 4b. Analysis and required outputs

Compute every indicator table, statistical test and figure reported in the paper:

```bash
python my_components/analyse.py --instances small medium large
```

Writes to `analysis/`: `indicators_raw.csv`, `indicators_summary.csv`,
`friedman.txt`, `posthoc_<instance>_<indicator>.csv`, `effect_sizes.csv`,
`doe_effects.csv`, `price_of_fairness.csv`, `reference_points.txt`, and
`figures/`.

Run and analyse the confirmatory crossed factorial:

```bash
python my_components/factorial_ablation.py --instance instances/ot_medium.json \
    --out results/factorial_medium/ --n-runs 30 --n-generations 100
python my_components/analyse_factorial.py --results results/factorial_medium/ \
    --reference-front instances/ot_medium_reference_front.json
```

Pass `--reference-front` or IGD+ is reported as missing.

Produce the solution outputs required by §8.1 of the brief — schedules,
objective values and feasibility certificates for every solution in the pooled
non-dominated set:

```bash
python my_components/export_solutions.py --instances small medium large
```

Writes to `solutions/<instance>/`: `pareto_front.json` (schedule, objectives
and certificate per solution), `schedules.csv` (flat surgery-level view),
`indicators_summary.csv`, and `certificate_summary.txt`. Certificates are
recomputed from the assignments rather than read from the decoder's own
bookkeeping, so they are independent of the code that produced the schedule.

### Note on the supplied statistics utility

`cio.analysis.holm_pairwise` passes the (runs × configurations) matrix directly
to `scikit_posthocs.posthoc_wilcoxon`, which interprets a bare 2-D array as one
group per row. On a 30 × 6 input it returns a 30 × 30 matrix of unit p-values
regardless of the data. `analyse.py` therefore calls scikit-posthocs directly
with each configuration passed as a separate sample
(`holm_pairwise_fixed`), verified against synthetic data with known
separation. `friedman` and `nemenyi_pairwise` are unaffected.

---

## 4c. External benchmark (brief §3.3 / §5.4)

Validation is performed on MMRCPSP from PSPLIB. That benchmark is chosen
because its renewable resources are cumulative over time in the same way ICU
and ward beds are — an activity occupies capacity for its whole duration, so a
placement decision constrains later periods. That is precisely the structural
property the repair decoder was designed to exploit, which makes it a genuine
transferability test rather than an unrelated second problem.

**Step 1 — obtain data.** From the PSPLIB site, MRCPSP section, download the
**j10 ZIP** (instance files) and the **j10 OPT** file (published optima).
Extract the instances to `external_benchmark/instances/` and put the optima
file at `external_benchmark/j10opt.mm`.

The j10 archive contains 536 files, not 640: PSPLIB omits the 104 instances
that have no feasible mode assignment. Those appear in the optima file with a
makespan of 16384, which `parse_psplib_solutions` treats as a sentinel and
skips, so the 536 files and 536 published optima correspond exactly.

**Step 2 — verify the pipeline** on the bundled synthetic instance before
committing to a long run:

```bash
python my_components/external/run_external.py --synthetic --n-runs 5 --n-generations 20
```

**Step 3 — run at full protocol**, with no re-tuning, as §5.4 requires:

```bash
python my_components/external/run_external.py \
    --instances external_benchmark/instances/ --limit 10 \
    --solutions-file external_benchmark/j10opt.mm \
    --n-runs 30 --n-generations 100
```

`--solutions-file` loads every published optimum automatically and reports the
gap per instance plus aggregate statistics (fraction of optima reached, mean,
median and worst gap). `--best-known name=value` still works for one-off
overrides. Allow roughly 12 minutes per instance for all three decoders.

Results are written to `results_external/`: per-run JSON logs,
`statistics.txt` per instance (Friedman, Holm-corrected pairwise,
Vargha-Delaney), `external_summary.csv`, `best_known_comparison.csv` and
`best_known_summary.txt`.

Three decoders are compared: `greedy` (feasibility by construction, analogous
to the OT baseline), `repair` (allow-then-repair with shortest-activity
displacement, analogous to the submitted OT decoder), and `repair_perm`
(genotype-dependent displacement, the ablation variant). Objectives are
makespan and renewable-resource levelling.

**Mode assignment.** Nonrenewable feasibility is a subproblem in its own right:
taking each activity's shortest mode exhausts the budget on roughly a third of
real j10 instances. A repair pass therefore applies the mode switch offering
the largest reduction in overconsumption per unit of added duration until the
assignment is feasible. This runs before scheduling and is deterministic, so
the permutation remains the only source of variation between individuals.

Every returned solution is re-decoded and checked for feasibility, and the
feasible fraction is reported per configuration.

---

## 5. Evaluation budget (important)

Brief §5.1 requires the termination criterion to be **fixed across
configurations being compared**. All four DoE cells therefore run at the same
total budget of `pop_size × n_generations` = 10,000 decoder evaluations.

Memetic cells **split** that budget rather than adding local search on top:
with the default `--ls-share 0.5`, the EA phase gets 5,000 evaluations
(50 generations at population 100) and the remaining 5,000 are divided evenly
across the individuals passed to local search.

Every run log records the split explicitly — `ea_evaluations`,
`ls_evaluations`, `evaluations` (their sum), `eval_budget`, and
`n_generations_effective` — so the budget match is auditable rather than
asserted. The local-search evaluation count is **measured** (via
`ConflictGraphLocalSearch.evals_used`), not estimated.

Passing `--ls-budget N` explicitly overrides this and gives local search `N`
evaluations per individual *on top of* the EA budget. That produces an
unmatched comparison; only do it deliberately, and say so in the paper.

---

## 6. DoE cells

| Cell | Decoder | Local search | Results directory |
|---|---|---|---|
| greedy, no LS | `GreedyDecoder` (baseline) | off | `results/<inst>/baseline_nsga2/` |
| repair, no LS | `RepairBasedDecoder` | off | `results/<inst>/doe_repair_noLS/` |
| greedy, LS | `GreedyDecoder` (baseline) | `ConflictGraphLocalSearch` | `results/<inst>/doe_greedy_LS/` |
| repair, LS | `RepairBasedDecoder` | `ConflictGraphLocalSearch` | `results/<inst>/doe_repair_LS/` |

The first cell is the same configuration as the required NSGA-II baseline —
same operators, same seeds, same output — so it is run once and reused rather
than recomputed. `doe_greedy_noLS_ALIAS.json` records this pointer.

---

## 7. Conflict-graph density

The local search restricts its swap neighbourhood to conflict-graph edges. Two
conflict definitions are implemented, and the restriction is only meaningful if
the graph is actually sparse — so measure it rather than assuming it:

```bash
PYTHONPATH=. python -c "
from cio.problem import Instance
from my_components.local_search import conflict_graph_density
for f in ['ot_small','ot_medium','ot_large']:
    inst = Instance.from_json(f'instances/{f}.json')
    for mode in ['ot','strict']:
        print(conflict_graph_density(inst, mode=mode))
"
```

Measured on the shipped instances:

| Instance | `ot` mode | `strict` mode |
|---|---|---|
| small | 100.0% | 86.6% |
| medium | 100.0% | 61.7% |
| large | 66.0% | 16.8% |

`ot` (shared eligible OT) is **completely dense on small and medium** — under
that definition the "restriction" is vacuous and the neighbourhood is identical
to an unrestricted random-pair search. `strict` additionally requires a shared
eligible team and overlapping admissible day-windows, which is the theatre–day–team
slot-conflict relation the method actually claims to exploit. `strict` is the
default; report both numbers in the paper.

---

## 8. Component summaries

### `RepairBasedDecoder` (constraint handling, problem-tailored)

Two phases. **Placement** processes the permutation in order and places each
surgery at the earliest day satisfying OT hours, team hours, the team
consecutive-day limit, specialty and equipment compatibility, and
`earliest_day` — but *not* ICU/ward capacity. **Repair** then scans for days
over ICU or ward capacity and, for each violation, evicts the
**lowest-priority** surgery occupying that day and reinserts it at the earliest
strictly-feasible later day, with all constraints including bed capacity
re-checked. A surgery becomes unscheduled only if no feasible day remains
before the horizon ends. Start times are re-packed afterwards so the emitted
schedule has no gaps or overlaps.

This differs from the shipped `GreedyDecoder`, which enforces bed capacity
during placement and therefore can only avoid overflow by refusing to schedule
a surgery at all — it never reshuffles the schedule to make room.

If the repair budget is exhausted with overflow still present, the decoder
raises rather than returning an infeasible schedule (`strict=True` by default).
Diagnostics after each decode: `last_repair_passes`, `last_evictions`,
`last_evictions_unplaceable`.

### `ConflictGraphLocalSearch` (local search)

Pareto local search over swap moves, with the candidate pairs restricted to
conflict-graph edges instead of all `C(n,2)` pairs. Accepts a move only on
strict Pareto dominance (first-improvement). Reports `evals_used` after each
call for honest budget accounting. Pass the same decoder the surrounding
configuration uses so local and global search share constraint-handling
behaviour.

---

## 9. Reproducing a specific result

Every run log is named `run_<index>_seed_<seed>.json` and contains the seed,
the final objective values (`F`), the decision variables (`X`), the evaluation
breakdown, and the runtime. Seeds are derived deterministically from
`--base-seed` (default 0) via `numpy.random.default_rng`, so rerunning any
configuration with the same base seed reproduces the same runs exactly.
