# Price of Fairness in Operating Theatre Scheduling

Code, experiments and results for the paper *Price of Fairness in Integrated Operating Theatre Scheduling: Problem-Tailored Multi-Objective Evolutionary Optimisation under Downstream Bed Constraints* by Sameena Siddiqua and Imran Medi, under review at *Engineering Applications of Artificial Intelligence*.

## The problem

Surgeries are scheduled into operating theatres over several weeks. Each patient then needs ICU and ward beds for several days, and daily bed capacity is a hard constraint. Three objectives are optimised together:

1. priority-weighted throughput
2. resource imbalance
3. fairness: the variance of mean waiting time across five patient priority classes

## Key findings

- **Price of fairness:** equalising waiting time across the five priority classes costs 130 to 5,866 units of priority-weighted throughput, with coefficients of variation of 0.11 to 0.14. The cost is stable and grows with instance size.
- **Constraint handling:** a repair-based constraint handler degrades the competing objective 74-fold. A crossed factorial design traces this to its eviction rule, which builds a fixed objective preference into decoding below the level where selection can correct it.
- **Local search:** a conflict-graph local search improves no indicator, because a static graph misrepresents contention that depends on the schedule.
- **Generalisation:** the mechanism reproduces on a withheld instance and on PSPLIB multi-mode project scheduling benchmarks.

All comparisons use 30 seeded runs per configuration on three instances, a fixed budget of 10,000 evaluations, and Friedman tests with Holm-corrected pairwise tests and Vargha-Delaney effect sizes.

## Repository layout

```
or-scheduling-price-of-fairness/
├── my_components/          # my work
│   ├── decoder.py          #   repair-based constraint handler
│   ├── local_search.py     #   conflict-graph local search
│   ├── run_everything.py   #   single entry point that reproduces every result
│   ├── analyse.py          #   indicators, Friedman and Holm tests, effect sizes
│   ├── external/           #   PSPLIB multi-mode benchmark parser and runner
│   └── tests/              #   32 tests for these components
├── cio/  scripts/  configs/  tests/  instances/  docs/
│                           # course framework supplied by Asia Pacific University (unmodified)
├── external_benchmark/     # PSPLIB instances
├── results/  results_external/   # per-run logs
├── analysis/               # indicator tables, statistical tests and figures
└── solutions/              # exported Pareto fronts
```

The framework is built on pymoo and provides the problem, the NSGA-II, MOEA/D and Pareto Local Search baselines, the experiment harness and the indicators. Its original README is in [`docs/SKELETON_README.md`](docs/SKELETON_README.md).

## Reproduce

```bash
pip install -r requirements.txt            # requirements-frozen.txt pins exact versions
python my_components/run_everything.py     # every table and figure in the paper
python -m pytest                           # framework and component tests
```

## Citation

```bibtex
@unpublished{siddiqua2026priceoffairness,
  author = {Siddiqua, Sameena and Medi, Imran},
  title  = {Price of Fairness in Integrated Operating Theatre Scheduling: Problem-Tailored
            Multi-Objective Evolutionary Optimisation under Downstream Bed Constraints},
  note   = {Under review at Engineering Applications of Artificial Intelligence},
  year   = {2026}
}
```
