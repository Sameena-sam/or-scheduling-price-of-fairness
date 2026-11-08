# CIO Coursework Skeleton — Operating Theatre Scheduling

This is the framework you will work in for the CT099-3-M Computational
Intelligence and Optimisation individual assignment. It provides:

- A multi-objective constrained scheduling problem (OT scheduling with
  downstream ICU/ward constraints over a multi-week horizon).
- A pluggable algorithm framework built on **pymoo**, with three baseline
  configurations (NSGA-II, MOEA/D, Pareto Local Search) already wired up.
- Named hooks for the components you are expected to design: initialiser,
  crossover, mutation, local search, decoder, and constraint handler.
- A reproducible experiment harness with YAML configs, CLI overrides,
  seeded runs, per-run logging, and statistical tooling (HV, IGD+, spread,
  Friedman, Holm/Nemenyi, Vargha-Delaney).
- Three public benchmark instances (small / medium / large) and reference
  Pareto fronts for IGD+ computation. A hidden instance is released 48h
  before submission for the final benchmark.

You **must** use this framework. You may not bypass it, rewrite it, or
implement your algorithm in a separate codebase. See the brief for the
component design requirements and the marking rubric.

---

## Quick Install

Tested on Python 3.10+.

```bash
pip install -r requirements.txt
```

If you're using a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Sanity Check

Verify the install works by running NSGA-II for 5 runs on the small
instance with 20 generations:

```bash
python scripts/run.py \
  --config configs/nsga2_baseline.yaml \
  --instance instances/ot_small.json \
  --out results/nsga2_small_quick/ \
  --n-runs 5 --n-generations 20
```

This should complete in under a minute and produce per-run JSON logs in
`results/nsga2_small_quick/`.

## Running the Baselines (Full)

The baseline configurations match what the marker uses to compare your
work. Each baseline takes the default 30 runs as required by the
experimental protocol.

```bash
# NSGA-II baseline
python scripts/run.py --config configs/nsga2_baseline.yaml \
                      --instance instances/ot_medium.json \
                      --out results/nsga2_baseline_medium/

# MOEA/D baseline
python scripts/run.py --config configs/moead_baseline.yaml \
                      --instance instances/ot_medium.json \
                      --out results/moead_baseline_medium/

# Pareto Local Search baseline
python scripts/run.py --config configs/pls_baseline.yaml \
                      --instance instances/ot_medium.json \
                      --out results/pls_baseline_medium/
```

You can override config values from the command line:

```bash
python scripts/run.py --config configs/nsga2_baseline.yaml \
                      --instance instances/ot_small.json \
                      --out results/nsga2_pop200/ \
                      --pop-size 200 --crossover ox --mutation insertion
```

## Where to Plug In Your Components

You are expected to design at least two novel algorithmic components, at
least one of which is problem-tailored to OT scheduling. The named hooks
where you can plug new components in are:

| Component             | Base class            | Module                              |
|-----------------------|-----------------------|-------------------------------------|
| Initialiser           | `BaseInitialiser`     | `cio/operators/operators.py`        |
| Crossover             | `BaseCrossover`       | `cio/operators/operators.py`        |
| Mutation              | `BaseMutation`        | `cio/operators/operators.py`        |
| Local search          | `BaseLocalSearch`     | `cio/operators/local_search.py`     |
| Decoder               | `BaseDecoder`         | `cio/problem/decoder.py`            |

To plug in a new component:

1. Write your component as a class with the same `__call__` (or `sample`)
   signature as the base classes above. Place it in your own module,
   for example `my_components/`.
2. Build an algorithm with your component:

```python
from cio.algorithms import build_nsga2
from cio.problem import Instance, OTSchedulingProblem
from my_components import MyTailoredCrossover, MyMemeticLocalSearch
from pymoo.optimize import minimize

instance = Instance.from_json("instances/ot_medium.json")
problem = OTSchedulingProblem(instance)

algo = build_nsga2(
    pop_size=100,
    crossover=MyTailoredCrossover(),
    mutation=MyTailoredMutation(),
    seed=0,
)
res = minimize(problem, algo, ("n_gen", 100), seed=0)
print(res.F)   # Pareto-front objective values
```

For a memetic configuration, run the algorithm to convergence then apply
your local search to each member of the final population.

For a custom decoder, subclass `cio.problem.decoder.BaseDecoder` and pass
it to `OTSchedulingProblem(instance, decoder=MyDecoder())`.

## Computing Indicators and Statistical Tests

```python
import numpy as np, json
from cio.analysis import hypervolume, igd_plus, friedman, vargha_delaney

# Load the reference front for IGD+
with open("instances/ot_medium_reference_front.json") as f:
    ref = np.array(json.load(f)["points"])

# Compute indicators per run
hv = hypervolume(F, ref_point=[10000, 5000, 50])
igd = igd_plus(F, ref)

# Compare two configurations across 30 runs
data = np.column_stack([hv_config_a, hv_config_b, hv_config_c])  # 30 x 3
print(friedman(data))
print(vargha_delaney(hv_config_a, hv_config_b))
```

## Project Layout

```
cio_skeleton/
├── cio/                       # Importable Python package
│   ├── problem/               # Instance, schedule, decoder, evaluator
│   ├── algorithms/            # NSGA-II, MOEA/D, PLS builders
│   ├── operators/             # Crossover, mutation, local-search hooks
│   ├── harness/               # Experiment runner
│   └── analysis/              # Indicators, statistical tests
├── configs/                   # YAML configs for baselines
├── instances/                 # Public benchmark instances
│   ├── ot_small.json
│   ├── ot_medium.json
│   ├── ot_large.json
│   ├── ot_*_reference_front.json
│   └── private/               # Hidden instance (NOT released to students)
├── scripts/                   # CLI tools
│   ├── run.py                 # Run an experiment
│   ├── generate_instances.py
│   └── generate_reference_fronts.py
├── docs/                      # Reference docs
│   ├── INSTANCE_FORMAT.md
│   └── STARTER_GUIDE.md
└── tests/                     # Unit tests
```

## Note on Reference Fronts

The reference fronts shipped with the skeleton (`instances/ot_*_reference_front.json`)
are tagged `"budget_profile": "preliminary"`. These are good enough for
development and self-comparison. For final grading, your lecturer will
regenerate them with much larger budgets:

```bash
python scripts/generate_reference_fronts.py --instances instances/ --generous
```

(This is documented for transparency; you do not need to do this yourself.)

## Tests

Run the test suite to verify your environment:

```bash
PYTHONPATH=. pytest tests/
```

## Further Reading

- The full assessment brief (separately distributed) for the problem
  specification, marking rubric, and submission requirements.
- `docs/STARTER_GUIDE.md` — a step-by-step walkthrough of your first
  experiment.
- `docs/INSTANCE_FORMAT.md` — full reference for the instance JSON format.
- pymoo documentation: https://pymoo.org

## Important Notes

- You may not modify files in `cio/problem/` (the problem definition is
  fixed). You may add new files alongside it.
- You may add new modules in your own directories, but the import path
  `from cio.problem import ...` must continue to work for the marker.
- The hidden benchmark instance is distributed 48 hours before submission.
- A 10-minute live viva is part of the assessment; see the brief.

---

If something doesn't work, post in the module forum or attend the weekly
lab session.
