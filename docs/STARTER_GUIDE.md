# Starter Guide — Your First Experiment

This guide walks you through running, analysing, and extending a baseline
algorithm. Read this once before starting your design work.

## Step 1 — Install and Verify

```bash
git clone <repo-url> cio_skeleton
cd cio_skeleton
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/
```

If all tests pass, you're ready.

## Step 2 — Look at an Instance

Open `instances/ot_small.json` in any editor. You'll see a single JSON
object with `ots`, `teams`, `surgery_types`, and `surgeries` arrays. This
is the small benchmark — 22 surgeries over 7 days, 2 OTs, 3 teams.

Load it interactively:

```bash
PYTHONPATH=. python
```

```python
from cio.problem import Instance
inst = Instance.from_json("instances/ot_small.json")
print(inst)
print(inst.surgeries[0])
print(inst.ots[0])
```

## Step 3 — Decode a Random Permutation

The encoding is a permutation of surgery indices. The decoder processes
the permutation in order, greedily slotting each surgery into the first
feasible `(day, OT, team)` it finds.

```python
import numpy as np
from cio.problem import Instance, OTSchedulingProblem, GreedyDecoder

inst = Instance.from_json("instances/ot_small.json")
problem = OTSchedulingProblem(inst)

rng = np.random.default_rng(0)
perm = rng.permutation(inst.n_surgeries)

decoder = GreedyDecoder()
sched = decoder(inst, perm)
print(f"scheduled: {sched.n_scheduled} of {inst.n_surgeries}")
print(f"unscheduled IDs: {sched.unscheduled_surgery_ids}")
print(f"first assignment: {sched.assignments[0]}")
```

## Step 4 — Evaluate

```python
from cio.problem.evaluator import evaluate

f1, f2, f3 = evaluate(inst, sched)
print(f"f1 = {f1:.1f}   f2 = {f2:.1f}   f3 = {f3:.3f}")
```

Remember: `f1` is **negated** internally, so a more-negative `f1` means
higher throughput.

## Step 5 — Run NSGA-II on Small

```bash
python scripts/run.py \
  --config configs/nsga2_baseline.yaml \
  --instance instances/ot_small.json \
  --out results/nsga2_small/ \
  --n-runs 5 --n-generations 50
```

This produces `results/nsga2_small/run_001_seed_*.json` for each run, plus
a `summary.json` describing the configuration.

Inspect one:

```python
import json
with open("results/nsga2_small/run_001_seed_*.json") as f:
    run = json.load(f)
print(f"final pop size: {len(run['F'])}")
print(f"f1 best: {min(row[0] for row in run['F']):.1f}")
print(f"runtime: {run['runtime_seconds']:.1f}s")
```

## Step 6 — Compute Indicators

```python
import numpy as np, json, glob
from cio.analysis import hypervolume, igd_plus, non_dominated

# Collect final fronts from 5 runs
fronts = []
for path in sorted(glob.glob("results/nsga2_small/run_*.json")):
    fronts.append(np.array(json.load(open(path))["F"]))

# Reference front for IGD+
ref = np.array(json.load(open("instances/ot_small_reference_front.json"))["points"])

# Reference point for HV — pick coordinates that dominate every observed point
ref_point = [500, 500, 50]   # tune by inspecting your data

for i, F in enumerate(fronts):
    hv = hypervolume(F, ref_point=ref_point)
    igd = igd_plus(F, ref)
    print(f"run {i+1}: HV = {hv:.3f}   IGD+ = {igd:.3f}")
```

## Step 7 — Statistical Comparison

You'll usually compare two or more configurations across 30 runs. Suppose
you have HV values for three configurations in `hv_a`, `hv_b`, `hv_c`
(each a length-30 array):

```python
import numpy as np
from cio.analysis import friedman, holm_pairwise, vargha_delaney, effect_size_magnitude

data = np.column_stack([hv_a, hv_b, hv_c])

# Omnibus test
result = friedman(data)
print(f"Friedman statistic: {result['statistic']:.3f}   p = {result['p_value']:.4f}")

# Pairwise post-hoc
print(holm_pairwise(data))

# Effect size between a and b
A = vargha_delaney(hv_a, hv_b)
print(f"A = {A:.3f}   magnitude: {effect_size_magnitude(A)}")
```

## Step 8 — Plug In Your First Component

Suppose you want to replace swap mutation with insertion mutation. Two
ways:

### Option A — CLI override

```bash
python scripts/run.py --config configs/nsga2_baseline.yaml \
                      --instance instances/ot_small.json \
                      --out results/nsga2_insertion/ \
                      --mutation insertion
```

### Option B — Programmatic

```python
from cio.algorithms import build_nsga2
from cio.operators import PMXCrossover, InsertionMutation
from cio.problem import Instance, OTSchedulingProblem
from pymoo.optimize import minimize

inst = Instance.from_json("instances/ot_medium.json")
problem = OTSchedulingProblem(inst)

algo = build_nsga2(
    pop_size=100,
    crossover=PMXCrossover(),
    mutation=InsertionMutation(),
    seed=0,
)
res = minimize(problem, algo, ("n_gen", 100), seed=0, verbose=True)
print(res.F)
```

## Step 9 — Design Your Own Operator

The minimum requirement is two novel components, at least one
problem-tailored. Here's the skeleton for a problem-tailored crossover:

```python
import numpy as np
from cio.operators import BaseCrossover

class UrgencyAwareCrossover:
    """Crossover that biases child orderings toward high-priority surgeries
    appearing earlier. Crosses parents using a priority-weighted blend."""

    def __init__(self, instance):
        self.instance = instance

    def __call__(self, parent_a, parent_b, rng):
        # Your design goes here. Must return two children, each a valid
        # permutation of length n_var.
        ...
        return child1, child2
```

Pass it to `build_nsga2(..., crossover=UrgencyAwareCrossover(inst))`.

## Step 10 — Compare to Baselines

Once you have a custom configuration, run it on all three instances with
30 runs each, then compare against the three baselines using HV, IGD+,
and statistical tests. The marking rubric (Code/C7) gives full marks for
demonstrating significant improvement with adequate effect size on at
least one indicator and at least one instance.

For full distinction credit, your design choices must be motivated by
empirical observation — see the rubric criterion on "evidence-driven
component design" and the rubric on the paper write-up.

## Pitfalls and Tips

- **Population size**: pymoo's NSGA-II needs `pop_size` even when the
  result has fewer points. With permutation problems, duplicates are
  possible; we set `eliminate_duplicates=False` to keep the dynamics
  predictable.
- **MOEA/D's effective pop**: with 3 objectives and `n_partitions=13`,
  Das-Dennis produces 105 weight vectors, so the effective population is
  fixed at 105 regardless of what you pass for `pop_size`.
- **Seeds**: every algorithm and operator that uses randomness accepts a
  seed. The harness derives a deterministic seed sequence from
  `--base-seed`, so you can reproduce any run.
- **Don't bypass the decoder**: if you want to design constraint-handling
  logic, do it via a custom `BaseDecoder` subclass, not by hand-crafting
  schedules. The Problem class only accepts permutations.
- **Read the brief**: the rubric specifies exactly what evidence is
  needed for each grade band. Read it before designing your experiments.
