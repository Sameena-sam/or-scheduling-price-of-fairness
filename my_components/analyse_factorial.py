"""Analysis of the crossed factorial (eviction rule x reinsertion policy).

Produces the two-way effect decomposition and the statistical tests needed to
support (or refute) the mechanism-attribution claim.

    python my_components/analyse_factorial.py --results results/factorial_medium/

Outputs to <results>/analysis/:
    cell_means.csv        per-cell mean/median/std/IQR for HV, IGD+, spread,
                          plus front cardinality and its variance across seeds
    two_way_effects.csv   main effect of each factor, interaction, and the
                          share of the greedy gap each factor accounts for
    factorial_tests.txt   Friedman across the six cells; Holm-corrected
                          pairwise p-values; Vargha-Delaney A vs the control
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import scikit_posthocs as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cio.analysis import hypervolume, igd_plus, spread, friedman, vargha_delaney, effect_size_magnitude
from analyse import nd_mask, derive_reference_point, summarise, write_csv, holm_pairwise_fixed

EVICTION = ["priority", "permutation", "random"]
REINSERT = ["dayplus1", "earliest"]
CELLS = [f"{e}_{r}" for e in EVICTION for r in REINSERT]
CONTROL = "control_greedy"
INDICATORS = ["HV", "IGD+", "spread"]


def load_cell(results: Path, cell: str):
    fronts = []
    for p in sorted(glob.glob(str(results / cell / "run_*.json"))):
        F = np.asarray(json.load(open(p))["F"], dtype=float)
        if F.ndim == 1 and F.size:
            F = F.reshape(1, -1)
        if F.size:
            fronts.append(F[nd_mask(F)])
    return fronts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--reference-front", type=Path, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    results = args.results if args.results.is_absolute() else root / args.results
    out = results / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    data = {c: load_cell(results, c) for c in [CONTROL] + CELLS}
    data = {c: v for c, v in data.items() if v}
    if CONTROL not in data:
        print(f"WARNING: control cell '{CONTROL}' missing; effects are relative only")

    ref_front = None
    if args.reference_front and args.reference_front.exists():
        ref_front = np.asarray(json.load(open(args.reference_front))["points"], dtype=float)

    # One reference point across every cell -- HV is otherwise not comparable.
    ref_point = derive_reference_point([F for fs in data.values() for F in fs])

    per_cell = {}
    rows = []
    for cell, fronts in data.items():
        vals = {"HV": [], "IGD+": [], "spread": [], "n_points": []}
        for F in fronts:
            vals["HV"].append(hypervolume(F, ref_point))
            vals["IGD+"].append(igd_plus(F, ref_front) if ref_front is not None else np.nan)
            vals["spread"].append(spread(F))
            vals["n_points"].append(len(np.unique(F, axis=0)))
        per_cell[cell] = vals
        for k in ["HV", "IGD+", "spread", "n_points"]:
            s = summarise(vals[k])
            rows.append([cell, k, s["mean"], s["median"], s["std"], s["iqr"], s["n"]])
    write_csv(out / "cell_means.csv",
              ["cell", "metric", "mean", "median", "std", "iqr", "n_runs"], rows)

    # ---- two-way decomposition -------------------------------------------
    eff_rows = []
    for metric in INDICATORS + ["n_points"]:
        m = {}
        for e, r in itertools.product(EVICTION, REINSERT):
            c = f"{e}_{r}"
            if c in per_cell:
                m[(e, r)] = float(np.nanmean(per_cell[c][metric]))
        if len(m) < len(EVICTION) * len(REINSERT):
            continue
        # main effect of reinsertion: mean over eviction rules of (earliest - dayplus1)
        eff_reinsert = float(np.mean([m[(e, "earliest")] - m[(e, "dayplus1")] for e in EVICTION]))
        # eviction effect: range of eviction-rule means, averaged over reinsertion
        ev_means = {e: float(np.mean([m[(e, r)] for r in REINSERT])) for e in EVICTION}
        eff_evict = max(ev_means.values()) - min(ev_means.values())
        # interaction: does the reinsertion effect depend on the eviction rule?
        per_rule = [m[(e, "earliest")] - m[(e, "dayplus1")] for e in EVICTION]
        interaction = float(max(per_rule) - min(per_rule))
        # how much of the gap to the greedy control does each factor span?
        ctrl = float(np.nanmean(per_cell[CONTROL][metric])) if CONTROL in per_cell else np.nan
        best_repair = (max(m.values()) if metric in ("HV", "spread", "n_points") else min(m.values()))
        gap = abs(ctrl - best_repair) if not np.isnan(ctrl) else np.nan
        eff_rows.append([metric, ctrl, min(m.values()), max(m.values()),
                         eff_evict, eff_reinsert, interaction,
                         gap, ev_means["priority"], ev_means["permutation"], ev_means["random"]])
    write_csv(out / "two_way_effects.csv",
              ["metric", "control_greedy", "worst_cell", "best_cell",
               "eviction_effect_range", "reinsertion_main_effect", "interaction_range",
               "residual_gap_best_cell_to_control",
               "evict_priority_mean", "evict_permutation_mean", "evict_random_mean"], eff_rows)

    # ---- statistics -------------------------------------------------------
    lines = []
    order = [c for c in [CONTROL] + CELLS if c in per_cell]
    for metric in INDICATORS:
        n = min(len(per_cell[c][metric]) for c in order)
        if n < 2:
            continue
        mat = np.column_stack([per_cell[c][metric][:n] for c in order])
        if np.isnan(mat).any():
            lines.append(f"[{metric}] skipped (missing values)\n")
            continue
        fr = friedman(mat)
        lines.append(f"[{metric}] Friedman chi2={fr['statistic']:.4f} p={fr['p_value']:.4e} "
                     f"(k={fr['n_configurations']}, n={fr['n_runs']})\n")
        lines.append(f"    column order: {', '.join(order)}\n")
        ph = holm_pairwise_fixed(mat)
        write_csv(out / f"posthoc_{metric.replace('+','plus')}.csv",
                  ["cell"] + order,
                  [[order[i]] + [ph[i, j] for j in range(len(order))] for i in range(len(order))])
        if CONTROL in order:
            ci = order.index(CONTROL)
            lines.append(f"    vs control ({CONTROL}):\n")
            for j, c in enumerate(order):
                if c == CONTROL:
                    continue
                A = vargha_delaney(mat[:, ci], mat[:, j])
                lines.append(f"      {c:26s} A={A:.4f} ({effect_size_magnitude(A):>10s})  "
                             f"Holm p={ph[ci, j]:.3e}  {'SIG' if ph[ci, j] < 0.05 else 'ns'}\n")
        lines.append("\n")
    (out / "factorial_tests.txt").write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {out}/")
    print((out / "factorial_tests.txt").read_text()[:2000])


if __name__ == "__main__":
    main()
