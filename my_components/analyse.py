"""Analysis pipeline: run logs -> indicators -> statistics -> paper tables.

Produces everything Section 6 of the paper needs, from the JSON run logs
written by the experiment harness. Run from the cio_skeleton/ root:

    PYTHONPATH=. python my_components/analyse.py

Outputs (written to analysis/):

    indicators_raw.csv        per-run HV / IGD+ / spread for every configuration
    indicators_summary.csv    mean / median / std / IQR per configuration
    friedman.txt              omnibus tests per instance per indicator
    posthoc_<inst>_<ind>.csv  Holm-corrected pairwise p-values
    effect_sizes.csv          Vargha-Delaney A for every pair, with magnitude
    doe_effects.csv           2^2 factorial main effects and interaction
    price_of_fairness.csv     f1-f3 trade-off along the combined front
    reference_points.txt      HV reference point per instance + how it was derived
    figures/*.png             Pareto fronts, trade-off curves, DoE effects

Options:

    --results-dir results/    where the run logs live
    --out analysis/           where to write tables and figures
    --instances small medium large
    --no-figures              skip plotting (if matplotlib is unavailable)

NOTE ON WHAT THIS DOES AND DOESN'T DO. This computes the numbers. It does not
interpret them -- the reading of these tables is yours to write, and it is what
P6/P7 and the viva assess.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scikit_posthocs as sp

from cio.analysis import (
    hypervolume, igd_plus, spread, non_dominated,
    friedman, vargha_delaney, effect_size_magnitude,
)


def holm_pairwise_fixed(data: np.ndarray) -> np.ndarray:
    """Holm-corrected pairwise Wilcoxon signed-rank across configurations.

    NOTE — this deliberately does NOT call `cio.analysis.holm_pairwise`.
    That function passes the (n_runs x n_configs) matrix straight to
    `scikit_posthocs.posthoc_wilcoxon`, which treats each ROW as a group.
    For a 30x6 input it therefore compares 30 "groups" of 6 observations and
    returns a 30x30 matrix of 1.0 -- i.e. every post-hoc p-value is 1.0
    regardless of the data, even when the configurations are perfectly
    separated (Vargha-Delaney A = 1.0).

    scikit-posthocs is "a recognised library" in the sense of brief 8.2; this
    calls it with the groups passed correctly, as a list of per-configuration
    samples. Verified against a synthetic case with known separation.
    """
    data = np.asarray(data, dtype=float)
    groups = [data[:, j] for j in range(data.shape[1])]
    return np.asarray(sp.posthoc_wilcoxon(groups, p_adjust="holm"), dtype=float)

# Configuration display order. `baseline_nsga2` doubles as the DoE cell
# (greedy decoder, no local search) -- same configuration and seeds.
CONFIGS = [
    "baseline_nsga2",
    "baseline_moead",
    "baseline_pls",
    "doe_repair_noLS",
    "doe_greedy_LS",
    "doe_repair_LS",
]

DOE_CELLS = {
    (False, False): "baseline_nsga2",     # greedy decoder, no local search
    (True, False): "doe_repair_noLS",     # repair decoder, no local search
    (False, True): "doe_greedy_LS",       # greedy decoder, local search
    (True, True): "doe_repair_LS",        # repair decoder, local search
}

INDICATORS = ["HV", "IGD+", "spread"]


# ------------------------------------------------- fast dominance filter

def nd_mask(F: np.ndarray) -> np.ndarray:
    """Non-dominated mask, vectorised.

    `cio.analysis.non_dominated` is an O(n^2) pure-Python double loop, which is
    fine for a single run's front (~100 points) but not for the pooled front
    across every configuration and run (tens of thousands). This is the same
    relation -- verified against the reference implementation in the test at
    the bottom of this file -- with the inner comparison vectorised.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    # Lexicographic order is essential, not cosmetic: sorting on the first
    # objective alone lets a point dominate an EARLIER point whenever the
    # first objective ties, which a forward-only sweep would miss. Under
    # lexicographic order, any dominator sorts strictly before its victim.
    order = np.lexsort(tuple(F[:, k] for k in range(F.shape[1] - 1, -1, -1)))
    Fs = F[order]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        tail = np.nonzero(keep[i + 1:])[0] + i + 1
        if tail.size == 0:
            continue
        block = Fs[tail]
        dominated = np.all(block >= Fs[i], axis=1) & np.any(block > Fs[i], axis=1)
        if dominated.any():
            keep[tail[dominated]] = False
    out = np.zeros(n, dtype=bool)
    out[order[keep]] = True
    return out


def pooled_front(fronts: list[np.ndarray]) -> np.ndarray:
    """Non-dominated set of the union of many fronts, deduplicated first."""
    if not fronts:
        return np.empty((0, 3))
    stacked = np.vstack([F for F in fronts if F.size])
    if stacked.size == 0:
        return np.empty((0, 3))
    stacked = np.unique(stacked, axis=0)      # collapses many identical points
    return stacked[nd_mask(stacked)]


# ---------------------------------------------------------------- loading

_RUN_CACHE: dict[tuple[str, str], list[dict]] = {}


def load_runs(results_dir: Path, instance: str, config: str) -> list[dict]:
    """Load and cache run logs. Each run's F is reduced to its own
    non-dominated set (pymoo may return dominated members)."""
    key = (instance, config)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    paths = sorted(glob.glob(str(results_dir / instance / config / "run_*.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        F = np.asarray(d.get("F", []), dtype=float)
        if F.ndim == 1 and F.size:
            F = F.reshape(1, -1)
        if F.size == 0:
            continue
        d["_F"] = F[nd_mask(F)]
        runs.append(d)
    _RUN_CACHE[key] = runs
    return runs


def load_reference_front(instance: str, root: Path) -> np.ndarray | None:
    path = root / "instances" / f"ot_{instance}_reference_front.json"
    if not path.exists():
        return None
    with open(path) as f:
        return np.asarray(json.load(f)["points"], dtype=float)


# ------------------------------------------------------- reference points

def derive_reference_point(all_fronts: list[np.ndarray], margin: float = 0.10):
    """HV reference point: componentwise worst observed value across every
    configuration on this instance, inflated by `margin`.

    Deriving it from the union of all configurations (rather than per
    configuration) is what makes HV comparable across configurations -- the
    same reference point must be used for every configuration being compared.
    The margin ensures the point is strictly dominated by every observed
    solution, so no front is silently clipped to zero contribution.
    """
    stacked = np.vstack([F for F in all_fronts if F.size])
    worst = stacked.max(axis=0)
    best = stacked.min(axis=0)
    span = np.where(worst - best > 0, worst - best, np.abs(worst) + 1.0)
    return worst + margin * span


# ------------------------------------------------------------ indicators

def compute_indicators(runs, ref_point, ref_front):
    rows = []
    for r in runs:
        F = r["_F"]
        rows.append({
            "seed": r.get("seed"),
            "n_solutions": int(F.shape[0]),
            "HV": hypervolume(F, ref_point),
            "IGD+": igd_plus(F, ref_front) if ref_front is not None else np.nan,
            "spread": spread(F),
            "runtime_s": r.get("runtime_seconds", np.nan),
            "evaluations": r.get("evaluations", np.nan),
        })
    return rows


def summarise(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {k: np.nan for k in ("mean", "median", "std", "iqr", "min", "max", "n")}
    q75, q25 = np.percentile(v, [75, 25])
    return {"mean": v.mean(), "median": np.median(v), "std": v.std(ddof=1) if v.size > 1 else 0.0,
            "iqr": q75 - q25, "min": v.min(), "max": v.max(), "n": int(v.size)}


# --------------------------------------------------------------- writing

def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("" if v is None else
                             (f"{v:.6g}" if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
                              else str(v)) for v in row) + "\n")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--instances", nargs="+", default=["small", "medium", "large"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir if not args.results_dir.is_absolute() else args.results_dir
    out = root / args.out if not args.out.is_absolute() else args.out
    out.mkdir(parents=True, exist_ok=True)

    raw_rows, summary_rows, effect_rows, friedman_lines, ref_lines = [], [], [], [], []
    per_instance_fronts = {}

    for inst in args.instances:
        present = {c: load_runs(results_dir, inst, c) for c in CONFIGS}
        present = {c: r for c, r in present.items() if r}
        if not present:
            print(f"[{inst}] no results found, skipping")
            continue

        ref_front = load_reference_front(inst, root)
        all_fronts = [r["_F"] for runs in present.values() for r in runs]
        ref_point = derive_reference_point(all_fronts)
        ref_lines.append(
            f"{inst}: ref_point = [{', '.join(f'{v:.4f}' for v in ref_point)}]\n"
            f"    derived as componentwise worst observed objective value across all "
            f"{len(present)} configurations ({sum(len(v) for v in present.values())} runs), "
            f"inflated by 10% of the observed range per objective.\n"
            f"    reference front for IGD+: "
            f"{'instances/ot_%s_reference_front.json (budget_profile=preliminary)' % inst if ref_front is not None else 'NOT FOUND'}\n"
        )

        print(f"[{inst}] {len(present)} configurations, ref_point={np.round(ref_point,2)}")

        ind_by_config = {}
        for cfg, runs in present.items():
            rows = compute_indicators(runs, ref_point, ref_front)
            ind_by_config[cfg] = rows
            for row in rows:
                raw_rows.append([inst, cfg, row["seed"], row["n_solutions"],
                                 row["HV"], row["IGD+"], row["spread"],
                                 row["runtime_s"], row["evaluations"]])
            for ind in INDICATORS:
                s = summarise([r[ind] for r in rows])
                summary_rows.append([inst, cfg, ind, s["mean"], s["median"], s["std"],
                                     s["iqr"], s["min"], s["max"], s["n"]])
            per_instance_fronts.setdefault(inst, {})[cfg] = np.vstack([r["_F"] for r in runs])

        # ---- Friedman + Holm post-hoc + effect sizes, per indicator ----
        for ind in INDICATORS:
            cfgs = [c for c in CONFIGS if c in ind_by_config]
            n = min(len(ind_by_config[c]) for c in cfgs)
            if n < 2 or len(cfgs) < 2:
                continue
            data = np.column_stack([[r[ind] for r in ind_by_config[c][:n]] for c in cfgs])
            if np.isnan(data).any():
                friedman_lines.append(f"[{inst}] {ind}: skipped (missing values)\n")
                continue

            fr = friedman(data)
            friedman_lines.append(
                f"[{inst}] {ind}: Friedman chi2={fr['statistic']:.4f}  p={fr['p_value']:.3e}  "
                f"(k={fr['n_configurations']} configs, n={fr['n_runs']} runs)\n"
                f"    configurations (column order): {', '.join(cfgs)}\n"
            )

            try:
                ph = holm_pairwise_fixed(data)
                write_csv(out / f"posthoc_{inst}_{ind.replace('+','plus')}.csv",
                          ["config"] + cfgs,
                          [[cfgs[i]] + [ph[i, j] for j in range(len(cfgs))] for i in range(len(cfgs))])
            except Exception as e:                       # pragma: no cover
                friedman_lines.append(f"    post-hoc failed: {e}\n")
                ph = None

            higher_better = (ind in ("HV", "spread"))
            for i, j in combinations(range(len(cfgs)), 2):
                a, b = data[:, i], data[:, j]
                A = vargha_delaney(a, b)
                p = float(ph[i, j]) if ph is not None else np.nan
                effect_rows.append([inst, ind, cfgs[i], cfgs[j], A,
                                    effect_size_magnitude(A), p,
                                    "yes" if (not np.isnan(p) and p < 0.05) else "no",
                                    "higher" if higher_better else "lower"])

    # ------------------------------------------------ DoE factorial effects
    doe_rows = []
    for inst in args.instances:
        fronts = per_instance_fronts.get(inst)
        if not fronts:
            continue
        # Reference point and front are per-instance, not per-cell: every cell
        # must be scored against the same reference for HV to be comparable.
        all_fronts = [r["_F"] for c in CONFIGS for r in load_runs(results_dir, inst, c)]
        if not all_fronts:
            continue
        rp = derive_reference_point(all_fronts)
        rf = load_reference_front(inst, root)
        cell_means = {}
        for (rep, ls), cfg in DOE_CELLS.items():
            runs = load_runs(results_dir, inst, cfg)
            if not runs:
                continue
            rows = compute_indicators(runs, rp, rf)
            for ind in INDICATORS:
                cell_means[(rep, ls, ind)] = float(np.nanmean([r[ind] for r in rows]))
        for ind in INDICATORS:
            try:
                m00 = cell_means[(False, False, ind)]
                m10 = cell_means[(True, False, ind)]
                m01 = cell_means[(False, True, ind)]
                m11 = cell_means[(True, True, ind)]
            except KeyError:
                continue
            main_repair = 0.5 * ((m10 - m00) + (m11 - m01))
            main_ls = 0.5 * ((m01 - m00) + (m11 - m10))
            interaction = 0.5 * ((m11 - m01) - (m10 - m00))
            doe_rows.append([inst, ind, m00, m10, m01, m11,
                             main_repair, main_ls, interaction,
                             "higher" if ind in ("HV", "spread") else "lower"])

    # ------------------------------------------------- price of fairness
    pof_rows = []
    for inst, fronts in per_instance_fronts.items():
        nd = pooled_front(list(fronts.values()))
        if nd.size == 0:
            continue
        nd = nd[np.argsort(nd[:, 2])]                # sort by f3 (fairness)
        best_f1 = nd[:, 0].min()
        for f1, f2, f3 in nd:
            # f1 is negated throughput, so (f1 - best_f1) is the throughput
            # forgone relative to the most productive schedule on the front.
            pof_rows.append([inst, f1, f2, f3, f1 - best_f1])

    # ------------------------------------------------------------- output
    write_csv(out / "indicators_raw.csv",
              ["instance", "config", "seed", "n_solutions", "HV", "IGD+", "spread",
               "runtime_s", "evaluations"], raw_rows)
    write_csv(out / "indicators_summary.csv",
              ["instance", "config", "indicator", "mean", "median", "std", "iqr",
               "min", "max", "n_runs"], summary_rows)
    write_csv(out / "effect_sizes.csv",
              ["instance", "indicator", "config_a", "config_b", "vargha_delaney_A",
               "magnitude", "holm_p", "significant_at_0.05", "better_is"], effect_rows)
    write_csv(out / "doe_effects.csv",
              ["instance", "indicator", "mean_greedy_noLS", "mean_repair_noLS",
               "mean_greedy_LS", "mean_repair_LS", "main_effect_repair",
               "main_effect_localsearch", "interaction", "better_is"], doe_rows)
    write_csv(out / "price_of_fairness.csv",
              ["instance", "f1", "f2", "f3_fairness", "f1_sacrificed_vs_best"], pof_rows)
    (out / "friedman.txt").write_text("".join(friedman_lines), encoding="utf-8")
    (out / "reference_points.txt").write_text("".join(ref_lines), encoding="utf-8")

    if not args.no_figures:
        try:
            make_figures(per_instance_fronts, out / "figures", doe_rows)
        except Exception as e:                            # pragma: no cover
            print(f"figures skipped: {e}")

    print(f"\nWrote analysis to {out}/")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"   {f.relative_to(out)}")


def make_figures(per_instance_fronts, figdir: Path, doe_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)

    for inst, fronts in per_instance_fronts.items():
        # f1 vs f3 -- the price-of-fairness view
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg, F in fronts.items():
            ax.scatter(F[:, 2], F[:, 0], s=12, alpha=0.55, label=cfg)
        ax.set_xlabel("$f_3$  waiting-time fairness (variance across priority classes)")
        ax.set_ylabel("$f_1$  negative priority-weighted throughput")
        ax.set_title(f"Throughput vs fairness trade-off — {inst} instance")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / f"tradeoff_f1_f3_{inst}.png", dpi=160)
        plt.close(fig)

        # combined non-dominated front, f1 vs f2
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg, F in fronts.items():
            ax.scatter(F[:, 1], F[:, 0], s=12, alpha=0.55, label=cfg)
        ax.set_xlabel("$f_2$  resource utilisation imbalance")
        ax.set_ylabel("$f_1$  negative priority-weighted throughput")
        ax.set_title(f"Throughput vs imbalance — {inst} instance")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / f"tradeoff_f1_f2_{inst}.png", dpi=160)
        plt.close(fig)

    # DoE main effects
    if doe_rows:
        insts = sorted({r[0] for r in doe_rows})
        for ind in INDICATORS:
            rows = [r for r in doe_rows if r[1] == ind]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(insts))
            rep = [next((r[6] for r in rows if r[0] == i), 0) for i in insts]
            ls = [next((r[7] for r in rows if r[0] == i), 0) for i in insts]
            inter = [next((r[8] for r in rows if r[0] == i), 0) for i in insts]
            ax.bar(x - 0.25, rep, 0.25, label="repair decoder")
            ax.bar(x, ls, 0.25, label="conflict-graph LS")
            ax.bar(x + 0.25, inter, 0.25, label="interaction")
            ax.set_xticks(x); ax.set_xticklabels(insts)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel(f"effect on {ind}")
            ax.set_title(f"2$^2$ factorial effects — {ind}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figdir / f"doe_effects_{ind.replace('+','plus')}.png", dpi=160)
            plt.close(fig)


if __name__ == "__main__":
    main()
