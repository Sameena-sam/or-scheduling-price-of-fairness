"""External benchmark validation on MMRCPSP (brief section 3.3 / 5.4).

Runs the same NSGA-II framework, the same permutation encoding and the same
two constraint-handling designs as the OT experiments, on PSPLIB project
scheduling instances. No parameter is re-tuned: population, generations,
crossover, mutation and run count are taken from the OT protocol.

The comparison answers two questions:

  1. Is the implementation competitive? Best makespan found is reported
     against the published best-known value for the instance, which is the
     conventional indicator for MMRCPSP.
  2. Does the OT finding transfer? The feasibility-by-construction decoder and
     the allow-then-repair decoder are compared under identical budgets, as in
     the OT design dataset.

USAGE

    # 1. download PSPLIB instances and extract into external_benchmark/instances/
    #    (j10.mm / j20.mm / j30.sm ... from the PSPLIB site)
    # 2. verify the pipeline on the bundled synthetic instance first
    python my_components/external/run_external.py --synthetic --n-runs 5 --n-generations 20

    # 3. run for real
    python my_components/external/run_external.py \\
        --instances external_benchmark/instances/j1010_1.mm \\
        --n-runs 30 --n-generations 100

    # 4. record published best-known makespans (optional but expected)
    python my_components/external/run_external.py \\
        --instances external_benchmark/instances/j1010_1.mm \\
        --best-known j1010_1=23
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cio.algorithms import build_nsga2
from cio.analysis import hypervolume, spread, friedman, vargha_delaney, effect_size_magnitude
from my_components.analyse import nd_mask, derive_reference_point, summarise, write_csv, holm_pairwise_fixed
from my_components.external.mmrcpsp import (
    ProjectInstance, parse_psplib, parse_psplib_solutions, make_synthetic_instance,
    GreedySGSDecoder, RepairSGSDecoder, evaluate_project,
)

DECODERS = {
    "greedy": lambda: GreedySGSDecoder(),
    "repair": lambda: RepairSGSDecoder(evict="shortest"),
    "repair_perm": lambda: RepairSGSDecoder(evict="permutation"),
}


class ProjectSchedulingProblem(ElementwiseProblem):
    """Permutation-encoded MMRCPSP, mirroring OTSchedulingProblem."""

    def __init__(self, instance: ProjectInstance, decoder):
        self.instance = instance
        self.decoder = decoder
        super().__init__(n_var=instance.n_activities, n_obj=2, n_constr=0,
                         xl=0, xu=instance.n_activities - 1)

    def _evaluate(self, x, out, *args, **kwargs):
        perm = np.asarray(x, dtype=int)
        n = self.instance.n_activities
        if perm.size != n or len(set(perm.tolist())) != perm.size:
            seen, rep = set(), []
            for v in perm.tolist():
                if v not in seen and 0 <= v < n:
                    rep.append(v); seen.add(v)
            rep.extend(v for v in range(n) if v not in seen)
            perm = np.asarray(rep, dtype=int)
        sched = self.decoder(self.instance, perm)
        f1, f2 = evaluate_project(self.instance, sched)
        out["F"] = [f1, f2]


def make_seeds(n_runs: int, base_seed: int = 0) -> list[int]:
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(n_runs)]


def run_config(inst, decoder_name, n_runs, n_generations, pop_size, base_seed, out_dir, verbose):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, seed in enumerate(make_seeds(n_runs, base_seed), start=1):
        decoder = DECODERS[decoder_name]()
        problem = ProjectSchedulingProblem(inst, decoder)
        t0 = time.time()
        res = minimize(problem, build_nsga2(pop_size=pop_size, seed=seed),
                       ("n_gen", n_generations), seed=seed, verbose=False)
        F = np.atleast_2d(res.F if res.F is not None else np.empty((0, 2)))
        X = np.atleast_2d(res.X if res.X is not None else np.empty((0, inst.n_activities), int))
        # feasibility of every returned solution, verified by re-decoding
        n_feasible = 0
        for row in X:
            n_feasible += bool(decoder(inst, np.asarray(row, int)).feasible)
        rec = {
            "instance": inst.name, "decoder": decoder_name, "seed": int(seed),
            "F": F.tolist(), "X": X.tolist(),
            "best_makespan": float(F[:, 0].min()) if F.size else None,
            "n_solutions": int(len(F)), "n_feasible": int(n_feasible),
            "evaluations": int(res.algorithm.evaluator.n_eval) if hasattr(res, "algorithm") else 0,
            "eval_budget": int(pop_size * n_generations),
            "runtime_seconds": float(time.time() - t0),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / f"run_{i:03d}_seed_{seed}.json").write_text(json.dumps(rec, indent=2))
        records.append(rec)
        if verbose:
            print(f"    run {i}/{n_runs}  best makespan {rec['best_makespan']}", end="\r", flush=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "instance": inst.name, "decoder": decoder_name, "n_runs": len(records),
        "eval_budget": int(pop_size * n_generations),
        "completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=[],
                    help="instance files, a directory, or a glob such as "
                         "'external_benchmark/instances/j10*.mm'")
    ap.add_argument("--limit", type=int, default=None,
                    help="use at most this many instances (after sorting), so a "
                         "640-file set can be sampled without listing paths")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on the bundled synthetic instance (pipeline check)")
    ap.add_argument("--out", type=Path, default=Path("results_external"))
    ap.add_argument("--decoders", nargs="+", default=["greedy", "repair", "repair_perm"])
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--n-generations", type=int, default=100)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--best-known", nargs="*", default=[],
                    help="published best-known makespans, e.g. j1010_1=23")
    ap.add_argument("--solutions-file", type=Path, default=None,
                    help="PSPLIB optimal-solutions file (e.g. external_benchmark/j10opt.mm); "
                         "published optima are then read automatically for every instance")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_root = args.out if args.out.is_absolute() else root / args.out
    best_known = {}
    if args.solutions_file:
        sf = args.solutions_file
        sf = sf if sf.is_absolute() else root / sf
        best_known.update(parse_psplib_solutions(sf))
        print(f"Loaded {len(best_known)} published optima from {sf.name}")
    for kv in args.best_known:                       # explicit values override the file
        k, _, v = kv.partition("=")
        best_known[k.strip()] = float(v)

    # Accept files, directories and globs; a PSPLIB set is 640 files, so
    # expanding a directory and taking --limit is the practical path.
    paths: list[Path] = []
    for spec in args.instances:
        p = Path(spec)
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.suffix in (".mm", ".sm")))
        elif any(ch in str(spec) for ch in "*?["):
            base = p.parent
            paths.extend(sorted(base.glob(p.name)))
        elif p.exists():
            paths.append(p)
        else:
            print(f"warning: {spec} not found, skipping")
    if args.limit:
        paths = paths[:args.limit]
    if args.synthetic or not paths:
        paths = [make_synthetic_instance(out_root / "instances" / "synth.mm")]
        print("Using the bundled synthetic instance (pipeline verification only).")
        print("For the reported benchmark, download PSPLIB .mm files and pass --instances.\n")

    summary_rows, verdict = [], []
    for path in paths:
        inst = parse_psplib(path)
        print(f"=== {inst.name}: {inst.n_activities} activities, "
              f"{inst.n_renewable} renewable, {inst.n_nonrenewable} nonrenewable ===")
        per_dec = {}
        for dname in args.decoders:
            t0 = time.time()
            print(f"  [{dname}] ...", flush=True)
            recs = run_config(inst, dname, args.n_runs, args.n_generations,
                              args.pop_size, args.base_seed,
                              out_root / inst.name / dname, not args.quiet)
            per_dec[dname] = recs
            print(f"  [{dname}] done in {time.time()-t0:.0f}s", flush=True)

        fronts = {d: [np.atleast_2d(np.asarray(r["F"], float)) for r in recs]
                  for d, recs in per_dec.items()}
        ref = derive_reference_point([F for fs in fronts.values() for F in fs])
        for d, fs in fronts.items():
            hv = [hypervolume(F[nd_mask(F)], ref) for F in fs]
            sp = [spread(F[nd_mask(F)]) for F in fs]
            mk = [r["best_makespan"] for r in per_dec[d]]
            nf = [r["n_feasible"] for r in per_dec[d]]
            ns = [r["n_solutions"] for r in per_dec[d]]
            for label, vals in [("HV", hv), ("spread", sp), ("best_makespan", mk)]:
                s = summarise(vals)
                summary_rows.append([inst.name, d, label, s["mean"], s["median"],
                                     s["std"], s["iqr"], s["min"], s["max"], s["n"]])
            summary_rows.append([inst.name, d, "feasible_fraction",
                                 float(np.sum(nf)) / max(1, float(np.sum(ns))),
                                 "", "", "", "", "", len(nf)])

        # statistics across decoders on HV and makespan
        order = [d for d in args.decoders if d in fronts]
        lines = [f"[{inst.name}] decoders: {', '.join(order)}"]

        # Saturation check. On small instances the evaluation budget can exhaust
        # the reachable objective space, so every seed returns the same front.
        # Statistical tests then compare constants: the difference between
        # decoders is real but deterministic, and p-values describe nothing.
        saturated = []
        for d in order:
            mk = [r["best_makespan"] for r in per_dec[d]]
            if len(set(mk)) == 1:
                saturated.append(d)
        if len(saturated) == len(order):
            lines.append("  NOTE: every seed returned an identical best makespan for every "
                         "decoder. The search has saturated the reachable space at this "
                         "instance size, so run-to-run variance is zero and the statistical "
                         "tests below compare constants. Report the ranking as deterministic, "
                         "not as a statistically significant difference; use a larger instance "
                         "set (j20) for tests with genuine variance.")
        elif saturated:
            lines.append(f"  NOTE: zero run-to-run variance for: {', '.join(saturated)}. "
                         f"Statistical comparisons involving these are not informative.")
        for label, getter in [("HV", lambda d: [hypervolume(F[nd_mask(F)], ref) for F in fronts[d]]),
                              ("best_makespan", lambda d: [r["best_makespan"] for r in per_dec[d]])]:
            n = min(len(getter(d)) for d in order)
            mat = np.column_stack([getter(d)[:n] for d in order])
            if len(order) >= 3:
                fr = friedman(mat)
                lines.append(f"  {label}: Friedman chi2={fr['statistic']:.4f} p={fr['p_value']:.4e}")
            ph = holm_pairwise_fixed(mat) if len(order) >= 2 else None
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    a_vals, b_vals = mat[:, i], mat[:, j]
                    A = vargha_delaney(a_vals, b_vals)
                    if np.allclose(a_vals, b_vals):
                        verdict_txt = "identical in every run (no test possible)"
                    else:
                        p = ph[i, j] if ph is not None else float("nan")
                        verdict_txt = ("Holm p=n/a (zero variance)" if np.isnan(p)
                                       else f"Holm p={p:.3e}")
                    lines.append(f"    {order[i]} vs {order[j]:12s} {label:14s} "
                                 f"A={A:.4f} ({effect_size_magnitude(A)}) {verdict_txt}")
        bk = best_known.get(inst.name)
        best_found = min(r["best_makespan"] for recs in per_dec.values() for r in recs)
        if bk:
            gap = 100.0 * (best_found - bk) / bk
            lines.append(f"  best makespan found = {best_found:.0f}; published best-known = {bk:.0f}; "
                         f"gap = {gap:+.2f}%")
            verdict.append([inst.name, best_found, bk, gap])
        else:
            lines.append(f"  best makespan found = {best_found:.0f}; no published value supplied "
                         f"(pass --best-known {inst.name}=<value>)")
            verdict.append([inst.name, best_found, "", ""])
        print("\n".join(lines) + "\n")
        (out_root / inst.name / "statistics.txt").write_text("\n".join(lines), encoding="utf-8")

    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "external_summary.csv",
              ["instance", "decoder", "metric", "mean", "median", "std", "iqr", "min", "max", "n_runs"],
              summary_rows)
    write_csv(out_root / "best_known_comparison.csv",
              ["instance", "best_makespan_found", "published_best_known", "gap_percent"], verdict)

    gaps = [r[3] for r in verdict if r[3] != ""]
    if gaps:
        gaps = np.asarray(gaps, dtype=float)
        optimal = int(np.sum(gaps <= 1e-9))
        print(f"\nAgainst published optima over {len(gaps)} instances:")
        print(f"  optimum reached : {optimal}/{len(gaps)} ({100*optimal/len(gaps):.1f}%)")
        print(f"  mean gap        : {gaps.mean():+.2f}%")
        print(f"  median gap      : {np.median(gaps):+.2f}%")
        print(f"  worst gap       : {gaps.max():+.2f}%")
        (out_root / "best_known_summary.txt").write_text(
            f"instances compared: {len(gaps)}\n"
            f"optimum reached: {optimal} ({100*optimal/len(gaps):.1f}%)\n"
            f"mean gap: {gaps.mean():+.4f}%\n"
            f"median gap: {np.median(gaps):+.4f}%\n"
            f"worst gap: {gaps.max():+.4f}%\n", encoding="utf-8")
    print(f"\nWrote {out_root}/external_summary.csv and best_known_comparison.csv")


if __name__ == "__main__":
    main()
