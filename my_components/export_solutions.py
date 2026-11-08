"""Export the deliverables required by Section 8.1 of the assessment brief.

For each instance, and for each solution in the pooled non-dominated set found
across all runs, this writes:

  * the schedule itself -- every scheduled surgery with its OT, day, team,
    start hour and duration, plus the list of unscheduled surgeries;
  * the three objective values;
  * a feasibility certificate -- every hard constraint checked explicitly,
    with the observed and permitted values recorded, so a marker can verify
    feasibility without re-running the solver.

It also writes the per-run indicator values summarised across runs
(mean/median/std/IQR), as required by the same section.

    python my_components/export_solutions.py --instances small medium large

Outputs to solutions/<instance>/:
    pareto_front.json        pooled non-dominated set: schedule + objectives +
                             certificate for every solution
    schedules.csv            flat surgery-level view of every solution
    indicators_summary.csv   per-run indicators summarised across runs
    certificate_summary.txt  one-line verdict per solution, plus totals
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cio.problem import Instance, GreedyDecoder
from cio.problem.evaluator import evaluate
from cio.analysis import hypervolume, igd_plus, spread
from my_components.decoder import RepairBasedDecoder
from analyse import nd_mask, derive_reference_point, summarise, write_csv

DECODERS = {"greedy": GreedyDecoder, "repair": RepairBasedDecoder}


def certify(inst: Instance, sched) -> dict:
    """Explicit check of every hard constraint in brief section 2.4.

    Returns a certificate recording, for each constraint, whether it holds and
    the worst observed value against its permitted bound. Bed occupancy is
    recomputed from the assignments rather than read from the decoder's own
    bookkeeping, so the certificate is independent of the decoder.
    """
    ots = {o.id: o for o in inst.ots}
    teams = {t.id: t for t in inst.teams}
    surg = {s.id: s for s in inst.surgeries}
    checks, violations = {}, []

    ids = [a.surgery_id for a in sched.assignments]
    unscheduled = sorted(set(sched.unscheduled_surgery_ids))
    checks["each_surgery_at_most_once"] = {
        "ok": len(ids) == len(set(ids)),
        "scheduled": len(ids), "distinct": len(set(ids)),
    }
    checks["surgery_accounting_complete"] = {
        "ok": len(set(ids)) + len(unscheduled) == inst.n_surgeries
              and not (set(ids) & set(unscheduled)),
        "scheduled": len(set(ids)), "unscheduled": len(unscheduled),
        "total_required": inst.n_surgeries,
    }

    ot_hours, team_hours = defaultdict(float), defaultdict(float)
    spec_ot = spec_team = equip = earliest = horizon = True
    for a in sched.assignments:
        s = surg[a.surgery_id]
        ot, team = ots[a.ot_id], teams[a.team_id]
        ot_hours[(a.ot_id, a.day)] += a.duration_hr
        team_hours[(a.team_id, a.day)] += a.duration_hr
        if s.specialty not in ot.permitted_specialties:
            spec_ot = False; violations.append(f"surgery {s.id}: specialty {s.specialty} not permitted in OT {ot.id}")
        if not team.can_do(s.specialty):
            spec_team = False; violations.append(f"surgery {s.id}: team {team.id} cannot perform {s.specialty}")
        if not s.required_equipment.issubset(ot.equipment):
            equip = False; violations.append(f"surgery {s.id}: equipment missing in OT {ot.id}")
        if a.day < s.earliest_day:
            earliest = False; violations.append(f"surgery {s.id}: day {a.day} before earliest_day {s.earliest_day}")
        if not (1 <= a.day <= inst.horizon_days):
            horizon = False; violations.append(f"surgery {s.id}: day {a.day} outside horizon")

    worst_ot = max(((v, k) for k, v in ot_hours.items()), default=(0.0, None))
    worst_team = max(((v, k) for k, v in team_hours.items()), default=(0.0, None))
    ot_ok = all(v <= ots[k[0]].daily_hours + 1e-9 for k, v in ot_hours.items())
    team_ok = all(v <= teams[k[0]].daily_hours_max + 1e-9 for k, v in team_hours.items())
    if not ot_ok:
        violations.append(f"OT daily hours exceeded (worst {worst_ot[0]:.2f}h at {worst_ot[1]})")
    if not team_ok:
        violations.append(f"team daily hours exceeded (worst {worst_team[0]:.2f}h at {worst_team[1]})")

    active = defaultdict(set)
    for a in sched.assignments:
        active[a.team_id].add(a.day)
    worst_run, cons_ok = 0, True
    for tid, days in active.items():
        days = sorted(days); run = best = 1 if days else 0
        for i in range(1, len(days)):
            run = run + 1 if days[i] == days[i - 1] + 1 else 1
            best = max(best, run)
        worst_run = max(worst_run, best)
        if best > teams[tid].max_consecutive_days:
            cons_ok = False
            violations.append(f"team {tid}: consecutive run {best} > {teams[tid].max_consecutive_days}")

    icu, ward = defaultdict(int), defaultdict(int)
    for a in sched.assignments:
        s = surg[a.surgery_id]
        for d in range(a.day, a.day + s.icu_days):
            if d <= inst.horizon_days:
                icu[d] += 1
        w0 = a.day + s.icu_days
        for d in range(w0, w0 + s.ward_days):
            if d <= inst.horizon_days:
                ward[d] += 1
    peak_icu = max(icu.values(), default=0)
    peak_ward = max(ward.values(), default=0)
    if peak_icu > inst.icu_capacity:
        violations.append(f"ICU capacity exceeded: peak {peak_icu} > {inst.icu_capacity}")
    if peak_ward > inst.ward_capacity:
        violations.append(f"ward capacity exceeded: peak {peak_ward} > {inst.ward_capacity}")

    checks.update({
        "ot_daily_hours":        {"ok": ot_ok, "worst_observed": round(worst_ot[0], 4), "permitted": ots[worst_ot[1][0]].daily_hours if worst_ot[1] else None},
        "team_daily_hours":      {"ok": team_ok, "worst_observed": round(worst_team[0], 4), "permitted": teams[worst_team[1][0]].daily_hours_max if worst_team[1] else None},
        "team_consecutive_days": {"ok": cons_ok, "worst_observed": worst_run, "permitted": max((t.max_consecutive_days for t in inst.teams), default=None)},
        "specialty_permitted_in_ot": {"ok": spec_ot},
        "specialty_matches_team":    {"ok": spec_team},
        "equipment_available":       {"ok": equip},
        "not_before_earliest_day":   {"ok": earliest},
        "within_horizon":            {"ok": horizon},
        "icu_daily_capacity":  {"ok": peak_icu <= inst.icu_capacity, "peak_observed": peak_icu, "permitted": inst.icu_capacity},
        "ward_daily_capacity": {"ok": peak_ward <= inst.ward_capacity, "peak_observed": peak_ward, "permitted": inst.ward_capacity},
    })
    return {"feasible": all(c["ok"] for c in checks.values()), "checks": checks, "violations": violations}


def load_runs(results: Path, instance: str):
    """Yield (config, seed, decoder_name, F, X) for every run of every config."""
    for cfg_dir in sorted(p for p in (results / instance).iterdir() if p.is_dir()):
        summary = cfg_dir / "summary.json"
        decoder = "greedy"
        if summary.exists():
            decoder = json.loads(summary.read_text()).get("decoder") or "greedy"
        for p in sorted(glob.glob(str(cfg_dir / "run_*.json"))):
            d = json.load(open(p))
            F = np.atleast_2d(np.asarray(d.get("F", []), dtype=float))
            X = np.atleast_2d(np.asarray(d.get("X", []), dtype=int))
            if F.size and X.size and len(F) == len(X):
                yield cfg_dir.name, d["seed"], decoder, F, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("solutions"))
    ap.add_argument("--instances", nargs="+", default=["small", "medium", "large"])
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    results = args.results if args.results.is_absolute() else root / args.results
    out_root = args.out if args.out.is_absolute() else root / args.out

    for inst_name in args.instances:
        inst_path = root / "instances" / f"ot_{inst_name}.json"
        if not (results / inst_name).exists() or not inst_path.exists():
            print(f"[{inst_name}] skipped (no results or instance)")
            continue
        inst = Instance.from_json(inst_path)
        surg = {s.id: s for s in inst.surgeries}

        rows, per_cfg = [], defaultdict(list)
        for cfg, seed, decoder, F, X in load_runs(results, inst_name):
            per_cfg[cfg].append(F)
            for f, x in zip(F, X):
                rows.append({"config": cfg, "seed": int(seed), "decoder": decoder,
                             "F": f, "X": np.asarray(x, dtype=int)})
        if not rows:
            print(f"[{inst_name}] no usable runs")
            continue

        allF = np.vstack([r["F"] for r in rows])
        mask = nd_mask(allF)
        chosen = [r for r, m in zip(rows, mask) if m]
        # De-duplicate identical objective vectors, keeping the first occurrence.
        seen, unique = set(), []
        for r in chosen:
            key = tuple(np.round(r["F"], 6))
            if key not in seen:
                seen.add(key); unique.append(r)

        out = out_root / inst_name
        out.mkdir(parents=True, exist_ok=True)

        solutions, flat, n_feasible = [], [], 0
        for i, r in enumerate(unique, start=1):
            sched = DECODERS[r["decoder"]]()(inst, r["X"])
            f1, f2, f3 = evaluate(inst, sched)
            cert = certify(inst, sched)
            n_feasible += bool(cert["feasible"])
            assignments = [{
                "surgery_id": a.surgery_id, "priority": surg[a.surgery_id].priority,
                "specialty": surg[a.surgery_id].specialty, "ot_id": a.ot_id, "day": a.day,
                "team_id": a.team_id, "start_hour": round(a.start_hour, 3),
                "duration_hr": round(a.duration_hr, 3),
                "icu_days": surg[a.surgery_id].icu_days, "ward_days": surg[a.surgery_id].ward_days,
                "waiting_days": a.day - surg[a.surgery_id].earliest_day,
            } for a in sorted(sched.assignments, key=lambda a: (a.day, a.ot_id, a.start_hour))]
            solutions.append({
                "solution_id": i, "source_config": r["config"], "source_seed": r["seed"],
                "decoder": r["decoder"],
                "objectives": {"f1_neg_priority_weighted_throughput": f1,
                               "f2_resource_imbalance": f2,
                               "f3_waiting_fairness": f3},
                "n_scheduled": len(assignments),
                "n_unscheduled": len(set(sched.unscheduled_surgery_ids)),
                "unscheduled_surgery_ids": sorted(set(sched.unscheduled_surgery_ids)),
                "permutation": r["X"].tolist(),
                "schedule": assignments,
                "feasibility_certificate": cert,
            })
            for a in assignments:
                flat.append([i, a["surgery_id"], a["priority"], a["specialty"], a["ot_id"],
                             a["day"], a["team_id"], a["start_hour"], a["duration_hr"],
                             a["waiting_days"], round(f1, 4), round(f2, 4), round(f3, 6)])

        (out / "pareto_front.json").write_text(json.dumps({
            "instance": inst.name, "n_solutions": len(solutions),
            "n_feasible": n_feasible,
            "note": "Pooled non-dominated set across all configurations and runs. "
                    "Each solution carries its schedule, objective values and an "
                    "independently recomputed feasibility certificate.",
            "solutions": solutions,
        }, indent=2), encoding="utf-8")

        write_csv(out / "schedules.csv",
                  ["solution_id", "surgery_id", "priority", "specialty", "ot_id", "day",
                   "team_id", "start_hour", "duration_hr", "waiting_days", "f1", "f2", "f3"],
                  flat)

        # Per-run indicators, summarised across runs (same section of the brief).
        ref_point = derive_reference_point([r["F"].reshape(1, -1) for r in rows])
        rf_path = root / "instances" / f"ot_{inst_name}_reference_front.json"
        ref_front = np.asarray(json.load(open(rf_path))["points"], float) if rf_path.exists() else None
        ind_rows = []
        for cfg, fronts in per_cfg.items():
            vals = {"HV": [], "IGD+": [], "spread": []}
            for F in fronts:
                Fn = F[nd_mask(F)]
                vals["HV"].append(hypervolume(Fn, ref_point))
                vals["IGD+"].append(igd_plus(Fn, ref_front) if ref_front is not None else np.nan)
                vals["spread"].append(spread(Fn))
            for k, v in vals.items():
                s = summarise(v)
                ind_rows.append([cfg, k, s["mean"], s["median"], s["std"], s["iqr"], s["n"]])
        write_csv(out / "indicators_summary.csv",
                  ["config", "indicator", "mean", "median", "std", "iqr", "n_runs"], ind_rows)

        lines = [f"Instance: {inst.name}",
                 f"Pooled non-dominated solutions: {len(solutions)}",
                 f"Certified feasible: {n_feasible}/{len(solutions)}", ""]
        for s in solutions:
            v = "FEASIBLE" if s["feasibility_certificate"]["feasible"] else "INFEASIBLE"
            lines.append(f"  solution {s['solution_id']:4d}  {v:10s}  "
                         f"scheduled={s['n_scheduled']:3d}  "
                         f"f1={s['objectives']['f1_neg_priority_weighted_throughput']:10.2f}  "
                         f"f2={s['objectives']['f2_resource_imbalance']:9.2f}  "
                         f"f3={s['objectives']['f3_waiting_fairness']:8.4f}")
            for msg in s["feasibility_certificate"]["violations"]:
                lines.append(f"        ! {msg}")
        (out / "certificate_summary.txt").write_text("\n".join(lines), encoding="utf-8")

        print(f"[{inst_name}] {len(solutions)} non-dominated solutions, "
              f"{n_feasible} certified feasible -> {out}")


if __name__ == "__main__":
    main()
