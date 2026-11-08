"""Objective evaluation for a decoded Schedule.

Three objectives, all to be MINIMISED in pymoo convention. The throughput
objective is negated internally so that "minimise" produces "maximise
throughput".

f1: -priority_weighted_throughput (minimise)
f2: resource_utilisation_imbalance (minimise)
f3: waiting_time_fairness (minimise; variance across priority classes)
"""
from __future__ import annotations

import numpy as np

from .instance import Instance
from .schedule import Schedule


# Penalty applied to unscheduled surgeries in the throughput objective.
# Calibrated so that an unscheduled surgery dominates the worst possible
# scheduled outcome for the same priority.
UNSCHEDULED_PENALTY_MULTIPLIER = 2.0


def evaluate(instance: Instance, sched: Schedule) -> tuple[float, float, float]:
    """Compute (f1, f2, f3) for a decoded schedule.

    Returns:
        Tuple of (f1, f2, f3), all to be minimised by pymoo.
    """
    f1 = _f1_neg_throughput(instance, sched)
    f2 = _f2_resource_imbalance(instance, sched)
    f3 = _f3_waiting_fairness(instance, sched)
    return f1, f2, f3


def _f1_neg_throughput(instance: Instance, sched: Schedule) -> float:
    """Negative priority-weighted throughput.

    Reward: + priority for each scheduled surgery.
    Penalty: - priority * days_late for surgeries scheduled beyond max wait.
    Penalty: - priority * horizon * UNSCHEDULED_PENALTY_MULTIPLIER for each
             unscheduled surgery.
    The whole quantity is negated so pymoo minimises it.
    """
    reward = 0.0
    for a in sched.assignments:
        s = next(s for s in instance.surgeries if s.id == a.surgery_id)
        reward += s.priority
        latest_no_penalty_day = s.earliest_day + s.max_wait_days
        if a.day > latest_no_penalty_day:
            days_late = a.day - latest_no_penalty_day
            reward -= s.priority * days_late
    for sid in sched.unscheduled_surgery_ids:
        s = next(s for s in instance.surgeries if s.id == sid)
        reward -= s.priority * instance.horizon_days * UNSCHEDULED_PENALTY_MULTIPLIER
    return -reward


def _f2_resource_imbalance(instance: Instance, sched: Schedule) -> float:
    w = instance.objective_weights
    ot_idle_w = float(w.get("ot_idle_weight", 1.0))
    icu_w = float(w.get("icu_overflow_weight", 10.0))
    ward_w = float(w.get("ward_overflow_weight", 5.0))

    # OT idle (only on active days; idle on inactive days doesn't count)
    ot_active_days: dict[int, set[int]] = {}
    for (ot_id, day), used in sched.ot_hours_used.items():
        if used > 0:
            ot_active_days.setdefault(ot_id, set()).add(day)

    ots_by_id = {o.id: o for o in instance.ots}
    ot_idle_sq = 0.0
    for ot_id, days in ot_active_days.items():
        ot = ots_by_id[ot_id]
        for d in days:
            used = sched.ot_hours_used.get((ot_id, d), 0.0)
            idle = ot.daily_hours - used
            ot_idle_sq += idle * idle

    # ICU and ward overflow
    icu_overflow_sq = 0.0
    ward_overflow_sq = 0.0
    for d in range(1, instance.horizon_days + 1):
        icu_o = max(0, sched.icu_occupancy.get(d, 0) - instance.icu_capacity)
        ward_o = max(0, sched.ward_occupancy.get(d, 0) - instance.ward_capacity)
        icu_overflow_sq += icu_o * icu_o
        ward_overflow_sq += ward_o * ward_o

    return ot_idle_w * ot_idle_sq + icu_w * icu_overflow_sq + ward_w * ward_overflow_sq


def _f3_waiting_fairness(instance: Instance, sched: Schedule) -> float:
    """Variance of priority-weighted average waiting time across priority classes.

    Waiting time for a scheduled surgery = (assigned_day - earliest_day).
    Unscheduled surgeries are assigned waiting time = horizon for fairness purposes.
    A class with no surgeries contributes nothing.
    """
    sched_map = {a.surgery_id: a for a in sched.assignments}
    class_wait_sums: dict[int, float] = {}
    class_counts: dict[int, int] = {}
    for s in instance.surgeries:
        if s.id in sched_map:
            w = sched_map[s.id].day - s.earliest_day
        else:
            w = instance.horizon_days
        class_wait_sums[s.priority] = class_wait_sums.get(s.priority, 0.0) + float(w)
        class_counts[s.priority] = class_counts.get(s.priority, 0) + 1

    if not class_counts:
        return 0.0

    class_means = [
        class_wait_sums[c] / class_counts[c]
        for c in class_counts
    ]
    if len(class_means) < 2:
        return 0.0
    return float(np.var(class_means))
