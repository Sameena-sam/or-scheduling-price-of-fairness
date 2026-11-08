"""Tests for the GreedyDecoder.

Validates that decoded schedules respect all hard constraints declared by
the instance: OT specialty/equipment, team specialty, daily hour caps,
consecutive-day limits, and ICU/ward capacities.
"""
from pathlib import Path

import numpy as np
import pytest

from cio.problem import Instance, GreedyDecoder

INSTANCES_DIR = Path(__file__).resolve().parent.parent / "instances"
SIZES = ["small", "medium", "large"]


def _decoded(size: str, seed: int = 0):
    inst = Instance.from_json(INSTANCES_DIR / f"ot_{size}.json")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(inst.n_surgeries)
    dec = GreedyDecoder()
    sched = dec(inst, perm)
    return inst, sched


@pytest.mark.parametrize("size", SIZES)
def test_decoder_assigns_subset(size):
    """Every assignment refers to a real surgery and every scheduled +
    unscheduled surgery accounts for every surgery in the instance."""
    inst, sched = _decoded(size)
    assigned_ids = {a.surgery_id for a in sched.assignments}
    unassigned_ids = set(sched.unscheduled_surgery_ids)
    all_ids = {s.id for s in inst.surgeries}
    assert assigned_ids.isdisjoint(unassigned_ids)
    assert assigned_ids | unassigned_ids == all_ids


@pytest.mark.parametrize("size", SIZES)
def test_ot_specialty_and_equipment(size):
    """Every assignment must use an OT that permits the surgery's specialty
    and has the required equipment."""
    inst, sched = _decoded(size)
    ots_by_id = {o.id: o for o in inst.ots}
    surgeries_by_id = {s.id: s for s in inst.surgeries}
    for a in sched.assignments:
        ot = ots_by_id[a.ot_id]
        s = surgeries_by_id[a.surgery_id]
        assert s.specialty in ot.permitted_specialties
        assert s.required_equipment.issubset(ot.equipment)


@pytest.mark.parametrize("size", SIZES)
def test_team_specialty(size):
    inst, sched = _decoded(size)
    teams_by_id = {t.id: t for t in inst.teams}
    surgeries_by_id = {s.id: s for s in inst.surgeries}
    for a in sched.assignments:
        team = teams_by_id[a.team_id]
        s = surgeries_by_id[a.surgery_id]
        assert team.can_do(s.specialty)


@pytest.mark.parametrize("size", SIZES)
def test_ot_daily_hours_not_exceeded(size):
    inst, sched = _decoded(size)
    ots_by_id = {o.id: o for o in inst.ots}
    # Sum durations by (ot, day) from assignments and compare
    by_ot_day = {}
    for a in sched.assignments:
        by_ot_day.setdefault((a.ot_id, a.day), 0.0)
        by_ot_day[(a.ot_id, a.day)] += a.duration_hr
    for (ot_id, day), used in by_ot_day.items():
        ot = ots_by_id[ot_id]
        assert used <= ot.daily_hours + 1e-9, (
            f"OT {ot_id} day {day}: used {used} > capacity {ot.daily_hours}"
        )


@pytest.mark.parametrize("size", SIZES)
def test_team_daily_hours_not_exceeded(size):
    inst, sched = _decoded(size)
    teams_by_id = {t.id: t for t in inst.teams}
    by_team_day = {}
    for a in sched.assignments:
        by_team_day.setdefault((a.team_id, a.day), 0.0)
        by_team_day[(a.team_id, a.day)] += a.duration_hr
    for (team_id, day), used in by_team_day.items():
        team = teams_by_id[team_id]
        assert used <= team.daily_hours_max + 1e-9


@pytest.mark.parametrize("size", SIZES)
def test_icu_and_ward_capacity(size):
    """ICU and ward occupancy must not exceed capacity on any day when
    overflow_allowed=False."""
    inst, sched = _decoded(size)
    if inst.overflow_allowed:
        pytest.skip("overflow allowed for this instance")
    surgeries_by_id = {s.id: s for s in inst.surgeries}
    icu = {d: 0 for d in range(1, inst.horizon_days + 1)}
    ward = {d: 0 for d in range(1, inst.horizon_days + 1)}
    for a in sched.assignments:
        s = surgeries_by_id[a.surgery_id]
        for d in range(a.day, a.day + s.icu_days):
            if d <= inst.horizon_days:
                icu[d] += 1
        ward_start = a.day + s.icu_days
        for d in range(ward_start, ward_start + s.ward_days):
            if d <= inst.horizon_days:
                ward[d] += 1
    for d, occ in icu.items():
        assert occ <= inst.icu_capacity, f"ICU overflow day {d}: {occ}"
    for d, occ in ward.items():
        assert occ <= inst.ward_capacity, f"Ward overflow day {d}: {occ}"


@pytest.mark.parametrize("size", SIZES)
def test_team_consecutive_days(size):
    inst, sched = _decoded(size)
    teams_by_id = {t.id: t for t in inst.teams}
    team_days: dict[int, set[int]] = {}
    for a in sched.assignments:
        team_days.setdefault(a.team_id, set()).add(a.day)
    for team_id, days in team_days.items():
        team = teams_by_id[team_id]
        # Compute longest consecutive run in the set of days
        sorted_days = sorted(days)
        longest = current = 1
        for i in range(1, len(sorted_days)):
            if sorted_days[i] == sorted_days[i - 1] + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        assert longest <= team.max_consecutive_days, (
            f"team {team_id}: consecutive run {longest} > "
            f"limit {team.max_consecutive_days}"
        )


def test_decoder_deterministic():
    """Same permutation -> same schedule (objective values)."""
    from cio.problem.evaluator import evaluate
    inst = Instance.from_json(INSTANCES_DIR / "ot_small.json")
    rng = np.random.default_rng(42)
    perm = rng.permutation(inst.n_surgeries)
    dec = GreedyDecoder()
    f1 = evaluate(inst, dec(inst, perm))
    f2 = evaluate(inst, dec(inst, perm))
    assert f1 == f2
