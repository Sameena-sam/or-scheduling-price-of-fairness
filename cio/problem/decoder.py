"""Decoder: permutation -> Schedule.

The default decoder is greedy first-fit. Surgeries are processed in the order
specified by the permutation; for each surgery, the decoder tries to find a
feasible (day, OT, team) triple and assigns the surgery to it if found.

The decoder is pluggable. Students who wish to design alternative decoders
(e.g., look-ahead, urgency-aware re-ordering, or partial-infeasibility
allowing decoders) can subclass `BaseDecoder` and pass their decoder to the
problem class.

Constraints handled in-decoder (always feasible by construction):
- OT daily hours
- OT specialty and equipment matching
- Team specialty matching
- Team daily hours
- Team consecutive-day limit
- ICU and ward daily capacities (when overflow_allowed=False)
- earliest_day respected

If `overflow_allowed=True`, the decoder permits ICU/ward overflow but the
overflow is captured by the resource-imbalance objective.
"""
from __future__ import annotations

import numpy as np
from typing import Protocol

from .instance import Instance
from .schedule import Assignment, Schedule


class BaseDecoder(Protocol):
    def __call__(self, instance: Instance, permutation: np.ndarray) -> Schedule:
        ...


class GreedyDecoder:
    """Default greedy first-fit decoder.

    Args:
        instance: the problem instance
        permutation: a 1-D numpy array of surgery indices (0-based, length =
            n_surgeries) giving the order in which surgeries are placed
    """

    def __call__(self, instance: Instance, permutation: np.ndarray) -> Schedule:
        sched = Schedule()
        # Initialise per-day occupancy tracking
        for day in range(1, instance.horizon_days + 1):
            sched.icu_occupancy[day] = 0
            sched.ward_occupancy[day] = 0

        # Iterate surgeries in permutation order
        for idx in permutation:
            surgery = instance.surgery_by_index(int(idx))
            assignment = self._try_place(instance, surgery, sched)
            if assignment is None:
                sched.unscheduled_surgery_ids.append(surgery.id)
            else:
                self._apply_assignment(instance, assignment, surgery, sched)
        return sched

    def _try_place(
        self,
        instance: Instance,
        surgery,
        sched: Schedule,
    ) -> Assignment | None:
        eligible_ot_ids = instance.eligible_ots_for_surgery[surgery.id]
        eligible_team_ids = instance.eligible_teams_for_surgery[surgery.id]
        if not eligible_ot_ids or not eligible_team_ids:
            return None

        ots_by_id = {o.id: o for o in instance.ots}
        teams_by_id = {t.id: t for t in instance.teams}

        latest_day = min(
            instance.horizon_days,
            surgery.earliest_day + surgery.max_wait_days,
        )

        # Try day-by-day starting from earliest_day. We allow scheduling past
        # the max wait (up to horizon) but it incurs penalty in the throughput
        # objective; first try within the no-penalty window.
        day_order = list(range(surgery.earliest_day, latest_day + 1))
        beyond_window = list(range(latest_day + 1, instance.horizon_days + 1))
        day_order.extend(beyond_window)

        for day in day_order:
            # Check downstream resource availability for the surgery's post-op stays
            if not self._post_op_resources_ok(instance, surgery, day, sched):
                continue
            for ot_id in eligible_ot_ids:
                ot = ots_by_id[ot_id]
                used = sched.ot_hours_used.get((ot_id, day), 0.0)
                if used + surgery.duration_hr > ot.daily_hours + 1e-9:
                    continue
                for team_id in eligible_team_ids:
                    team = teams_by_id[team_id]
                    used_t = sched.team_hours_used.get((team_id, day), 0.0)
                    if used_t + surgery.duration_hr > team.daily_hours_max + 1e-9:
                        continue
                    # Check team consecutive-day limit
                    if not self._consecutive_days_ok(team, day, sched):
                        continue
                    return Assignment(
                        surgery_id=surgery.id,
                        ot_id=ot_id,
                        day=day,
                        team_id=team_id,
                        start_hour=used,
                        duration_hr=surgery.duration_hr,
                    )
        return None

    def _post_op_resources_ok(
        self,
        instance: Instance,
        surgery,
        scheduled_day: int,
        sched: Schedule,
    ) -> bool:
        # ICU days follow the surgery day; then ward days follow ICU.
        for d in range(scheduled_day, scheduled_day + surgery.icu_days):
            if d > instance.horizon_days:
                break
            occ = sched.icu_occupancy.get(d, 0)
            if not instance.overflow_allowed and occ + 1 > instance.icu_capacity:
                return False
        ward_start = scheduled_day + surgery.icu_days
        for d in range(ward_start, ward_start + surgery.ward_days):
            if d > instance.horizon_days:
                break
            occ = sched.ward_occupancy.get(d, 0)
            if not instance.overflow_allowed and occ + 1 > instance.ward_capacity:
                return False
        return True

    def _consecutive_days_ok(self, team, day: int, sched: Schedule) -> bool:
        active = sched.team_active_days.get(team.id, [])
        # Find longest run ending at or before `day` that would extend if day added
        # Simple approach: collect all consecutive sequences including `day`,
        # check the longest one's length doesn't exceed the limit.
        candidate = sorted(set(active) | {day})
        # Find the run containing `day`
        run_length = 1
        prev = day
        for d in reversed(candidate):
            if d >= day:
                continue
            if d == prev - 1:
                run_length += 1
                prev = d
            else:
                break
        prev = day
        for d in candidate:
            if d <= day:
                continue
            if d == prev + 1:
                run_length += 1
                prev = d
            else:
                break
        return run_length <= team.max_consecutive_days

    def _apply_assignment(
        self,
        instance: Instance,
        assignment: Assignment,
        surgery,
        sched: Schedule,
    ) -> None:
        sched.assignments.append(assignment)
        # Update OT and team hour tracking
        sched.ot_hours_used[(assignment.ot_id, assignment.day)] = (
            sched.ot_hours_used.get((assignment.ot_id, assignment.day), 0.0)
            + assignment.duration_hr
        )
        sched.team_hours_used[(assignment.team_id, assignment.day)] = (
            sched.team_hours_used.get((assignment.team_id, assignment.day), 0.0)
            + assignment.duration_hr
        )
        # Track team active days
        sched.team_active_days.setdefault(assignment.team_id, []).append(assignment.day)
        # Update downstream resources
        for d in range(assignment.day, assignment.day + surgery.icu_days):
            if d > instance.horizon_days:
                break
            sched.icu_occupancy[d] = sched.icu_occupancy.get(d, 0) + 1
        ward_start = assignment.day + surgery.icu_days
        for d in range(ward_start, ward_start + surgery.ward_days):
            if d > instance.horizon_days:
                break
            sched.ward_occupancy[d] = sched.ward_occupancy.get(d, 0) + 1
