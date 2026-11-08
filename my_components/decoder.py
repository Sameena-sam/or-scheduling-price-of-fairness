"""Repair-based constraint-handling decoder.

Design (student-authored): candidate schedules are built permissively --
ICU/ward daily capacity is *not* enforced during initial placement, so a
surgery is only rejected during that pass for OT-hours, team-hours,
consecutive-day, or specialty/equipment reasons (all still hard-enforced,
same as the shipped GreedyDecoder). Downstream ICU/ward overflow is
detected only after the full permutation has been placed, then repaired by
evicting surgeries from over-capacity days (via `_select_surgery_to_evict`,
below) and re-inserting each evicted surgery at the earliest later day that
is *strictly* feasible (including ICU/ward capacity).

This is a genuine repair-based constraint-handling mechanism per Section
4.2 of the brief -- as opposed to the shipped GreedyDecoder's
feasibility-by-construction approach, which can only ever avoid overflow by
leaving a surgery unscheduled outright rather than reshuffling the schedule
to make room for it.

Fill in `_select_surgery_to_evict` with your own eviction-selection rule --
that is the actual design decision. Everything else here is shared
plumbing, deliberately written to mirror the shipped GreedyDecoder's
day-ordering behaviour (earliest-day-first, within-max-wait-window
preferred) so that a comparison between this decoder and the baseline
isolates your repair mechanism, not incidental behavioural drift.
"""
from __future__ import annotations

import numpy as np

from cio.problem.instance import Instance
from cio.problem.schedule import Assignment, Schedule


class RepairBasedDecoder:
    """Allow-then-repair decoder: permissive ICU/ward placement followed by
    an explicit repair pass that resolves overflow by eviction + reinsertion.
    """

    def __init__(self, max_repair_passes: int | None = None,
                 strict: bool = True):
        # Safety bound so repair can't loop forever. Default is generous
        # (2x horizon + number of surgeries) -- should never be hit in
        # practice unless occupancy bookkeeping has a bug.
        self.max_repair_passes = max_repair_passes
        # If True, raise when repair exits with residual overflow instead of
        # silently returning an infeasible schedule. Keep True: a decoder
        # that quietly emits infeasible schedules would invalidate every
        # feasibility claim in the paper.
        self.strict = strict
        # Diagnostics, useful for the paper's constraint-handling analysis.
        self.last_repair_passes = 0
        self.last_evictions = 0
        self.last_evictions_unplaceable = 0

    def __call__(self, instance: Instance, permutation: np.ndarray) -> Schedule:
        sched = self._initial_pass(instance, permutation)
        self._repair(instance, sched)
        self._compact_start_hours(instance, sched)
        residual = self._find_first_overflow(instance, sched)
        if residual is not None and self.strict:
            day, resource = residual
            raise RuntimeError(
                f"RepairBasedDecoder exhausted its repair budget with residual "
                f"{resource.upper()} overflow on day {day} "
                f"(passes={self.last_repair_passes}). The returned schedule "
                f"would be infeasible; refusing to return it silently."
            )
        return sched

    # ------------------------------------------------------------------
    # Pass 1: permissive placement (ICU/ward capacity NOT enforced here)
    # ------------------------------------------------------------------
    def _initial_pass(self, instance: Instance, permutation: np.ndarray) -> Schedule:
        sched = Schedule()
        for day in range(1, instance.horizon_days + 1):
            sched.icu_occupancy[day] = 0
            sched.ward_occupancy[day] = 0

        ots_by_id = {o.id: o for o in instance.ots}
        teams_by_id = {t.id: t for t in instance.teams}

        for idx in permutation:
            surgery = instance.surgery_by_index(int(idx))
            assignment = self._try_place(
                instance, surgery, sched, ots_by_id, teams_by_id,
                enforce_downstream_capacity=False,
            )
            if assignment is None:
                sched.unscheduled_surgery_ids.append(surgery.id)
            else:
                self._apply_assignment(instance, assignment, surgery, sched)
        return sched

    # ------------------------------------------------------------------
    # Pass 2: repair overflow
    # ------------------------------------------------------------------
    def _repair(self, instance: Instance, sched: Schedule) -> None:
        # Every pass either resolves one bed-day of overflow or makes one
        # surgery unscheduled, so the number of passes is bounded by the
        # total bed-days that could be over capacity plus one eviction per
        # scheduled surgery.
        max_passes = self.max_repair_passes or (
            4 * instance.horizon_days + 2 * len(sched.assignments) + 16
        )
        surgeries_by_id = {s.id: s for s in instance.surgeries}
        ots_by_id = {o.id: o for o in instance.ots}
        teams_by_id = {t.id: t for t in instance.teams}

        self.last_repair_passes = 0
        self.last_evictions = 0
        self.last_evictions_unplaceable = 0

        for _ in range(max_passes):
            self.last_repair_passes += 1
            overflow = self._find_first_overflow(instance, sched)
            if overflow is None:
                break
            day, resource = overflow

            candidates = self._occupants(sched, surgeries_by_id, day, resource)
            if not candidates:
                # Bookkeeping should make this unreachable; guard anyway.
                break

            evict_id = self._select_surgery_to_evict(instance, sched, candidates, day, resource)
            evicted_surgery = surgeries_by_id[evict_id]
            self._evict(instance, sched, evicted_surgery)
            self.last_evictions += 1

            # Re-insert at the earliest later day that is *strictly*
            # feasible (including ICU/ward capacity), so we don't just
            # recreate the same overflow immediately.
            reassignment = self._try_place(
                instance, evicted_surgery, sched, ots_by_id, teams_by_id,
                enforce_downstream_capacity=True,
                earliest_search_day=day + 1,
            )
            if reassignment is not None:
                self._apply_assignment(instance, reassignment, evicted_surgery, sched)
            else:
                sched.unscheduled_surgery_ids.append(evicted_surgery.id)
                self.last_evictions_unplaceable += 1

    def _compact_start_hours(self, instance: Instance, sched: Schedule) -> None:
        """Re-pack start times within each (OT, day).

        `start_hour` is assigned when a surgery is placed; an eviction later
        in the repair pass can leave a hole, so a printed schedule would show
        a theatre idle at the start of the day with cases running after it.
        Objectives are unaffected (the evaluator sums hours), but brief 8.1
        requires the schedule itself to be emitted, so it must be coherent.
        """
        from collections import defaultdict
        by_slot: dict[tuple[int, int], list] = defaultdict(list)
        for a in sched.assignments:
            by_slot[(a.ot_id, a.day)].append(a)
        for _slot, group in by_slot.items():
            group.sort(key=lambda a: (a.start_hour, a.surgery_id))
            cursor = 0.0
            for a in group:
                a.start_hour = cursor
                cursor += a.duration_hr

    # ------------------------------------------------------------------
    # >>> YOUR DESIGN DECISION GOES HERE <<<
    # ------------------------------------------------------------------
    def _select_surgery_to_evict(
        self,
        instance: Instance,
        sched: Schedule,
        candidates: list,
        day: int,
        resource: str,
    ) -> int:
        """Return the surgery_id to evict from `candidates` (a list of
        Surgery objects occupying `day` for `resource`, resource is 'icu'
        or 'ward') in order to resolve the overflow on that day.

        This is the core of your constraint-handling design. Implement
        whichever rule you've decided on and can justify -- e.g.:

          * lowest priority first:
                return min(candidates, key=lambda s: s.priority).id
          * shortest remaining footprint first (minimal disruption):
                return min(candidates, key=lambda s: s.icu_days + s.ward_days).id
          * the surgery causing the overflow itself (last one placed among
            the candidates) -- you'd need to track placement order
            separately to identify this precisely.

        Replace the line below with your chosen rule.
        """
        # Design rule (student-chosen): evict the lowest-priority surgery
        # among the occupants of the overflowing day. This protects
        # high-priority patients from being displaced by scheduling
        # decisions elsewhere -- the surgery isn't cancelled, it's handed
        # to `_try_place` immediately afterwards, which searches forward
        # for the earliest day that is genuinely feasible (OT hours, team
        # hours, consecutive-day limit, and ICU/ward capacity all
        # re-checked). It only ends up unscheduled if no such day exists
        # before the planning horizon ends.
        return min(candidates, key=lambda s: s.priority).id

    # ------------------------------------------------------------------
    # Mechanical helpers below -- shared plumbing, not the design itself
    # ------------------------------------------------------------------
    def _find_first_overflow(self, instance: Instance, sched: Schedule) -> tuple[int, str] | None:
        for d in range(1, instance.horizon_days + 1):
            if sched.icu_occupancy.get(d, 0) > instance.icu_capacity:
                return d, "icu"
            if sched.ward_occupancy.get(d, 0) > instance.ward_capacity:
                return d, "ward"
        return None

    def _occupants(self, sched: Schedule, surgeries_by_id: dict, day: int, resource: str) -> list:
        occupants = []
        for a in sched.assignments:
            s = surgeries_by_id[a.surgery_id]
            if resource == "icu":
                start, span = a.day, s.icu_days
            else:
                start, span = a.day + s.icu_days, s.ward_days
            if start <= day < start + span:
                occupants.append(s)
        return occupants

    def _evict(self, instance: Instance, sched: Schedule, surgery) -> None:
        a = sched.assignment_for(surgery.id)
        if a is None:
            return
        sched.assignments.remove(a)
        sched.ot_hours_used[(a.ot_id, a.day)] = max(
            0.0, sched.ot_hours_used.get((a.ot_id, a.day), 0.0) - a.duration_hr
        )
        sched.team_hours_used[(a.team_id, a.day)] = max(
            0.0, sched.team_hours_used.get((a.team_id, a.day), 0.0) - a.duration_hr
        )
        if a.team_id in sched.team_active_days and a.day in sched.team_active_days[a.team_id]:
            sched.team_active_days[a.team_id].remove(a.day)
        for d in range(a.day, a.day + surgery.icu_days):
            if d > instance.horizon_days:
                break
            sched.icu_occupancy[d] = max(0, sched.icu_occupancy.get(d, 0) - 1)
        ward_start = a.day + surgery.icu_days
        for d in range(ward_start, ward_start + surgery.ward_days):
            if d > instance.horizon_days:
                break
            sched.ward_occupancy[d] = max(0, sched.ward_occupancy.get(d, 0) - 1)

    def _try_place(
        self,
        instance: Instance,
        surgery,
        sched: Schedule,
        ots_by_id: dict,
        teams_by_id: dict,
        enforce_downstream_capacity: bool,
        earliest_search_day: int | None = None,
    ) -> Assignment | None:
        eligible_ot_ids = instance.eligible_ots_for_surgery[surgery.id]
        eligible_team_ids = instance.eligible_teams_for_surgery[surgery.id]
        if not eligible_ot_ids or not eligible_team_ids:
            return None

        start_from = max(surgery.earliest_day, earliest_search_day or surgery.earliest_day)
        latest_no_penalty_day = surgery.earliest_day + surgery.max_wait_days
        within_window = list(range(start_from, min(instance.horizon_days, latest_no_penalty_day) + 1))
        beyond_window = list(range(max(start_from, latest_no_penalty_day + 1), instance.horizon_days + 1))
        day_order = within_window + beyond_window

        for day in day_order:
            if enforce_downstream_capacity and not self._post_op_ok(instance, surgery, day, sched):
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
                    if not self._consecutive_days_ok(team, day, sched):
                        continue
                    return Assignment(
                        surgery_id=surgery.id, ot_id=ot_id, day=day,
                        team_id=team_id, start_hour=used, duration_hr=surgery.duration_hr,
                    )
        return None

    def _post_op_ok(self, instance: Instance, surgery, day: int, sched: Schedule) -> bool:
        for d in range(day, day + surgery.icu_days):
            if d > instance.horizon_days:
                break
            if sched.icu_occupancy.get(d, 0) + 1 > instance.icu_capacity:
                return False
        ward_start = day + surgery.icu_days
        for d in range(ward_start, ward_start + surgery.ward_days):
            if d > instance.horizon_days:
                break
            if sched.ward_occupancy.get(d, 0) + 1 > instance.ward_capacity:
                return False
        return True

    def _consecutive_days_ok(self, team, day: int, sched: Schedule) -> bool:
        active = sched.team_active_days.get(team.id, [])
        candidate = sorted(set(active) | {day})
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

    def _apply_assignment(self, instance: Instance, assignment: Assignment, surgery, sched: Schedule) -> None:
        sched.assignments.append(assignment)
        sched.ot_hours_used[(assignment.ot_id, assignment.day)] = (
            sched.ot_hours_used.get((assignment.ot_id, assignment.day), 0.0) + assignment.duration_hr
        )
        sched.team_hours_used[(assignment.team_id, assignment.day)] = (
            sched.team_hours_used.get((assignment.team_id, assignment.day), 0.0) + assignment.duration_hr
        )
        sched.team_active_days.setdefault(assignment.team_id, []).append(assignment.day)
        for d in range(assignment.day, assignment.day + surgery.icu_days):
            if d > instance.horizon_days:
                break
            sched.icu_occupancy[d] = sched.icu_occupancy.get(d, 0) + 1
        ward_start = assignment.day + surgery.icu_days
        for d in range(ward_start, ward_start + surgery.ward_days):
            if d > instance.horizon_days:
                break
            sched.ward_occupancy[d] = sched.ward_occupancy.get(d, 0) + 1
