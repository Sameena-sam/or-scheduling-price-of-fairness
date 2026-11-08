"""Schedule representation.

A Schedule is the decoded form of a candidate solution: a concrete assignment
of (some) surgeries to (OT, day, team) slots. Surgeries with no feasible slot
in the greedy decoder remain unscheduled and contribute a heavy penalty to
the throughput objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Assignment:
    surgery_id: int
    ot_id: int
    day: int            # 1-indexed
    team_id: int
    start_hour: float   # Start time within the day, 0.0 to ot.daily_hours
    duration_hr: float


@dataclass
class Schedule:
    """Decoded schedule plus derived metadata."""
    assignments: list[Assignment] = field(default_factory=list)
    unscheduled_surgery_ids: list[int] = field(default_factory=list)

    # Daily resource usage tables, populated by the decoder.
    # Keys are (resource_id, day); the day is 1-indexed.
    ot_hours_used: dict[tuple[int, int], float] = field(default_factory=dict)
    team_hours_used: dict[tuple[int, int], float] = field(default_factory=dict)
    icu_occupancy: dict[int, int] = field(default_factory=dict)
    ward_occupancy: dict[int, int] = field(default_factory=dict)
    team_active_days: dict[int, list[int]] = field(default_factory=dict)

    @property
    def n_scheduled(self) -> int:
        return len(self.assignments)

    @property
    def n_unscheduled(self) -> int:
        return len(self.unscheduled_surgery_ids)

    def assignment_for(self, surgery_id: int) -> Assignment | None:
        for a in self.assignments:
            if a.surgery_id == surgery_id:
                return a
        return None
