"""Typed access to a loaded OT scheduling instance.

Students load an instance via:

    from cio.problem import Instance
    inst = Instance.from_json("instances/ot_small.json")

All entities are exposed as plain dataclasses. The Instance object also
caches a few derived structures (eligibility tables) that speed up the
decoder. It is read-only after construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OT:
    id: int
    permitted_specialties: tuple[str, ...]
    equipment: frozenset[str]
    daily_hours: float
    daily_activation_cost: float


@dataclass(frozen=True)
class Team:
    id: int
    primary_specialty: str
    secondary_specialties: tuple[str, ...]
    daily_hours_max: float
    max_consecutive_days: int
    daily_cost: float

    def can_do(self, specialty: str) -> bool:
        return specialty == self.primary_specialty or specialty in self.secondary_specialties


@dataclass(frozen=True)
class SurgeryType:
    name: str
    specialty: str
    duration_hr: float
    icu_days: int
    ward_days: int
    equipment: frozenset[str]


@dataclass(frozen=True)
class Surgery:
    id: int
    type: str
    specialty: str
    duration_hr: float
    icu_days: int
    ward_days: int
    required_equipment: frozenset[str]
    priority: int
    earliest_day: int
    max_wait_days: int


@dataclass
class Instance:
    name: str
    horizon_days: int
    icu_capacity: int
    ward_capacity: int
    overflow_allowed: bool
    objective_weights: dict[str, float]
    ots: tuple[OT, ...]
    teams: tuple[Team, ...]
    surgery_types: dict[str, SurgeryType]
    surgeries: tuple[Surgery, ...]
    seed: int

    # Derived eligibility tables (cached for decoder speed)
    eligible_ots_for_surgery: dict[int, tuple[int, ...]] = field(default_factory=dict)
    eligible_teams_for_surgery: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "Instance":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        gp = data["global_parameters"]
        ots = tuple(
            OT(
                id=o["id"],
                permitted_specialties=tuple(o["permitted_specialties"]),
                equipment=frozenset(o["equipment"]),
                daily_hours=float(o["daily_hours"]),
                daily_activation_cost=float(o["daily_activation_cost"]),
            )
            for o in data["ots"]
        )
        teams = tuple(
            Team(
                id=t["id"],
                primary_specialty=t["primary_specialty"],
                secondary_specialties=tuple(t["secondary_specialties"]),
                daily_hours_max=float(t["daily_hours_max"]),
                max_consecutive_days=int(t["max_consecutive_days"]),
                daily_cost=float(t["daily_cost"]),
            )
            for t in data["teams"]
        )
        surgery_types = {
            st["name"]: SurgeryType(
                name=st["name"],
                specialty=st["specialty"],
                duration_hr=float(st["duration_hr"]),
                icu_days=int(st["icu_days"]),
                ward_days=int(st["ward_days"]),
                equipment=frozenset(st["equipment"]),
            )
            for st in data["surgery_types"]
        }
        surgeries = tuple(
            Surgery(
                id=s["id"],
                type=s["type"],
                specialty=s["specialty"],
                duration_hr=float(s["duration_hr"]),
                icu_days=int(s["icu_days"]),
                ward_days=int(s["ward_days"]),
                required_equipment=frozenset(s["required_equipment"]),
                priority=int(s["priority"]),
                earliest_day=int(s["earliest_day"]),
                max_wait_days=int(s["max_wait_days"]),
            )
            for s in data["surgeries"]
        )

        inst = cls(
            name=data["name"],
            horizon_days=int(data["horizon_days"]),
            icu_capacity=int(gp["icu_capacity"]),
            ward_capacity=int(gp["ward_capacity"]),
            overflow_allowed=bool(gp["overflow_allowed"]),
            objective_weights=dict(gp["objective_weights"]),
            ots=ots,
            teams=teams,
            surgery_types=surgery_types,
            surgeries=surgeries,
            seed=int(gp["seed"]),
        )
        inst._build_eligibility_tables()
        return inst

    def _build_eligibility_tables(self) -> None:
        ots_by_id = {o.id: o for o in self.ots}
        teams_by_id = {t.id: t for t in self.teams}
        for s in self.surgeries:
            eligible_ots = tuple(
                o.id for o in self.ots
                if s.specialty in o.permitted_specialties
                and s.required_equipment.issubset(o.equipment)
            )
            eligible_teams = tuple(
                t.id for t in self.teams
                if t.can_do(s.specialty)
            )
            self.eligible_ots_for_surgery[s.id] = eligible_ots
            self.eligible_teams_for_surgery[s.id] = eligible_teams

    @property
    def n_surgeries(self) -> int:
        return len(self.surgeries)

    @property
    def n_ots(self) -> int:
        return len(self.ots)

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    def surgery_by_index(self, idx: int) -> Surgery:
        """0-indexed access for use inside operators."""
        return self.surgeries[idx]

    def __repr__(self) -> str:
        return (
            f"Instance(name={self.name!r}, horizon={self.horizon_days}, "
            f"n_surgeries={self.n_surgeries}, n_ots={self.n_ots}, "
            f"n_teams={self.n_teams})"
        )
