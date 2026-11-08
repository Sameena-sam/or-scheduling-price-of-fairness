"""
Instance generator for the OT scheduling problem.

Produces JSON instances at three public sizes (small / medium / large) plus
one hidden instance for examination. The generator is parametric and seeded
so instances are fully reproducible.

Usage:
    python generate_instances.py --out instances/

This is part of the skeleton distributed to students. The hidden instance
is generated to instances/private/ and should be moved out of the student
distribution before release.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Catalogue: surgical specialties and equipment
# ---------------------------------------------------------------------------

SPECIALTIES = [
    "general_surgery",
    "orthopaedics",
    "cardiothoracic",
    "neurosurgery",
    "obs_gynae",
]

EQUIPMENT = [
    "robotic_arm",         # for selected ortho/general
    "c_arm",               # imaging, common
    "neurology_microscope",
    "perfusion_machine",   # cardio
    "laparoscopy_tower",   # general
]

# Surgery types (catalogue). Each entry: specialty, base duration (hrs),
# ICU stay (days, point estimate), ward stay (days), required equipment.
SURGERY_TYPE_CATALOGUE = [
    # general_surgery
    {"name": "lap_chole",          "specialty": "general_surgery", "duration_hr": 1.5, "icu_days": 0, "ward_days": 2, "equipment": ["laparoscopy_tower"]},
    {"name": "hernia_repair",      "specialty": "general_surgery", "duration_hr": 1.0, "icu_days": 0, "ward_days": 1, "equipment": []},
    {"name": "colectomy",          "specialty": "general_surgery", "duration_hr": 3.5, "icu_days": 1, "ward_days": 5, "equipment": ["laparoscopy_tower"]},
    # orthopaedics
    {"name": "hip_replacement",    "specialty": "orthopaedics",    "duration_hr": 2.5, "icu_days": 0, "ward_days": 4, "equipment": ["c_arm"]},
    {"name": "knee_replacement",   "specialty": "orthopaedics",    "duration_hr": 2.0, "icu_days": 0, "ward_days": 3, "equipment": ["c_arm"]},
    {"name": "spinal_fusion",      "specialty": "orthopaedics",    "duration_hr": 4.0, "icu_days": 1, "ward_days": 5, "equipment": ["c_arm", "neurology_microscope"]},
    # cardiothoracic
    {"name": "cabg",               "specialty": "cardiothoracic",  "duration_hr": 4.5, "icu_days": 2, "ward_days": 6, "equipment": ["perfusion_machine"]},
    {"name": "valve_replacement",  "specialty": "cardiothoracic",  "duration_hr": 4.0, "icu_days": 2, "ward_days": 6, "equipment": ["perfusion_machine"]},
    # neurosurgery
    {"name": "craniotomy",         "specialty": "neurosurgery",    "duration_hr": 5.0, "icu_days": 2, "ward_days": 7, "equipment": ["neurology_microscope"]},
    {"name": "lumbar_disc",        "specialty": "neurosurgery",    "duration_hr": 2.5, "icu_days": 0, "ward_days": 3, "equipment": ["neurology_microscope"]},
    # obs_gynae
    {"name": "elective_cs",        "specialty": "obs_gynae",       "duration_hr": 1.5, "icu_days": 0, "ward_days": 3, "equipment": []},
    {"name": "hysterectomy",       "specialty": "obs_gynae",       "duration_hr": 2.5, "icu_days": 0, "ward_days": 3, "equipment": ["laparoscopy_tower"]},
]


# ---------------------------------------------------------------------------
# Configurations for the three public sizes plus the hidden instance
# ---------------------------------------------------------------------------

@dataclass
class InstanceConfig:
    name: str
    horizon_days: int
    n_surgeries: int
    n_ots: int
    n_teams: int
    specialties_used: list[str]
    icu_capacity: int
    ward_capacity: int
    ot_daily_hours: float
    team_daily_hours_max: float
    team_max_consecutive_days: int
    overflow_allowed: bool
    seed: int


CONFIGS = {
    "small": InstanceConfig(
        name="small",
        horizon_days=7,
        n_surgeries=22,
        n_ots=2,
        n_teams=3,
        specialties_used=["general_surgery"],
        icu_capacity=1,
        ward_capacity=5,
        ot_daily_hours=8.0,
        team_daily_hours_max=8.0,
        team_max_consecutive_days=5,
        overflow_allowed=False,
        seed=20260201,
    ),
    "medium": InstanceConfig(
        name="medium",
        horizon_days=28,
        n_surgeries=120,
        n_ots=4,
        n_teams=6,
        specialties_used=["general_surgery", "orthopaedics", "obs_gynae"],
        icu_capacity=1,
        ward_capacity=12,
        ot_daily_hours=8.0,
        team_daily_hours_max=8.0,
        team_max_consecutive_days=5,
        overflow_allowed=False,
        seed=20260202,
    ),
    "large": InstanceConfig(
        name="large",
        horizon_days=56,
        n_surgeries=280,
        n_ots=6,
        n_teams=12,
        specialties_used=SPECIALTIES,
        icu_capacity=2,
        ward_capacity=22,
        ot_daily_hours=8.0,
        team_daily_hours_max=8.0,
        team_max_consecutive_days=5,
        overflow_allowed=False,
        seed=20260203,
    ),
    "hidden": InstanceConfig(
        name="hidden",
        horizon_days=42,
        n_surgeries=200,
        n_ots=5,
        n_teams=9,
        specialties_used=["general_surgery", "orthopaedics", "cardiothoracic", "obs_gynae"],
        icu_capacity=1,
        ward_capacity=16,
        ot_daily_hours=8.0,
        team_daily_hours_max=8.0,
        team_max_consecutive_days=5,
        overflow_allowed=False,
        seed=20260204,
    ),
}


# Objective weights (resource utilisation imbalance)
DEFAULT_OBJECTIVE_WEIGHTS = {
    "ot_idle_weight": 1.0,
    "icu_overflow_weight": 10.0,
    "ward_overflow_weight": 5.0,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _filter_surgery_types(specialties_used: list[str]) -> list[dict[str, Any]]:
    return [st for st in SURGERY_TYPE_CATALOGUE if st["specialty"] in specialties_used]


def _generate_ots(
    cfg: InstanceConfig,
    surgery_types: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate OTs, ensuring every surgery type has at least one feasible OT.

    Each OT supports a subset of the used specialties. Its equipment is the
    union of (a) the equipment required by all surgery types whose specialty
    it supports, and (b) a small random set of extras to give variety.
    """
    # Map specialty -> union of equipment required by its surgery types
    spec_required_eq: dict[str, set[str]] = {s: set() for s in cfg.specialties_used}
    for st in surgery_types:
        spec_required_eq[st["specialty"]].update(st["equipment"])

    ots = []
    specs = list(cfg.specialties_used)
    rng.shuffle(specs)
    for i in range(cfg.n_ots):
        # Each OT supports 1-3 specialties from those used in the instance
        n_specs = rng.randint(1, min(3, len(cfg.specialties_used)))
        # Round-robin assignment ensures every specialty has at least one OT
        primary = specs[i % len(specs)]
        candidates = [s for s in cfg.specialties_used if s != primary]
        others = rng.sample(candidates, k=min(n_specs - 1, len(candidates)))
        ot_specs = [primary] + others
        # Required equipment: union over permitted specialties
        required_eq: set[str] = set()
        for s in ot_specs:
            required_eq.update(spec_required_eq[s])
        # Add 0-2 random extras for variety (excluding already-included items)
        extras_pool = [e for e in EQUIPMENT if e not in required_eq]
        n_extras = min(rng.randint(0, 2), len(extras_pool))
        extras = rng.sample(extras_pool, k=n_extras) if n_extras > 0 else []
        equipment = sorted(required_eq | set(extras))
        ots.append({
            "id": i + 1,
            "permitted_specialties": ot_specs,
            "equipment": equipment,
            "daily_hours": cfg.ot_daily_hours,
            "daily_activation_cost": 800 + 200 * len(ot_specs),
        })
    return ots


def _generate_teams(cfg: InstanceConfig, rng: random.Random) -> list[dict[str, Any]]:
    teams = []
    specs = list(cfg.specialties_used)
    rng.shuffle(specs)
    for i in range(cfg.n_teams):
        # Ensure each used specialty has at least one team via round-robin
        primary = specs[i % len(specs)]
        # Secondary specialties: 0-2 others (small chance of cross-specialty teams)
        n_secondaries = rng.choices([0, 1, 2], weights=[6, 3, 1])[0]
        candidates = [s for s in cfg.specialties_used if s != primary]
        secondaries = rng.sample(candidates, k=min(n_secondaries, len(candidates)))
        teams.append({
            "id": i + 1,
            "primary_specialty": primary,
            "secondary_specialties": secondaries,
            "daily_hours_max": cfg.team_daily_hours_max,
            "max_consecutive_days": cfg.team_max_consecutive_days,
            "daily_cost": 1500 + 100 * len(secondaries),
        })
    return teams


def _generate_surgeries(
    cfg: InstanceConfig,
    surgery_types: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    surgeries = []
    # Priority distribution: skewed toward middle, with a tail of urgent
    priority_weights = [1, 3, 5, 3, 2]  # for priority 1..5

    # Max-wait as a fraction of horizon, by priority. Tight enough that
    # ordering choices in the schedule meaningfully affect throughput.
    # (low, high) fraction of horizon
    max_wait_fractions = {
        5: (0.10, 0.25),
        4: (0.15, 0.35),
        3: (0.25, 0.50),
        2: (0.40, 0.70),
        1: (0.60, 1.00),
    }

    H = cfg.horizon_days
    for sid in range(1, cfg.n_surgeries + 1):
        stype = rng.choice(surgery_types)
        priority = rng.choices([1, 2, 3, 4, 5], weights=priority_weights)[0]
        # Earliest day: spread across the horizon but biased toward earlier
        earliest_day = rng.randint(1, max(1, H - 2))
        # Max waiting time scaled to horizon and priority
        lo_frac, hi_frac = max_wait_fractions[priority]
        lo = max(1, int(round(H * lo_frac)))
        hi = max(lo, int(round(H * hi_frac)))
        max_wait = rng.randint(lo, hi)
        # Cap so it doesn't exceed remaining horizon (so lateness *can* occur
        # rather than being trivially impossible)
        max_wait = min(max_wait, max(1, H - earliest_day))
        surgeries.append({
            "id": sid,
            "type": stype["name"],
            "specialty": stype["specialty"],
            "duration_hr": stype["duration_hr"],
            "icu_days": stype["icu_days"],
            "ward_days": stype["ward_days"],
            "required_equipment": list(stype["equipment"]),
            "priority": priority,
            "earliest_day": earliest_day,
            "max_wait_days": max_wait,
        })
    return surgeries


def generate_instance(cfg: InstanceConfig) -> dict[str, Any]:
    """Build a JSON-serialisable instance dict given a config."""
    rng = random.Random(cfg.seed)
    surgery_types = _filter_surgery_types(cfg.specialties_used)
    if not surgery_types:
        raise ValueError(f"No surgery types match the specialties used in {cfg.name}")

    ots = _generate_ots(cfg, surgery_types, rng)
    teams = _generate_teams(cfg, rng)
    surgeries = _generate_surgeries(cfg, surgery_types, rng)

    return {
        "name": cfg.name,
        "horizon_days": cfg.horizon_days,
        "global_parameters": {
            "icu_capacity": cfg.icu_capacity,
            "ward_capacity": cfg.ward_capacity,
            "overflow_allowed": cfg.overflow_allowed,
            "objective_weights": DEFAULT_OBJECTIVE_WEIGHTS,
            "seed": cfg.seed,
        },
        "ots": ots,
        "teams": teams,
        "surgery_types": surgery_types,
        "surgeries": surgeries,
    }


# ---------------------------------------------------------------------------
# Validation: a sanity check that an instance is internally consistent.
# Does NOT check that the instance is "easy" or feasible to schedule completely.
# ---------------------------------------------------------------------------

def validate_instance(instance: dict[str, Any]) -> list[str]:
    issues = []
    # Every surgery must have at least one OT that permits its specialty and has its equipment
    ots = instance["ots"]
    teams = instance["teams"]
    for s in instance["surgeries"]:
        valid_ots = [
            o for o in ots
            if s["specialty"] in o["permitted_specialties"]
            and all(eq in o["equipment"] for eq in s["required_equipment"])
        ]
        if not valid_ots:
            issues.append(f"surgery {s['id']} has no OT supporting its specialty and equipment")
        valid_teams = [
            t for t in teams
            if s["specialty"] == t["primary_specialty"] or s["specialty"] in t["secondary_specialties"]
        ]
        if not valid_teams:
            issues.append(f"surgery {s['id']} has no team supporting its specialty")
    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=Path("instances"),
                        help="Output directory for public instances")
    parser.add_argument("--hidden-out", type=Path, default=None,
                        help="Output directory for the hidden instance (default: out/private)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    hidden_out = args.hidden_out or (args.out / "private")
    hidden_out.mkdir(parents=True, exist_ok=True)

    for name, cfg in CONFIGS.items():
        instance = generate_instance(cfg)
        issues = validate_instance(instance)
        if issues:
            print(f"WARNING: validation issues in {name}:")
            for issue in issues:
                print(f"  - {issue}")
        target_dir = hidden_out if name == "hidden" else args.out
        target_path = target_dir / f"ot_{name}.json"
        with open(target_path, "w") as f:
            json.dump(instance, f, indent=2, sort_keys=False)
        n_surgeries = len(instance["surgeries"])
        print(f"Wrote {target_path}  ({n_surgeries} surgeries, "
              f"horizon {instance['horizon_days']} days)")


if __name__ == "__main__":
    main()
