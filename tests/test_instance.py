"""Tests for instance loading and basic structural validity."""
import json
from pathlib import Path

import pytest

from cio.problem import Instance


INSTANCES_DIR = Path(__file__).resolve().parent.parent / "instances"
SIZES = ["small", "medium", "large"]


@pytest.mark.parametrize("size", SIZES)
def test_instance_loads(size):
    path = INSTANCES_DIR / f"ot_{size}.json"
    inst = Instance.from_json(path)
    assert inst.name == size
    assert inst.horizon_days >= 1
    assert inst.n_surgeries > 0
    assert inst.n_ots > 0
    assert inst.n_teams > 0


@pytest.mark.parametrize("size", SIZES)
def test_instance_eligibility_nonempty(size):
    """Every surgery must have at least one eligible OT and team."""
    inst = Instance.from_json(INSTANCES_DIR / f"ot_{size}.json")
    for s in inst.surgeries:
        ot_ids = inst.eligible_ots_for_surgery[s.id]
        team_ids = inst.eligible_teams_for_surgery[s.id]
        assert len(ot_ids) > 0, f"surgery {s.id} has no eligible OT"
        assert len(team_ids) > 0, f"surgery {s.id} has no eligible team"


@pytest.mark.parametrize("size", SIZES)
def test_surgery_invariants(size):
    """Surgery fields are internally consistent."""
    inst = Instance.from_json(INSTANCES_DIR / f"ot_{size}.json")
    for s in inst.surgeries:
        assert s.priority in {1, 2, 3, 4, 5}
        assert 1 <= s.earliest_day <= inst.horizon_days
        assert s.max_wait_days >= 1
        assert s.duration_hr > 0
        assert s.icu_days >= 0 and s.ward_days >= 0
        # Surgery type must exist in catalogue
        assert s.type in inst.surgery_types


def test_unique_surgery_ids():
    """Surgery IDs are unique within an instance."""
    for size in SIZES:
        inst = Instance.from_json(INSTANCES_DIR / f"ot_{size}.json")
        ids = [s.id for s in inst.surgeries]
        assert len(ids) == len(set(ids))


def test_repr_does_not_crash():
    inst = Instance.from_json(INSTANCES_DIR / "ot_small.json")
    repr(inst)
