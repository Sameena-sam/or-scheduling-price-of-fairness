"""Tests for the MMRCPSP external-benchmark module.

Run with:  PYTHONPATH=. pytest tests/test_external.py -v

These verify that the PSPLIB parser reads the format correctly and that both
decoders produce schedules satisfying every constraint of the problem --
precedence, renewable capacity per period, and the nonrenewable budget.
Without these, a benchmark comparison would be meaningless: an infeasible
schedule can attain an arbitrarily good makespan.
"""
from __future__ import annotations

import numpy as np
import pytest

from my_components.external.mmrcpsp import (
    make_synthetic_instance, parse_psplib, evaluate_project,
    GreedySGSDecoder, RepairSGSDecoder,
)

SEEDS = [0, 1, 2, 3, 4, 5]


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    p = make_synthetic_instance(tmp_path_factory.mktemp("ext") / "synth.mm")
    return parse_psplib(p)


def assert_project_feasible(inst, sched):
    """Independent check of every hard constraint."""
    for j, preds in inst.predecessors.items():
        for p in preds:
            end = sched.start[p] + inst.activities[p].modes[sched.mode[p]].duration
            assert sched.start[j] >= end, f"precedence {p}->{j} violated"

    horizon = sched.makespan + 1
    usage = np.zeros((inst.n_renewable, horizon + 1), dtype=int)
    nonren = [0] * inst.n_nonrenewable
    for j, t in sched.start.items():
        m = inst.activities[j].modes[sched.mode[j]]
        for r in range(inst.n_renewable):
            usage[r, t:t + m.duration] += m.renewable[r]
        for r in range(inst.n_nonrenewable):
            nonren[r] += m.nonrenewable[r]

    for r in range(inst.n_renewable):
        peak = int(usage[r].max())
        assert peak <= inst.renewable_capacity[r], \
            f"renewable resource {r+1} peak {peak} > capacity {inst.renewable_capacity[r]}"
    for r in range(inst.n_nonrenewable):
        assert nonren[r] <= inst.nonrenewable_capacity[r], \
            f"nonrenewable resource {r+1} used {nonren[r]} > budget {inst.nonrenewable_capacity[r]}"


# ------------------------------------------------------------------- parser

def test_parser_reads_header(inst):
    assert inst.n_activities == 8
    assert inst.n_renewable == 2 and inst.n_nonrenewable == 1
    assert inst.renewable_capacity == (6, 5)
    assert inst.nonrenewable_capacity == (30,)


def test_parser_reads_modes_and_precedence(inst):
    assert len(inst.activities[1].modes) == 1        # dummy source
    assert len(inst.activities[2].modes) == 2
    assert inst.activities[1].successors == (2, 3, 4)
    assert inst.activities[8].successors == ()
    assert inst.predecessors[8] == (7,)
    assert inst.activities[2].modes[0].duration == 4
    assert inst.activities[2].modes[0].renewable == (3, 2)
    assert inst.activities[2].modes[0].nonrenewable == (4,)


def test_topological_order_respects_precedence(inst):
    order = inst.topological_order()
    pos = {a: i for i, a in enumerate(order)}
    for j, preds in inst.predecessors.items():
        for p in preds:
            assert pos[p] < pos[j]


# ------------------------------------------------------------------ decoders

@pytest.mark.parametrize("seed", SEEDS)
def test_greedy_decoder_feasible(inst, seed):
    perm = np.random.default_rng(seed).permutation(inst.n_activities)
    sched = GreedySGSDecoder()(inst, perm)
    assert sched.feasible, sched.violations
    assert_project_feasible(inst, sched)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("evict", ["shortest", "permutation"])
def test_repair_decoder_feasible(inst, seed, evict):
    perm = np.random.default_rng(seed).permutation(inst.n_activities)
    sched = RepairSGSDecoder(evict=evict)(inst, perm)
    assert sched.feasible, sched.violations
    assert_project_feasible(inst, sched)


@pytest.mark.parametrize("evict", ["shortest", "permutation"])
def test_decoders_are_deterministic(inst, evict):
    perm = np.random.default_rng(11).permutation(inst.n_activities)
    a = evaluate_project(inst, RepairSGSDecoder(evict=evict)(inst, perm))
    b = evaluate_project(inst, RepairSGSDecoder(evict=evict)(inst, perm))
    assert a == b


def test_makespan_at_least_critical_path(inst):
    """No schedule can finish sooner than the longest precedence chain using
    each activity's shortest mode."""
    best = {j: min(m.duration for m in a.modes) for j, a in inst.activities.items()}
    length = {}
    for j in inst.topological_order():
        length[j] = best[j] + max((length[p] for p in inst.predecessors.get(j, ())), default=0)
    lower_bound = max(length.values())
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(inst.n_activities)
        for dec in (GreedySGSDecoder(), RepairSGSDecoder("shortest")):
            assert dec(inst, perm).makespan >= lower_bound


def test_objectives_are_finite_and_ordered(inst):
    perm = np.random.default_rng(3).permutation(inst.n_activities)
    f1, f2 = evaluate_project(inst, GreedySGSDecoder()(inst, perm))
    assert np.isfinite(f1) and np.isfinite(f2)
    assert f1 > 0 and f2 >= 0


def test_permutation_ordering_influences_schedule(inst):
    """If no permutation changed the outcome the encoding would be inert and
    the evolutionary search meaningless."""
    outcomes = set()
    for seed in range(40):
        perm = np.random.default_rng(seed).permutation(inst.n_activities)
        outcomes.add(evaluate_project(inst, GreedySGSDecoder()(inst, perm)))
    assert len(outcomes) > 1


# ------------------------------------------------- nonrenewable budget repair

TIGHT_BUDGET = """************************************************************************
file with basedata            : tight.bas
initial value random generator: 1
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  6
horizon                       :  40
RESOURCES
  - renewable                 :  1   R
  - nonrenewable              :  1   N
  - doubly constrained        :  0   D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          2           2   3
   2        2          1           4
   3        2          1           4
   4        2          1           5
   5        2          1           6
   6        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  N 1
------------------------------------------------------------------------
   1      1     0       0    0
   2      1     2       3    9
          2     8       1    2
   3      1     2       3    9
          2     8       1    2
   4      1     2       3    9
          2     8       1    2
   5      1     2       3    9
          2     8       1    2
   6      1     0       0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  N 1
    6   12
************************************************************************
"""


@pytest.fixture(scope="module")
def tight_inst(tmp_path_factory):
    """An instance whose shortest modes blow the nonrenewable budget four times
    over (4 x 9 = 36 against a budget of 12), so a decoder that assigns modes
    greedily by duration cannot produce a feasible solution without repair.

    This is the case that a loose synthetic instance hides: the original
    implementation passed every test while producing nonrenewable-infeasible
    schedules on a third of real PSPLIB instances.
    """
    p = tmp_path_factory.mktemp("tight") / "tight.mm"
    p.write_text(TIGHT_BUDGET)
    return parse_psplib(p)


def test_shortest_modes_alone_would_break_the_budget(tight_inst):
    """Confirms the fixture actually exercises the repair path."""
    naive = sum(min(a.modes, key=lambda m: m.duration).nonrenewable[0]
                for a in tight_inst.activities.values())
    assert naive > tight_inst.nonrenewable_capacity[0]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("decoder", [GreedySGSDecoder(), RepairSGSDecoder("shortest"),
                                     RepairSGSDecoder("permutation")])
def test_nonrenewable_budget_respected_under_pressure(tight_inst, seed, decoder):
    perm = np.random.default_rng(seed).permutation(tight_inst.n_activities)
    sched = decoder(tight_inst, perm)
    assert sched.feasible, sched.violations
    assert_project_feasible(tight_inst, sched)
