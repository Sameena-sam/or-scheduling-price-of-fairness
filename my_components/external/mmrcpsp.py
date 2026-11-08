"""External benchmark: multi-mode resource-constrained project scheduling.

WHY THIS BENCHMARK. Brief section 3.3 requires validation on an external
benchmark to show the method is not specific to OT scheduling. MMRCPSP is the
structurally appropriate choice here because its renewable resources are
*cumulative over time* in exactly the way ICU and ward beds are: an activity
occupies capacity for the whole of its duration, so a placement decision
constrains availability in later periods. That is the precise structural
property the repair-based decoder was designed to exploit, which makes MMRCPSP
a genuine test of transferability rather than an unrelated second problem.

WHAT IS TESTED. The OT experiments found that allow-then-repair constraint
handling is outperformed by feasibility-by-construction, and diagnosed the
cause as a fixed objective preference embedded in the decoder. Here the same
two decoder designs are re-implemented for MMRCPSP and compared under the same
protocol, to establish whether that finding transfers to another
cumulative-resource scheduling problem.

INSTANCE FORMAT. Standard PSPLIB `.mm` (multi-mode) and `.sm` (single-mode)
files are parsed. Download from the PSPLIB site (j10.mm, j20.mm, j30.sm, ...)
and place the extracted files in `external_benchmark/instances/`.
`make_synthetic_instance()` writes a small instance in the identical format so
the pipeline can be verified without the download.

OBJECTIVES. Two, both minimised:
    f1  makespan
    f2  renewable-resource levelling -- the sum over renewable resources and
        periods of the squared deviation of usage from that resource's mean
        usage over the project duration.
Makespan is the indicator by which MMRCPSP results are conventionally
reported, so f1 can be compared directly against published best-known values;
f2 plays the role f2 played in the OT problem, penalising uneven use of a
capacitated resource.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- model

@dataclass(frozen=True)
class Mode:
    number: int
    duration: int
    renewable: tuple[int, ...]        # requirement per renewable resource
    nonrenewable: tuple[int, ...]     # requirement per nonrenewable resource


@dataclass(frozen=True)
class Activity:
    number: int                       # 1-based, as in the file
    successors: tuple[int, ...]
    modes: tuple[Mode, ...]


@dataclass
class ProjectInstance:
    name: str
    n_activities: int                 # includes the two dummies
    renewable_capacity: tuple[int, ...]
    nonrenewable_capacity: tuple[int, ...]
    activities: dict[int, Activity]
    horizon: int
    predecessors: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self):
        preds: dict[int, list[int]] = {a: [] for a in self.activities}
        for a, act in self.activities.items():
            for s in act.successors:
                preds.setdefault(s, []).append(a)
        self.predecessors = {k: tuple(sorted(v)) for k, v in preds.items()}

    @property
    def n_renewable(self) -> int:
        return len(self.renewable_capacity)

    @property
    def n_nonrenewable(self) -> int:
        return len(self.nonrenewable_capacity)

    def topological_order(self) -> list[int]:
        indeg = {a: len(self.predecessors.get(a, ())) for a in self.activities}
        ready = sorted(a for a, d in indeg.items() if d == 0)
        order = []
        while ready:
            a = ready.pop(0)
            order.append(a)
            for s in self.activities[a].successors:
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
            ready.sort()
        if len(order) != len(self.activities):
            raise ValueError("precedence graph contains a cycle")
        return order

    def __repr__(self):
        return (f"ProjectInstance(name={self.name!r}, activities={self.n_activities}, "
                f"renewable={self.n_renewable}, nonrenewable={self.n_nonrenewable})")


# -------------------------------------------------------------------- parser

def _ints(line: str) -> list[int]:
    return [int(t) for t in re.findall(r"-?\d+", line)]


def parse_psplib(path: str | Path) -> ProjectInstance:
    """Parse a PSPLIB .mm or .sm file.

    The format is fixed-layout with three blocks of interest: the precedence
    relations, the mode/requirement table, and the resource availabilities.
    Single-mode (.sm) files are multi-mode files with one mode per activity,
    and parse through the same path.
    """
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    n_act = horizon = None
    n_ren = n_non = 0
    for i, ln in enumerate(lines):
        if "jobs (incl. supersource/sink" in ln:
            n_act = _ints(ln)[-1]
        elif ln.strip().startswith("horizon"):
            horizon = _ints(ln)[-1]
        elif "- renewable" in ln:
            n_ren = _ints(ln)[0]
        elif "- nonrenewable" in ln:
            n_non = _ints(ln)[0]
    if n_act is None:
        raise ValueError(f"{path.name}: could not find activity count")

    # --- precedence block ---
    successors: dict[int, tuple[int, ...]] = {}
    try:
        start = next(i for i, l in enumerate(lines) if "PRECEDENCE RELATIONS" in l)
    except StopIteration:
        raise ValueError(f"{path.name}: no PRECEDENCE RELATIONS block")
    i = start + 1
    while i < len(lines) and not lines[i].lstrip().startswith("jobnr"):
        i += 1
    i += 1
    while i < len(lines) and not lines[i].startswith("*"):
        vals = _ints(lines[i])
        if len(vals) >= 3:
            job, _nmodes, nsucc = vals[0], vals[1], vals[2]
            successors[job] = tuple(vals[3:3 + nsucc])
        i += 1

    # --- requests/durations block ---
    modes: dict[int, list[Mode]] = {}
    try:
        start = next(i for i, l in enumerate(lines) if "REQUESTS/DURATIONS" in l)
    except StopIteration:
        raise ValueError(f"{path.name}: no REQUESTS/DURATIONS block")
    i = start + 1
    while i < len(lines) and not lines[i].lstrip().startswith("jobnr"):
        i += 1
    i += 2  # skip header and the dashed rule
    current = None
    while i < len(lines) and not lines[i].startswith("*"):
        vals = _ints(lines[i])
        if vals:
            if len(vals) >= 3 + n_ren + n_non:
                current = vals[0]
                mode_no, dur = vals[1], vals[2]
                res = vals[3:3 + n_ren + n_non]
            else:
                mode_no, dur = vals[0], vals[1]
                res = vals[2:2 + n_ren + n_non]
            modes.setdefault(current, []).append(Mode(
                number=mode_no, duration=dur,
                renewable=tuple(res[:n_ren]),
                nonrenewable=tuple(res[n_ren:n_ren + n_non]),
            ))
        i += 1

    # --- availabilities ---
    try:
        start = next(i for i, l in enumerate(lines) if "RESOURCEAVAILABILITIES" in l)
    except StopIteration:
        raise ValueError(f"{path.name}: no RESOURCEAVAILABILITIES block")
    caps: list[int] = []
    i = start + 1
    while i < len(lines) and len(caps) < n_ren + n_non:
        v = _ints(lines[i])
        if v and not lines[i].lstrip().startswith("R "):
            caps.extend(v)
        i += 1

    activities = {
        j: Activity(number=j,
                    successors=successors.get(j, ()),
                    modes=tuple(modes.get(j, [Mode(1, 0, (0,) * n_ren, (0,) * n_non)])))
        for j in sorted(set(successors) | set(modes))
    }
    if len(activities) != n_act:
        raise ValueError(f"{path.name}: parsed {len(activities)} activities, header says {n_act}")

    return ProjectInstance(
        name=path.stem,
        n_activities=n_act,
        renewable_capacity=tuple(caps[:n_ren]),
        nonrenewable_capacity=tuple(caps[n_ren:n_ren + n_non]),
        activities=activities,
        horizon=horizon or sum(max(m.duration for m in a.modes) for a in activities.values()),
    )


# ------------------------------------------------------------------ decoding

@dataclass
class ProjectSchedule:
    start: dict[int, int]
    mode: dict[int, int]              # activity -> index into its modes tuple
    makespan: int
    renewable_usage: np.ndarray       # (n_renewable, horizon)
    nonrenewable_used: tuple[int, ...]
    feasible: bool
    violations: list[str] = field(default_factory=list)


def _assign_modes(inst: ProjectInstance, order: list[int]) -> dict[int, int]:
    """Assign a mode to every activity, respecting the nonrenewable budget.

    Nonrenewable feasibility is a genuine subproblem in MMRCPSP: a purely
    greedy pass that takes the shortest mode for each activity in turn will
    routinely exhaust the budget before the later activities are assigned, and
    no amount of rescheduling can then recover feasibility because
    nonrenewable consumption does not depend on timing.

    Two phases are therefore used. First every activity takes its shortest
    mode. If that exceeds a nonrenewable budget, a repair pass repeatedly
    applies the single mode switch offering the largest reduction in total
    overconsumption per unit of added duration, until the assignment is
    feasible or no improving switch remains. This is the standard mode-repair
    approach for MMRCPSP and keeps the assignment deterministic, so the
    permutation remains the only source of variation between individuals.
    """
    chosen: dict[int, int] = {}
    for j in order:
        modes = inst.activities[j].modes
        chosen[j] = min(range(len(modes)), key=lambda k: (modes[k].duration, sum(modes[k].nonrenewable)))

    def total_nonrenewable(assign):
        tot = [0] * inst.n_nonrenewable
        for a, mi in assign.items():
            m = inst.activities[a].modes[mi]
            for r in range(inst.n_nonrenewable):
                tot[r] += m.nonrenewable[r]
        return tot

    def overuse(tot):
        return sum(max(0, tot[r] - inst.nonrenewable_capacity[r])
                   for r in range(inst.n_nonrenewable))

    tot = total_nonrenewable(chosen)
    if overuse(tot) == 0:
        return chosen

    for _ in range(8 * len(order) + 64):
        current = overuse(tot)
        if current == 0:
            break
        best = None                       # (ratio, added_duration, activity, mode)
        for j in order:
            modes = inst.activities[j].modes
            cur = modes[chosen[j]]
            for k, cand in enumerate(modes):
                if k == chosen[j]:
                    continue
                trial = list(tot)
                for r in range(inst.n_nonrenewable):
                    trial[r] += cand.nonrenewable[r] - cur.nonrenewable[r]
                gain = current - overuse(trial)
                if gain <= 0:
                    continue
                added = max(0, cand.duration - cur.duration)
                ratio = added / gain           # duration paid per unit of overuse removed
                key = (ratio, added, j, k)
                if best is None or key < best:
                    best = key
        if best is None:
            break                          # no improving switch exists
        _, _, j, k = best
        cur = inst.activities[j].modes[chosen[j]]
        cand = inst.activities[j].modes[k]
        for r in range(inst.n_nonrenewable):
            tot[r] += cand.nonrenewable[r] - cur.nonrenewable[r]
        chosen[j] = k
    return chosen


def _finalise(inst, start, mode, usage) -> ProjectSchedule:
    makespan = max((start[j] + inst.activities[j].modes[mode[j]].duration
                    for j in start), default=0)
    non_used = [0] * inst.n_nonrenewable
    for j, mi in mode.items():
        for r in range(inst.n_nonrenewable):
            non_used[r] += inst.activities[j].modes[mi].nonrenewable[r]
    violations = []
    for r in range(inst.n_renewable):
        over = int(np.max(usage[r]) - inst.renewable_capacity[r]) if usage.shape[1] else 0
        if over > 0:
            violations.append(f"renewable resource {r+1} exceeded by {over}")
    for r in range(inst.n_nonrenewable):
        if non_used[r] > inst.nonrenewable_capacity[r]:
            violations.append(f"nonrenewable resource {r+1} over budget "
                              f"({non_used[r]} > {inst.nonrenewable_capacity[r]})")
    for j, preds in inst.predecessors.items():
        for p in preds:
            if start[j] < start[p] + inst.activities[p].modes[mode[p]].duration:
                violations.append(f"precedence {p}->{j} violated")
    return ProjectSchedule(start=start, mode=mode, makespan=int(makespan),
                           renewable_usage=usage, nonrenewable_used=tuple(non_used),
                           feasible=not violations, violations=violations)


class GreedySGSDecoder:
    """Serial schedule generation scheme with resource feasibility enforced
    during placement. Each activity is started at the earliest period at which
    precedence is satisfied AND renewable capacity is available for its whole
    duration. Analogous to the OT baseline decoder."""

    name = "greedy"

    def __call__(self, inst: ProjectInstance, permutation: np.ndarray) -> ProjectSchedule:
        order = _precedence_feasible_order(inst, permutation)
        mode = _assign_modes(inst, order)
        H = inst.horizon + sum(max(m.duration for m in a.modes) for a in inst.activities.values())
        usage = np.zeros((inst.n_renewable, H + 1), dtype=int)
        start: dict[int, int] = {}
        for j in order:
            m = inst.activities[j].modes[mode[j]]
            est = max((start[p] + inst.activities[p].modes[mode[p]].duration
                       for p in inst.predecessors.get(j, ())), default=0)
            t = est
            while True:
                if m.duration == 0:
                    break
                ok = all(np.all(usage[r, t:t + m.duration] + m.renewable[r]
                                <= inst.renewable_capacity[r])
                         for r in range(inst.n_renewable))
                if ok or t > H - m.duration:
                    break
                t += 1
            start[j] = t
            for r in range(inst.n_renewable):
                usage[r, t:t + m.duration] += m.renewable[r]
        return _finalise(inst, start, mode, usage)


class RepairSGSDecoder:
    """Allow-then-repair: activities are placed at their earliest
    precedence-feasible period with renewable capacity ignored, after which
    over-capacity periods are repaired by delaying activities. Analogous to
    the OT repair decoder, and using the same displacement principle -- the
    activity selected for delay is the one whose removal relieves the
    violation at least cost, measured here by shortest duration, since
    MMRCPSP has no priority attribute.

    `evict` selects the displacement rule, mirroring the OT ablation:
        "shortest"    delay the shortest activity occupying the period
        "permutation" delay whichever occupant appears latest in the
                      permutation (genotype-dependent)
    """

    def __init__(self, evict: str = "shortest", max_passes: int | None = None):
        if evict not in ("shortest", "permutation"):
            raise ValueError(evict)
        self.evict = evict
        self.max_passes = max_passes
        self.name = f"repair_{evict}"

    def __call__(self, inst: ProjectInstance, permutation: np.ndarray) -> ProjectSchedule:
        order = _precedence_feasible_order(inst, permutation)
        rank = {int(v): i for i, v in enumerate(np.asarray(permutation, dtype=int))}
        mode = _assign_modes(inst, order)
        H = inst.horizon + sum(max(m.duration for m in a.modes) for a in inst.activities.values())

        # pass 1: earliest precedence-feasible placement, capacity ignored
        start: dict[int, int] = {}
        for j in order:
            start[j] = max((start[p] + inst.activities[p].modes[mode[p]].duration
                            for p in inst.predecessors.get(j, ())), default=0)

        def build_usage():
            u = np.zeros((inst.n_renewable, H + 1), dtype=int)
            for a, t in start.items():
                m = inst.activities[a].modes[mode[a]]
                for r in range(inst.n_renewable):
                    u[r, t:t + m.duration] += m.renewable[r]
            return u

        max_passes = self.max_passes or (4 * len(order) + 64)
        for _ in range(max_passes):
            usage = build_usage()
            hit = None
            for r in range(inst.n_renewable):
                over = np.nonzero(usage[r] > inst.renewable_capacity[r])[0]
                if over.size:
                    hit = (r, int(over[0]))
                    break
            if hit is None:
                break
            r, t = hit
            occupants = [a for a, s in start.items()
                         if s <= t < s + inst.activities[a].modes[mode[a]].duration
                         and inst.activities[a].modes[mode[a]].renewable[r] > 0]
            if not occupants:
                break
            if self.evict == "shortest":
                victim = min(occupants, key=lambda a: (inst.activities[a].modes[mode[a]].duration, a))
            else:
                victim = max(occupants, key=lambda a: rank.get(a - 1, 0))
            # delay the victim past the violating period, then cascade successors
            start[victim] = t + 1
            for a in _precedence_feasible_order(inst, permutation):
                est = max((start[p] + inst.activities[p].modes[mode[p]].duration
                           for p in inst.predecessors.get(a, ())), default=0)
                if start[a] < est:
                    start[a] = est
        return _finalise(inst, start, mode, build_usage())


def _precedence_feasible_order(inst: ProjectInstance, permutation: np.ndarray) -> list[int]:
    """Turn an arbitrary permutation into a precedence-feasible activity list.

    The permutation expresses a preference order; activities are taken in that
    order whenever their predecessors are already placed, which is the standard
    way of making an activity-list encoding valid for RCPSP.
    """
    acts = sorted(inst.activities)
    pref = {a: i for i, a in enumerate(acts)}
    for i, v in enumerate(np.asarray(permutation, dtype=int)):
        if 0 <= int(v) < len(acts):
            pref[acts[int(v)]] = i
    remaining = set(acts)
    placed: set[int] = set()
    order: list[int] = []
    while remaining:
        ready = [a for a in remaining if set(inst.predecessors.get(a, ())) <= placed]
        if not ready:
            raise ValueError("precedence deadlock")
        nxt = min(ready, key=lambda a: pref[a])
        order.append(nxt)
        placed.add(nxt)
        remaining.discard(nxt)
    return order


# ------------------------------------------------------------------ evaluate

def evaluate_project(inst: ProjectInstance, sched: ProjectSchedule) -> tuple[float, float]:
    """f1 = makespan, f2 = renewable-resource levelling (squared deviation of
    per-period usage from that resource's mean over the project duration)."""
    f1 = float(sched.makespan)
    if sched.makespan <= 0:
        return f1, 0.0
    u = sched.renewable_usage[:, :sched.makespan].astype(float)
    f2 = float(np.sum((u - u.mean(axis=1, keepdims=True)) ** 2))
    return f1, f2


# ------------------------------------------------------------- self-contained

SYNTHETIC = """************************************************************************
file with basedata            : synth.bas
initial value random generator: 1
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  8
horizon                       :  40
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  1   N
  - doubly constrained        :  0   D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        2          1           5
   3        2          1           6
   4        2          1           6
   5        2          1           7
   6        2          1           7
   7        2          1           8
   8        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1
------------------------------------------------------------------------
   1      1     0       0    0    0
   2      1     4       3    2    4
          2     6       2    1    3
   3      1     3       2    3    5
          2     5       1    2    3
   4      1     5       4    1    6
          2     7       2    1    4
   5      1     2       1    4    2
          2     4       1    2    2
   6      1     6       3    2    5
          2     8       2    1    4
   7      1     3       2    2    3
          2     5       1    1    2
   8      1     0       0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1
    6    5   30
************************************************************************
"""


def make_synthetic_instance(path: str | Path) -> Path:
    """Write a small instance in the exact PSPLIB layout, so the parser and
    both decoders can be verified without the PSPLIB download."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SYNTHETIC)
    return path


# ------------------------------------------------- published optimal results

INFEASIBLE_MARKER = 16384


def parse_psplib_solutions(path: str | Path) -> dict[str, float]:
    """Read a PSPLIB optimal-solutions file (e.g. `j10opt.mm`).

    The file carries a header block, then a table:

        Paramter Instance Makespan CPU-Time[sec.]
        1       1         16384    0.00
        2       2         20       0.12

    The first two columns identify the instance as parameter set and instance
    number, which correspond to the filename `j10<param>_<instance>.mm`. A
    makespan of 16384 is the sentinel for "no feasible solution exists" -- for
    the j10 set 104 of 640 instances are infeasible, and PSPLIB omits those
    from the instance download, which is why the extracted set contains 536
    files rather than 640. Sentinel rows are excluded from the returned map.

    Returns a mapping from instance stem (as it appears in the filename,
    without the `.mm` suffix) to published optimal makespan.
    """
    path = Path(path)
    prefix = path.stem.split("opt")[0] or "j10"      # "j10opt" -> "j10"
    lines = path.read_text(errors="ignore").splitlines()

    # Parsing must begin after the table header. The metadata block above it
    # contains lines such as "Date : 2/15/96", which would otherwise be read
    # as parameter 2, instance 15, makespan 96.
    start = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if "instance" in low and "makespan" in low:
            start = i + 1
            break

    out: dict[str, float] = {}
    for line in lines[start:]:
        if set(line.strip()) <= {"-", "=", ""}:          # rules under the header
            continue
        vals = _ints(line)
        if len(vals) < 3:
            continue
        param, instance, makespan = vals[0], vals[1], vals[2]
        if makespan == INFEASIBLE_MARKER:
            continue
        if param <= 0 or instance <= 0 or makespan <= 0:
            continue
        out[f"{prefix}{param}_{instance}"] = float(makespan)
    return out
