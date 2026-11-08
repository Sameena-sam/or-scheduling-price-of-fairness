# Instance JSON Format

Each instance file is a single JSON object with the following top-level keys:

```json
{
  "name": "small",
  "horizon_days": 7,
  "global_parameters": { ... },
  "ots": [ ... ],
  "teams": [ ... ],
  "surgery_types": [ ... ],
  "surgeries": [ ... ]
}
```

Throughout this document, "day" is a 1-indexed integer from 1 to `horizon_days`.

---

## Top-Level Fields

### `name` (string)

Identifier for the instance. One of `"small"`, `"medium"`, `"large"`, or
`"hidden"` for the standard distribution.

### `horizon_days` (integer)

Planning horizon length in days. Surgeries must be scheduled within this
horizon.

### `global_parameters` (object)

```json
{
  "icu_capacity": 1,
  "ward_capacity": 5,
  "overflow_allowed": false,
  "objective_weights": {
    "ot_idle_weight": 1.0,
    "icu_overflow_weight": 10.0,
    "ward_overflow_weight": 5.0
  },
  "seed": 20260201
}
```

- `icu_capacity`: number of ICU beds available per day.
- `ward_capacity`: number of ward beds available per day.
- `overflow_allowed`: if `false`, the decoder must keep ICU and ward
  occupancy within capacity; surgeries that cannot fit are unscheduled.
  If `true`, overflow is allowed but is captured in the resource-imbalance
  objective `f2`.
- `objective_weights`: weights for `f2` components (OT idle squared,
  ICU overflow squared, ward overflow squared).
- `seed`: the seed used to generate this instance (informational only).

---

## OTs (Operating Theatres)

```json
{
  "id": 1,
  "permitted_specialties": ["general_surgery"],
  "equipment": ["laparoscopy_tower", "c_arm"],
  "daily_hours": 8.0,
  "daily_activation_cost": 1000
}
```

- `id`: unique integer.
- `permitted_specialties`: list of specialty names this OT supports.
- `equipment`: list of equipment items installed in this OT.
- `daily_hours`: available operating hours per day in this OT.
- `daily_activation_cost`: cost of opening the OT on a day (not used by
  the default objectives, but available for student extensions).

---

## Teams

```json
{
  "id": 1,
  "primary_specialty": "general_surgery",
  "secondary_specialties": [],
  "daily_hours_max": 8.0,
  "max_consecutive_days": 5,
  "daily_cost": 1500
}
```

- `id`: unique integer.
- `primary_specialty`: the specialty the team is trained for.
- `secondary_specialties`: additional specialties this team can handle.
- `daily_hours_max`: maximum hours this team can operate per day.
- `max_consecutive_days`: maximum number of consecutive days this team
  can operate without a rest day.
- `daily_cost`: cost per active day (informational).

---

## Surgery Types

A catalogue of the surgery types appearing in this instance. Each surgery
in the `surgeries` array refers to one of these by name.

```json
{
  "name": "lap_chole",
  "specialty": "general_surgery",
  "duration_hr": 1.5,
  "icu_days": 0,
  "ward_days": 2,
  "equipment": ["laparoscopy_tower"]
}
```

- `name`: unique string identifier.
- `specialty`: the specialty required to perform this surgery.
- `duration_hr`: operating time in hours.
- `icu_days`: post-op ICU days required (starting the same day as surgery).
- `ward_days`: post-op ward days required (after any ICU stay).
- `equipment`: list of equipment items required for this surgery.

---

## Surgeries

The individual surgery cases to be scheduled.

```json
{
  "id": 1,
  "type": "lap_chole",
  "specialty": "general_surgery",
  "duration_hr": 1.5,
  "icu_days": 0,
  "ward_days": 2,
  "required_equipment": ["laparoscopy_tower"],
  "priority": 5,
  "earliest_day": 1,
  "max_wait_days": 5
}
```

- `id`: unique integer.
- `type`: name of the surgery type (matches an entry in `surgery_types`).
- `specialty`, `duration_hr`, `icu_days`, `ward_days`, `required_equipment`:
  inherited from the surgery type, repeated here for convenience.
- `priority`: integer 1-5, where 5 is most urgent.
- `earliest_day`: the earliest day on which this surgery can be scheduled.
- `max_wait_days`: the number of days after `earliest_day` within which the
  surgery should be scheduled. Scheduling later incurs a lateness penalty
  in the throughput objective `f1`.

---

## Objectives

The three objectives evaluated for any schedule, all minimised in pymoo
convention:

### f1 — Negative priority-weighted throughput

```
reward = sum(priority of each scheduled surgery)
       - sum(priority * days_late for each surgery scheduled past its
             max_wait window)
       - sum(priority * horizon_days * 2.0 for each unscheduled surgery)
f1 = -reward
```

Lower is better (i.e. higher throughput). Note that f1 can be negative.

### f2 — Resource utilisation imbalance

Computed by `cio.problem.evaluator._f2_resource_imbalance`:

```
f2 = w_idle * sum(OT idle hours squared, over active OT-days)
   + w_icu  * sum(ICU overflow squared, per day)
   + w_ward * sum(ward overflow squared, per day)
```

With default `objective_weights`, ICU overflow dominates (weight 10),
then ward overflow (weight 5), then OT idle (weight 1).

### f3 — Waiting-time fairness

```
For each priority class with at least one surgery:
    mean_wait_p = mean(assigned_day - earliest_day) over scheduled
                  + horizon_days for each unscheduled
f3 = variance(mean_wait_p across priority classes)
```

Lower is better — equal mean waiting times across priority classes.

---

## Eligibility Constraints

A surgery `s` is eligible for an `(OT, team, day)` triple iff:

- `s.specialty` is in `OT.permitted_specialties`.
- `s.required_equipment` is a subset of `OT.equipment`.
- `team.can_do(s.specialty)` — i.e. `s.specialty` is either the team's
  primary or one of its secondaries.
- `OT.daily_hours - used_OT_hours_on_day >= s.duration_hr`.
- `team.daily_hours_max - used_team_hours_on_day >= s.duration_hr`.
- Adding `day` to `team.active_days` does not create a consecutive run
  exceeding `team.max_consecutive_days`.
- For each day in the surgery's ICU stay window, ICU occupancy + 1
  does not exceed `icu_capacity` (when `overflow_allowed = false`).
- Same for the ward stay window.

The default `GreedyDecoder` enforces these constraints by construction,
producing feasible schedules. Surgeries that cannot be placed remain
unscheduled.
