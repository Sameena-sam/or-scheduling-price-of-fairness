"""OT scheduling problem module.

Public API:
    Instance       - Typed access to a loaded instance file.
    Schedule       - A decoded schedule.
    Assignment     - One surgery -> (OT, day, team) assignment.
    GreedyDecoder  - Default permutation decoder.
    OTSchedulingProblem - pymoo Problem subclass.
    evaluate       - Compute (f1, f2, f3) for a Schedule.
"""
from .instance import Instance, OT, Team, Surgery, SurgeryType
from .schedule import Schedule, Assignment
from .decoder import BaseDecoder, GreedyDecoder
from .problem import OTSchedulingProblem
from .evaluator import evaluate

__all__ = [
    "Instance",
    "OT",
    "Team",
    "Surgery",
    "SurgeryType",
    "Schedule",
    "Assignment",
    "BaseDecoder",
    "GreedyDecoder",
    "OTSchedulingProblem",
    "evaluate",
]
