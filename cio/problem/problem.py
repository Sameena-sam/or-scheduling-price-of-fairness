"""pymoo Problem subclass for the OT scheduling problem.

Solution representation:
    A permutation of integers 0..(n_surgeries-1) giving the order in which
    surgeries are placed by the decoder. pymoo's vectorised Problem interface
    expects a 2D numpy array of shape (pop_size, n_var); we treat each row as
    one permutation.
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .decoder import BaseDecoder, GreedyDecoder
from .evaluator import evaluate
from .instance import Instance


class OTSchedulingProblem(ElementwiseProblem):
    """Three-objective constrained permutation problem.

    f1: minimise (-priority-weighted throughput)
    f2: minimise resource utilisation imbalance
    f3: minimise waiting-time fairness (variance across priority classes)

    Constraints are handled in-decoder, so this problem has no explicit
    constraint output to pymoo (feasibility is guaranteed by construction).
    Students who design alternative constraint-handling schemes should
    subclass this problem.
    """

    def __init__(self, instance: Instance, decoder: BaseDecoder | None = None):
        self.instance = instance
        self.decoder = decoder or GreedyDecoder()
        # n_var = number of surgeries; we use a permutation encoding
        super().__init__(
            n_var=instance.n_surgeries,
            n_obj=3,
            n_constr=0,
            xl=0,
            xu=instance.n_surgeries - 1,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        perm = np.asarray(x, dtype=int)
        # Validate that x is a permutation (defensive; pymoo permutation
        # operators should preserve this, but custom operators might not).
        if perm.size != self.instance.n_surgeries or len(set(perm.tolist())) != perm.size:
            # Repair: take unique values in order, then append missing
            seen = set()
            repaired = []
            for v in perm.tolist():
                if v not in seen and 0 <= v < self.instance.n_surgeries:
                    repaired.append(v)
                    seen.add(v)
            for v in range(self.instance.n_surgeries):
                if v not in seen:
                    repaired.append(v)
            perm = np.asarray(repaired, dtype=int)
        schedule = self.decoder(self.instance, perm)
        f1, f2, f3 = evaluate(self.instance, schedule)
        out["F"] = [f1, f2, f3]
        # Stash the schedule on out for downstream analysis if anyone needs it
        out["schedule"] = schedule
