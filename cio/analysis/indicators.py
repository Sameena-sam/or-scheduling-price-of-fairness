"""Pareto-front quality indicators.

Wraps pymoo's indicator implementations and adds a couple of conveniences.

Indicators provided:
    hypervolume(F, ref_point)        - HV (higher = better)
    igd_plus(F, reference_front)     - IGD+ (lower = better)
    spread(F)                        - spread / maximum spread (lower = better
                                       for spread metric)
"""
from __future__ import annotations

import numpy as np

from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus


def hypervolume(F: np.ndarray, ref_point: np.ndarray | list | tuple) -> float:
    """Compute hypervolume of a non-dominated front w.r.t. a reference point.

    Args:
        F: shape (n, n_obj) array of objective values (assumed minimisation)
        ref_point: shape (n_obj,) reference point dominated by all points in F
    """
    hv = HV(ref_point=np.asarray(ref_point, dtype=float))
    return float(hv(F))


def igd_plus(F: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute IGD+ between F and a known reference front.

    Args:
        F: shape (n, n_obj) approximation set
        reference_front: shape (m, n_obj) reference Pareto front
    """
    ind = IGDPlus(reference_front)
    return float(ind(F))


def spread(F: np.ndarray) -> float:
    """Maximum spread: sum across objectives of (max - min) on that objective.

    Higher is better (more diverse front). Use the negation if you want a
    "lower-is-better" indicator for consistency in statistical tests.
    """
    F = np.asarray(F)
    if F.shape[0] < 2:
        return 0.0
    return float(np.sum(F.max(axis=0) - F.min(axis=0)))


def non_dominated(F: np.ndarray) -> np.ndarray:
    """Return a boolean mask selecting non-dominated rows of F."""
    F = np.asarray(F)
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # j dominates i?
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                mask[i] = False
                break
    return mask


def union_non_dominated(*fronts: np.ndarray) -> np.ndarray:
    """Union several fronts and return only the non-dominated solutions.

    Useful for constructing reference fronts.
    """
    if not fronts:
        return np.empty((0, 0))
    F = np.vstack([f for f in fronts if f.size > 0])
    return F[non_dominated(F)]
