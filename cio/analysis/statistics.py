"""Statistical tests for comparing algorithm configurations.

Standard tooling for the experimental-protocol requirement:
    friedman(data)             - Friedman test across configurations
    holm_pairwise(data)        - Pairwise Wilcoxon with Holm correction
    nemenyi_pairwise(data)     - Pairwise Nemenyi post-hoc
    vargha_delaney(a, b)       - Effect size (Vargha-Delaney A-measure)

All tests expect `data` as a 2-D array of shape (n_runs, n_configurations).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
import scikit_posthocs as sp


def friedman(data: np.ndarray) -> dict:
    """Friedman test.

    Returns dict with keys:
        statistic, p_value, n_runs, n_configurations
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data must be 2-D with at least 2 configurations")
    columns = [data[:, j] for j in range(data.shape[1])]
    stat, p = sps.friedmanchisquare(*columns)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_runs": int(data.shape[0]),
        "n_configurations": int(data.shape[1]),
    }


def holm_pairwise(data: np.ndarray) -> np.ndarray:
    """Pairwise Wilcoxon signed-rank with Holm correction.

    Returns shape (n_configs, n_configs) matrix of corrected p-values.
    """
    data = np.asarray(data)
    return sp.posthoc_wilcoxon(data, p_adjust="holm")


def nemenyi_pairwise(data: np.ndarray) -> np.ndarray:
    """Friedman + Nemenyi post-hoc.

    Returns shape (n_configs, n_configs) matrix of corrected p-values.
    """
    data = np.asarray(data)
    return sp.posthoc_nemenyi_friedman(data)


def vargha_delaney(a: np.ndarray, b: np.ndarray) -> float:
    """Vargha-Delaney A-measure (probability of superiority).

    Returns: A in [0, 1]. A=0.5 -> no effect; A>0.5 -> a tends to exceed b.
    Magnitude conventions (Vargha & Delaney, 2000):
        |A - 0.5| < 0.06 : negligible
        |A - 0.5| < 0.14 : small
        |A - 0.5| < 0.21 : medium
        otherwise: large
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.5
    # Pairwise comparisons
    gt = 0
    eq = 0
    for x in a:
        gt += int(np.sum(b < x))
        eq += int(np.sum(b == x))
    return (gt + 0.5 * eq) / (n_a * n_b)


def effect_size_magnitude(A: float) -> str:
    """Return the magnitude label for a Vargha-Delaney A-measure."""
    diff = abs(A - 0.5)
    if diff < 0.06:
        return "negligible"
    if diff < 0.14:
        return "small"
    if diff < 0.21:
        return "medium"
    return "large"
