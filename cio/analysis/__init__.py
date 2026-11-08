"""Indicators and statistical tests for Pareto-front evaluation."""
from .indicators import hypervolume, igd_plus, spread, non_dominated, union_non_dominated
from .statistics import (
    friedman, holm_pairwise, nemenyi_pairwise,
    vargha_delaney, effect_size_magnitude,
)

__all__ = [
    "hypervolume", "igd_plus", "spread", "non_dominated", "union_non_dominated",
    "friedman", "holm_pairwise", "nemenyi_pairwise",
    "vargha_delaney", "effect_size_magnitude",
]
