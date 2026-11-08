"""Algorithm builders for baselines.

    build_nsga2          - NSGA-II with permutation operators
    build_moead          - MOEA/D with permutation operators (Tchebycheff)
    pareto_local_search  - Stand-alone Pareto local search
"""
from .builders import build_nsga2, build_moead
from .pls import pareto_local_search, PLSResult

__all__ = ["build_nsga2", "build_moead", "pareto_local_search", "PLSResult"]
