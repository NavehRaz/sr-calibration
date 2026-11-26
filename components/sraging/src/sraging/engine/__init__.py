"""sraging.engine: the generalised Fokker-Planck engine (spec 002; contracts/python-api.md).

    from sraging.engine import evaluator, curves, Segment, AgeSegments, Heterogeneity, GridPolicy
    r = evaluator(cohort, heterogeneity=Heterogeneity(parameter="xc", sd=0.2)).mixture_value(params)   # spec 004
    with NodePool(ev, workers=4) as pool: ev.mixture_value(params, grad=True, pool=pool)               # same bytes

Every input is named; every result carries its grid and numerics. Numerics "v033" reproduces v0.3.3
(and SRK's formulas for the features v0.3.3 lacks); "v1" (the default) differs from it only by the
divergence-register entries that carry a machine-checkable condition (D-004, D-106, D-107, D-108).
"""
from sraging.engine.curves import Curves, curves
from sraging.engine.evaluator import NUMERICS, Evaluator, evaluator
from sraging.engine.grid import Grid, GridPolicy
from sraging.engine.inputs import AgeSegments, Heterogeneity, Segment, SegmentConflictError
from sraging.engine.layout import ParamLayout
from sraging.engine.cost import CostStatement
from sraging.engine.mixture import CAP, RUNGS, THRESHOLD, MixtureResult, NonFiniteNode, mixture_value
from sraging.engine.pool import NodeIndicesError, NodePool
from sraging.engine.quadrature import Rule, truncated_normal_rule
from sraging.engine.series import AdaptivePolicy, MaximisationResult, SeriesRecord, maximise_fixed_sd

__all__ = ["Evaluator", "evaluator", "curves", "Curves", "Segment", "AgeSegments", "Heterogeneity",
           "SegmentConflictError", "GridPolicy", "Grid", "ParamLayout", "NUMERICS", "MixtureResult", "NonFiniteNode",
           "mixture_value", "RUNGS", "THRESHOLD", "CAP", "Rule", "truncated_normal_rule", "NodePool",
           "NodeIndicesError", "CostStatement", "AdaptivePolicy", "MaximisationResult", "SeriesRecord",
           "maximise_fixed_sd"]
