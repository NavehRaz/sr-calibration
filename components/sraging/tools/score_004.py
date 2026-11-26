"""Spec 004's S5, homogeneous half (tasks T089, T127; research R20): spec 002's simulator-anchored score harness
(`tools/score_test.py`, untouched) with its simulator replaced by sraging's bridge kernel at the default wall (clamp).

score_test.simulate draws through the oracle worker's `simulate_sr_hetro` (SRK's frozen kernel, numba seeded per
cohort) and then applies the harness's own observation transform. Here the oracle session is replaced, in this
process, by `SraSession`, whose `call("simulate_sr_hetro", ...)` draws the same configuration with
`sraging.langevin.simulate` (the cohort's seed as its key, SRK's horizon (steps - 1) dt as t_max, error="ks"); every
other line of the harness, its transform and its verdict rules, are 002's.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VARS = ("eta", "beta", "epsilon", "kappa", "xc")


class SraSession:
    """The oracle session's one operation the harness calls, on sraging's kernel."""

    def call(self, op, **kw):
        if op != "simulate_sr_hetro":
            raise ValueError(f"score_004 replaces simulate_sr_hetro only, not {op!r}")
        from sraging import SRParams
        from sraging.engine.inputs import Heterogeneity
        from sraging.langevin import simulate
        p = SRParams(eta=float(kw["eta"]), beta=float(kw["beta"]), epsilon=float(kw["epsilon"]), kappa=float(kw["kappa"]),
                     xc=float(kw.get("xc", 1.0)),
                     h_ext=math.exp(-float(kw["external_hazard"])) if "external_hazard" in kw else 0.0)
        het = [Heterogeneity(parameter=v, sd=float(kw[v + "_var"])) for v in VARS if float(kw.get(v + "_var", 0)) > 0]
        if len(het) > 1:
            raise ValueError("one heterogeneous parameter per configuration")
        t_max = (int(kw["steps"]) - 1) * float(kw["dt"])
        s = simulate(p, n=int(kw["npeople"]), key=int(kw["seed"]), dt=float(kw["dt"]), t_max=t_max,
                     heterogeneity=het[0] if het else None, error="ks")
        return {"death_times": s.times, "events": (s.times < t_max).astype(int)}


@contextlib.contextmanager
def session():
    yield SraSession()


def install():
    """Replace the oracle session the harness opens, in this process."""
    import oracle_fp_piecewise as OP
    OP.session = session


def run_feature(feat, zcrit, smoke=False) -> dict:
    """One of 002's features, its verdict by 002's rules, on sraging's kernel."""
    import score_test as ST
    install()
    return ST.run_feature(feat, zcrit, smoke)


def zcrit() -> tuple:
    """002's Bonferroni bound over every (feature, component) pair at family-wise alpha 0.01."""
    from statistics import NormalDist
    import acceptance_002 as A
    import score_test as ST
    n_pairs = sum(len(A.PARAMS) + (1 if "ext" in ST.FEATURES[f] else 0) for f in ST.FEATURES)
    return NormalDist().inv_cdf(1 - ST.ALPHA / (2 * n_pairs)), n_pairs


def summarise(features: dict) -> str:
    """002's close: pass iff every feature passes with its engine grid certified."""
    ok = all(v.get("verdict") == "pass" and v.get("engine_grid_certified") for v in features.values())
    return "pass" if ok and features else "fail"

