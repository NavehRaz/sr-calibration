"""The per-node truncated-normal mixture, its ladder and its check (tasks T012; research R3-R4; contracts/python-api.md
§2; data-model "Ladder and check", "Mixture result"). Each verdict has a planted case beside its twin; normal_frac_sd's
fused path is pinned unchanged by its program text."""
import hashlib
import math

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine import mixture as MIX
from sraging.engine import quadrature as Q
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import Heterogeneity
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(3)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 300)) + 1.0)
E = (RNG.random(300) < 0.9).astype(int)
BROAD = Cohort.from_arrays(T, E, delta=1.0)                       # tests/unit/engine/test_mixture.py's cohort
GRID = GridPolicy("fixed", N=200, dt=2.0 ** -8, tau_max=2.5, L=124.0)
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)

# a sharp cohort: library v6's synthetic sharp truth (het-k0.01-p00-c1-s0: CV_c 0.04, L 100), with lifetimes spread by
# x_c heterogeneity at sd 0.2 (T ~ sqrt(x_c) near the drift-dominated limit) -- each death pins its node narrowly
P_SHARP = SRParams(eta=2.125427697723628e-4, beta=1e-3, epsilon=3.9810717055349735e-5, kappa=0.01)
_R = np.random.default_rng(11)
_z = Q.truncated_normal_rule(0.2, 192).z
_s = 1.0 + 0.2 * _R.standard_normal(1000)
_s = _s[_s > 0.05][:800]
SHARP = Cohort.from_arrays(np.round(100.0 * np.sqrt(_s) * (1.0 + 0.03 * _R.standard_normal(_s.size))),
                           np.ones(_s.size, int), delta=1.0)
GRID_SHARP = GridPolicy("fixed", N=200, dt=2.0 ** -9, tau_max=2.5, L=100.0)


def ev(het, cohort=BROAD, grid=GRID, **kw):
    return Evaluator(cohort, numerics=kw.pop("numerics", "v1"), grid=grid, heterogeneity=het, **kw)


@pytest.fixture(scope="module")
def broad_auto():
    return ev(Heterogeneity(parameter="xc", sd=0.2))


@pytest.fixture(scope="module")
def sharp_fixed11():
    return ev(Heterogeneity(parameter="xc", sd=0.2, M=11), SHARP, GRID_SHARP)


# ------------------------------------------------------------------------------------------------------ the ladder
def test_the_ladder_constants():
    assert MIX.RUNGS == (5, 8, 11, 17, 25, 38, 57, 85, 128, 192)
    assert MIX.THRESHOLD == 0.05 and MIX.CAP == 192
    assert (MIX.C_R, MIX.G_N0, MIX.G_EXP) == (10.0, 10_000, 0.3)


@pytest.mark.parametrize("r,n,want", [(0.0, 1000, 5), (0.5, 1000, 5), (0.51, 1000, 8), (1.05, 10_000, 11),
                                      (1.95, 10_000, 25), (1.95, 100_000, 57), (2.84, 10_000, 38), (30.0, 10, 192)])
def test_the_start_is_the_smallest_rung_at_least_ten_r_g(r, n, want):
    assert MIX.g(n) == (1.0 if n <= 10_000 else (n / 10_000) ** 0.3)
    assert MIX.start_rung(r, n) == want


@pytest.mark.parametrize("M,want", [(11, 17), (38, 57), (128, 192), (5, 8), (8, 17), (17, 38), (1, 5), (10, 17)])
def test_a_fixed_m_is_checked_against_the_smallest_rung_at_least_one_and_a_half_m(M, want):
    assert MIX.check_rung(M) == want


def test_the_start_comes_from_the_model_r_at_theta_and_sd():
    """OD-10: r = sqrt(CV_mix^2 - CV_c^2) / CV_c with CV_c from the homogeneous march and CV_mix from the 5-node rule,
    recomputed here from homogeneous evaluators at the node values (beta heterogeneity: one compiled evaluator)."""
    e = ev(Heterogeneity(parameter="beta", sd=0.3))
    rec = e.mixture_value(P).node_record["check"]["start"]
    hom = ev(None)
    rule = Q.truncated_normal_rule(0.3, 5)

    def moments(beta):
        tau, S, _, _ = hom.survival_on_grid(SRParams(eta=P.eta, beta=beta, epsilon=P.epsilon, kappa=P.kappa))
        return np.trapezoid(S, tau), np.trapezoid(2.0 * tau * S, tau)
    m0 = moments(P.beta)
    cv_c = math.sqrt(m0[1] - m0[0] ** 2) / m0[0]
    mk = np.array([moments(P.beta * s) for s in rule.s])
    m1, m2 = float(rule.w @ mk[:, 0]), float(rule.w @ mk[:, 1])
    cv_mix = math.sqrt(m2 - m1 ** 2) / m1
    r = math.sqrt(max(cv_mix ** 2 - cv_c ** 2, 0.0)) / cv_c
    assert rec["r"] == pytest.approx(r, rel=1e-9) and rec["n"] == BROAD.n
    assert rec["guide"] == pytest.approx(10.0 * r * MIX.g(BROAD.n), rel=1e-9)
    assert rec["rung"] == MIX.start_rung(rec["r"], BROAD.n)


class _FakeProgs:
    """The guide's inputs only: a march grid and a survival per scale vector (hr3's crash, Naveh 2026-10-30)."""

    def __init__(self, tau, S_of):
        from types import SimpleNamespace
        self.ev = SimpleNamespace(_asm=SimpleNamespace(tau=np.asarray(tau, float)),
                                  het=SimpleNamespace(parameter="epsilon"))
        self._S = S_of

    def survival(self, z, s):
        return self._S(np.asarray(s, float))


def _old_guide(progs, z, sd, n):
    """The guide as it was before the fix (2f09ab9), for the twin."""
    tau = np.asarray(progs.ev._asm.tau, float)

    def moments(s):
        S = progs.survival(z, s)
        return float(np.trapezoid(S, tau)), float(np.trapezoid(2.0 * tau * S, tau))
    m0 = moments(np.ones(len(MIX.SCALED)))
    rule = Q.truncated_normal_rule(sd, MIX.GUIDE_M)
    mk = np.array([moments(MIX._scales("epsilon", float(sk))) for sk in rule.s])
    m1, m2 = float(rule.w @ mk[:, 0]), float(rule.w @ mk[:, 1])
    cv_c, cv_mix = MIX._cv(*m0), MIX._cv(m1, m2)
    r = math.sqrt(max(cv_mix ** 2 - cv_c ** 2, 0.0)) / cv_c
    return {"rung": MIX.start_rung(r, n), "guide": MIX.C_R * r * MIX.g(n), "r": r, "n": int(n), "cv_c": cv_c,
            "cv_mix": cv_mix}


def test_a_degenerate_point_starts_the_ladder_at_the_lowest_rung_and_says_why():
    flat = _FakeProgs([0.0, 1.0], lambda s: np.ones(2))           # m1 = 1, m2 = 1: m2 - m1^2 = 0, so cv_c = 0
    with pytest.raises(ZeroDivisionError):
        _old_guide(flat, None, 0.2, 500)                           # what hr3 hit
    rec = MIX.guide(flat, None, 0.2, 500)
    assert rec["rung"] == MIX.RUNGS[0] and rec["r"] is None and rec["guide"] is None and rec["cv_c"] == 0.0
    assert "cv_c = 0" in rec["undefined"] and rec["n"] == 500


def test_an_ordinary_point_gives_exactly_the_dict_it_gave_before():
    tau = np.linspace(0.0, 3.0, 601)
    progs = _FakeProgs(tau, lambda s: np.exp(-(tau * s[2] * 1.3) ** 2))      # epsilon's slot scales the curve
    new, old = MIX.guide(progs, None, 0.2, 500), _old_guide(progs, None, 0.2, 500)
    assert new == old and "undefined" not in new and new["r"] > 0


def test_the_ladder_and_the_series_take_an_undefined_guide(monkeypatch):
    from sraging.engine.series import AdaptivePolicy, Series
    real = MIX.guide

    def undefined(progs, z, sd, n):
        rec = real(progs, z, sd, n)
        return {**rec, "rung": MIX.RUNGS[0], "guide": None, "r": None, "undefined": "planted"}
    monkeypatch.setattr(MIX, "guide", undefined)
    e = ev(Heterogeneity(parameter="xc", sd=0.2))
    r = e.mixture_value(P)
    c = r.node_record["check"]
    assert c["start"]["undefined"] == "planted" and c["ladder"][0]["M"] == MIX.RUNGS[0] and c["verdict"] == "converged"
    s = Series(e, AdaptivePolicy(), start=P)
    assert s.M_low == MIX.RUNGS[0]


# ------------------------------------------------------------------------------------------------- the verdicts
def test_converged_returns_the_value_at_the_passing_rung_and_is_clean(broad_auto):
    r = broad_auto.mixture_value(P)
    c = r.node_record["check"]
    assert c["verdict"] == "converged" and r.clean and r.warnings == ()
    assert c["abs_dll"] < 0.05 and c["abs_dll"] == abs(c["ll_M"] - c["ll_check"])
    assert r.value == c["ll_M"] and r.node_record["M_used"] == c["M"]
    assert c["M_check"] == MIX.RUNGS[MIX.RUNGS.index(c["M"]) + 1]
    assert [x["M"] for x in c["ladder"]][-2:] == [c["M"], c["M_check"]]
    assert c["ladder"][0]["M"] == c["start"]["rung"] and all(x["cpu_s"] >= 0 for x in c["ladder"])
    rule = Q.truncated_normal_rule(0.2, c["M"])
    assert r.node_record["rule_sha256"] == rule.sha256 and r.node_record["s_min"] == rule.s_min
    assert (r.node_record["definition"], r.node_record["parameter"], r.node_record["sd"]) == ("truncated_normal", "xc", 0.2)


def test_unconverged_at_the_cap_returns_l192_warns_and_is_not_clean(broad_auto, monkeypatch):
    """Planted: no pair can pass a threshold of 0, so the ladder climbs to the cap."""
    twin = broad_auto.mixture_value(P)
    assert twin.node_record["check"]["verdict"] == "converged"                                     # the twin
    monkeypatch.setattr(MIX, "THRESHOLD", 0.0)
    r = broad_auto.mixture_value(P)
    c = r.node_record["check"]
    assert c["verdict"] == "unconverged_at_cap" and (c["M"], c["M_check"]) == (128, 192)
    assert r.node_record["M_used"] == 192 and r.value == c["ll_check"] and not r.clean
    assert [x["M"] for x in c["ladder"]][-1] == 192 and max(x["M"] for x in c["ladder"]) == 192
    (w,) = r.warnings
    assert w.code == "HET_UNCONVERGED" and w.data == {"M": 128, "M_check": 192, "abs_dll": c["abs_dll"],
                                                       "threshold": 0.0}
    assert "M = 192" in w.message and "ℓ(128) − ℓ(192)" in w.message


def test_not_checked_is_never_clean():
    r = ev(Heterogeneity(parameter="xc", sd=0.2, M=11, check=False)).mixture_value(P)
    c = r.node_record["check"]
    assert c["verdict"] == "not_checked" and c["M_check"] is None and r.node_record["M_used"] == 11
    assert [w.code for w in r.warnings] == ["HET_NOT_CHECKED"] and not r.clean
    t = ev(Heterogeneity(parameter="xc", sd=0.2, M=11)).mixture_value(P)                          # the twin
    assert t.node_record["check"]["verdict"] == "converged" and t.clean and t.value == r.value


def test_a_forced_m11_on_a_sharp_cohort_fires_the_check_against_m17(sharp_fixed11):
    """US1 scenario 2: the warning names the measured change |l(11) - l(17)|."""
    r = sharp_fixed11.mixture_value(P_SHARP)
    c = r.node_record["check"]
    assert (c["M"], c["M_check"], c["verdict"]) == (11, 17, "unconverged") and not r.clean
    assert c["abs_dll"] >= 0.05 and r.value == c["ll_M"]
    (w,) = r.warnings
    assert w.code == "HET_UNCONVERGED" and (w.data["M"], w.data["M_check"]) == (11, 17)
    assert w.data["abs_dll"] == c["abs_dll"] and "ℓ(11) − ℓ(17)" in w.message
    t = ev(Heterogeneity(parameter="xc", sd=0.2, M=11)).mixture_value(P)                          # the twin: broad
    assert t.node_record["check"]["verdict"] == "converged" and t.clean


def test_a_checked_fixed_m_is_compared_with_the_smallest_rung_at_least_one_and_a_half_m():
    r = ev(Heterogeneity(parameter="xc", sd=0.2, M=38)).mixture_value(P)
    c = r.node_record["check"]
    assert (c["M"], c["M_check"]) == (38, 57) and [x["M"] for x in c["ladder"]] == [38, 57]
    assert c["start"] is None and r.node_record["M_used"] == 38


def test_an_sd_above_the_measured_range_is_not_clean():
    r = ev(Heterogeneity(parameter="xc", sd=0.6, M=11, check=False)).mixture_value(P)
    assert {w.code for w in r.warnings} == {"HET_OUTSIDE_MEASURED_RANGE", "HET_NOT_CHECKED"} and not r.clean


# -------------------------------------------------------------------------------------------- refused methods
@pytest.mark.parametrize("method", ["value", "value_and_grad", "terms"])
def test_the_plain_methods_are_refused_with_automatic_m(broad_auto, method):
    with pytest.raises(UnsupportedOptionError, match="use mixture_value"):
        getattr(broad_auto, method)(P)


def test_weighted_grad_and_objective_are_refused(broad_auto):
    with pytest.raises(UnsupportedOptionError, match="use mixture_value"):
        broad_auto.weighted_grad(P, np.ones(BROAD.n))
    with pytest.raises(UnsupportedOptionError, match="not one jitted program"):
        broad_auto.objective()
    with pytest.raises(UnsupportedOptionError, match="not one jitted program"):
        ev(Heterogeneity(parameter="xc", sd=0.2, M=11)).objective()


def test_with_a_fixed_m_the_plain_methods_are_the_mixtures_bit_for_bit():
    e = ev(Heterogeneity(parameter="xc", sd=0.2, M=11))
    r = e.mixture_value(P, grad=True)
    assert e.value(P) == r.value
    v, g = e.value_and_grad(P)
    assert v == r.value and g == {k: r.grad[k] for k in ("eta", "beta", "epsilon", "kappa")}
    terms, aux = e.terms(P)
    assert math.isclose(float(np.sum(terms)), r.value, rel_tol=1e-13) and terms.shape == (BROAD.n,)
    wg = e.weighted_grad(P, np.ones(BROAD.n))
    assert all(math.isclose(wg[k], g[k], rel_tol=1e-10) for k in g)


def test_a_pool_that_is_not_a_nodepool_is_refused():
    with pytest.raises(UnsupportedOptionError) as ex:
        ev(Heterogeneity(parameter="xc", sd=0.2, M=5)).mixture_value(P, pool=object())
    assert ex.value.option == "pool" and "NodePool" in str(ex.value)


def test_mixture_value_needs_the_truncated_normal():
    with pytest.raises(UnsupportedOptionError):
        ev(None).mixture_value(P)
    with pytest.raises(UnsupportedOptionError, match="normal_frac_sd"):
        ev(Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7)).mixture_value(P)


def test_a_node_with_a_non_finite_point_is_refused_naming_it_and_its_s(monkeypatch):
    e = ev(Heterogeneity(parameter="xc", sd=0.2, M=11, check=False))
    progs = MIX.programs(e)
    rule = Q.truncated_normal_rule(0.2, 11)
    good = progs.points
    target = rule.s[3]

    def planted(z, s):
        pts = good(z, s)
        if s[4] == target:
            pts = dict(pts, S_hi=np.where(np.arange(pts["S_hi"].size) == 7, np.nan, pts["S_hi"]))
        return pts
    twin = e.mixture_value(P)
    assert np.isfinite(twin.value)                                                                 # the twin
    monkeypatch.setattr(progs, "points", planted)
    with pytest.raises(MIX.NonFiniteNode) as ex:
        e.mixture_value(P)
    assert ex.value.k == 3 and ex.value.s == target and "node k = 3" in str(ex.value) and "S_hi" in str(ex.value)


# ------------------------------------------------------------------------------------------------ m_report: whole
def test_m_report_gains_the_whole_block_for_the_truncated_normal(broad_auto):
    rep = broad_auto.m_report(P, early_S=0.99)
    w = rep["whole"]
    r = broad_auto.mixture_value(P)
    c = r.node_record["check"]
    assert set(w) == {"M", "M_more", "ll_M", "ll_more", "abs_dll", "threshold", "verdict"}
    assert (w["M"], w["M_more"], w["ll_M"], w["ll_more"], w["abs_dll"], w["threshold"], w["verdict"]) == \
        (c["M"], c["M_check"], c["ll_M"], c["ll_check"], c["abs_dll"], 0.05, "converged")
    assert rep["M"] == c["M"] and rep["M_more"] == c["M_check"]


def test_m_report_gains_the_whole_block_for_normal_frac_sd():
    e = ev(Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7), numerics="v033")
    rep = e.m_report(P, early_S=0.99)
    w = rep["whole"]
    more = ev(Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=11), numerics="v033")
    assert (w["M"], w["M_more"], w["threshold"]) == (7, 11, 0.05)
    assert w["ll_M"] == e.value(P) and w["ll_more"] == more.value(P)
    assert w["abs_dll"] == abs(w["ll_M"] - w["ll_more"])
    assert w["verdict"] == ("converged" if w["abs_dll"] < 0.05 else "unconverged")
    assert rep["M"] == 7 and rep["M_more"] == 11 and "dterm" in rep                    # the early-death fields stay


# ------------------------------------------------------------------------------------- per-node against vmapped
@pytest.mark.parametrize("par,sd,M", [("xc", 0.2, 11), ("eta", 0.3, 17), ("epsilon", 0.2, 25)])
def test_the_per_node_mixture_and_todays_vmapped_one_agree(par, sd, M):
    rule = Q.truncated_normal_rule(sd, M)
    old = ev(Heterogeneity(parameter=par, definition="normal_frac_sd", v=sd, M=M))
    old._het_z, old._het_w = np.asarray(rule.z), np.asarray(rule.w)        # today's program, the new nodes
    new = ev(Heterogeneity(parameter=par, sd=sd, M=M, check=False)).mixture_value(P).value
    assert abs(new - old.value(P)) <= 1e-8 * abs(old.value(P))


def test_the_per_node_mixture_agrees_on_the_general_march_with_a_hazard_and_its_ladder_runs_there():
    """Segments take the general march (stepper_gen) and the guide its survival; a hazard enters the assembly."""
    from sraging.engine.inputs import AgeSegments, Segment
    kw = dict(segments=AgeSegments(Segment(id="s", parameter="eta", value=1.3, window=(60.0, 120.0))),
              fixed={"h_ext": 1e-3})
    p = SRParams(**{**P.as_dict(), "h_ext": 1e-3})
    rule = Q.truncated_normal_rule(0.2, 11)
    old = ev(Heterogeneity(parameter="beta", definition="normal_frac_sd", v=0.2, M=11), **kw)
    old._het_z, old._het_w = np.asarray(rule.z), np.asarray(rule.w)
    e = ev(Heterogeneity(parameter="beta", sd=0.2, M=11, check=False), **kw)
    assert abs(e.mixture_value(p).value - old.value(p)) <= 1e-8 * abs(old.value(p))
    r = MIX.mixture_value(e, p, het=Heterogeneity(parameter="beta", sd=0.2))
    assert r.node_record["check"]["verdict"] == "converged" and r.node_record["check"]["start"]["r"] > 0


# ------------------------------------------------------------------------- normal_frac_sd's path is untouched
#: the canonical program text's sha256 at 3a7b3ce (before spec 004 touched src/), this file's cohort and grid
PINNED_TEXT = {
    ("v033", "value"): "9728ca75c3b859f8f2547be6f1c0151c1978b4fc9709d2f49136a487ff373fa4",
    ("v033", "value_and_grad"): "051f6d65bc96d94e94e57c55ff7543a18b358af52fd09fc3a7df8b987d8ea9c4",
    ("v1", "value"): "9138ea0967a2f43e0b2f753d8a61aa08cfc719ebcb1ac2ee456da5178b73c423",
    ("v1", "value_and_grad"): "4540d4613736c7b7ddd59adeee26e9aec3ed1f3dddfd5fb67d8cd8773a020d8d",
}
#: its value and gradient bytes at 3a7b3ce
PINNED_BYTES = {"v033": ("-0x1.7b2509b49f151p+10", {"eta": "-0x1.7963ce5a23c68p+9", "beta": "0x1.355e41477128ap+9",
                                                    "epsilon": "-0x1.d8ac2be3cd537p+6",
                                                    "kappa": "-0x1.84a2694c85ca8p+6"}),
                "v1": ("-0x1.7b2509b49f16dp+10", {"eta": "-0x1.7963ce5a23c0cp+9", "beta": "0x1.355e414771205p+9",
                                                  "epsilon": "-0x1.d8ac2be3cee1bp+6", "kappa": "-0x1.84a2694c85bfap+6"})}


@pytest.mark.parametrize("numerics", ["v033", "v1"])
def test_normal_frac_sds_fused_path_is_unchanged(numerics):
    e = Evaluator(BROAD, numerics=numerics, grid=GRID,
                  heterogeneity=Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7))
    for kind in ("value", "value_and_grad"):
        assert hashlib.sha256(e.program_text(kind).encode()).hexdigest() == PINNED_TEXT[(numerics, kind)], kind
    v, g = e.value_and_grad(P)
    want_v, want_g = PINNED_BYTES[numerics]
    assert e.value(P).hex() == v.hex() == want_v and {k: x.hex() for k, x in g.items()} == want_g
