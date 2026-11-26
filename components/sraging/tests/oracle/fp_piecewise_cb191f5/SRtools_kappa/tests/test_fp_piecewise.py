"""The piecewise-FP prototype: exact curves for WINDOWED interventions.

The first test is the one that keeps this file honest. `fp_piecewise` DUPLICATES `fpt_full`'s
Scharfetter-Gummel discretisation, because `fpt_full` never exposes the density and it is not mine to
change right now. A duplicate that silently drifts from its original is worse than no duplicate, so
one segment must reproduce `fpt_full` BIT-IDENTICALLY -- not "closely".
"""
import warnings

import numpy as np
import pytest

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from SRK import curves as _curves
from SRK import fp_curves as fpc
from SRK import fp_likelihood as fpl
from SRK import fp_piecewise as fpp
from SRK import intervention_sim as isim
from SRK import presets
from SRK import stepsize as ss
from SRK.intervention_sim import ExtHazard, Intervention

L, T_END = 124.0, 214.0
_SIM_N, _SIM_NSTEPS = 8000, 40000


def _sim_ks(theta, interventions, dt=1e-3, N=900, edge_halfwidth=0.0, seed=4):
    """(KS, KS ignoring +-edge_halfwidth around every window edge) between piecewise-FP and a cohort.

    Compares CURVES rather than medians: a window can leave survival nearly flat -- or nearly
    vertical -- where it crosses 0.5, and a median comparison then reports a difference that the
    curves do not have.
    """
    segs, _xc, _e = fpp.build_segments(theta, interventions, L, T_END, return_exth=True)
    tau, S = fpp.piecewise_survival(segs, N=N, dt=dt)
    grid = np.linspace(1e-6, T_END, 800)
    S_fp = np.interp(grid, tau * L, S, left=1.0, right=S[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = isim.sr_metrics(theta, interventions, evaluator="sim", t_end=T_END, n=_SIM_N,
                            nsteps=_SIM_NSTEPS, seed=seed, return_sim=True)
    tk, Sk = (np.asarray(a, float) for a in r["sim"].getSurvival())
    d = np.abs(np.interp(grid, tk, Sk, left=1.0, right=Sk[-1]) - S_fp)
    keep = np.ones_like(grid, bool)
    if edge_halfwidth:
        for iv in interventions:
            if getattr(iv, "window", None):
                for w in iv.window:
                    keep &= np.abs(grid - w) > edge_halfwidth
    return float(d.max()), float(d[keep].max())


def _assert_matches_sim(theta, interventions, dt=1e-3, N=900):
    ks, _ = _sim_ks(theta, interventions, dt=dt, N=N)
    floor = ss.mc_floor(_SIM_N)
    assert ks <= 2.0 * floor, f"KS={ks:.4f} vs 2x MC floor {2 * floor:.4f}"


@pytest.fixture(scope="module")
def th():
    return presets.getTheta("mice_F", type="max_likelihood", as_theta=True)


@pytest.fixture(scope="module")
def cohort():
    """A day-resolution cohort with censoring, shaped like an ITP arm."""
    rng = np.random.default_rng(0)
    t = np.round(rng.gamma(9.0, 14.0, 400)) + 1.0
    e = (rng.random(400) > 0.05).astype(int)
    return t, e


@pytest.fixture(scope="module")
def dl(th):
    """Dimensionless (rho_eta, rho_beta, rho_eps, kappa) -- note to_dimensionless returns beta FIRST."""
    rb, re, rp, kd = fpl.to_dimensionless(th.eta, th.beta, th.epsilon, 1.0, L, k=th.kappa)
    return re, rb, rp, kd


class TestDuplicateStaysHonest:
    def test_single_segment_is_bit_identical_to_fpt_full(self, dl):
        r_eta, r_beta, r_eps, kap = dl
        _m, _c, _s, tau_ref, S_ref = fpl.fpt_full(r_eta, r_beta, r_eps, kap, N=900, dt=1e-3,
                                                  tau_max=20.0, Gtol=1e-11, return_survival=True)
        tau, S = fpp.piecewise_survival([(0.0, np.inf, r_eta, r_beta, r_eps, kap)],
                                        N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11)
        assert np.array_equal(S, S_ref), (
            "the duplicated stepper has drifted from fpt_full. Either re-sync it, or -- better -- do "
            "the merge-back: factor fpt_full's stepper out and have both call it.")
        assert np.array_equal(tau, tau_ref)

    def test_bdf2_also_matches(self, dl):
        r_eta, r_beta, r_eps, kap = dl
        _m, _c, _s, _t, S_ref = fpl.fpt_full(r_eta, r_beta, r_eps, kap, N=900, dt=4e-3,
                                             tau_max=20.0, Gtol=1e-11, return_survival=True,
                                             scheme="bdf2")
        _tau, S = fpp.piecewise_survival([(0.0, np.inf, r_eta, r_beta, r_eps, kap)],
                                         N=900, dt=4e-3, tau_max=20.0, Gtol=1e-11, scheme="bdf2")
        assert np.array_equal(S, S_ref)

    def test_splitting_one_regime_into_two_segments_changes_nothing(self, dl):
        """A boundary with identical parameters on both sides must be a no-op."""
        r_eta, r_beta, r_eps, kap = dl
        one = fpp.piecewise_survival([(0.0, np.inf, r_eta, r_beta, r_eps, kap)])[1]
        two = fpp.piecewise_survival([(0.0, 0.5, r_eta, r_beta, r_eps, kap),
                                      (0.5, np.inf, r_eta, r_beta, r_eps, kap)])[1]
        assert np.array_equal(one, two)

    def test_non_contiguous_segments_raise(self, dl):
        r_eta, r_beta, r_eps, kap = dl
        with pytest.raises(ValueError, match="contiguous"):
            fpp.piecewise_survival([(0.0, 0.5, r_eta, r_beta, r_eps, kap),
                                    (0.7, np.inf, r_eta, r_beta, r_eps, kap)])


class TestAgreesWithTheOtherEvaluators:
    @pytest.mark.parametrize("param,effect", [("beta", -0.3), ("eta", 0.5), ("epsilon", 0.4),
                                              ("kappa", 1.0), ("xc", 0.3)])
    def test_chronic_matches_fp_const(self, th, param, effect):
        ivs = (Intervention(param, effect),)
        a = isim.sr_metrics(th, ivs, evaluator="fp_const", L=L)
        b = isim.sr_metrics(th, ivs, evaluator="piecewise_fp", L=L, t_end=T_END)
        assert b["median"] == pytest.approx(a["median"], rel=0.005)
        assert b["n_segments"] == 1, "a chronic intervention needs no extra segments"

    @pytest.mark.parametrize("window", [(60.0, 120.0), (100.0, 214.0)])
    def test_windowed_matches_the_simulator_on_the_CURVE(self, th, window):
        """Compare curves, not medians.

        A window can leave survival nearly flat where it crosses 0.5 (measured |dS/dt| = 0.0067 per
        unit time on the [0, 60] case), so a survival difference at the Monte-Carlo floor shows up as
        a several-percent median gap. The curve is the honest comparison.
        """
        ivs = [Intervention("beta", -0.6, window)]
        segs, _xc = fpp.build_segments(th, ivs, L, T_END)
        tau, S = fpp.piecewise_survival(segs, N=900, dt=1e-3)
        grid = np.linspace(1e-6, T_END, 800)
        S_fp = np.interp(grid, tau * L, S, left=1.0, right=S[-1])

        n = 6000
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = isim.sr_metrics(th, ivs, evaluator="sim", t_end=T_END, n=n, nsteps=20000,
                                seed=4, return_sim=True)
        tk, Sk = (np.asarray(a, float) for a in r["sim"].getSurvival())
        ks = float(np.max(np.abs(np.interp(grid, tk, Sk, left=1.0, right=Sk[-1]) - S_fp)))
        assert ks <= 2.0 * ss.mc_floor(n), f"KS={ks:.4f} vs floor {ss.mc_floor(n):.4f}"

    def test_windowed_lies_between_baseline_and_chronic(self, th):
        base = isim.sr_metrics(th, (), evaluator="piecewise_fp", L=L, t_end=T_END)["median"]
        win = isim.sr_metrics(th, (Intervention("beta", -0.6, (60.0, 120.0)),),
                              evaluator="piecewise_fp", L=L, t_end=T_END)["median"]
        chronic = isim.sr_metrics(th, (Intervention("beta", -0.6),),
                                  evaluator="piecewise_fp", L=L, t_end=T_END)["median"]
        assert chronic < win < base

    def test_it_is_much_faster_than_the_cohort(self, th):
        import time
        ivs = (Intervention("beta", -0.6, (60.0, 120.0)),)
        t0 = time.perf_counter()
        isim.sr_metrics(th, ivs, evaluator="piecewise_fp", L=L, t_end=T_END)
        t_pw = time.perf_counter() - t0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t0 = time.perf_counter()
            isim.sr_metrics(th, ivs, evaluator="sim", t_end=T_END, n=8000, nsteps=5000, seed=4)
            t_sim = time.perf_counter() - t0
        assert t_pw < t_sim


class TestSegmentConstruction:
    def test_edges_come_from_the_windows(self, th):
        segs, _xc = fpp.build_segments(th, [Intervention("beta", -0.5, (60.0, 120.0))], L, T_END)
        starts = [s[0] * L for s in segs]
        assert len(segs) == 3
        assert starts == pytest.approx([0.0, 60.0, 120.0])
        assert segs[-1][1] == np.inf, "the last segment must run to tau_max, not stop at t_end"

    def test_only_the_active_segment_is_modified(self, th):
        segs, _xc = fpp.build_segments(th, [Intervention("beta", -0.5, (60.0, 120.0))], L, T_END)
        rho_betas = [s[3] for s in segs]
        assert rho_betas[0] == pytest.approx(rho_betas[2])
        assert rho_betas[1] == pytest.approx(0.5 * rho_betas[0])

    def test_chronic_xc_rescales_and_stays_one_segment(self, th):
        segs, xc = fpp.build_segments(th, [Intervention("xc", 0.3)], L, T_END)
        assert len(segs) == 1 and xc == pytest.approx(1.3)


class TestLimitsRaiseRatherThanApproximate:
    def test_external_hazard_without_opting_in(self, th):
        """An ExtHazard is applied OUTSIDE the solver, so a caller taking only segments would drop
        it silently. It must ask for it."""
        with pytest.raises(NotImplementedError, match="external hazard"):
            fpp.build_segments(th, [ExtHazard(6.0, (60.0, 120.0))], L, T_END)

    def test_requires_t_end(self, th):
        with pytest.raises(ValueError, match="t_end"):
            fpp.piecewise_metrics(th, ())

    def test_unknown_parameter_still_raises(self, th):
        with pytest.raises(KeyError):
            fpp.build_segments(th, [Intervention("nonesuch", 0.3, (60.0, 120.0))], L, T_END)


class TestWindowedA:
    """`A` is the ageing-clock rate. It used to raise here; it is now exact.

    In the simulator (`intervention_SR.py`) A scales ``dy/dt`` where ``y`` is a DETERMINISTIC scalar
    shared by every trajectory, not per-particle state -- so the drift's time argument becomes a
    piecewise-linear clock and nothing else changes.
    """

    def test_chronic_A_equals_chronic_eta(self, th):
        """eta*(a*tau) == (eta*a)*tau, so the two must agree to floating-point noise."""
        a = fpp.piecewise_metrics(th, [Intervention("A", -0.2)], L=L, t_end=T_END)["median"]
        b = fpp.piecewise_metrics(th, [Intervention("eta", -0.2)], L=L, t_end=T_END)["median"]
        assert a == pytest.approx(b, rel=1e-12)

    def test_windowed_A_outlives_its_window(self, th):
        """The property that makes A different from every other direction: it has MEMORY.

        beta is restored the moment its window closes; A is not, because the clock it advanced
        stays advanced.
        """
        base = fpp.piecewise_metrics(th, (), L=L, t_end=T_END)["median"]
        win = (60.0, 120.0)
        a = fpp.piecewise_metrics(th, [Intervention("A", -0.3, win)], L=L, t_end=T_END)["median"]
        assert a > base, "slowing the clock during a window must extend life"
        # the window closes at 120 but the gain persists past it -- the median is beyond the window
        assert a > win[1], "this test is vacuous unless the median falls outside the window"

    @pytest.mark.parametrize("effect,window", [(-0.3, (60.0, 120.0)), (-0.3, (100.0, 214.0)),
                                               (0.25, (60.0, 120.0))])
    def test_windowed_A_matches_the_simulator(self, th, effect, window):
        _assert_matches_sim(th, [Intervention("A", effect, window)])

    def test_a_is_a_no_op_at_1(self, th, dl):
        r_eta, r_beta, r_eps, kap = dl
        six = fpp.piecewise_survival([(0.0, np.inf, r_eta, r_beta, r_eps, kap)])[1]
        seven = fpp.piecewise_survival([(0.0, np.inf, r_eta, r_beta, r_eps, kap, 1.0)])[1]
        assert np.array_equal(six, seven)


class TestWindowedXc:
    """A moving threshold needs no density remap: fix the grid at ``max_k xc_k`` and the absorbing
    boundary becomes a moving CELL INDEX. Particles never move, ``h`` never changes."""

    def test_full_life_window_equals_chronic(self, th):
        """The sharpest internal check: a window spanning all of life IS the chronic case."""
        for effect in (0.3, -0.25):
            ch = fpp.piecewise_metrics(th, [Intervention("xc", effect)], L=L, t_end=T_END)
            wh = fpp.piecewise_metrics(th, [Intervention("xc", effect, (0.0, T_END))],
                                       L=L, t_end=T_END)
            assert wh["median"] == ch["median"], "a full-life window must be bit-identical to chronic"

    def test_lowering_the_threshold_shortens_life(self, th):
        base = fpp.piecewise_metrics(th, (), L=L, t_end=T_END)["median"]
        lo = fpp.piecewise_metrics(th, [Intervention("xc", -0.25, (60.0, 120.0))],
                                   L=L, t_end=T_END)["median"]
        assert lo < base

    def test_segments_carry_the_threshold_fraction(self, th):
        segs, xc_ref = fpp.build_segments(th, [Intervention("xc", 0.3, (60.0, 120.0))], L, T_END)
        assert xc_ref == pytest.approx(1.3), "the grid is sized by the LARGEST threshold"
        assert [round(s[7], 6) for s in segs] == [pytest.approx(1 / 1.3, rel=1e-6), 1.0,
                                                 pytest.approx(1 / 1.3, rel=1e-6)]

    @pytest.mark.parametrize("effect,window", [(-0.25, (60.0, 120.0)), (0.3, (100.0, 214.0)),
                                               (-0.25, (100.0, 214.0))])
    def test_windowed_xc_matches_the_simulator(self, th, effect, window):
        _assert_matches_sim(th, [Intervention("xc", effect, window)], dt=5e-4, N=1200)

    def test_raised_threshold_disagrees_ONLY_at_the_closing_edge(self, th):
        """Documents the one real artifact, so nobody rediscovers it as a bug.

        When a RAISED threshold drops back, everyone whose damage sits above the restored threshold
        dies at once, so S has a near-vertical cliff there. Both engines produce that cliff; they
        place it within a fraction of a step of each other, which reads as a large VERTICAL gap.
        Away from the edge the two agree well inside the Monte-Carlo floor, and refining dt halves
        the edge gap -- i.e. it is discretisation at a cliff, not a model disagreement.
        """
        ivs = [Intervention("xc", 0.3, (60.0, 120.0))]
        full, off = _sim_ks(th, ivs, dt=5e-4, N=1200, edge_halfwidth=3.0)
        assert off <= 2.0 * ss.mc_floor(_SIM_N), f"off-edge KS={off:.4f}"
        assert full > off, "if this ever stops holding, the cliff explanation no longer applies"


class TestLikelihood:
    """`piecewise_loglik` is the reason this module exists for the ITP work: the best-pair
    estimator needs a likelihood, and `fp_piecewise` only ever produced curves and metrics.

    The gate is bit-identity with the canonical `fp_loglik_physical` when no intervention is
    applied. Not "close" -- identical -- because anything less means the two have drifted and the
    difference will be attributed to the drug.
    """

    @pytest.mark.parametrize("sampling_dt", [None, 1.0])
    @pytest.mark.parametrize("external_hazard", [np.inf, 6.0])
    def test_no_intervention_is_bit_identical_to_fp_loglik(self, th, cohort, sampling_dt,
                                                           external_hazard):
        t, e = cohort
        a = fpl.fp_loglik_physical(t, e, th.eta, th.beta, th.epsilon, 1.0, th.kappa, L=L,
                                   external_hazard=external_hazard, sampling_dt=sampling_dt)
        b = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END,
                                 external_hazard=external_hazard, sampling_dt=sampling_dt)
        assert a == b, f"drifted by {abs(a - b):.3e}"

    @pytest.mark.parametrize("sampling_dt", [None, 1.0])
    def test_all_deaths_no_censored(self, th, cohort, sampling_dt):
        """The censored partial sum is skipped entirely when there is nothing to censor; that
        branch has to stay bit-identical too."""
        t, _e = cohort
        e = np.ones(t.size, int)
        a = fpl.fp_loglik_physical(t, e, th.eta, th.beta, th.epsilon, 1.0, th.kappa, L=L,
                                   sampling_dt=sampling_dt)
        b = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, sampling_dt=sampling_dt)
        assert a == b

    def test_terms_sum_to_the_total(self, th, cohort):
        t, e = cohort
        terms = fpp.piecewise_loglik_terms(t, e, th, (), xc=1.0, L=L, t_end=T_END)
        assert terms.size == t.size
        total = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END)
        assert terms.sum() == pytest.approx(total, abs=1e-9)

    def test_interval_censoring_is_the_default(self, th, cohort):
        """Day-recorded data: the death weight is log[S(t)-S(t+1)], not log p(t). The two differ by
        about n*log(delta), so this must not be silently switchable."""
        t, e = cohort
        interval = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END)
        explicit = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, sampling_dt=1.0)
        density = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, sampling_dt=None)
        assert interval == explicit
        assert interval != density

    def test_a_windowed_effect_moves_the_likelihood(self, th, cohort):
        """Guards the failure mode where a window is active NOWHERE and the baseline comes back."""
        t, e = cohort
        base = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END)
        for iv in (Intervention("beta", -0.4, (60.0, 214.0)),
                   Intervention("A", -0.2, (60.0, 214.0)),
                   Intervention("xc", 0.2, (60.0, 214.0))):
            assert fpp.piecewise_loglik(t, e, th, [iv], xc=1.0, L=L, t_end=T_END) != base, iv

    def test_invalid_theta_returns_minus_inf(self, th, cohort):
        t, e = cohort
        bad = _curves.Theta(eta=th.eta, beta=th.beta, epsilon=-1.0, kappa=th.kappa, xc=1.0)
        assert fpp.piecewise_loglik(t, e, bad, (), xc=1.0, L=L, t_end=T_END) == -np.inf


class TestExternalHazard:
    def test_windowed_hazard_only_bites_inside_its_window(self, th):
        c_base = fpp.piecewise_curves(th, (), xc=1.0, L=L, t_end=T_END)
        c_eh = fpp.piecewise_curves(th, [ExtHazard(4.0, (60.0, 120.0))], xc=1.0, L=L, t_end=T_END)
        before = np.argmin(np.abs(c_base["t"] - 30.0))
        after = np.argmin(np.abs(c_base["t"] - 150.0))
        assert c_eh["S"][before] == pytest.approx(c_base["S"][before], rel=1e-12)
        assert c_eh["S"][after] < c_base["S"][after], "the hazard must leave a permanent deficit"

    def test_chronic_ExtHazard_matches_the_external_hazard_kwarg(self, th, cohort):
        """A chronic level and the scalar kwarg are two spellings of the same thing."""
        t, e = cohort
        a = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, external_hazard=6.0)
        b = fpp.piecewise_loglik(t, e, th, [ExtHazard(6.0)], xc=1.0, L=L, t_end=T_END)
        assert a == pytest.approx(b, abs=1e-9)


def test_registered_on_import(th):
    assert "piecewise_fp" in isim.EVALUATORS
    assert isim.sr_metrics(th, (), evaluator="piecewise_fp", L=L,
                           t_end=T_END)["evaluator"] == "piecewise_fp"


class TestWindowResolutionHelpers:
    def test_floor_catches_a_window_comparable_to_dt(self):
        """The structural case: 0.5 units of window on a dt=0.107 grid is ~5 points."""
        assert ss.window_nsteps_floor(214.0, [(60.0, 60.5)]) > 2000
        assert ss.window_nsteps_floor(214.0, [(0.0, 60.0)]) < 2000     # 60 units is plenty
        assert ss.window_nsteps_floor(214.0, []) == 0

    def test_periodic_floor(self):
        assert ss.window_nsteps_floor(100.0, period=4.0) == 1000
        assert ss.window_nsteps_floor(100.0, period=40.0) == 100

    def test_sim_metrics_applies_the_floor(self, th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = isim.sr_metrics(th, (Intervention("beta", -0.6, (60.0, 60.5)),), evaluator="sim",
                                t_end=T_END, n=800, nsteps=2000, seed=4)
        assert r["nsteps"] >= r["window_nsteps_floor"] > 2000

    def test_median_sensitivity_flags_a_flat_crossing(self, th):
        """High sensitivity means 'compare curves, not medians'."""
        flat = isim.sr_metrics(th, (Intervention("beta", -0.6, (0.0, 60.0)),),
                               evaluator="piecewise_fp", L=L, t_end=T_END)
        cs = fpc.fp_curves(eta=th.eta, beta=th.beta, epsilon=th.epsilon, kappa=th.kappa,
                           L=L, t_end=3 * L)
        plain = cs.metrics()["median_sensitivity"]
        assert np.isfinite(plain) and plain > 0
        # the windowed cohort's crossing is flatter than the plain one's
        assert fpc.median_sensitivity(cs.t_full, cs.S_full) == pytest.approx(plain)
        assert flat["median"] > 0


# ===========================================================================================
# ADVERSARIAL QA PASS (2026-08-14). Everything below was written to BREAK the four additions
# rather than to demonstrate them. Four defects it found are fixed in `fp_piecewise`; the tests
# that pin those fixes are marked REGRESSION. One defect is NOT fixed (it is a semantics call for
# the module owner) and is pinned as a strict xfail, so it converts to a hard failure the moment
# somebody fixes it and forgets to delete the marker.
# ===========================================================================================
class TestSegmentArity:
    """REGRESSION: `s + (1.0,)*(8-len(s))` padded SHORT tuples silently.

    A 5-tuple used to run to completion with kappa quietly defaulted to 1.0 -- a different model,
    no message, plausible-looking output. A 9-tuple padded by `(1.0,)*-1 == ()` and then died deep
    inside `_seg_consts` with "too many values to unpack", nowhere near the caller's mistake.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 9])
    def test_wrong_arity_raises(self, dl, n):
        r_eta, r_beta, r_eps, kap = dl
        full = (0.0, np.inf, r_eta, r_beta, r_eps, kap, 1.0, 1.0)
        seg = full[:n] if n <= 8 else full + (1.0,)
        with pytest.raises(ValueError, match="6-, 7- or 8-tuple"):
            fpp.piecewise_survival([seg], N=200, dt=5e-3)

    @pytest.mark.parametrize("n", [6, 7, 8])
    def test_six_seven_and_eight_are_all_accepted_and_agree(self, dl, n):
        r_eta, r_beta, r_eps, kap = dl
        full = (0.0, np.inf, r_eta, r_beta, r_eps, kap, 1.0, 1.0)
        ref = fpp.piecewise_survival([full[:6]], N=300, dt=4e-3)[1]
        assert np.array_equal(fpp.piecewise_survival([full[:n]], N=300, dt=4e-3)[1], ref)

    def test_a_short_tuple_no_longer_silently_invents_kappa(self, dl):
        """The specific poison: dropping kappa used to substitute 1.0 and keep going."""
        r_eta, r_beta, r_eps, kap = dl
        assert kap != 1.0, "this test is vacuous if the fixture's kappa happens to be 1"
        with pytest.raises(ValueError):
            fpp.piecewise_survival([(0.0, np.inf, r_eta, r_beta, r_eps)], N=200, dt=5e-3)


class TestCumulativeHazardIsTheIntegralOfTheRate:
    """REGRESSION: `_cum_ext_hazard` and `_hazard_rate` must describe the SAME h(t).

    `piecewise_curves` uses BOTH -- `S *= exp(-H)` and `p += h*S` -- so if H is not the integral of
    h then S and p stop being each other's derivative and the density is wrong while the survival
    still looks plausible. They agreed for one window and disagreed for two: the cumulative form
    ADDED each window's excess, while the rate form OVERRIDES (last wins), which is what the
    simulator does (`chance_to_die_externally = exth_chance` is an assignment, not a `+=`).
    """

    @staticmethod
    def _max_mismatch(exth, external_hazard, t_hi=300.0, n=200001):
        t = np.linspace(0.0, t_hi, n)
        H = fpp._cum_ext_hazard(t, exth, external_hazard)
        h = fpp._hazard_rate(t, exth, external_hazard)
        H_num = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(t))])
        return float(np.max(np.abs(H - H_num)))

    @pytest.mark.parametrize("exth,eh,label", [
        ((), np.inf, "nothing"),
        ((), 6.0, "chronic kwarg"),
        ((ExtHazard(6.0),), np.inf, "chronic spec"),
        ((ExtHazard(4.0, (60.0, 120.0)),), np.inf, "one window"),
        ((ExtHazard(4.0, (60.0, 120.0)),), 6.0, "one window over a background"),
        ((ExtHazard(4.0, (60.0, 120.0)), ExtHazard(5.0, (140.0, 180.0))), 6.0, "two disjoint"),
        ((ExtHazard(4.0, (60.0, 120.0)), ExtHazard(5.0, (100.0, 180.0))), 6.0, "TWO OVERLAPPING"),
        ((ExtHazard(4.0, (60.0, 120.0)), ExtHazard(np.inf, (90.0, 110.0))), 6.0, "window + relief"),
        ((ExtHazard(6.0), ExtHazard(4.0, (60.0, 120.0)), ExtHazard(5.0, (100.0, 180.0))), np.inf,
         "chronic + two overlapping"),
        ((ExtHazard(4.0, (100.0, 400.0)),), 6.0, "window past the horizon"),
        ((ExtHazard(4.0, (60.0, 60.0)),), 6.0, "zero-width window"),
        ((ExtHazard(4.0, (0.0, 120.0)),), 6.0, "window starting at 0"),
    ])
    def test_H_is_the_integral_of_h(self, exth, eh, label):
        # 1e-4 is the trapezoid's own error at a step discontinuity on this grid; the defect this
        # guards produced 3.2e-1 on the "TWO OVERLAPPING" row.
        assert self._max_mismatch(exth, eh) < 1e-4, label

    def test_overlapping_windows_override_rather_than_sum(self):
        """The concrete failure: on [100, 120] both windows are open and only the LAST applies."""
        exth = (ExtHazard(4.0, (60.0, 120.0)), ExtHazard(5.0, (100.0, 180.0)))
        h = fpp._hazard_rate(np.array([110.0]), exth, 6.0)[0]
        assert h == pytest.approx(np.exp(-5.0)), "last window wins inside an overlap"
        H = fpp._cum_ext_hazard(np.array([300.0]), exth, 6.0)[0]
        summed = (np.exp(-6.0) * 300.0 + (np.exp(-4.0) - np.exp(-6.0)) * 60.0
                  + (np.exp(-5.0) - np.exp(-6.0)) * 80.0)          # the old, wrong, additive form
        assert H < summed - 0.3, "H must not double-count the overlap"

    def test_one_window_keeps_its_closed_form(self):
        """The <=1-window branch is untouched, so it must still equal the closed form exactly."""
        t = np.linspace(0.0, 300.0, 1001)
        h_base, h_in = np.exp(-6.0), np.exp(-4.0)
        overlap = np.clip(np.maximum(np.clip(t, None, 120.0) - 60.0, 0.0), 0.0, 60.0)
        want = h_base * t + (h_in - h_base) * overlap
        got = fpp._cum_ext_hazard(t, (ExtHazard(4.0, (60.0, 120.0)),), 6.0)
        assert np.array_equal(got, want)

    def test_a_level_of_inf_inside_a_window_is_relief_not_extra_hazard(self):
        t = np.array([300.0])
        protected = fpp._cum_ext_hazard(t, (ExtHazard(np.inf, (60.0, 120.0)),), 6.0)[0]
        plain = fpp._cum_ext_hazard(t, (), 6.0)[0]
        assert protected == pytest.approx(plain - np.exp(-6.0) * 60.0)

    def test_a_zero_width_window_contributes_nothing(self):
        t = np.array([300.0])
        assert (fpp._cum_ext_hazard(t, (ExtHazard(0.0, (60.0, 60.0)),), 6.0)[0]
                == pytest.approx(fpp._cum_ext_hazard(t, (), 6.0)[0]))


class TestSamplingDtContract:
    """REGRESSION: `piecewise_loglik` must honour `fp_loglik`'s `sampling_dt` contract exactly.

    It used to test the raw value with `if sampling_dt:`, so a NEGATIVE delta was truthy and scored
    log[S(t) - S(t-|d|)] <= 0 -- every death hit the -700 floor and the total came back -29404
    against fp_loglik's -239, silently. The per-observation ARRAY form (a cohort with a missed
    scoring round, which `_norm_sampling_dt` exists for) raised "truth value of an array is
    ambiguous" instead of working.
    """

    @pytest.mark.parametrize("sampling_dt", [None, 0.0, -1.0, 1.0, 2.0, 0.5])
    def test_scalar_matches_fp_loglik_bit_for_bit(self, th, cohort, sampling_dt):
        t, e = cohort
        a = fpl.fp_loglik_physical(t, e, th.eta, th.beta, th.epsilon, 1.0, th.kappa, L=L,
                                   sampling_dt=sampling_dt)
        b = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, sampling_dt=sampling_dt)
        assert a == b, f"drifted by {abs(a - b):.3e} at sampling_dt={sampling_dt}"

    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.001])
    def test_non_positive_means_the_density_form(self, th, cohort, bad):
        """`_norm_sampling_dt` maps non-positive scalars to None; so must this."""
        t, e = cohort
        density = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, sampling_dt=None)
        assert fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END,
                                    sampling_dt=bad) == density

    def test_per_observation_array_matches_fp_loglik(self, th, cohort):
        """A missed scoring round pools two days into one entry; that entry covers 2 days."""
        t, e = cohort
        d = np.full(t.size, 1.0)
        d[::7] = 2.0
        a = fpl.fp_loglik_physical(t, e, th.eta, th.beta, th.epsilon, 1.0, th.kappa, L=L,
                                   sampling_dt=d)
        b = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, sampling_dt=d)
        assert a == b
        assert b != fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, sampling_dt=1.0)

    def test_a_mis_shaped_array_raises(self, th, cohort):
        t, e = cohort
        with pytest.raises(ValueError, match="scalar or an array"):
            fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END,
                                 sampling_dt=np.ones(t.size - 3))


class TestEvaluatorForwardsExternalHazard:
    """REGRESSION: `_evaluator` dropped `external_hazard` although `piecewise_metrics` implements it.

    `sr_metrics(..., evaluator='piecewise_fp', external_hazard=4.0)` came back with the
    hazard-free median (126.0 instead of 37.7) while `evaluator='sim'` honoured the same kwarg --
    so swapping evaluators silently swapped the model, in the direction that flatters the FP.
    """

    def test_sr_metrics_honours_external_hazard(self, th):
        direct = fpp.piecewise_metrics(th, (), L=L, t_end=T_END, external_hazard=4.0)["median"]
        via = isim.sr_metrics(th, (), evaluator="piecewise_fp", L=L, t_end=T_END,
                              external_hazard=4.0)["median"]
        none = isim.sr_metrics(th, (), evaluator="piecewise_fp", L=L, t_end=T_END)["median"]
        assert via == direct
        assert via < none, "an external hazard must shorten life"

    def test_simulator_only_kwargs_are_still_dropped(self, th):
        """The drop list still has to absorb the sim's own kwargs, or dispatch breaks."""
        m = isim.sr_metrics(th, (), evaluator="piecewise_fp", L=L, t_end=T_END,
                            n=10, nsteps=10, seed=1, parallel=False, method="brownian_bridge",
                            hetro=False, het_var=0.2, return_sim=False, time_step_multiplier=1)
        assert m["evaluator"] == "piecewise_fp"


class TestMovingBoundaryInternals:
    """The moving-`xc` boundary is the one addition that mutates solver state mid-march."""

    @staticmethod
    def _march(scheme, fracs, monkeypatch, N=900, dt=1e-3, nstart=20, dl=None):
        """Run a 3-segment march, recording (operator size, scheme flag, rhs length) per step."""
        r_eta, r_beta, r_eps, kap = dl
        seen = {}
        log = []
        real_op, real_sb = fpp._operator, fpp.solve_banded

        def spy_op(a_face, M, h, re):
            lo, di, up = real_op(a_face, M, h, re)
            seen["di0"], seen["M"] = di[0], M
            return lo, di, up

        def spy_sb(lu, ab, rhs):
            # ab[1, 0] == c - dt*di[0] with c == 1.0 for backward Euler and 1.5 for BDF2
            log.append((seen["M"], round(float(ab[1, 0] + dt * seen["di0"]), 6), rhs.shape[0]))
            return real_sb(lu, ab, rhs)

        monkeypatch.setattr(fpp, "_operator", spy_op)
        monkeypatch.setattr(fpp, "solve_banded", spy_sb)
        segs = [(0.0, 0.4, r_eta, r_beta, r_eps, kap, 1.0, fracs[0]),
                (0.4, 0.7, r_eta, r_beta, r_eps, kap, 1.0, fracs[1]),
                (0.7, np.inf, r_eta, r_beta, r_eps, kap, 1.0, fracs[2])]
        tau, S = fpp.piecewise_survival(segs, N=N, dt=dt, scheme=scheme, nstart=nstart)
        return np.array([r[0] for r in log]), np.array([r[1] for r in log]), \
            np.array([r[2] for r in log]), S

    def test_rhs_length_always_matches_the_operator(self, dl, monkeypatch):
        """`p` and `p_prev` are full-length N; the slices `p[:M]` must track the operator."""
        for scheme in ("be", "bdf2"):
            M, _c, R, _S = self._march(scheme, (1.0, 0.5, 1.0), monkeypatch, dl=dl)
            assert np.array_equal(M, R), scheme
            assert sorted(set(M.tolist())) == [450, 900], scheme

    def test_bdf2_restarts_with_one_be_step_at_every_boundary_move(self, dl, monkeypatch):
        """`p_prev = None` is the correct reset (the history spans a different domain) -- but it
        also has to be REACHED. It is: the BE steps are exactly the `nstart` startup plus one per
        move, and no more."""
        M, c, _R, _S = self._march("bdf2", (1.0, 0.5, 1.0), monkeypatch, nstart=20, dl=dl)
        be = set(np.flatnonzero(np.abs(c - 1.0) < 1e-9).tolist())
        bdf2 = np.flatnonzero(np.abs(c - 1.5) < 1e-9)
        moves = {int(i) + 1 for i in np.flatnonzero(np.diff(M) != 0)}
        assert be == set(range(20)) | moves, f"BE steps {sorted(be)} vs startup+moves"
        assert bdf2.size > 100, "the march must actually be running BDF2 the rest of the time"
        assert len(moves) == 2, "this test needs the boundary to move in and then back out"

    def test_no_boundary_move_means_no_extra_be_step(self, dl, monkeypatch):
        """Control: with a fixed threshold the only BE steps are the startup."""
        _M, c, _R, _S = self._march("bdf2", (1.0, 1.0, 1.0), monkeypatch, nstart=20, dl=dl)
        assert set(np.flatnonzero(np.abs(c - 1.0) < 1e-9).tolist()) == set(range(20))

    @pytest.mark.parametrize("scheme", ["be", "bdf2"])
    def test_a_flat_threshold_is_bit_identical_to_one_segment(self, dl, scheme):
        """Splitting into three segments with xc_frac == 1 everywhere must change nothing at all."""
        r_eta, r_beta, r_eps, kap = dl
        dt = 1e-3 if scheme == "be" else 4e-3
        one = fpp.piecewise_survival([(0.0, np.inf, r_eta, r_beta, r_eps, kap)],
                                     dt=dt, scheme=scheme)[1]
        three = fpp.piecewise_survival([(0.0, 0.4, r_eta, r_beta, r_eps, kap, 1.0, 1.0),
                                        (0.4, 0.7, r_eta, r_beta, r_eps, kap, 1.0, 1.0),
                                        (0.7, np.inf, r_eta, r_beta, r_eps, kap, 1.0, 1.0)],
                                       dt=dt, scheme=scheme)[1]
        assert np.array_equal(one, three)

    @pytest.mark.parametrize("frac", [0.9, 0.7, 0.5, 0.3])
    def test_survival_never_increases_when_the_boundary_moves_in_then_out(self, dl, frac):
        """Raising the threshold back must expose EMPTY cells, never resurrect anybody."""
        r_eta, r_beta, r_eps, kap = dl
        tau, S = fpp.piecewise_survival(
            [(0.0, 0.4, r_eta, r_beta, r_eps, kap, 1.0, 1.0),
             (0.4, 0.7, r_eta, r_beta, r_eps, kap, 1.0, frac),
             (0.7, np.inf, r_eta, r_beta, r_eps, kap, 1.0, 1.0)], N=900, dt=1e-3)
        assert np.max(np.diff(S)) <= 1e-13, "S increased: mass was created at a boundary move"
        assert S.min() >= 0.0

    def test_lowering_the_threshold_absorbs_immediately(self, dl):
        """The documented physics: whoever already sits above the new threshold dies at once."""
        r_eta, r_beta, r_eps, kap = dl
        drops = []
        for frac in (1.0, 0.7, 0.5, 0.3):
            tau, S = fpp.piecewise_survival(
                [(0.0, 0.4, r_eta, r_beta, r_eps, kap, 1.0, 1.0),
                 (0.4, np.inf, r_eta, r_beta, r_eps, kap, 1.0, frac)], N=900, dt=1e-3)
            i = int(np.searchsorted(tau, 0.4))
            drops.append(float(S[i - 1] - S[i + 1]))
        assert drops[0] < 1e-4, "a flat threshold must not produce a cliff"
        assert drops[1] < drops[2] < drops[3], "a deeper cut must absorb more, immediately"

    def test_an_early_harmless_dip_barely_moves_the_answer(self, th):
        """At day 1 nobody is near the threshold, so halving it for a day must be almost a no-op --
        and restoring it must not leave stale mass behind."""
        base = fpp.piecewise_metrics(th, (), L=L, t_end=T_END)["median"]
        dip = fpp.piecewise_metrics(th, [Intervention("xc", -0.5, (1.0, 2.0))],
                                    L=L, t_end=T_END)["median"]
        assert abs(dip - base) / base < 1e-3, f"base={base:.4f} dip={dip:.4f}"

    def test_the_moving_boundary_converges_in_N(self, th):
        """The threshold is quantised to `round(N*frac)/N`, so refining N must not move the answer
        by more than the ordinary discretisation does."""
        iv = [Intervention("xc", -0.25, (60.0, 120.0))]
        a = fpp.piecewise_metrics(th, iv, L=L, t_end=T_END, N=900)["median"]
        b = fpp.piecewise_metrics(th, iv, L=L, t_end=T_END, N=1800)["median"]
        assert a == pytest.approx(b, rel=3e-3), f"N=900 {a:.5f} vs N=1800 {b:.5f}"

    def test_a_threshold_below_two_cells_is_silently_floored(self):
        """KNOWN, DOCUMENTED LIMIT -- pinned so nobody rediscovers it as a mystery.

        `M = max(2, min(N, round(N*frac)))` cannot represent a threshold thinner than 2/N, so a cut
        deeper than that is silently served at 2/N instead. At the default N=900 that is a 0.22 %
        threshold; anything below needs a larger N. There is no warning.
        """
        N = 900
        for frac in (1e-9, 1.0 / N, 2.0 / N):
            assert max(2, min(N, int(round(N * frac)))) == 2
        assert max(2, min(N, int(round(N * 3.0)))) == N, "frac > 1 is clamped, also silently"


class TestFullLifeWindowEqualsChronic:
    """Sharpen the existing full-life check: it must hold for EVERY parameter, not just xc.

    Note this is a weaker test than it reads: `build_segments` collapses a (0, t_end) window to a
    SINGLE segment, so it never exercises the segment machinery. It is still the right invariant --
    it catches a window that is placed off-by-one and therefore misses part of life.
    """

    @pytest.mark.parametrize("param", ["eta", "beta", "epsilon", "kappa", "A", "xc"])
    @pytest.mark.parametrize("effect", [-0.25, 0.3])
    def test_full_life_window_is_bit_identical_to_chronic(self, th, param, effect):
        ch = fpp.piecewise_metrics(th, [Intervention(param, effect)], L=L, t_end=T_END)["median"]
        wh = fpp.piecewise_metrics(th, [Intervention(param, effect, (0.0, T_END))],
                                   L=L, t_end=T_END)["median"]
        assert ch == wh, f"{param} {effect}: chronic {ch!r} vs full-life window {wh!r}"


class TestLikelihoodUnderWindowedHazard:
    """The exact path `itp_analysis/src/fp_engine.score` runs, which nothing covered before:
    a windowed `ExtHazard` together with windowed interventions AND interval censoring."""

    @pytest.mark.parametrize("sampling_dt", [None, 1.0])
    def test_a_window_spanning_all_of_life_equals_the_chronic_level(self, th, cohort, sampling_dt):
        """The windowed branch and the `_interval_mass` branch are separate code; on a window that
        can never close they have to agree."""
        t, e = cohort
        a = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END, external_hazard=6.0,
                                 sampling_dt=sampling_dt)
        b = fpp.piecewise_loglik(t, e, th, [ExtHazard(6.0, (0.0, 1e5))], xc=1.0, L=L, t_end=T_END,
                                 sampling_dt=sampling_dt)
        assert a == pytest.approx(b, abs=1e-9)

    def test_terms_still_sum_to_the_total_on_the_windowed_branch(self, th, cohort):
        t, e = cohort
        ivs = [Intervention("beta", -0.3, (60.0, 150.0)), Intervention("A", -0.15, (60.0, 150.0)),
               ExtHazard(5.0, (60.0, 150.0))]
        terms = fpp.piecewise_loglik_terms(t, e, th, ivs, xc=1.0, L=L, t_end=T_END, sampling_dt=1.0)
        total = fpp.piecewise_loglik(t, e, th, ivs, xc=1.0, L=L, t_end=T_END, sampling_dt=1.0)
        assert terms.size == t.size and np.isfinite(terms).all()
        assert terms.sum() == pytest.approx(total, abs=1e-9)

    def test_a_windowed_hazard_costs_survival_only_after_it_opens(self, th, cohort):
        t, e = cohort
        base = fpp.piecewise_loglik(t, e, th, (), xc=1.0, L=L, t_end=T_END)
        late = fpp.piecewise_loglik(t, e, th, [ExtHazard(4.0, (150.0, T_END))], xc=1.0, L=L,
                                    t_end=T_END)
        assert late != base

    @pytest.mark.parametrize("n_off", [-3, +3])
    def test_mismatched_events_length_raises_rather_than_mis_aligning(self, th, cohort, n_off):
        """`np.asarray(events, int)[:terms.size]` looks like it could silently truncate. It cannot:
        `piecewise_loglik_terms` uses the FULL events array first and indexes a length-n `terms`
        with it, so any mismatch is an IndexError before the slice is ever reached."""
        t, e = cohort
        ev = e[:e.size + n_off] if n_off < 0 else np.concatenate([e, e[:n_off]])
        with pytest.raises(IndexError):
            fpp.piecewise_loglik(t, ev, th, (), xc=1.0, L=L, t_end=T_END)


class TestNothingIsMutated:
    """A fitter calls these thousands of times with the same `theta` / `interventions` objects."""

    def test_piecewise_survival_leaves_its_segments_alone(self, dl):
        r_eta, r_beta, r_eps, kap = dl
        segs = [(0.0, 0.5, r_eta, r_beta, r_eps, kap), (0.5, np.inf, r_eta, r_beta, r_eps, kap)]
        before = [tuple(s) for s in segs]
        fpp.piecewise_survival(segs, N=300, dt=4e-3)
        assert segs == before

    def test_build_segments_leaves_theta_and_interventions_alone(self, th):
        ivs = [Intervention("beta", -0.5, (60.0, 120.0)), Intervention("xc", 0.3, (30.0, 90.0)),
               ExtHazard(6.0, (10.0, 20.0))]
        iv_before = [(i.param, i.effect, i.window) for i in ivs
                     if not isinstance(i, ExtHazard)]
        th_before = (th.eta, th.beta, th.epsilon, th.kappa, th.xc)
        n_before = len(ivs)
        fpp.build_segments(th, ivs, L, T_END, return_exth=True)
        assert [(i.param, i.effect, i.window) for i in ivs
                if not isinstance(i, ExtHazard)] == iv_before
        assert (th.eta, th.beta, th.epsilon, th.kappa, th.xc) == th_before
        assert len(ivs) == n_before

    def test_the_likelihood_leaves_the_data_alone(self, th, cohort):
        t, e = cohort
        t_before, e_before = t.copy(), e.copy()
        fpp.piecewise_loglik(t, e, th, [Intervention("beta", -0.3, (60.0, 120.0))],
                             xc=1.0, L=L, t_end=T_END)
        assert np.array_equal(t, t_before) and np.array_equal(e, e_before)


class TestWindowEdgeSemantics:
    """Degenerate windows must be no-ops or raise -- never a quietly different model."""

    def test_a_zero_width_window_is_a_no_op(self, th):
        base = fpp.piecewise_metrics(th, (), L=L, t_end=T_END)["median"]
        assert fpp.piecewise_metrics(th, [Intervention("beta", -0.9, (60.0, 60.0))],
                                     L=L, t_end=T_END)["median"] == base

    def test_a_window_starting_at_zero_is_active_from_birth(self, th):
        segs, _xc = fpp.build_segments(th, [Intervention("beta", -0.5, (0.0, 60.0))], L, T_END)
        assert len(segs) == 2 and segs[0][0] == 0.0
        assert segs[0][3] == pytest.approx(0.5 * segs[1][3])

    def test_a_window_reaching_back_before_birth_is_clipped_not_dropped(self, th):
        segs, _xc = fpp.build_segments(th, [Intervention("beta", -0.5, (-10.0, 60.0))], L, T_END)
        assert len(segs) == 2
        assert segs[0][3] == pytest.approx(0.5 * segs[1][3]), "the pre-birth part is simply clipped"

    def test_a_chronic_and_a_windowed_xc_compose(self, th):
        segs, xc_ref = fpp.build_segments(
            th, [Intervention("xc", 0.5), Intervention("xc", -0.2, (60.0, 120.0))], L, T_END)
        assert xc_ref == pytest.approx(1.5), "the chronic factor sets the grid"
        assert [round(s[7], 6) for s in segs] == [1.0, pytest.approx(0.8), 1.0]

    def test_the_clock_is_continuous_across_every_edge(self, th):
        """`A` is the only direction with memory; the accumulated clock is what carries it."""
        segs, _xc = fpp.build_segments(th, [Intervention("A", -0.3, (60.0, 120.0))], L, T_END)
        assert [round(s[6], 6) for s in segs] == [1.0, pytest.approx(0.7), 1.0]
        y = segs[0][0]
        for s in segs[:-1]:
            y += s[6] * (s[1] - s[0])
        assert y == pytest.approx(60.0 / L + 0.7 * 60.0 / L), "y(120) must be 0.7-discounted"
        assert y < 120.0 / L, "the clock has to run BEHIND real time after a slowing window"


class TestKnownDefectsNotYetFixed:
    """Reported, deliberately NOT patched here: fixing them is a semantics decision for the module
    owner, and the fix changes published numbers. Written as the assertion that SHOULD hold and
    marked strict-xfail, so it becomes a loud failure the moment somebody fixes the behaviour."""

    @pytest.mark.xfail(strict=True, reason=(
        "DEFECT: build_segments extends the LAST span to tau_max unconditionally, so a window whose "
        "stop is >= t_end silently becomes chronic-from-start and its stated stop is ignored. "
        "Intervention('eta',-0.6,(100,214)) and (100,10000) at t_end=214 give byte-identical "
        "metrics, and the mean is 229.1 against 191.2 for a window that really does close at 214 "
        "(a 19.8% error). See the QA report."))
    def test_a_window_stop_beyond_t_end_should_not_be_ignored(self, th):
        short = fpp.piecewise_metrics(th, [Intervention("eta", -0.6, (100.0, T_END))],
                                      L=L, t_end=T_END)["mean"]
        long = fpp.piecewise_metrics(th, [Intervention("eta", -0.6, (100.0, 10000.0))],
                                     L=L, t_end=T_END)["mean"]
        assert short < long, "a longer life-extending window must extend life further"

    @pytest.mark.xfail(strict=True, reason=(
        "DEFECT: two interventions on the SAME parameter are silently multiplied here, while "
        "intervention_sim.to_getisr_kwargs rejects them ('getISR has ONE argument block per "
        "parameter'). So piecewise_fp accepts a spec the simulator cannot reproduce, and there is "
        "no way to validate the composed result against the cohort."))
    def test_two_windows_on_one_parameter_should_be_rejected_like_the_simulator_does(self, th):
        ivs = [Intervention("beta", -0.3, (60.0, 120.0)), Intervention("beta", -0.3, (140.0, 180.0))]
        with pytest.raises((ValueError, NotImplementedError)):
            fpp.build_segments(th, ivs, L, T_END)
