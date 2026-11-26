"""S5's instrument against gen-1 (tasks T085; research R20): the score sum_cells counts x d log P_cell / d log theta
against gen-1's binned law, differentiated once by central differences; gen-1's own objects, read-only; a planted law
at a perturbed sd shifts the mean score and its twin does not. Toy laws keep it cheap: gen-1's own 161-node law at
N = 6,000 is S5's run's, not a unit test's."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import norm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import s5_gen1_004 as S5  # noqa: E402


def toy(x):
    """A two-cell law in one log parameter: P = (a, 1) / (1 + a), a = e^x."""
    a = math.exp(x[0])
    return np.array([a / (1 + a), 1 / (1 + a)])


def test_the_score_is_the_hand_computed_toy_score():
    x0, counts = np.array([0.3]), np.array([3.0, 5.0])
    a = math.exp(0.3)
    want = 3 / (1 + a) - 5 * a / (1 + a)                       # d log P / dx = (1/(1+a), -a/(1+a))
    d = S5.log_law_gradient(toy, x0)
    assert d.shape == (2, 1) and S5.score(counts, d)[0] == pytest.approx(want, rel=1e-8)


def binned_tn(sd, edges=np.linspace(0.0, 2.5, 26)):
    """A toy binned law: s = 1 + sd z, z standard normal truncated to s > 0, in cells of s."""
    a = -1.0 / sd
    F = (norm.cdf((edges - 1.0) / sd) - norm.cdf(a)) / (1 - norm.cdf(a))
    P = np.diff(np.clip(F, 0, 1))
    P[-1] += 1 - F[-1]
    return P


def test_a_planted_law_at_a_perturbed_sd_shifts_the_mean_score_and_its_twin_does_not():
    x0 = np.array([math.log(0.2)])
    law = lambda x: binned_tn(math.exp(x[0]))                  # noqa: E731
    d = S5.log_law_gradient(law, x0)
    twin = S5.expected_score(binned_tn(0.2), d, n=1)           # draws from the true law: mean score 0
    plant = S5.expected_score(binned_tn(0.23), d, n=1)
    assert abs(twin[0]) < 1e-6 and abs(plant[0]) > 0.1


def test_gen1s_own_objects_and_gen1_is_never_written():
    from sraging import generator
    assert S5.GEN is generator
    for name in ("binned_distribution", "_record", "_het_nodes", "_S_obs", "_model"):
        assert getattr(S5.GEN, name) is getattr(generator, name)
    assert S5.law.__code__.co_names.count("binned_distribution") == 1
    r = subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", "src/sraging/generator.py"])
    assert r.returncode == 0                                    # gen-1 unchanged against the commit


def test_counts_map_each_record_to_its_cell():
    law = {"ages": np.array([0.0, 1.0, 2.0, 3.0]), "events": np.array([1, 1, 1, 0])}
    c = S5.counts(np.array([1.0, 1.0, 0.0, 3.0, 2.0]), np.array([1, 1, 1, 0, 1]), law)
    assert c.tolist() == [1, 2, 1, 1]
    with pytest.raises(ValueError, match="not a cell"):
        S5.counts(np.array([1.5]), np.array([1]), law)


def test_the_spec_and_the_srparams_agree():
    from sraging import generator as GEN
    spec = GEN.CohortSpec(name="t", rho_eta=2.0, rho_beta=10.0, rho_eps=1.5, kappa=0.1, n=1000, L_phys=100.0, seed=1,
                          delta=1.0, external_hazard=5.0, het_param="xc", het_var=0.2)
    p, het = S5.params_of(spec)
    assert (p.eta, p.beta, p.epsilon, p.kappa) == pytest.approx((2.0 / 1e4, 0.1, 0.015, 0.1))
    assert p.h_ext == pytest.approx(math.exp(-5.0)) and (het.parameter, het.sd) == ("xc", 0.2)


# ------------------------------------------- P2's 2026-11-03 error: a survivor at the horizon (31ec891, cohorts 53, 69)
TOY_LAW = {"ages": np.array([0.0, 1.0, 2.0, 3.0]), "events": np.array([1, 1, 1, 1])}   # gen-1 at censor_frac 0
TOY_SPEC = type("S", (), {"delta": 1.0, "censor_frac": 0.0, "time_range": None})()


def test_planted_a_survivor_at_the_horizon_is_gen1s_last_node_death_and_counts():
    t_end = 3.7                                                    # gen-1's support end, tau[-1] L
    ages, events, n_beyond = S5.record(np.array([0.2, 2.4, t_end]), t_end, TOY_SPEC)
    assert ages.tolist() == [0.0, 2.0, 3.0] and events.tolist() == [1, 1, 1] and n_beyond == 1
    assert S5.counts(ages, events, TOY_LAW).tolist() == [1, 0, 1, 1]
    with pytest.raises(ValueError, match="not a cell"):            # the old record: censored at the sampler's t_max
        S5.counts(np.array([0.0, 2.0, t_end]), np.array([1, 1, 0]), TOY_LAW)


def test_twin_an_ordinary_record_is_its_floor_cell_and_nothing_is_beyond():
    ages, events, n_beyond = S5.record(np.array([2.4, 0.2, 3.69]), 3.7, TOY_SPEC)
    assert ages.tolist() == [0.0, 2.0, 3.0] and events.tolist() == [1, 1, 1] and n_beyond == 0
    assert S5.counts(ages, events, TOY_LAW).tolist() == [1, 0, 1, 1]


def test_a_time_past_the_horizon_and_a_censored_or_windowed_spec_are_refused():
    from sraging.errors import UnsupportedOptionError
    with pytest.raises(ValueError, match="past gen-1's horizon"):
        S5.record(np.array([3.8]), 3.7, TOY_SPEC)
    for f, v in (("censor_frac", 0.2), ("time_range", (0.0, 10.0))):
        spec = type("S", (), {"censor_frac": 0.0, "time_range": None, f: v})()
        with pytest.raises(UnsupportedOptionError, match=f):
            S5.plain(spec)
    S5.plain(TOY_SPEC)                                              # twin: a plain spec passes


def test_the_horizons_cell_is_gen1s_last_law_cell():
    """On gen-1's own (homogeneous, one-curve) law at S5's point: a survivor at the horizon lands on the law's last
    death cell, and the law's survival there is below the march's Gtol scale."""
    from sraging import generator as GEN
    spec = GEN.CohortSpec(name="h", rho_eta=2.125427697723628, rho_beta=0.1, rho_eps=0.003981071705534973,
                          kappa=0.01, n=2000, L_phys=100.0, seed=1, delta=1.0)
    t_end = S5.horizon(spec)
    lw = S5.law(spec)
    ages, events, n_beyond = S5.record(np.array([t_end]), t_end, spec)
    assert (ages[0], events[0], n_beyond) == (lw["ages"][-1], 1, 1) and set(lw["events"].tolist()) == {1}
    assert S5.counts(ages, events, lw).sum() == 1
    m = GEN._model(spec, dict(GEN.GRID))
    assert float(GEN._S_obs(m, spec, np.array([t_end]))[0]) < 1e-11
