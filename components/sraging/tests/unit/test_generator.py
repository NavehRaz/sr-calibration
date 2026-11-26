"""Spec 003d's cohort generator (src/sraging/generator.py): the spec and its refusals (contracts/python-api.md), the
survival curve from the engine's v1 march (R2-R4), the draws (R5), heterogeneity (R6), administrative censoring and
recording (R7), time_range as the tester's apply_window (R8), and materialize (FR-001, FR-013, FR-014).

Engine-backed tests run on a coarse grid (monkeypatched GRID) so the suite stays fast; the version's grid is fixed by
the refinement sweep (T019-T021), not here.
"""
import ast
import math
import os

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import truncnorm

import sraging.generator as G
from sraging.errors import UnsupportedOptionError

COARSE = {"N": 300, "dt": 1e-3, "tau_max": 30.0, "Gtol": 1e-12, "nstart": 20, "K": 9}
# library v5 points (tester manifest, read 2026-10-13)
K2_P05 = dict(rho_eta=2.113047133591786, rho_beta=0.4923882631706739, rho_eps=0.001, kappa=2.0)       # sharp
K01_P20 = dict(rho_eta=0.16193367995080452, rho_beta=2.424462017082328, rho_eps=1.0, kappa=0.1)       # broad
K01_P00 = dict(rho_eta=2.2434680350647502, rho_beta=0.22189823414589713, rho_eps=0.001995262314968879, kappa=0.1)
BASE = dict(name="t", **K01_P00, n=500, L_phys=100.0, seed=7, delta=1.0, censor_frac=0.0)
TESTER = os.environ.get("SRAGING_TESTER_MAIN", os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset",
                                                            "sr_calibration_tester"))   # release 0.1.1


@pytest.fixture
def coarse(monkeypatch):
    monkeypatch.setattr(G, "GRID", dict(COARSE))
    return COARSE


def _spec(**kw):
    return G.CohortSpec(**{**BASE, **kw})


# ------------------------------------------------------------------------------------------------- T003: the spec
def test_spec_accepts_the_testers_keyword_dict():
    # as srbench/library/build.py:_materialize builds it (169-178)
    kw = dict(name="censor-x", rho_eta=1.0, rho_beta=0.5, rho_eps=0.01, kappa=0.1, n=100, L_phys=50.0, seed=3,
              delta=1.0, censor_frac=0.2)
    s = G.CohortSpec(**kw)
    assert (s.n, s.censor_frac, s.external_hazard, s.het_param, s.het_var, s.time_range) == (100, 0.2, None, None, 0.0,
                                                                                            None)
    assert G.CohortSpec(**kw, external_hazard=5.5).external_hazard == 5.5
    assert G.CohortSpec(**kw, het_param="xc", het_var=0.2).het_var == 0.2


@pytest.mark.parametrize("field", ["defect", "defect_kw", "group", "notes", "foo"])
def test_spec_refuses_unknown_fields_by_name(field):
    with pytest.raises(UnsupportedOptionError) as e:
        _spec(**{field: 1})
    assert e.value.option == field


def test_spec_refuses_positional_construction():
    with pytest.raises(UnsupportedOptionError) as e:
        G.CohortSpec("t", 1.0, 0.5, 0.01, 0.1, 100, 50.0, 3, 1.0, 0.0)
    assert e.value.option == "positional"


@pytest.mark.parametrize("field,bad", [
    ("rho_eta", 0.0), ("rho_eta", float("nan")), ("rho_eps", 0.0), ("kappa", -1.0), ("rho_beta", -0.1),
    ("n", 0), ("L_phys", 0.0), ("seed", -1), ("delta", None), ("delta", 0.0), ("censor_frac", 1.0),
    ("censor_frac", -0.1), ("het_var", -0.1), ("time_range", (5.0, 5.0)), ("time_range", (0.5, 10.0)),
])
def test_spec_refuses_out_of_range_values_by_name(field, bad):
    with pytest.raises(UnsupportedOptionError) as e:
        _spec(**{field: bad})
    assert e.value.option == field


def test_spec_refuses_heterogeneity_other_than_xc():
    with pytest.raises(UnsupportedOptionError) as e:
        _spec(het_param="eta", het_var=0.2)
    assert e.value.option == "het_param"


def test_spec_is_frozen():
    s = _spec()
    with pytest.raises(Exception):
        s.n = 3


def test_spec_has_no_required_field_missing_silently():
    kw = dict(BASE)
    kw.pop("seed")
    with pytest.raises(UnsupportedOptionError) as e:
        G.CohortSpec(**kw)
    assert e.value.option == "seed"


# ------------------------------------------------------------------------------------------- T005: the curve
@pytest.mark.parametrize("point", [K2_P05, K01_P20])
def test_curve_is_a_cdf_that_absorbs(point):
    c = G._curve(point, COARSE)
    F, tau = c["F"], c["tau"]
    assert F[0] == 0.0 and np.all(np.diff(F) >= 0) and F[-1] >= 1 - 1e-11
    assert tau.size == F.size and abs(tau[1] - tau[0] - COARSE["dt"]) < 1e-15
    assert 0 <= c["max_abs_S_plus_F_minus_1"] < 1e-6        # a recorded diagnostic; 4e-8 at k2-p05 on this coarse grid


def test_curve_refuses_a_horizon_that_does_not_absorb():
    with pytest.raises(UnsupportedOptionError) as e:
        G._curve(K01_P20, {**COARSE, "tau_max": 0.5})
    assert e.value.option == "tau_max"


def test_curve_converges_second_order_in_N_at_the_sharp_point():
    # N 300-1200 is pre-asymptotic at this, the hardest point (ratio 2.3); the gate measured 4.1-4.4 from N 1500 up
    cv = [G.curve_moments(G._curve(K2_P05, {**COARSE, "N": N, "dt": 2e-4}))["cv"] for N in (1500, 3000, 6000)]
    d1, d2 = math.log(cv[0] / cv[1]), math.log(cv[1] / cv[2])
    assert 3.0 < d1 / d2 < 5.0


# ------------------------------------------------------------------------------------------- T007: the draws
class _Recorder:
    def __init__(self, rng):
        self.rng, self.calls = rng, []

    def __getattr__(self, name):
        f = getattr(self.rng, name)

        def wrapped(*a, **k):
            self.calls.append((name, a, k))
            return f(*a, **k)
        return wrapped


@pytest.mark.parametrize("hazard,het", [(False, False), (True, False), (False, True), (True, True)])
def test_draws_use_only_uniforms_in_the_documented_order(monkeypatch, hazard, het):
    recs = []
    monkeypatch.setattr(G, "_rng", lambda seed: recs.append(_Recorder(np.random.default_rng(seed))) or recs[-1])
    s = _spec(external_hazard=5.5 if hazard else None, het_param="xc" if het else None, het_var=0.2 if het else 0.0)
    u = G._uniforms(s)
    calls = recs[0].calls
    assert [c[0] for c in calls] == ["random"] * (1 + hazard + het)
    assert all(c[1] == (s.n,) or c[2].get("size") == s.n for c in calls)
    assert np.array_equal(u["death"], np.random.default_rng(s.seed).random(s.n))      # SRK's stream first (R5)
    assert (u["hazard"] is not None) == hazard and (u["het"] is not None) == het


def test_inverse_cdf_is_exact_on_a_linear_cdf_and_counts_draws_beyond_its_end():
    tau = np.linspace(0.0, 2.0, 2001)
    F = np.clip(tau / 2.0, 0.0, 0.999)
    u = np.array([0.0, 0.25, 0.5, 0.998, 0.9995])
    t, beyond = G._inverse(u, tau, F[None, :], np.zeros(u.size, int), np.zeros(u.size))
    assert np.allclose(t[:4], 2.0 * u[:4], atol=1e-12) and t[4] == tau[-1] and beyond == 1


def test_hazard_times_are_the_exponential_inverse_cdf():
    u = np.array([0.0, 0.1, 0.5, 0.9])
    assert np.allclose(G._hazard_times(u, 5.5), -np.log1p(-u) / math.exp(-5.5), rtol=0, atol=0)


# ------------------------------------------------------------------------------------------- T009: heterogeneity
def test_het_z_is_the_truncated_normal_quantile():
    u = np.linspace(1e-6, 1 - 1e-6, 101)
    want = truncnorm.ppf(u, a=-1 / 0.2, b=np.inf)
    assert np.max(np.abs(G._het_z(u, 0.2) - want)) < 1e-9


def test_het_nodes_span_the_range_and_their_weights_sum_to_one():
    nodes = G._het_nodes(0.2, 41)
    z = nodes["z"]
    assert z.size == 41 and np.allclose(np.diff(z), z[1] - z[0])
    assert abs(z[0] - max(ndtri(1e-9), -0.95 / 0.2)) < 1e-12 and abs(z[-1] - ndtri(1 - 1e-9)) < 1e-12
    assert np.all(1 + 0.2 * z >= 0.05 - 1e-12) and abs(nodes["W"].sum() - 1.0) < 1e-12 and np.all(nodes["W"] > 0)


def test_node_parameters_rescale_as_xc():
    p = G._node_point(K01_P00, 0.5)
    assert p == dict(rho_eta=K01_P00["rho_eta"] / 0.5, rho_beta=K01_P00["rho_beta"] / 0.5,
                     rho_eps=K01_P00["rho_eps"] / 0.25, kappa=K01_P00["kappa"] / 0.5)


def test_het_draw_interpolates_the_cdf_between_bracketing_nodes():
    tau = np.linspace(0.0, 4.0, 4001)
    Fm = np.vstack([np.clip(tau / T, 0, 1) for T in (1.0, 2.0, 4.0)])          # node j: uniform on [0, T_j]
    u = np.full(3, 0.5)
    j = np.array([0, 0, 1])
    w = np.array([0.0, 0.5, 1.0])
    t, _ = G._inverse(u, tau, Fm, j, w)
    # at w = 0, node 0's quantile; at w = 1 on interval 1, node 2's; at w = 0.5 the root of 0.5 F_0 + 0.5 F_1 = 0.5
    assert abs(t[0] - 0.5) < 1e-9 and abs(t[2] - 2.0) < 1e-9 and abs(t[1] - 2.0 / 3.0) < 1e-6


def test_individuals_outside_the_node_range_are_counted():
    z = np.array([-10.0, 0.0, 10.0])
    j, w, outside = G._bracket(z, G._het_nodes(0.2, 9)["z"])
    assert outside == 2 and (j[0], w[0]) == (0, 0.0) and w[2] in (0.0, 1.0)


# ------------------------------------------------------------------------------------------- T011: censoring, recording
def test_censoring_time_is_the_grid_point_whose_survival_is_nearest_the_target():
    # Naveh 2026-10-14 ("Nearest in probability"): of the two grid points around t*, the one whose model survival is
    # closer to censor_frac; never 0
    S_obs = lambda t: np.exp(-t / 10.0)                                       # noqa: E731
    assert G._censor_time(S_obs, delta=1.0, f=0.3, t_end=400.0) == 12.0          # t* = 12.04; S(12) 0.301, S(13) 0.273
    sharp = lambda t: 1.0 / (1.0 + np.exp((np.asarray(t, float) - 10.4) / 0.1))   # noqa: E731
    # t* = 10.485: nearest in time is 10 (S 0.982), nearest in probability 11 (S 0.0025)
    assert G._censor_time(sharp, delta=1.0, f=0.3, t_end=40.0) == 11.0
    assert G._censor_time(S_obs, delta=1.0, f=0.99, t_end=400.0) == 1.0          # t* = 0.1: never 0
    with pytest.raises(UnsupportedOptionError) as e:
        G._censor_time(lambda t: np.exp(-t / 10.0) * (t < 50), delta=1.0, f=1e-15, t_end=50.0)
    assert e.value.option == "censor_frac"


def test_recording_floors_to_the_grid_with_first_bin_deaths_at_zero_and_no_death_at_C():
    d, C = 1.0, 12.0
    t = np.array([0.3, 1.0, 1.7, 11.999999999999998, 12.0, 15.0, 11.0])
    ages, ev = G._record(t, C, d)
    assert list(ages) == [0.0, 1.0, 1.0, 11.0, 12.0, 12.0, 11.0]
    assert list(ev) == [1, 1, 1, 1, 0, 0, 1]
    ages2, ev2 = G._record(np.array([0.3, 2.5]), None, d)
    assert list(ages2) == [0.0, 2.0] and list(ev2) == [1, 1]


# ------------------------------------------------------------------------------------------- T013: time_range
def _testers_apply_window():
    path = os.path.join(TESTER, "srbench", "library", "observation.py")
    if not os.path.exists(path):
        pytest.skip("NOT APPLICABLE: the tester's observation.py is not readable here")
    tree = ast.parse(open(path).read())
    ns = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "apply_window":
            exec(compile(ast.Module([node], []), "observation.py", "exec"), ns)
    return ns["apply_window"]


def test_time_range_equals_the_testers_apply_window():
    aw = _testers_apply_window()
    ages = np.array([0.0, 2.0, 3.0, 5.0, 7.0, 7.0, 8.0, 9.0])
    ev = np.array([1, 1, 1, 1, 1, 0, 1, 0])
    a1, e1, n_obs = G._apply_time_range(ages, ev, (3.0, 7.0))
    a0, e0 = aw(ages, ev, (3.0, 7.0))
    assert np.array_equal(a1, a0) and np.array_equal(e1, e0) and n_obs == a0.size
    assert list(e1) == [1, 1, 1, 0, 0, 0]               # a death at hi stays a death; beyond hi censored at hi


def test_a_window_that_keeps_no_one_is_refused():
    with pytest.raises(UnsupportedOptionError) as e:
        G._apply_time_range(np.array([1.0, 2.0]), np.array([1, 1]), (3.0, 7.0))
    assert e.value.option == "time_range"


# ------------------------------------------------------------------------------------------- T015: materialize
def test_materialize_returns_the_contract_and_is_deterministic(coarse):
    co = G.materialize(_spec(censor_frac=0.2))
    co2 = G.materialize(_spec(censor_frac=0.2))
    assert np.array_equal(co.death_times, co2.death_times) and np.array_equal(co.events, co2.events)
    a, e = co.death_times, co.events
    assert a.dtype == np.float64 and set(np.unique(e)) <= {0, 1} and np.all(np.diff(a) >= 0)
    assert np.allclose(a / 1.0, np.round(a / 1.0), atol=0)
    C = co.censor_time
    assert C is not None and np.all(a[e == 0] == C) and not np.any((a == C) & (e == 1))
    assert abs(co.realised_censored_fraction - float((e == 0).mean())) < 1e-15 and co.n_observed == a.size
    p = co.provenance
    for k in ("generator_version", "grid", "prng", "seed", "package", "libraries"):
        assert k in p
    assert p["prng"] == "PCG64" and p["seed"] == 7 and p["grid"]["N"] == COARSE["N"]
    for k in ("lambda", "dlncv", "n_beyond_cdf", "n_outside_het_nodes"):
        assert k in co.grid_error


def test_materialize_with_hazard_and_heterogeneity_runs_and_counts(coarse):
    co = G.materialize(_spec(n=200, external_hazard=5.5, het_param="xc", het_var=0.2))
    assert co.death_times.size == 200 and np.all(co.events == 1)
    assert co.grid_error["n_outside_het_nodes"] >= 0


def test_materialize_time_range_equals_the_window_of_the_untruncated_cohort(coarse):
    aw = _testers_apply_window()
    full = G.materialize(_spec(n=300))
    lo, hi = float(np.quantile(full.death_times, 0.2)), float(np.quantile(full.death_times, 0.8))
    win = G.materialize(_spec(n=300, time_range=(lo, hi)))
    a0, e0 = aw(full.death_times, full.events, (lo, hi))
    assert np.array_equal(win.death_times, a0) and np.array_equal(win.events, e0) and win.n_observed == a0.size


def test_the_reference_law_uses_the_generators_censoring_time(coarse, monkeypatch):
    # C is the cohort's design, fixed per version: at fine delta a refinement can flip t* across a rounding edge, and
    # comparing two laws with different C measured the flip, not the grid (sweep pilot, 2026-10-13: censor-k2-p01).
    s = _spec(censor_frac=0.2)
    b = G.binned_distribution(s, COARSE)
    forced = b["C"] + s.delta
    b2 = G.binned_distribution(s, COARSE, C=forced)
    assert b2["C"] == forced and b2["ages"][-1] == forced and b2["events"][-1] == 0 and abs(b2["P"].sum() - 1) < 1e-12
    seen = []
    real = G.binned_distribution
    monkeypatch.setattr(G, "binned_distribution", lambda spec, grid, model=None, C=None: seen.append(C) or
                        real(spec, grid, model, C))
    G.grid_error(s, COARSE)
    assert seen[0] is None and seen[1] == b["C"]


def test_a_heterogeneous_cohort_above_the_accepted_n_carries_the_caveat(coarse):
    big = G.materialize(_spec(n=G.HET_N_CAP + 1, het_param="xc", het_var=0.2))
    small = G.materialize(_spec(n=1000, het_param="xc", het_var=0.2))
    hom = G.materialize(_spec(n=G.HET_N_CAP + 1))
    assert big.grid_error["accepted_n"] == G.HET_N_CAP and "accepted at n <= 20000" in big.grid_error["caveat"]
    # on this coarse test grid every cohort's own grid error exceeds its bar (D1 caveats); the size caveat is the point
    assert not any("accepted at n <=" in c for c in small.caveats) and small.grid_error["accepted_n"] == 1000
    assert not any("accepted at n <=" in c for c in hom.caveats) and hom.grid_error["accepted_n"] == G.HET_N_CAP + 1


def test_the_first_curve_in_a_fresh_process_is_computed_in_float64():
    # 2026-10-14: generator.py built theta with jnp before the engine's import enabled x64, so the first curve in a
    # process ran on float32-rounded parameters (|dF| up to 7.8e-6 at a human anchor); later calls were float64.
    import subprocess
    import sys
    code = ("import hashlib, sraging.generator as G\n"
            "P = dict(rho_eta=25348.5, rho_beta=82515.3, rho_eps=2789.06, kappa=0.186)\n"
            "g = {**G.GRID, 'N': 300, 'dt': 1e-3}\n"
            "h = [hashlib.sha256(G._solve_curve(P, g)['F'].tobytes()).hexdigest() for _ in range(2)]\n"
            "import jax; print(h[0] == h[1], jax.config.jax_enable_x64)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300).stdout.split()
    assert out[-2:] == ["True", "True"]


def test_lambda_merges_cells_no_test_of_n_individuals_could_resolve():
    # Naveh 2026-10-14 (after the full sweep, run 1): cells are merged left to right until n Q >= 5, the G-test's rule,
    # so far-tail cells where one grid gives exactly 0 and the other 1e-8 cannot dominate lambda
    P = np.array([0.5, 0.49, 1e-8, 1e-8, 0.01 - 2e-8])
    Q = np.array([0.5, 0.49, 0.0, 2e-8, 0.01 - 2e-8])
    assert G.kl(P, Q) > 1e-6                                       # the floored 0 dominates the raw KL
    n = 100000
    merged = G.kl_binned(P, Q, n)
    assert merged < 1e-12                                          # the tail cells merge into the last: KL ~ 0
    # hand computation: [0.5, 0.49, 0.01] vs the same
    assert G.kl_binned(np.array([0.3, 0.7]), np.array([0.4, 0.6]), 20) == pytest.approx(     # 8 and 12 expected
        0.3 * math.log(0.3 / 0.4) + 0.7 * math.log(0.7 / 0.6))
    assert G.kl_binned(np.array([0.3, 0.7]), np.array([0.4, 0.6]), 10) == 0.0    # 4 < 5 expected: one cell


def test_the_grid_is_the_sweeps_and_the_packaged_table_is_its_point_table():
    # T021: the version's grid and grid-error table are frozen from the committed sweep (spec 003d SC-001, research R3)
    import json as _json
    import os as _os
    repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    sweep = _json.load(open(_os.path.join(repo, "tests", "acceptance", "results", "sweep_003d.json")))
    assert sweep["status"] == "pass" and sweep["status_for_acceptance"] == "acceptance"
    ch = sweep["choice"]
    assert (G.GRID["N"], G.GRID["dt"], G.GRID["K"]) == (ch["N"], ch["dt"], ch["K"])
    assert (G.GRID["tau_max"], G.GRID["Gtol"], G.GRID["nstart"]) == (30.0, 1e-12, 20)
    pt = _json.load(open(_os.path.join(repo, "tests", "acceptance", "results", "point_table_003d.json")))
    tab = _json.load(open(G.TABLE_PATH))
    assert tab["generator_version"] == G.GENERATOR_VERSION and tab["grid"] == G.GRID
    assert [r["key"] for r in tab["rows"]] == [r["key"] for r in pt["rows"]] and len(tab["rows"]) == 569
    assert all(a["kl"] == b["kl"] and a["dlncv"] == b["dlncv"] for a, b in zip(tab["rows"], pt["rows"]))


def test_a_library_configuration_reads_its_grid_error_from_the_table():
    s = G.CohortSpec(name="pure-k0.1-p20-c0-s0", rho_eta=0.16193367995080452, rho_beta=2.424462017082328,
                     rho_eps=1.0, kappa=0.1, n=1000, L_phys=100.0, seed=106800, delta=1.0)
    rows = G._table(G.TABLE_PATH, G.GENERATOR_VERSION, tuple(sorted(G.GRID.items())))
    assert rows is not None and _json_key(s) in rows


def _json_key(s):
    import json as _json
    return _json.dumps(G.configuration_key(s))


def test_the_curve_cache_returns_what_a_fresh_solve_gives():
    a = G._curve(K01_P00, COARSE)
    b = G._solve_curve(K01_P00, COARSE)
    assert np.array_equal(a["F"], b["F"]) and np.array_equal(a["S"], b["S"]) and np.array_equal(a["tau"], b["tau"])
    assert G._curve(K01_P00, COARSE) is a                                     # the second call is the cached one


# SC-007: every refusal case in contracts/python-api.md has a test (T029). Numbered as the contract lists them.
REFUSAL_TESTS = {
    1: ["test_spec_refuses_unknown_fields_by_name"],
    2: ["test_spec_refuses_positional_construction"],
    3: ["test_spec_refuses_out_of_range_values_by_name"],                     # delta=None
    4: ["test_spec_refuses_heterogeneity_other_than_xc"],
    5: ["test_spec_refuses_out_of_range_values_by_name"],                     # time_range off the delta grid
    6: ["test_curve_refuses_a_horizon_that_does_not_absorb"],
    7: ["test_censoring_time_is_the_grid_point_whose_survival_is_nearest_the_target"],
    8: ["test_spec_refuses_out_of_range_values_by_name"],                     # censor_frac outside [0, 1)
    9: ["test_a_window_that_keeps_no_one_is_refused"],
    10: ["test_spec_refuses_out_of_range_values_by_name", "test_spec_has_no_required_field_missing_silently"],
}


def test_every_contract_refusal_has_a_test():
    import os as _os
    import re as _re
    repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    text = open(_os.path.join(repo, "specs", "003d-cohort-generator", "contracts", "python-api.md")).read()
    section = text.split("## Refusals", 1)[1].split("\n## ", 1)[0]
    items = _re.findall(r"^(\d+)\. ", section, _re.M)
    assert [int(i) for i in items] == sorted(REFUSAL_TESTS)
    for names in REFUSAL_TESTS.values():
        for name in names:
            assert callable(globals().get(name)), name


def test_any_cohort_outside_its_bars_says_so(coarse):
    # Naveh 2026-10-14 (D1 at the tag gate): a caveat whenever the cohort's own lambda or dlnCV, at its real n, exceeds
    # its bar, homogeneous or heterogeneous; metadata only
    big = G.materialize(G.CohortSpec(name="b", **K2_P05, n=100000, L_phys=100.0, seed=1, delta=0.1))
    assert big.grid_error["lambda"] > big.grid_error["lambda_tolerance"]
    assert any("lambda" in c for c in big.caveats) and big.grid_error["caveat"]
    small = G.materialize(G.CohortSpec(name="s", **K01_P20, n=50, L_phys=100.0, seed=1, delta=1.0))
    assert small.caveats == () and small.grid_error["caveat"] is None


def test_a_censoring_miss_over_a_tenth_says_so(coarse):
    # Naveh 2026-10-14 (D2): keep nearest-in-probability, and say so when the realised fraction misses by > 0.1
    co = G.materialize(G.CohortSpec(name="c", **K2_P05, n=2000, L_phys=100.0, seed=3, delta=10.0, censor_frac=0.2))
    miss = abs(co.realised_censored_fraction - 0.2)
    assert (miss > 0.1) == any("censored fraction" in c for c in co.caveats)
