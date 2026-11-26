"""The engine switch (tasks T100; research R21; contracts/python-api.md §6; data-model "Population"): every planted
configuration and its twin (SC-009). `fp` refuses what FP cannot do, by name; `langevin` needs n; `auto` takes FP for
n = None, gen-1 inside its domain for n = int, and outside it the Langevin with ENGINE_AUTO_LANGEVIN naming why: P3's
boundary (tests/acceptance/results/p3_cost_004.json#boundary, encoded by T130).
gen-1 runs on a coarse grid here (its full grid costs minutes per heterogeneous draw), as tests/unit/test_generator.py
does."""
import json
import os
import sys

import numpy as np
import pytest

from sraging import SRParams
from sraging import generator as G
from sraging.engine.inputs import Heterogeneity
from sraging.errors import UnsupportedOptionError
from sraging.population import Population, population

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
AGES = [20.0, 60.0, 100.0]
COARSE = {"N": 300, "dt": 1e-3, "tau_max": 30.0, "Gtol": 1e-12, "nstart": 20, "K": 9}
XC, ETA = Heterogeneity(parameter="xc", sd=0.2), Heterogeneity(parameter="eta", sd=0.2)


@pytest.fixture
def coarse(monkeypatch):
    monkeypatch.setattr(G, "GRID", dict(COARSE))


def codes(pop):
    return [w.code for w in pop.warnings]


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_auto_takes_p3s_committed_boundary_on_every_one_of_its_rows():
    """T130 (C10): P3's result records, per configuration and n, the engine `auto` takes; the encoded rule gives the
    same engine on each row (the human preset, sd 0.2, delta 1, P3's key), without drawing."""
    from sraging.population import P3_RULE, auto_engine
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import presets_004 as PR
    p3 = json.load(open(os.path.join(REPO, "tests", "acceptance", "results", "p3_cost_004.json")))
    cfg, rows = p3["config"], p3["boundary"]["rows"]
    assert p3["status"] == "written" and P3_RULE["rule"] == p3["boundary"]["rule"] and len(rows) == 6
    param = dict(cfg["configs"])
    for r in rows:
        het = Heterogeneity(parameter=param[r["config"]], sd=cfg["sd"])
        engine, why = auto_engine(PR.preset(cfg["preset"]), het, r["n"], cfg["key"], cfg["delta"])
        assert engine == r["engine"], (r, why)
    assert {r["engine"] for r in rows} == {"gen-1", "langevin"}                    # both sides of it are on the rows


def test_eta_heterogeneity_at_n_5000_under_auto_is_langevin_with_a_warning():
    pop = population(P, AGES, heterogeneity=ETA, engine="auto", n=5_000, key=1, delta=1.0, dt=0.5)
    assert pop.engine_used == "langevin" and "ENGINE_AUTO_LANGEVIN" in codes(pop)
    w = next(w for w in pop.warnings if w.code == "ENGINE_AUTO_LANGEVIN")
    assert "eta" in str(w) and "eta" in pop.reason and pop.provenance["requested_engine"] == "auto"


def test_xc_heterogeneity_at_n_5000_under_auto_is_gen1_with_no_warning(coarse):
    pop = population(P, AGES, heterogeneity=XC, engine="auto", n=5_000, key=1, delta=1.0)
    assert isinstance(pop, Population) and pop.engine_used == "gen-1" and pop.kind == "sample"
    assert "ENGINE_AUTO_LANGEVIN" not in codes(pop) and pop.sample.spec.het_param == "xc"
    assert pop.provenance["engine"] == "gen-1" and pop.provenance["key"] == 1
    lo, hi = pop.bands["S_lo"], pop.bands["S_hi"]
    assert np.all(lo <= pop.S) and np.all(pop.S <= hi) and pop.S.shape == (3,)


def test_fp_refuses_two_heterogeneous_parameters_and_runs_one():
    with pytest.raises(UnsupportedOptionError) as ex:
        population(P, AGES, heterogeneity=(XC, ETA), engine="fp")
    assert ex.value.option == "heterogeneity"
    pop = population(P, AGES, heterogeneity=XC, engine="fp")                                           # the twin
    assert pop.kind == "exact" and pop.engine_used == "fp" and pop.S.shape == (3,) and np.all(np.diff(pop.S) < 0)
    assert pop.hazard.shape == (3,) and pop.sample is None
    nc = pop.provenance["node_count"]                                  # R26 for an exact population: sup|dS| <= 1e-4
    assert nc["verdict"] == "converged" and nc["sup_dS"] <= nc["tol"] == 1e-4 and nc["M"] in (5, 8, 11, 17, 25)


def test_fp_with_n_outside_gen1s_domain_is_refused_naming_why(coarse):
    with pytest.raises(UnsupportedOptionError) as ex:
        population(P, AGES, heterogeneity=ETA, engine="fp", n=500, key=1, delta=1.0)
    assert "xc" in str(ex.value)
    assert population(P, AGES, engine="fp", n=500, key=1, delta=1.0).engine_used == "gen-1"           # the twin


def test_langevin_needs_n():
    with pytest.raises(UnsupportedOptionError, match="pass n"):
        population(P, AGES, engine="langevin", key=1, dt=0.5)
    pop = population(P, AGES, engine="langevin", n=1_000, key=1, dt=0.5)                              # the twin
    assert pop.engine_used == "langevin" and pop.kind == "sample" and pop.sample.n == 1_000
    assert pop.provenance["wall"] == "clamp" and pop.provenance["step"] == 0.5


def test_a_tuple_under_auto_needs_n():
    with pytest.raises(UnsupportedOptionError, match="pass n"):
        population(P, AGES, heterogeneity=(XC, ETA), engine="auto")


def test_a_tuple_at_n_5000_under_auto_is_langevin_naming_more_than_one_parameter():
    pop = population(P, AGES, heterogeneity=(XC, ETA), engine="auto", n=5_000, key=1, dt=0.5)
    assert pop.engine_used == "langevin"
    w = [w for w in pop.warnings if w.code == "ENGINE_AUTO_LANGEVIN"]
    assert w and "more than one heterogeneous parameter" in str(w[0])


def test_the_wall_is_the_langevins_only():
    with pytest.raises(UnsupportedOptionError) as ex:
        population(P, AGES, engine="fp", wall="reflect")
    assert ex.value.option == "wall"
    pop = population(P, AGES, engine="langevin", n=1_000, key=1, dt=0.5, wall="reflect")             # the twin
    assert pop.provenance["wall"] == "reflect" and pop.sample.provenance["wall"] == "reflect"
    with pytest.raises(UnsupportedOptionError) as ex:
        population(P, AGES, engine="langevin", n=1_000, key=1, dt=0.5, wall="mirror")
    assert "clamp" in str(ex.value) and "reflect" in str(ex.value)


def test_auto_without_n_is_fp_and_options_fp_cannot_use_are_refused_by_name():
    pop = population(P, AGES, heterogeneity=XC)
    assert pop.engine_used == "fp" and pop.kind == "exact"
    for kw in ({"key": 1}, {"delta": 1.0}, {"dt": 0.5}, {"censor_frac": 0.1}, {"window": (0.0, 50.0)}):
        with pytest.raises(UnsupportedOptionError) as ex:
            population(P, AGES, engine="fp", **kw)
        assert ex.value.option == next(iter(kw))
    with pytest.raises(UnsupportedOptionError) as ex:
        population(P, AGES, engine="gibbs")
    assert ex.value.option == "engine"
