"""The stated cost (tasks T029; research R8; data-model "Cost statement"; FR-008): stated_cpu = M x t_hom x f_mode,
with compilation apart; one statement per rung of the ladder; the factor carries its source (P0's result file and its
sha256, read from panel_004.toml [cost] after freeze-H, "unfrozen" before); an M-blind predictor falls outside [0.5, 2]
where M >= 31, its twin inside."""
import hashlib
import math

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine import cost as COST
from sraging.engine import mixture as MIX
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import Heterogeneity
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(3)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 300)) + 1.0)
E = (RNG.random(300) < 0.9).astype(int)
C = Cohort.from_arrays(T, E, delta=1.0)
GRID = GridPolicy("fixed", N=200, dt=2.0 ** -8, tau_max=2.5, L=124.0)
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)


@pytest.fixture(scope="module")
def ev():
    return Evaluator(C, numerics="v1", grid=GRID, heterogeneity=Heterogeneity(parameter="xc", sd=0.2))


def test_the_statement_is_m_times_t_hom_times_the_factor():
    f = COST.factors(panel_path="/nonexistent/panel_004.toml")
    for mode, key in (("value", "value"), ("value_and_grad", "value_and_grad")):
        s = COST.state(mode, 38, 0.01, factors=f, workers=1, phase="rung")
        assert s.stated_cpu_s == 38 * 0.01 * f[key] and s.factor == f[key] and s.M == 38 and s.t_hom_cpu_s == 0.01
        assert s.stated_wall_s == s.stated_cpu_s
    s3 = COST.state("value", 38, 0.01, factors=f, workers=3)
    assert s3.stated_cpu_s == 38 * 0.01 * f["value"] and s3.stated_wall_s == math.ceil(38 / 3) * 0.01 * f["value"]
    p2 = COST.state("pass2", 38, 0.01, factors=f)
    assert p2.stated_cpu_s == 38 * 0.01 * (f["value_and_grad"] - f["value"])
    m = s.measured(0.5, 0.6)
    assert (m.measured_cpu_s, m.measured_wall_s, m.ratio) == (0.5, 0.6, 0.5 / s.stated_cpu_s)
    with pytest.raises(UnsupportedOptionError):
        COST.state("jacfwd", 38, 0.01, factors=f)


def test_every_rung_of_the_ladder_has_its_statement_and_compilation_is_apart(ev):
    r = ev.mixture_value(P)
    st = r.cost_record["statements"]
    rungs = [x for x in st if x["phase"] == "rung"]
    assert [x["M"] for x in rungs] == [x["M"] for x in r.node_record["check"]["ladder"]]
    t = r.cost_record["t_hom_cpu_s"]
    f = r.cost_record["factor"]
    for x in rungs:
        assert x["stated_cpu_s"] == x["M"] * t * f["value"] and x["measured_cpu_s"] > 0
    assert [x["phase"] for x in st] == ["guide"] + ["rung"] * len(rungs)
    assert "compile_cpu_s" in r.cost_record and r.cost_record["measured_cpu_s"] == pytest.approx(
        sum(x["measured_cpu_s"] for x in st))
    g = ev.mixture_value(P, grad=True).cost_record["statements"]
    assert g[-1]["phase"] == "pass 2" and g[-1]["mode"] == "pass2"


def test_before_freeze_h_the_factor_reads_unfrozen():
    f = COST.factors(panel_path="/nonexistent/panel_004.toml")
    assert f["source"]["status"] == "unfrozen" and "unfrozen" in f["source"]["note"]
    assert f["value"] > 0 and f["value_and_grad"] > f["value"]


def test_after_freeze_h_the_factor_carries_p0s_file_and_sha256(tmp_path):
    res = tmp_path / "p0_het_cost_004.json"
    res.write_text('{"per_node_cpu_s": {"value": 0.009, "value_and_grad": 0.03}}')
    sha = hashlib.sha256(res.read_bytes()).hexdigest()
    panel = tmp_path / "panel_004.toml"
    panel.write_text(f'naveh_ok_h = "2026-11-02: OK"\n[cost]\nf_value = 0.88\nf_grad = 3.2\nnoise_log_sd = 0.1\n'
                     f'source = "{res.name}"\nsha256 = "{sha}"\n')
    f = COST.factors(panel_path=str(panel), root=str(tmp_path))
    assert (f["value"], f["value_and_grad"]) == (0.88, 3.2)
    assert f["source"] == {"status": "frozen", "file": res.name, "sha256": sha, "panel": str(panel)}
    panel.write_text(panel.read_text().replace(sha, "0" * 64))
    with pytest.raises(ValueError, match="sha256"):
        COST.factors(panel_path=str(panel), root=str(tmp_path))
    panel.write_text(panel.read_text().replace('naveh_ok_h = "2026-11-02: OK"', 'naveh_ok_h = ""'))
    assert COST.factors(panel_path=str(panel), root=str(tmp_path))["source"]["status"] == "unfrozen"


def test_an_m_blind_predictor_falls_outside_the_band_where_m_is_large_and_its_twin_inside(ev):
    """HR-5's plant: a predictor that ignores M (it states every evaluation at M_BLIND nodes)."""
    het = Heterogeneity(parameter="xc", sd=0.2, M=38, check=False)
    MIX.mixture_value(ev, P, het=het)                                    # warm: compilation is apart anyway
    r = MIX.mixture_value(ev, P, het=het)
    (rung,) = [x for x in r.cost_record["statements"] if x["phase"] == "rung"]
    assert rung["M"] == 38 >= 31
    blind = COST.blind(rung)
    assert 0.5 <= rung["measured_cpu_s"] / rung["stated_cpu_s"] <= 2.0                         # the twin
    assert not 0.5 <= rung["measured_cpu_s"] / blind["stated_cpu_s"] <= 2.0
    assert blind["M"] == COST.M_BLIND == 11


def test_price_states_without_evaluating_the_mixture(ev):
    s = ev.price(P, M=57, mode="value_and_grad", workers=3)
    assert s.M == 57 and s.mode == "value_and_grad" and s.t_hom_cpu_s > 0 and s.measured_cpu_s is None
    assert s.stated_cpu_s == 57 * s.t_hom_cpu_s * s.factor and s.workers == 3
    with pytest.raises(UnsupportedOptionError):
        ev.price(P, M=500)
