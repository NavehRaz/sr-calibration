"""HR-11's phase `multi` (tasks T109; research R27-R29; contracts/acceptance.md §4): (i) the reductions under both
walls and the shared-stream plant; (ii) the 6 cells on the default clamp at P1's heterogeneous steps, judged on the
dt/2 rung against the tensor reference, the composite cross-check of one 2-parameter cell, the dropped-spread plants
with their predicted sizes computed first; (iii) one above-budget statement. The configuration is frozen at freeze-L
([multi]); a spec of that shape is built here."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

import multi_004 as MU  # noqa: E402
import s4_fp_langevin_004 as S4  # noqa: E402
from sraging import SRParams  # noqa: E402
from sraging import sampler_rules as R  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
CELLS = [{"id": "xc+eta", "members": {"xc": 0.2, "eta": 0.2}},
         {"id": "xc+epsilon", "members": {"xc": 0.2, "epsilon": 0.2}},
         {"id": "eta+beta", "members": {"eta": 0.2, "beta": 0.2}},
         {"id": "epsilon+kappa", "members": {"epsilon": 0.2, "kappa": 0.2}},
         {"id": "xc+eta+epsilon", "members": {"xc": 0.2, "eta": 0.2, "epsilon": 0.2}},
         {"id": "eta+beta+kappa", "members": {"eta": 0.2, "beta": 0.2, "kappa": 0.2}}]
SPEC = {"preset": "mice_F", "cells": CELLS, "n": 25_000, "G": 6, "key": 4200}


def p1(missing=()):
    steps = {S4.step_key(p, h, w): {"m": 8, "dt": 0.05} for p in ("mice_F",) for h in ("hom", "het")
             for w in ("clamp", "reflect")}
    for k in missing:
        steps.pop(k)
    return {"steps": steps}


def test_the_six_cells_on_the_default_clamp_at_p1s_heterogeneous_step():
    rows = MU.build_cells(SPEC, p1())
    assert len(rows) == 6 and {r["id"] for r in rows} == {c["id"] for c in CELLS}
    assert all(r["wall"] == "clamp" and r["n"] == 25_000 and r["m"] == 8 and r["dt"] == 0.05 and r["m_max"] == 16
               and r["p1_step"] == S4.step_key("mice_F", "het", "clamp") for r in rows)
    het = MU.members(rows[0]["members"])
    assert [h.parameter for h in het] == ["eta", "xc"] and all(h.sd == 0.2 for h in het)


def test_cells_without_a_p1_step_or_off_the_family_are_refused():
    with pytest.raises(MU.Refused, match="no P1 step"):
        MU.build_cells(SPEC, p1(missing=(S4.step_key("mice_F", "het", "clamp"),)))
    with pytest.raises(MU.Refused, match="G = 6"):
        MU.build_cells({**SPEC, "cells": CELLS[:5]}, p1())
    with pytest.raises(MU.Refused, match="d <= 3"):
        MU.build_cells({**SPEC, "cells": CELLS[:5] + [{"id": "four", "members": {"xc": 0.2, "eta": 0.2, "beta": 0.2,
                                                                                    "kappa": 0.2}}]}, p1())


def test_the_reductions_are_exact_under_both_walls_and_the_shared_stream_plant_breaks_them():
    rows = MU.reductions(P, [{"single": {"xc": 0.2}, "zero": ["eta"]}], n=300, key=5, dt=0.5, t_max=300.0)
    cases = [r for r in rows if r["kind"] == "case"]
    assert sorted(r["wall"] for r in cases) == ["clamp", "reflect"] and all(r["equal"] for r in cases)
    plant = [r for r in rows if r["kind"] == "plant"]
    assert len(plant) == 1 and plant[0]["breaks"]


def test_the_dropped_spread_plants_size_is_computed_first():
    d = MU.plant_size(P, {"xc": 0.2, "eta": 0.2}, "eta", bar=1e-3)
    assert d["predicted_sup_dS"] > 0 and d["drop"] == "eta" and d["M"]["eta"] >= 5


def test_a_cell_is_judged_on_the_dt2_rung_against_the_tensor_reference():
    row = {"id": "xc+eta", "members": {"xc": 0.2, "eta": 0.2}, "wall": "clamp", "n": 800, "m": 1, "dt": 0.5,
           "m_max": 2, "p1_step": "k"}
    c = MU.run_cell(row, P, 3, 0.5)
    assert c["reference_kind"] == "tensor" and c["reference_verdict"] == "converged" and c["D"] is not None
    assert set(c) >= {"id", "n", "D", "D_bias", "p_hat", "m", "m_max"} and 1 <= c["m"] <= 2


def test_the_composite_cross_check_reports_its_sup_ds():
    out = MU.composite(P, {"xc": 0.2, "eta": 0.2}, panels=(2, 4), order=4, agree=1.0, M={"xc": 5, "eta": 5})
    assert out["sup_dS"] >= 0 and out["panels"] == 4 and out["converged"] and out["nodes"] == 16 * 16


def test_an_above_budget_configuration_reads_no_fp_reference(monkeypatch):
    from sraging.engine import tensor_reference as TR
    monkeypatch.setattr(TR, "BUDGET", 20)
    a = MU.above_budget(P, {"xc": 0.2, "eta": 0.2}, n=300, key=4, dt=0.5, t_max=300.0)
    assert a["kind"] == "none" and not a["clean"]
    v = R.hr11_verdict([], [], [], None, a, cpu_spent=0.0)
    assert not v.statistic["above_budget_wrong"]


def test_the_phase_is_registered():
    import acceptance_004 as ACC
    assert ACC.PHASES["multi"] is ACC.phase_multi


def test_a_cell_runs_once_at_the_borrowed_frozen_step():
    """Amendment A1 (Naveh 2026-11-03): no ladder; the frozen cell's pooled bias, borrowed and guarded."""
    row = {"id": "xc+eta", "members": {"xc": 0.2, "eta": 0.2}, "wall": "clamp", "n": 600, "m": 1, "dt": 0.5,
           "m_max": 2, "p1_step": "k"}
    c = MU.run_frozen(row, P, 3, {"p_hat": 0.5, "pooled_b": 0.0008, "D_key_mean": 0.0015})
    assert c["borrowed"] and c["frozen_b"] == 0.0008 and c["m"] == 1 and "ladder" not in c
    assert c["reference_kind"] == "tensor" and c["D"] is not None and c["D_own"] >= 0
