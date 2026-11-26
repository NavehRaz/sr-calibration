"""S4's grid (tasks T077; research R18; spec FR-015): per parameter x cv, v = 0 cells for Jack Russell and E. coli
(C13), about 60 clamp rows at n = 25,000 and 8 reflect rows equal to P1's cells; each row's step is P1's (refused
without one); the eta(1 + delta) plant's delta sized for a predicted sup|dS| = 0.033 on its 2 cells. The grid's
members are frozen at freeze-L ([s4]); a spec of that shape is built here."""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import s4_fp_langevin_004 as S4  # noqa: E402

PRESETS = ("mice_F", "Jack_Russell", "ecoli", "human")
SPEC = {"het_presets": list(PRESETS), "parameters": ["eta", "beta", "epsilon", "kappa", "xc"], "cvs": [0.1, 0.2, 0.3],
        "v0_presets": ["Jack_Russell", "ecoli"], "n": 25_000}


def p1(missing=()):
    """P1's steps per (preset, hom | het, wall), as its result writes them."""
    steps = {}
    for p in PRESETS:
        for het in ("hom", "het"):
            for wall in ("clamp", "reflect"):
                k = S4.step_key(p, het, wall)
                if k not in missing:
                    steps[k] = {"m": 8, "dt": 0.05}
    return {"steps": steps, "cells": [{"preset": p, "het": h} for p in PRESETS for h in ("hom", "het")]}


def test_the_grid_is_per_parameter_and_cv_with_v0_cells_for_jack_russell_and_ecoli():
    g = S4.build_grid(SPEC, p1())
    clamp = g["clamp"]
    assert 55 <= len(clamp) <= 65 and all(r["n"] == 25_000 and r["wall"] == "clamp" for r in clamp)
    het = [r for r in clamp if r["cv"] > 0]
    assert len(het) == 4 * 5 * 3 and {r["parameter"] for r in het} == set(SPEC["parameters"])
    v0 = [r for r in clamp if r["cv"] == 0]
    assert sorted(r["preset"] for r in v0) == ["Jack_Russell", "ecoli"] and all(r["parameter"] is None for r in v0)
    assert len({r["id"] for r in clamp}) == len(clamp)


def test_the_reflect_rows_are_p1s_cells():
    g = S4.build_grid(SPEC, p1())
    refl = g["reflect"]
    assert len(refl) == 8 and all(r["wall"] == "reflect" for r in refl)
    assert sorted((r["preset"], str(r["parameter"]), r["cv"]) for r in refl) == sorted(
        (p, "None", 0.0) if h == "hom" else (p, "xc", 0.2) for p in PRESETS for h in ("hom", "het"))


def test_each_row_carries_p1s_step_and_m_max_twice_it():
    g = S4.build_grid(SPEC, p1())
    assert all(r["m"] == 8 and r["dt"] == 0.05 and r["m_max"] == 16 for r in g["clamp"] + g["reflect"])


def test_a_row_without_a_p1_step_is_refused():
    with pytest.raises(S4.Refused, match="no P1 step"):
        S4.build_grid(SPEC, p1(missing=(S4.step_key("ecoli", "het", "clamp"),)))


def test_the_plants_delta_is_sized_for_a_predicted_sup_ds_of_0033():
    from sraging import SRParams
    from sraging.engine.inputs import Heterogeneity
    p = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
    for het in (None, Heterogeneity(parameter="xc", sd=0.2)):
        d = S4.plant_delta(p, het)
        assert 0 < d["delta"] < 1 and d["predicted_sup_dS"] == pytest.approx(0.033, abs=2e-4)
        assert d["target"] == S4.PLANT_TARGET == 2 * 0.0165


def test_the_crit_and_bar_follow_the_family():
    assert S4.crit("clamp", 25_000) == pytest.approx(1.973 / math.sqrt(25_000))
    assert S4.crit("reflect", 25_000) == pytest.approx(1.698 / math.sqrt(25_000))


def test_ecoli_leaves_the_clamp_family_and_stays_in_reflect():
    """Naveh 2026-11-03 (ff794f5): the default clamp is not shown to agree with FP for E. coli at any affordable
    step; every E. coli clamp row leaves the clamp family; its reflect rows stay."""
    g = S4.build_grid({**SPEC, "clamp_excluded": ["ecoli"]}, p1())
    assert not any(r["preset"] == "ecoli" for r in g["clamp"]) and len(g["clamp"]) == 3 * 5 * 3 + 1
    assert sorted(r["preset"] for r in g["reflect"] if r["preset"] == "ecoli") == ["ecoli", "ecoli"]
    twin = S4.build_grid(SPEC, p1())                                      # the twin: no exclusion
    assert sum(r["preset"] == "ecoli" for r in twin["clamp"]) == 16


def test_a_row_borrows_its_cells_step_unless_it_is_xc_at_cv_02():
    g = S4.build_grid(SPEC, p1())
    by = {r["id"]: r for r in g["clamp"]}
    assert not S4.borrowed(by["mice_F/xc/cv0.2/clamp"]) and not S4.borrowed(by["Jack_Russell/v0/clamp"])
    assert S4.borrowed(by["mice_F/eta/cv0.2/clamp"]) and S4.borrowed(by["mice_F/xc/cv0.1/clamp"])
