"""Spec 003d T033 (SC-006): the 12 witness cohorts' checksums at the frozen grid. Any byte change in generation fails
here; a deliberate change is a new generator version with a new panel."""
import json
import os

import pytest

import sraging.generator as G
import v5_reproduce_003d as R

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "data", "checksum_panel_003d.json")))


def test_the_panel_was_taken_at_this_grid():
    assert FIX["grid"] == G.GRID and len(FIX["cohorts"]) == 12


@pytest.mark.parametrize("item", FIX["cohorts"], ids=[c["spec"]["name"] for c in FIX["cohorts"]])
def test_the_witness_cohorts_bytes_are_unchanged(item):
    co = G.materialize(G.CohortSpec(**item["spec"]))
    assert R.cohort_digest(co.death_times, co.events) == item["digest"]


PATHS = [pytest.param(p, id=p["path"], marks=[pytest.mark.slow] if p["slow"] else [])
         for p in FIX.get("paths", [])]


@pytest.mark.parametrize("item", PATHS)
def test_each_generation_path_keeps_its_bytes(item):
    # Naveh 2026-10-14 (D4 at the tag gate): one cohort per path -- censoring, hazard, heterogeneity (slow: 161 node
    # curves), realfull, big n, and a window generated with time_range
    spec = dict(item["spec"])
    if "time_range" in spec:
        spec["time_range"] = tuple(spec["time_range"])
    co = G.materialize(G.CohortSpec(**spec))
    assert R.cohort_digest(co.death_times, co.events) == item["digest"]


def test_the_panel_covers_every_path():
    assert {p["path"] for p in FIX["paths"]} == {"censor", "hazard", "heterogeneous", "realfull", "big n",
                                                  "window (time_range)"}
