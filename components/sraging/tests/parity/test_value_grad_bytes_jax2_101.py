"""Value and value+gradient bytes of the jax2 1.0.1 port against the frozen 1.0.1 oracle where the first-interval
mass is below 1e-16, with its twin on v0.3.3's numerics (spec 003b, T013; research R8; 003e close item 5).

PANEL [parity.bytes] holds one theta per hazard / age-0 cohort of the parity panel, found by the deterministic scan
`tools/jax2_cases_101.py scan`. At each, side by side in ONE process of its own (the 1.0.1 oracle cannot share a
process with the 1.0.0 oracle the rest of this suite loads): the port's and the oracle's value and value+gradient
bytes are equal (the J objective; the _h entry also the hazard objective at two hazard rates), the age-0 term is the
recorded one, finite and below log(1e-16), and v0.3.3's S(0) - S(delta) scores the -700 floor there.
NOT APPLICABLE (never a pass) away from this Mac or the pinned environment.
"""
import json
import math
import os
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from conftest import not_applicable

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))
from parity_checks import bytes_diff  # noqa: E402

PANEL = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "panel_003b.toml"), "rb"))
ENTRIES = (PANEL.get("parity") or {}).get("bytes") or []


@pytest.fixture(scope="module")
def checked():
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "jax2_cases_101.py"), "check"], cwd=REPO,
                       capture_output=True, text=True, timeout=1800)
    lines = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
    assert r.returncode == 0 and lines, r.stdout[-2000:] + r.stderr[-4000:]
    if "not_applicable" in lines[0]:
        not_applicable("; ".join(lines[0]["not_applicable"]))
    return {(ln["cid"], ln["name"]): ln for ln in lines}


def test_the_panel_has_one_entry_per_hazard_age0_cohort():
    pairs = {(p["cid"], p["name"]) for p in PANEL["parity"]["pairs"] if p["cid"].startswith("hazard-")}
    assert {(e["cid"], e["name"]) for e in ENTRIES} == pairs and len(ENTRIES) == 2


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["cid"])
def test_port_and_oracle_bytes_are_equal_below_1e16_and_v033_scores_the_floor(checked, entry):
    got = checked[(entry["cid"], entry["name"])]
    assert got["bytes_equal"], got["diffs"]
    assert got["term_bytes_equal"] and float(got["term"]).hex() == float(entry["term"]).hex()
    assert got["mass"] < 1e-16 and got["term"] > -690.0 and math.isfinite(got["term"])
    assert got["n_age0"] == entry["n_age0"] and got["n_age0_v033"] == entry["n_age0"]
    assert got["v033_terms"] == [-700.0]                     # the twin: S(0) - S(delta) cancels to the floor
    if entry["name"] == "profile_jax2_h":
        assert {"jax2", "jax2_h@box_floor", "jax2_h@hL=0.1"} == set(got["diffs"])


def test_a_one_ulp_change_of_the_term_is_caught():
    """Constitution V: the byte comparison fails on a planted one-ulp perturbation."""
    v = np.asarray(ENTRIES[0]["term"])
    assert bytes_diff(np.nextafter(v, np.inf), v, "term") != [] and bytes_diff(v.copy(), v, "term") == []
