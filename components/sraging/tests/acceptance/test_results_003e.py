"""Spec 003e's committed acceptance results (tests/acceptance/results/*_003e.json): each phase's outcome, and SC-009 --
every result file carries FR-011's provenance (the library and its manifest sha256, the tester commit on a clean
clone, the reference run manifests' sha256, the content pins' sha256, the host) and was written by a clean run. The
checker is shown to fail on planted omissions in tests/unit/test_acceptance_003e.py."""
import glob
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "src")]
import acceptance_003e as ACC  # noqa: E402

EXPECTED = {"diff": "no change", "pin": "written", "known_answer": "PASS", "self_pair": "PASS", "refusals": "PASS",
            "v033": "PASS", "jcross": "REPORTED", "loader": "PASS", "age0_bytes": "PASS", "unchanged": "PASS"}


def load(name):
    return json.load(open(os.path.join(ACC.RESULTS_DIR, f"{name}_003e.json")))


def test_every_phase_has_a_committed_result():
    have = sorted(os.path.basename(p)[:-len("_003e.json")] for p in glob.glob(os.path.join(ACC.RESULTS_DIR,
                                                                                           "*_003e.json")))
    assert have == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_phase_has_its_outcome(name):
    assert load(name)["status"] == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_sc009_every_result_carries_fr011_provenance(name):
    assert ACC.provenance_problems(name, load(name)) == []


#: what a 003e result depends on: a change to any of these after a result ran makes it stale (sr-naveh finding 14)
CODE = ["src/sraging", "tools/acceptance_003e.py", "tools/acceptance_003.py", "tests/acceptance/panel_003e.toml",
        "tests/acceptance/pins_003e.json"]
#: spec 003e's merge (PR #6). Later specs change the code legitimately; the rule pins 003e's results to 003e's own
#: last code (spec 003f, 2026-10-18), not to whatever HEAD is.
CLOSED_AT = "739d37e"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_result_ran_at_the_code_now_committed(name):
    import subprocess
    d = load(name)
    # pin writes the pins; diff is git only and never reads them (its provenance carries none, as SC-009 allows)
    paths = [p for p in CODE if not (name in ("pin", "diff") and p.endswith("pins_003e.json"))]
    r = subprocess.run(["git", "-C", REPO, "diff", "--name-only", d["repo_head"], CLOSED_AT, "--", *paths],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not r.stdout.split(), f"{name} ran at {d['repo_head'][:7]}; changed since: {r.stdout.split()}"
