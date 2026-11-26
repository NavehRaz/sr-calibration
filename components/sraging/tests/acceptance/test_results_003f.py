"""Spec 003f's committed results (tests/acceptance/results/*_003f.json): each phase's outcome, every result's
leverage provenance (FR-010, SC-009), a clean run, the rule in force, and no result older than a change to its code.

Every phase ran after Naveh's decisions of 2026-10-19 (the review's batched change, 2a3b45d), so every result is judged
strictly: nothing under src/sraging, the 003f tool, spec 003's shared Run or the panel may have changed between the
commit it ran at and 003f's merge (CLOSED_AT, as 003e's test does; spec 003b T019): later specs change the code
legitimately, and HEAD is not 003f's code."""
import json
import os
import subprocess
import sys
import tomllib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
PANEL = "tests/acceptance/panel_003f.toml"
STRICT = ["src/sraging", "tools/acceptance_003f.py", "tools/acceptance_003.py", PANEL]

#: spec 003f's merge (PR #7, cce0623): its results are pinned to 003f's own last code, not to HEAD.
CLOSED_AT = "cce0623"

EXPECTED = {"p1": "PASS", "p2": "PASS", "p3": "written", "itp": "PASS", "panel": "PASS", "age0": "REPORTED",
            "readme_example": "REPORTED", "unchanged": "PASS"}
NOT_RERUN = {"p3_panel"}          # names P3's cohorts (no fit, no leverage code); its [p3] block is unchanged


def load(name):
    return json.load(open(os.path.join(RESULTS, f"{name}_003f.json")))


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


def rule():
    sys.path[:0] = [os.path.join(REPO, "src")]
    from sraging.leverage_rule import RULE
    return dict(RULE)


def test_every_phase_has_a_committed_result():
    have = sorted(f[:-len("_003f.json")] for f in os.listdir(RESULTS) if f.endswith("_003f.json"))
    assert have == sorted(set(EXPECTED) | NOT_RERUN)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_phase_has_its_outcome_from_a_clean_run(name):
    d = load(name)
    assert d["status"] == EXPECTED[name]
    assert d["status_for_acceptance"] == "acceptance"
    assert not d["tree_dirty"] and not d["tree_changed_during_run"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_result_carries_the_leverage_provenance_and_the_rule_in_force(name):
    p = load(name)["provenance"]
    for k in ("calibrator", "version", "host", "machine", "cpu", "rule", "panel_commit"):
        assert p.get(k), k
    assert p["calibrator"] == "profile_jax2" and p["rule"] == rule()
    if name in ("p3", "panel", "age0"):
        assert p["library"] == "v6"


def changed_since(head, until=CLOSED_AT) -> list:
    """The STRICT paths that changed between `head` (where a result ran) and `until` (003f's merge)."""
    r = git("diff", "--name-only", head, until, "--", *STRICT)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_result_ran_at_the_code_now_committed(name):
    head = load(name)["repo_head"]
    changed = changed_since(head)
    assert not changed, f"{name} ran at {head[:7]}; changed before 003f's merge: {changed}"


def test_the_bound_counts_a_pre_merge_change_and_not_a_later_one():
    """Planted: a result run at 1d00f02 (before 003f's own batched change 2a3b45d) is stale at the merge. Twin: at the
    merge itself nothing counts, although later specs (003b) changed src/sraging since."""
    assert "src/sraging/leverage.py" in changed_since("1d00f02")
    assert changed_since(CLOSED_AT) == []
    assert git("diff", "--name-only", CLOSED_AT, "HEAD", "--", "src/sraging").stdout.split() != []


def test_p3_panel_still_names_the_cohorts_p3_ran():
    d, pn = load("p3_panel"), tomllib.load(open(os.path.join(REPO, PANEL), "rb"))["p3"]
    assert d["strata"] == {k: pn[k] for k in d["strata"]}
    assert sorted(r["cohort_id"] for r in load("p3")["rows"]) == sorted(c for v in d["strata"].values() for c in v)


def test_the_rule_in_force_is_the_committed_panels():
    assert rule() == tomllib.load(open(os.path.join(REPO, PANEL), "rb"))["rule"]


def test_every_leverage_block_carries_its_provenance():
    for name in ("itp", "readme_example"):
        lev = load(name)["leverage"]
        assert lev["version"] == "003f/2" and lev["rule"] == rule()
        for k in ("calibrator", "version", "host", "cpu"):
            assert lev["provenance"].get(k), (name, k)
    for r in load("p3")["rows"]:
        assert r["production"]["provenance"]["library"] == "v6", r["cohort_id"]


def test_the_gates_and_the_positive_control():
    i = load("itp")
    assert i["SC-001"] and i["SC-002"] and all(i["FR-014"].values())
    assert i["leverage"]["label"] == "flag" and i["leverage"]["dropped"]["row"] == i["age0_row"]
    p = load("panel")
    gate = tomllib.load(open(os.path.join(REPO, PANEL), "rb"))["gate"]
    assert p["SC-003"] == "REPORTED" and p["SC-004"] is True
    assert p["overhead"]["p90"] <= gate["overhead_p90_ceiling"]
    p2 = load("p2")
    assert p2["refit_vs_stored_without"]["equal_to_round_off"] and p2["warm_start"]["not_below_its_start"]


def test_the_readme_example_is_the_stored_one_and_panel_and_age0_read_the_current_p3():
    r = load("readme_example")
    assert r["claims_equal_to_stored"] and r["theta_hat_equal_to_stored"]
    head = load("p3")["repo_head"]
    assert load("panel")["p3"]["repo_head"] == head and load("age0")["p3"]["repo_head"] == head


def test_p3_repeats_the_production_drop_bit_for_bit():
    s = load("p3")["summary"]
    assert s["top_repeats_production"] and all(s["top_repeats_production"].values())
