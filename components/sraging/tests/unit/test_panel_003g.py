"""Spec 003g's frozen gate panel (tasks T004; research R1; contracts/gate-and-panel.md): the committed id list is what
`tools/panel_003g.py` rebuilds from the six pinned sources, its counts are the spec's, and the builder refuses a source
whose bytes moved, a register list whose ids do not match its own digest, and a window id in J's domain."""
import hashlib
import json
import os
import sys
import tomllib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import panel_003g as P  # noqa: E402

PANEL = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "panel_003g.toml"), "rb"))
HAVE_SOURCES = all(os.path.exists(os.path.join(P.DROPBOX, s["path"])) for k, s in PANEL["sources"].items()
                   if k != "ids")

#: spec 003g FR-001, FR-002 and research R1 [M, 2026-10-22]
EXPECTED = {"BASIN-J": 32, "BASIN-P15": 3, "BASIN-H": 13, "LB-2": 264, "LB-1x": 35, "SRK-1": 238, "SRK-1b": 1,
            "SRK-2": 4, "P-ROUNDOFF": 324, "P-DTCEIL": 16, "P-CENSORTIE": 0}


def test_the_committed_counts_are_the_specs():
    got = {k: v["count"] for k, v in PANEL["strata"].items() if k in EXPECTED}
    assert got == EXPECTED


def test_the_register_union_in_js_domain_is_507_and_every_valid_row_is_informative():
    rows = P.read_ids(os.path.join(REPO, "tests", "acceptance", "panel_003g_ids.tsv"))
    register = {"BASIN-J", "BASIN-P15", "LB-2", "LB-1x", "SRK-1", "SRK-1b", "SRK-2"}
    union = [r for r in rows if set(r["strata"]) & register]
    assert len(union) == 507
    assert sum(r["half"] == "inside" for r in union) == 269 and sum(r["half"] == "outside" for r in union) == 238
    assert all(r["informative"] for r in union if r["valid"])      # the register's strata: valid == informative
    # the data-rule strata span the whole v6 J domain, valid het included: W is undefined there (J cannot score a
    # heterogeneous truth), so PR-1 reads valid informative rows only; coverage reads every valid row [M]
    pr1 = {s: sum(r["valid"] and r["informative"] for r in rows if s in r["strata"]) for s in ("P-ROUNDOFF", "P-DTCEIL")}
    assert pr1 == {"P-ROUNDOFF": 178, "P-DTCEIL": 12}
    assert not any(r["arm"] == "window" for r in rows)


def test_the_committed_tsv_is_the_one_the_toml_pins():
    b = open(os.path.join(REPO, "tests", "acceptance", "panel_003g_ids.tsv"), "rb").read()
    assert hashlib.sha256(b).hexdigest() == PANEL["sources"]["ids"]["sha256"]


@pytest.mark.skipif(not HAVE_SOURCES, reason="the pinned sources are not on this host")
def test_a_rebuild_from_the_pinned_sources_equals_the_committed_panel():
    assert P.check(PANEL) == []


# ------------------------------------------------------------------------------------------ planted refusals + twins
def test_a_source_whose_bytes_moved_is_refused(tmp_path):
    f = tmp_path / "src.txt"
    f.write_bytes(b"abc")
    good = {"s": {"path": "src.txt", "sha256": hashlib.sha256(b"abc").hexdigest()}}
    assert P.read_source(good, "s", root=str(tmp_path)) == b"abc"                       # twin
    bad = {"s": {"path": "src.txt", "sha256": hashlib.sha256(b"abd").hexdigest()}}
    with pytest.raises(P.Refused, match="sha256"):
        P.read_source(bad, "s", root=str(tmp_path))


def test_a_register_list_whose_ids_do_not_match_its_digest_is_refused():
    ids = ["b", "a"]
    ok = {"key": "X", "ids": ids, "ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}
    assert P.checked_ids(ok) == ["a", "b"]                                               # twin
    with pytest.raises(P.Refused, match="ids_sha256"):
        P.checked_ids(dict(ok, ids=["a", "b", "c"]))


def test_a_window_id_in_js_domain_is_refused():
    assert P.j_domain({"arm": "pure"}) and not P.j_domain({"arm": "hazard"})            # twin
    with pytest.raises(P.Refused, match="window"):
        P.require_no_window([{"cohort_id": "window-k0.1-p00-c0-s0", "arm": "window", "calibrator": "profile_jax2"}])
    P.require_no_window([{"cohort_id": "pure-k0.1-p00-c0-s0", "arm": "pure", "calibrator": "profile_jax2"}])


def test_the_content_digest_is_the_testers():
    """sraging's copy of srbench.library.build.cohort_digest: repr floats joined by |, then #, then the events."""
    want = hashlib.sha256(b"0.0|1.5#1|0").hexdigest()
    assert P.cohort_digest([0.0, 1.5], [1, 0]) == want
    assert P.cohort_digest([0.0, 1.5], [1, 1]) != want
