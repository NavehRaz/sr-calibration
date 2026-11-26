"""Spec 003b's committed results (tests/acceptance/results/*_003b.json; tasks T043): each phase's outcome from a clean
run on this Mac, every result's FR-014 provenance, and no result older than a change to its code.

Strict staleness: nothing under src/sraging, the 003b tools, spec 003's shared Run and port helpers, or the panel may
have changed between the commit a result ran at and 003b's merge (CLOSED_AT, as 003f's test does; spec 003g FR-017):
later specs change the code legitimately, and HEAD is not 003b's code. The pilot (lev-measure) is exempt only from
what Naveh's post-pilot step was allowed to change: his rule decision (T027, which changed nothing) and the [ac11]
registration (T029), committed after the pilot by design. [ac11] must be committed before ac11's run and unchanged
since."""
import json
import os
import subprocess
import tomllib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
PANEL = "tests/acceptance/panel_003b.toml"
BYTE_HOST = "MC06268844"

#: spec 003b's merge (PR #8, 9326f87): its results are pinned to 003b's own last code, not to HEAD.
CLOSED_AT = "9326f87"
STRICT = ["src/sraging", "tools/acceptance_003b.py", "tools/acceptance_003.py", "tools/acceptance_003_port.py",
          "tools/oracle_jax2_101.py", "tools/oracle_jax2.py", "tools/oracle_v033.py", "tools/jax2_cases_101.py",
          "tools/jax2_cases.py", "tools/parity_checks.py", "tools/port_jax2_101.py", PANEL]

#: The one change to a STRICT file after 003b's results ran (Naveh, 2026-10-21, "Allow D2's snapshots, re-run in a
#: window"): spec 003's untouched rule (tools/acceptance_003.py `NEW_ORACLE`, read only by its `unchanged` phase)
#: permits 003b's two oracle snapshots. No 003b phase runs that code. A result is exempt from exactly these lines.
REVIEWED = {"tools/acceptance_003.py": {
    '-NEW_ORACLE = "tests/oracle/jax2_3bc5775/"          # the one addition FR-030 permits (research R2)',
    "+#: the additions FR-030 permits under tests/oracle: spec 003's jax2 1.0.0 snapshot (research R2), and spec 003b's two",
    "+#: (Naveh D2, 2026-10-19: the jax2 1.0.1 snapshot and the srbench tree it runs against; spec 003b FR-002)",
    '+NEW_ORACLE = ("tests/oracle/jax2_3bc5775/", "tests/oracle/jax2_6884fc8/", "tests/oracle/tester_f94ec2b/")',
    '-    main to HEAD); only additions under the jax2 oracle snapshot are allowed."""',
    '+    main to HEAD); only additions under the permitted oracle snapshots (NEW_ORACLE) are allowed."""'}}

EXPECTED = {"verify": "pass", "census": "pass", "identity_off": "pass", "parity": "pass", "bytes": "PASS",
            "v100_unchanged": "pass", "lev_measure": "written", "itp": "PASS", "ac11": "PASS",
            "readme_example": "REPORTED", "unchanged": "PASS"}
BYTE_RESULTS = ("identity_off", "parity", "v100_unchanged", "bytes")
PILOT = "lev_measure"
PROVENANCE = ("calibrator", "version", "lock_sha256", "host", "machine", "cpu", "panel_commit")


def load(name):
    return json.load(open(os.path.join(RESULTS, f"{name}_003b.json")))


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


def changed_since(head, until=CLOSED_AT) -> list:
    """The STRICT paths changed between `head` (where a result ran) and `until` (003b's merge), less a file whose every
    changed line is a REVIEWED one."""
    r = git("diff", "--name-only", head, until, "--", *STRICT)
    assert r.returncode == 0, r.stderr
    out = []
    for path in r.stdout.split():
        lines = [ln for ln in git("diff", "-U0", head, until, "--", path).stdout.splitlines()
                 if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
        if not exempt(path, lines):
            out.append(path)
    return out


def exempt(path, changed_lines) -> bool:
    """A STRICT file's change is exempt when every changed line (+/-) is one of its REVIEWED lines."""
    return path in REVIEWED and bool(changed_lines) and set(changed_lines) <= REVIEWED[path]


def panel_at(rev):
    r = git("show", f"{rev}:{PANEL}")
    assert r.returncode == 0, r.stderr
    return tomllib.loads(r.stdout)


def test_every_phase_has_a_committed_result():
    have = sorted(f[:-len("_003b.json")] for f in os.listdir(RESULTS) if f.endswith("_003b.json"))
    assert have == sorted(EXPECTED)
    for name in EXPECTED:
        assert git("ls-files", "--error-unmatch", f"tests/acceptance/results/{name}_003b.json").returncode == 0, name


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_phase_has_its_outcome_from_a_clean_run(name):
    d = load(name)
    assert d["status"] == EXPECTED[name]
    assert d["status_for_acceptance"] == "acceptance"
    assert not d["tree_dirty"] and not d["tree_changed_during_run"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_result_carries_fr014_provenance(name):
    p = load(name)["provenance"]
    for k in PROVENANCE:
        assert p.get(k), (name, k)
    assert p["calibrator"] == "profile_jax2"
    assert p["version"] == ("1.0.0" if name == "v100_unchanged" else "1.0.1")
    if name in ("census", "lev_measure", "ac11"):
        assert p["library"] == "v6", name


SUPPLEMENT = os.path.join(REPO, "specs", "003b-jax2-101-port", "review", "evidence", "provenance_003b.json")


def test_the_provenance_supplement_covers_the_multi_library_and_two_port_phases():
    """sr-naveh finding 2 (Naveh, 2026-10-21: a supplement, no re-runs): parity and bytes ran on libraries v5 and v6
    and identity-off on both ports; their one-line provenance does not say so, the supplement does, row by row."""
    sup = json.load(open(SUPPLEMENT))
    pins = panel_at(CLOSED_AT)["pins"]
    rows = {(r["cohort_id"], r["name"], r["library"]) for r in load("parity")["rows"]}
    assert {(r["cohort_id"], r["calibrator"], r["library"]) for r in sup["parity_rows"]} == rows
    assert all(r["manifest_sha256"] == pins[f"library_{r['library']}_manifest"] for r in sup["parity_rows"])
    assert [r["cohort_id"] for r in sup["bytes_items"]] == [x["cid"] for x in load("bytes")["bytes_item"]]
    assert {r["library"] for r in sup["bytes_items"]} == {"v5"}
    locks = sup["identity_off"]["ports"]
    assert locks["1.0.1"] == load("identity_off")["provenance"]["lock_sha256"]
    assert locks["1.0.0"] and locks["1.0.0"] != locks["1.0.1"]
    assert sup["identity_off"]["calibrators"] == ["profile_jax2", "profile_jax2_h"]
    assert sup["checks"]["v5_now_equals_pin"] and sup["checks"]["v6_now_equals_pin"]


@pytest.mark.parametrize("name", BYTE_RESULTS)
def test_the_byte_results_ran_on_the_byte_host(name):
    d = load(name)
    host = d["host"]["node"] if isinstance(d["host"], dict) else d["host"]   # identity-off and parity record the node
    assert host == BYTE_HOST and d["provenance"]["host"] == BYTE_HOST


@pytest.mark.parametrize("name", sorted(set(EXPECTED) - {PILOT}))
def test_each_result_ran_at_the_code_now_committed(name):
    head = load(name)["repo_head"]
    changed = changed_since(head)
    assert not changed, f"{name} ran at {head[:7]}; changed before 003b's merge: {changed}"


def test_the_pilot_ran_at_the_code_now_committed_but_for_the_ac11_registration():
    head = load(PILOT)["repo_head"]
    changed = changed_since(head)
    assert set(changed) <= {PANEL}, changed
    then, now = panel_at(head), panel_at(CLOSED_AT)
    assert {k: v for k, v in now.items() if k != "ac11"} == {k: v for k, v in then.items() if k != "ac11"}
    assert not then.get("ac11", {}).get("held_out")          # the pilot ran before the registration


def test_ac11_was_registered_before_it_ran_and_is_unchanged_since():
    head = load("ac11")["repo_head"]
    assert panel_at(head)["ac11"] == panel_at(CLOSED_AT)["ac11"]
    reg = load("ac11")["registration_commit"]
    assert git("merge-base", "--is-ancestor", reg, head).returncode == 0
    assert git("merge-base", "--is-ancestor", load(PILOT)["repo_head"], reg).returncode == 0


def test_the_reviewed_exemption_is_exactly_one_change():
    """The twin: the reviewed lines are exempt. Planted: one more changed line, or the same lines in another STRICT
    file, is not."""
    lines = sorted(REVIEWED["tools/acceptance_003.py"])
    assert exempt("tools/acceptance_003.py", lines)
    assert not exempt("tools/acceptance_003.py", lines + ["+BASE = 'some-other-branch'"])
    assert not exempt("tools/acceptance_003b.py", lines) and not exempt("tools/acceptance_003.py", [])


def test_the_gates():
    c = load("census")
    assert c["equal"] == c["rows"] == 2798 and c["in_scope"]["sraging"] == c["in_scope"]["jax2"] == {"J": 290, "H": 52}
    i = load("identity_off")
    assert i["lines"] == i["equal"] == 11 and not i["differences"]
    p = load("parity")
    assert p["pairs"] == p["done"] == 7 and not p["differences"] and p["full_pair_in_scope_equal"]
    assert all(r["equal"] for r in p["rows"] if r.get("equal") is not None)
    assert all(r["prepared_matches_record"] for r in p["rows"] if "prepared_matches_record" in r)
    v = load("v100_unchanged")
    assert all(v["theta_hat_equal"].values()) and v["loglik_equal"] and v["claims_equal_frozen_adapter"]
    assert v["leverage"]["equal"] and v["leverage"]["added_keys"] == ["g0"]
    t = load("itp")
    assert t["AC1"] and t["AC2"] and t["refit_equal_to_round_off"]
    a = load("ac11")
    assert a["nulls"]["held_out"]["pass"] and a["nulls"]["small_n"]["pass"] and len(a["rows"]) == 113
    b = load("bytes")
    assert all(x["bytes_equal"] and x["v033_terms"] == [-700.0] for x in b["bytes_item"])


def test_the_stored_jax2_example_is_the_readme_example_run():
    r = load("readme_example")
    ex = json.load(open(os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")))
    assert r["version"] == "1.0.1" == ex["provenance"]["calibrator"]["version"]
    assert r["repo_head"].startswith(ex["provenance"]["package"]["git_sha"][:12])
    assert r["loglik"]["new"] == ex["loglik"]
