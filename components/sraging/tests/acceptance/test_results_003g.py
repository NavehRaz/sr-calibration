"""Spec 003g's committed results (tests/acceptance/results/*_003g.json; contracts/acceptance.md; tasks T058): every phase
has its result from a clean run, every scored row carries research R14's provenance checked against the pins, the panel
and rules were frozen before any v6 record was pinned, v0.3.3's v6 records were read only after the seal, and no result
is older than a change to its code.

Staleness (by commit since T074, the sr-naveh interim review: matching line text exempted blank lines and brackets
anywhere). A result is stale when a commit between the one it ran at and 003g's merge (CLOSED_AT, as 003b's and 003f's
tests do; spec 004 T001, research R23: later specs change the code legitimately, and HEAD is not 003g's code) touched
STRICT and is not REVIEWED for it. Each REVIEWED commit names the STRICT files it may touch, a scope ("all" results, or only the two Mac "fits") and
its reason. The zero-fit phases are re-run at the final head (tasks T057-T058). The two Mac fits are too costly to
re-run: they are judged on src/sraging and on their own phase code (the functions they call in
tools/acceptance_003g.py), unchanged. The rule code is frozen at pin-w1, except for named changes (RULE_CHANGES)."""
import ast
import json
import os
import subprocess
import tomllib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
PANEL = "tests/acceptance/panel_003g.toml"
PINS = "tests/acceptance/pins_003g.json"
ACC = "tools/acceptance_003g.py"

#: spec 003g's merge (PR #9, 4e17529): its results are judged up to 003g's own last code, not HEAD.
CLOSED_AT = "4e17529"
STRICT =["src/sraging", ACC, "tools/panel_003g.py", "tools/v033_points.py", "tools/v033_mle.py",
          "tools/witness_003g.py", "tools/window_003g.py", "tests/acceptance/panel_003g_ids.tsv"]

#: commits checked not to change what an earlier result measured: sha -> (the STRICT files it may touch, scope, reason).
#: Scope "all": no result's numbers depend on the change; "fits": only the two Mac fits are exempt (they never call it).
REVIEWED = {
    "40a56cf": (("src/sraging/default_gate.py",), "all",
                "default_gate.null_verdict reads the information outcome (PR-4, Naveh 2026-10-22); committed late, "
                "before pin-w1, used by score phases only, which ran after it and re-run at the close"),
    "85072be": (("src/sraging/engine/inputs.py", "src/sraging/engine/stepper_gen.py"), "all",
                "the production-clock wording (comments and docstrings in src/sraging/engine)"),
    "6335950": (("src/sraging/warnings.py",), "all",
                "the _h note's v6 numbers (warnings.HAZARD_SHARP_LIMIT text and data; T046); no verdict reads it"),
    "f90bd21": (("src/sraging/default_gate.py",), "fits",
                "PR-2 with no measured axis reads 'reported' (DV-2, Naveh 2026-10-28; T064); no fit calls default_gate; "
                "the predecessor phase, which does, re-ran after it"),
    "c0e26c2": (("src/sraging/leverage.py", "src/sraging/leverage_rule.py", "src/sraging/warnings.py", ACC), "all",
                "not_assessed per individual (DV-3, Naveh 2026-10-28; T065): the labels of all 10 measured fits "
                "recomputed and unchanged (lev-pilot re-ran at this commit; fit 1 |g|/n 1.9e-6, fit 2 2.8e-2); it adds "
                "two recorded fields (gradient per individual, scipy's status) the fits' stored blocks lack; the "
                "phase-code change is phase_lev_pilot's only"),
    "3e94b42": (("src/sraging/warnings.py",), "all",
                "the _h note's wording (T066: 36 valid, the reference grid on 34); no verdict reads it"),
    "ad8a46c": (("src/sraging/registry.py", "src/sraging/__init__.py", "src/sraging/cli.py"), "all",
                "the switch (T052): the default name only; every 003g phase names its calibrator (fit 1 and fit 2 "
                "profile_jax2 1.0.1, the adapter its own), so no result depends on the default"),
    "079a83a": (("src/sraging/default_gate.py",), "fits",
                "the close (Naveh 2026-10-29): PR-1's n_witnessed (N10), PR-3's no-claim reason (N10), DV-5 (N13); no "
                "fit calls default_gate; the score, predecessor and gate phases re-ran after it"),
    "cfc43ca": (("src/sraging/warnings.py", "src/sraging/calibrators/profile_jax2.py", "src/sraging/cli.py"), "fits",
                "N2 (Naveh 2026-10-29): a 1.0.1 fit in DT_RULE's regime carries SHARP_COHORT_LIMIT, a note (text), and "
                "the CLI prints it on the DT_RULE line; no claim, theta-hat or log-likelihood of a fit changes, so the "
                "Mac fits' numbers stand (fit 1, in the regime, would now also print the note)"),
}

#: the rule code (PR-1..PR-5 and SC-007's comparator) frozen at pin-w1; a change after it is named here
RULE_FILES = {"src/sraging/default_gate.py": None, "tools/window_003g.py": ("coverage_counts", "info_widths",
                                                                            "claim_compare")}
RULE_CHANGES = {
    "eca3766": "claim_compare compares the parameters both sides claim and lists the rest (fit 2, 2026-10-27)",
    "f90bd21": "PR-2 with no measured axis reads 'reported' (DV-2, Naveh 2026-10-28; T064)",
    "d0c9f56": "claim_compare takes a dict of claims (B1) and fails a raw parameter claimed on one side (T067)",
    "079a83a": "PR-1 n_witnessed under upper95, PR-3's no-claim reason (N10), and DV-5: a PR-2 PASS where 'loses' "
               "cannot occur reads 'reported' (N13; Naveh 2026-10-29)",
}

#: phase-code changes reviewed for the two Mac fits: function -> the commit that changed it, with its reason
REVIEWED_FUNCTIONS = {
    "v6_cohort": ("02bf714", "an optional Library argument; with none it loads 003b's snapshot Library on v6 and calls "
                             "its observation(), which is what OJ1.observation(cid, 'v6') did for fit 1"),
    "row_provenance": ("630aff2", "N10 (Naveh 2026-10-29): host/cpu renamed scored_on_host/scored_on_cpu, and "
                                  "record_host/record_cpu added (None for a fit made here); a fit's numbers unchanged"),
}

EXPECTED = {"panel": "pass", "v033_points_verify": "pass", "v033_mle_verify": "pass", "lev_pilot": "PASS",
            "p15_v033": "written", "fit1": "written", "sut": "pass", "score_w1": "written", "fit2_pick": "picked",
            "fit2": "written", "fit2_compare": "reported", "predecessor": "written", "score_w2": "reported",
            "pr3_power": "written", "score_w3": "written", "gate": ("passed", "blocked; decided (11 of 11)"),
            "readme": "pass", "lev_small_n": "written",
            "unchanged": "PASS"}
FITS = {"fit1": ("phase_fit1", "v6_cohort", "_fit_summary", "row_provenance"),
        "fit2": ("phase_fit2", "j_view", "_fit_summary", "row_provenance")}


def load(name):
    return json.load(open(os.path.join(RESULTS, f"{name}_003g.json")))


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


def touched(sha, paths=STRICT) -> set:
    """The files under `paths` that commit `sha` changed."""
    r = git("show", "--name-only", "--format=", sha, "--", *paths)
    assert r.returncode == 0, r.stderr
    return set(r.stdout.split())


def changed_since(head, paths=STRICT, *, fits=False, until=CLOSED_AT) -> list:
    """The commits between `head` (where a result ran) and `until` (003g's merge) that touched `paths` and are not
    REVIEWED for this result (scope "all"; or "fits" too, when judging a Mac fit)."""
    r = git("rev-list", f"{head}..{until}", "--", *paths)
    assert r.returncode == 0, r.stderr
    scopes = ("all", "fits") if fits else ("all",)
    return [s[:7] for s in r.stdout.split() if not (s[:7] in REVIEWED and REVIEWED[s[:7]][1] in scopes)]


def function_source(rev, path, names) -> dict:
    src = git("show", f"{rev}:{path}").stdout
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in names}


def toml_at(rev, path=PANEL):
    r = git("show", f"{rev}:{path}")
    assert r.returncode == 0, r.stderr
    return tomllib.loads(r.stdout)


# ------------------------------------------------------------------------------------------------- presence, status
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_phase_has_a_committed_result_from_a_clean_run(name):
    assert git("ls-files", "--error-unmatch", f"tests/acceptance/results/{name}_003g.json").returncode == 0, name
    d = load(name)
    want = EXPECTED[name]
    assert d["status"] in (want if isinstance(want, tuple) else (want,)), d["status"]
    assert d["status_for_acceptance"] == "acceptance"
    assert not d["tree_dirty"] and not d["tree_changed_during_run"]


# ------------------------------------------------------------------------------------------------------- provenance
@pytest.mark.parametrize("name", ["score_w1", "score_w3"])
def test_every_scored_row_carries_its_provenance_and_it_is_the_pinned_one(name):
    pins = json.load(open(os.path.join(REPO, PINS)))
    with open(os.path.join(REPO, PANEL), "rb") as f:
        windows = tomllib.load(f)["windows"]
    #: the cohorts jax2 named as stopped without a record (W3: 629 of 632); their rows carry no record or witness
    not_stored = {c for w in windows.values() if isinstance(w, dict) for c in w.get("not_stored", ())}
    d = load(name)
    rows = d["provenance_rows"]
    assert rows
    for cid, p in rows.items():
        for k in ("calibrator", "version", "lock_sha256", "library", "manifest_sha256", "run_manifest",
                  "run_manifest_sha256", "scored_on_host", "scored_on_cpu"):
            assert p.get(k), (cid, k)
        assert p["manifest_sha256"] == pins["libraries"]["v6"]["sha256"]
        assert p["run_manifest_sha256"] == pins["runs"][p["run_manifest"]]["sha256"]
        if cid in not_stored:
            assert p["record_sha256"] is None and p["witness_sha256"] is None, cid
            assert cid not in pins["records"][p["run_manifest"]], cid
            continue
        assert p.get("record_sha256") and p.get("record_host") and p.get("record_cpu"), (cid, "record")
        assert p["record_sha256"] == pins["records"][p["run_manifest"]][cid]["sha256"]
        assert p["witness_sha256"] in (None, (pins["witness"].get("mle_profile_jax2") or {}).get(cid))


# ---------------------------------------------------------------------------------------------------------- order
def _first_commit_with(path, text):
    r = git("log", "--reverse", "--format=%H", "-S", text, "--", path)
    return r.stdout.split()[0]


def test_the_panel_and_rules_were_frozen_before_w1_was_pinned():
    pin = _first_commit_with(PINS, '"jax2-v6-full-bign"')
    before, now = toml_at(f"{pin}^"), toml_at("HEAD")
    assert before["rules"] == now["rules"] and before["strata"] == now["strata"]
    assert git("show", f"{pin}^:tests/acceptance/panel_003g_ids.tsv").stdout == \
        open(os.path.join(REPO, "tests", "acceptance", "panel_003g_ids.tsv")).read()


def test_v033s_v6_records_were_read_only_after_the_seal():
    d = load("predecessor")
    assert toml_at(d["repo_head"])["seal"]["jax033_v6"].startswith("GATE-V6 registered run")
    assert d["seal"].startswith("GATE-V6 registered run")
    seal = _first_commit_with(PANEL, "GATE-V6 registered run 2026-10-27")
    pinned = _first_commit_with(PINS, '"jax033-v6-full-bign"')
    assert git("merge-base", "--is-ancestor", seal, pinned).returncode == 0


# ------------------------------------------------------------------------------------------------------ staleness
@pytest.mark.parametrize("name", sorted(set(EXPECTED) - set(FITS)))
def test_each_zero_fit_result_ran_at_the_code_now_committed(name):
    head = load(name)["repo_head"]
    changed = changed_since(head, STRICT)
    assert not changed, f"{name} ran at {head[:7]}; changed before 003g's merge: {changed}"


@pytest.mark.parametrize("name", sorted(FITS))
def test_each_mac_fit_ran_at_the_src_and_phase_code_now_committed(name):
    head = load(name)["repo_head"]
    assert not changed_since(head, ["src/sraging"], fits=True), f"{name} ran at {head[:7]}"
    then, now = function_source(head, ACC, FITS[name]), function_source("HEAD", ACC, FITS[name])
    assert set(then) == set(FITS[name]) or name == "fit1"     # fit1 ran before j_view existed; it does not call it
    for f in then:
        if then[f] == now.get(f):
            continue
        sha, _why = REVIEWED_FUNCTIONS.get(f, (None, None))
        assert sha is not None, f"{name}: {f} changed since {head[:7]}"
        assert function_source(f"{sha}^", ACC, [f])[f] == then[f] and function_source(sha, ACC, [f])[f] == now[f], \
            f"{name}: {f} changed outside its reviewed commit {sha}"


@pytest.mark.parametrize("sha", sorted(REVIEWED))
def test_each_reviewed_commit_touches_only_its_declared_files(sha):
    files, scope, _why = REVIEWED[sha]
    assert scope in ("all", "fits")
    assert touched(sha) <= set(files), f"{sha} touches {sorted(touched(sha) - set(files))}, not declared"


def test_a_commit_outside_the_reviewed_set_is_not_exempt_and_a_reviewed_one_is():
    """Planted and twin, on this branch's own history: d0c9f56 (score-w3's code) is not reviewed, 3e94b42 (the _h note)
    is; f90bd21 (PR-2) exempts a Mac fit and not a zero-fit result; an undeclared file is caught."""
    assert changed_since("d0c9f56^", ["tools/window_003g.py"]) and "d0c9f56" in changed_since("d0c9f56^", STRICT)
    assert "3e94b42" not in changed_since("3e94b42^", STRICT)
    assert "f90bd21" in changed_since("f90bd21^", STRICT) and "f90bd21" not in changed_since("f90bd21^", STRICT,
                                                                                               fits=True)
    assert touched("c0e26c2") - {"src/sraging/leverage.py"}                     # declaring one file would fail


def test_closed_at_is_the_merge_and_nothing_changed_before_it():
    """Spec 004 T001 (research R23), as 003f's test does: CLOSED_AT is PR #9's merge, on this branch's history. Planted:
    a result run before 003g's own d0c9f56 (score-w3's code) is stale at the merge. Twin: at the merge nothing counts,
    whatever later specs change under src/sraging."""
    parents, subject = git("log", "-1", "--format=%P%n%s", CLOSED_AT).stdout.strip().split("\n")
    assert len(parents.split()) == 2 and "pull request #9" in subject, subject
    assert git("merge-base", "--is-ancestor", CLOSED_AT, "HEAD").returncode == 0
    assert "d0c9f56" in changed_since("d0c9f56^")
    assert changed_since(CLOSED_AT) == []


# ------------------------------------------------------------------------------------------- the frozen rule code
def _pin(window_run):
    r = git("log", "--reverse", "--format=%H", "-S", f'"{window_run}"', "--", PINS).stdout.split()
    return r[0] if r else None


def test_the_rule_code_is_frozen_at_pin_w1_except_named_changes():
    """The interim review: the order test froze the toml's rules, not the code computing them, and the zero-fit phases
    re-run at the close, so staleness cannot see a rule-code edit made after W1 was read."""
    pin = _pin("jax2-v6-full-bign")
    for path, names in RULE_FILES.items():
        for sha in git("rev-list", f"{pin}..HEAD", "--", path).stdout.split():
            if names is not None and function_source(f"{sha}^", path, names) == function_source(sha, path, names):
                continue                                         # the file changed, not its rule functions
            assert sha[:7] in RULE_CHANGES, f"{path} rule code changed after pin-w1 in {sha[:7]}, not named"


def test_the_rule_freeze_sees_a_function_change_and_not_a_neighbour():
    """Twin: d0c9f56 changed claim_compare; 686d41f added functions beside the frozen three and changed none of them."""
    names = RULE_FILES["tools/window_003g.py"]
    assert function_source("d0c9f56^", "tools/window_003g.py", names) != function_source("d0c9f56", "tools/window_003g.py",
                                                                                         names)
    assert function_source("686d41f^", "tools/window_003g.py", names) == function_source("686d41f", "tools/window_003g.py",
                                                                                         names)


def test_the_pr3_power_table_was_committed_before_w3_was_pinned():
    """T069: the table is built from W1 and the panel's N only, and committed before pin-w3; a re-run at the close
    must give the same rows."""
    first = git("log", "--reverse", "--format=%H", "--", "tests/acceptance/results/pr3_power_003g.json").stdout.split()
    assert first, "pr3_power_003g.json is not committed"
    pin3 = _pin("jax2-v6-sharp-nladder")
    if pin3:
        assert git("merge-base", "--is-ancestor", first[0], pin3).returncode == 0
    then = json.loads(git("show", f"{first[0]}:tests/acceptance/results/pr3_power_003g.json").stdout)
    assert then["rows"] == load("pr3_power")["rows"]



# --------------------------------------------------------------------------- the close (Naveh 2026-10-29)
def undecided(gate: dict) -> list:
    """N9: the gate's blocking entries that carry no decision from the panel's [[decisions]]."""
    return [(b["stratum"], b["rule"], b.get("arm"), b.get("quantity")) for b in gate.get("blocking", [])
            if not b.get("decision")]


def test_every_blocking_entry_carries_navehs_decision_before_the_default_is_switched():
    """SC-003 (N9): the switch stands on a blocked gate only if every blocking entry is decided. Planted: one entry
    with its decision removed is caught."""
    from sraging import registry
    g = load("gate")
    if registry.DEFAULT == "profile_jax2" and g["blocking"]:
        assert not undecided(g), undecided(g)
        assert g["decisions"] == {"decided": len(g["blocking"]), "of": len(g["blocking"])}
    planted = json.loads(json.dumps(g))
    if planted["blocking"]:
        planted["blocking"][0]["decision"] = None
        assert len(undecided(planted)) == 1


def test_the_sharp_cohort_note_says_what_score_w3_measured():
    """N2: every number SHARP_COHORT_LIMIT prints is the committed score-w3 result's."""
    from sraging.warnings import SHARP_COHORT_LIMIT
    d = SHARP_COHORT_LIMIT["data"]
    w3 = load("score_w3")
    v = {(x["stratum"], x["arm"], x["rule"], x.get("quantity")): x for x in w3["verdicts"]}
    for arm in ("pure", "censor"):
        for q in ("eta", "epsilon"):
            st = v[("SRK-1/outside", arm, "PR-3", q)]["statistic"]
            assert d[f"{q}_closed_covered"][arm] == [st["k_out"], st["n_out"]], (arm, q)
        shapes = v[("SRK-1/outside", arm, "PR-4", None)]["statistic"]["shapes_j"]
        assert d["eta_open_above"][arm] == shapes["eta"]["open_above"]
        assert d["beta_closed"][arm] == shapes["beta"]["closed"]
        assert d[f"n_{arm}"] == shapes["eta"]["n"]
        lim = w3["limits"][f"SRK-1/outside/{arm}"]
        for q in ("eta", "epsilon"):
            assert abs(d["median_closed_dex"][q][arm] - lim["closed_widths"][q]["median_dex"]) < 0.005, (arm, q)
    assert d["n_cohorts"] == d["n_pure"] + d["n_censor"]
    one = [w3["limits"][f"SRK-1/outside/{a}"]["one_sided"]["eta"] for a in ("pure", "censor")]
    assert d["eta_lower_bounds_covered"] == [sum(o["hits"] for o in one), sum(o["claimed"] for o in one)]
    assert d["eta_lower_bounds_covered"][1] == d["eta_open_above"]["pure"] + d["eta_open_above"]["censor"]
