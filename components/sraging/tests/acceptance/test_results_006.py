"""Spec 006's committed results (tests/acceptance/results/*_006.json; contracts/acceptance.md; tasks T001, T068).

- **Bounded at 006's merge (FR-001 of spec 006b; applied on release 0.1.0's branch, Naveh 2026-11-16):** staleness and
  the closure edits read the repository at CLOSED_AT (894eb0c, PR #12), not HEAD, as test_results_005.py reads
  d604a97: later commits (release 0.1.0's path fix, spec 006b) must not stale, or fail, what 006 measured.
- **The closure edits are listed (C9, T001):** spec 006 changes spec 001's claims closure by design (W, H's windows,
  the adapter, the κ ends, v033's leverage check). Every closure file that differs from 005's merge is a key of
  CLOSURE_EDITS_006 with the task that edits it; any other closure change fails. The claims re-run (T076) then
  stands for them.
- **The records are pinned (R11, T001):** every stored file 006 reads -- the v6 manifest, J's, H's and the null's
  run manifests and their window-arm records, and the window-arm cohort arrays -- matches tests/acceptance/
  pins_003g.json (the arrays through the pinned manifest's own checksums), with no new pins. Without the tester's
  data the check is not_applicable, never a pass.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import subprocess
import tomllib

import numpy as np
import pytest

import acceptance_003 as A3
import acceptance_006 as ACC
from conftest import not_applicable
from window_003g import file_sha, store_path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLOSED_AT_005 = "d604a97"                          # 005's merge (PR #11): 006's base
CLOSED_AT = "894eb0c"                              # 006's merge (PR #12): staleness and the closure edits read here
PINS = os.path.join(REPO, "tests", "acceptance", "pins_003g.json")
SNAPSHOT_BUILD = os.path.join(REPO, "tests", "oracle", "tester_f94ec2b", "srbench", "library", "build.py")


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


# ======================================================================================= the closure edits (T001)
#: spec 001's claims-closure files spec 006 may change, each with its task (tasks.md); nothing else may change
CLOSURE_EDITS_006 = {
    "src/sraging/calibrators/profile_jax2.py": "T008, T016, T017, T030, T043, T067, T072",
    "src/sraging/calibrators/profile_v033.py": "T040, T043, T072",
    "src/sraging/compat/tester_f94ec2b/__init__.py": "T006 (new: the verbatim apply_window port)",
    "src/sraging/compat/tester_f94ec2b/observation.py": "T006 (new: the verbatim apply_window port)",
    "src/sraging/v033_mle.py": "T038 (new: moved from tools/v033_mle.py)",
    "src/sraging/leverage.py": "T010, T040",
    "src/sraging/floors.py": "T010, T017",
    "src/sraging/result.py": "T011",
    "src/sraging/warnings.py": "T011, T067, T072",
    "src/sraging/adapters/srbench.py": "T024",
    "src/sraging/adapters/observation.py": "T024",
    "src/sraging/engine/observation.py": "T034",
    "src/sraging/engine/evaluator.py": "T034",
}


def closure_edits(base=CLOSED_AT_005, rev=CLOSED_AT) -> list:
    """The files of spec 001's claims closure (walked on the working tree) that differ between `base` and `rev`."""
    return git("diff", "--name-only", base, rev, "--", *A3.import_closure()).stdout.split()


def test_the_bound_is_006s_merge():
    """CLOSED_AT is PR #12's merge commit, on this branch's history."""
    assert git("log", "-1", "--format=%s", CLOSED_AT).stdout.startswith("Merge pull request #12 ")
    assert git("merge-base", "--is-ancestor", CLOSED_AT, "HEAD").returncode == 0


def unlisted(changed) -> list:
    return [f for f in changed if f not in CLOSURE_EDITS_006]


def test_closure_edits_are_listed():
    """C9: every closure change since 005's merge is on CLOSURE_EDITS_006. Planted twin: an unlisted closure file
    (cohort.py, which 006 does not edit) is named."""
    assert unlisted(closure_edits()) == [], unlisted(closure_edits())
    assert "src/sraging/cohort.py" in A3.import_closure()
    assert unlisted(["src/sraging/cohort.py", "src/sraging/result.py"]) == ["src/sraging/cohort.py"]


def test_the_listed_engine_files_are_the_engine_files_edited():
    """FR-017: every engine file on the list is named in spec 003's SPEC_ENGINE_CHANGES["006"] once it changes
    (T036), so spec 003's unchanged records it as spec-exempt and owes the claims re-run, never a silent pass."""
    edited = [f for f in closure_edits() if f.startswith(A3.ENGINE_PREFIX)]
    named = {f for f, _task in A3.SPEC_ENGINE_CHANGES.get("006", ())}
    assert set(edited) <= named, sorted(set(edited) - named)


# ============================================================================================ the records (T001)
#: the pinned runs 006 reads (R11): J, H and the null on v6's full and big-n groups
RUNS = {"J": "jax2-v6-full-bign", "H": "jax2h-v6-full-bign", "null": "null_prior-v6-full-bign"}
WINDOW_GROUPS = ("main", "bign")                   # the gate's window arm (research R14); the sharp 52 not covered


def _tester_root() -> str:
    from sraging.adapters.srbench import DEFAULT_TESTER
    return os.environ.get("SRAGING_TESTER_ROOT", DEFAULT_TESTER)


def cohort_digest():
    """The tester's own `cohort_digest`, taken verbatim from the snapshot (srbench/library/build.py)."""
    tree = ast.parse(open(SNAPSHOT_BUILD).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cohort_digest")
    ns = {"hashlib": __import__("hashlib"), "np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SNAPSHOT_BUILD, "exec"), ns)
    return ns["cohort_digest"]


def window_arm(manifest) -> list:
    return sorted(s["cohort_id"] for s in manifest["specs"] if s["arm"] == "window" and s["group"] in WINDOW_GROUPS)


def records_problems(pins: dict, root: str, digest=None) -> list:
    """Every way the files 006 reads differ from their pins: the v6 manifest and the three run manifests by sha256;
    each window-arm cohort's record in each run by its pinned key and sha256; each window-arm cohort's arrays by the
    pinned manifest's checksum. [] when all match."""
    out = []
    lib = pins["libraries"]["v6"]
    man_path = os.path.join(root, lib["path"])
    if file_sha(man_path) != lib["sha256"]:
        return [f"{lib['path']}: sha256 differs from its pin"]
    manifest = json.load(open(man_path))
    ids = window_arm(manifest)
    store = os.path.join(root, "data", "store")
    for label, run in RUNS.items():
        ent = pins["runs"][run]
        rp = os.path.join(root, ent["path"])
        if file_sha(rp) != ent["sha256"]:
            out.append(f"{ent['path']}: sha256 differs from its pin")
            continue
        keys = json.load(open(rp))["keys"]
        recs = pins["records"][run]
        for cid in ids:
            pin = recs.get(cid)
            if pin is None or keys.get(cid) != pin["key"]:
                out.append(f"{run}/{cid}: no pinned record, or the manifest's key is not the pinned one")
            elif file_sha(store_path(store, pin["key"])) != pin["sha256"]:
                out.append(f"{run}/{cid}: the record's sha256 differs from its pin")
    if digest is not None:
        for cid in ids:
            with np.load(os.path.join(root, lib["path"].rsplit("/", 1)[0], "cohorts", f"{cid}.npz")) as z:
                got = digest(np.asarray(z["ages"], dtype=float), np.asarray(z["events"], dtype=int))
            if got != manifest["checksums"].get(cid):
                out.append(f"{cid}: the cohort's arrays differ from the pinned manifest's checksum")
    return out


@pytest.fixture(scope="module")
def pinned():
    root = _tester_root()
    pins = json.load(open(PINS))
    missing = [p for p in [pins["libraries"]["v6"]["path"], *(pins["runs"][r]["path"] for r in RUNS.values())]
               if not os.path.exists(os.path.join(root, p))]
    if missing:
        not_applicable(f"the tester's data is not here ({root}): {missing}")
    return pins, root


def test_records_match_pins(pinned):
    """R11: the 240 window-arm cohorts' records in J's, H's and the null's pinned runs, and their arrays, are the
    pinned bytes; no new pins."""
    pins, root = pinned
    manifest = json.load(open(os.path.join(root, pins["libraries"]["v6"]["path"])))
    ids = window_arm(manifest)
    assert len(ids) == 240 and len({s["cluster_key"] for s in manifest["specs"] if s["cohort_id"] in ids}) == 54
    assert records_problems(pins, root, cohort_digest()) == []


def test_a_changed_pin_is_named(pinned):
    """Planted: one record's pin altered, the run manifest's pin altered, and a digest that matches nothing, are each
    named (the last on all 240 cohorts)."""
    pins, root = pinned
    cid = window_arm(json.load(open(os.path.join(root, pins["libraries"]["v6"]["path"]))))[0]
    assert len(records_problems(pins, root, lambda ages, events: "planted")) == 240
    bad = json.loads(json.dumps(pins))
    bad["records"][RUNS["H"]][cid]["sha256"] = "0" * 64
    assert records_problems(bad, root) == [f"{RUNS['H']}/{cid}: the record's sha256 differs from its pin"]
    bad["runs"][RUNS["null"]]["sha256"] = "0" * 64
    assert f"{pins['runs'][RUNS['null']]['path']}: sha256 differs from its pin" in records_problems(bad, root)


# ================================================================================ the committed results (T068)
BOUND = CLOSED_AT                                  # 006 is closed: staleness reads its merge, not HEAD
FREEZE = "85b8604"                                 # rules_006.toml frozen = true with naveh_ok (T058)
#: phases that ran before the freeze by design: parity reads no rule (RULE_FREE); the pilot measured r and Dg's price,
#: which the freeze set ([pricing]), and gates nothing (FR-019)
PRE_FREEZE = ("parity", "pilot")
#: each phase's literal status in its committed JSON
EXPECTED = {"parity": "pass", "identity": "pass", "stored": "pass", "pilot": "pass", "a1": "fail", "a2": "fail",
            "realfull": "fail", "v033-leverage": "capped", "unchanged": "pass"}
#: a status other than pass, accepted by Naveh's recorded decision (spec.md Clarifications): phase -> (the rules that
#: failed, the decision). Nothing else may fail.
DECIDED = {
    "a1": (("R-4",), "R-4 failed on A1 (15 of 196, 0.29-1.80 nats, all below 1.92): 'Ship W, state the limit' "
                     "(Naveh 2026-11-12); its consequence, 'stops W', is overridden"),
    "a2": (("truncated_above", "R-4"), "truncated above: eta 28 of 32 closed claims, epsilon 29 of 29, so W warns at "
                                       "n >= 25,000 (A2's rule, Naveh 2026-11-11; W_LARGE_N, T067). R-4 on 3 of 44 "
                                       "(0.54, 0.43, 0.16 nats, below 1.92, R-3 passing; the worst refit reads W's "
                                       "own gap): 'Realfull + A2, same kind' (Naveh 2026-11-15)"),
    "realfull": (("R-4",), "R-4 on 2 of 20 (0.81, 0.17 nats, below 1.92, R-3 passing; a refit reads W's own gap "
                           "0.81): 'Realfull + A2, same kind' (Naveh 2026-11-15) extends the A1 override"),
    "v033-leverage": (("R-9",), "R-9 failed on its null (6 flags among 18 assessed) and the cell stopped at its 8 "
                                "CPU-h cap before the positive control: 'Fail; kappa-ends only' (Naveh 2026-11-11)"),
}
#: development records, each with why it cannot be an acceptance one; their rows are one job head's
DEVELOPMENT = {
    "pilot": "rows at 7398c32, collected and scored at 958a3e7 with --resume --job-head (rows of another commit are "
             "development); pre-freeze by design",
    "v033-leverage": "the null's rows at 966bbe7 (the acceptance run, stopped at its cap), scored at 8d2bbed with "
                     "--dev --resume --job-head after the control's cap check (5341340)",
}
#: the parsed sections of rules_006.toml each phase reads (start() reads `frozen` for all but RULE_FREE)
_CELL_READS = ("frozen", "rules", "panel", "pricing", "budget", "pins")
FREEZE_READS = {"parity": (), "identity": (), "stored": ("frozen", "rules", "pins"),
                **{p: _CELL_READS + (f"caps.{p}",) for p in ("pilot", "a1", "a2", "realfull", "v033-leverage")},
                "unchanged": ("frozen", "caps.unchanged")}
_GATE_ELEMENT = (("tools/gate_006.py", "element_main"),)
#: each phase's entry functions: the driver's, and on WEXAC the array element (tools/gate_006.py element)
ENTRIES = {p: (("tools/acceptance_006.py", ACC.PHASE_FUNCS[p].__name__),)
              + (_GATE_ELEMENT if ACC.WHERE[p] == "wexac" else ()) for p in ACC.PHASES}


def name(phase):
    return f"tests/acceptance/results/{phase.replace('-', '_')}_006.json"


def load(phase) -> dict:
    return json.load(open(os.path.join(REPO, name(phase))))


def ran_at(phase) -> str:
    """The commit whose code computed the result: a WEXAC cell's job head (its rows'), else the run's HEAD."""
    d = load(phase)
    return d.get("head") or d["repo_head"]


def ancestor(a, b) -> bool:
    return git("merge-base", "--is-ancestor", a, b).returncode == 0


def toml_at(rev, path):
    r = git("show", f"{rev}:{path}")
    return tomllib.loads(r.stdout) if r.returncode == 0 else None


def failed_rules(phase, d) -> tuple:
    """The rules a record failed: a cell's false gates; R-9's null or control."""
    if phase == "v033-leverage":
        return ("R-9",) if d["status"] != "pass" else ()
    return tuple(k for k, v in (d.get("gates") or {}).items() if not v)


@pytest.mark.parametrize("phase", ACC.PHASES)
def test_each_phase_has_a_committed_result_from_a_clean_run_after_the_freeze(phase):
    """Every phase's JSON is tracked, from a clean run with its expected status: an acceptance run at or after the
    freeze reading the frozen rules, except PRE_FREEZE and DEVELOPMENT, each with its reason. A status other than pass
    fails only the rules Naveh's decision names."""
    assert git("ls-files", "--error-unmatch", name(phase)).returncode == 0, phase
    d = load(phase)
    assert not d["tree_dirty"] and not d["tree_changed_during_run"], phase
    assert d["status_for_acceptance"] == ("development" if phase in DEVELOPMENT else "acceptance"), phase
    assert d["status"] == EXPECTED[phase], (phase, d["status"])
    if phase not in PRE_FREEZE:
        assert ancestor(FREEZE, ran_at(phase)) and ancestor(FREEZE, d["repo_head"]), phase
        assert toml_at(d["manifests"][ACC.RULES_REL], ACC.RULES_REL)["frozen"] is True, phase
    assert ancestor(d["repo_head"], BOUND) and ancestor(ran_at(phase), d["repo_head"]), phase
    assert failed_rules(phase, d) == (DECIDED[phase][0] if phase in DECIDED else ()), (phase, failed_rules(phase, d))


def test_the_freeze_is_the_commit_that_set_navehs_ok():
    before, after = toml_at(f"{FREEZE}^", ACC.RULES_REL), toml_at(FREEZE, ACC.RULES_REL)
    assert before["frozen"] is False and not before.get("naveh_ok")
    assert after["frozen"] is True and "Freeze and start A1" in after["naveh_ok"]


@pytest.mark.parametrize("phase", ACC.PHASES)
def test_each_result_records_its_cap_and_stayed_under_it(phase):
    """FR-020: the cap is the frozen table's (a1 and a2 set at the freeze); CPU under it and never `capped`, except
    where Naveh's decision accepts the cell's stop. A WEXAC cell counts every wave's CPU (its ledger)."""
    d = load(phase)
    cap = d["cap"]
    assert cap["cap_cpu_h"] == ACC.cap_for(phase) == toml_at(BOUND, ACC.RULES_REL)["caps"][phase]
    assert cap["where"] == ACC.WHERE[phase]
    if d["status"] == "capped":
        assert phase in DECIDED and cap["capped"] is True, (phase, cap)
    else:
        assert cap["cpu_spent_h"] < cap["cap_cpu_h"] and cap["capped"] is False, (phase, cap)


#: the collector's own Mac CPU (scoring, with its ssh and rsync children), counted by the cap on top of the ledger:
#: at most 35 CPU-s measured (A1)
COLLECTOR_CPU_H = 0.05


@pytest.mark.parametrize("phase", [p for p in ACC.PHASES if ACC.WHERE[p] == "wexac"])
def test_each_cells_rows_are_one_job_heads_and_its_ledger_counts_its_cpu(phase):
    """Every committed row of a WEXAC cell was written at the record's head; the record's waves are the ledger's (a
    resumed collection submits none), and the cap counted every wave in the ledger (WEXAC's and any Mac step's) plus
    the collector's own CPU."""
    d = load(phase)
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(ACC.RESULTS_GATE, phase, "*.json"))]
    assert len(rows) == d["n_rows"] and {r["head"] for r in rows} == {d["head"]}, phase
    led = json.load(open(os.path.join(REPO, d["ledger"])))["waves"]
    assert [w["job_id"] for w in d["waves"]] in ([w["job_id"] for w in led if w.get("where", "wexac") == "wexac"],
                                                 []), phase
    assert d["waves"] or phase in DEVELOPMENT, phase
    spent, counted = d["cap"]["cpu_spent_h"], ACC.ledger_spent_h(phase)
    assert counted <= spent <= counted + COLLECTOR_CPU_H, (phase, spent, counted)


# ------------------------------------------------------------------------------------- R-1 to R-4 from the rows
@pytest.fixture(scope="module")
def v6():
    root = _tester_root()
    pins = json.load(open(PINS))
    if not os.path.exists(os.path.join(root, pins["libraries"]["v6"]["path"])):
        not_applicable(f"the tester's data is not here ({root})")
    return ACC.connect_v6()


@pytest.mark.parametrize("phase", ("a1", "a2", "realfull"))
def test_each_cell_is_rescored_from_its_stored_rows(phase, v6):
    """R-1 to R-4, FR-022, the gates, the consequence and the status, recomputed here from the committed rows under the
    rules the cell was scored with, equal the record."""
    d = load(phase)
    mods, pins, lib = v6
    rules = toml_at(d["repo_head"], ACC.RULES_REL)
    ids = ACC.cell_ids(phase, lib, rules)
    rows = ACC.read_rows(os.path.join(ACC.RESULTS_GATE, phase), ids, d["head"])
    assert len(rows) == len(ids) == d["n_cohorts"], phase
    got = json.loads(json.dumps(ACC.score_cell(phase, rows, ids, mods, pins, lib, rules,
                                               ACC.j_prices(mods, pins, rules, ids))))
    for k in ("R-1", "one_sided", "R-2", "R-3", "R-4", "FR-022", "truncation", "gates", "consequence", "status"):
        assert got[k] == d[k], (phase, k)


def test_r9_is_rescored_from_its_stored_rows():
    """R-9's null from the 26 committed rows, with the record's control (not run: cell capped), equals the record."""
    d = load("v033-leverage")
    rules = toml_at(d["repo_head"], ACC.RULES_REL)
    ids = ACC.v033_ids(rules)
    rows = ACC.read_rows(os.path.join(ACC.RESULTS_GATE, "v033-leverage"), ids, d["head"])
    got = json.loads(json.dumps(ACC.score_v033(rows, ids, {}, d["R-9"]["positive_control"], rules["rules"]["R-9"])))
    assert got["R-9"] == d["R-9"] and d["R-9"]["null"]["passes"] is False
    assert d["R-9"]["positive_control"]["not_run"].startswith("cell capped")


# ------------------------------------------------------------------------------------------------------ staleness
# 005's rule (test_results_005.py:117-317, copied), read at BOUND: a result is stale when something it depends on
# changed since the commit that computed it, unless the change is REVIEWED: in tools/, a module-level name reachable
# from its phase's entries; in src/, any file of the import closure of the sraging modules that code uses (both as
# syntax, docstrings and comments removed); the rules sections it reads. `unchanged` runs whole suites, so every
# Python file under src/, tools/ and tests/ counts for it.
#: the pilot measured what the freeze set, before it (PRE_FREEZE, DEVELOPMENT): the gate's code changed after it by
#: design (T057), and it gates nothing; it is held to its record, not to HEAD
STALENESS_EXEMPT = {"pilot": "pre-freeze development: its r and Dg's price are the frozen [pricing]; it gates nothing"}
#: commits checked not to change what an earlier result measured: sha -> (the files it may touch, reason)
REVIEWED = {
    "bde8717": (("src/sraging/calibrators/profile_jax2.py", "src/sraging/warnings.py"),
                "T067 adds W_LARGE_N to W's warnings after the fit, on windowed fits of n >= 25,000 rows given: "
                "theta-hat, claims, log-likelihoods, the leverage block and every field the gate scores are unchanged "
                "(test_profile_jax2_w.py's wiring test: equal with it and without it). identity's cohorts are "
                "unwindowed or n <= 2,000 (test_t067_cannot_reach_identity)"),
    "dc451e0": (("src/sraging/calibrators/profile_jax2.py", "src/sraging/calibrators/profile_v033.py",
                 "src/sraging/warnings.py"),
                "T072 (Naveh 2026-11-15): message text only. W_LARGE_N's wording and its constants (A2's ends by kind, "
                "A1's real-table rate; checked against the records), and a caveat closing profile_v033's flag "
                "message. Same trigger, same fit: theta-hat, claims, log-likelihoods and every field the gate scores "
                "are unchanged; identity's cohorts never reach W_LARGE_N (test_t067_cannot_reach_identity)"),
}
#: tools/ names changed after a result, each checked: "tools/<file>.py:<name>" -> ((shas), reason)
_COLLECTOR = ("the Mac-side collector, changed after A1's and v033's rows (966bbe7): 5341340 put ssh timeouts on every "
              "remote call, read a partly purged array from LSF's history, kept a Mac step's spend out of the spec's "
              "WEXAC bound and put R-9's control under the cell's cap; 622465a routed v033's --dev --resume to the "
              "collection, never the ITP pilot. Neither touches the element (tools/gate_006.py) or score_cell: the rows "
              "are WEXAC's at the job head, R-1 to R-4 are rescored from them here, and the spend is the ledger's")
REVIEWED_FUNCTIONS = {f"tools/acceptance_006.py:{n}": (shas, _COLLECTOR) for n, shas in {
    "SSH_OPTS": ("5341340",), "_remote": ("5341340",), "poll_array": ("5341340",), "lsf_cpu": ("5341340",),
    "running_job": ("5341340",), "ledger_spent_h": ("5341340",), "spec_spent_wexac_h": ("5341340",),
    "R9_CONTROL_PRICE_H": ("5341340",), "r9_not_run": ("5341340",), "r9_control": ("5341340",),
    "phase_cell": ("5341340", "622465a"),
}.items()}
#: rules sections changed after a result, each checked: section -> ((shas), reason)
REVIEWED_SECTIONS = {
    "budget": (("e694cbc", "dc451e0"),
               "the spec's WEXAC bound 300 -> 350 for A2 (Naveh 2026-11-14), after A1 and v033's null: it decides only "
               "whether a later wave may start; A1 (179.01) and v033 (4.93 WEXAC) ended within 300 with every wave run. "
               "The Mac bound 15 -> 21 (Naveh 2026-11-15) for the claims re-run: no phase reads it"),
}
#: `unchanged` runs whole suites by pytest, so it depends on every Python file they import or collect
UNCHANGED_PATHS = ("src", "tools", "tests/unit", "tests/parity", "tests/acceptance", "tests/conftest.py")


def _exists_at(rev, path) -> bool:
    return git("cat-file", "-e", f"{rev}:{path}").returncode == 0


def _module_file_at(rev, mod):
    """spec 003's _module_file, read at `rev`: the repo-relative file of module `mod`, else None."""
    if mod == "sraging" or mod.startswith("sraging."):
        base = "/".join(["src", *mod.split(".")])
        return next((p for p in (base + ".py", base + "/__init__.py") if _exists_at(rev, p)), None)
    p = f"tools/{mod}.py"
    return p if "." not in mod and _exists_at(rev, p) else None


def _walk_at(rev, roots, follow) -> list:
    """spec 003's import walk (import_closure) over the tree at `rev`: every module reachable from `roots` by any
    import statement that `follow` accepts, plus each sraging package's __init__."""
    seen, todo = {}, [r for r in roots if follow(r)]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        f = seen[mod] = _module_file_at(rev, mod)
        if f is None:
            continue
        parts = mod.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts)) if parts[0] == "sraging"]
        pkg = parts if f.endswith("__init__.py") else parts[:-1]
        for node in ast.walk(ast.parse(git("show", f"{rev}:{f}").stdout, f)):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names if follow(a.name)]
            elif isinstance(node, ast.ImportFrom):
                base = (pkg[:len(pkg) - node.level + 1] if node.level else []) + (node.module.split(".")
                                                                                  if node.module else [])
                todo += [m for m in [".".join(base)] + [".".join(base + [a.name]) for a in node.names] if follow(m)]
    return sorted(f for f in seen.values() if f is not None)


def _top(rev, path) -> tuple:
    """`path` at rev: its module-level names (functions, classes, assignment targets) -> node, and its module-level
    import aliases (alias -> ("mod", module) or ("from", module, name))."""
    r = git("show", f"{rev}:{path}")
    tree = ast.parse(r.stdout) if r.returncode == 0 else ast.Module(body=[], type_ignores=[])
    top, alias = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        top.setdefault(n.id, node)
        else:
            alias.update(_aliases([node]))
    return top, alias


def _aliases(nodes) -> dict:
    out = {}
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for a in n.names:
                    out[a.asname or a.name.split(".")[0]] = ("mod", a.name if a.asname else a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                for a in n.names:
                    out[a.asname or a.name] = ("from", n.module, a.name)
    return out


def _defs(rev, path) -> dict:
    """name -> its syntax (docstrings removed) for every module-level name of `path` at rev."""
    return {k: A3.without_docstrings(ast.unparse(v)) for k, v in _top(rev, path)[0].items()}


def _tool(mod):
    return f"tools/{mod}.py" if "." not in mod and os.path.exists(os.path.join(REPO, "tools", f"{mod}.py")) else None


def _sraging(mod):
    return mod == "sraging" or mod.startswith("sraging.")


def reach_code(phase) -> tuple:
    """(the (tools file, module-level name) pairs reachable from the phase's entries at BOUND, the sraging modules they
    use): 004's rule -- the call graph and the module-level names its functions and tables read (a name passed as a
    string included), through the import aliases into other tools modules. A static over-approximation."""
    todo = list(ENTRIES[phase])
    seen, src, info = set(), set(), {}
    while todo:
        path, nm = todo.pop()
        if (path, nm) in seen:
            continue
        seen.add((path, nm))
        top, mod_alias = info.setdefault(path, _top(BOUND, path))
        node = top.get(nm)
        if node is None:
            continue
        alias = {**mod_alias, **_aliases([node])}
        for n in ast.walk(node):
            hit = None
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
                hit = (alias[n.value.id], n.attr)
            elif isinstance(n, ast.Name) and n.id in top:
                todo.append((path, n.id))
            elif isinstance(n, ast.Name) and n.id in alias:
                hit = (alias[n.id], None)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in top:
                todo.append((path, n.value))
            if hit:
                kind, attr = hit
                m, nm2 = (kind[1], attr) if kind[0] == "mod" else (kind[1], kind[2])
                if nm2 is not None and _tool(m):
                    todo.append((_tool(m), nm2))
                elif _sraging(m):
                    src.update({m} | ({f"{m}.{nm2}"} if nm2 else set()))
    return seen, src


def src_closure(mods, rev=BOUND) -> list:
    """The src/ files of every sraging module reachable from `mods` by any import statement, plus each package's
    __init__ (spec 003's import_closure walk), as the tree at `rev` has them."""
    return _walk_at(rev, sorted(mods), _sraging)


def _file_syntax(rev, path):
    r = git("show", f"{rev}:{path}")
    return None if r.returncode != 0 else (A3.without_docstrings(r.stdout) if path.endswith(".py") else r.stdout)


def _commits(base, path) -> list:
    return git("rev-list", "--reverse", f"{base}..{BOUND}", "--", path).stdout.split()


def name_changes(path, nm, base) -> list:
    """The commits in base..BOUND whose own diff (against the first parent) changed module-level `nm` of `path`."""
    return [s[:7] for s in _commits(base, path) if _defs(f"{s}^", path).get(nm) != _defs(s, path).get(nm)]


def file_changes(path, base) -> list:
    """The commits in base..BOUND whose own diff changed `path` as syntax (docstrings and comments are not content)."""
    return [s[:7] for s in _commits(base, path) if _file_syntax(f"{s}^", path) != _file_syntax(s, path)]


def section(doc, dotted):
    for part in dotted.split("."):
        if not isinstance(doc, dict) or part not in doc:
            return None
        doc = doc[part]
    return doc


def section_changes(sec, base) -> list:
    """The commits in base..BOUND whose own diff changed rules section `sec`."""
    return [s[:7] for s in _commits(base, ACC.RULES_REL)
            if section(toml_at(f"{s}^", ACC.RULES_REL), sec) != section(toml_at(s, ACC.RULES_REL), sec)]


def stale_items(phase) -> list:
    """What `phase`'s result depends on and that changed between the commit that computed it and BOUND, unreviewed:
    (file, the reachable name, or a rules section, or None for a whole src/ file, the commits)."""
    base, out = ran_at(phase), []
    names, mods = reach_code(phase)
    for path, nm in sorted(names):
        if _defs(base, path).get(nm) != _defs(BOUND, path).get(nm):
            shas = name_changes(path, nm, base)
            key = f"{path}:{nm}"
            if not (key in REVIEWED_FUNCTIONS and set(shas) <= set(REVIEWED_FUNCTIONS[key][0])):
                out.append((path, nm, shas))
    files = [f for f in src_closure(sorted(mods)) if f.startswith("src/")]
    if phase == "unchanged":
        files += [f for f in git("diff", "--name-only", base, BOUND, "--", *UNCHANGED_PATHS).stdout.split()
                  if f.endswith(".py") and f not in files]
    for f in files:
        if _file_syntax(base, f) != _file_syntax(BOUND, f):
            unreviewed = [x for x in file_changes(f, base) if not (x in REVIEWED and f in REVIEWED[x][0])]
            if unreviewed:
                out.append((f, None, unreviewed))
    a, b = toml_at(base, ACC.RULES_REL), toml_at(BOUND, ACC.RULES_REL)
    for sec in FREEZE_READS[phase]:
        if section(a, sec) != section(b, sec):
            shas = section_changes(sec, base)
            if not (sec in REVIEWED_SECTIONS and set(shas) <= set(REVIEWED_SECTIONS[sec][0])):
                out.append((ACC.RULES_REL, f"[{sec}]", shas))
    return out


def stale_reasons(phase) -> list:
    return [f"{f}{'' if what is None else ' ' + what if what.startswith('[') else ':' + what} changed in {shas}"
            for f, what, shas in stale_items(phase)]


@pytest.mark.parametrize("phase", [p for p in ACC.PHASES if p not in STALENESS_EXEMPT])
def test_each_result_ran_at_the_code_and_freeze_it_reads_now(phase):
    reasons = stale_reasons(phase)
    assert reasons == [], f"{phase} is stale: re-run it, or add a REVIEWED entry with its reason: {reasons}"


def test_t067_cannot_reach_identity():
    """REVIEWED's bde8717: W_LARGE_N needs a window and n >= 25,000; identity's windowed cohorts are A1's (n < 25,000),
    its unwindowed ones carry no window."""
    from sraging.calibrators import profile_jax2 as PJ
    ident = toml_at(BOUND, ACC.RULES_REL)["panel"]["identity"]
    for c in ident["windowed"]:
        assert json.load(open(os.path.join(ACC.RESULTS_GATE, "a1", f"{c}.json")))["n"] < PJ.W_LARGE_N_MIN, c
    assert all(c.endswith(".csv") or c.startswith("pure-") for c in ident["unwindowed"])


def test_staleness_sees_a_code_change_and_a_rules_change_and_not_a_comment():
    """Planted: T057 (d4397b6) rewrote score_cell after the pilot's rows, and a cell reaches it through phase_cell; the
    budget raise (e694cbc, first) changed [budget] after A1's head. Twin: a comment or a docstring is not content."""
    assert ("tools/acceptance_006.py", "score_cell") in reach_code("a1")[0]
    assert "d4397b6" in name_changes("tools/acceptance_006.py", "score_cell", "7398c32")
    assert section_changes("budget", "966bbe7")[0] == "e694cbc" and section_changes("rules", "966bbe7") == []
    assert A3.without_docstrings('def f():\n    """a"""\n    return 1  # b\n') == \
        A3.without_docstrings("def f():\n    return 1\n")
