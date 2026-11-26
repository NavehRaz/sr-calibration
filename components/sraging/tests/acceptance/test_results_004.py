"""Spec 004's committed results (tests/acceptance/results/*_004.json; contracts/acceptance.md §6; tasks T137).

- **Committed results:** every phase has its result, from a clean acceptance run, with its expected status. The
  expected status is the JSON's literal one; where Naveh read a result in the spec's Clarifications, the reading is a
  named entry citing its session (READINGS), never a loosened check. multi's is MULTI_EXPECTED, set after A9's re-run.
- **Ordering, by commit ancestry:** T001 before the first src/ commit; P0, hr2-noise and P5 before freeze-H; P1-P4
  before freeze-L; freeze-H before every §3 phase and freeze-L before every §4 phase; the reference's pin before hr1;
  hr3-pilot before hr3; P3 before T130.
- **Strict staleness:** a result is stale when, between the commit it ran at and 004's merge (CLOSED_AT, spec 005 T001), something it
  depends on changed,
  unless the change is REVIEWED: in tools/, a function or module-level name reachable from its phase's entry (the call
  graph and the names its functions and dispatch tables read, through the imports into other tools modules), compared
  as syntax trees without docstrings; in src/, the library under test, any file of the import closure of the sraging
  modules that code uses, compared alike; and a parsed section of its freeze file that it reads (comments are not
  content). Long phases (hr3, s4, s5-het) are judged on exactly the code they call, as every phase is.
- **The rules:** each HR verdict is recomputed from the stored rows by sampler_rules and must equal the stored one; a
  rule function changed after its freeze is named in RULE_CHANGES.
- **Caps:** a capped result records its projection, cap, CPU spent and naveh_ok; CPU over the cap without `capped`
  needs a named reading. **Never PASS:** capped and incomplete never count as PASS. **Denominators** on every verdict.
- **Not yet run:** docs, unchanged and spec 001's claims re-run (T145) read NOT APPLICABLE until their results exist.

It starts from T055's FR-009 finding (register D-412).
"""
import ast
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tomllib

import pytest

from conftest import not_applicable

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
sys.path.insert(0, os.path.join(REPO, "tools"))

import acceptance_003 as A3  # noqa: E402
import acceptance_004 as ACC  # noqa: E402
from sraging import generator as GEN  # noqa: E402
from sraging import sampler_rules as R  # noqa: E402

PANEL, LANG, ACC_PY = ACC.PANEL_PATH, ACC.LANG_PATH, "tools/acceptance_004.py"
CLOSED_AT_003G = "4e17529"                         # 003g's merge (PR #9): 004's base
CLOSED_AT = "759c003"                              # 004's merge (PR #10): spec 005 T001 bounds staleness here

# ----------------------------------------------------------------------------------------------- expected statuses
#: each phase's literal status in its committed JSON
EXPECTED = {"p5-inventory": "written", "p0-het-cost": "written", "hr2-noise": "written",
            "panel": "pass", "reference": "written", "hr1": "fail", "hr5": "pass", "hr2": "pass",
            "hr3-pilot": "capped", "hr3": "fail", "hr4": "fail", "d401": "written",
            "p1-langevin": "written", "p1-pooled": "capped", "p1-pooled-b": "written", "p2-s5-power": "written",
            "p3-cost": "written", "p4-jax": "written",
            "plants": "pass", "hr8": "pass", "s4": "pass", "paths": "pass", "s5-hom": "fail", "s5-het": "pass",
            "docs": "pass", "unchanged": "PASS"}
#: multi's expected status after amendment A9's re-run (key 60600, composite bar 1e-4; a fail in the re-run stands).
#: The re-run at 878872f (fe6a416) read pass; the first run's fail is evidence (review/evidence/multi_004_first_252ad3a.json).
MULTI_EXPECTED = "pass"
#: Naveh's readings of results whose literal status is not a plain pass: phase -> (literal, reading, Clarifications
#: session). The close counts the reading.
READINGS = {
    "hr1": ("fail", "fail: HR-1 passes (159 valid, 0 false passes); FR-009 fails as written with gen-1's limit above "
                    "n = 20,000 (register D-412), no re-run",
            "Session 2026-10-29 (Naveh, on hr1, hr4 and hr3-pilot's results); Session 2026-10-31 (T056)"),
    "hr3-pilot": ("capped", "capped: the pilot stopped at its 1 CPU-h cap after its sharp cell; hr3 ran from its "
                            "one-cell projection under the 30 CPU-h OK",
                  "Session 2026-10-29 (Naveh, on hr1, hr4 and hr3-pilot's results)"),
    "hr3": ("fail", "fail: recorded and attributed to the maximiser, no re-run; no adaptive-M series for spec 006",
            "Session 2026-10-31 (Naveh, on HR-3 and the heterogeneity half, T056)"),
    "hr4": ("fail", "fail as written on its remainder plant; FR-007's byte identity stands on the 90 rows",
            "Session 2026-10-29 (Naveh, on hr1, hr4 and hr3-pilot's results)"),
    "p1-pooled": ("capped", "capped after 15 of 18 cells; p1-pooled-b met the other 3 under the follow-up OK",
                  "Session 2026-11-01 (Naveh, on P1's steps and S5's configuration); Session 2026-11-02"),
    "s5-hom": ("fail", "pass under spec 002's condition (d): four features pass, heterogeneity's plant not detected "
                       "at R 200 as in 002; the mixture's conditioning check stays Not Covered for spec 006",
               "Session 2026-11-05 (Naveh, T127: s5-hom's reading)"),
}
#: results that spent more CPU than their cap without reading `capped`, each with the OK that let it finish
OVER_CAP_OK = {"s5-hom": "Session 2026-11-04 (Naveh, T121 and T127): \"Let it finish\" (its projection read "
                         "score_002.json alone; the true projection was 4.8 CPU-h)"}

# ------------------------------------------------------------------------------------------------- ordering commits
FREEZE_H = "c9cb9ef"                               # panel_004.toml with naveh_ok_h
FREEZE_L = "252ad3a"                               # langevin_004.toml with naveh_ok_l (T119, T120)
REFERENCE_PIN = "3c935c1"                          # reference_004.json's sha256 in panel_004.toml [reference]
T001 = "853deb0"
T130 = "1445a1e"
A9 = "3bb031d"                                     # amendment A9: multi's re-run at key 60600, composite bar 1e-4
PILOTS_BEFORE_H = ("p0-het-cost", "hr2-noise", "p5-inventory")
PILOTS_BEFORE_L = ("p1-langevin", "p1-pooled", "p1-pooled-b", "p2-s5-power", "p3-cost", "p4-jax")

# ----------------------------------------------------------------------------------------------------- staleness
#: commits checked not to change what an earlier result measured: sha -> (the files it may touch, reason). The src/
#: engine changes are backed by the FP-side phases re-run at 878872f (review/evidence/rerun_head_878872f/).
_RERUN = ("re-run at 878872f (review/evidence/rerun_head_878872f/): hr2-noise, panel, reference, hr1, hr2, hr5 and d401 "
          "give the same rows, hr4 the same outside its wall timings")
REVIEWED = {
    "e46258c": (("src/sraging/engine/density.py", "src/sraging/engine/evaluator.py", "src/sraging/engine/inputs.py",
                 "src/sraging/engine/tensor_reference.py", "src/sraging/langevin/sample.py", "src/sraging/population.py",
                 "src/sraging/sampler_rules.py", "tools/acceptance_004.py", "tools/multi_004.py"),
                "US8 (T103-T109): inputs.py adds Heterogeneity.zero (sd 0, which the constructor still refuses) and "
                "refuse_zero_spread, which raises for that member only; the Evaluator calls the refusal; the FP entry "
                "points refuse a tuple. No single-parameter path changes; " + _RERUN),
    "370e536": (("src/sraging/engine/mixture.py", "tools/acceptance_004.py"),
                "hr3's crash fix: mixture.guide changes only where the old code divided by zero (cv_c = 0, or a zero "
                "first moment) and raised; every point where it returned keeps its dict (a twin test against the old "
                "code). A result that ran before it never raised there; " + _RERUN),
    "1445a1e": (("src/sraging/population.py",),
                "T130: auto's n = int boundary (P3_RULE, auto_engine) and auto's Langevin branch; the plants read "
                "population.kaplan_meier, unchanged, and never call auto"),
    "8856ecb": (("src/sraging/langevin/sample.py", "src/sraging/warnings.py"),
                "T141 finding 1 (Naveh 2026-11-05): LANGEVIN_STEP_BIAS_ABOVE_CRIT (b-hat(dt) > crit_95; never under "
                "error=\"ks\") joins a sample's warnings, so clean reads it. The sample's bytes are unchanged (times, ages "
                "and events equal on four configurations), and no phase's verdict, status or judged number reads a "
                "ladder draw's clean or warnings: P1 records each rung's codes (a re-run would add the new one to 58 of "
                "its 80 rungs, its steps and p-hat unchanged), P3 records clean (none of its 6 rows has b-hat above "
                "crit), and HR-11 (iii) and S5 draw with error=\"ks\""),
    "a313d6f": (("src/sraging/langevin/sample.py",),
                "T141 finding 6: p_hat_source's text and docstrings ('the conservative 1/2' is now the scheme's weak "
                "order at the wall, not a bound) and the constant's name (P_HAT_DEFAULT); p-hat and every number are "
                "unchanged, and P3 records the text only"),
    "2602ea3": (("src/sraging/langevin/sample.py",),
                "T142 C4 (Naveh 2026-11-05): p_hat() returns 1/2 for every configuration and reads no test result. "
                "Only a P1 configuration's stated p_hat, bias_at_dt and clean can change, and no committed result "
                "recorded the library's P1-cell p-hat: p3-cost's six rows read the 1/2 source; S4, multi, paths, P2 "
                "and S5 judge with the freeze's p-hat from D_dt_vs_dt2; p1-pooled and plants draw error=\"ks\". The "
                "draws never call p_hat, so the sample bytes are unchanged"),
}
#: tools/ names changed after a result, each checked: "tools/<file>.py:<name>" -> ((shas), reason)
_S5_HOM_PROJ = ("s5-hom's projection only (635967e): the stop rule's price reads spec 002's three score runs, not one; "
                "the verdict path is unchanged, and the run went to its end under Naveh's OK (OVER_CAP_OK)")
REVIEWED_FUNCTIONS = {
    "tools/acceptance_004.py:PILOT_CAPS": (("07fc9db", "7072a64"),
                                           "two keys added (p1-pooled 45, p1-pooled-b 15 CPU-h); every existing cap "
                                           "unchanged, and each earlier phase reads its own"),
    "tools/acceptance_004.py:run_capped": (("370e536",),
                                           "opt-in on_error / on_cell (hr3 only); without them the same loop, a "
                                           "raising cell ending the run as before"),
    "tools/acceptance_004.py:progress_path": (("84d327d",),
                                              "new: robust_run's progress file, diagnostic and never read back; "
                                              "phase_hr3 is unchanged since hr3's run (370e536)"),
    "tools/acceptance_004.py:S5_HOM_002_RUNS": (("635967e",), _S5_HOM_PROJ),
    "tools/acceptance_004.py:phase_s5_hom": (("635967e",), _S5_HOM_PROJ),
    "tools/acceptance_004.py:s5_hom_projected_cpu_h": (("635967e",), _S5_HOM_PROJ),
    "tools/p1_pooled_004.py:projected_cpu_h": (("7072a64",),
                                               "an `overhead` factor for p1-pooled-b's projection, default 1.0, as "
                                               "p1-pooled ran; the stop rule's price only"),
    "tools/p1_pooled_004.py:s5_projected_cpu_h": (("7072a64",), "the same `overhead` factor, default 1.0"),
    "tools/presets_004.py:ROLES": (("07fc9db", "57ac7cb"),
                                   "the hr8, s5 and plants roles only; ROLES['p1'], ['p4'] and ['s5'] as p1-pooled, "
                                   "p1-pooled-b and p4-jax read them are equal by value at their runs and now"),
    "tools/p2_s5_power_004.py:coupled_halving": (("98110b1",),
                                                 "optional base / d, reused when given (A6's re-check in s5-het); "
                                                 "P2's calls pass neither, so it computes S5.law and the gradient as "
                                                 "before"),
}
#: src/ files judged by their own rules instead: the rule code by the recomputed verdicts and RULE_FREEZE below
BY_RULES = ("src/sraging/sampler_rules.py",)
#: the parsed sections of its freeze file each phase reads (dotted paths into the TOML)
_PANEL_ALL = ("sources", "panel", "subsets", "ladder", "guide", "cost", "reference", "rules")
#: the panel and the reference phases come before the reference's pin (reference.sha256, which hr1 reads)
_PANEL_UNPINNED = ("sources", "panel", "subsets", "ladder", "guide", "cost", "reference.M_ref", "reference.M_check",
                   "reference.valid_below", "reference.composite", "reference.composite_cells", "rules")
FREEZE_READS = {
    **{p: (PANEL, _PANEL_ALL) for p in ("hr1", "hr5", "hr2", "hr3-pilot", "hr3", "hr4", "d401")},
    **{p: (PANEL, _PANEL_UNPINNED) for p in ("panel", "reference")},
    "plants": (LANG, ("rules", "judged", "plants", "limits")),
    "hr8": (LANG, ("rules", "judged")),
    "s4": (LANG, ("rules", "judged", "s4", "limits")),
    "paths": (LANG, ("rules", "judged", "paths", "limits")),
    "multi": (LANG, ("rules", "judged", "multi")),
    "s5-hom": (LANG, ("rules",)),
    "s5-het": (LANG, ("rules", "s5")),
}

# ---------------------------------------------------------------------------------------------------- the rules
#: each HR rule function's freeze: HR-1 to HR-5 at freeze-H, HR-6 to HR-11 at freeze-L (T120)
RULE_FREEZE = {**{f: FREEZE_H for f in ("hr1_verdict", "hr2_verdict", "hr3_verdict", "hr4_verdict", "hr5_verdict")},
               **{f: FREEZE_L for f in ("hr6_verdict", "hr7_verdict", "hr8_verdict", "hr9_verdict", "hr10_verdict",
                                        "hr11_verdict", "s5_wall_branch", "frozen_bias", "hr10_censoring")}}
RULES_PY = "src/sraging/sampler_rules.py"
#: a rule function changed after its freeze: sha -> reason
RULE_CHANGES = {}


# ----------------------------------------------------------------------------------------------------- helpers
def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


def ancestor(a, b) -> bool:
    return git("merge-base", "--is-ancestor", a, b).returncode == 0


def name(phase):
    return ACC.result_rel(phase)


def load(phase) -> dict:
    return json.load(open(os.path.join(REPO, name(phase))))


def exists(phase) -> bool:
    return os.path.exists(os.path.join(REPO, name(phase)))


def _json(x):
    return json.loads(json.dumps(x, default=ACC._json_default))


def toml_at(rev, path):
    r = git("show", f"{rev}:{path}")
    return tomllib.loads(r.stdout) if r.returncode == 0 else None


def section(doc, dotted):
    for part in dotted.split("."):
        if not isinstance(doc, dict) or part not in doc:
            return None
        doc = doc[part]
    return doc


def _strip_doc(node):
    for n in ast.walk(node):
        if (isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                and isinstance(n.body[0].value.value, str)):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(node)


_DEFS = {}


def module_defs(rev, path=ACC_PY) -> dict:
    """name -> ast.dump (docstrings removed) of every module-level function, class and assignment target of `path` at
    rev (empty if the file is not there)."""
    if (rev, path) not in _DEFS:
        r = git("show", f"{rev}:{path}")
        out = {}
        for node in (ast.parse(r.stdout).body if r.returncode == 0 else []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[node.name] = _strip_doc(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            out.setdefault(n.id, ast.dump(node))
        _DEFS[(rev, path)] = out
    return _DEFS[(rev, path)]


def _aliases(nodes) -> dict:
    """alias -> ("mod", module) or ("from", module, name) of the import statements in `nodes`."""
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


_INFO = {}


def _info(path):
    """CLOSED_AT's module-level names of `path` (functions, classes, assignment targets) -> node, and its module-level
    import aliases."""
    if path not in _INFO:
        tree = ast.parse(git("show", f"{CLOSED_AT}:{path}").stdout)
        top = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            top.setdefault(n.id, node)
        _INFO[path] = (top, _aliases([x for x in tree.body
                                      if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]))
    return _INFO[path]


def _tool(mod):
    """tools/<mod>.py if `mod` is a tools module, else None."""
    return f"tools/{mod}.py" if "." not in mod and os.path.exists(os.path.join(REPO, "tools", f"{mod}.py")) else None


def _sraging(mod):
    return mod == "sraging" or mod.startswith("sraging.")


def reach_code(phase):
    """(the (tools file, module-level name) pairs reachable from the phase's entry function at HEAD, the sraging
    modules they use). Reachable: the call graph and the module-level names its functions, classes and dispatch tables
    read (a name passed as a string, as robust_run_pool's, included), through the import aliases into other tools
    modules; docstrings play no part. A static over-approximation."""
    todo, seen, src = [(ACC_PY, ACC.PHASES[phase].__name__)], set(), set()
    while todo:
        path, nm = todo.pop()
        if (path, nm) in seen:
            continue
        seen.add((path, nm))
        top, mod_alias = _info(path)
        node = top.get(nm)
        if node is None:
            continue
        alias = {**mod_alias, **_aliases([node])}

        def follow(a, attr=None):
            kind = alias[a]
            m, name_ = (kind[1], attr) if kind[0] == "mod" else (kind[1], kind[2])
            if name_ is not None and _tool(m):
                todo.append((_tool(m), name_))
            elif _sraging(m):
                src.update({m} | ({f"{m}.{name_}"} if name_ else set()))
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
                follow(n.value.id, n.attr)
            elif isinstance(n, ast.Name):
                if n.id in top:
                    todo.append((path, n.id))
                elif n.id in alias:
                    follow(n.id)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in top:
                todo.append((path, n.value))
    return seen, src


def closure(roots) -> list:
    """tools/acceptance_003.import_closure's walk (every import statement, function-level ones included, and each sraging
    package's __init__), from the sraging modules a phase uses; its src/ files."""
    seen, todo = {}, [r for r in roots if _sraging(r)]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        f = A3._module_file(mod)
        seen[mod] = f
        if f is None:
            continue
        parts = mod.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts)) if parts[0] == "sraging"]
        pkg = parts if f.endswith("__init__.py") else parts[:-1]
        for node in ast.walk(ast.parse(open(f).read(), f)):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names if _sraging(a.name)]
            elif isinstance(node, ast.ImportFrom):
                base = (pkg[:len(pkg) - node.level + 1] if node.level else []) + (node.module.split(".")
                                                                                  if node.module else [])
                if base and base[0] == "sraging":
                    todo.append(".".join(base))
                    todo += [".".join(base + [a.name]) for a in node.names]
    return sorted(os.path.relpath(f, REPO) for f in seen.values() if f is not None)


def name_changes(path, name, head) -> list:
    """The commits in head..CLOSED_AT whose own diff changed module-level `name` of `path` (as syntax, docstrings removed)."""
    return [sha[:7] for sha in git("rev-list", "--reverse", f"{head}..{CLOSED_AT}", "--", path).stdout.split()
            if module_defs(f"{sha}^", path).get(name) != module_defs(sha, path).get(name)]


_FILES = {}


def _file_ast(rev, path):
    if (rev, path) not in _FILES:
        r = git("show", f"{rev}:{path}")
        _FILES[(rev, path)] = None if r.returncode != 0 else (A3.without_docstrings(r.stdout) if path.endswith(".py")
                                                              else r.stdout)
    return _FILES[(rev, path)]


def file_changes(path, head) -> list:
    """The commits in head..CLOSED_AT whose own diff changed `path` as syntax (docstrings and comments are not content)."""
    return [sha[:7] for sha in git("rev-list", "--reverse", f"{head}..{CLOSED_AT}", "--", path).stdout.split()
            if _file_ast(f"{sha}^", path) != _file_ast(sha, path)]


def section_changes(path, sec, head) -> list:
    """The commits in head..CLOSED_AT whose own diff changed the parsed section `sec` of the TOML `path`."""
    return [sha[:7] for sha in git("rev-list", "--reverse", f"{head}..{CLOSED_AT}", "--", path).stdout.split()
            if section(toml_at(f"{sha}^", path), sec) != section(toml_at(sha, path), sec)]


def stale_items(phase) -> list:
    """What `phase`'s committed result depends on and that changed since its run, unreviewed: (file, the reachable
    names that changed, or a freeze section, or None for a whole src/ file, the commits)."""
    head = load(phase)["repo_head"]
    out = []
    names, src = reach_code(phase)
    for path, nm in sorted(names):
        if module_defs(head, path).get(nm) != module_defs(CLOSED_AT, path).get(nm):
            shas = name_changes(path, nm, head)
            key = f"{path}:{nm}"
            if not (key in REVIEWED_FUNCTIONS and set(shas) <= set(REVIEWED_FUNCTIONS[key][0])):
                out.append((path, nm, shas))
    for f in closure(tuple(sorted(src))):
        if not f.startswith("src/") or f in BY_RULES or _file_ast(head, f) == _file_ast(CLOSED_AT, f):
            continue
        unreviewed = [x for x in file_changes(f, head) if not (x in REVIEWED and f in REVIEWED[x][0])]
        if unreviewed:
            out.append((f, None, unreviewed))
    if phase in FREEZE_READS:
        path, reads = FREEZE_READS[phase]
        a, b = toml_at(head, path), toml_at(CLOSED_AT, path)
        for sec in reads:
            if section(a, sec) != section(b, sec):
                unreviewed = [x for x in section_changes(path, sec, head)
                              if not (x in REVIEWED and path in REVIEWED[x][0])]
                if unreviewed:
                    out.append((path, f"[{sec}]", unreviewed))
    return out


def test_closed_at_is_prs_10_merge_and_bounds_staleness():
    """Spec 005 T001 (FR-001), as 004's T001 did for 003g: CLOSED_AT is PR #10's merge, on this branch's history, and
    staleness is judged up to it, so spec 005's own commits never stale 004's results. Planted: a result run before
    004's own 8856ecb (the step-bias warning in langevin/sample.py) is stale at the merge. Twin: from the merge on,
    nothing counts, whatever later specs change under src/."""
    parents, subject = git("log", "-1", "--format=%P%n%s", CLOSED_AT).stdout.strip().split("\n")
    assert len(parents.split()) == 2 and "pull request #10" in subject, subject
    assert git("merge-base", "--is-ancestor", CLOSED_AT, "HEAD").returncode == 0
    assert "8856ecb" in file_changes("src/sraging/langevin/sample.py", "8856ecb^")
    later = git("diff", "--name-only", CLOSED_AT, "HEAD", "--", "src").stdout.split()
    assert all(file_changes(f, CLOSED_AT) == [] for f in later + ["src/sraging/langevin/sample.py"])


def stale_reasons(phase) -> list:
    """Why `phase`'s committed result is stale at 004's merge (empty: it is not)."""
    return [f"{f}{'' if what is None else ' ' + what if what.startswith('[') else ':' + what} changed in {shas}"
            for f, what, shas in stale_items(phase)]


def effective(phase) -> str:
    """The status the close counts: Naveh's reading where there is one, else the literal status."""
    d = load(phase)
    if phase in READINGS:
        return READINGS[phase][1].split()[0].rstrip(":")          # "pass under ...", "fail: ...", "capped: ..."
    return d["status"]


RUN_PHASES = [p for p in ACC.PHASE_NAMES if p not in ("docs", "unchanged")]


# ---------------------------------------------------------------------------------------------- committed results
@pytest.mark.parametrize("phase", RUN_PHASES)
def test_each_phase_has_a_committed_result_from_a_clean_run(phase):
    assert git("ls-files", "--error-unmatch", name(phase)).returncode == 0, phase
    d = load(phase)
    assert d["status_for_acceptance"] == "acceptance" and not d["tree_dirty"] and not d["tree_changed_during_run"]
    if phase == "multi":
        return                                                    # its status: test_multi_after_a9
    assert d["status"] == EXPECTED[phase], (phase, d["status"])
    if phase in READINGS:
        assert READINGS[phase][0] == d["status"], f"{phase}: the reading was made on a {READINGS[phase][0]!r} result"


def test_multi_is_a9s_rerun_and_reads_its_expected_status():
    """A9 (Naveh 2026-11-05): multi re-runs once at the fresh key base 60600 with the composite cross-checking the
    tensor reference at 1e-4; a fail in the re-run stands. The first run is kept as evidence."""
    d = load("multi")
    assert os.path.exists(os.path.join(REPO, "specs/004-samplers/review/evidence/multi_004_first_252ad3a.json"))
    if not ancestor(A9, d["repo_head"]):
        not_applicable(f"multi_004.json ran at {d['repo_head'][:7]}, before A9 ({A9}): its re-run is not committed")
    lang = toml_at(d["repo_head"], LANG)
    assert lang["multi"]["key"] == 60600 and lang["multi"]["composite"]["bar"] == 1e-4
    if MULTI_EXPECTED is None:
        not_applicable("MULTI_EXPECTED is not set: the coordinator sets it from A9's committed re-run")
    assert d["status"] == MULTI_EXPECTED


def test_every_reading_names_its_session_in_the_spec():
    spec = open(os.path.join(REPO, "specs/004-samplers/spec.md")).read()
    for phase, (_lit, reading, session) in {**READINGS, **{p: (None, "", s) for p, s in OVER_CAP_OK.items()}}.items():
        first = session.split(";")[0].split(" (")[0]
        assert first in spec, (phase, session)


# ------------------------------------------------------------------------------------------------------- ordering
def _first_commit(path):
    r = git("log", "--reverse", "--format=%H", "--", path).stdout.split()
    return r[0] if r else None


def test_t001_came_before_the_first_src_commit_and_after_003gs_merge():
    first_src = git("rev-list", "--reverse", f"{CLOSED_AT_003G}..HEAD", "--", "src").stdout.split()[0]
    assert ancestor(CLOSED_AT_003G, T001) and ancestor(T001, first_src) and T001 != first_src[:7]


@pytest.mark.parametrize("phase", PILOTS_BEFORE_H)
def test_the_heterogeneity_pilots_ran_before_freeze_h(phase):
    assert ancestor(load(phase)["repo_head"], FREEZE_H), phase


@pytest.mark.parametrize("phase", PILOTS_BEFORE_L)
def test_the_langevin_pilots_ran_before_freeze_l(phase):
    assert ancestor(load(phase)["repo_head"], FREEZE_L), phase


@pytest.mark.parametrize("phase", ACC.HETEROGENEITY)
def test_every_heterogeneity_phase_ran_after_freeze_h(phase):
    assert ancestor(FREEZE_H, load(phase)["repo_head"]), phase


@pytest.mark.parametrize("phase", ACC.LANGEVIN)
def test_every_langevin_phase_ran_after_freeze_l(phase):
    assert ancestor(FREEZE_L, load(phase)["repo_head"]), phase


def test_the_freezes_are_the_commits_that_set_their_oks():
    assert toml_at(f"{FREEZE_H}^", PANEL) is None or not toml_at(f"{FREEZE_H}^", PANEL).get("naveh_ok_h")
    assert toml_at(FREEZE_H, PANEL)["naveh_ok_h"] and toml_at(FREEZE_L, LANG)["naveh_ok_l"]
    assert not (toml_at(f"{FREEZE_L}^", LANG) or {}).get("naveh_ok_l")


def test_the_reference_was_pinned_before_hr1_and_hr1_read_the_pinned_file():
    pinned = toml_at(REFERENCE_PIN, PANEL)["reference"]["sha256"]
    assert not section(toml_at(f"{REFERENCE_PIN}^", PANEL), "reference.sha256")
    hr1 = load("hr1")
    assert ancestor(REFERENCE_PIN, hr1["repo_head"]) and hr1["reference_sha256"] == pinned
    assert ACC.file_sha256(name("reference")) == pinned


def test_hr3_pilot_came_before_hr3():
    assert ancestor(_first_commit(name("hr3-pilot")), load("hr3")["repo_head"])


def test_p3s_result_came_before_t130():
    assert ancestor(_first_commit(name("p3-cost")), T130)
    from sraging.population import P3_RULE
    assert P3_RULE["rule"] == load("p3-cost")["boundary"]["rule"]


# ------------------------------------------------------------------------------------------------------ staleness
@pytest.mark.parametrize("phase", RUN_PHASES)
def test_each_result_ran_at_the_code_and_freeze_it_reads_now(phase):
    reasons = stale_reasons(phase)
    assert not reasons, f"{phase} ran at {load(phase)['repo_head'][:7]}; since then:\n  " + "\n  ".join(reasons)


@pytest.mark.parametrize("sha", sorted(REVIEWED))
def test_each_reviewed_commit_touches_only_its_declared_files(sha):
    files, _why = REVIEWED[sha]
    touched = set(git("show", "--name-only", "--format=", sha).stdout.split())
    assert touched <= set(files) | {f for f in touched if not f.startswith(("src/", "tools/", "tests/acceptance/"))}


def test_staleness_sees_a_code_change_and_not_a_comment():
    """Name-level reach: hr8 reaches plants_004.hr8_rows; p1-pooled reaches s5_gen1_004.params_of and not replicate
    (5076c42 changed replicate only). Twin: a docstring-only difference is not content (28ec2f2, the DPU rename)."""
    names, src = reach_code("hr8")
    assert (ACC_PY, "phase_hr8") in names and ("tools/plants_004.py", "hr8_rows") in names
    pooled, _ = reach_code("p1-pooled")                      # p1-pooled reads s5_gen1_004's params_of, not replicate
    assert ("tools/s5_gen1_004.py", "params_of") in pooled and ("tools/s5_gen1_004.py", "replicate") not in pooled
    assert _file_ast("28ec2f2^", "src/sraging/engine/stepper_gen.py") == _file_ast("28ec2f2",
                                                                                   "src/sraging/engine/stepper_gen.py")
    assert module_defs("c0f7732")["phase_s5_het"] != module_defs("98110b1")["phase_s5_het"]   # A6's re-check landed


# ---------------------------------------------------------------------------------------------------- the rules
def _recomputed(phase) -> list:
    """(stored, recomputed) verdict pairs of a phase, the rule called as the phase calls it on its stored rows."""
    d = load(phase)
    pan = toml_at(d["repo_head"], PANEL) or {}
    vs = d.get("verdicts") or []
    if phase == "hr1":
        rules = (pan.get("rules") or {}).get("HR-1") or {}
        v = R.hr1_verdict(d["rows"], **{k: rules[k] for k in ("target", "threshold", "sd_strata") if k in rules})
        return [(vs[0], v.as_dict()), (vs[1], ACC.fr009_verdict(d["fr009"]))]
    if phase == "hr2":
        cells = [p for c in d["cohorts"] for p in c["points"]]
        return [(vs[0], R.hr2_verdict(cells, d["plants"], tol=d["tol"]).as_dict())]
    if phase == "hr3":
        pairs, plants = ACC.hr3_rows(d["cells"])
        return [(vs[0], R.hr3_verdict(pairs, plants, cpu_spent=vs[0]["cpu_spent"], cap=vs[0]["cap"],
                                      threshold=d["rules"]["threshold"]).as_dict())]
    if phase == "hr4":
        rules = ACC.hr4_rules(pan)
        rows = [x for o in d["pools"] for c in o["cells"] for x in c["rows"]]
        plants = [c["plant"] for o in d["pools"] for c in o["cells"]]
        walls = [w for o in d["pools"] for c in o["cells"] for w in c["walls"]]
        return [(vs[0], R.hr4_verdict(rows, plants, walls=walls, stated_W=int(rules["stated_W"])).as_dict())]
    if phase == "hr5":
        rules = (pan.get("rules") or {}).get("HR-5") or {}
        kw = {k: tuple(rules[k]) if isinstance(rules[k], list) else rules[k] for k in ("band", "share", "median_band")
              if k in rules}
        return [(vs[0], R.hr5_verdict(d["cells"], **kw).as_dict())]
    if phase == "hr8":
        return [(vs[0], R.hr8_verdict(d["rows"], d["plants"]).as_dict())]
    if phase == "plants":
        p = next(x for x in d["plants"] if x["id"] == "D-402")
        alpha = toml_at(d["repo_head"], LANG)["plants"]["hr9"]["alpha"]
        return [(d["hr9"], R.hr9_verdict({"plant": p["plant"], "twin": p["twin"], "fix": p["fix"]},
                                         alpha=alpha).as_dict())]
    if phase == "s4":
        return [(v, R.hr6_verdict([c for c in d["cells"] if c["wall"] == v["wall"]],
                                  d["plants"] if v["wall"] == "clamp" else [], wall=v["wall"],
                                  cpu_spent=v["cpu_spent"], cap=v["cap"], G=d["G"][v["wall"]]).as_dict()) for v in vs]
    if phase == "paths":
        out = []
        for v in vs:
            w = v["wall"]
            mine = [r for r in d["configurations"] if r["wall"] == w]
            checks = dict(d["checks"].get(w, {}))
            done = [r for r in mine if not r.get("error")]
            if done:
                checks["nan_iff_dead"] = {"holds": all(r["nan_check"]["holds"] for r in done),
                                          "per_configuration": {r["id"]: r["nan_check"] for r in done}}
            out.append((v, R.hr10_verdict([t for r in mine for t in r["tests"]], checks,
                                          [p for p in d["plants"] if p.get("wall") in (w, None)], wall=w,
                                          cpu_spent=v["cpu_spent"], cap=v["cap"]).as_dict()))
        return out
    if phase == "multi":
        return [(vs[0], R.hr11_verdict(d["reductions"], d["cells"], d["plants"], d["composite"], d["above_budget"],
                                       cpu_spent=vs[0]["cpu_spent"], cap=vs[0]["cap"]).as_dict())]
    if phase == "s5-het":
        res, rc = d["s5"], d.get("recheck")
        rc_error = (f"{rc['error'].get('type')}: {rc['error'].get('message')}" if rc is not None and rc.get("error")
                    else None)
        st = vs[0]["statistic"]
        return [(vs[0], R.hr7_verdict({"plant": {"z": res["plant"]["z"]}, "twin": {"z": res["twin"]["z"]}},
                                      branch=st["branch"], smallest_detectable=st["smallest_detectable_nats_per_1e5"],
                                      n_errors=len(res["errors"]), recheck_error=rc_error).as_dict())]
    raise KeyError(phase)


HR_PHASES = ("hr1", "hr2", "hr3", "hr4", "hr5", "hr8", "plants", "s4", "paths", "multi", "s5-het")


@pytest.mark.parametrize("phase", HR_PHASES)
def test_each_hr_verdict_is_the_rule_recomputed_on_the_stored_rows(phase):
    for stored, again in _recomputed(phase):
        assert _json(again) == stored, (phase, stored.get("rule"), stored.get("wall"))


def test_the_rule_code_is_frozen_at_its_freeze_except_named_changes():
    for fn, freeze in RULE_FREEZE.items():
        shas = git("rev-list", f"{freeze}..{CLOSED_AT}", "--", RULES_PY).stdout.split()
        for sha in shas:
            before, after = module_defs(f"{sha}^", RULES_PY).get(fn), module_defs(sha, RULES_PY).get(fn)
            if before != after:
                assert sha[:7] in RULE_CHANGES, f"{fn} changed after its freeze {freeze} in {sha[:7]}, not named"


def test_the_rule_freeze_sees_a_rule_change_and_not_a_neighbour():
    """Twin and planted on this branch: 252ad3a (freeze-L) added hr7_verdict's recheck_error, and changed no HR-1 to
    HR-5 rule."""
    assert module_defs("252ad3a^", RULES_PY)["hr7_verdict"] != module_defs("252ad3a", RULES_PY)["hr7_verdict"]
    assert module_defs("252ad3a^", RULES_PY)["hr1_verdict"] == module_defs("252ad3a", RULES_PY)["hr1_verdict"]


# --------------------------------------------------------------------------------------- caps, never PASS, Ns
@pytest.mark.parametrize("phase", RUN_PHASES)
def test_the_stop_rule_record_and_the_cap(phase):
    d = load(phase)
    sr = d["stop_rule"]
    for k in ("projected_cpu_h", "cap_cpu_h", "cpu_spent_h", "naveh_ok", "capped"):
        assert k in sr, (phase, k)
    if d["status"] == "capped":
        assert sr["capped"] and sr["cpu_spent_h"] >= sr["cap_cpu_h"], phase
    elif sr["cpu_spent_h"] > sr["cap_cpu_h"]:
        assert phase in OVER_CAP_OK, f"{phase} spent {sr['cpu_spent_h']:.3g} over its cap {sr['cap_cpu_h']:g}"


def test_capped_and_incomplete_never_count_as_pass():
    statuses = {p: effective(p) for p in RUN_PHASES if p != "multi"}
    for p, s in statuses.items():
        if load(p)["status"] in ("capped", "incomplete"):
            assert not ACC.counts_as_pass(s), p
    tally = ACC.close_tally(statuses)
    assert not tally["all_pass"] and set(tally["capped"]) == {"hr3-pilot", "p1-pooled"}


@pytest.mark.parametrize("phase", HR_PHASES)
def test_every_verdict_carries_its_denominators(phase):
    for v in load(phase).get("verdicts") or [load(phase)["hr9"]]:
        if str(v.get("rule", "")).startswith("HR-"):
            assert isinstance(v.get("n_valid"), int) and isinstance(v.get("n_unassessable"), int), (phase, v["rule"])


# ------------------------------------------------------------------------------------------- the FR-009 finding
def test_fr009_misses_only_above_gen1s_n_cap():
    """D-412: gen-1's 161-node law is within FR-009's 0.1 nats of the converged reference on every panel cohort at
    n <= HET_N_CAP, and every cohort beyond 0.1 is above the cap; the stored verdict is the recomputed one (FAIL)."""
    hr1 = load("hr1")
    n = {r["cohort_id"]: int(r["n"]) for r in csv.DictReader(open(os.path.join(REPO, ACC.IDS_PATH)), delimiter="\t")}
    rows = hr1["fr009"]
    stored = next(v for v in hr1["verdicts"] if v["rule"] == "FR-009")
    assert len(rows) == 40 and ACC.fr009_verdict(rows) == stored and stored["status"] == "fail"
    over = {r["cohort_id"] for r in rows if not r["abs_diff"] < ACC.FR009_TOL}
    assert over == set(stored["over_tol"]) and over and all(n[c] > GEN.HET_N_CAP for c in over)
    assert all(r["abs_diff"] < ACC.FR009_TOL for r in rows if n[r["cohort_id"]] <= GEN.HET_N_CAP)
    assert hr1["gen1_rule_sha256"] == ACC.gen1_rule().sha256


# ------------------------------------------------------------------------------------- not yet run (the close)
def test_docs_built_and_every_marker_recomputes():
    if not exists("docs"):
        not_applicable("docs_004.json is not committed: T135 has not run")
    d = load("docs")
    assert d["status"] == EXPECTED["docs"] and d["build"]["ok"] and d["markers"]
    assert all(m["match"] for m in d["markers"])
    again = ACC.docs_markers(ACC.DOCS_SECTIONS)                 # the sections now recompute as the result says
    assert [(m["path"], m["text"]) for m in again] == [(m["path"], m["text"]) for m in d["markers"]]
    assert all(m["match"] for m in again)


def test_unchanged_passed_at_the_final_code():
    if not exists("unchanged"):
        not_applicable("unchanged_004.json is not committed: T139 has not run")
    d = load("unchanged")
    assert d["status"] == EXPECTED["unchanged"] and d["verdict"]["groups_ok"]
    assert all(v["ok"] for v in d["verdict"]["named"].values()) and not d["verdict"]["skips_unlisted"]
    changed = git("diff", "--name-only", d["repo_head"], CLOSED_AT, "--", "src", "tools", "tests").stdout.split()
    assert not [f for f in changed if not f.startswith("tests/acceptance/results/")], changed


def _results_003():
    spec = importlib.util.spec_from_file_location("test_results_003_for_004",
                                                  os.path.join(REPO, "tests", "acceptance", "test_results_003.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_spec_001s_claims_rerun_t145_holds_for_this_branch():
    # spec 006 T077: read at CLOSED_AT, as the rest of this file is; spec 006's own claims re-run replaced the file
    d = json.loads(git("show", f"{CLOSED_AT}:tests/acceptance/results/claims-001_003.json").stdout)
    changed = git("diff", "--name-only", d["repo_head"], CLOSED_AT, "--", *d["closure"]["files"]).stdout.split()
    if changed or not d["untouched"].get("spec_exempt"):
        not_applicable(f"claims-001_003.json ran at {d['repo_head'][:7]}, before T145's re-run on this branch "
                       f"(closure changed since: {changed[:5]})")
    _results_003().check_claims_001(d)
    exempt = {e["path"] for e in d["untouched"]["spec_exempt"]}
    assert exempt <= {f for f, _t in A3.SPEC_ENGINE_CHANGES["004"]} | {f for f, *_ in A3.SPEC_ORACLE_ADDITIONS["004"]}
