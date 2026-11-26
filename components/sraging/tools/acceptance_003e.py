"""Spec 003e's acceptance phases (specs/003e-library-move/contracts/acceptance.md). No phase calibrates (C1).

    .venv/bin/python tools/acceptance_003e.py diff           # the srbench diff 55fc444 -> the pinned commit, recorded
    .venv/bin/python tools/acceptance_003e.py pin            # writes tests/acceptance/pins_003e.json (then commit it)
    .venv/bin/python tools/acceptance_003e.py known-answer   # SC-001: the tester's v5 score file, line by line
    .venv/bin/python tools/acceptance_003e.py self-pair      # SC-002
    .venv/bin/python tools/acceptance_003e.py refusals       # SC-003
    .venv/bin/python tools/acceptance_003e.py v033           # SC-004: 0 calibrations, the 22's claims equal
    .venv/bin/python tools/acceptance_003e.py jcross         # SC-005, reported
    .venv/bin/python tools/acceptance_003e.py loader         # SC-006
    .venv/bin/python tools/acceptance_003e.py age0-bytes     # SC-007, optional
    .venv/bin/python tools/acceptance_003e.py unchanged      # SC-008

Every phase runs through spec 003's `acceptance_003.Run` (suffix "003e"): a clean committed tree, its price printed
first, its result in tests/acceptance/results/<phase>_003e.json. The tester's code comes from the read-only clone at
the commit it named (`srbench.TESTER_PINNED`); its data (library, store, runs) from its live checkout, read-only and
checked against tests/acceptance/pins_003e.json before use. The tester's sealed v6 comparator run is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sraging.adapters import srbench as ADP  # noqa: E402

PANEL_PATH = "tests/acceptance/panel_003e.toml"
PANEL = tomllib.load(open(os.path.join(REPO, PANEL_PATH), "rb"))
PINS_PATH = os.path.join(REPO, "tests", "acceptance", "pins_003e.json")
RESULTS_DIR = os.path.join(REPO, "tests", "acceptance", "results")
TESTER = os.environ.get("SRAGING_TESTER_MAIN", ADP.DEFAULT_TESTER)
LIBRARY = "v5"
SHARED = ("srbench/contract", "srbench/library", "srbench/run", "srbench/store", "srbench/score",
          "srbench/reference", "srbench/report/comparison.py")


def _git(cwd, *a):
    return subprocess.run(["git", "--no-optional-locks", "-C", cwd, *a], capture_output=True, text=True).stdout.strip()


def canon_sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def pins_drift(old: dict, new: dict) -> list[str]:
    """The pin sections whose content differs between two pin sets (made_at aside)."""
    return [sec for sec in ("libraries", "runs", "arrays", "records") if canon_sha(old.get(sec)) != canon_sha(new.get(sec))]


def sha256_file(path):
    return ADP._sha256(path)


def keys_sha256(keys: dict) -> str:
    """The tester's (jax2's) cross-check of a run's key set: `jq -cS .keys <run.json> | shasum -a 256`."""
    return hashlib.sha256((json.dumps(keys, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def run_path(run_id):
    if run_id in PANEL["runs"]["never_read"]:
        raise ADP.UnsupportedOptionError("run", f"{run_id} is sealed (jax2's no-peek): spec 003e never reads it")
    return os.path.join("runs", f"{run_id}.json")


def provenance(pins=None, library=LIBRARY) -> dict:
    """FR-011: the library and its manifest sha256, the clone's HEAD and status, the pins' sha256, the run files'
    sha256 (pin-checked when pins exist), and the host."""
    clone = ADP.TESTER_PINNED
    lib_rel = os.path.join("data", "library", library, "manifest.json")
    runs = {}
    for r in (PANEL["runs"]["reference"], PANEL["runs"]["method_known_answer"], PANEL["runs"]["null"]):
        rel = run_path(r)
        runs[r] = pins.check_file(rel) if pins else sha256_file(os.path.join(TESTER, rel))
    cpu = ""
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return {"library": library,
            "library_manifest_sha256": pins.check_file(lib_rel) if pins else sha256_file(os.path.join(TESTER, lib_rel)),
            "tester_clone": clone, "tester_clone_head": _git(clone, "rev-parse", "HEAD"),
            "tester_clone_dirty": bool(_git(clone, "status", "--porcelain")),
            "tester_data_root": TESTER, "pins_sha256": pins.sha256() if pins else None, "runs_sha256": runs,
            "host": {"node": platform.node(), "machine": platform.machine(), "cpu": cpu},
            "header_tester_note": HEADER_NOTE}


def _run(phase, args, cost):
    import acceptance_003 as A
    return A.Run(phase, manifests=[PANEL_PATH], dev=args.dev, suffix="003e", cost=cost)


def _mods():
    return ADP.connect_pinned(head=PANEL["tester"]["commit"])


def _pins():
    if not os.path.exists(PINS_PATH):
        sys.exit("REFUSED: tests/acceptance/pins_003e.json does not exist (run the pin phase and commit it first)")
    return ADP.load_pins(PINS_PATH, TESTER)


# --------------------------------------------------------------------------------------------------------- diff
def diff(args):
    """R1: what changed in the shared and called srbench modules between the audited head and the pinned commit."""
    run = _run("diff", args, "git diff in the pinned clone; seconds")
    clone, frm, to = ADP.TESTER_PINNED, PANEL["tester"]["audited_from"], PANEL["tester"]["commit"]
    bad = ADP.pinned_problems(clone, to)
    if bad:
        sys.exit("REFUSED: " + "; ".join(bad))
    names = [ln for ln in _git(clone, "diff", "--name-only", frm, to, "--", *SHARED).split("\n") if ln]
    stat = _git(clone, "diff", "--stat", frm, to, "--", *SHARED)
    allfiles = [ln for ln in _git(clone, "diff", "--name-only", frm, to).split("\n") if ln]
    run.write({"status": "no change" if not names else "changed", "from": frm, "to": to,
               "shared_and_called": list(SHARED), "changed_files": names, "stat": stat,
               "all_changed_files": allfiles, "provenance": provenance()}, name="diff")


# ---------------------------------------------------------------------------------------------------------- pin
def pin(args):
    """R5: the content pins of every tester file the 003e phases read, written to pins_003e.json."""
    run = _run("pin", args, "sha256 of 2 manifests, 3 run files, 3 x 1,776 store records and 15 arrays; seconds")
    if ADP.pinned_problems(ADP.TESTER_PINNED, PANEL["tester"]["commit"]):
        sys.exit("REFUSED: the pinned clone is missing, at another HEAD, or dirty")
    mods = _mods()
    store = mods["store.store"].ResultStore(os.path.join(TESTER, "data", "store"))
    data = {"made_at": {"tester_commit": PANEL["tester"]["commit"], "sraging_head": _git(REPO, "rev-parse", "HEAD"),
                        "tester_data_root": TESTER},
            "libraries": {}, "runs": {}, "records": {}, "arrays": {}, "tester_stated": {}}
    for v, prefix in PANEL["libraries"].items():
        rel = os.path.join("data", "library", v, "manifest.json")
        sha = sha256_file(os.path.join(TESTER, rel))
        if not sha.startswith(prefix):
            sys.exit(f"REFUSED: {rel} sha256 {sha[:16]} is not the tester-stated {prefix}")
        data["libraries"][v] = {"path": rel, "sha256": sha}
    for role in ("reference", "method_known_answer", "null"):
        r = PANEL["runs"][role]
        rel = run_path(r)
        man = json.load(open(os.path.join(TESTER, rel)))
        ks = keys_sha256(man["keys"])
        stated = PANEL["runs_keys_sha256"][r]
        if ks != stated:
            sys.exit(f"REFUSED: {r}'s keys sha256 {ks[:16]} is not the tester-stated {stated[:16]}")
        data["runs"][r] = {"path": rel, "sha256": sha256_file(os.path.join(TESTER, rel)), "role": role,
                           "library_version": man["library_version"], "n_keys": len(man["keys"])}
        data["tester_stated"][r] = {"keys_sha256": stated, "matches": True,
                                    "definition": "sha256(json.dumps(keys, sort_keys=True, separators=(',', ':')) + '\\n')"}
        data["records"][r] = {c: {"key": k, "sha256": sha256_file(store._path(k))} for c, k in man["keys"].items()}
    for c in PANEL["loader"]["byte_compared"] + PANEL["loader"]["refused_windowed"]:
        rel = os.path.join("data", "library", LIBRARY, "cohorts", f"{c}.npz")
        data["arrays"][c] = {"path": rel, "sha256": sha256_file(os.path.join(TESTER, rel))}
    drift = []
    if os.path.exists(PINS_PATH):                     # a re-pin must find the tester's data unchanged
        old = json.load(open(PINS_PATH))
        drift = pins_drift(old, data)
        if drift and not args.repin:
            sys.exit(f"REFUSED: the tester's data differ from the committed pins in {drift} (--repin to accept)")
        data["made_at"]["first_made_at"] = old["made_at"].get("first_made_at", old["made_at"])
    # the result is written first, with the sha256 of the bytes about to be written: writing the tracked pins file
    # before it would make the run's own tree dirty
    blob = (json.dumps(data, indent=1, sort_keys=True) + "\n").encode()
    res = run.write({"status": "written", "pins": os.path.relpath(PINS_PATH, REPO),
                     "pins_sha256": hashlib.sha256(blob).hexdigest(),
                     "n_records": {r: len(v) for r, v in data["records"].items()}, "n_arrays": len(data["arrays"]),
                     "drift_from_committed_pins": drift, "libraries": data["libraries"],
                     "keys_sha256_cross_check": "all 3 equal the tester-stated values",
                     "provenance": provenance()}, name="pin")
    with open(PINS_PATH, "wb") as fh:
        fh.write(blob)
    if res["status_for_acceptance"] == "acceptance" and sha256_file(PINS_PATH) != res["pins_sha256"]:
        sys.exit("FAIL: the pins file written is not the one the result names")


# ------------------------------------------------------------------------------------- the score file (SC-001)
def split_score_file(text: str):
    """A `srbench score` output cut into (header lines, {quantity: block lines}); blank lines dropped. A block is its
    `=== q ===  OVERALL` line, its axis lines and its reason lines; any line after the last block goes under
    "_trailer" (the tester's UNTESTED REGIMES / POOLED notes), so a trailer on one side only is a mismatch."""
    header, blocks, cur = [], {}, None
    for ln in text.split("\n"):
        if ln.startswith("=== "):
            cur = ln.split(" ")[1]
            blocks[cur] = [ln]
        elif not ln.strip():
            if cur is not None and cur != "_trailer":
                cur = None
        elif cur is not None:
            blocks[cur].append(ln)
        elif blocks:
            cur = "_trailer"
            blocks[cur] = [ln]
        else:
            header.append(ln)
    return header, blocks


def render_block(quantity: str, entry: dict, axis_cls) -> list[str]:
    """One quantity's block as `srbench score` prints it, from build_comparison's entry: each axis rebuilt as the
    tester's own `AxisResult(**to_dict())` and printed by its own `format()` (cli.cmd_score's loop)."""
    return ([f"=== {quantity} ===  {entry['overall'].upper()}"]
            + ["   " + axis_cls(**d).format() for d in entry["axes"].values()]
            + [f"   - {r}" for r in entry["reasons"]])


def render_header(method_manifest: dict, ref_manifest: dict, paired: int) -> list[str]:
    """cli.cmd_score's 4 header lines; the reference line is the pointer's stamp, rebuilt from the reference run."""
    m, r = method_manifest, ref_manifest
    return [f"method     {m['method']} v{m['method_version']}",
            f"reference: {r['method']} v{r['method_version']} (run {r['run_id']}, library {r['library_version']})",
            f"library    {m['library_version']}   paired cohorts {paired}",
            "0 calibrations performed -- scoring reads the store."]


def compare_blocks(want: dict, got: dict) -> list[dict]:
    """Every difference between two {quantity: lines}: a block missing on either side, the blocks' order, and each
    line that differs (by index). An empty list is a line-by-line match."""
    out = []
    for q in list(want) + [q for q in got if q not in want]:
        if q not in want or q not in got:
            out.append({"quantity": q, "line": None, "want": want.get(q), "got": got.get(q)})
            continue
        a, b = want[q], got[q]
        for i in range(max(len(a), len(b))):
            x, y = (a[i] if i < len(a) else None), (b[i] if i < len(b) else None)
            if x != y:
                out.append({"quantity": q, "line": i, "want": x, "got": y})
    if not out and list(want) != list(got):
        out.append({"quantity": "_order", "line": None, "want": list(want), "got": list(got)})
    return out


def _side(role, pins):
    """{manifest, store_root, run_id} of a tester run, its file pin-checked before it is read."""
    r = PANEL["runs"][role]
    rel = run_path(r)
    pins.check_file(rel)
    with open(os.path.join(TESTER, rel), encoding="utf-8") as fh:
        man = json.load(fh)
    return {"manifest": man, "store_root": os.path.join(TESTER, "data", "store"), "run_id": r}


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def _widths(sets, q):
    """The closed claims' widths in dex (log10 upper/lower), median and count: the null's beside the reference's."""
    import math
    w = [math.log10(c.upper / c.lower) for cs in sets for c in cs.claims
         if c.parameter == q and c.shape == "closed" and c.lower and c.upper and c.lower > 0]
    return {"n_closed": len(w), "median_width_dex": _median(w)}


def _raw_records(run_id, manifest, pins, mods):
    """{cohort: the raw store record}, each read once and pin-checked (key and bytes)."""
    store = mods["store.store"].ResultStore(os.path.join(TESTER, "data", "store"))
    return {c: json.loads(pins.read_record(run_id, c, k, store)) for c, k in manifest["keys"].items()}


def _utc_day(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime("%Y-%m-%d") if ts else None


def _composition(recs):
    """What the reference's records are (sr-naveh 003e finding 12d): by key library, harness revision and UTC day
    written; and each parameter's claim shapes."""
    import collections
    by = collections.Counter()
    for r in recs.values():
        prov = r.get("code_provenance") or {}
        by[(r["key"].get("library_version"), prov.get("revision"), _utc_day(r.get("written_at")))] += 1
    shapes = {}
    for q in ("eta", "beta", "epsilon", "kappa"):
        cnt = collections.Counter()
        for r in recs.values():
            cl = [c for c in (r["claim_set"].get("claims") or []) if c["parameter"] == q]
            cnt[cl[0]["shape"] if cl else "no claim"] += 1
        shapes[q] = dict(sorted(cnt.items()))
    return {"records": len(recs),
            "groups": [{"key_library": a, "harness_revision": b, "written_utc": c, "n": n}
                       for (a, b, c), n in sorted(by.items(), key=lambda kv: tuple(str(x) for x in kv[0]))],
            "shapes": shapes}


HEADER_NOTE = ("each result's header field `tester` is spec 003's J2 checkout (acceptance_003.Run, shared); the "
               "tester commit 003e pins is provenance.tester_clone_head")


def known_answer(args):
    """SC-001 (research R4): the tester's own v5 score file (jax033-v5 against jax2-v5), reproduced line by line
    through compare_v5 and the tester's AxisResult.format."""
    run = _run("known-answer", args, "3 x 1,776 store records read and sha256-checked, build_comparison once; "
               "about a minute; 0 calibrations")
    pins, mods = _pins(), _mods()
    ka = PANEL["known_answer"]
    score_path = os.path.join(os.path.dirname(os.path.normpath(TESTER)), ka["path"])
    sha = sha256_file(score_path)
    if sha != ka["sha256"]:
        sys.exit(f"REFUSED: {ka['path']} sha256 {sha[:16]} is not the pinned {ka['sha256'][:16]}")
    with open(score_path, encoding="utf-8") as fh:
        want_header, want = split_score_file(fh.read())
    method, ref, null = (_side(r, pins) for r in ("method_known_answer", "reference", "null"))
    cmp = ADP.compare_v5(mods, method, ref, null, library=LIBRARY, pins=pins, tester_root=TESTER)
    paired = cmp["cohort_set"]["paired"]
    V = mods["score.verdict"]
    got = {q: render_block(q, e, V.AxisResult) for q, e in cmp["by_parameter"].items()}
    got_header = render_header(method["manifest"], ref["manifest"], paired)
    header_bad = [{"line": i, "want": a, "got": b} for i, (a, b) in enumerate(zip(want_header, got_header)) if a != b]
    if len(want_header) != len(got_header):
        header_bad.append({"line": None, "want": want_header, "got": got_header})
    bad = compare_blocks(want, got)
    # the planted defect: the last digit of eta's calibration statistic moved by one must be caught
    planted = {q: list(v) for q, v in got.items()}
    planted["eta"][1] = planted["eta"][1].replace("-0.0357", "-0.0358", 1)
    caught = compare_blocks(want, planted) if planted["eta"][1] != got["eta"][1] else []
    n_lines = len(want_header) + sum(len(v) for v in want.values())
    ok = not bad and not header_bad and paired == ka["paired"] and len(caught) == 1
    # beside it (sr-naveh 003e finding 11): the null's coverage says nothing without its width, and the reference's
    # composition (finding 12d)
    sets = {r: ADP.load_pinned_sets(mods, side["manifest"], side["store_root"], pins=pins, run_id=side["run_id"])
            for r, side in (("reference", ref), ("null", null))}
    widths = {q: {r: _widths(v, q) for r, v in sets.items()} for q in ("eta", "beta", "epsilon", "kappa")}
    composition = _composition(_raw_records(ref["run_id"], ref["manifest"], pins, mods))
    run.write({"status": "PASS" if ok else "FAIL", "score_file": ka["path"], "score_file_sha256": sha,
               "paired": paired, "paired_expected": ka["paired"], "lines_compared": n_lines,
               "planted_one_digit_change": {"caught": len(caught) == 1, "mismatch": caught},
               "quantities": list(want), "header_mismatches": header_bad, "mismatches": bad,
               "overall": {q: e["overall"] for q, e in cmp["by_parameter"].items()},
               "null_coverage": {q: e["null"] for q, e in cmp["by_parameter"].items()},
               "closed_widths": widths, "reference_composition": composition,
               "not_proven": "the score file prints no null lines, so the null path is checked by self-pair's round "
                             "trip, not here; both sides come from the tester's store",
               "rendered": {"header": got_header, "blocks": got},
               "sraging_provenance": cmp["sraging_provenance"], "provenance": provenance(pins)})
    if not ok:
        sys.exit(f"FAIL: {len(header_bad)} header and {len(bad)} block mismatches (paired {paired})")


def expected_denominators(ref_sets, truths, paired):
    """Each axis's n on a self-pair, counted independently of the tester from the reference's own claims: closed for
    calibration and information, one-sided for calibration_one_sided, both for accuracy (a point to score), the
    cohorts with a truth for availability, and the paired set for cost."""
    out = {}
    for q in ("eta", "beta", "epsilon", "kappa", "external_hazard"):
        shapes = [c.shape for cs in ref_sets for c in cs.claims if c.parameter == q]
        closed = shapes.count("closed")
        one = shapes.count("upper_bound") + shapes.count("lower_bound")
        out[q] = {"calibration": closed, "calibration_one_sided": one, "availability":
                  sum(1 for t in truths.values() if t.get(q) is not None), "information": closed,
                  "accuracy": closed + one, "cost": paired}
    return out


def self_pair_problems(cmp: dict, expected: dict, want_set: str) -> list[str]:
    """SC-002, strictly (sr-naveh 003e finding 5): every axis "no difference", or "not measured" only where the
    reference holds no claim of that class; n equal to the independent count; the cohort set the paired one; a
    measured statistic at its identity (0 for a difference, 1 for cost's ratio of medians)."""
    bad = []
    for q, e in cmp["by_parameter"].items():
        for k, d in e["axes"].items():
            n = expected.get(q, {}).get(k)
            identity = 1.0 if k == "cost" else 0.0
            if d["outcome"] == "not measured":
                ok = n == 0 and d["denominator"] == 0
            else:
                ok = d["outcome"] == "no difference" and d["statistic"] == identity and d["denominator"] == n
            if not ok or d["cohort_set"] != want_set:
                bad.append(f"{q}.{k}")
    return bad


def _round_trip(cs, T):
    """A stored ClaimSet rebuilt through claimset_from_kwargs, the path sraging's own claims take into compare_v5."""
    if cs.status == "ok":
        return ADP.claimset_from_kwargs(T, cs.cohort_id, cs.method_name, [c.to_dict() for c in cs.claims],
                                        cs.diagnostics, wall_s=cs.wall_clock_seconds, cpu_s=cs.cpu_seconds)
    return T.ClaimSet(cohort_id=cs.cohort_id, method_name=cs.method_name, status=cs.status,
                      wall_clock_seconds=cs.wall_clock_seconds, cpu_seconds=cs.cpu_seconds,
                      diagnostics=dict(cs.diagnostics or {}))


def self_pair(args):
    """SC-002: jax2-v5 against itself through compare_v5, gated strictly (self_pair_problems). Then the known answer
    for the `sets` path (sr-naveh 003e finding 11): the 1,776 reference records rebuilt through claimset_from_kwargs
    and paired, as sraging's own claims, against the reference -- the comparison must equal the self-pair's exactly."""
    from sraging.result import canon
    run = _run("self-pair", args, "2 x 1,776 store records read and sha256-checked, build_comparison twice; "
               "about a minute; 0 calibrations")
    pins, mods = _pins(), _mods()
    ref, null = _side("reference", pins), _side("null", pins)
    cmp = ADP.compare_v5(mods, ref, ref, null, library=LIBRARY, pins=pins, tester_root=TESTER)
    paired = cmp["cohort_set"]["paired"]
    want_set = f"{PANEL['known_answer']['paired']} paired"
    ref_sets = ADP.load_pinned_sets(mods, ref["manifest"], ref["store_root"], pins=pins, run_id=ref["run_id"])
    truths = mods["library.load"].Library(root=ADP.library_root(LIBRARY, TESTER)).truths(list(ref["manifest"]["keys"]))
    expected = expected_denominators(ref_sets, truths, paired)
    bad = self_pair_problems(cmp, expected, want_set)
    rt = [_round_trip(cs, mods["contract.types"]) for cs in ref_sets]
    mine = {"manifest": dict(ref["manifest"], run_id="round-trip"), "sets": rt, "computed_on": LIBRARY, "run_id": None}
    cmp_rt = ADP.compare_v5(mods, mine, ref, null, library=LIBRARY, pins=pins, tester_root=TESTER)
    strip = lambda c: {q: {k: e[k] for k in ("overall", "reasons", "axes", "coverage", "null", "null_one_sided")}  # noqa: E731
                       for q, e in c["by_parameter"].items()}
    rt_equal = canon(strip(cmp_rt)) == canon(strip(cmp))
    rt_diff = [q for q in strip(cmp) if canon(strip(cmp)[q]) != canon(strip(cmp_rt).get(q))]
    ok = not bad and paired == PANEL["known_answer"]["paired"] and rt_equal
    axes = {f"{q}.{k}": {f: d[f] for f in ("outcome", "statistic", "uncertainty", "denominator", "cohort_set")}
            for q, e in cmp["by_parameter"].items() for k, d in e["axes"].items()}
    run.write({"status": "PASS" if ok else "FAIL", "paired": paired, "cohort_set_expected": want_set,
               "failing_axes": bad, "axes": axes, "expected_denominators": expected,
               "overall": {q: e["overall"] for q, e in cmp["by_parameter"].items()},
               "round_trip": {"equal_to_self_pair": rt_equal, "differing_quantities": rt_diff, "records": len(rt),
                              "path": "stored ClaimSet -> to_dict claims -> claimset_from_kwargs -> compare_v5 sets"},
               "reading": "SC-002's 'denominator 1,776' is the paired cohort set on every axis (spec erratum): the "
                          "tester's n is per claim class, checked here against an independent count",
               "sraging_provenance": cmp["sraging_provenance"], "provenance": provenance(pins)})
    if not ok:
        sys.exit(f"FAIL: axes {bad}; round trip equal {rt_equal} {rt_diff} (paired {paired})")


# ------------------------------------------------------------------------------------------------ refusals (SC-003)
def _outcome(fn, *tokens):
    """Run `fn`: a refusal (UnsupportedOptionError or the tester's ValueError) is recorded with its message and
    whether the message names every token of its cause; anything else that raises is recorded as not a refusal."""
    try:
        v = fn()
    except (ADP.UnsupportedOptionError, ValueError) as ex:
        msg = str(ex)
        return {"refused": True, "error": type(ex).__name__, "message": msg,
                "names_cause": all(t in msg for t in tokens), "tokens": list(tokens)}
    except Exception as ex:                                     # noqa: BLE001 -- recorded, and the case fails
        return {"refused": False, "error": type(ex).__name__, "message": str(ex)[:500]}
    return {"refused": False, "ran": True, "result": v}


def _copy_records(run_id, cohorts, pins, dst):
    """Byte copies of a run's pinned store records for `cohorts`, in the store layout, under `dst` (never in T)."""
    for c in cohorts:
        dg = pins.data["records"][run_id][c]["key"]
        rel = os.path.join(dg[:2], dg[2:4], f"{dg}.json")
        os.makedirs(os.path.join(dst, os.path.dirname(rel)), exist_ok=True)
        shutil.copyfile(os.path.join(TESTER, "data", "store", rel), os.path.join(dst, rel))
        yield c, os.path.join(dst, rel)


def _flip_last_byte(path):
    b = bytearray(open(path, "rb").read())
    b[-2] = ord("1") if b[-2] != ord("1") else ord("2")      # before the trailing newline/brace: still a change
    open(path, "wb").write(bytes(b))


def refusals(args):
    """SC-003: the planted cases of User Story 2 and of sr-naveh's review, each refused with a message naming its
    cause, each beside its unplanted twin, which must run. Comparisons run on the 22 pairing cohorts; nothing
    calibrates; every planted file is a temp copy, never the tester's."""
    run = _run("refusals", args, "11 planted cases and their twins: about 14 comparisons on the 22 (seconds each), "
               "temp copies of 22 records and 4 tester files, a temp clone of the pinned clone; about a minute; "
               "0 calibrations")
    pins, mods = _pins(), _mods()
    ids = PANEL["pairing"]["cohorts"]
    ref, null, meth = _side("reference", pins), _side("null", pins), _side("method_known_answer", pins)
    head, other = PANEL["tester"]["commit"], PANEL["tester"]["audited_from"]
    T = mods["contract.types"]
    jsets = ADP.load_pinned_sets(mods, ref["manifest"], ref["store_root"], pins=pins, run_id=ref["run_id"],
                                 cohort_ids=ids)
    jlab = {"method": "sraging_profile_jax2", "method_version": "test", "library_version": LIBRARY,
            "contract_version": T.CONTRACT_VERSION, "run_id": None}

    def mine(method="sraging_profile_jax2", label=LIBRARY, computed_on=LIBRARY, sets=None):
        """sraging's own claims (the reference's 22, rebuilt) as a `sets` side."""
        return {"manifest": dict(jlab, method=method, library_version=label),
                "sets": sets if sets is not None else [_round_trip(cs, T) for cs in jsets],
                "computed_on": computed_on, "run_id": None}

    def cmp(m=meth, r=ref, library=LIBRARY, p=pins, cohort_ids=ids):
        out = ADP.compare_v5(mods, m, r, null, library=library, pins=p, tester_root=TESTER, cohort_ids=cohort_ids)
        return {"paired": out["cohort_set"]["paired"]}

    cases = {}
    # 1. no library
    cases["no_library"] = {"planted": _outcome(lambda: cmp(library=None), "library"), "twin": _outcome(cmp)}
    # 2. a run labelled v1 or v2 against the v5 reference: sraging's refusal (on sraging's own claims, labelled), and
    #    the tester's own require_comparable; a tester run's label cannot be planted (its file is pinned: case 3c)
    for lv in ("v1", "v2"):
        cases[f"method_labelled_{lv}"] = {
            "planted": _outcome(lambda: cmp(m=mine(label=lv)), lv, LIBRARY),
            "tester_require_comparable": _outcome(lambda: mods["reference.pointer"].require_comparable(
                dict(meth["manifest"], library_version=lv), ref["manifest"]), lv, LIBRARY),
            "twin": _outcome(lambda: cmp(m=mine()))}
    with tempfile.TemporaryDirectory(prefix="sraging003e_") as tmp:
        # 3a. a changed byte in a copy of one reference store record
        planted_store, twin_store = os.path.join(tmp, "store_planted"), os.path.join(tmp, "store_twin")
        copied = dict(_copy_records(ref["run_id"], ids, pins, planted_store))
        list(_copy_records(ref["run_id"], ids, pins, twin_store))
        _flip_last_byte(copied[ids[0]])
        cases["reference_record_changed"] = {
            "planted_cohort": ids[0],
            "planted": _outcome(lambda: cmp(r=dict(ref, store_root=planted_store)), ids[0]),
            "twin": _outcome(lambda: cmp(r=dict(ref, store_root=twin_store)))}
        # 3b. a changed byte in a copy of the reference run file, read through compare_v5 itself
        roots = {}
        for d in ("planted", "twin"):
            roots[d] = os.path.join(tmp, f"root_{d}")
            for r in (ref["run_id"], meth["run_id"], null["run_id"]):
                rel = run_path(r)
                os.makedirs(os.path.join(roots[d], "runs"), exist_ok=True)
                shutil.copyfile(os.path.join(TESTER, rel), os.path.join(roots[d], rel))
            lib_rel = os.path.join("data", "library", LIBRARY, "manifest.json")
            os.makedirs(os.path.dirname(os.path.join(roots[d], lib_rel)), exist_ok=True)
            shutil.copyfile(os.path.join(TESTER, lib_rel), os.path.join(roots[d], lib_rel))
        _flip_last_byte(os.path.join(roots["planted"], run_path(ref["run_id"])))
        cases["reference_run_file_changed"] = {
            "planted": _outcome(lambda: cmp(p=ADP.Pins(pins.data, roots["planted"])), run_path(ref["run_id"])),
            "twin": _outcome(lambda: cmp(p=ADP.Pins(pins.data, roots["twin"])))}
        # 3c. a manifest passed with a tester run that is not its pinned file (a relabelled or re-keyed copy)
        rekeyed = dict(ref["manifest"], keys=dict(ref["manifest"]["keys"], **{ids[0]: ref["manifest"]["keys"][ids[1]]}))
        cases["manifest_not_the_pinned_file"] = {
            "planted": _outcome(lambda: cmp(r=dict(ref, manifest=rekeyed)), ref["run_id"], "pinned run file"),
            "twin": _outcome(lambda: cmp(r=dict(ref, manifest=dict(ref["manifest"]))))}
        # 3d. a key that is not the pinned key, at the reading itself (load_pinned_sets)
        cases["key_not_the_pinned_key"] = {
            "planted": _outcome(lambda: len(ADP.load_pinned_sets(mods, rekeyed, ref["store_root"], pins=pins,
                                                                 run_id=ref["run_id"], cohort_ids=ids)), ids[0], "key"),
            "twin": _outcome(lambda: len(ADP.load_pinned_sets(mods, ref["manifest"], ref["store_root"], pins=pins,
                                                              run_id=ref["run_id"], cohort_ids=ids)))}
        # 4. the _h port at v5: built, run and paired (run_records gets no modules, so a missed refusal stops at once)
        cases["h_port_at_v5"] = {
            "planted": _outcome(lambda: ADP.make_jax2_method(T, ADP.H_PORT, library="v5") and None, ADP.H_PORT, "v5"),
            "planted_run_records": _outcome(lambda: ADP.run_records(
                {}, ADP.make_jax2_method(T, ADP.H_PORT, library="v2"), [], library="v5"), ADP.H_PORT, "v5"),
            "planted_compare_v5": _outcome(lambda: cmp(m=mine(method=ADP.H_PORT)), ADP.H_PORT, "v5"),
            "twin": _outcome(lambda: ADP.make_jax2_method(T, ADP.H_PORT, library="v2").descriptor.name),
            "twin_compare_v5": _outcome(lambda: cmp(m=mine()))}
        # 5. a clone at another HEAD: refused before any import; the twin connects in a fresh process
        clone = os.path.join(tmp, "clone")
        subprocess.run(["git", "clone", "-q", "--no-checkout", ADP.TESTER_PINNED, clone], check=True)
        subprocess.run(["git", "-C", clone, "checkout", "-q", "--detach", other], check=True)
        planted = _outcome(lambda: ADP.connect_pinned(clone=clone, head=head), head, other[:7])
        subprocess.run(["git", "-C", clone, "checkout", "-q", "--detach", head], check=True)
        code = ("import sys; from sraging.adapters import srbench as A; A.connect_pinned(clone=sys.argv[1], "
                "head=sys.argv[2]); import srbench; print(srbench.__file__)")
        p = subprocess.run([sys.executable, "-c", code, clone, head], capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                "PYTHONPATH": os.path.join(REPO, "src")})
        twin = ({"refused": False, "ran": True, "result": {"srbench_from_temp_clone": p.stdout.strip().startswith(
                    os.path.realpath(clone)) or p.stdout.strip().startswith(clone)}}
                if p.returncode == 0 else {"refused": True, "message": p.stderr[-1000:]})
        cases["clone_at_another_head"] = {"planted": planted, "twin": twin}
    # 6. claims computed on v2 for a cohort whose arrays changed v2 -> v5 (sr-naveh finding 4), and no computed_on
    ck2 = json.load(open(os.path.join(TESTER, "data", "library", "v2", "manifest.json")))["checksums"]
    ck5 = json.load(open(os.path.join(TESTER, "data", "library", LIBRARY, "manifest.json")))["checksums"]
    changed = next(c for c in ref["manifest"]["keys"] if c in ck2 and ck2[c] != ck5[c])
    with_changed = ADP.load_pinned_sets(mods, ref["manifest"], ref["store_root"], pins=pins, run_id=ref["run_id"],
                                        cohort_ids=[changed])
    sets6 = [_round_trip(cs, T) for cs in jsets + with_changed]
    cases["claims_on_changed_arrays"] = {
        "planted_cohort": changed,
        "planted": _outcome(lambda: cmp(m=mine(computed_on="v2", sets=sets6), cohort_ids=ids + [changed]),
                            changed, "v2", LIBRARY),
        "planted_no_computed_on": _outcome(lambda: cmp(m={k: v for k, v in mine().items() if k != "computed_on"}),
                                           "computed_on"),
        "twin": _outcome(lambda: cmp(m=mine(computed_on="v2", sets=sets6), cohort_ids=ids))}

    def case_ok(c):
        planted = [v for k, v in c.items() if k.startswith("planted") and isinstance(v, dict)
                   or k == "tester_require_comparable"]
        twins = [v for k, v in c.items() if k.startswith("twin")]
        return (all(v.get("refused") and v.get("names_cause") for v in planted)
                and all(t.get("ran") is True for t in twins)
                and all((t["result"].get("srbench_from_temp_clone", True)
                         and t["result"].get("paired", len(ids)) == len(ids)) if isinstance(t.get("result"), dict)
                        else True for t in twins))
    verdicts = {k: case_ok(c) for k, c in cases.items()}
    ok = all(verdicts.values())
    run.write({"status": "PASS" if ok else "FAIL", "verdicts": verdicts, "cases": cases, "cohorts": len(ids),
               "provenance": provenance(pins)})
    if not ok:
        sys.exit(f"FAIL: {[k for k, v in verdicts.items() if not v]}")


# ------------------------------------------------------------------------------------------------- v033 (SC-004)
V033_RUN_ID = "sraging_v033-v5-pair22"


def v033(args):
    """SC-004 (research R8): the v0.3.3 port on the 22 at v5 with 0 calibrations. The tester's own plan_run must show
    0 to compute (refused before run() otherwise: a computed failure would itself be stored), run() must record
    n_computed 0, and the port's records must equal jax033-v5-full-bign's claim for claim (canon, exact floats).
    Reported beside it: the port paired against the jax2 v5 reference through compare_v5."""
    from sraging.result import canon
    run = _run("v033", args, "plan_run on the 22 (store lookups), run() reusing all 22, 2 x 22 records compared, one "
               "comparison on the 22; seconds; 0 calibrations (refused before run() otherwise)")
    pins, mods = _pins(), _mods()
    ids = PANEL["pairing"]["cohorts"]
    method = ADP.register_with(mods["contract.method"], ADP.make_method(mods["contract.types"]))
    lib = mods["library.load"].Library(root=ADP.library_root(LIBRARY, TESTER))
    plan = mods["run.execute"].plan_run(method, lib, ids, mods["store.store"].ResultStore(ADP.PORT_STORE))
    if plan.compute:
        sys.exit(f"REFUSED: the plan would calibrate {len(plan.compute)} of {len(ids)} (first {plan.compute[:3]}); "
                 "spec 003e computes nothing (C1)")
    manifest, path = ADP.run_records(mods, method, ids, tester_root=TESTER, store_root=ADP.PORT_STORE,
                                     run_id=V033_RUN_ID, library=LIBRARY)
    port = {cs.cohort_id: cs for cs in ADP.load_pinned_sets(mods, manifest, ADP.PORT_STORE)}
    ka = _side("method_known_answer", pins)
    ref = {cs.cohort_id: cs for cs in ADP.load_pinned_sets(mods, ka["manifest"], ka["store_root"], pins=pins,
                                                           run_id=ka["run_id"], cohort_ids=ids)}
    per = {}
    for c in ids:
        a, b = port[c], ref[c]
        per[c] = {"equal": (a.status == b.status
                            and canon([x.to_dict() for x in a.claims]) == canon([x.to_dict() for x in b.claims])),
                  "status": [a.status, b.status], "n_claims": [len(a.claims), len(b.claims)],
                  "port_key": manifest["keys"][c], "jax033_key": ka["manifest"]["keys"][c]}
    n_equal = sum(v["equal"] for v in per.values())
    cmp = ADP.compare_v5(mods, {"manifest": manifest, "store_root": ADP.PORT_STORE, "run_id": None},
                         _side("reference", pins), _side("null", pins), library=LIBRARY, pins=pins,
                         tester_root=TESTER, cohort_ids=ids)
    ok = (not plan.compute and manifest["n_computed"] == 0 and manifest["n_reused"] == len(ids)
          and n_equal == len(ids) and manifest["library_version"] == LIBRARY)
    run.write({"status": "PASS" if ok else "FAIL", "cohorts": len(ids),
               "plan": {"reuse": len(plan.reuse), "compute": len(plan.compute)},
               "run_manifest": {k: manifest[k] for k in ("method", "method_version", "library_version", "n_cohorts",
                                                         "n_reused", "n_computed", "n_failed")},
               "run_manifest_path": os.path.relpath(path, REPO), "claims_equal": n_equal, "per_cohort": per,
               "paired_with_reference": {"paired": cmp["cohort_set"]["paired"],
                                         "overall": {q: e["overall"] for q, e in cmp["by_parameter"].items()},
                                         "axes": {q: {k: d["outcome"] for k, d in e["axes"].items()}
                                                  for q, e in cmp["by_parameter"].items()}},
               "sraging_provenance": cmp["sraging_provenance"], "provenance": provenance(pins)})
    if not ok:
        sys.exit(f"FAIL: plan compute {len(plan.compute)}, n_computed {manifest['n_computed']}, "
                 f"claims equal {n_equal}/{len(ids)}")



# ------------------------------------------------------------------------------------------------ jcross (SC-005)
CLAIM_FIELDS = ("parameter", "shape", "lower", "upper", "point", "level", "n_components")
HIT_PARAMS = ("eta", "beta", "epsilon")          # kappa is reported, not counted: its truth sits on a flat ridge


def _q(xs):
    """n, median, 90th percentile, max and the count above 0.03 dex of a list of |dex| values."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    pick = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"n": len(xs), "median": pick(0.5), "p90": pick(0.9), "max": xs[-1],
            "n_above_0_03": sum(x > 0.03 for x in xs)}


def _claims_by_param(cs):
    return {c.parameter: {f: c.to_dict().get(f) for f in CLAIM_FIELDS} for c in cs.claims}


def _hit(c, t):
    """Whether the truth lies inside a claim (closed, or the bound's side); None for a shape with no interval."""
    if c is None or t is None:
        return None
    if c["shape"] == "closed":
        return c["lower"] <= t <= c["upper"]
    if c["shape"] == "upper_bound":
        return t <= c["upper"]
    if c["shape"] == "lower_bound":
        return t >= c["lower"]
    return None


def _cross(port_sets, ref_sets, arms, truths, floors, log_error):
    """Per cohort and per parameter: bit-equal (canon), shape agreement, coverage-hit agreement against the truth, and
    |dex| of the ends and of closed claims' points, each clamped at the tester's accuracy floor (score.floors, through
    its own claims.log_error) so that two values below it -- both meaning zero -- do not count as a move."""
    from sraging.result import canon
    ref = {cs.cohort_id: cs for cs in ref_sets}
    per, shape_flips, hit_flips, ends, points, hits = {}, [], [], {}, {}, {"claims": 0, "flips": 0}
    for cs in port_sets:
        c = cs.cohort_id
        a, b = _claims_by_param(cs), _claims_by_param(ref[c])
        row = {"bit_equal": cs.status == ref[c].status and canon(a) == canon(b), "params": {}}
        for p in sorted(set(a) | set(b)):
            x, y = a.get(p), b.get(p)
            if x is None or y is None:
                row["params"][p] = {"present": [x is not None, y is not None]}
                shape_flips.append(f"{c}:{p}")
                continue
            fl = floors[p].get(c)
            dx = lambda u, v: log_error(u, v, fl) if u is not None and v is not None else None  # noqa: E731
            d = {"shape": [x["shape"], y["shape"]], "lower": dx(x["lower"], y["lower"]),
                 "upper": dx(x["upper"], y["upper"]),
                 "point": dx(x["point"], y["point"]) if x["shape"] == y["shape"] == "closed" else None,
                 "floor": fl}
            if p in HIT_PARAMS:
                t = truths[c].get(p)
                d["hit"] = [_hit(x, t), _hit(y, t)]
                if d["hit"][0] is not None or d["hit"][1] is not None:
                    hits["claims"] += 1
                    if d["hit"][0] != d["hit"][1]:
                        hits["flips"] += 1
                        hit_flips.append(f"{c}:{p}")
            row["params"][p] = d
            if x["shape"] != y["shape"]:
                shape_flips.append(f"{c}:{p}")
            for k in (arms[c], "all"):
                ends.setdefault(k, []).extend([d["lower"], d["upper"]])
                points.setdefault(k, []).append(d["point"])
        per[c] = row
    above = [{"cohort": c, "parameter": p, "shape": d["shape"],
              **{k: d[k] for k in ("lower", "upper", "point") if d[k] is not None and d[k] > 0.03}}
             for c, r in per.items() for p, d in r["params"].items()
             if "shape" in d and any(d[k] is not None and d[k] > 0.03 for k in ("lower", "upper", "point"))]
    by_arm = {}
    for k in sorted(set(arms.values())) + ["all"]:
        ids = [c for c in per if k == "all" or arms[c] == k]
        by_arm[k] = {"cohorts": len(ids), "bit_equal": sum(per[c]["bit_equal"] for c in ids),
                     "ends_dex": _q(ends.get(k, [])), "closed_points_dex": _q(points.get(k, []))}
    return {"by_arm": by_arm, "shape_flips": shape_flips, "n_shape_flips": len(shape_flips),
            "coverage_hits": {**hits, "params": list(HIT_PARAMS), "flipped": hit_flips},
            "above_0_03_dex": above, "n_ends_above_0_03": by_arm["all"]["ends_dex"].get("n_above_0_03", 0),
            "n_ends": by_arm["all"]["ends_dex"].get("n", 0), "per_cohort": per}


def _fisher(a, b):
    """Two-sided Fisher exact p of two (equal, total) counts."""
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1])


def jcross(args):
    """SC-005 (research R9), REPORTED, not gated: the J port's claims against the stored jax2 v5 reference without any
    fit -- (a) the 22 from spec 003's fits on this Mac, (b) the 118 WEXAC coverage rows selected by PANEL's rule. Both
    are sraging's own claims computed on v2 (`computed_on`), which compare_v5 accepts only for cohorts whose arrays
    are unchanged v2 -> v5. Beside it, from sr-naveh's review: coverage-hit agreement, floor-clamped ends and closed
    points, the rows' and the sources' sha256, the two port heads, and bit-equality by the reference record's batch
    and the port row's CPU."""
    import glob
    run = _run("jcross", args, "fits_003.json and 215 WEXAC rows read, 140 reference records sha256-checked, two "
               "comparisons; seconds; 0 calibrations")
    pins, mods = _pins(), _mods()
    T = mods["contract.types"]
    jc = PANEL["jcross"]
    ref, null = _side("reference", pins), _side("null", pins)
    v2_path = os.path.join(TESTER, "data", "library", "v2", "manifest.json")
    ck5 = json.loads(pins.read_file(os.path.join("data", "library", LIBRARY, "manifest.json")))["checksums"]
    ck2 = json.load(open(v2_path))["checksums"]
    unchanged = lambda c: c in ck2 and ck2[c] == ck5.get(c)  # noqa: E731
    name = "sraging_profile_jax2"
    port_manifest = {"method": name, "method_version": ADP.descriptor_fields(name)["version"],
                     "library_version": LIBRARY, "contract_version": T.CONTRACT_VERSION}
    mac_path = os.path.join(REPO, jc["mac_source"])
    fits = json.load(open(mac_path))

    # (a) the 22, this Mac (spec 003's fits, port claims)
    ids22 = PANEL["pairing"]["cohorts"]
    rows_a = {r["cohort_id"]: r for r in fits["rows"] if r["name"] == "jax2" and r["cohort_id"] in ids22}
    missing = [c for c in ids22 if c not in rows_a or not unchanged(c) or c not in ref["manifest"]["keys"]]
    if missing:
        sys.exit(f"REFUSED: (a) {len(missing)} of the 22 lack a jax2 fits row, unchanged v2->v5 arrays or a v5 "
                 f"reference record: {missing[:5]}")
    sets_a = [ADP.claimset_from_kwargs(T, c, name, rows_a[c]["port_claims"], {}, cpu_s=rows_a[c]["port_cpu_s"])
              for c in ids22]

    # (b) the WEXAC coverage rows by PANEL's rule
    shards = sorted(glob.glob(os.path.join(REPO, jc["wexac_rows"], "*.jsonl")))
    rows = [json.loads(ln) for f in shards for ln in open(f) if ln.strip()]
    sel = {r["cohort_id"]: r for r in rows if r["calibrator"] == "profile_jax2" and unchanged(r["cohort_id"])
           and r["cohort_id"] in ref["manifest"]["keys"]}
    by_arm = {}
    for r in sel.values():
        by_arm[r["arm"]] = by_arm.get(r["arm"], 0) + 1
    if len(sel) != jc["wexac_expected"] or by_arm != dict(jc["wexac_by_arm"]):
        sys.exit(f"REFUSED: the rule selects {len(sel)} rows {by_arm}, not {jc['wexac_expected']} {jc['wexac_by_arm']}")
    if any(r["status"] != "ok" for r in sel.values()):
        sys.exit("REFUSED: a selected WEXAC row is not ok")
    ids_b = sorted(sel)
    sets_b = [ADP.claimset_from_kwargs(T, c, name, sel[c]["claims"], {}, cpu_s=sel[c].get("cpu_s")) for c in ids_b]

    lib = mods["library.load"].Library(root=ADP.library_root(LIBRARY, TESTER))
    raw = _raw_records(ref["run_id"], {"keys": {c: ref["manifest"]["keys"][c] for c in ids22 + ids_b}}, pins, mods)
    out = {}
    for tag, ids, sets, hosts in (("mac_22", ids22, sets_a, [fits["host"]]),
                                  ("wexac_118", ids_b, sets_b, sorted({(r["host"]["node"], r["host"]["cpu"])
                                                                       for r in sel.values()}))):
        ref_sets = ADP.load_pinned_sets(mods, ref["manifest"], ref["store_root"], pins=pins, run_id=ref["run_id"],
                                        cohort_ids=ids)
        arms = {c: c.split("-")[0] for c in ids}
        specs = {c: lib.spec(c) for c in ids}
        floors = {p: mods["score.floors"].cohort_floors(specs, p) for p in ("eta", "beta", "epsilon", "kappa")}
        cross = _cross(sets, ref_sets, arms, lib.truths(ids), floors, mods["score.claims"].log_error)
        cmp = ADP.compare_v5(mods, {"manifest": dict(port_manifest, run_id=f"jcross-{tag}"), "sets": sets,
                                    "computed_on": "v2", "run_id": None}, ref, null, library=LIBRARY, pins=pins,
                             tester_root=TESTER, cohort_ids=ids)
        verdicts = {q: {"overall": e["overall"], "axes": {k: d["outcome"] for k, d in e["axes"].items()}}
                    for q, e in cmp["by_parameter"].items()}
        other = [f"{q}.{k}" for q, e in verdicts.items() for k, o in e["axes"].items()
                 if o not in ("no difference", "not measured")]
        out[tag] = {"cohorts": len(ids), "port_hosts": hosts,
                    "reference_records": {"host": "none recorded",
                                          "written_utc": sorted({_utc_day(raw[c].get("written_at")) for c in ids}),
                                          "harness_revision": sorted({str((raw[c].get("code_provenance") or {})
                                                                          .get("revision")) for c in ids})},
                    **cross, "paired": cmp["cohort_set"]["paired"], "verdicts": verdicts,
                    "axes_not_no_difference": other, "sraging_provenance": cmp["sraging_provenance"]}

    # bit-equality by the reference record's batch (UTC day written) and the port row's CPU (sr-naveh finding 1)
    cells = {}
    for c in ids_b:
        k = (_utc_day(raw[c].get("written_at")), sel[c]["host"]["cpu"])
        n = cells.setdefault(k, [0, 0])
        n[0] += out["wexac_118"]["per_cohort"][c]["bit_equal"]
        n[1] += 1
    table = [{"reference_written_utc": d, "port_cpu": cpu, "bit_equal": v[0], "of": v[1]}
             for (d, cpu), v in sorted(cells.items())]

    def pooled(f):
        e = [sum(v[0] for k, v in cells.items() if f(k)), sum(v[1] for k, v in cells.items() if f(k))]
        return e
    cpus = sorted({k[1] for k in cells})
    days = sorted({k[0] for k in cells})
    tests = {}
    if len(days) == 2:
        for cpu in cpus:
            a, b = pooled(lambda k: k == (days[0], cpu)), pooled(lambda k: k == (days[1], cpu))
            if a[1] and b[1]:
                tests[f"batch, {cpu}: {days[0]} vs {days[1]}"] = {"a": a, "b": b, "fisher_p": _fisher(a, b)}
    if len(cpus) == 2:
        for d in days:
            a, b = pooled(lambda k: k == (d, cpus[0])), pooled(lambda k: k == (d, cpus[1]))
            if a[1] and b[1]:
                tests[f"cpu, batch {d}: {cpus[0]} vs {cpus[1]}"] = {"a": a, "b": b, "fisher_p": _fisher(a, b)}

    pred = {"zero_shape_flips_on_the_22": not out["mac_22"]["shape_flips"],
            "every_axis_no_difference_or_not_measured": {k: not v["axes_not_no_difference"] for k, v in out.items()}}
    run.write({"status": "REPORTED", "gated": False, "prediction_r9": pred,
               "coverage_hits_all": {"claims": sum(v["coverage_hits"]["claims"] for v in out.values()),
                                     "flips": sum(v["coverage_hits"]["flips"] for v in out.values())},
               "bit_equal_by_batch_and_cpu": {"table": table, "fisher": tests},
               "selection": {"rule": jc["wexac_rule"], "by_arm": by_arm, "v2_manifest_sha256": sha256_file(v2_path),
                             "excluded_by_the_rule": "every hazard cohort, every human life table, every age-0 cohort "
                                                     "and every record recomputed at v5 (arrays changed v2 -> v5)"},
               "sources": {"mac_fits": {"path": jc["mac_source"], "sha256": sha256_file(mac_path),
                                        "port_head": fits.get("repo_head")},
                           "wexac_rows": {"dir": jc["wexac_rows"], "port_heads": sorted({r["head"] for r in rows}),
                                          "shards_sha256": {os.path.basename(f): sha256_file(f) for f in shards},
                                          "selected_rows_sha256": {c: hashlib.sha256(json.dumps(
                                              sel[c], sort_keys=True).encode()).hexdigest() for c in ids_b}},
                           "port_heads_note": "the two sources are at two port heads; profile_jax2 changed between "
                                              "them in the combination walker only (339d113), and FR-019 keeps the "
                                              "parameter claims"},
               **out, "provenance": provenance(pins)})

# ----------------------------------------------------------------------------------------------- loader (SC-006)
def loader(args):
    """SC-006 (research R10): the loader on v5's first-bin convention. For the 12, the pinned srbench Library gives
    observation(c) and cohort_from_observation must reproduce its ages and events byte for byte (the npz pin-checked
    first); the 3 windowed cohorts must be refused, naming the window. Positive control: every age-0 death term at the
    library truth is finite and above the -700 floor under the port's own likelihood (profile_jax2, or profile_jax2_h
    for the hazard cohorts) on the fit's own grid -- an evaluation, not a fit."""
    import types
    import numpy as np
    import acceptance_003 as A3
    from sraging.adapters.observation import cohort_from_observation
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import fp_jax as fj
    run = _run("loader", args, "15 arrays sha256-checked and loaded; one likelihood evaluation at the truth (a JIT "
               "compile each) on each cohort with an age-0 death; a few minutes; 0 calibrations")
    pins, mods = _pins(), _mods()
    lib = mods["library.load"].Library(root=ADP.library_root(LIBRARY, TESTER))
    byte_ids, win_ids = PANEL["loader"]["byte_compared"], PANEL["loader"]["refused_windowed"]
    truths = lib.truths(byte_ids)
    out_b, out_w = {}, {}
    for c in byte_ids:
        pins.check_file(pins.data["arrays"][c]["path"])
        o = lib.observation(c)
        a_ref, e_ref = np.asarray(list(o.ages), float), np.asarray(list(o.events), int)
        coh = cohort_from_observation(o)
        a, e = np.asarray(coh.ages, float), np.asarray(coh.events, int)
        row = {"n": int(a_ref.size), "delta": [o.delta, coh.delta], "window": o.window,
               "ages_bytes_equal": a.tobytes() == a_ref.tobytes(), "events_bytes_equal": e.tobytes() == e_ref.tobytes(),
               "age0_deaths": int(((a_ref == 0) & (e_ref == 1)).sum())}
        if row["age0_deaths"]:
            name = "profile_jax2_h" if c.startswith("hazard-") else "profile_jax2"
            t = truths[c]
            theta = types.SimpleNamespace(eta=t["eta"], beta=t["beta"], epsilon=t["epsilon"], kappa=t["kappa"],
                                          h_ext=t.get("external_hazard"))
            inp = PJ.fit_inputs(coh, name)
            term = A3._jax2_terms(coh, theta, name)["term"]
            ti, ei = np.asarray(inp["t"], float), np.asarray(inp["e"], int)
            mask = (ti == 0) & (ei == 1)
            age0 = term[mask] if term.size == ti.size else np.array([])
            row.update({"calibrator": name, "truth": t, "age0_deaths_in_fit_inputs": int(mask.sum()),
                        "age0_terms": [float(x) for x in age0], "log_floor": fj._LOG_FLOOR,
                        "age0_terms_finite_above_floor": bool(age0.size and np.all(np.isfinite(age0))
                                                               and np.all(age0 > fj._LOG_FLOOR)),
                        "other_terms_min": float(np.min(term[~mask])) if term.size == ti.size else None})
        out_b[c] = row
    for c in win_ids:
        pins.check_file(pins.data["arrays"][c]["path"])
        o = lib.observation(c)
        try:
            cohort_from_observation(o)
            out_w[c] = {"refused": False, "window": o.window}
        except ADP.UnsupportedOptionError as ex:
            out_w[c] = {"refused": True, "window": o.window, "message": str(ex),
                        "names_window": "window" in str(ex) and repr(o.window) in str(ex)}
    with_age0 = [c for c, r in out_b.items() if r["age0_deaths"]]
    ok = (all(r["ages_bytes_equal"] and r["events_bytes_equal"] and r["window"] is None for r in out_b.values())
          and len(with_age0) == len(byte_ids) - 2
          and all(out_b[c]["age0_terms_finite_above_floor"] for c in with_age0)
          and all(r["refused"] and r["names_window"] for r in out_w.values()))
    run.write({"status": "PASS" if ok else "FAIL", "byte_compared": out_b, "refused_windowed": out_w,
               "cohorts_with_age0_deaths": with_age0,
               "age0_rule": "the 10 panel cohorts with an age-0 death on v5 plus the 2 bytes cohorts (research R10)",
               "provenance": provenance(pins)})
    if not ok:
        sys.exit("FAIL: see tests/acceptance/results/loader_003e.json")



# ------------------------------------------------------------------------------------------- age0-bytes (SC-007)
def _arrays_sha(ages, events):
    import numpy as np
    return hashlib.sha256(np.asarray(ages, float).tobytes() + np.asarray(events, int).tobytes()).hexdigest()


def _age0_bytes_child(path):
    """In a fresh process with no pinned srbench: the J port and its frozen oracle side by side (tests/parity's
    pattern), value and value+gradient bytes at the truth on each cohort's v5 arrays. Prints JSON."""
    import types
    import numpy as np
    import jax2_cases as C
    import oracle_jax2 as OJ
    from parity_checks import bytes_diff
    problems = OJ.preflight()
    if problems:
        print(json.dumps({"applicable": False, "reason": problems}))
        return
    port, orc = C.port_modules(), C.oracle_modules()
    cases = json.load(open(path))
    out, planted = {}, None
    for c, d in cases.items():
        obs = types.SimpleNamespace(cohort_id=c, ages=tuple(d["ages"]), events=tuple(d["events"]), delta=d["delta"],
                                    window=None)
        name = "jax2_h" if c.startswith("hazard-") else "jax2"
        po, pi = C.objective(port, obs, name)
        oo, oi = C.objective(orc, obs, name)
        t = d["truth"]
        z = C.z_of(t, t["external_hazard"] if name == "jax2_h" else None)
        diffs = []
        a, b = np.asarray(po.f(z)), np.asarray(oo.f(z))
        diffs += bytes_diff(a, b, "value")
        (va, ga), (vb, gb) = po.vg(z), oo.vg(z)
        diffs += bytes_diff(np.asarray(va), np.asarray(vb), "vg.value")
        diffs += bytes_diff(np.asarray(ga), np.asarray(gb), "vg.grad")
        if planted is None:
            planted = bytes_diff(np.nextafter(a, np.inf), a, "value") != []
        out[c] = {"name": name, "arrays_sha256": _arrays_sha(obs.ages, obs.events), "grid_equal": pi == oi,
                  "bytes_equal": diffs == [], "diffs": [str(x) for x in diffs[:5]], "value": float(a)}
    print(json.dumps({"applicable": True, "cohorts": out, "planted_one_ulp_caught": planted}))


def age0_bytes(args):
    """SC-007 (optional): value+grad bytes of the J port (and _h on the hazard cohorts) against the frozen oracle at
    the truth on the 12 v5 arrays. The arrays come from the pinned library here; the comparison runs in a fresh
    process that never imports the pinned srbench (the oracle brings its own snapshot), and the arrays' sha256 is
    checked on both sides."""
    run = _run("age0-bytes", args, "12 arrays; one fresh process loading the jax2 oracle and the port; 2 x 12 "
               "objective builds and value+grad evaluations at the truth; a few minutes; 0 calibrations")
    import numpy as np
    pins, mods = _pins(), _mods()
    lib = mods["library.load"].Library(root=ADP.library_root(LIBRARY, TESTER))
    ids = PANEL["loader"]["byte_compared"]
    truths = lib.truths(ids)
    cases, sent = {}, {}
    for c in ids:
        pins.check_file(pins.data["arrays"][c]["path"])
        o = lib.observation(c)
        a, e = [float(x) for x in o.ages], [int(x) for x in o.events]
        cases[c] = {"ages": a, "events": e, "delta": o.delta, "truth": truths[c]}
        sent[c] = _arrays_sha(a, e)
    with tempfile.TemporaryDirectory(prefix="sraging003e_") as tmp:
        path = os.path.join(tmp, "cases.json")
        json.dump(cases, open(path, "w"))
        code = "import sys; import acceptance_003e as A; A._age0_bytes_child(sys.argv[1])"
        p = subprocess.run([sys.executable, "-c", code, path], capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "PYTHONPYCACHEPREFIX": os.path.join(REPO, ".cache", "pycache"),
                                "PYTHONPATH": os.pathsep.join([os.path.join(REPO, "tools"), os.path.join(REPO, "src")])})
    if p.returncode != 0:
        sys.exit(f"FAIL: the side-by-side process failed: {p.stderr[-2000:]}")
    res = json.loads(p.stdout.strip().split("\n")[-1])
    if not res["applicable"]:
        run.write({"status": "NOT APPLICABLE", "reason": res["reason"], "provenance": provenance(pins)})
        return
    got = res["cohorts"]
    same_arrays = all(got[c]["arrays_sha256"] == sent[c] for c in ids)
    ok = (same_arrays and res["planted_one_ulp_caught"]
          and all(got[c]["bytes_equal"] and got[c]["grid_equal"] for c in ids))
    run.write({"status": "PASS" if ok else "FAIL", "optional": True, "cohorts": got,
               "arrays_identical_across_processes": same_arrays,
               "planted_one_ulp_caught": res["planted_one_ulp_caught"],
               "bytes_equal": sum(got[c]["bytes_equal"] for c in ids), "of": len(ids),
               "theta": "the library truth (with its external hazard for the hazard cohorts, under jax2_h)",
               "provenance": provenance(pins)})
    if not ok:
        sys.exit(f"FAIL: {[c for c in ids if not got[c]['bytes_equal']]}")



# -------------------------------------------------------------------------------------------- unchanged (SC-008)
#: NOT APPLICABLE skips that are superseded by design, never a gap: {test id suffix: why}
SUPERSEDED = {
    "test_results_002::test_claims_001_rerun_is_identical_and_valid_for_this_branch":
        "spec 002's claims re-run; once spec 003 changed the fit path it is NOT APPLICABLE for good, and spec 003's "
        "re-run (results/claims-001_003.json, test_results_003.py) is the check, as its own comment rules",
}

#: every other skip or xfail the groups may show, each with its recorded reason (sr-naveh 003e finding 15)
RECORDED_SKIPS = {
    "test_results_003::test_coverage_no_combination_loses_on_a_gated_arm":
        "xfail: spec 003 SC-005 fails as recorded (eta/eps on censored cohorts; Naveh 2026-10-10, [coverage_result])",
}

UNCHANGED_GROUPS = {
    "unit": ["tests/unit"],                                   # includes 003d's result and rule tests
    "parity": ["tests/parity"],                               # side by side with the frozen oracles
    "results_001_002_003": ["tests/acceptance/test_results.py", "tests/acceptance/test_results_002.py",
                            "tests/acceptance/test_results_003.py"],
}


def unchanged(args):
    """SC-008: nothing earlier moved -- the unit suite, the parity suite and the result tests that pin specs 001-003d,
    each run by pytest in its own process at the committed results. Every group must have 0 failures and 0 errors;
    skips are recorded with their reasons."""
    import time
    import xml.etree.ElementTree as ET
    run = _run("unchanged", args, "pytest in 3 processes: the unit suite (~2 min), the parity suite and the 001-003 "
               "result tests; several minutes; 0 calibrations")
    out = {}
    with tempfile.TemporaryDirectory(prefix="sraging003e_") as tmp:
        for g, paths in UNCHANGED_GROUPS.items():
            xml = os.path.join(tmp, f"{g}.xml")
            t0 = time.time()
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}",
                                *paths], capture_output=True, text=True, cwd=REPO)
            root = ET.parse(xml).getroot()
            suites = [root] if root.tag == "testsuite" else list(root)
            c = {k: sum(int(x.get(k, 0)) for x in suites) for k in ("tests", "failures", "errors", "skipped")}
            listed = lambda tag: [(f"{tc.get('classname')}::{tc.get('name')}", (el.get("message") or "")[:300])  # noqa: E731
                                  for tc in root.iter("testcase") for el in tc.findall(tag)]
            out[g] = {**c, "passed": c["tests"] - c["failures"] - c["errors"] - c["skipped"],
                      "returncode": p.returncode, "summary": (p.stdout.strip().split("\n") or [""])[-1],
                      "wall_s": round(time.time() - t0, 1), "paths": paths, "skipped_list": listed("skipped"),
                      "failed_list": listed("failure") + listed("error")}
    ok, unlisted = unchanged_verdict(out)
    run.write({"status": "PASS" if ok else "FAIL", "groups": out, "skips_unlisted": unlisted,
               "superseded": SUPERSEDED, "recorded_skips": RECORDED_SKIPS, "provenance": provenance(_pins())})
    if not ok:
        sys.exit(f"FAIL: {[(g, v['failed_list'][:3]) for g, v in out.items() if v['failures'] or v['errors']]}; "
                 f"skips not named: {unlisted}")


def unchanged_verdict(groups: dict):
    """(ok, unlisted): every group has 0 failures and 0 errors, and pytest's exit is 0 -- or 1 only because of NOT
    APPLICABLE skips (the conftest's rule: never a pass). Every skip or xfail is named: a NOT APPLICABLE one on
    SUPERSEDED, any other on RECORDED_SKIPS."""
    unlisted = []
    ok = True
    for g, v in groups.items():
        na = [t for t, msg in v["skipped_list"] if "NOT APPLICABLE" in msg]
        plain = [t for t, msg in v["skipped_list"] if "NOT APPLICABLE" not in msg]
        bad = ([t for t in na if not any(t.endswith(k) for k in SUPERSEDED)]
               + [t for t in plain if not any(t.endswith(k) for k in RECORDED_SKIPS)])
        unlisted += [f"{g}: {t}" for t in bad]
        if v["failures"] or v["errors"] or bad or v["returncode"] not in ((0, 1) if na else (0,)):
            ok = False
    return ok, unlisted



# ------------------------------------------------------------------------------------------- provenance (SC-009)
def provenance_problems(name: str, d: dict) -> list[str]:
    """What a 003e result file lacks of FR-011's provenance (SC-009), and whether a clean run wrote it. `diff` ran
    before the pins existed (T006's order), and `pin` records the pins' sha256 at its top level; every other result
    carries it in its provenance."""
    out = []
    if d.get("status_for_acceptance") != "acceptance":
        out.append(f"status_for_acceptance is {d.get('status_for_acceptance')!r}, not 'acceptance'")
    p = d.get("provenance") or {}
    pins = json.load(open(PINS_PATH))
    if p.get("library") != LIBRARY:
        out.append(f"library is {p.get('library')!r}, not {LIBRARY}")
    if p.get("library_manifest_sha256") != pins["libraries"][LIBRARY]["sha256"]:
        out.append("the library manifest's sha256 is missing or not the pinned one")
    if p.get("tester_clone_head") != PANEL["tester"]["commit"] or p.get("tester_clone_dirty") is not False:
        out.append("the tester commit is missing, not the pinned one, or the clone was dirty")
    if p.get("runs_sha256") != {r: v["sha256"] for r, v in pins["runs"].items()}:
        out.append("the reference run manifests' sha256 are missing or not the pinned ones")
    if not all((p.get("host") or {}).get(k) for k in ("node", "machine", "cpu")):
        out.append("the host is missing")
    pins_sha = sha256_file(PINS_PATH)
    if name == "pin":
        if d.get("pins_sha256") != pins_sha:
            out.append("the pins' sha256 is missing or not the committed pins'")
    elif name != "diff" and p.get("pins_sha256") != pins_sha:
        out.append("the pins' sha256 is missing or not the committed pins'")
    return out

PHASES = {"diff": diff, "pin": pin, "known-answer": known_answer, "self-pair": self_pair, "refusals": refusals, "v033": v033, "jcross": jcross, "loader": loader, "age0-bytes": age0_bytes, "unchanged": unchanged}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--repin", action="store_true", help="pin: accept tester data that differ from the committed pins")
    args = ap.parse_args(argv)
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
