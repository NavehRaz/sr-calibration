"""Spec 003b's acceptance phases (specs/003b-jax2-101-port/contracts/acceptance.md). Mac only; nothing runs on WEXAC,
and no v6 reference record is read.

    .venv/bin/python tools/acceptance_003b.py verify          # the handoff on disk (FR-001; no fit)
    .venv/bin/python tools/acceptance_003b.py census          # 1.0.1's grid vs jax2's census, 2,798 rows (no fit)
    .venv/bin/python tools/acceptance_003b.py identity-off    # 1.0.1 switches off == the 1.0.0 port, G1's 11 lines
    .venv/bin/python tools/acceptance_003b.py parity          # the port == the 1.0.1 oracle, side by side
    .venv/bin/python tools/acceptance_003b.py v100-unchanged  # profile_jax2@1.0.0 still bit-equal on ITP+age0
    .venv/bin/python tools/acceptance_003b.py lev-measure     # pilot: g0 and the bound tolerance (then STOP)
    .venv/bin/python tools/acceptance_003b.py itp             # AC1/AC2 on 1.0.1
    .venv/bin/python tools/acceptance_003b.py ac11            # the pre-registered nulls and planted panel
    .venv/bin/python tools/acceptance_003b.py bytes           # the parity suite at the frozen src, the bytes item
    .venv/bin/python tools/acceptance_003b.py readme-example  # the README's jax2 example under 1.0.1
    .venv/bin/python tools/acceptance_003b.py unchanged       # the suites and the 001-003f results tests

Every phase runs through spec 003's `acceptance_003.Run` (suffix "003b"): a clean committed tree, its price printed
first, its result in tests/acceptance/results/<phase>_003b.json with the provenance of FR-014.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")

PANEL_PATH = "tests/acceptance/panel_003b.toml"
RESULTS_DIR = os.path.join(REPO, "tests", "acceptance", "results")
BYTE_HOST = "MC06268844"            # the host jax2's freeze identity ran on; byte identity is claimed here only
BYTE_PHASES = ("identity-off", "parity", "v100-unchanged", "bytes")
DROPBOX = os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset"   # release 0.1.1: the lab's checkouts
J2_REPO = os.environ.get("SRAGING_J2F_ROOT", os.path.join(DROPBOX, "sr_calibration_tester_jax2_freeze"))
FREEZE = os.path.join(J2_REPO, "reports", "freeze101")
J2_REL = os.path.join(DROPBOX, "sr_calibration_tester_jax2_rel", "reports", "rel101")
CENSUS = os.path.join(J2_REL, "ADD3", "census_dev5", "rows_v6.jsonl")
#: the files the panel pins by sha256 (read-only; the reports are pinned, the code is read from git objects)
PINNED_FILES = {
    "handoff": os.path.join(FREEZE, "HANDOFF.txt"),
    "switches_tsv": os.path.join(FREEZE, "snapshot", "SWITCHES.tsv"),
    "srbench_files_sha256_file": os.path.join(FREEZE, "snapshot", "SRBENCH_FILES.sha256"),
    "defect_register": os.path.join(FREEZE, "defects", "DEFECT_REGISTER.txt"),
    "srk_record": os.path.join(FREEZE, "identity", "fits", "realfull-real-Denmark_M_1900_homo-c1-s1__jax2__on101.json"),
    "census_rows_v6": CENSUS,
    "g1_cohorts": os.path.join(J2_REL, "G1", "cohorts.txt"),
    "library_v6_manifest": None,            # the tester's library v6 (oracle_jax2_101.LIBRARY_V6), set below
}
SWITCHES_COPY = os.path.join(REPO, "tests", "data", "jax2_101_SWITCHES.tsv")
REGISTERED = ("mle_profile_jax2", "mle_profile_jax2_h")
N_LISTING, N_SRBENCH, N_SRK = 39, 46, 18           # handoff section 2 and 2.1; the identity record's srk_sha


def panel() -> dict:
    return tomllib.load(open(os.path.join(REPO, PANEL_PATH), "rb"))


def _git(*a, repo=REPO):
    return subprocess.run(["git", "--no-optional-locks", "-C", repo, *a], capture_output=True, text=True).stdout.strip()


def cpu_model() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return platform.processor()


def provenance(name="profile_jax2", version="1.0.1", library=None, manifest_sha256=None) -> dict:
    """FR-014: the calibrator, its version and lock sha256, the host, the CPU, the library and its manifest (library
    cohorts), and the commit of the panel file."""
    import hashlib
    snap = {"1.0.0": "jax2_3bc5775", "1.0.1": "jax2_6884fc8"}.get(version)
    lock_path = os.path.join(REPO, "tests", "oracle", snap or "-", "mle_profile_jax2", "VERSION_LOCK.json")
    lock = hashlib.sha256(open(lock_path, "rb").read()).hexdigest() if os.path.exists(lock_path) else None
    return {"calibrator": name, "version": version, "lock_sha256": lock, "host": platform.node(),
            "machine": platform.machine(), "cpu": cpu_model(), "library": library,
            "manifest_sha256": manifest_sha256, "panel_commit": _git("log", "-1", "--format=%H", "--", PANEL_PATH)}


def require_byte_host(phase):
    if phase in BYTE_PHASES and platform.node() != BYTE_HOST:
        sys.exit(f"REFUSED: {phase} compares bytes, which hold per host only (handoff issue 1); this is "
                 f"{platform.node()!r}, not {BYTE_HOST!r}")


def _run(phase, args, cost):
    import acceptance_003 as A
    return A.Run(phase, manifests=[PANEL_PATH], dev=args.dev, suffix="003b", cost=cost)


def _todo(name):
    def phase(args):
        sys.exit(f"{name}: not implemented yet (specs/003b-jax2-101-port/tasks.md)")
    return phase


def _sha(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _file_sha(p):
    return _sha(open(p, "rb").read()) if p and os.path.exists(p) else None


def pinned_files() -> dict:
    import oracle_jax2_101 as OJ1
    return dict(PINNED_FILES, library_v6_manifest=os.path.join(OJ1.LIBRARY_V6, "manifest.json"))


# ------------------------------------------------------------------------------------------------- verify (FR-001)
def verify_checks(pins: dict, *, repo: str = J2_REPO, files: dict | None = None) -> list[dict]:
    """FR-001 / SC-001: every pinned item, from git objects at the tag's commit (the code) and by sha256 on disk (the
    reports), against the panel's pins; the two snapshots are compared byte for byte with what they were built from.
    Read-only. One row per check: {check, ok, detail}; a git failure fails its check, never the others."""
    import json
    import oracle_jax2_101 as OJ1
    import oracle_v033 as V
    import snapshot_jax2 as SJ
    files = pinned_files() if files is None else files
    out = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except (SJ.Refused, OSError, ValueError, KeyError) as ex:
            ok, detail = False, f"{type(ex).__name__}: {ex}"
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    commit, tag = pins["commit"], pins["tag"]

    def tag_check():
        obj, com = SJ._gitq(repo, "rev-parse", tag), SJ._gitq(repo, "rev-parse", f"{tag}^{{commit}}")
        return obj == pins["tag_object"] and com == commit, f"{tag} -> tag object {obj[:8]}, commit {com[:8]}"

    def lock_check():
        b = SJ._blob(repo, commit, f"{SJ.METHOD}/{SJ.LOCK}")
        lock = json.loads(b)
        snap = open(os.path.join(OJ1.SNAP3, "mle_profile_jax2", SJ.LOCK), "rb").read()
        vers = {n: lock["names"][n]["version"] for n in REGISTERED}
        pinned = all(lock["names"][n]["sha256"] == dict(pins["runtime"]) for n in REGISTERED)
        ok = _sha(b) == pins["version_lock_sha256"] and snap == b and set(vers.values()) == {"1.0.1"} and pinned
        return ok, f"sha256 {_sha(b)[:8]}; snapshot equal {snap == b}; versions {vers}; pins the 4 files {pinned}"

    def runtime_check(f, want):
        def fn():
            b = SJ._blob(repo, commit, f"{SJ.METHOD}/{f}")
            snap = open(os.path.join(OJ1.SNAP3, "mle_profile_jax2", f), "rb").read()
            return _sha(b) == want and snap == b, f"sha256 {_sha(b)[:8]}; snapshot equal {snap == b}"
        return fn

    def listing_check():
        rows = [f"{_sha(SJ._blob(repo, commit, ln.split(chr(9))[1]))}  {ln.split(chr(9))[1]}"
                for ln in SJ._gitq(repo, "ls-tree", "-r", commit, SJ.METHOD).splitlines()]
        digest = _sha(("\n".join(rows) + "\n").encode())
        man = json.load(open(os.path.join(OJ1.SNAP3, "MANIFEST.json")))
        ok = digest == pins["listing_digest"] and len(rows) == N_LISTING and man["listing_digest"] == digest
        return ok, f"{len(rows)} files, digest {digest[:8]}; the snapshot records {man['listing_digest'][:8]}"

    def srbench_check():
        raw = open(files["srbench_files_sha256_file"], "rb").read()
        lines = [ln for ln in raw.decode().splitlines() if ln and not ln.startswith("#")]
        want = dict(reversed(ln.split("  ", 1)) for ln in lines)
        tree = SJ._gitq(repo, "rev-parse", f"{commit}:srbench")
        have = [ln.split("\t")[1] for ln in SJ._gitq(repo, "ls-tree", "-r", commit, "srbench").splitlines()]
        git_ok = [p for p in have if p in want and _sha(SJ._blob(repo, commit, p)) == want[p]]
        snap_ok = [p for p in have if _file_sha(os.path.join(OJ1.SRB, p)) == want.get(p)]
        ok = (_sha(raw) == pins["srbench_files_sha256_file"] and _sha(("\n".join(lines) + "\n").encode())
              == pins["srbench_listing_digest"] and tree == pins["srbench_tree"] and set(have) == set(want)
              and len(git_ok) == len(snap_ok) == len(want) == N_SRBENCH)
        return ok, (f"tree {tree[:8]}; {len(git_ok)}/{len(want)} at the commit and {len(snap_ok)}/{len(want)} in "
                    f"tests/oracle/tester_f94ec2b match the sums file")

    def srk_check():
        rec = files["srk_record"]
        srk = SJ._find(json.load(open(rec)), "srk_sha")["files"]
        same = [r for r, h in srk.items() if _file_sha(os.path.join(V.SNAP_SRK_ROOT, "SRK", r)) == h]
        ok = _file_sha(rec) == pins["srk_record"] and len(same) == len(srk) == N_SRK
        return ok, f"{len(same)}/{len(srk)} SRK files a 1.0.1 fit loads equal tests/oracle/v033's"

    check("tag", tag_check)
    check("version_lock", lock_check)
    for f, want in pins["runtime"].items():
        check(f"runtime:{f}", runtime_check(f, want))
    check("listing_digest", listing_check)
    check("srbench", srbench_check)
    check("srk", srk_check)
    for key in sorted(files):
        if key in ("srbench_files_sha256_file", "srk_record"):
            continue
        got = _file_sha(files[key])
        check(f"file:{key}", lambda got=got, key=key: (got == pins.get(key), f"sha256 {(got or 'missing')[:8]}"))
    check("file:switches_tsv(copy)", lambda: (_file_sha(SWITCHES_COPY) == pins["switches_tsv"],
                                              "tests/data/jax2_101_SWITCHES.tsv"))
    return out


def phase_verify(args):
    P = panel()["pins"]
    run = _run("verify", args, "< 0.1 Mac CPU-h; no fit (git objects and sha256s)")
    rows = verify_checks(P)
    bad = [r["check"] for r in rows if not r["ok"]]
    for r in rows:
        print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['check']}: {r['detail']}", flush=True)
    run.write({"status": "pass" if not bad else "fail", "failed": bad, "n_checks": len(rows), "checks": rows,
               "provenance": provenance()})
    if bad:
        sys.exit(1)


# ------------------------------------------------------------------------------------------------- census (FR-006)
#: compared bit for bit: the five the census records for the rule (tasks T014) and L, dl and n beside them
CENSUS_FIELDS = ("cv", "cv_c", "in5", "N5", "dt5", "L", "dl", "n")
CENSUS_EXPECT = {"rows": 2798, "J": 2408, "H": 390}


def _kind(x) -> str:
    n = type(x).__name__
    if n in ("bool", "bool_"):
        return "bool"
    if isinstance(x, float) or n.startswith("float"):
        return "float"
    if isinstance(x, int) or n.startswith(("int", "uint")):
        return "int"
    return n


def bit_equal(a, b) -> bool:
    """Bit-for-bit equality of two census values: the same kind (bool, int, float; numpy's or Python's), floats by
    their IEEE bytes (NaN equal to NaN, -0.0 not 0.0), None only to None."""
    import struct
    if a is None or b is None:
        return a is None and b is None
    ka, kb = _kind(a), _kind(b)
    if ka != kb:
        return False
    if ka == "float":
        fa, fb = float(a), float(b)
        return (fa != fa and fb != fb) or struct.pack("<d", fa) == struct.pack("<d", fb)
    if ka in ("bool", "int"):
        return int(a) == int(b)
    return a == b


def census_rows(path=CENSUS, want_sha256=None) -> list[dict]:
    import json
    raw = open(path, "rb").read()
    if want_sha256 is not None and _sha(raw) != want_sha256:
        raise SystemExit(f"REFUSED: {path} has sha256 {_sha(raw)[:12]}, not the panel's {want_sha256[:12]}")
    return [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]


def port_grid(obs, view: str) -> dict:
    """1.0.1's grid of the observation as `view` sees it (J: profile_jax2, window off; H: profile_jax2_h, window on),
    through the port's `fit_inputs` (its prepare and `_grid` statements, the class constants); no fit."""
    from acceptance_003_port import port_observation
    from sraging.calibrators import profile_jax2 as PJ
    name = "profile_jax2_h" if view == "H" else "profile_jax2"
    fi = PJ.fit_inputs(port_observation(obs), name, PJ.PORTS["1.0.1"])
    gr, df = fi["dt_floor"]["grid_rule"], fi["dt_floor"]
    return {"cv": gr["cv"], "cv_c": gr["cv_c"], "in5": gr["dt_in_scope"], "N5": fi["kw"]["N"], "dt5": fi["kw"]["dt"],
            "L": fi["L"], "dl": df["dl"], "n": int(fi["t"].size)}


def census_compare(rows: list[dict], grid_of) -> dict:
    """Each census row against `grid_of(cid, view)`: {rows, equal, differences, in_scope, table}. The table holds
    sraging's values (equal to jax2's on an equal row); a differing row lists both sides of each field."""
    table, diffs = [], []
    scope = {"jax2": {"J": 0, "H": 0}, "sraging": {"J": 0, "H": 0}}
    for r in rows:
        g = grid_of(r["cid"], r["view"])
        bad = {f: [r.get(f), g.get(f)] for f in CENSUS_FIELDS if not bit_equal(r.get(f), g.get(f))}
        table.append([r["cid"], r["view"], not bad] + [_plain(g.get(f)) for f in CENSUS_FIELDS])
        scope["jax2"][r["view"]] += bool(r.get("in5"))
        scope["sraging"][r["view"]] += bool(g.get("in5"))
        if bad:
            diffs.append({"cid": r["cid"], "view": r["view"], "fields": {f: [_plain(x) for x in v]
                                                                          for f, v in bad.items()}})
    views = {v: sum(1 for r in rows if r["view"] == v) for v in ("J", "H")}
    return {"rows": len(rows), "views": views, "equal": len(rows) - len(diffs), "differences": diffs,
            "in_scope": scope, "columns": ["cid", "view", "equal", *CENSUS_FIELDS], "table": table}


def _plain(x):
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    if type(x).__name__ == "bool_":
        return bool(x)
    return int(x) if type(x).__name__.startswith(("int", "uint")) else float(x)


def real_grid_of():
    """`grid_of` on the tester's library v6 through the 1.0.1 oracle's srbench loader (this loads the 1.0.1 oracle
    into the process: one oracle per process)."""
    import sraging.compat.jax2_101  # noqa: F401  (configures the runtime before the oracle's fp_jax)
    import oracle_jax2_101 as OJ1
    OJ1.load()
    return lambda cid, view: port_grid(OJ1.observation_v6(cid), view)


def phase_census(args):
    P = panel()["pins"]
    run = _run("census", args, "< 0.3 Mac CPU-h; no fit (1.0.1's grid on 2,798 rows)")
    rows = census_rows(CENSUS, P["census_rows_v6"])
    if args.only:
        rows = [r for r in rows if r["cid"] in set(args.only)]
    import oracle_jax2_101 as OJ1
    man = _file_sha(os.path.join(OJ1.LIBRARY_V6, "manifest.json"))
    if man != P["library_v6_manifest"]:
        sys.exit(f"[census] REFUSED: the v6 manifest is {str(man)[:12]}, not the panel's")
    if {r.get("lib") for r in rows} != {"v6"}:
        sys.exit("[census] REFUSED: the census holds rows of another library")
    try:
        grid_of = real_grid_of()
    except OJ1.NotApplicable as ex:
        run.write({"status": "not_applicable", "reasons": ex.problems})
        sys.exit(3)
    res = census_compare(rows, grid_of)
    full = not args.only
    counts_ok = not full or (res["rows"] == CENSUS_EXPECT["rows"] and res["views"] == {"J": CENSUS_EXPECT["J"],
                                                                                     "H": CENSUS_EXPECT["H"]}
                             and res["in_scope"]["sraging"] == res["in_scope"]["jax2"])
    ok = res["equal"] == res["rows"] and counts_ok
    print(f"[census] {res['equal']} of {res['rows']} rows equal; in scope J/H: sraging "
          f"{res['in_scope']['sraging']}, jax2 {res['in_scope']['jax2']}", flush=True)
    for d in res["differences"][:20]:
        print(f"  DIFFERS {d['cid']} {d['view']}: {d['fields']}", flush=True)
    run.write({"status": "pass" if ok else "fail", "full": full, "census_sha256": P["census_rows_v6"],
               "provenance": provenance(library="v6", manifest_sha256=man), **res})
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------------------------------------- parity (FR-005)
#: handoff 7.2's TIMING, dropped at any depth (= profile_jax2.TIMING_LEAVES)
TIMING = ("wall_s", "compile_s", "seed_s", "mle_s", "cpu_s", "refit_cpu_s", "fit_wall_s")
CS_TIMING = ("wall_clock_seconds", "cpu_seconds")                   # the ClaimSet's top level
J2_SHORT = {"profile_jax2": "jax2", "profile_jax2_h": "jax2_h"}
J2_FULL = {"profile_jax2": "mle_profile_jax2", "profile_jax2_h": "mle_profile_jax2_h"}
SRK_NOTE = ("diagnostics.srk_sha: .root is a SUT-location key (handoff 7.2 'Any level'); .files compared file by file "
            "with the 18 sha256 tests/oracle/jax2_6884fc8 cites; absent on the port, whose run_fit drops the SRK "
            "fingerprint (one of the 10 guard deletions tools/port_jax2_101.py asserts)")
EXCLUDED = {
    "run_fit": ["TIMING at any depth: " + ", ".join(TIMING), SRK_NOTE],
    "fitresult": ["TIMING at any depth: " + ", ".join(TIMING), SRK_NOTE,
                  "CS-TIMING: the ClaimSet's wall_clock_seconds and cpu_seconds (top level; not in the view)",
                  "code_provenance (SUT location; not in the view)", "diagnostics.code (SUT location)",
                  "sraging's own diagnostics, not ClaimSet fields (not compared): "],
    "mle_only": ["TIMING at any depth: " + ", ".join(TIMING)],
}


def sraging_own() -> tuple:
    """The FitResult's diagnostics that are sraging's, not the method's: SRAGING_DIAGNOSTICS and the keys only
    sraging writes (the no-external-mortality assumption, L_SR and the box in L_SR units)."""
    from sraging.result import SRAGING_DIAGNOSTICS
    return tuple(SRAGING_DIAGNOSTICS) + ("assumptions", "L_SR", "box_lsr")


def cited_srk() -> dict:
    """{SRK-relative path: sha256} the 1.0.1 snapshot cites (the 18 files a 1.0.1 fit loads)."""
    import json
    import oracle_jax2_101 as OJ1
    man = json.load(open(os.path.join(OJ1.SNAP3, "MANIFEST.json")))
    return {c["path"].split("SRK/", 1)[1]: c["sha256"] for c in man["cited"]}


def _strip(x):
    """`x` without handoff 7.2's TIMING keys, at any depth."""
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, (list, tuple)):
        return [_strip(v) for v in x]
    return x


def _take_srk(tree: dict, side: str, cited: dict, problems: list, required: bool) -> dict:
    """`tree` (a result or a view, with `diagnostics`) without diagnostics.srk_sha; the fingerprint's files are
    checked against `cited` into `problems`. `required`: its absence is a problem (the oracle side)."""
    diag = tree.get("diagnostics")
    if not isinstance(diag, dict) or "srk_sha" not in diag:
        if required:
            problems.append(f"{side}: no diagnostics/srk_sha")
        return tree
    diag = dict(diag)
    files = dict((diag.pop("srk_sha") or {}).get("files") or {})
    for rel in sorted(set(files) | set(cited)):
        if files.get(rel) != cited.get(rel):
            problems.append(f"{side}: diagnostics/srk_sha/files/{rel} {str(files.get(rel))[:12]} != cited "
                            f"{str(cited.get(rel))[:12]}")
    return dict(tree, diagnostics=diag)


def run_fit_diff(port: tuple, oracle: tuple, cited: dict) -> dict:
    """Level run_fit (L0 at the class constants): (result, claims, self-consistency block) of the port against the
    oracle's, TIMING dropped at any depth, srk_sha as SRK_NOTE. Returns {equal, diff_paths, srk}."""
    from parity_checks import tree_diff
    srk = []
    pres = _take_srk(dict(port[0]), "port", cited, srk, required=False)
    ores = _take_srk(dict(oracle[0]), "oracle", cited, srk, required=True)
    diffs = (tree_diff(_strip(pres), _strip(ores), "/result") + tree_diff(_strip(port[1]), _strip(oracle[1]), "/claims")
             + tree_diff(_strip(port[2]), _strip(oracle[2]), "/sc_block"))
    return {"equal": not diffs and not srk, "diff_paths": diffs, "srk": srk}


def fitresult_view(fr, name: str) -> dict:
    """The FitResult as the ClaimSet's fields: cohort id, the method's name, status, the claims' reference fields and
    the method's diagnostics (sraging's own keys left out, and recorded)."""
    own = sraging_own()
    return {"cohort_id": fr.cohort["name"], "method_name": J2_FULL[name], "status": fr.status,
            "claims": [c.reference_dict() for c in fr.claims],
            "diagnostics": {k: v for k, v in fr.diagnostics.items() if k not in own}}


def claimset_view(cs) -> dict:
    """A frozen 1.0.1 ClaimSet's compared fields: the top-level CS-TIMING keys and code_provenance are left out, and
    diagnostics.code (SUT location)."""
    from parity_checks import CLAIM_FIELDS
    return {"cohort_id": cs.cohort_id, "method_name": cs.method_name, "status": cs.status,
            "claims": [{k: getattr(c, k) for k in CLAIM_FIELDS} for c in cs.claims],
            "diagnostics": {k: v for k, v in (cs.diagnostics or {}).items() if k != "code"}}


def fitresult_diff(fview: dict, cview: dict, cited: dict) -> dict:
    """Level fitresult (L0): the FitResult's view against the frozen ClaimSet's, TIMING dropped at any depth, srk_sha
    as SRK_NOTE; diagnostics.estimate compared. Returns {equal, diff_paths, srk}."""
    from parity_checks import tree_diff
    srk = []
    f = _take_srk(fview, "port", cited, srk, required=False)
    c = _take_srk(cview, "oracle", cited, srk, required=True)
    diffs = tree_diff(_strip(f), _strip(c), "")
    return {"equal": not diffs and not srk, "diff_paths": diffs, "srk": srk}


def mechanisms_101(res: dict) -> dict:
    """What fired in one 1.0.1 fit: inner polish, bracket bisection, extent z stored, the basin re-fit, DT_RULE's
    scope (the panel picks one cohort per mechanism)."""
    dg = res.get("diagnostics") or {}
    ip, bb = dg.get("inner_polish") or {}, dg.get("bracket_bisect") or {}
    ext = ((dg.get("self_consistency") or {}).get("extents") or {})
    return {"inner_polish_improved": int(ip.get("n_improved") or 0), "inner_polish_solves": int(ip.get("n_solves") or 0),
            "bracket_bisect_nodes": int(bb.get("n_nodes") or 0),
            "extent_z": any("z_qmin" in (e or {}) for e in ext.values()),
            "basin_refit": bool((dg.get("basin_fix") or {}).get("fired")),
            "dt_in_scope": bool(((dg.get("dt_floor") or {}).get("grid_rule") or {}).get("dt_in_scope")),
            "hazard": "hazard" in dg}


def stage_timers(res: dict) -> dict:
    dg = res.get("diagnostics") or {}
    return {k: dg.get(k) for k in ("compile_s", "seed_s", "mle_s")} | {"wall_s": res.get("wall_s")}


def mle_only_kwargs(name: str, port) -> dict:
    """`estimator.fit`'s keywords for the MLE stage on `port`, from the port's own run_fit call (`run_kwargs`), as the
    1.0.1 oracle's `_mle_only` builds them on the frozen side; `stages` is added by the caller."""
    from sraging.calibrators import profile_jax2 as PJ
    kw = PJ.run_kwargs(port.classes[name])
    extra = {k: True for k in ("shadow", "exact_from_F", "self_consistent", "floor_early", "n_rule", "dt_rule",
                               "early_all", "bracket_bisect", "extent_z", "box_flag") if kw.get(k)}
    if kw.get("inner_polish_j"):
        extra["inner_polish"] = True
    return {"levels": (port.claims.LEVEL,), "kappa_max": port.claims.KAPPA_MAX, "hazard": kw["hazard"], **extra}


def port_mle_only(obs, name: str, port, stages=("mle",)) -> dict:
    """The port's MLE stage on `obs`: run_fit's two steps (prepare, estimator.fit) with `stages` added."""
    from sraging.calibrators import profile_jax2 as PJ
    kw = PJ.run_kwargs(port.classes[name])
    prep = port.observation_model.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=kw["window"],
                                          relabel_on=kw["relabel"])
    return port.estimator.fit(port.fp_jax, prep.ages, prep.events, delta=obs.delta, window=prep.window_eff,
                              stages=stages, **mle_only_kwargs(name, port))


def oracle_claimset(o, short: str, obs, res):
    """The frozen 1.0.1 adapter's own ClaimSet of the oracle's run_fit result: its `claimset` (estimate included),
    with a stand-in code provenance (a SUT-location key, excluded)."""
    import time
    from srbench.provenance import CodeProvenance
    prov = CodeProvenance(package="mle_profile_jax2", resolved_path="oracle", revision="6884fc8", dirty=False)
    return o.classes[short]().claimset(obs, res, prov, time.perf_counter(), time.process_time())


def prepared_sha256(obs, name: str, port) -> str:
    """The prepared data's digest as jax2's S3 driver records it (ADD1/cluster/common_rel.py `prepared_sha256`), on
    the port's `prepare` with the class's window and relabel: equal to a record's, it shows the pair runs on the data
    the record's mechanism fired on."""
    import hashlib
    import json
    from sraging.calibrators import profile_jax2 as PJ
    kw = PJ.run_kwargs(port.classes[name])
    prep = port.observation_model.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=kw["window"],
                                          relabel_on=kw["relabel"])
    payload = dict(ages=[repr(float(a)) for a in prep.ages], events=[int(x) for x in prep.events],
                   window_eff=None if prep.window_eff is None else [repr(float(x)) for x in prep.window_eff],
                   delta=None if obs.delta is None else repr(float(obs.delta)))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parity_worker(cid: str, name: str, mode: str, cited: dict, library: str = "v6") -> dict:
    """One pair, side by side in this process: the port and the 1.0.1 oracle on the same observation of `library`.
    "full": run_fit level, then the FitResult of the same port run against the frozen adapter's ClaimSet of the
    oracle's run (non-windowed cohorts: the public fit refuses windows). "mle_only": the MLE stage's result."""
    import time
    import traceback
    try:
        import sraging.compat.jax2_101  # noqa: F401  (configures the runtime before the oracle's fp_jax)
        import oracle_jax2_101 as OJ1
        from acceptance_003_port import port_observation
        from parity_checks import tree_diff
        from sraging import Cohort
        from sraging.calibrators import profile_jax2 as PJ
        o = OJ1.load()
        P1 = PJ.PORTS["1.0.1"]
        obs = OJ1.observation(cid, library)
        short = J2_SHORT[name]
        base = {"cohort_id": cid, "name": name, "mode": mode, "library": library,
                "prepared_sha256": prepared_sha256(obs, name, P1)}
        rows = []
        if mode == "mle_only":
            c0 = time.process_time()
            pres = port_mle_only(obs, name, P1)
            pc = time.process_time() - c0
            c0 = time.process_time()
            ores = OJ1._mle_only(o, obs, short, ("mle",))
            oc = time.process_time() - c0
            d = tree_diff(_strip(pres), _strip(ores), "/result")
            rows.append({**base, "level": "mle_only", "equal": not d, "diff_paths": d[:40], "n_diffs": len(d),
                         "excluded_keys": EXCLUDED["mle_only"], "port_cpu_s": pc, "oracle_cpu_s": oc,
                         "stage_timers": {"port": stage_timers(pres), "oracle": stage_timers(ores)},
                         "in_scope": mechanisms_101(pres)["dt_in_scope"], "mechanisms": mechanisms_101(pres)})
            return {"cohort_id": cid, "rows": rows}
        c0 = time.process_time()
        out = PJ.run_port(port_observation(obs), name, capture=True, port=P1)
        pc = time.process_time() - c0
        c0 = time.process_time()
        ores = o.run_fit(obs, **OJ1.fit_kwargs(short))
        oclaims, osc = o.final_claim_kwargs(ores, o.classes[short].SELF_CONSISTENT)
        oc = time.process_time() - c0
        lv = run_fit_diff((out["result"], out["claims"], out["sc_block"]), (ores, oclaims, osc), cited)
        mech = mechanisms_101(out["result"])
        common = {"port_cpu_s": pc, "oracle_cpu_s": oc, "in_scope": mech["dt_in_scope"], "mechanisms": mech,
                  "stage_timers": {"port": stage_timers(out["result"]), "oracle": stage_timers(ores)}}
        rows.append({**base, "level": "run_fit", "equal": lv["equal"], "diff_paths": lv["diff_paths"][:40],
                     "n_diffs": len(lv["diff_paths"]), "srk": lv["srk"], "excluded_keys": EXCLUDED["run_fit"],
                     **common})
        if obs.window is not None:
            rows.append({**base, "level": "fitresult", "equal": None, "status": "not_applicable",
                         "reason": "a windowed cohort: the public fit refuses windows (FR-007)"})
            return {"cohort_id": cid, "rows": rows}
        c0 = time.process_time()
        cohort = Cohort.from_arrays(list(obs.ages), list(obs.events), delta=obs.delta,
                                    continuous=obs.delta is None, name=cid)
        fr = PJ.fit_result(cohort, name, out, port=P1, combinations=False)
        fc = time.process_time() - c0
        cs = oracle_claimset(o, short, obs, ores)
        lf = fitresult_diff(fitresult_view(fr, name), claimset_view(cs), cited)
        own = sorted(k for k in fr.diagnostics if k in sraging_own())
        rows.append({**base, "level": "fitresult", "equal": lf["equal"], "diff_paths": lf["diff_paths"][:40],
                     "n_diffs": len(lf["diff_paths"]), "srk": lf["srk"],
                     "excluded_keys": EXCLUDED["fitresult"][:-1] + [EXCLUDED["fitresult"][-1] + ", ".join(own)],
                     "estimate_compared": "estimate" in fr.diagnostics and "estimate" in (cs.diagnostics or {}),
                     "fitresult_cpu_s": fc, **common})
        return {"cohort_id": cid, "rows": rows}
    except Exception as ex:  # noqa: BLE001 -- a crashed pair is a row, never a silent pass
        return {"cohort_id": cid, "name": name, "mode": mode, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-3000:]}


def _init101():
    """A parity child's set-up: the single-thread XLA and BLAS flags of spec 003's workers (acceptance_003_port's
    THREAD_ENV), the bytecode cache and the paths. Not spec 003's `_init`: that one loads the 1.0.0 oracle, and the
    1.0.1 oracle refuses to share a process with another srbench."""
    import acceptance_003_port as AP
    for k, v in AP.THREAD_ENV.items():
        os.environ[k] = v
    sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
    for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def is_difference(r: dict) -> bool:
    """A pair's result stops the phase: an error, or any row with a difference."""
    return "error" in r or any(row.get("equal") is False for row in r.get("rows", []))


def _report(i, n, p, r):
    what = "ERROR " + r["error"][:80] if "error" in r else ("DIFFERS" if is_difference(r) else "equal")
    print(f"  [{i}/{n}] {p['cid']} {p['name']} {p['mode']}: {what}", flush=True)


def run_pairs(pairs: list[dict], workers: int, cited: dict, *, worker=None, pool=True,
              stop_on_difference=True) -> tuple[list, bool]:
    """Each pair in its own spawned process (one oracle per process, and one pair per process). Returns (results,
    stopped): at the first pair with a difference, the pending pairs are cancelled (the running ones finish), unless
    `stop_on_difference` is False (a measurement phase). `pool=False` runs the pairs in this process, in order (the
    unit tests' fake workers)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    worker = worker or parity_worker
    out, stopped = [], False
    if not pool:
        for p in pairs:
            r = worker(p["cid"], p["name"], p["mode"], cited, p.get("library", "v6"))
            out.append(r)
            _report(len(out), len(pairs), p, r)
            if is_difference(r) and stop_on_difference:
                return out, True
        return out, False
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init101,
                                max_tasks_per_child=1) as ex:
        futs = {ex.submit(worker, p["cid"], p["name"], p["mode"], cited, p.get("library", "v6")): p for p in pairs}
        for f in cf.as_completed(futs):
            try:
                r = f.result()
            except cf.CancelledError:
                continue
            except Exception as e:  # noqa: BLE001
                r = {"cohort_id": futs[f]["cid"], "error": f"crash: {type(e).__name__}: {e}"}
            out.append(r)
            _report(len(out), len(pairs), futs[f], r)
            if is_difference(r) and stop_on_difference and not stopped:
                stopped = True
                for g in futs:
                    g.cancel()
    return out, stopped


def phase_parity(args):
    import json  # noqa: F401
    pan = panel()
    pairs = list((pan.get("parity") or {}).get("pairs") or [])
    if args.only:
        pairs = [p for p in pairs if p["cid"] in set(args.only)]
    if not pairs:
        sys.exit("[parity] REFUSED: PANEL [parity] has no pairs (tasks T016)")
    price = sum(float(p.get("price_cpu_s") or 0) for p in pairs) / 3600
    run = _run("parity", args, f"~{price:.1f} Mac CPU-h by the panel's stored timers ({len(pairs)} pairs; one process "
                               f"per pair, {args.workers} at a time)")
    import oracle_jax2_101 as OJ1
    problems = OJ1.preflight()
    if problems:
        run.write({"status": "not_applicable", "reasons": problems})
        sys.exit(3)
    pins = pan["pins"]
    for lib in sorted({p.get("library", "v6") for p in pairs}):
        man = _file_sha(os.path.join(OJ1.LIBRARIES[lib], "manifest.json"))
        if man != pins[f"library_{lib}_manifest"]:
            sys.exit(f"[parity] REFUSED: the {lib} manifest is {str(man)[:12]}, not the panel's")
    for p in pairs:
        got = _file_sha(os.path.join(OJ1.LIBRARIES[p.get("library", "v6")], "cohorts", f"{p['cid']}.npz"))
        if got != p["sha256"]:
            sys.exit(f"[parity] REFUSED: {p['cid']}'s array is {str(got)[:12]}, not the panel's {p['sha256'][:12]}")
    results, stopped = run_pairs(pairs, args.workers, cited_srk())
    rows = [row for r in results for row in r.get("rows", [])]
    want_prep = {(p["cid"], p["name"]): p.get("record_prepared_sha256") for p in pairs}
    for row in rows:
        want = want_prep.get((row["cohort_id"], row["name"]))
        if want is not None:
            row["prepared_matches_record"] = row.get("prepared_sha256") == want
    errors = [r for r in results if "error" in r]
    differ = [row for row in rows if row.get("equal") is False]
    wrong_data = [row for row in rows if row.get("prepared_matches_record") is False]
    in_scope_full = any(row["level"] == "run_fit" and row.get("in_scope") and row.get("equal") for row in rows)
    ok = (not errors and not differ and not stopped and not wrong_data and len(results) == len(pairs)
          and in_scope_full)
    run.write({"status": "pass" if ok else ("stopped" if (stopped or differ) else "fail"), "stopped": stopped,
               "pairs": len(pairs), "done": len(results), "rows": rows, "errors": errors,
               "differences": [{k: row.get(k) for k in ("cohort_id", "name", "level", "diff_paths", "srk")}
                               for row in differ],
               "full_pair_in_scope_equal": in_scope_full, "prepared_not_the_records": wrong_data,
               "excluded_keys": EXCLUDED,
               "host": platform.node(), "cpu": cpu_model(), "provenance": provenance(library="v6")})
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------------------------------------- identity-off (FR-004)
TEN = ("relabel", "hazard", "window", "shadow", "exact_from_F", "self_consistent", "floor_early", "n_rule", "dt_rule",
       "early_all")


def g1_lines(path=None) -> list[dict]:
    """jax2's G1/cohorts.txt as {name, cid, path, cpu_s} (comment lines skipped)."""
    out = []
    for ln in open(path or PINNED_FILES["g1_cohorts"]):
        if ln.strip() and not ln.startswith("#"):
            name, cid, p, cpu = ln.split()
            out.append({"name": name, "cid": cid, "path": p, "cpu_s": float(cpu)})
    return out


def identity_worker(cid: str, name: str, mode: str = "identity-off", cited=None, library: str = "v2") -> dict:
    """One G1 line: the 1.0.1 port's run_fit with 1.0.0's ten keywords (dt_rule as 1.0.0 passes it, no 1.0.1
    keyword) against the 1.0.0 port's, on G1's observation through the 1.0.0 oracle's loader (library v2; the 1.0.1
    oracle is not loaded). Result, claims and self-consistency block compared, TIMING only excluded (L1, I1f)."""
    import time
    import traceback
    try:
        import sraging.compat.jax2  # noqa: F401  (configures the runtime before the oracle's modules)
        import sraging.compat.jax2_101  # noqa: F401
        import oracle_jax2 as OJ
        from acceptance_003_port import port_observation
        from parity_checks import tree_diff
        from sraging.calibrators import profile_jax2 as PJ
        OJ.load()
        obs = port_observation(OJ.observation(cid))
        P0, P1 = PJ.PORTS["1.0.0"], PJ.PORTS["1.0.1"]
        cls0 = P0.classes[name]
        kw = PJ.run_kwargs(cls0)
        if tuple(sorted(kw)) != tuple(sorted(TEN)):
            raise AssertionError(f"1.0.0's keywords are {sorted(kw)}")
        side = {}
        for tag, P in (("1.0.0", P0), ("1.0.1_off", P1)):
            c0 = time.process_time()
            res = P.claims.run_fit(obs, **kw)
            claims, sc = P.claims.final_claim_kwargs(res, cls0.SELF_CONSISTENT)
            side[tag] = (res, claims, sc, time.process_time() - c0)
        (r0, c0_, s0, t0), (r1, c1, s1, t1) = side["1.0.0"], side["1.0.1_off"]
        dr = tree_diff(_strip(r1), _strip(r0), "/result")
        dc = tree_diff(_strip(c1), _strip(c0_), "/claims")
        ds = tree_diff(_strip(s1), _strip(s0), "/sc_block")
        row = {"cohort_id": cid, "name": name, "level": "identity-off", "equal": not (dr or dc or ds),
               "result_equal": not dr, "claims_equal": not dc, "sc_equal": not ds, "diff_paths": (dr + dc + ds)[:40],
               "n_diffs": len(dr + dc + ds), "excluded_keys": ["TIMING at any depth: " + ", ".join(TIMING)],
               "cpu_s": {"1.0.0": t0, "1.0.1_off": t1}, "keywords": sorted(kw),
               "mechanisms_1_0_1_off": mechanisms_101(r1)}
        return {"cohort_id": cid, "rows": [row]}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "name": name, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-3000:]}


def phase_identity_off(args):
    pan = panel()
    lines = list(pan["identity"]["lines"])
    g1 = g1_lines()
    if _file_sha(PINNED_FILES["g1_cohorts"]) != pan["pins"]["g1_cohorts"]:
        sys.exit("[identity-off] REFUSED: G1/cohorts.txt is not the pinned file")
    if [(x["name"], x["cid"]) for x in g1] != [(x["name"], x["cid"]) for x in lines]:
        sys.exit("[identity-off] REFUSED: PANEL [identity] is not G1/cohorts.txt's 11 lines")
    if args.only:
        lines = [x for x in lines if x["cid"] in set(args.only)]
    cluster = sum(x["cpu_s"] for x in lines)
    run = _run("identity-off", args, f"~{2 * cluster / 2.9 / 3600:.1f} Mac CPU-h [I] (G1's {cluster:,.0f} cluster CPU-s "
                                     f"per pass, two passes, / 2.9); {len(lines)} lines, {args.workers} at a time")
    import oracle_jax2 as OJ
    problems = OJ.preflight()
    if problems:
        run.write({"status": "not_applicable", "reasons": problems})
        sys.exit(3)
    pairs = [{"cid": x["cid"], "name": f"profile_{x['name']}", "mode": "identity-off", "library": "v2"}
             for x in lines]
    results, stopped = run_pairs(pairs, args.workers, {}, worker=identity_worker)
    rows = [row for r in results for row in r.get("rows", [])]
    errors = [r for r in results if "error" in r]
    differ = [row for row in rows if not row["equal"]]
    ok = not errors and not differ and not stopped and len(rows) == len(pairs) == (11 if not args.only else len(pairs))
    run.write({"status": "pass" if ok else ("stopped" if (stopped or differ) else "fail"), "stopped": stopped,
               "lines": len(pairs), "equal": sum(r["equal"] for r in rows), "rows": rows, "errors": errors,
               "differences": [{k: row.get(k) for k in ("cohort_id", "name", "diff_paths")} for row in differ],
               "g1_sha256": pan["pins"]["g1_cohorts"], "host": platform.node(), "cpu": cpu_model(),
               "provenance": provenance(library="v2")})
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------------------------------------- v100-unchanged (US2)
ITP_AGE0 = os.path.join(REPO, "tests", "data", "itp_plus_age0.csv")
CHECKS_003 = os.path.join(RESULTS_DIR, "checks_003.json")
ITP_003F = os.path.join(RESULTS_DIR, "itp_003f.json")
LEVERAGE_TIMING = ("cost",)          # the check's CPU and its share of the fit: timing, like every cpu_s


def leverage_diff(new: dict, stored: dict) -> dict:
    """The leverage block against 003f's stored one, on 003f's keys, TIMING (any depth) and `cost` aside. Keys the
    block gained since (spec 003b: g0) are listed, not compared."""
    from parity_checks import tree_diff
    keys = [k for k in stored if k not in LEVERAGE_TIMING]
    d = tree_diff(_strip({k: new.get(k) for k in keys}), _strip({k: stored[k] for k in keys}), "/leverage")
    return {"equal": not d, "diff_paths": d[:40], "added_keys": sorted(set(new) - set(stored)),
            "missing_keys": sorted(set(stored) - set(new))}


def phase_v100_unchanged(args):
    """US2: ITP+age0 under profile_jax2@1.0.0. theta-hat and loglik equal checks_003.json's fit with the death; the
    claims equal the frozen 1.0.0 adapter's final_claim_kwargs of the same run_fit result (no claims of this fit are
    stored: checks_003.json keeps theta-hat and loglik only); diagnostics.leverage equals itp_003f.json's block on its
    keys, timing aside. The 1.0.0 oracle is loaded for the claims; no 1.0.1 oracle."""
    import json
    run = _run("v100-unchanged", args, "~0.6 Mac CPU-h (one profile_jax2@1.0.0 fit of ITP+age0 with the check; 003f "
                                       "itp 2,050 s wall)")
    import sraging.compat.jax2  # noqa: F401
    import oracle_jax2 as OJ
    problems = OJ.preflight()
    if problems:
        run.write({"status": "not_applicable", "reasons": problems})
        sys.exit(3)
    o = OJ.load()
    from parity_checks import CLAIM_FIELDS, tree_diff
    from sraging import Cohort, registry
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.v033 import runtime
    spec = registry.get("profile_jax2", "1.0.0")
    stored = json.load(open(CHECKS_003))["itp_age0"]["absorbed"]
    lev_stored = json.load(open(ITP_003F))["leverage"]
    c = Cohort.from_csv(ITP_AGE0, delta=1.0, time_unit="days")
    runtime.check()
    out = PJ.run_port(c, "profile_jax2", capture=True, port=PJ.V100)
    fr = PJ.fit_result(c, "profile_jax2", out, port=PJ.V100, combinations=False)
    th = fr.theta_hat.as_dict()
    want = stored["theta_hat_with"]
    theta_eq = {p: float(th[p]).hex() == float(want[p]).hex() for p in ("eta", "beta", "epsilon", "kappa")}
    ll_eq = float(fr.loglik).hex() == float(stored["loglik_with"]).hex()
    oclaims, _osc = o.final_claim_kwargs(out["result"], PJ.V100.classes["profile_jax2"].SELF_CONSISTENT)
    mine = [c_.reference_dict() for c_ in fr.claims]
    theirs = [{k: kw.get(k, {"n_components": 1}.get(k)) for k in CLAIM_FIELDS} for kw in oclaims]
    dclaims = tree_diff(mine, theirs, "/claims")
    lev = leverage_diff(fr.diagnostics.get("leverage") or {}, lev_stored)
    ok = all(theta_eq.values()) and ll_eq and not dclaims and lev["equal"] and spec.version == "1.0.0"
    run.write({"status": "pass" if ok else "fail", "theta_hat_equal": theta_eq, "loglik_equal": ll_eq,
               "claims_equal_frozen_adapter": not dclaims, "claims_diff": dclaims, "claims": mine,
               "leverage": lev, "leverage_block": fr.diagnostics.get("leverage"),
               "fit_cpu_s": out["cpu_s"], "provenance": provenance(version="1.0.0")})
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------------------------------------- lev-measure (pilot)
PANEL_003F = os.path.join(REPO, "tests", "acceptance", "panel_003f.toml")
ITP_TAG = "ITP+age0"
#: the bound tolerances the pilot reads the free set under (no refit): 003f's 1e-6 and three wider ones
BOUND_TOLS = (1e-6, 1e-4, 1e-3, 1e-2)


def pilot_cohorts() -> list[dict]:
    """003f's P3 human hazard cohorts (the `hazard-real-` ids of its human strata), with the array sha256s 003f
    pinned, in panel order; then ITP+age0."""
    p = tomllib.load(open(PANEL_003F, "rb"))["p3"]
    sha = p["arrays_sha256"]
    ids = [c for k in ("human_no_basin_fix", "human_basin_fix") for c in p[k] if c.startswith("hazard-real-")]
    return [{"cid": c, "sha256": sha[c], "library": "v6"} for c in ids] + [{"cid": ITP_TAG, "library": None}]


def bound_tol_sensitivity(cohort, name, z, port, rule_q, tols=BOUND_TOLS) -> list[dict]:
    """The influence step of the check (sraging.leverage) under each bound tolerance: which coordinates are free, how
    many information directions survive, and the top-influence row under the rule's quantities. No refit."""
    from sraging import leverage as LEV
    from sraging.calibrators import profile_jax2 as PJ
    hazard = bool(port.classes[name].HAZARD)
    td = LEV.terms_and_derivatives(cohort, name, z, port=port)
    lo, hi = LEV._bounds(PJ.fit_inputs(cohort, name, port)["box"], hazard)
    out, orig = [], LEV.BOUND_TOL
    try:
        for tol in tols:
            LEV.BOUND_TOL = tol
            inf = LEV.influence(td["G"], td["H"], z, lo, hi)
            r = LEV.ranking(LEV.shifts(inf["DZ"], z), rule_q) if inf["n_kept"] else None
            out.append({"bound_tol": tol, "free": inf["free"], "n_kept": inf["n_kept"],
                        "top_row": None if r is None else int(r["top"]) + 1})
    finally:
        LEV.BOUND_TOL = orig
    return out


def lev_worker(cid: str, name: str = "profile_jax2", mode: str = "lev-measure", cited=None, library="v6") -> dict:
    """One pilot fit under profile_jax2@1.0.1 (the public fit; combinations off, as 003f's P3): what the check records
    (label, R, Delta_j, g0, the log distances to the bounds, the gradient) and what Naveh's decision reads: Delta_j - g0,
    the label under it, and the free set under each bound tolerance."""
    import math
    import time
    import traceback
    try:
        import sraging
        from sraging import Cohort
        from sraging import leverage as LEV
        from sraging.adapters.observation import cohort_from_observation
        from sraging.calibrators import profile_jax2 as PJ
        from sraging.leverage_rule import RULE
        P1 = PJ.PORTS["1.0.1"]
        if cid == ITP_TAG:
            c = Cohort.from_csv(ITP_AGE0, delta=1.0, time_unit="days")
        else:
            import oracle_jax2_101 as OJ1
            c = cohort_from_observation(OJ1.observation(cid, library))
        t0 = time.time()
        r = sraging.fit_cohort(c, calibrator=f"{name}@1.0.1", combinations=False)
        lev = r.diagnostics.get("leverage") or {}
        z = LEV._z(r.theta_hat, bool(P1.classes[name].HAZARD))
        q = RULE["quantities"]
        dt, g = lev.get("delta_total"), (lev.get("g0") or {}).get("g0")
        R = (lev.get("R") or {}).get(q)
        net = None if dt is None or g is None else dt - g
        lab_g0 = None if net is None else LEV.label(R, net, ratio=RULE["ratio"],
                                                    delta_total_nats=RULE["delta_total_nats"],
                                                    fail_nats=RULE["fail_nats"])
        a = lev.get("at_theta_hat") or {}
        dist = a.get("log_distance_to_bounds") or {}
        names = list(dist)
        grad = a.get("gradient") or []
        coords = {p: {"lower": v[0], "upper": v[1], "min": min(v), "at_bound": min(v) < LEV.BOUND_TOL,
                      "gradient": grad[k] if k < len(grad) else None} for k, (p, v) in enumerate(dist.items())}
        sens = bound_tol_sensitivity(c, name, z, P1, q)
        row = {"cohort_id": cid, "name": name, "n": c.n, "level": "lev-measure", "status": r.status,
               "fit_cpu_s": r.timing.get("cpu_s"), "wall_s": time.time() - t0,
               "theta_hat": r.theta_hat.as_dict(), "loglik": r.loglik,
               "mechanisms": mechanisms_101({"diagnostics": r.diagnostics}),
               "label": lev.get("label"), "reason": lev.get("reason"), "R": lev.get("R"), "delta_total": dt,
               "g0": lev.get("g0"), "delta_minus_g0": net, "label_if_read_against_g0": lab_g0,
               "dropped": lev.get("dropped"), "refit_start": (lev.get("refit") or {}).get("start"),
               "ll_by_start": lev.get("ll_by_start"), "coordinates": coords,
               "free_gradient_norm": a.get("free_gradient_norm"), "bound_tol": LEV.BOUND_TOL,
               "bound_tol_sensitivity": sens, "check_cost": lev.get("cost"), "basin_fix_fired": lev.get("basin_fix_fired"),
               "warnings": [w.code for w in r.warnings], "leverage_block": lev,
               "finite": all(math.isfinite(x) for x in (dt, g) if x is not None)}
        return {"cohort_id": cid, "rows": [row]}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-3000:]}


def phase_lev_measure(args):
    """The pilot (FR-010.1; research R11): 003f's 7 human hazard cohorts of P3 and ITP+age0 under 1.0.1. Written, not
    gated; then STOP for Naveh (tasks T027): whether Delta_j is read against g0, and whether BOUND_TOL changes."""
    cohorts = pilot_cohorts()
    if args.only:
        cohorts = [x for x in cohorts if x["cid"] in set(args.only)]
    run = _run("lev-measure", args, f"~4-5 Mac CPU-h [I] (003f's P3 fits of these 7 took 8,488 s wall at 1.0.0; 1.0.1 "
                                    f"adds its mechanisms and g0; + ITP+age0); {len(cohorts)} fits, {args.workers} at a "
                                    f"time (keep the lid open)")
    import oracle_jax2_101 as OJ1
    for x in cohorts:
        if x["library"]:
            got = _file_sha(os.path.join(OJ1.LIBRARIES[x["library"]], "cohorts", f"{x['cid']}.npz"))
            if got != x["sha256"]:
                sys.exit(f"[lev-measure] REFUSED: {x['cid']}'s array is {str(got)[:12]}, not 003f's {x['sha256'][:12]}")
    pairs = [{"cid": x["cid"], "name": "profile_jax2", "mode": "lev-measure", "library": x["library"] or "v6"}
             for x in cohorts]
    results, _stopped = run_pairs(pairs, args.workers, {}, worker=lev_worker, stop_on_difference=False)
    rows = [row for r in results for row in r.get("rows", [])]
    errors = [r for r in results if "error" in r]
    from sraging.leverage_rule import RULE
    summary = {
        "n": len(rows), "labels": {r["cohort_id"]: r["label"] for r in rows},
        "labels_if_read_against_g0": {r["cohort_id"]: r["label_if_read_against_g0"] for r in rows},
        "g0": {r["cohort_id"]: (r["g0"] or {}).get("g0") for r in rows},
        "delta_total": {r["cohort_id"]: r["delta_total"] for r in rows},
        "beta_upper_distance": {r["cohort_id"]: (r["coordinates"].get("beta") or {}).get("upper") for r in rows},
        "free_by_tol": {r["cohort_id"]: {str(s["bound_tol"]): s["free"] for s in r["bound_tol_sensitivity"]}
                        for r in rows},
        "top_row_by_tol": {r["cohort_id"]: {str(s["bound_tol"]): s["top_row"] for s in r["bound_tol_sensitivity"]}
                           for r in rows}}
    run.write({"status": "written" if not errors and len(rows) == len(pairs) else "incomplete", "rows": rows,
               "errors": errors, "summary": summary, "rule": dict(RULE), "bound_tols": list(BOUND_TOLS),
               "provenance": provenance(library="v6")})
    if errors:
        sys.exit(1)


# ------------------------------------------------------------------------------------------------- itp (FR-010.4)
EXAMPLE_CSV = os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv")
ITP_TAG_WITHOUT = "ITP (example, no age-0 death)"
ROUND_OFF_LOG = 1e-9                 # the MLE-stage refit and the fit of the cohort without the death, in log theta


def itp_worker(cid: str, name: str = "profile_jax2", mode: str = "itp", cited=None, library=None) -> dict:
    """One 1.0.1 fit (the public fit, combinations off): ITP+age0 with the check, or the README example (ITP without
    its age-0 death), in its own process."""
    import traceback
    try:
        import sraging
        from sraging import Cohort
        c = Cohort.from_csv(ITP_AGE0 if cid == ITP_TAG else EXAMPLE_CSV, delta=1.0, time_unit="days")
        r = sraging.fit_cohort(c, calibrator=f"{name}@1.0.1", combinations=False)
        return {"cohort_id": cid, "rows": [{"cohort_id": cid, "level": "itp", "status": r.status,
                                            "theta_hat": r.theta_hat.as_dict(), "loglik": r.loglik,
                                            "leverage": r.diagnostics.get("leverage"),
                                            "warnings": [w.code for w in r.warnings],
                                            "ages": [float(a) for a in c.ages[:3]], "n": c.n,
                                            "fit_cpu_s": r.timing.get("cpu_s"),
                                            "provenance_version": (r.provenance.get("calibrator") or {}).get("version")}]}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-3000:]}


def itp_verdict(with_row: dict, without_row: dict, age0_row: int) -> dict:
    """AC1 (SC-006): the check drops the age-0 death; AC2: it flags, LEVERAGE fires, and its MLE-stage refit equals
    1.0.1's own fit of the cohort without the death to round-off (log theta within ROUND_OFF_LOG)."""
    import math
    lev = with_row.get("leverage") or {}
    rm = ((lev.get("refits") or {}).get("mle_stage") or {}).get("theta")
    wo = without_row["theta_hat"]
    diff = None if not rm else max(abs(math.log(rm[k]) - math.log(wo[p])) for k, p in
                                   enumerate(("eta", "beta", "epsilon", "kappa")))
    ac1 = (lev.get("dropped") or {}).get("row") == age0_row
    refit_eq = diff is not None and diff < ROUND_OFF_LOG
    ac2 = lev.get("label") == "flag" and "LEVERAGE" in with_row.get("warnings", []) and refit_eq
    return {"AC1": ac1, "AC2": ac2, "refit_vs_fit_without_max_log": diff, "refit_equal_to_round_off": refit_eq,
            "label": lev.get("label"), "dropped": lev.get("dropped"), "start_that_stood": (lev.get("refit") or {})
            .get("start"), "g0": lev.get("g0"), "R": lev.get("R"), "delta_total": lev.get("delta_total")}


def phase_itp(args):
    run = _run("itp", args, "~1.2 Mac CPU-h (two 1.0.1 fits, in parallel: ITP+age0 with the check, and the example "
                            "without the death)")
    import numpy as np
    from sraging import Cohort
    age0_row = int(np.flatnonzero(np.asarray(Cohort.from_csv(ITP_AGE0, delta=1.0).ages, float) == 0.0)[0]) + 1
    pairs = [{"cid": ITP_TAG, "name": "profile_jax2", "mode": "itp"},
             {"cid": ITP_TAG_WITHOUT, "name": "profile_jax2", "mode": "itp"}]
    results, _ = run_pairs(pairs, 2, {}, worker=itp_worker, stop_on_difference=False)
    errors = [r for r in results if "error" in r]
    rows = {r["cohort_id"]: r["rows"][0] for r in results if "rows" in r}
    if errors or len(rows) != 2:
        run.write({"status": "FAIL", "errors": errors, "provenance": provenance()})
        sys.exit(1)
    v = itp_verdict(rows[ITP_TAG], rows[ITP_TAG_WITHOUT], age0_row)
    ok = v["AC1"] and v["AC2"] and all(r["provenance_version"] == "1.0.1" for r in rows.values())
    run.write({"status": "PASS" if ok else "FAIL", **v, "age0_row": age0_row, "fits": rows,
               "provenance": provenance()})
    if not ok:
        sys.exit(f"FAIL: AC1 {v['AC1']}, AC2 {v['AC2']} (refit vs fit without: {v['refit_vs_fit_without_max_log']})")


# ------------------------------------------------------------------------------------------------- ac11 (FR-010.2-3)
NULLS = ("held_out", "small_n", "twins")


def wilson(k, n, z=1.959963984540054):
    import acceptance_003f as F
    return F.wilson(k, n, z)


def ac11_registration() -> dict:
    """PANEL [ac11], refused when empty or when its rule is not the one in force."""
    from sraging.leverage_rule import RULE
    reg = panel().get("ac11") or {}
    need = ("rule", "held_out", "small_n", "planted", "ceilings", "late_rule", "generator_version")
    missing = [k for k in need if k not in reg]
    if missing:
        sys.exit(f"[ac11] REFUSED: PANEL [ac11] lacks {missing} (tasks T029: registered before any AC11 fit)")
    if dict(reg["rule"]) != dict(RULE):
        sys.exit("[ac11] REFUSED: PANEL [ac11].rule is not sraging.leverage_rule.RULE")
    return reg


def planted_cohorts(reg: dict) -> list[dict]:
    """The planted panel: for each registered base spec and n, the unplanted twin and the two planted variants (one
    age-0 death; one extreme late death at late_rule's age), as {cid, spec, n, seed, variant}."""
    out = []
    for p in reg["planted"]:
        for n_s, seed in sorted(p["seeds"].items(), key=lambda kv: int(kv[0])):
            for variant in ("twin", "age0", "late"):
                out.append({"cid": f"planted-{p['name']}-n{n_s}-{variant}", "base": p["name"], "n": int(n_s),
                            "seed": int(seed), "variant": variant})
    return out


def planted_cohort(reg: dict, cid: str):
    """(Cohort, planted 1-based row or None) for a planted id: sraging.generator.materialize of the registered spec,
    the planted death appended to the materialised arrays (an age-0 death, or one at late_rule's age)."""
    import numpy as np
    from sraging import Cohort
    from sraging import generator as GEN
    if GEN.GENERATOR_VERSION != reg["generator_version"]:
        raise RuntimeError(f"the generator is {GEN.GENERATOR_VERSION}, the registration {reg['generator_version']}")
    item = next(x for x in planted_cohorts(reg) if x["cid"] == cid)
    base = next(p for p in reg["planted"] if p["name"] == item["base"])
    spec = GEN.CohortSpec(name=cid, n=item["n"], seed=item["seed"], **dict(base["spec"]))
    g = GEN.materialize(spec)
    ages, events = np.asarray(g.death_times, float), np.asarray(g.events, int)
    row = None
    if item["variant"] != "twin":
        lr = reg["late_rule"]
        extra = 0.0 if item["variant"] == "age0" else round(float(np.max(ages)) * float(lr["factor_of_max_age"])
                                                          / spec.delta) * spec.delta
        ages, events = np.append(ages, extra), np.append(events, 1)
        row = ages.size
    return Cohort.from_arrays(ages, events, delta=spec.delta, name=cid), row


def ac11_worker(cid: str, name: str = "profile_jax2", mode: str = "ac11", cited=None, library="v6") -> dict:
    """One AC11 fit under profile_jax2@1.0.1 (combinations off): a library cohort (held-out, small-n) or a planted
    one; the check's labels per quantity set and, for a planted variant, whether the planted death was dropped."""
    import traceback
    try:
        import sraging
        if mode == "planted":
            c, planted_row = planted_cohort(ac11_registration(), cid)
        else:
            import oracle_jax2_101 as OJ1
            from sraging.adapters.observation import cohort_from_observation
            c, planted_row = cohort_from_observation(OJ1.observation(cid, library)), None
        r = sraging.fit_cohort(c, calibrator=f"{name}@1.0.1", combinations=False)
        lev = r.diagnostics.get("leverage") or {}
        dropped = (lev.get("dropped") or {}).get("row")
        return {"cohort_id": cid, "rows": [{
            "cohort_id": cid, "mode": mode, "n": c.n, "status": r.status, "label": lev.get("label"),
            "labels": lev.get("labels"), "R": lev.get("R"), "delta_total": lev.get("delta_total"), "g0": lev.get("g0"),
            "dropped": lev.get("dropped"), "planted_row": planted_row,
            "planted_dropped": planted_row is not None and dropped == planted_row, "fit_cpu_s": r.timing.get("cpu_s"),
            "check_cost": lev.get("cost"), "warnings": [w.code for w in r.warnings]}]}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-3000:]}


def ac11_score(rows: list[dict], reg: dict) -> dict:
    """Flag rates per null and per quantity set, with Wilson 95 % intervals against the ceilings (the gate reads the
    rule's set); detection by n and variant on the planted panel (reported): flagged with the planted death dropped."""
    from sraging import leverage as LEV
    q = reg["rule"]["quantities"]
    by_null = {"held_out": [r for r in rows if r["mode"] == "held_out"],
               "small_n": [r for r in rows if r["mode"] == "small_n"],
               "twins": [r for r in rows if r["mode"] == "planted" and r["planted_row"] is None]}
    nulls = {}
    for null, rs in by_null.items():
        rs = [r for r in rs if r.get("labels")]
        per = {}
        for s in LEV.QUANTITY_SETS:
            k = sum(r["labels"].get(s) == "flag" for r in rs)
            per[s] = {"flags": k, "n": len(rs), "rate": k / len(rs) if rs else None, "wilson": wilson(k, len(rs))}
        ceiling = reg["ceilings"].get(null)
        nulls[null] = {"per_set": per, "ceiling": ceiling, "gate_set": q,
                       "pass": None if ceiling is None else per[q]["wilson"][1] <= ceiling}
    det = {}
    for r in rows:
        if r["mode"] != "planted" or r["planted_row"] is None:
            continue
        variant = r["cohort_id"].rsplit("-", 1)[1]
        n = r["cohort_id"].split("-n")[1].split("-")[0]
        d = det.setdefault(variant, {}).setdefault(n, {"detected": 0, "n": 0})
        d["n"] += 1
        d["detected"] += bool(r["label"] == "flag" and r["planted_dropped"])
    for variant in det.values():
        for d in variant.values():
            d["rate"] = d["detected"] / d["n"] if d["n"] else None
            d["wilson"] = wilson(d["detected"], d["n"])
    return {"nulls": nulls, "detection": det}


def phase_ac11(args):
    reg = ac11_registration()
    pins = panel()["pins"]
    import oracle_jax2_101 as OJ1
    jobs = []
    for null in ("held_out", "small_n"):
        for cid, sha in reg[null]["sha256"].items():
            got = _file_sha(os.path.join(OJ1.LIBRARY_V6, "cohorts", f"{cid}.npz"))
            if got != sha:
                sys.exit(f"[ac11] REFUSED: {cid}'s array is {str(got)[:12]}, not the registration's {sha[:12]}")
            jobs.append({"cid": cid, "name": "profile_jax2", "mode": null, "library": "v6"})
    jobs += [{"cid": x["cid"], "name": "profile_jax2", "mode": "planted", "library": None}
             for x in planted_cohorts(reg)]
    if args.only:
        jobs = [j for j in jobs if j["cid"] in set(args.only)]
    if _file_sha(os.path.join(OJ1.LIBRARY_V6, "manifest.json")) != pins["library_v6_manifest"]:
        sys.exit("[ac11] REFUSED: the v6 manifest is not the pinned one")
    run = _run("ac11", args, f"~8-12 Mac CPU-h [I] ({len(jobs)} fits with the check, {args.workers} at a time; keep "
                             f"the lid open)")
    results, _ = run_pairs(jobs, args.workers, {}, worker=ac11_worker, stop_on_difference=False)
    rows = [row for r in results for row in r.get("rows", [])]
    errors = [r for r in results if "error" in r]
    score = ac11_score(rows, reg)
    gates = [v["pass"] for v in score["nulls"].values() if v["pass"] is not None]
    ok = not errors and len(rows) == len(jobs) and all(gates)
    run.write({"status": "PASS" if ok else "FAIL", **score, "rows": rows, "errors": errors,
               "registration_commit": _git("log", "-1", "--format=%H", "--", PANEL_PATH),
               "provenance": provenance(library="v6", manifest_sha256=pins["library_v6_manifest"])})
    if not ok:
        sys.exit(f"FAIL: gates {score['nulls']}; errors {len(errors)}")


# ------------------------------------------------------------------------------------------------- readme-example
STORED_JAX2_EXAMPLE = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")


def phase_readme_example(args):
    """FR-012: the README's jax2 example under 1.0.1 (the README's own command; lead combinations). Its differences
    from the stored 1.0.0 example are reported, not required equal (a new version); the stored example is regenerated
    from this run, written only after an acceptance result."""
    import json
    import math
    import shutil
    import tempfile
    run = _run("readme-example", args, "~1.6 Mac CPU-h (one profile_jax2@1.0.1 fit of the README example, lead "
                                       "combinations; 003f's took 5,925 s wall)")
    with tempfile.TemporaryDirectory(prefix="sraging003b_") as tmp:
        out = os.path.join(tmp, "fit.json")
        p = subprocess.run([os.path.join(REPO, ".venv", "bin", "sraging"), "fit-cohort", "--csv", EXAMPLE_CSV,
                            "--delta", "1", "--time-unit", "days", "--calibrator", "profile_jax2", "--json", out],
                           capture_output=True, text=True, cwd=REPO)
        if p.returncode not in (0, 1) or not os.path.exists(out):
            sys.exit(f"FAIL: fit-cohort exited {p.returncode}: {p.stderr[-1500:]}")
        new_d, old_d = json.load(open(out)), json.load(open(STORED_JAX2_EXAMPLE))
        th_new, th_old = new_d["theta_hat"], old_d["theta_hat"]
        dex = {k: math.log10(th_new[k] / th_old[k]) for k in ("eta", "beta", "epsilon", "kappa")
               if th_new.get(k) and th_old.get(k)}
        claims = {k: {"old": {f: old_d["claims"].get(k, {}).get(f) for f in ("shape", "lower", "upper")},
                      "new": {f: v.get(f) for f in ("shape", "lower", "upper")}}
                  for k, v in new_d["claims"].items()}
        version = (new_d["provenance"].get("calibrator") or {}).get("version")
        res = run.write({"status": "REPORTED" if version == "1.0.1" else "FAIL", "version": version,
                         "theta_hat_dex_from_1_0_0": dex, "loglik": {"new": new_d["loglik"], "old": old_d["loglik"]},
                         "claims": claims, "leverage": new_d["diagnostics"].get("leverage"),
                         "printed": p.stdout.strip().split("\n"), "provenance": provenance()})
        if res["status_for_acceptance"] == "acceptance" and version == "1.0.1":
            shutil.copyfile(out, STORED_JAX2_EXAMPLE)
    if version != "1.0.1":
        sys.exit(f"FAIL: the README's command ran {version}, not 1.0.1")


# ------------------------------------------------------------------------------------------------- bytes, unchanged
def _pytest_groups(groups: dict) -> dict:
    """Each group of test paths by pytest in its own process, junit-counted, every skip and failure listed."""
    import tempfile
    import time
    import xml.etree.ElementTree as ET
    out = {}
    with tempfile.TemporaryDirectory(prefix="sraging003b_") as tmp:
        for g, paths in groups.items():
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
    return out


def phase_bytes(args):
    """Tasks T034: the parity suite once at the frozen src, T013's 1.0.1 bytes test included; its summary, and the
    bytes item's check lines (tools/jax2_cases_101.py check), beside parity_003b.json."""
    import acceptance_003e as E
    import json
    run = _run("bytes", args, "the parity suite in one pytest process (minutes) + the 1.0.1 bytes item's check")
    out = _pytest_groups({"parity": ["tests/parity"]})
    p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "jax2_cases_101.py"), "check"],
                       capture_output=True, text=True, cwd=REPO)
    lines = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
    ok, unlisted = E.unchanged_verdict(out)
    ok = ok and bool(lines) and all(x.get("bytes_equal") and x.get("v033_terms") == [-700.0] for x in lines)
    run.write({"status": "PASS" if ok else "FAIL", "groups": out, "skips_unlisted": unlisted, "bytes_item": lines,
               "provenance": provenance()})
    if not ok:
        sys.exit(1)


def phase_unchanged(args):
    """FR-013 / SC-005, as 003f's: the unit and parity suites and the results tests of 001-003f (003f bounded by its
    merge, T019), every skip named."""
    import acceptance_003e as E
    groups = dict(E.UNCHANGED_GROUPS)
    groups["results_001_002_003"] = groups["results_001_002_003"] + ["tests/acceptance/test_results_003e.py",
                                                                     "tests/acceptance/test_results_003f.py"]
    run = _run("unchanged", args, "pytest in 3 processes: the unit suite (~2.5 min), the parity suite and the "
                                  "001-003f results tests; several minutes; 0 calibrations")
    out = _pytest_groups(groups)
    ok, unlisted = E.unchanged_verdict(out)
    run.write({"status": "PASS" if ok else "FAIL", "groups": out, "skips_unlisted": unlisted,
               "superseded": E.SUPERSEDED, "recorded_skips": E.RECORDED_SKIPS, "provenance": provenance()})
    if not ok:
        sys.exit(f"FAIL: {[(g, v['failed_list'][:3]) for g, v in out.items() if v['failures'] or v['errors']]}; "
                 f"skips not named: {unlisted}")


PHASES = {"lev-measure": phase_lev_measure, "itp": phase_itp, "ac11": phase_ac11,
          "readme-example": phase_readme_example, "bytes": phase_bytes, "unchanged": phase_unchanged}
PHASES.update({"verify": phase_verify, "census": phase_census, "parity": phase_parity,
               "identity-off": phase_identity_off, "v100-unchanged": phase_v100_unchanged})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", help="restrict to these cohorts (a smoke run; never counts)")
    args = ap.parse_args(argv)
    if args.only:
        args.dev = True
    require_byte_host(args.phase)
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
