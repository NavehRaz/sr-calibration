"""Spec 003 acceptance phases (specs/003-calibrator/contracts/acceptance.md; tasks T017-T019, T024, T032-T033,
T037, T039, T042, T044, T049).

    .venv/bin/python tools/acceptance_003.py <phase> [--dev] [--workers W] [options]

Every phase prints its price first and writes tests/acceptance/results/<phase>_003.json with its provenance (the
repo HEAD, whether the tracked tree is dirty, the host, the J2 revision). A phase REFUSES a dirty tracked tree or
a manifest that is not committed unchanged; `--dev` runs it anyway and labels the result "development", which
never counts (tests/acceptance/test_results_003.py). Long phases run from a worktree at the committed HEAD
(`git worktree add --detach .cache/wt/run HEAD`), never while tracked files are being edited.

Phases: pilot (writes tests/acceptance/panel_003.toml), bytes, fits, speed, checks, adapter, order, floor,
walker, kmax, coverage, unchanged.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import math
import os
import random
import subprocess
import sys
import time
import tomllib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
# This checkout's src first: the venv's editable install points at the MAIN tree's src, so a run in a worktree
# (.cache/wt/...) would otherwise fit with the main tree's code in this process (spawned workers inherit sys.path).
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools"), os.path.join(REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import oracle_jax2 as OJ  # noqa: E402
import oracle_v033 as O  # noqa: E402

RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
PANEL = "tests/acceptance/panel_003.toml"
PANEL_V1 = "tests/acceptance/panel_v1.toml"
J2 = OJ.J2
T = O.TESTER
SETS_C = os.path.join(J2, "phaseC", "tasks", "sets_C.json")
STORED_FITS = ("reports/rc2/C/freeze_1.0.0/identity", "reports/rc2/dev6/identity", "reports/rc2/R3d6/L5")
MECHANISMS = [   # research R5: each listed mechanism fires in the stored Mac fit of the same runtime
    ("realfull-real-Sweden_F_1910_homo-c1-s1", ["basin re-fit (jax2)", "relabel (6)", "N=1800"]),
    ("realfull-real-Denmark_M_1900_homo-c1-s1", ["relabel (2)"]),
    ("pure-k0.01-p00-c0-s0", ["N=1800"]),
    ("pure-k0.01-p05-c0-s0", ["N=3600", "Nelder-Mead fallback (41)", "widening under _h"]),
    ("hazard-k0.01-p00-c0-s0", ["_h's planted hazard", "widening under _h"]),
]
SPEED = ["pure-k0.01-p02-c1-s1", "pure-k0.01-p03-c0-s1", "pure-k0.01-p04-c0-s0", "pure-k0.01-p02-c0-s0",
         "pure-k0.01-p03-c1-s0"]
SEED = 20261005


def now():
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args, repo=REPO) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()


def tree_dirty() -> bool:
    return bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)tests/acceptance/results"))


def committed(rel: str) -> bool:
    return bool(git("ls-files", rel)) and not git("status", "--porcelain", "--", rel)


def last_commit(rel: str) -> str:
    return git("log", "-1", "--format=%H", "--", rel)


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    raise TypeError(type(x).__name__)


def own_src_problem():
    """None if `sraging` imports from this checkout's src (the tree whose HEAD the result records), else why not."""
    import sraging
    got, want = os.path.realpath(sraging.__file__), os.path.realpath(os.path.join(REPO, "src", "sraging"))
    return None if got.startswith(want + os.sep) else f"sraging imports from {got}, not this checkout's {want}"


class Run:
    """One phase run: the provenance gate at the start, the re-check and the result file at the end
    (tools/acceptance_002.py's Run, for spec 003's results and the J2 tester)."""

    def __init__(self, phase, manifests=(), dev=False, cost="", suffix="003"):
        self.phase, self.manifests, self.dev, self.suffix = phase, list(manifests), dev, suffix
        self.head0, self.dirty0 = git("rev-parse", "HEAD"), tree_dirty()
        self.uncommitted = [m for m in self.manifests if not committed(m)]
        print(f"[{phase}] price: {cost}", flush=True)
        own = own_src_problem()
        if own:
            sys.exit(f"[{phase}] REFUSED: {own}")
        if (self.dirty0 or self.uncommitted) and not dev:
            why = (["the tracked tree is dirty"] if self.dirty0 else []) + \
                  ([f"manifest(s) not committed unchanged: {', '.join(self.uncommitted)}"] if self.uncommitted else [])
            sys.exit(f"[{phase}] REFUSED: {'; '.join(why)} (commit first, or pass --dev for a development run that "
                     f"never counts)")
        self.t0, self.c0 = time.time(), time.process_time()

    def write(self, payload, name=None):
        os.makedirs(RESULTS, exist_ok=True)
        head1, dirty1 = git("rev-parse", "HEAD"), tree_dirty()
        changed = head1 != self.head0 or dirty1 != self.dirty0
        dev = self.dev or self.dirty0 or dirty1 or bool(self.uncommitted) or changed
        out = {"phase": self.phase, "written_at": now(), "repo_head": self.head0, "tree_dirty": self.dirty0 or dirty1,
               "tree_changed_during_run": changed, "manifests": {m: last_commit(m) for m in self.manifests},
               "manifests_uncommitted": self.uncommitted,
               "status_for_acceptance": "development" if dev else "acceptance",
               "host": O.host_identity(), "tester": {"root": J2, "head": git("rev-parse", "HEAD", repo=J2)},
               "wall_s": time.time() - self.t0, "cpu_s": time.process_time() - self.c0, **payload}
        path = os.path.join(RESULTS, f"{name or self.phase.replace('-', '_')}_{self.suffix}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, allow_nan=True, default=_jsonable)
            fh.write("\n")
        print(f"[{self.phase}] status {out.get('status')}; wrote {os.path.relpath(path, REPO)} "
              f"({out['status_for_acceptance']})", flush=True)
        return out


def preflight_or_exit(run):
    try:
        OJ.load()
    except OJ.NotApplicable as ex:
        run.write({"status": "not_applicable", "reasons": ex.problems})
        sys.exit(3)


def load_toml(rel):
    return tomllib.load(open(os.path.join(REPO, rel), "rb"))


# ------------------------------------------------------------------------------------------- the stored records
def stored_fits() -> dict:
    """{(cohort, name): stored Mac fit of jax2 1.0.0's runtime} from the tester's reports (read-only): the freeze's
    `__v100on`, dev6's `__dev6on` and R3d6/L5's plain `__<name>.json` records, the same files `stored_runtime_thetas`
    reads (L5 keys the cohort as `cohort` and its CPU at the top level). dev4, dev5on and dev6off are other runtimes:
    the earlier filter let `__dev4` through, and it sorts before `__v100on` (sr-manager gate, agent report)."""
    out = {}
    for rel in STORED_FITS:
        d = os.path.join(J2, rel)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            parts = f[:-5].split("__") if f.endswith(".json") else []
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in ("v100on", "dev6on")):
                continue
            try:
                rec = json.load(open(os.path.join(d, f)))
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict):
                continue
            cid, name = rec.get("cid") or rec.get("cohort"), rec.get("name")
            if cid and name in ("jax2", "jax2_h") and rec.get("result"):
                meta = dict(rec.get("meta") or {})
                meta.setdefault("cpu_s", rec.get("cpu_s"))
                out.setdefault((cid, name), {**rec, "result": j2_floats(rec["result"]), "meta": meta,
                                             "_path": os.path.join(rel, f)})
    return out


J2_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def j2_floats(x):
    """A stored result with the strings R3d6/L5's writer put in place of non-finite floats read back as floats: that
    writer stores NaN as "nan" (measured 2026-10-06: 12 of the 18 stored fits the fits phase compares, every such
    string under /result/ci; the identity records keep JSON's NaN). Any other string is left as it is."""
    if isinstance(x, dict):
        return {k: j2_floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [j2_floats(v) for v in x]
    return J2_NONFINITE.get(x, x) if isinstance(x, str) else x


def stored_runtime_thetas() -> dict:
    """{(cohort, name): {theta, path}} from the tester's stored Mac fits of jax2 1.0.0's runtime only: the freeze's
    `__v100on`, dev6's `__dev6on` and R3d6/L5's plain `__<name>.json` records (L5 keys the cohort as `cohort`);
    dev4, dev5on and dev6off are other runtimes. First found wins, in STORED_FITS order (read-only)."""
    out = {}
    for rel in STORED_FITS:
        d = os.path.join(J2, rel)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            parts = f[:-5].split("__") if f.endswith(".json") else []
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in ("v100on", "dev6on")):
                continue
            try:
                rec = json.load(open(os.path.join(d, f)))
            except (OSError, ValueError):
                continue
            cid, name, th = rec.get("cid") or rec.get("cohort"), rec.get("name"), (rec.get("result") or {}).get("theta")
            if cid and name in ("jax2", "jax2_h") and th:
                out.setdefault((cid, name), {"theta": [float(x) for x in th], "path": os.path.join(rel, f)})
    return out


def sets_c() -> dict:
    return json.load(open(SETS_C))


# ------------------------------------------------------------------------------------------- pilot
def _stratified_draw(items, key, k_per, seed):
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    out = []
    for g in sorted(groups):
        xs = sorted(groups[g])
        rng.shuffle(xs)
        out.extend(xs[:k_per])
    return out


def round_robin(items, key, n, seed):
    """`n` items drawn round-robin over the groups `key` defines, each group shuffled with `seed` (sorted groups)."""
    rng = random.Random(seed)
    groups = {}
    for it in sorted(items):
        groups.setdefault(key(it), []).append(it)
    for g in groups.values():
        rng.shuffle(g)
    out, i = [], 0
    while len(out) < n and any(groups.values()):
        for g in sorted(groups):
            if groups[g] and len(out) < n:
                out.append(groups[g].pop(0))
        i += 1
    return out


def kmax_candidates():
    """Research R15: valid pure ratio-lo cohorts with a beta upper bound in v0.3.3's stored full-tier records."""
    rows = {r["cohort_id"]: r for r in csv.DictReader(open(os.path.join(T, "reports", "validity", "validity.csv")))}
    full = json.load(open(os.path.join(T, "runs", "mle_profile_jax-full.json")))["keys"]
    lib = OJ.library()
    cands, closed = [], []
    for cid, r in rows.items():
        if r.get("arm") != "pure" or r.get("flag_invalid") not in ("False", "false", "0"):
            continue
        try:
            sp = lib.spec(cid)
        except Exception:  # noqa: BLE001 -- a v1-only cohort is not in library v2
            continue
        if not sp.stratum.startswith("ratio-lo") or cid not in full:
            continue
        d = full[cid]
        rec = json.load(open(os.path.join(T, "data", "store", d[:2], d[2:4], f"{d}.json")))
        beta = next((c for c in (rec.get("claim_set") or {}).get("claims", []) if c.get("parameter") == "beta"), None)
        if beta is None:
            continue
        (cands if beta.get("shape") == "upper_bound" else closed if beta.get("shape") == "closed" else []).append(
            (cid, sp.stratum))
    return sorted(cands), sorted(closed)


WALKER_PURE, WALKER_HAZARD, WALKER_K300 = 12, 5, 3


def walker_draw(lib, pure_medium, hazard_h):
    """SC-004's pilot (sr-manager gate 3c, sr-naveh review 8): WALKER_K300 cohorts whose stored same-runtime jax2 fit
    has kappa-hat at kappa_max (within 0.01 dex of 300, under that fit's own name; seeded if more exist, the shortfall
    recorded if fewer), then WALKER_PURE pure medium cohorts under profile_jax2 drawn round-robin over the regime
    strata (the [pilot] rule) and WALKER_HAZARD hazard-arm cohorts under profile_jax2_h, both seeded, none repeated."""
    st = stored_runtime_thetas()
    k300 = sorted({(c, n) for (c, n), r in st.items() if r["theta"][3] >= 300.0 / 10 ** 0.01})
    random.Random(SEED + 4).shuffle(k300)
    rows, seen = [], set()
    for c, n in k300:
        if c in seen or len(rows) >= WALKER_K300:
            continue
        w = lib.spec(c).window
        rows.append({"id": c, "name": f"profile_{n}", "why": f"kappa-hat at 300 in the stored fit {st[(c, n)]['path']}"
                     + ("" if w is None else f"; windowed {list(w)}: fit on the oracle's Observation")})
        seen.add(c)
    for c in round_robin([c for c in pure_medium if c not in seen], lambda c: lib.spec(c).stratum, WALKER_PURE, SEED):
        rows.append({"id": c, "name": "profile_jax2", "why": f"pure, stratum {lib.spec(c).stratum}"})
        seen.add(c)
    for c in round_robin([c for c in hazard_h if c not in seen], lambda c: lib.spec(c).stratum, WALKER_HAZARD,
                         SEED + 3):
        rows.append({"id": c, "name": "profile_jax2_h", "why": f"hazard arm, stratum {lib.spec(c).stratum}"})
        seen.add(c)
    return {"cohorts": rows, "kappa300_found": len(k300), "kappa300_shortfall": max(0, WALKER_K300 - len(
        {c for c, _ in k300})), "rule": "3 kappa-hat-at-300 (stored same-runtime fits: v100on, dev6on, R3d6/L5), "
                                  "12 pure (round-robin over strata, seed SEED), 5 hazard under _h (seed SEED+3)"}


def phase_pilot(args):
    run = Run("pilot", manifests=[PANEL_V1], dev=args.dev, cost="minutes: stored-record reads, one timed value+grad "
              "per C. elegans point (the port), the panel draws")
    preflight_or_exit(run)
    panel, extra = build_panel()
    with open(os.path.join(REPO, PANEL), "w") as fh:
        fh.write(_toml(panel))
    run.write({"status": "written", "panel": PANEL, **extra})


def build_panel():
    """(panel, extra): the panel_003.toml tables and the pilot result's prices and counts; reads stored records only,
    writes nothing (phase_pilot writes both)."""
    lib = OJ.library()
    S = sets_c()
    stored = stored_fits()
    quick = [s.cohort_id for s in lib.tier("quick")]
    # prices from the tester's stored Mac fits of the same runtime (meta.cpu_s), by cohort and name
    price = {f"{c}__{n}": rec.get("meta", {}).get("cpu_s") for (c, n), rec in stored.items()}
    celegans = list(S["C1"]["J"]["CEILING"])[:2]
    # time one port value+grad per C. elegans truth point (the floored ceiling's grid)
    from sraging.adapters.observation import cohort_from_observation
    from sraging.calibrators import profile_jax2 as PJ
    ce_cost = {}
    for cid in celegans:
        obs = lib.observation(cid)
        c = cohort_from_observation(obs) if obs.window is None else None
        ce_cost[cid] = None if c is None else "timed in the bytes phase"
    medium = list(S["C2"]["J"]["MEDIUM"])
    has_window = sorted(c for c in medium if lib.spec(c).window is not None)
    cov_all = sorted(c for c in medium if lib.spec(c).window is None)
    cov_h = sorted(c for c in S["C2"]["H"]["MEDIUM_H"] if c.startswith("hazard-") and lib.spec(c).window is None)
    # sr-manager gate CHEAPEN and 5: what the coverage run does not fit under profile_jax2, and why
    cuts = {
        "het": {"cohorts": sorted(c for c in cov_all if lib.spec(c).arm == "het"),
                "reason": "het arm: 0 informative cohorts (informative = het_var == 0, research R14), so it is scored "
                          "on nothing; about 7.6 CPU-h on the stored v0.3.3 basis"},
        "hazard_under_profile_jax2": {
            "cohorts": sorted(c for c in cov_all if lib.spec(c).arm == "hazard"),
            "reason": "profile_jax2 is misspecified on the hazard arm (seed change 5): reported only, never gated, "
                      "so not fitted; the arm is fitted and gated under profile_jax2_h ([coverage].profile_jax2_h)"},
        "realfull_celegans": {
            "cohorts": sorted(c for c in cov_all
                              if lib.spec(c).arm == "realfull" and lib.spec(c).real_name == "celegans"),
            "reason": "realfull C. elegans: reported, never gated, and never priced; under jax2 the cheapest is "
                      "11,990 CPU-s per fit (research R21), 8.7 CPU-h for the four on v0.3.3"},
    }
    cut = {c for v in cuts.values() for c in v["cohorts"]}
    cov_jax2 = [c for c in cov_all if c not in cut]
    pure_medium = sorted(c for c in medium if c.startswith("pure-k"))
    pilot = round_robin(pure_medium, lambda c: lib.spec(c).stratum, 20, SEED)
    walker = walker_draw(lib, pure_medium, cov_h)
    cands, closed = kmax_candidates()
    strat = dict(cands)
    # FR-025 (sr-manager gate 4): the WHOLE candidate list in seeded, stratified order; the kmax phase confirms
    # along it until KEEP are confirmed or the list ends
    draw = round_robin([c for c, _ in cands], lambda c: strat[c], len(cands), SEED + 1)
    neg = round_robin([c for c, _ in closed], lambda c: "all", 3, SEED + 2)
    panel = {
        "meta": {"written_at": now(), "seed": SEED, "library": OJ.LIBRARY_V2, "sets_c": SETS_C,
                 "rule": "contracts/acceptance.md; research R4, R5, R14, R15"},
        "bytes": {"panel": PANEL_V1, "h_points": ["box_floor", "hL=0.1"], "celegans": celegans,
                  "celegans_theta": ["truth", "eta x1.5"]},
        "fits": {"quick": quick, "mechanisms": [c for c, _ in MECHANISMS],
                 "mechanism_notes": {c: m for c, m in MECHANISMS}},
        "pilot": {"cohorts": pilot, "strata": sorted({lib.spec(c).stratum for c in pilot}),
                  "rule": "C2 medium pure arm, 20 drawn round-robin over the library's regime strata, seeded"},
        "walker": walker,
        "kmax": {"candidates_total": len(cands), "draw": draw, "negative_controls": neg, "keep": 10,
                 "rule": "valid pure ratio-lo with a beta upper bound (v0.3.3 full); every candidate, in seeded "
                         "round-robin order over strata; confirmed along that order (kappa at kappa_max at the cap "
                         "node, by a captured fit) until 10 are confirmed or the list ends; the first 10 kept"},
        "coverage": {"profile_jax2": cov_jax2, "profile_jax2_h": cov_h, "excluded_window": has_window,
                     "gated": [["profile_jax2", "pure"], ["profile_jax2", "censor"], ["profile_jax2_h", "hazard"]],
                     "cuts": cuts},
        "speed": {"cohorts": SPEED, "replicates": 3},
    }
    return panel, {"prices_cpu_s_stored": price, "celegans": ce_cost,
                   "counts": {"quick": len(quick), "coverage_jax2": len(cov_jax2), "coverage_h": len(cov_h),
                              "excluded_window": len(has_window),
                              "coverage_cut": {k: len(v["cohorts"]) for k, v in cuts.items()},
                              "kmax_candidates": len(cands), "kmax_draw": len(draw), "kmax_negative": len(neg),
                              "pilot": len(pilot), "walker": len(walker["cohorts"]),
                              "walker_kappa300_shortfall": walker["kappa300_shortfall"]}}


def _toml(d, prefix=""):
    """A minimal TOML writer for the panel (strings, numbers, lists, inline tables in lists, nested tables)."""
    def val(v):
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{key(k)} = {val(x)}" for k, x in v.items()) + "}"
        raise TypeError(type(v))

    def key(k):        # a bare key only when TOML reads it as one: a cohort id's dots would make nested tables
        return k if k and all(ch.isascii() and (ch.isalnum() or ch in "_-") for ch in k) else json.dumps(k)
    lines, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict):
            tables.append((k, v))
        else:
            lines.append(f"{key(k)} = {val(v)}")
    out = "\n".join(lines) + ("\n" if lines else "")
    for k, v in tables:
        name = f"{prefix}{key(k)}"
        out += f"\n[{name}]\n" + _toml(v, f"{name}.")
    return out


# ------------------------------------------------------------------------------------------- bytes and fits
MECH_CHECKS = {   # research R5: (cohort, name) -> what must fire in the port's fit
    ("realfull-real-Sweden_F_1910_homo-c1-s1", "jax2"): lambda m: m["basin_refit"]["fired"] and m["relabelled_rows"] > 0,
    ("realfull-real-Denmark_M_1900_homo-c1-s1", "jax2"): lambda m: m["relabelled_rows"] > 0,
    ("pure-k0.01-p00-c0-s0", "jax2"): lambda m: m["N"] == 1800,
    ("pure-k0.01-p05-c0-s0", "jax2"): lambda m: m["N"] == 3600 and m["n_fallback"] > 0,
    ("pure-k0.01-p05-c0-s0", "jax2_h"): lambda m: bool(m["widened"]),
    ("hazard-k0.01-p00-c0-s0", "jax2_h"): lambda m: bool(m["widened"]),
}


def _summ(rows, key):
    ok = sum(1 for r in rows if r.get(key))
    return {"passed": ok, "total": len(rows)}


def phase_bytes(args):
    import jax2_cases as C
    import acceptance_003_port as AP
    panel = load_toml(PANEL)
    lib = OJ.library()
    ids = C.panel_ids()
    jobs = [(cid, None) for cid in ids]
    for cid in panel["bytes"]["celegans"]:
        tr = lib.truth(cid)
        th = {k: float(tr[k]) for k in C.PARAMS}
        jobs.append((cid, [("truth", th), ("eta x1.5", dict(th, eta=th["eta"] * 1.5))]))
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    run = Run("bytes", manifests=[PANEL, PANEL_V1], dev=args.dev or bool(args.only),
              cost=f"{len(jobs)} cohorts x (6 theta jax2 + 12 jax2_h) value+grad, port and oracle: about 1 CPU-min "
                   f"per cohort (compile-dominated), {args.workers} workers")
    preflight_or_exit(run)
    res = AP.run_pool(AP.bytes_worker, jobs, args.workers)
    rows = [r for c in res for r in c.get("rows", [])]
    errors = [c for c in res if "error" in c]
    checks = {}
    for name in ("jax2", "jax2_h"):
        rr = [r for r in rows if r["name"] == name]
        checks[name] = {"value": _summ(rr, "value_equal"), "vg": _summ(rr, "vg_equal"),
                        "grid_equal": _summ(rr, "grid_equal")}
    celeg = [r for r in rows if r["cohort_id"] in panel["bytes"]["celegans"]]
    expected = sum(6 + 12 if pts is None else 3 * len(pts) for _, pts in jobs)   # jax2: 1 per theta; _h: 2 h each
    ok = (not errors and all(r["value_equal"] and r["vg_equal"] and r["grid_equal"] for r in rows)
          and len(rows) == expected)
    run.write({"status": ("smoke-pass" if ok else "smoke-fail") if args.only else ("pass" if ok else "fail"),
               "checks": checks, "celegans_floored": [r["floored"] for r in celeg if r["name"] == "jax2"],
               "errors": errors, "rows": rows})


def phase_fits(args):
    import acceptance_003_port as AP
    panel = load_toml(PANEL)
    ids = list(panel["fits"]["quick"]) + [c for c in panel["fits"]["mechanisms"] if c not in panel["fits"]["quick"]]
    if args.only:
        ids = [c for c in ids if c in args.only]
    stored = {f"{c}__{n}": rec for (c, n), rec in stored_fits().items()}
    if args.stored_only:
        return phase_fits_stored(args, ids, stored)
    price = [v.get("meta", {}).get("cpu_s") for v in stored.values() if v.get("meta", {}).get("cpu_s")]
    med = float(np.median(price)) if price else 400.0
    run = Run("fits", manifests=[PANEL], dev=args.dev or bool(args.only),
              cost=f"{len(ids)} cohorts x 2 names x (port + oracle); stored Mac fits of the same runtime cost a median "
                   f"{med:.0f} CPU-s each (n = {len(price)}): about {len(ids) * 4 * med / 3600:.1f} CPU-h, "
                   f"{args.workers} workers")
    preflight_or_exit(run)
    jobs = [(cid, ("jax2", "jax2_h"), {k: v for k, v in stored.items() if k.startswith(cid + "__")}) for cid in ids]
    res = AP.run_pool(AP.fits_worker, jobs, args.workers)
    rows = [r for c in res for r in c.get("rows", [])]
    errors = [c for c in res if "error" in c]
    mech = {f"{c}__{n}": next((bool(chk(r["mechanisms"])) for r in rows if r["cohort_id"] == c and r["name"] == n), None)
            for (c, n), chk in MECH_CHECKS.items() if c in ids}
    with_stored = [r for r in rows if "stored" in r]
    checks = {k: _summ(rows, k) for k in ("theta_bytes_equal", "result_equal", "claims_equal", "sc_equal")}
    checks["stored_result_equal"] = {"passed": sum(r["stored"]["result_equal"] for r in with_stored),
                                     "total": len(with_stored)}
    checks["stored_claims_equal"] = {"passed": sum(r["stored"]["claims_equal"] for r in with_stored),
                                     "total": len(with_stored)}
    checks["mechanisms_fire"] = {"passed": sum(1 for v in mech.values() if v), "total": len(mech)}
    ok = (not errors and all(v["passed"] == v["total"] for v in checks.values())
          and len(rows) == 2 * len(ids))
    run.write({"status": ("smoke-pass" if ok else "smoke-fail") if args.only else ("pass" if ok else "fail"),
               "checks": checks, "mechanisms": mech, "errors": errors, "rows": rows,
               "cpu_ratio_port_over_oracle": [r["port_cpu_s"] / r["oracle_cpu_s"] for r in rows if r["oracle_cpu_s"]]})


def phase_fits_stored(args, ids, stored):
    """`fits --stored-only`: the port's fit against every stored Mac fit of the [fits] cohorts, with the whole diff
    (no oracle; results/fits_stored_003.json). The fits run of 2026-10-06 read R3d6/L5's NaN strings as strings, and
    its diff list stops at 4, so this re-check stands for that run's two stored checks."""
    import acceptance_003_port as AP
    jobs = [(cid, tuple(n for n in ("jax2", "jax2_h") if f"{cid}__{n}" in stored),
             {k: v for k, v in stored.items() if k.startswith(cid + "__")}) for cid in ids]
    jobs = [j for j in jobs if j[1]]
    n = sum(len(j[1]) for j in jobs)
    cpu = sum(float(v.get("meta", {}).get("cpu_s") or 0.0) for j in jobs for v in j[2].values())
    run = Run("fits", manifests=[PANEL], dev=args.dev or bool(args.only),
              cost=f"{n} port fits ({len(jobs)} cohorts) against the tester's stored Mac fits, no oracle: about "
                   f"{cpu / 3600:.1f} CPU-h on the stored CPU, {args.workers} workers")
    preflight_or_exit(run)
    res = AP.run_pool(AP.stored_worker, jobs, args.workers)
    rows = [r for c in res for r in c.get("rows", [])]
    errors = [c for c in res if "error" in c]
    checks = {k: {"passed": sum(bool(r[k]) for r in rows), "total": len(rows)}
              for k in ("result_equal", "claims_equal")}
    ok = not errors and len(rows) == n and all(v["passed"] == v["total"] for v in checks.values())
    run.write({"status": ("smoke-pass" if ok else "smoke-fail") if args.only else ("pass" if ok else "fail"),
               "checks": checks, "errors": errors, "rows": rows,
               "decoding": "R3d6/L5's non-finite strings read as floats (j2_floats)"}, name="fits_stored")


def _jax2_terms(cohort, theta, name):
    """Per-row log-likelihood terms and death masses of a jax2 calibrator at theta, on the fit's own grid (the
    evaluation floors.report_jax2 makes before refining)."""
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import fp_jax as fj
    inp = PJ.fit_inputs(cohort, name)
    t, e, L, kw = inp["t"], inp["e"], inp["L"], inp["kw"]
    z = [math.log(theta.eta), math.log(theta.beta), math.log(theta.epsilon), math.log(theta.kappa)]
    if inp["hazard"]:
        z.append(math.log(theta.h_ext))
    _f, _vg, tf = fj.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                                 sampling_dt=kw.get("sampling_dt"), hazard=inp["hazard"], window=inp["window"],
                                 with_terms=True, exact_from_F=bool(kw.get("exact_from_F")), early=kw.get("early"))
    d = tf(np.asarray(z, float))
    term, core = np.asarray(d["term"], float), np.asarray(d["core"], float)
    mass = core * kw["dt"] * L if cohort.continuous else core
    return {"term": term, "mass": mass, "total": float(np.sum(term) + float(np.asarray(d["cond"])))}


def phase_checks(args):
    """SC-009 (task T024, amended 2026-10-06): on the ITP example plus an age-0 death, AGE0_DEATHS fires under
    profile_jax2, and FLOORED_DEATHS at the theta-hat fitted without that death lists it with its refined mass and
    share of the log-likelihood (the fit with it absorbs it; the pull is recorded, not gated). CENSOR_TIE fires under
    profile_jax2 on the first medium censor-arm cohort (library order) where v0.3.3's rule fires and jax2's relabel
    does not apply, else on the recorded construction."""
    import numpy as np
    from sraging import Cohort
    from sraging.adapters.observation import cohort_from_observation
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import observation_model as om
    from sraging.warnings import _censor_tie
    run = Run("checks", manifests=[PANEL], dev=args.dev, cost="three profile_jax2 fits, combinations off (ITP n = "
              "304 with and without its age-0 death, ~1,700 CPU-s each, measured 2026-10-05; one censor-arm cohort, "
              "n 1,000-10,000): about 60-75 CPU-min")
    preflight_or_exit(run)
    import sraging
    from sraging import floors
    itp = Cohort.from_csv(os.path.join(REPO, "tests", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
    itp0 = Cohort.from_csv(os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv"), delta=1.0,
                           time_unit="days")
    r1 = sraging.fit_cohort(itp, calibrator="profile_jax2", combinations=False)   # SC-009 needs no walks
    r0 = sraging.fit_cohort(itp0, calibrator="profile_jax2", combinations=False)
    # SC-009 as amended 2026-10-06 (Naveh): the fit with the death absorbs it (its mass ~1e-6 at that theta-hat), so
    # FLOORED_DEATHS is evaluated at the theta-hat fitted WITHOUT it; AGE0_DEATHS must fire on the fit with it.
    fr1 = r1.diagnostics.get("floor_report") or {}
    at_own = [d for d in fr1.get("deaths", []) if d["age"] == 0.0]
    rep0 = floors.report_jax2(itp, r0.theta_hat, "profile_jax2")
    age0 = [d for d in rep0["deaths"] if d["age"] == 0.0]
    age0_fires = any(w.code == "AGE0_DEATHS" for w in r1.warnings)
    itp_ok = age0_fires and bool(age0) and age0[0]["mass_refined"] < 1e-12 and age0[0]["ll_share"] is not None
    t1, t0 = _jax2_terms(itp, r1.theta_hat, "profile_jax2"), _jax2_terms(itp, r0.theta_hat, "profile_jax2")
    i0 = int(np.flatnonzero(np.asarray(itp.ages, float) == 0.0)[0])
    others1, others0 = t1["total"] - t1["term"][i0], t0["total"] - t0["term"][i0]
    pull = {p: math.log10(getattr(r1.theta_hat, p) / getattr(r0.theta_hat, p)) for p in ("eta", "beta", "epsilon")}
    absorbed = {"theta_hat_with": r1.theta_hat.as_dict(), "theta_hat_without": r0.theta_hat.as_dict(),
                "loglik_with": r1.loglik, "loglik_without": r0.loglik, "shift_dex": pull,
                "shift_fold": {p: 10 ** v for p, v in pull.items()},
                "age0_term_at_with": float(t1["term"][i0]), "age0_mass_at_with": float(t1["mass"][i0]),
                "age0_term_at_without": float(t0["term"][i0]), "age0_mass_at_without": float(t0["mass"][i0]),
                "others_loglik_at_with": float(others1), "others_loglik_at_without": float(others0),
                "others_cost_nats": float(others0 - others1),
                "others": "the cohort's other deaths and censorings, on the with-age0 cohort's grid at both points",
                "listed_at_own_theta_hat": bool(at_own), "age0_deaths_fires": age0_fires}
    lib = OJ.library()
    chosen, constructed, base = None, None, None
    for cid in [c for c in sets_c()["C2"]["J"]["MEDIUM"] if c.startswith("censor-")]:
        obs = lib.observation(cid)
        if obs.window is not None:
            continue
        a, e = np.asarray(obs.ages, float), np.asarray(obs.events, int)
        rec = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=False, relabel_on=True).censor_tie
        if _censor_tie(a, e, obs.delta) and not rec.get("applied"):
            chosen = cid
            break
        if base is None and _censor_tie(a, e, obs.delta):
            base = cid
    # Measured 2026-10-05: no library cohort qualifies (0 of 1,990: every censored cohort is censored at one age, the
    # maximum, where the relabel applies). Then the staggered cohort is built by a recorded rule: the first medium
    # censor cohort where v0.3.3's rule fires, with every 10th censored row (by row order) censored instead at the
    # median death age (on the recording grid) -- censoring at two ages, the last still the maximum, the ties kept.
    if chosen is None and base is not None:
        obs = lib.observation(base)
        a, e = np.asarray(obs.ages, float).copy(), np.asarray(obs.events, int)
        cens = np.flatnonzero(e == 0)
        moved = cens[::10]
        med = float(np.median(a[e == 1]))
        if obs.delta:
            med = float(np.floor(med / obs.delta) * obs.delta)
        a[moved] = med
        constructed = {"base": base, "rule": "every 10th censored row (row order) censored at the median death age "
                       "on the recording grid", "rows_moved": int(moved.size), "censored_at": med}
        tie_cohort = Cohort.from_arrays(a, e, delta=obs.delta, name=f"{base}+staggered")
        rec = om.prepare(tuple(a), tuple(e), obs.delta, None, window_on=False, relabel_on=True).censor_tie
        constructed["v033_rule_fires"] = bool(_censor_tie(a, e, obs.delta))
        constructed["relabel_applied"] = bool(rec.get("applied"))
        chosen = tie_cohort.name
    else:
        tie_cohort = None if chosen is None else cohort_from_observation(lib.observation(chosen))
    tie_ok, tie_msg = False, None
    if tie_cohort is not None:
        r2 = sraging.fit_cohort(tie_cohort, calibrator="profile_jax2", combinations=False)
        w = next((w for w in r2.warnings if w.code == "CENSOR_TIE"), None)
        tie_ok = w is not None and "relabel did not apply" in w.message
        tie_msg = None if w is None else w.message
    ok = itp_ok and tie_ok
    run.write({"status": "pass" if ok else "fail",
               "itp_age0": {"ok": itp_ok, "rule": "SC-009 as amended 2026-10-06 (Naveh): AGE0_DEATHS fires on the "
                            "fit with the age-0 death; FLOORED_DEATHS at the theta-hat fitted without it lists it "
                            "with refined mass < 1e-12 and its share of the log-likelihood",
                            "death": age0[0] if age0 else None, "listed_at": "theta_hat_without",
                            "floor_report_without": rep0, "absorbed": absorbed},
               "censor_tie": {"ok": tie_ok, "cohort": chosen, "message": tie_msg, "constructed": constructed,
                              "rule": "first medium censor-arm cohort where v0.3.3's rule fires and the relabel does "
                                      "not; else the recorded construction"}})


C5 = os.path.dirname(J2)
NEG_FLOORS_JSON = os.path.join(C5, "sr_fp_fit_share", "benchmark_reports", "neg_floors.json")
SPACE_SEARCH = os.path.join(C5, "SRtools_kappa", "SRK", "benchmark", "space_search.py")


NEG_FLOORS_MD5 = "609c91996f3696546bc34051255a93bc"     # neg_floors.json of 2026-08-17 (research R11), pinned
FLOOR_REGIMES = ("sharp", "mid", "kappa_large", "diffuse")


def floor_coefficients(neg: dict) -> dict:
    """FR-022's coefficients, measured from neg_floors.json's beta sweep (sr-manager gate 10): per regime, c_n =
    floor_dlogl(n) * sqrt(n) at each swept n; c = their mean over the n whose floor is not right-censored
    (negligible.py `summarise`'s rule: a floor >= 0.999 * ratio_max_clean sits at the top of the grid), with the
    spread over n. If every n is censored, c is a LOWER bound, the smallest c_n (any single c above every censored
    floor is above this one; the safe direction: a smaller floor labels less). r-bar's coefficient is c * f(kappa),
    f = 1 - kappa ln(1 + 1/kappa), at the regime's own kappa (tau-mean 1)."""
    import math
    import statistics as st
    out = {}
    for s in neg["beta"]:
        vmax = s.get("ratio_max_clean", float("nan"))
        by_n, cens = {}, []
        for f in s["floors"]:
            dl = f["floor_dlogl"]
            if not math.isfinite(dl):
                continue
            by_n[int(f["n"])] = dl * math.sqrt(f["n"])
            if math.isfinite(vmax) and dl >= 0.999 * vmax:
                cens.append(int(f["n"]))
        used = [c for n, c in sorted(by_n.items()) if n not in cens]
        lower = not used
        c = min(by_n.values()) if lower else st.fmean(used)
        k = float(s["kappa"])
        fk = 1.0 - k * math.log1p(1.0 / k)
        vals = used or list(by_n.values())
        out[s["regime"]] = {"rho_eps": float(s["rho_eps"]), "kappa": k, "c_beta": c, "lower_bound": lower,
                            "c_beta_by_n": {str(n): v for n, v in sorted(by_n.items())}, "censored_n": cens,
                            "spread_over_n": {"min": min(vals), "max": max(vals), "sd": st.pstdev(vals),
                                              "max_over_min": max(vals) / min(vals), "n_used": len(used)},
                            "rbar_factor": fk, "c_rbar": c * fk}
    return out


def phase_floor(args):
    """FR-022 / T037: the negligibility floor on r-bar/(eta L_SR), by regime, MEASURED from the stored sec. 8b sweep
    (neg_floors.json, md5-pinned; `floor_coefficients`) rather than copied from space_search.py's comment, which is
    recorded beside it for comparison. Each regime of `negligible.REGIMES` has a fixed kappa and tau-mean 1, so its
    raw-beta coefficient converts to r-bar's as c * f(kappa). The diffuse regime is right-censored at every n, so its
    floor is a lower bound: a cap below it is below the true floor too. No fit, no re-run of the sweep."""
    import hashlib
    import re
    run = Run("floor", dev=args.dev, cost="seconds (reads two files)")
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()   # noqa: E731
    got = md5(NEG_FLOORS_JSON)
    regimes = floor_coefficients(json.load(open(NEG_FLOORS_JSON))) if got == NEG_FLOORS_MD5 else {}
    text = open(SPACE_SEARCH).read()
    pat = re.compile(r"#\s+(sharp|mid|kappa_large|diffuse)\s+\(rho_eps ([0-9.]+), kappa ([0-9.]+)\)\s+c (=|>) ([0-9.]+)")
    comment = {name: {"c_beta": float(c), "lower_bound": op == ">"} for name, _r, _k, op, c in pat.findall(text)}
    differs = sorted(n for n in comment if n in regimes and round(regimes[n]["c_beta"], 2) != comment[n]["c_beta"])
    ok = got == NEG_FLOORS_MD5 and set(regimes) == set(FLOOR_REGIMES)
    exact = [regimes[n] for n in regimes if not regimes[n]["lower_bound"]]
    spread = {q: {"max_over_min": max(r[q] for r in regimes.values()) / min(r[q] for r in regimes.values()),
                  "max_over_min_exact": (max(r[q] for r in exact) / min(r[q] for r in exact)) if exact else None}
              for q in ("c_beta", "c_rbar")} if regimes else {}
    run.write({"status": "written" if ok else "fail",
               "reasons": [] if got == NEG_FLOORS_MD5 else [f"neg_floors.json md5 {got} != pinned {NEG_FLOORS_MD5}"],
               "regimes": regimes, "spread_across_regimes": spread, "n_eff_cap": 2000,
               "space_search_comment": comment, "comment_differs_at_2dp": differs,
               "sources": {"space_search.py": {"path": SPACE_SEARCH, "md5": md5(SPACE_SEARCH), "lines": "66-80",
                                               "use": "comparison only"},
                           "neg_floors.json": {"path": NEG_FLOORS_JSON, "md5": got, "md5_pinned": NEG_FLOORS_MD5}},
               "rule": "c = mean over uncensored n of floor_dlogl(n) sqrt(n) (all censored: min, a lower bound); floor "
                       "on rbar/(eta L_SR) = c f(kappa) / sqrt(min(n, 2000)); the regime choice is the label's"},
              name="neg_floor")


KF = os.path.join(C5, "sr_calibration_tester_kfix")
INVARIANCE_MD = os.path.join(KF, "reports", "kfix", "invariance.md")
KF_NAMES = {"eta_over_eps": "eta_over_eps", "sqrt_eta_over_eps": "sqrt_eta_over_eps", "beta_rbar": "rbar",
            "beta_rbar_over_eta": "rbar_over_eta", "beta_over_kp1": "beta_over_kp1",
            "beta_over_kp1_eta": "beta_over_eta_kp1"}


def _invariance_tables(text):
    """{section header ('panel valid', 'panel', 'quick valid', 'quick'): {kf name: (median, p90, n)}} from kfix's report."""
    out, cur = {}, None
    for line in text.splitlines():
        if ": cohorts" in line and "kappa0 values" in line:
            cur = line.split(":")[0].strip()
            out[cur] = {}
        elif cur and line.startswith("| `") and line.count("|") == 5:
            cells = [c.strip() for c in line.strip("|").split("|")]
            out[cur].setdefault(cells[0].strip("`"), (float(cells[1]), float(cells[2]), int(cells[3])))
        elif line.startswith("| quantity | cohorts where"):
            cur = None
    return out


def phase_order(args):
    """FR-016 / T033 (Naveh's statistic, 2026-10-05): the lead group is the combinations invariant on the scaling ridge
    by construction whose median spread along the stored kfix kappa-slices is <= 0.05 dex on the validity-valid panel
    (19 cohorts); the rest follow by that spread; ties (equal to 3 decimals) would be broken by the walker pilot's
    median widths. The spreads are kfix's own measurement (reports/kfix/invariance.md, its commit and md5 recorded);
    every cohort set's spreads are reported."""
    import hashlib
    from sraging.claims import combinations as K
    run = Run("order", dev=args.dev, cost="seconds (reads kfix's committed report; runs nothing there)")
    text = open(INVARIANCE_MD).read()
    tables = _invariance_tables(text)
    valid = tables["panel valid"]
    spread = {KF_NAMES[n]: v[0] for n, v in valid.items() if n in KF_NAMES}
    lead = sorted((k for k in K.KEYS if k in K.RIDGE_INVARIANT and spread[k] <= 0.05), key=lambda k: spread[k])
    rest = sorted((k for k in K.KEYS if k not in lead), key=lambda k: spread[k])
    order = lead + rest
    ties = [(a, b) for a, b in zip(order, order[1:]) if (a in lead) == (b in lead) and spread[a] == spread[b]]
    departures = [(i, a, b) for i, (a, b) in enumerate(zip(order, K.HYPOTHESIS_ORDER)) if a != b]
    run.write({"status": "written" if not ties else "ties: break them with the walker pilot's widths",
               "order": order, "lead_group": lead, "spread_valid_panel_dex": spread,
               "all_sets": {sec: {KF_NAMES.get(n, n): v for n, v in t.items()} for sec, t in tables.items()},
               "hypothesis": list(K.HYPOTHESIS_ORDER), "departures_from_hypothesis": departures, "ties": ties,
               "source": {"path": INVARIANCE_MD, "md5": hashlib.md5(text.encode()).hexdigest(),
                          "kf_commit": git("log", "-1", "--format=%H", "--", "reports/kfix/invariance.md", repo=KF)},
               "rule": "lead = ridge-invariant by construction and <= 0.05 dex on the validity-valid panel; rest by spread"})


def oracle_claimset(name, cid, claims_kw, sc_block=None):
    """The frozen method's OWN ClaimSet for one fit: its `claimset` method (mle_profile_jax2/__init__.py `claimset`,
    the tester's `Claim(**kw)` mapping, diagnostics and method name), run on the fit's claim keywords. Only the fit
    itself is stood in for: `final_claim_kwargs` (called by `claimset` on the run_fit result, which the fits phase does
    not keep) is replaced for the call by the recorded claims, then restored. Nothing of sraging's adapter is used."""
    import types
    from srbench.provenance import CodeProvenance
    o = OJ.load()
    ad = o.adapter
    inst = o.classes[name]()
    prov = CodeProvenance(package="mle_profile_jax2", resolved_path="oracle", revision="3bc5775", dirty=False)
    orig = ad.final_claim_kwargs
    ad.final_claim_kwargs = lambda res, sc: (list(claims_kw), sc_block)
    try:
        obs = types.SimpleNamespace(cohort_id=cid)
        return inst.claimset(obs, {"diagnostics": {}, "loglik": None, "dt_ceiling": None, "wall_s": 0.0},
                             prov, time.perf_counter(), time.process_time())
    finally:
        ad.final_claim_kwargs = orig


def _four(cs):
    return tuple(c for c in cs.claims if c.parameter in ("eta", "beta", "epsilon", "kappa"))


def phase_adapter(args):
    """SC-003 / T042: the adapter's records of the port, built from the `fits` phase's port claims by sraging's
    `claimset_from_kwargs`, are valid tester ClaimSets, EQUAL (claim for claim) to the ClaimSets the frozen method
    itself builds from the oracle's claims (`oracle_claimset`: the frozen `claimset`, not sraging's code), and the
    tester's own coverage scoring of the two is equal on every parameter and shape. No refit: the claims come from
    results/fits_003.json. The tester is the read-only clone pinned to AD.J2_HEAD (refused if its HEAD differs or its
    tree is dirty); the library's data files are untracked in J2, so they are read from the live checkout."""
    from sraging.adapters import srbench as AD
    fits = json.load(open(os.path.join(RESULTS, "fits_003.json")))
    run = Run("adapter", manifests=[PANEL], dev=args.dev or fits.get("status_for_acceptance") != "acceptance",
              cost="minutes: no fit (the fits phase's claims), the tester's scoring")
    bad = AD.pinned_problems(AD.J2_TESTER, AD.J2_HEAD)
    if bad:
        run.write({"status": "not_applicable", "reasons": bad})
        sys.exit(3)
    import sraging.compat.jax2  # noqa: F401  (configures the runtime before the oracle's fp_jax)
    preflight_or_exit(run)                       # loads the frozen oracle (its srbench) before the tester's is joined
    mods = AD.connect(AD.J2_TESTER, AD.J2_HEAD, require_clean=True)
    import importlib
    T = mods["contract.types"]
    V = importlib.import_module("srbench.contract.validate")
    C = mods["score.claims"]
    lib = mods["library.load"].Library(root=OJ.LIBRARY_V2, version="v2")
    quick = set(load_toml(PANEL)["fits"]["quick"])
    out, ok = {}, True
    for name in ("jax2", "jax2_h"):
        mname = f"sraging_profile_{name}"
        desc = T.MethodDescriptor(**AD.descriptor_fields(mname))
        rows = [r for r in fits["rows"] if r["name"] == name and r["cohort_id"] in quick]
        port = [AD.claimset_from_kwargs(T, r["cohort_id"], mname, r["port_claims"], {}) for r in rows]
        orc = [oracle_claimset(name, r["cohort_id"], r["oracle_claims"]) for r in rows]
        invalid = []
        for cs in port + orc:
            try:
                V.validate_claimset(cs, desc if cs.method_name == mname else _same_name(T, cs, desc))
            except Exception as ex:  # noqa: BLE001
                invalid.append((cs.method_name, cs.cohort_id, str(ex)[:200]))
        unequal = [(a.cohort_id, [(x, y) for x, y in zip(_four(a), _four(b)) if x != y][:2])
                   for a, b in zip(port, orc) if _four(a) != _four(b)]
        truths = lib.truths([r["cohort_id"] for r in rows])
        scores = {}
        for p in ("eta", "beta", "epsilon", "kappa"):
            a = C.coverage_all_shapes(port, truths, p)
            b = C.coverage_all_shapes(orc, truths, p)
            fa = {k: {f: getattr(v, f) for f in ("hits", "claimed", "declined", "failed", "total")} for k, v in a.items()}
            fb = {k: {f: getattr(v, f) for f in ("hits", "claimed", "declined", "failed", "total")} for k, v in b.items()}
            scores[p] = {"port": fa, "oracle": fb, "equal": fa == fb}
        good = (not invalid and not unequal and len(rows) == len(quick) and all(v["equal"] for v in scores.values()))
        ok = ok and good
        out[name] = {"n": len(rows), "invalid": invalid, "claims_unequal": unequal, "scores": scores, "ok": good}
    run.write({"status": "pass" if ok else "fail", "methods": out,
               "adapter_tester": {"root": AD.J2_TESTER, "head": git("rev-parse", "HEAD", repo=AD.J2_TESTER),
                                  "clean": True, "library_root": OJ.LIBRARY_V2,
                                  "library_note": "data files untracked in J2: read from the live checkout, read-only"},
               "oracle_side": "frozen method's claimset() on the oracle's claims (oracle_claimset)",
               "fits_result": {"repo_head": fits["repo_head"], "status": fits["status"]}})


def _same_name(T, cs, desc):
    """The descriptor under the ClaimSet's own method name (validation checks the name; the bounds are the same)."""
    return T.MethodDescriptor(**{**{f: getattr(desc, f) for f in ("version", "bounds", "capabilities", "description",
                                                                  "contact")}, "name": cs.method_name})


def phase_walker(args):
    import acceptance_003_walker as WK
    WK.phase_walker(args)


def phase_coverage(args):
    import acceptance_003_coverage as CV
    CV.phase_coverage(args)


def phase_coverage_xhost(args):
    import acceptance_003_coverage as CV
    CV.phase_xhost(args)


def phase_kmax(args):
    import acceptance_003_kmax as KM
    KM.phase_kmax(args)


def phase_speed(args):
    import acceptance_003_speed as SP
    SP.phase_speed(args)


# ------------------------------------------------------------------------------------------- unchanged (SC-011)
V033_ROOTS = ("sraging.calibrators.profile_v033", "sraging.adapters.observation", "sraging.adapters.srbench",
              "sraging.cohort", "sraging.compat.v033.estimator", "sraging.compat.v033.fp_jax")
CLAIMS_TOOLS = ("acceptance", "acceptance_claims", "parity_checks", "oracle_v033")   # tools/: spec 001's claims phase
UNTOUCHED = ("src/sraging/compat/v033", "src/sraging/engine", "tests/oracle", "tools/oracle_v033.py",
             "tools/oracle_fp_piecewise.py")
#: the additions FR-030 permits under tests/oracle: spec 003's jax2 1.0.0 snapshot (research R2), and spec 003b's two
#: (Naveh D2, 2026-10-19: the jax2 1.0.1 snapshot and the srbench tree it runs against; spec 003b FR-002)
NEW_ORACLE = ("tests/oracle/jax2_3bc5775/", "tests/oracle/jax2_6884fc8/", "tests/oracle/tester_f94ec2b/")
BASE = "main"
#: spec 004 T144 (Naveh 2026-10-29, spec 004 Clarifications "on spec 001's claims re-run"): the engine files a later
#: spec creates or changes on purpose, each with its task. A listed file is recorded under untouched["spec_exempt"],
#: never as a violation, and sets rerun_required; the phase passes only if the claims re-run is then 40/40 on claims
#: identity, theta-hat, value bytes and gradient bytes (claims_exempt_ok), and short of that the listed files are
#: violations again (exempt_violations). Only files under ENGINE_PREFIX can be exempt: compat/v033, tests/oracle, the
#: oracle tools and every unlisted engine file stay strict. stepper.py and stepper_gen.py are not on the list.
SPEC_ENGINE_CHANGES = {"004": (("src/sraging/engine/inputs.py", "T017"), ("src/sraging/engine/quadrature.py", "T016"),
                               ("src/sraging/engine/mixture.py", "T018-T020, T034"),
                               ("src/sraging/engine/evaluator.py", "T020, T034"),
                               ("src/sraging/engine/__init__.py", "T020, T034"), ("src/sraging/engine/pool.py", "T031"),
                               ("src/sraging/engine/cost.py", "T032"), ("src/sraging/engine/series.py", "T033"),
                               ("src/sraging/engine/density.py", "T094"),
                               ("src/sraging/engine/tensor_reference.py", "T107")),
                       # spec 006 (FR-017; C9): the window at hi = inf (D-601), the v1 clip of S(t_lo) (D-108) and
                       # Evaluator.window_report (specs/006-calibrator-capabilities/tasks.md T034)
                       "006": (("src/sraging/engine/observation.py", "T034"),
                               ("src/sraging/engine/evaluator.py", "T034"))}
ENGINE_PREFIX = "src/sraging/engine/"
#: Naveh 2026-11-05 (beside T144's list): the oracle tool spec 004 extended with SRK read-only operations, with its tasks
#: and commits. It is exempt (untouched["spec_exempt"], rule "additions-only") iff its diff from the merge base deletes
#: or changes no line, every added top-level statement is a new `def _op_*` or helper `def _*` (no existing name
#: redefined, no module-level import or other code) plus new entries in its operations table (ORACLE_OPS_TABLE), and the
#: claims re-run is 40/40 (exempt_violations, as the engine files). Every other oracle tool, tests/oracle and compat/v033
#: stay strict.
SPEC_ORACLE_ADDITIONS = {"004": (("tools/oracle_fp_piecewise.py", "T070, T072, T074", "66bcc26, e893cc3, 1106277"),)}
ORACLE_OPS_TABLE = "OPS"
CLAIMS_EXEMPT_CHECKS = ("claims_identity", "theta_hat", "value_bytes", "grad_bytes")
CLAIMS_N = 40


def _module_file(name):
    """The repo file of module `name`: src/sraging/... for sraging.*, tools/<name>.py for a tool; None otherwise."""
    if name == "sraging" or name.startswith("sraging."):
        base = os.path.join(REPO, "src", *name.split("."))
        for p in (base + ".py", os.path.join(base, "__init__.py")):
            if os.path.exists(p):
                return p
        return None
    p = os.path.join(REPO, "tools", f"{name}.py")
    return p if "." not in name and os.path.exists(p) else None


def import_closure(roots=V033_ROOTS + CLAIMS_TOOLS) -> list[str]:
    """Repo-relative files of every sraging module and tool reachable from `roots` by ANY import statement (function-
    level ones included) plus each sraging package's __init__.py (it runs on import). Static, so an over-approximation:
    the safe side for deciding that nothing the claims depend on changed."""
    import ast
    seen, todo = {}, list(roots)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        f = _module_file(name)
        seen[name] = f
        if f is None:
            continue
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts)) if parts[0] == "sraging"]
        pkg = parts if f.endswith("__init__.py") else parts[:-1]
        for node in ast.walk(ast.parse(open(f).read(), f)):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (pkg[:len(pkg) - node.level + 1] if node.level else []) + (node.module.split(".")
                                                                                  if node.module else [])
                todo.append(".".join(base))
                todo += [".".join(base + [a.name]) for a in node.names]
    return sorted(os.path.relpath(f, REPO) for f in seen.values() if f is not None)


def without_docstrings(src: str) -> str | None:
    """`ast.dump` of `src` with every module, class and function docstring removed (comments never reach the tree);
    None if it does not parse."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def docstring_only(path: str, base: str) -> bool:
    """Spec 003g T075 (Naveh 2026-10-29): a modified `src/` .py file whose syntax tree, docstrings removed, equals the
    merge base's, i.e. a change of comments and docstrings only. The oracles stay byte-frozen: never exempt."""
    if not (path.startswith("src/") and path.endswith(".py")):
        return False
    a, b = (without_docstrings(git("show", f"{rev}:{path}")) for rev in (base, "HEAD"))
    return a is not None and a == b


def _spec_listed(path):
    """(spec, task) of an engine file a later spec names in SPEC_ENGINE_CHANGES, else None (never outside the engine)."""
    if not path.startswith(ENGINE_PREFIX):
        return None
    return next(((spec, task) for spec, files in SPEC_ENGINE_CHANGES.items() for f, task in files if f == path), None)


def _oracle_listed(path):
    """(spec, tasks, commits) of an oracle tool SPEC_ORACLE_ADDITIONS names, else None."""
    return next(((spec, tasks, commits) for spec, files in SPEC_ORACLE_ADDITIONS.items()
                 for f, tasks, commits in files if f == path), None)


def _top_names(stmt) -> set:
    """The module-level names a top-level statement binds."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [getattr(stmt, "target", None)]
    return {n.id for t in targets if t is not None for n in ast.walk(t) if isinstance(n, ast.Name)}


def _ops_table(stmt):
    """The operations table's dict literal if `stmt` is `ORACLE_OPS_TABLE = {...}`, else None."""
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == ORACLE_OPS_TABLE and isinstance(stmt.value, ast.Dict)):
        return stmt.value
    return None


def oracle_additions_problems(path, base) -> list[str]:
    """Why `path`'s change from `base` to HEAD is not additions-only (empty: it is). From the diff: no line deleted or
    changed (numstat deletions 0 and no '-' line in the hunks). From the syntax trees: every top-level statement of
    `base` is still there, unchanged, but the operations table, which may only gain entries (a new string key to a
    module-level `_op_*` function); every added statement is a new `def _op_*` or `def _*` whose name `base` does not
    bind and that is defined once."""
    out = []
    raw = git("diff", "--numstat", base, "HEAD", "--", path)
    if not raw.strip():
        return out                                              # unchanged: nothing to judge
    num = raw.split("\t")
    if len(num) < 2 or not num[1].isdigit() or int(num[1]) != 0:
        out.append(f"the diff deletes or changes lines (numstat {num[:2]})")
    minus = [x for x in git("diff", "-U0", base, "HEAD", "--", path).splitlines()
             if x.startswith("-") and not x.startswith("---")]
    if minus:
        out.append(f"the diff deletes or changes {len(minus)} line(s), e.g. {minus[0][:80]!r}")
    try:
        old, new = ast.parse(git("show", f"{base}:{path}")), ast.parse(git("show", f"HEAD:{path}"))
    except SyntaxError as ex:
        return out + [f"does not parse: {ex}"]
    old_names = set().union(*(_top_names(x) for x in old.body)) if old.body else set()
    old_dumps = [ast.dump(x) for x in old.body]
    old_table = next((t for t in map(_ops_table, old.body) if t is not None), None)
    funcs = {x.name for x in new.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
    seen_new = []
    for stmt in new.body:
        d = ast.dump(stmt)
        if d in old_dumps:
            old_dumps.remove(d)
            continue
        table = _ops_table(stmt)
        if table is not None and old_table is not None:
            pair = lambda k, v: (ast.dump(k), ast.dump(v))                                    # noqa: E731
            olds = [pair(k, v) for k, v in zip(old_table.keys, old_table.values)]
            news = [pair(k, v) for k, v in zip(table.keys, table.values)]
            old_keys = {ast.dump(k) for k in old_table.keys}
            added = [(k, v) for k, v in zip(table.keys, table.values) if pair(k, v) not in olds]
            if [p_ for p_ in news if p_ in olds] != olds:
                out.append(f"the operations table {ORACLE_OPS_TABLE} changes or drops an existing entry")
            for k, v in added:
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str) and ast.dump(k) not in old_keys
                        and isinstance(v, ast.Name) and v.id.startswith("_op_") and v.id in funcs):
                    out.append(f"an operations-table entry is not a new key to a module-level _op_ function: "
                               f"{ast.unparse(k)}: {ast.unparse(v)}")
            old_dumps.remove(ast.dump(next(x for x in old.body if _ops_table(x) is not None)))
            continue
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("_"):
            if stmt.name in old_names:
                out.append(f"redefines the existing name {stmt.name}")
            elif stmt.name in seen_new:
                out.append(f"defines {stmt.name} twice")
            seen_new.append(stmt.name)
            continue
        out.append(f"line {stmt.lineno}: not a new function `def _op_*` or `def _*` "
                   f"({type(stmt).__name__}: {ast.unparse(stmt).splitlines()[0][:60]!r})")
    if old_dumps:
        out.append(f"{len(old_dumps)} top-level statement(s) of {base} changed or removed")
    return out


def merge_base() -> str:
    """Where this branch left BASE (`git merge-base main HEAD`): the one base of every `unchanged` diff (the untouched
    paths, the claims closure, the oracle tool's additions). Refused by name when git cannot resolve it: in a clone with
    no local main every diff against main printed nothing, and `unchanged` read "none changed" and skipped spec 001's
    claims re-run (found at spec 004 T145). A missing base never reads as no change."""
    base = git("merge-base", BASE, "HEAD")
    if not base:
        sys.exit(f"[unchanged] REFUSED: no merge-base with {BASE} (create or fetch the {BASE} branch)")
    return base


def untouched_problems(docstring_only_out: list | None = None, spec_exempt_out: list | None = None,
                       base: str | None = None) -> list[str]:
    """FR-030 / SC-011: no change to compat/v033, the engine or the existing oracles on this branch (merge base with
    main to HEAD); only additions under the permitted oracle snapshots (NEW_ORACLE) are allowed, and (spec 003g T075)
    modifications of `src/` files that change comments and docstrings only, each appended to `docstring_only_out`.
    Spec 004 T144: an engine file SPEC_ENGINE_CHANGES names is appended to `spec_exempt_out` ({path, status, spec,
    task}) instead of the violations, when the caller collects them (and so owes the claims re-run's 40/40,
    exempt_violations); without a collector it stays a violation. `base` is `merge_base()` unless given."""
    out, base = [], base or merge_base()
    for line in git("diff", "--name-status", base, "HEAD", "--", *UNTOUCHED).splitlines():
        st, path = line.split("\t", 1)
        if st == "A" and path.startswith(NEW_ORACLE):
            continue
        listed = _spec_listed(path)
        if listed is not None and spec_exempt_out is not None:
            spec_exempt_out.append({"path": path, "status": st, "spec": listed[0], "task": listed[1]})
            continue
        oracle = _oracle_listed(path)
        if oracle is not None and spec_exempt_out is not None and st == "M":
            if not oracle_additions_problems(path, base):
                spec_exempt_out.append({"path": path, "status": st, "spec": oracle[0], "task": oracle[1],
                                        "commits": oracle[2], "rule": "additions-only"})
                continue
        if st == "M":
            if docstring_only(path, base):
                if docstring_only_out is not None:
                    docstring_only_out.append(path)
                continue
        out.append(line)
    return out


def claims_exempt_ok(checks) -> bool:
    """T144: the claims re-run is 40/40 on claims identity, theta-hat, value bytes and gradient bytes."""
    return isinstance(checks, dict) and all(
        isinstance(checks.get(k), dict) and checks[k].get("passed") == checks[k].get("total") == CLAIMS_N
        for k in CLAIMS_EXEMPT_CHECKS)


def exempt_violations(spec_exempt, checks) -> list[str]:
    """T144: the spec-exempt engine files as violations again ("status\tpath", as untouched_problems writes them)
    unless the claims re-run is 40/40 on all four checks."""
    return [] if claims_exempt_ok(checks) else [f"{e['status']}\t{e['path']}" for e in spec_exempt]


def phase_unchanged(args):
    """SC-011 (FR-030; research R18; sr-manager gate 6): git shows compat/v033, the engine and the oracles untouched,
    and spec 001's claims acceptance (tools/acceptance.py `claims`, 40 cohorts, port against the frozen v0.3.3 oracle)
    is re-run on this branch, written to results/claims-001_003.json with this phase's provenance. Recorded as evidence
    beside it: the import closure of profile_v033's fit path and spec 001's claims tools, and the files of it that
    changed on this branch (against main) and since spec 002's re-run. The re-run is skipped only when that closure
    diff is empty (nothing the claims depend on changed). The srbench record is not re-run (no store is written)."""
    import math
    import types
    import acceptance as A1
    import acceptance_claims as AC
    base = merge_base()                                 # refused by name when main cannot be resolved (T145)
    pv1 = load_toml(PANEL_V1)
    ids = list(pv1["quick"]) + [n["cohort_id"] for n in pv1["named"]]
    if args.only:
        ids = [c for c in ids if c in args.only]
    closure = import_closure()
    changed = git("diff", "--name-only", base, "HEAD", "--", *closure).split()
    prev = os.path.join(RESULTS, "claims-001_002.json")
    prev_head = json.load(open(prev)).get("repo_head") if os.path.exists(prev) else None
    # evidence only; an unresolvable spec 002 head records None (not known), never an empty list (T145)
    since_prev = (git("diff", "--name-only", f"{prev_head}..HEAD", "--", *closure).split()
                  if prev_head and git("rev-parse", "--verify", "--quiet", f"{prev_head}^{{commit}}") else None)
    doc_only, exempt = [], []
    violations = untouched_problems(doc_only, exempt, base)
    rerun = bool(changed) or bool(exempt) or bool(args.only)
    known = [s for s in (A1.stored_cpu(c) for c in ids) if math.isfinite(s)]
    run = Run("unchanged", manifests=[PANEL_V1], dev=args.dev or bool(args.only),
              cost=(f"spec 001's claims re-run, {len(ids)} cohorts x 2 sides: {2 * sum(known) / 3600:.1f} CPU-h on the "
                    f"stored v0.3.3 CPU ({len(ids) - len(known)} unknown), {args.workers} workers; {len(changed)} of "
                    f"{len(closure)} closure files changed against {BASE}") if rerun else
              f"seconds: git only ({len(closure)} closure files, none changed against {BASE})")
    evidence = {"closure": {"roots": list(V033_ROOTS + CLAIMS_TOOLS), "files": closure, "base": BASE,
                            "base_commit": base, "changed_vs_base": changed,
                            "spec002_rerun_head": prev_head, "changed_since_spec002_rerun": since_prev},
                "untouched": {"paths": list(UNTOUCHED), "allowed_additions": NEW_ORACLE,
                              "violations": violations, "docstring_only": doc_only, "spec_exempt": exempt,
                              "spec_engine_changes": {k: [list(x) for x in v] for k, v in SPEC_ENGINE_CHANGES.items()},
                              "spec_oracle_additions": {k: [list(x) for x in v] for k, v in SPEC_ORACLE_ADDITIONS.items()},
                              "oracle_additions_problems": {
                                  f: oracle_additions_problems(f, base)
                                  for files in SPEC_ORACLE_ADDITIONS.values() for f, *_ in files}},
                "rerun_required": bool(changed) or bool(exempt)}
    untouched_ok = not evidence["untouched"]["violations"]
    if not rerun:
        run.write({"status": "pass" if untouched_ok else "fail", "claims_status": "not re-run: closure unchanged",
                   **evidence}, name="claims-001")
        return
    problems = O.preflight()                           # checked here: spec 001's own exits write results/claims.json
    ok, info = AC.panel_status()
    if not ok and not args.only:
        problems.append(info)
    if problems:
        run.write({"status": "not_applicable", "reasons": problems, **evidence}, name="claims-001")
        sys.exit(3)
    reason = (f"SC-011 re-runs spec 001's claims identity; the srbench record (spec 001 SC-004) is not re-run, so no "
              f"srbench store is written (the tester checkout is at {git('rev-parse', 'HEAD', repo=O.TESTER)[:7]})")
    AC.phase_claims(types.SimpleNamespace(only=args.only, workers=args.workers, no_srbench=reason,
                                          result_name="claims-001_003"))
    d = json.load(open(os.path.join(RESULTS, "claims-001_003.json")))
    inner = str(d.get("status", ""))
    again = exempt_violations(exempt, d.get("checks"))          # T144: short of 40/40 the listed files count again
    if again:
        evidence["untouched"]["violations"] = violations + again
        untouched_ok = False
    good = inner.removeprefix("smoke-") == "pass" and untouched_ok
    run.write({**{k: v for k, v in d.items() if k not in ("phase", "written_at", "repo_head", "host", "tester",
                                                          "status")},
               "status": ("smoke-" if args.only else "") + ("pass" if good else "fail"), "claims_status": inner,
               "claims_run": {"written_at": d.get("written_at"), "repo_head": d.get("repo_head"),
                              "tester_v033": d.get("tester")}, **evidence}, name="claims-001")


PHASES = {"pilot": phase_pilot, "bytes": phase_bytes, "fits": phase_fits, "speed": phase_speed,
          "checks": phase_checks, "floor": phase_floor, "order": phase_order,
          "kmax": phase_kmax, "coverage": phase_coverage, "coverage-xhost": phase_coverage_xhost,
          "walker": phase_walker, "adapter": phase_adapter, "unchanged": phase_unchanged}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--dev", action="store_true", help="development run: never counts")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", nargs="*", help="restrict to these cohorts (a smoke run; never counts)")
    ap.add_argument("--walker-step", type=int, choices=(1, 2), default=1,
                    help="walker: 1 = the fits and walks (cost, widths); 2 = the brute force, after Naveh decides")
    ap.add_argument("--n-sobol", type=int, default=32, help="walker: Sobol starts per brute-force point")
    ap.add_argument("--score-only", action="store_true",
                    help="coverage: score the rows imported from the cluster (acceptance_003_coverage.py import-rows); "
                         "fit nothing here")
    ap.add_argument("--stored-only", action="store_true",
                    help="fits: only the port against the tester's stored Mac fits (results/fits_stored_003.json)")
    args = ap.parse_args(argv)
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
