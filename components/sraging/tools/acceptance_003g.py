"""Spec 003g's acceptance phases (specs/003g-default-switch/contracts/acceptance.md). Mac only; nothing runs on WEXAC;
nothing is written to the tester checkout, any tester store or `data/store`; v0.3.3's v6 records are read only by the
`predecessor` phase, and only once the panel's [seal] names jax2's gate run.

Before W1:   panel, v033-points-verify, v033-mle-verify, lev-pilot, p15-v033, fit1
Each window: pin-w1 -> sut -> score-w1 (+ null-local) -> fit2-pick -> fit2;  [seal] -> predecessor;
             pin-w2 -> score-w2;  pin-w3 -> score-w3
After W3:    gate, readme, unchanged

Every phase runs through spec 003's `acceptance_003.Run` (suffix "003g"): a clean committed tree, its price printed
first, its result in tests/acceptance/results/<phase>_003g.json; every row carries research R14's provenance.
"""
from __future__ import annotations

import argparse
import hashlib
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

PANEL_PATH = "tests/acceptance/panel_003g.toml"
IDS_PATH = "tests/acceptance/panel_003g_ids.tsv"
PINS_PATH = "tests/acceptance/pins_003g.json"
RESULTS_DIR = os.path.join(REPO, "tests", "acceptance", "results")
DROPBOX = os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset"   # release 0.1.1: the lab's checkouts
TESTER = os.path.join(DROPBOX, "sr_calibration_tester")                 # data root (read-only; never git here)
GATEV6 = os.path.join(DROPBOX, "sr_calibration_tester_jax2", "reports", "rc2", "GATEV6")
WITNESS_MIRROR = os.path.join(GATEV6, "cluster_home", "witness")     # jax2's pull mirror (fallback only)
EXPORT_V6 = os.path.join(GATEV6, "export_v6")          # jax2's per-window export (sesync cd40): <W>/witness/


def witness_root(window: str) -> str:
    """Where a window's witness files are read: jax2's export for it when present, else the pull mirror."""
    d = os.path.join(EXPORT_V6, window, "witness")
    return d if os.path.isdir(d) else WITNESS_MIRROR
MANIFESTS = [PANEL_PATH, IDS_PATH, PINS_PATH]


class Refused(SystemExit):
    """A phase that must not run (a prerequisite missing, the seal closed, an input unpinned)."""


def panel() -> dict:
    return tomllib.load(open(os.path.join(REPO, PANEL_PATH), "rb"))


def pins_data() -> dict:
    return json.load(open(os.path.join(REPO, PINS_PATH)))


def _git(*a, repo=REPO):
    return subprocess.run(["git", "--no-optional-locks", "-C", repo, *a], capture_output=True, text=True).stdout.strip()


def cpu_model() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return platform.processor()


def tester_clone(head: str) -> str:
    """The read-only clone of the tester commit jax2 names for a window (003e's connect_pinned refuses another HEAD
    or a dirty tree)."""
    return os.path.join(REPO, ".cache", f"tester_{head[:7]}")


def lock_sha256(version: str) -> str | None:
    snap = {"1.0.0": "jax2_3bc5775", "1.0.1": "jax2_6884fc8"}.get(version)
    p = os.path.join(REPO, "tests", "oracle", snap or "-", "mle_profile_jax2", "VERSION_LOCK.json")
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def row_provenance(*, calibrator, version, library="v6", run_manifest=None, run_manifest_sha256=None,
                   record_sha256=None, witness_sha256=None, record_host=None, record_cpu=None) -> dict:
    """Research R14: what every 003g row carries. N10 (Naveh 2026-10-29): `scored_on_*` is where this phase ran;
    `record_*` is where a stored record was computed (its own code_provenance), None for a fit made here."""
    pins = pins_data()
    return {"calibrator": calibrator, "version": version, "lock_sha256": lock_sha256(version), "library": library,
            "manifest_sha256": (pins.get("libraries", {}).get(library) or {}).get("sha256"),
            "run_manifest": run_manifest, "run_manifest_sha256": run_manifest_sha256, "record_sha256": record_sha256,
            "witness_sha256": witness_sha256, "scored_on_host": platform.node(), "scored_on_cpu": cpu_model(),
            "record_host": record_host, "record_cpu": record_cpu}


# ------------------------------------------------------------------------------------------------------- the seal
def require_unsealed(pan: dict) -> str:
    """The evidence that jax2's gate run happened (research R5); refused while the seal is closed."""
    ev = (pan.get("seal") or {}).get("jax033_v6", "")
    if not ev.strip():
        raise Refused("REFUSED: v0.3.3's v6 records stay sealed until jax2's gate run is on record "
                      "(panel_003g.toml [seal] jax033_v6 is empty; tasks T042)")
    return ev


# --------------------------------------------------------------------------------------------------- phase frame
def _run(phase, args, cost):
    import acceptance_003 as A
    return A.Run(phase, manifests=MANIFESTS, dev=args.dev, suffix="003g", cost=cost)


def _todo(name):
    def phase(args):
        sys.exit(f"{name}: not implemented yet (specs/003g-default-switch/tasks.md)")
    return phase


def phase_panel(args):
    """SC-001's first half: the committed panel is the rebuild from the pinned sources, and its counts are the spec's."""
    import panel_003g as P
    run = _run("panel", args, "0 fits; ~10 s (reads the six sources and the 1,820 J-domain v6 arrays)")
    pan = panel()
    diffs = P.check(pan)
    rows = P.read_ids(os.path.join(REPO, IDS_PATH))
    counts = {k: v["count"] for k, v in pan["strata"].items()}
    run.write({"status": "pass" if not diffs else "fail", "differences": diffs, "counts": counts, "rows": len(rows),
               "ids_sha256": pan["sources"]["ids"]["sha256"], "rules": pan["rules"], "naveh_ok": pan["naveh_ok"]})


def v6_cohort(cohort_id, lib=None):
    """A library cohort as sraging's Cohort. `lib`: a Library already loaded in this process (the pinned tester
    clone's, when a phase has it open: one srbench per process); else 003b's srbench snapshot on v6. Either Library
    checks the array's content digest against its manifest on load."""
    from sraging.adapters.observation import cohort_from_observation
    if lib is None:
        import oracle_jax2_101 as OJ1
        lib = OJ1.library("v6")
    return cohort_from_observation(lib.observation(cohort_id))


def j_view(cohort_id, lib=None):
    """J's own view of a library cohort: J does not model fit windows (its class's WINDOW is off; the census's "J view:
    window off"), so a realfull cohort's window is set aside exactly as jax2's record set it aside. (sraging Cohort
    without the window, the port's Observation with it -- fit_inputs prepares it with J's switches --, the window)."""
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    if lib is None:
        import oracle_jax2_101 as OJ1
        lib = OJ1.library("v6")
    obs = lib.observation(cohort_id)
    w = getattr(obs, "window", None)
    delta = obs.delta if obs.delta else None
    c = Cohort.from_arrays(list(obs.ages), list(obs.events), delta=delta, continuous=delta is None, name=cohort_id)
    o = PJ.Observation(cohort_id=cohort_id, ages=tuple(float(a) for a in obs.ages),
                       events=tuple(int(e) for e in obs.events), delta=obs.delta, window=w)
    return c, o, w


def _fit_summary(r) -> dict:
    """What a cross-check fit records (data-model "Cross-check fit"): claims, the leverage block's label and reason,
    notes, mechanisms, DT_RULE scope and the fit's own timers."""
    d = json.loads(r.to_json())
    dg = d.get("diagnostics") or {}
    rule = (dg.get("dt_floor") or {}).get("grid_rule") or {}
    lev = dg.get("leverage") or {}
    return {"status": d.get("status"), "claims": d.get("claims"), "theta_hat": d.get("theta_hat"),
            "loglik": d.get("loglik"), "timing": d.get("timing"), "combinations": d.get("combinations"),
            "mechanisms": d.get("mechanisms"), "dt_in_scope": rule.get("dt_in_scope"), "cv_c": rule.get("cv_c"),
            "leverage": {k: lev.get(k) for k in ("label", "reason", "g0", "warm_termination", "at_theta_hat",
                                                 "dropped", "R", "delta_total", "cost")},
            "notes": dg.get("notes"), "warnings": [w.get("code") for w in d.get("warnings") or []],
            "basin_fix_fired": bool((dg.get("basin_fix") or {}).get("fired"))}


def phase_fit1(args):
    """S4 fit 1 (FR-009; research R10): the default settings on the in-scope sharp cohort, priced before it runs. Its
    FitResult goes to sraging's own store (.cache/store_003g), never a tester store; its comparison with W3's record
    is score-w3's."""
    import time
    import sraging
    s4 = panel()["s4"]["fit1"]
    run = _run("fit1", args, f"1 Mac fit, {s4['price_cpu_h']} CPU-h; keep the lid open")
    cid = s4["cohort"]
    c = v6_cohort(cid)
    t0, c0 = time.time(), time.process_time()
    r = sraging.fit_cohort(c, calibrator=s4["calibrator"], version=s4["version"])
    cpu, wall = time.process_time() - c0, time.time() - t0
    store = os.path.join(REPO, ".cache", "store_003g")
    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, f"fit1_{cid}.json")
    with open(path, "w") as fh:
        fh.write(r.to_json())
    run.write({"status": "written", "cohort_id": cid, "settings": s4["settings"], "price_cpu_h": s4["price_cpu_h"],
               "cpu_s": cpu, "wall_s_fit": wall, "n": c.n, "fit": _fit_summary(r),
               "store_file": os.path.relpath(path, REPO),
               "store_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
               "provenance": row_provenance(calibrator=s4["calibrator"], version=s4["version"])})


def _scope_sources():
    """The basin scope's v1 records and table, each read only after its sha256 matches the panel's."""
    import csv
    import io
    import panel_003g as P
    pan = panel()
    rec = json.loads(P.read_source(pan["sources"], "basin_records", root=DROPBOX))
    scope = {r["cohort_id"]: r for r in csv.DictReader(io.StringIO(
        P.read_source(pan["sources"], "basin_scope", root=DROPBOX).decode()))}
    return rec, scope


def verify_points_rows(scope: dict, basin_j: list) -> list:
    """T031's pre-registered rows ([verify] points_rows): (cohort_id, role)."""
    import panel_003g as P
    jd = [r for r in scope.values() if r["arm"] not in ("window", "hazard")]
    small = lambda src, k: [r["cohort_id"] for r in sorted((r for r in jd if r["point_source"] == src),  # noqa: E731
                                                            key=lambda r: (int(float(r["n_obs"])), r["cohort_id"]))][:k]
    return [(c, "basin") for c in sorted(basin_j)] + [(c, "recon") for c in small("recon", 5)] + \
           [(c, "recon_failed") for c in small("recon_failed", 2)]


def phase_v033_points_verify(args):
    """T031 (research R6, VERIFY FIRST, no v6): sraging's recovery of v0.3.3's point reproduces the basin scope's on its
    own v1 records -- the path taken, the kept decision, and the value within [verify] value_rel_tol of the scope's."""
    import oracle_jax2_101 as OJ1
    import v033_points as VP
    pan = panel()
    run = _run("v033-points-verify", args, "0 fits; likelihood evaluations on 39 v1 cohorts (~2-5 min)")
    OJ1.load()
    from srbench.library.load import Library
    lib = Library(root=os.path.join(TESTER, "data", "library", "v1"), version="v1")
    rec, scope = _scope_sources()
    tol, ptol = float(pan["verify"]["value_rel_tol"]), float(pan["verify"]["point_rel_tol"])
    basin_j = [r["cohort_id"] for r in read_panel_rows() if "BASIN-J" in r["strata"]]
    rows = []
    for cid, role in verify_points_rows(scope, basin_j):
        r, s = rec[cid], scope[cid]
        o = lib.observation(cid)
        out = VP.recover(r["claims"], r["diagnostics"]["solver"], r["diagnostics"]["loglik"], o.ages, o.events)
        want_kept = s["point_ok"] == "True"
        row = {"cohort_id": cid, "role": role, "source": out["source"], "scope_source": s["point_source"],
               "kept": out["kept"], "scope_kept": want_kept, "value": out.get("value"),
               "scope_point_jax": float(s["point_jax"]) if s["point_jax"] not in ("", "nan") else None,
               "kappa": out.get("kappa"), "scope_kappa_hat": s["kappa_hat"], "recon_dll": out.get("recon_dll"),
               "scope_recon_dll": s["recon_dll_jax"], "n": len(o.ages), "oracle_host": s["oracle_host"]}
        ok = row["source"] == row["scope_source"] and row["kept"] == want_kept
        if out["kept"] and row["scope_point_jax"] is not None:
            rel = abs(out["value"] - row["scope_point_jax"]) / max(1.0, abs(row["scope_point_jax"]))
            row["value_rel_diff"], row["value_diff_nats"] = rel, out["value"] - row["scope_point_jax"]
            if out["source"] == "claims":           # the point is read, not computed: gate the point ([verify])
                want = [float(s[f"{p}_point"]) for p in VP.P]
                prel = max(abs(a - b) / abs(b) for a, b in zip(out["point"], want))
                row["point_rel_diff"] = prel
                ok = ok and prel <= ptol
            else:                                   # the kappa rebuild evaluates: gate the value
                ok = ok and rel <= tol
        row["ok"] = ok
        rows.append(row)
    claims = [r for r in rows if r["source"] == "claims" and r.get("value_diff_nats") is not None]
    run.write({"status": "pass" if all(r["ok"] for r in rows) else "fail", "rows": rows, "n": len(rows),
               "n_ok": sum(r["ok"] for r in rows), "value_rel_tol": tol, "point_rel_tol": ptol,
               "max_value_rel_diff_recon": max((r.get("value_rel_diff") or 0.0) for r in rows if r["source"] == "recon"),
               "v033_host_dependence_on_claims_rows": {
                   "n": len(claims), "n_above_1_nat": sum(abs(r["value_diff_nats"]) > 1 for r in claims),
                   "max_abs_nats": max((abs(r["value_diff_nats"]) for r in claims), default=None),
                   "note": "v0.3.3's value at its own point, Mac vs the scope's WEXAC: reported, not gated ([verify])"},
               "provenance": row_provenance(calibrator="profile_v033", version="0.3.3", library="v1")})


def phase_v033_mle_verify(args):
    """T032 (research R7, VERIFY FIRST): the MLE-stage replay equals a real v0.3.3 fit's theta-hat and log-likelihood
    bit for bit, side by side on this Mac, on the two smallest-n v6 nladder pure cohorts."""
    import v033_mle as M
    run = _run("v033-mle-verify", args, "2 full v0.3.3 fits at n = 150 + their replays (~10-20 CPU-min)")
    import oracle_jax2_101 as OJ1
    ids = sorted(c for c in OJ1.library("v6").manifest["checksums"] if c.startswith("nlad-") and "-n150-" in c)[:2]
    rows = []
    for cid in ids:
        c = v6_cohort(cid)
        r = M.mle_stage(c.ages, c.events, c.delta)
        f = M.full_fit_theta(c.ages, c.events, c.delta)
        rows.append({"cohort_id": cid, "n": c.n, "replay": r, "full": f,
                     "equal": r["status"] == "ok" and r["theta"] == f["theta"] and r["loglik"] == f["loglik"]})
    run.write({"status": "pass" if all(r["equal"] for r in rows) else "fail", "rows": rows,
               "provenance": row_provenance(calibrator="profile_v033", version="0.3.3")})


def phase_p15_v033(args):
    """T034 (Naveh Dd): the 3 BASIN-P15 ids have no v0.3.3 v6 record. Price from their v1 records' MLE-stage timers;
    replay v0.3.3's MLE stage on the v6 arrays under the panel's cap; W33 in 1.0.1's likelihood on 1.0.1's grid."""
    import v033_mle as M
    import v033_points as VP
    pan = panel()
    cap_s = float(pan["s4"]["dd_cap_cpu_h"]) * 3600.0
    ids = [r["cohort_id"] for r in read_panel_rows() if "BASIN-P15" in r["strata"]]
    rec, _scope = _scope_sources()
    price = {c: sum(float(rec[c]["diagnostics"][k]) for k in ("compile_s", "seed_s", "mle_s")) for c in ids}
    run = _run("p15-v033", args, f"3 MLE-stage replays; priced {sum(price.values()):.0f} s from their v1 records "
                                  f"(cap {cap_s:.0f} s)")
    if sum(price.values()) > cap_s:
        run.write({"status": "capped", "reason": "the price exceeds Naveh's cap; the gap is stated", "price_s": price})
        return
    import oracle_jax2_101 as OJ1
    lib = OJ1.library("v6")
    rows, spent = [], 0.0
    for cid in ids:
        if spent > cap_s:
            rows.append({"cohort_id": cid, "status": "not run: the cap was reached"})
            continue
        c = v6_cohort(cid)
        r = M.mle_stage(c.ages, c.events, c.delta)
        spent += r["cpu_s"]
        w = VP.witness_j(c, lib.truth(cid), r["theta"]) if r["status"] == "ok" else None
        rows.append({"cohort_id": cid, "n": c.n, "price_s": price[cid], "mle": r, "witness": w,
                     "shown": None if w is None else w["shown"],
                     "provenance": row_provenance(calibrator="profile_v033", version="0.3.3")})
    run.write({"status": "written", "rows": rows, "cpu_s_total": spent, "cap_s": cap_s,
               "shown": sum(bool(r.get("shown")) for r in rows), "n": len(ids)})


def phase_lev_pilot(args):
    """T033 (FR-011, Naveh Da; SC-005): 003b's pilot's leverage check re-run at the pilot's own theta-hats under the new
    label (the label depends only on the cohort and theta-hat; ~50-100 CPU-s per check [M, lev_measure_003b.json]).
    PASS iff ITP+age0 is `flag` and the 7 human hazard fits are `not_assessed`."""
    from sraging import Cohort
    from sraging import leverage as LEV
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.leverage_rule import G_CRIT_PER_IND, RULE
    from sraging.params import SRParams
    pilot = json.load(open(os.path.join(RESULTS_DIR, "lev_measure_003b.json")))
    run = _run("lev-pilot", args, f"{len(pilot['rows'])} leverage checks at stored theta-hats, ~8 CPU-min [I]")
    P1 = PJ.PORTS["1.0.1"]
    rows = []
    for pr in pilot["rows"]:
        cid = pr["cohort_id"]
        c = (Cohort.from_csv(os.path.join(REPO, "tests", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
             if cid == "ITP+age0" else v6_cohort(cid))
        th = pr["theta_hat"]
        theta = SRParams(**{k: th[k] for k in ("eta", "beta", "epsilon", "kappa")}, h_ext=th.get("h_ext", 0.0),
                         time_unit=th.get("time_unit", "days"))
        lev = LEV.check(c, "profile_jax2", theta, {}, library="v6", port=P1)
        want = "flag" if cid == "ITP+age0" else "not_assessed"
        rows.append({"cohort_id": cid, "n": c.n, "label_003b": pr["label"], "label": lev.get("label"), "want": want,
                     "ok": lev.get("label") == want, "reason": lev.get("reason"), "g0": lev.get("g0"),
                     "free_gradient_norm": (lev.get("at_theta_hat") or {}).get("free_gradient_norm"),
                     "free_gradient_per_individual": (lev.get("at_theta_hat") or {}).get("free_gradient_per_individual"),
                     "warm_termination": lev.get("warm_termination"), "cost": lev.get("cost"),
                     "provenance": row_provenance(calibrator="profile_jax2", version="1.0.1")})
    run.write({"status": "PASS" if all(r["ok"] for r in rows) else "FAIL", "rows": rows, "g_crit_per_ind": G_CRIT_PER_IND,
               "rule": dict(RULE), "labels": {r["cohort_id"]: r["label"] for r in rows}})


# ---------------------------------------------------------------------------------------------------- the windows
STORE = os.path.join(TESTER, "data", "store")
W_ROBUST = 10.0          # a human cohort's W4 above this is shown whatever W_REF is (W4 - W_REF ~ 1 nat, issue 17c)


def _pins_path(args):
    return os.path.join(REPO, args.pins) if getattr(args, "pins", None) else os.path.join(REPO, PINS_PATH)


def _pin_phase(n):
    def phase(args):
        """pin-wN (research R4): the window's runs, records and witness files by sha256, appended to the pins file
        (committed before any scoring of the window). Not a Run: the pins file is its record."""
        import window_003g as WIN
        if not args.head:
            sys.exit("pin-w%d: --head <the tester commit jax2 named for the window> is required" % n)
        if not args.outside_window:
            sys.exit(f"REFUSED: pass --outside-window once jax2 has closed window W{n} (its M7): the clone reads the "
                     f"tester checkout's .git")
        pan = panel()
        w = pan["windows"][f"W{n}"]
        path = _pins_path(args)
        data = json.load(open(path))
        WIN.ensure_clone(TESTER, tester_clone(args.head), args.head)
        runs = args.runs.split(",") if args.runs else w["runs"]
        exported = None
        if args.witness_sums:                    # the SHA256SUMS sha256 from jax2's WINDOW ENDS message
            exported = WIN.export_witness(witness_root(f"W{n}"), args.witness_sums,
                                          ("mle_profile_jax2", "mle_profile_jax2_h"))
        counts = WIN.pin_window(data, window=f"W{n}", runs=runs, witness_name=w.get("witness"), tester_root=TESTER,
                                mirror=witness_root(f"W{n}"), library=args.library, head=args.head,
                                clone_rel=os.path.relpath(tester_clone(args.head), REPO), exported=exported)
        data["tester"][f"W{n}"].update(witness_root=os.path.relpath(witness_root(f"W{n}"), DROPBOX),
                                       witness_sums_sha256=args.witness_sums, store_fingerprint=args.store_fingerprint,
                                       window_ends=args.window_ends)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(json.dumps(counts, indent=1))
        print(f"pinned W{n} into {os.path.relpath(path, REPO)}: commit it before scoring")
    return phase


def _context(n, args):
    """(pins data, adapter mods, Pins, the v6 Library, the window's head) for scoring window n."""
    from sraging.adapters import srbench as ADP
    path = _pins_path(args)
    data = json.load(open(path))
    t = data["tester"].get(f"W{n}")             # never another window's head (T067: no silent fallback to W1)
    if not t:
        raise Refused(f"REFUSED: window W{n} is not pinned (run pin-w{n} and commit the pins)")
    mods = ADP.connect_pinned(clone=tester_clone(t["head"]), head=t["head"])
    pins = ADP.Pins(data, TESTER, path)
    lib = mods["library.load"].Library(root=ADP.library_root(args.library, TESTER))
    return data, mods, pins, lib, t["head"]


def _sets(mods, pins, run, ids):
    """A pinned run's claim sets for `ids` (those it keys), each record read once and checked against its pin."""
    from sraging.adapters import srbench as ADP
    man = json.loads(pins.read_file(os.path.join("runs", f"{run}.json")))
    keep = [c for c in ids if c in man["keys"]]
    return ADP.load_pinned_sets(mods, man, STORE, pins=pins, run_id=run, cohort_ids=keep), man


def _record_env(mods, pins, run, man, cid) -> dict:
    """N10: where a pinned record was computed (its top-level code_provenance.environment); {} when not stored."""
    if cid not in man["keys"]:
        return {}
    rec = json.loads(pins.read_record(run, cid, man["keys"][cid], mods["store.store"].ResultStore(STORE)))
    return ((rec.get("code_provenance") or {}).get("environment")) or {}


def _gating_strata(rows, half):
    names = sorted({s for r in rows for s in r["strata"] if r["calibrator"] == "profile_jax2"})
    return [s for s in names if any(s in r["strata"] and r["half"] == half for r in rows)]


def _score(n, args, half):
    """score-w1 (inside) / score-w3 (outside): per stratum x arm, PR-1 from the witness files, PR-4 against the null,
    PR-3 (outside) against J's coverage on the same arm inside; the Db mechanism counts."""
    import window_003g as WIN
    pan = panel()
    data, mods, pins, lib, head = _context(n, args)
    V, CL = mods["score.verdict"], mods["score.claims"]
    runs = args.runs.split(",") if args.runs else pan["windows"][f"W{n}"]["runs"]       # --runs: dry runs only
    run_j = [r for r in runs if not r.startswith("null_prior")][0]
    null_run = ([r for r in runs if r.startswith("null_prior")] or ["null_prior-v6-full-bign"])[0]
    wname = pan["windows"][f"W{n}"]["witness"]
    rows = [r for r in read_panel_rows() if r["calibrator"] == "profile_jax2" and r["half"] == half]
    verdicts, wi, prov, mech, limits = [], [], {}, {}, {}
    for stratum in _gating_strata(read_panel_rows(), half):
        srows = [r for r in rows if stratum in r["strata"]]
        if not srows:
            verdicts.append(G_verdict(stratum, half, "PR-1", "empty"))
            continue
        for arm in sorted({r["arm"] for r in srows}):
            arows = [r for r in srows if r["arm"] == arm]
            ids = [r["cohort_id"] for r in arows]
            valid = [r["cohort_id"] for r in arows if r["valid"]]
            j_sets, man = _sets(mods, pins, run_j, ids)
            label = f"{stratum}/{half}/{arm}"
            wr = WIN.witness_rows(arows, data, wname, man["keys"], witness_root(f"W{n}"))
            verdicts.append(DG().witness_verdict(f"{stratum}/{half}", wr, arm=arm))
            wi.append(DG().witness_verdict(f"{stratum}/{half}", WIN.with_invalid(wr), arm=arm))     # FR-005, T068
            jv = [cs for cs in j_sets if cs.cohort_id in set(valid)]

            def nulls(keep):
                if half == "inside":
                    return _sets(mods, pins, null_run, keep)[0] if null_run in data["runs"] else []
                return _local_null_sets(mods, data, keep)

            def pr4(js, nul, lab):
                v = (DG().null_verdict(f"{stratum}/{half}", arm, WIN.info_widths(V, js, nul, lab)) if js and nul else
                     DG().Verdict(f"{stratum}/{half}", "PR-4", "reported" if not js else "incomplete", arm=arm,
                                  reasons=["no valid cohort" if not js else "no null"]))
                v.statistic["shapes_j"] = WIN.shape_shares(js)                 # T070: what PR-4's widths cannot see
                return v
            verdicts.append(pr4(jv, nulls(valid), label))
            wi.append(pr4(j_sets, nulls(ids), f"{label}/with_invalid"))
            if half == "outside":
                w1_run = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
                for out_sets, inside, into in ((jv, False, verdicts), (j_sets, True, wi)):
                    truths = lib.truths([cs.cohort_id for cs in out_sets])
                    inside_ids = [r["cohort_id"] for r in read_all_inside_valid(lib, arm, include_invalid=inside)]
                    j_in, _ = _sets(mods, pins, w1_run, inside_ids)
                    t_in = lib.truths([cs.cohort_id for cs in j_in])
                    for q in CL.scored_quantities(lib.truths(valid)):
                        k_out, n_out = WIN.coverage_counts(CL, out_sets, truths, q)
                        k_in, n_in = WIN.coverage_counts(CL, j_in, t_in, q)
                        into.append(DG().coverage_outside(
                            f"{stratum}/{half}", arm, q, k_out=k_out, n_out=n_out, k_in=k_in, n_in=n_in,
                            n_valid_out=sum((truths.get(cs.cohort_id) or {}).get(q) is not None for cs in out_sets)))
            mech[label] = WIN.mechanisms(j_sets)
            t_valid = lib.truths(valid)
            lim = {"closed_widths": {q: WIN.closed_widths(jv, q) for q in ("eta", "epsilon")},        # N4, reported
                   "eta_beta": WIN.eta_beta_crosstab(jv)}
            if half == "outside":
                lim["one_sided"] = WIN.one_sided_coverage(CL, jv, t_valid)                           # N8, reported
            if arm == "realfull":                                                                    # N6, reported
                n_of = {cid: lib.spec(cid).n for cid in valid}
                lim["realfull"] = {
                    "rows": [{"cohort_id": r["cohort_id"], "n": n_of[r["cohort_id"]], "W": r["witness"]["W"],
                              "W_per_individual": (None if r["witness"]["W"] is None else
                                                   r["witness"]["W"] / n_of[r["cohort_id"]])}
                             for r in wr if r["cohort_id"] in n_of],
                    "closed": {q: dict(zip(("hits", "claimed"), WIN.coverage_counts(CL, jv, t_valid, q)))
                               for q in ("eta", "epsilon")}}
            limits[label] = lim
            for cid in ids:
                rec = (data["records"].get(run_j) or {}).get(cid) or {}
                env = _record_env(mods, pins, run_j, man, cid)
                prov[cid] = row_provenance(calibrator="profile_jax2", version="1.0.1", run_manifest=run_j,
                                           run_manifest_sha256=data["runs"][run_j]["sha256"],
                                           record_sha256=rec.get("sha256"),
                                           witness_sha256=(data["witness"].get(wname) or {}).get(cid),
                                           record_host=env.get("host"), record_cpu=env.get("cpu"))
    return {"verdicts": [v.as_dict() for v in verdicts], "mechanisms": mech, "provenance_rows": prov,
            "tester_head": head, "pr1_verdicts_scored": sum(1 for v in verdicts if v.rule == "PR-1"),
            "limits": limits,
            "with_invalid": [v.as_dict() for v in wi]}           # FR-005 (T068): reported, never read by the gate


def DG():
    from sraging import default_gate
    return default_gate


def G_verdict(stratum, half, rule, verdict):
    return DG().Verdict(f"{stratum}/{half}", rule, verdict)


def read_all_inside_valid(lib, arm, include_invalid=False):
    """PR-3's comparator: every valid cohort of `arm` in the reference set (main + bign), J's domain; with
    `include_invalid`, every cohort (FR-005's reported numbers, T068)."""
    import csv
    import io
    import panel_003g as P
    pan = panel()
    ref = set()
    for line in P.read_source(pan["sources"], "reference_set", root=DROPBOX).decode().splitlines():
        if line and not line.startswith("#"):
            ref.add(line.split("\t")[1])
    val = {r["cohort_id"]: r for r in csv.DictReader(io.StringIO(P.read_source(pan["sources"], "validity_v6",
                                                                                root=DROPBOX).decode()))}
    return [{"cohort_id": c} for c in sorted(ref) if lib.spec(c).arm == arm
            and (include_invalid or str(val[c]["flag_invalid"]).lower() not in ("true", "1"))]


def _local_null_sets(mods, data, ids):
    """The null outside the reference set, computed locally (null-local) and pinned in pins["null_local"]."""
    nl = data.get("null_local") or {}
    if not nl.get("manifest"):
        return []
    from sraging.adapters import srbench as ADP
    man_path = os.path.join(REPO, nl["manifest"])
    import window_003g as WIN
    if WIN.file_sha(man_path) != nl["manifest_sha256"]:
        raise Refused("REFUSED: the local null's manifest differs from its pin")
    man = json.load(open(man_path))
    keep = [c for c in ids if c in man["keys"]]
    for c in keep:
        if WIN.file_sha(WIN.store_path(os.path.join(REPO, nl["store"]), man["keys"][c])) != nl["records"][c]:
            raise Refused(f"REFUSED: the local null's record for {c} differs from its pin")
    return ADP.load_pinned_sets(mods, man, os.path.join(REPO, nl["store"]), cohort_ids=keep)


def phase_pr3_power(args):
    """T069 (the interim review, B3): PR-3's power at the panel's N, from W1's inside closed counts only (no W3 record
    is read), committed before pin-w3. Per outside stratum x arm x quantity: the smallest passing k_out for every
    closed-claim count up to the stratum's valid N, and whether equal coverage passes at the count the inside closed
    share predicts. The gate prints the row beside every PR-3 FAIL."""
    import window_003g as WIN
    run = _run("pr3-power", args, "0 fits; reads W1's pinned inside records")
    pan = panel()
    data, mods, pins, lib, head = _context(1, args)
    CL = mods["score.claims"]
    w1_run = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
    rows = [r for r in read_panel_rows() if r["calibrator"] == "profile_jax2" and r["half"] == "outside"]
    inside, table = {}, []
    for stratum in _gating_strata(read_panel_rows(), "outside"):
        srows = [r for r in rows if stratum in r["strata"]]
        for arm in sorted({r["arm"] for r in srows}):
            valid = [r["cohort_id"] for r in srows if r["arm"] == arm and r["valid"]]
            if arm not in inside:
                j_in, _ = _sets(mods, pins, w1_run, [r["cohort_id"] for r in read_all_inside_valid(lib, arm)])
                inside[arm] = (j_in, lib.truths([cs.cohort_id for cs in j_in]))
            j_in, t_in = inside[arm]
            t_out = lib.truths(valid)
            for q in CL.scored_quantities(t_out):
                c = CL.coverage(j_in, t_in, q, "closed").to_dict()
                n_valid_out = sum((t_out.get(cid) or {}).get(q) is not None for cid in valid)
                share = c["claimed"] / c["total"] if c["total"] else 0.0
                table.append({"stratum": f"{stratum}/outside", "arm": arm, "quantity": q, "closed_share_in": share,
                              **WIN.pr3_power_row(k_in=int(c["hits"]), n_in=int(c["claimed"]),
                                                  n_valid_out=n_valid_out, n_expected=round(n_valid_out * share))})
    run.write({"status": "written", "rows": table, "tester_head": head,
               "cannot_pass_at_expected": sorted({f"{r['stratum']} {r['arm']} {r['quantity']}" for r in table
                                                  if not r["equal_coverage_passes_at_expected"]})})


def phase_score_w1(args):
    run = _run("score-w1", args, "0 fits; reads W1's pinned records and witness files")
    out = _score(1, args, "inside")
    run.write({"status": "written", "window": "W1", **out, "human_box_edge": _human_box_edge(args),
               "realfull_all": _realfull_all(args, out["limits"])})


def _realfull_all(args, limits: dict) -> dict:
    """N6 (Naveh 2026-10-29), reported: the union of the inside strata's valid realfull cohorts (a cohort in several
    strata counted once): J's closed eta/epsilon coverage and widths, and W and W/n from the per-stratum rows."""
    import statistics
    import window_003g as WIN
    pan = panel()
    data, mods, pins, lib, head = _context(1, args)
    CL = mods["score.claims"]
    run_j = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
    ids = sorted({r["cohort_id"] for r in read_panel_rows() if r["calibrator"] == "profile_jax2"
                  and r["half"] == "inside" and r["arm"] == "realfull" and r["valid"]})
    sets, _ = _sets(mods, pins, run_j, ids)
    t = lib.truths(ids)
    rows = {r["cohort_id"]: r for lim in limits.values() for r in (lim.get("realfull") or {}).get("rows", [])}
    w = [r["W"] for r in rows.values() if r["W"] is not None]
    wn = [r["W_per_individual"] for r in rows.values() if r["W"] is not None]
    return {"n": len(ids), "closed": {q: dict(zip(("hits", "claimed"), WIN.coverage_counts(CL, sets, t, q)))
                                      for q in ("eta", "epsilon")},
            "closed_widths": {q: WIN.closed_widths(sets, q) for q in ("eta", "epsilon")},
            "W": {"n": len(w), "below_minus_10": sum(x < -10 for x in w), "min": min(w, default=None),
                  "max": max(w, default=None),
                  "per_individual": {"median": statistics.median(wn) if wn else None, "min": min(wn, default=None),
                                     "max": max(wn, default=None)}}}


def _human_box_edge(args) -> dict:
    """N5 (Naveh 2026-10-29), reported: on every human record of W1's J run, how often sraging's BOX_EDGE check (theta-hat
    within 0.1 % of the prior box's upper edge, warnings._box_edge, the box 1.0.1's calibrator passes) fires, by
    parameter. Reads diagnostics.estimate.theta_hat; 0 fits."""
    import witness_003g as WIT
    from sraging import warnings as SW
    from sraging.calibrators import profile_jax2 as PJ
    pan = panel()
    data, mods, pins, lib, head = _context(1, args)
    run_j = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
    man = json.loads(pins.read_file(os.path.join("runs", f"{run_j}.json")))
    box = PJ.settings("profile_jax2", PJ.PORTS["1.0.1"])["box"]
    sets, _ = _sets(mods, pins, run_j, sorted(c for c in man["keys"] if WIT.is_human(c)))
    n, k, by, no_theta = 0, 0, {}, 0
    for cs in sets:
        th = ((cs.diagnostics or {}).get("estimate") or {}).get("theta_hat") if cs.ok else None
        if not th:
            no_theta += 1
            continue
        n += 1
        w = SW._box_edge(th, box)
        if w:
            k += 1
            for p in w[0].data["parameters"]:
                by[p] = by.get(p, 0) + 1
    import window_003g as WIN
    return {"run": run_j, "n": n, "box_edge": k, "by_parameter": dict(sorted(by.items())), "no_theta_hat": no_theta,
            "box": {p: list(v) for p, v in box.items()},
            "closed_widths": {q: WIN.closed_widths(sets, q) for q in ("eta", "epsilon")},             # N4, reported
            "shapes": WIN.shape_shares(sets)}


def phase_score_w3(args):
    """The outside strata (PR-1, PR-3, PR-4), and fit 1 against its W3 record (SC-007)."""
    import window_003g as WIN
    run = _run("score-w3", args, "0 fits; reads W3's pinned records, witness files and the local null")
    out = _score(3, args, "outside")
    data, mods, pins, lib, head = _context(3, args)
    pan = panel()
    meta = {r["cohort_id"]: r for r in read_panel_rows() if r["calibrator"] == "profile_jax2"}
    p15 = json.load(open(os.path.join(RESULTS_DIR, "p15_v033_003g.json")))        # T067 (B2): BASIN-P15's PR-5
    out["verdicts"].append(DG().predecessor_verdict("BASIN-P15", WIN.p15_rows(p15, meta),
                                                    measure=pan["rules"]["pr5_measures"]["BASIN-P15"]).as_dict())
    out["scorer"] = WIN.scorer_diff(tester_clone(head), data["tester"]["W1"]["head"], head)   # T067: one scorer
    f1p = os.path.join(RESULTS_DIR, "fit1_003g.json")
    cmp = None
    if os.path.exists(f1p):
        f1 = json.load(open(f1p))
        run_j = [r for r in pan["windows"]["W3"]["runs"] if not r.startswith("null_prior")][0]
        man = json.loads(pins.read_file(os.path.join("runs", f"{run_j}.json")))
        cid = f1["cohort_id"]
        if cid in man["keys"]:
            rec = json.loads(pins.read_record(run_j, cid, man["keys"][cid], mods["store.store"].ResultStore(STORE)))
            cmp = WIN.claim_compare(list(f1["fit"]["claims"].values()), rec["claim_set"]["claims"], lib.truth(cid))
            cmp.update(cohort_id=cid, record_cpu_s=rec["claim_set"].get("cpu_seconds"), mac_cpu_s=f1.get("cpu_s"))
    run.write({"status": "written", "window": "W3", **out, "fit1_vs_record": cmp})


def phase_sut(args):
    """FR-010 (research R11): the SRK files jax2's V6-J fits loaded equal sraging's v0.3.3 oracle's, on one het record
    and the one record inside DT_RULE's scope (SRK-1b). No fit."""
    import window_003g as WIN
    run = _run("sut", args, "0 fits; reads 2 pinned records")
    pan = panel()
    data, mods, pins, lib, head = _context(1, args)
    run_j = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
    man = json.loads(pins.read_file(os.path.join("runs", f"{run_j}.json")))
    het = sorted(c for c in man["keys"] if c.startswith("het-"))[0]
    inscope = [r["cohort_id"] for r in read_panel_rows() if "SRK-1b" in r["strata"]][0]
    oracle = os.path.join(REPO, "tests", "oracle", "v033", "SRtools_kappa", "SRK")
    rows = []
    for cid in (het, inscope):
        rec = json.loads(pins.read_record(run_j, cid, man["keys"][cid], mods["store.store"].ResultStore(STORE)))
        files = ((rec.get("claim_set") or {}).get("diagnostics") or {}).get("srk_sha", {}).get("files") or {}
        rows.append({"cohort_id": cid, "files": files, **WIN.srk_diff(files, oracle)})
    ok = all(r["n_files"] == 18 and not r["different"] and not r["missing_in_oracle"] for r in rows)
    run.write({"status": "pass" if ok else "fail", "rows": rows, "tester_head": head})


def phase_gate(args):
    """PR-6: every committed verdict, the blocking list for Naveh (tasks T049-T050)."""
    import window_003g as WIN
    run = _run("gate", args, "0 fits; reads the committed verdicts")
    pan = panel()
    raw, loaded = [], {}
    for name in ("score_w1", "score_w3", "predecessor", "pr3_power"):
        p = os.path.join(RESULTS_DIR, f"{name}_003g.json")
        if os.path.exists(p):
            loaded[name] = json.load(open(p))
            raw += loaded[name].get("verdicts", [])
    raw, deviations = WIN.apply_deviations(raw, pan.get("deviations", []), strata=set(pan["strata"]))   # T071
    fields = DG().Verdict.__dataclass_fields__
    verdicts = [DG().Verdict(**{k: v[k] for k in fields if k in v}) for v in raw]
    rows = read_panel_rows()
    required = []
    for half, rules in (("inside", ("PR-1", "PR-2", "PR-4")), ("outside", ("PR-1", "PR-3", "PR-4"))):
        for s in _gating_strata(rows, half):
            required += [(f"{s}/{half}", r) for r in rules]
    for s in ("BASIN-J", "BASIN-P15", "P-ROUNDOFF", "P-DTCEIL", "P-CENSORTIE"):
        required.append((s, "PR-5"))
    summary = DG().gate_summary(verdicts, required=required)
    if "score_w3" in loaded:                         # T067: inside and outside verdicts from one tester scorer
        summary["blocking"] += WIN.scorer_blocking(loaded["score_w3"].get("scorer"))
        summary["status"] = "blocked" if summary["blocking"] else summary["status"]
    summary["blocking"], decided = WIN.apply_decisions(summary["blocking"], pan.get("decisions", []))    # N9
    if summary["blocking"] and decided["decided"] == decided["of"]:
        summary["status"] = f"blocked; decided ({decided['decided']} of {decided['of']})"
    power = {(r["stratum"], r["arm"], r["quantity"]): r for r in (loaded.get("pr3_power") or {}).get("rows", [])}
    pr3_notes = [dict(stratum=v.stratum, arm=v.arm, quantity=v.quantity, **(
        WIN.pr3_note(power[(v.stratum, v.arm, v.quantity)], n_out=v.statistic["n_out"], k_out=v.statistic["k_out"])
        if (v.stratum, v.arm, v.quantity) in power else {"reading": "no power-table row"}))
        for v in verdicts if v.rule == "PR-3" and v.verdict == "FAIL"]                       # T069
    pr4_declines = [{"stratum": v.stratum, "arm": v.arm,
                     **{f"{q}_{k}": f"{(v.statistic.get('shapes_j') or {}).get(q, {}).get(k)} of "
                                    f"{(v.statistic.get('shapes_j') or {}).get(q, {}).get('n')}"
                        for q, k in (("eta", "open_above"), ("epsilon", "open_above"),
                                     ("beta", "upper_end_at_kappa_box"))}}
                    for v in verdicts if v.rule == "PR-4" and v.verdict == "PASS"]            # T070
    run.write({"status": summary["status"], **summary, "required": [list(x) for x in required],
               "deviations": deviations, "decisions": decided, "pr3_notes": pr3_notes, "pr4_declines": pr4_declines})


def phase_null_local(args):
    """FR-006 / research R12: the data-ignoring null on sharp + nladder (jax2's sharp null is never merged), run from
    the pinned tester clone into sraging's own store (.cache/null_003g), nothing under the tester checkout; pinned into
    the pins file's null_local (then committed). Not a Run: the pins file is its record. Seconds."""
    import window_003g as WIN
    path = _pins_path(args)
    data = json.load(open(path))
    t = data["tester"].get("W1")
    if not t:
        raise Refused("REFUSED: pin W1 first (the null runs from the clone of the tester commit jax2 named)")
    clone = tester_clone(t["head"])
    base = os.path.join(REPO, ".cache", "null_003g")
    store, out = os.path.join(base, "store"), os.path.join(base, "null_prior-v6-sharp-nladder.json")
    os.makedirs(store, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=clone, PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-B", "-m", "srbench.cli", "--store", store, "--library-root",
           os.path.join(TESTER, "data", "library", "v6"), "run", "--method", "null_prior", "--group", "sharp",
           "--group", "nladder", "--run-id", "null_prior-v6-sharp-nladder-sraging", "--out", out, "--no-explorer"]
    r = subprocess.run(cmd, cwd=base, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"null-local failed: {r.stderr[-2000:]}")
    man = json.load(open(out))
    recs = {c: WIN.file_sha(WIN.store_path(store, k)) for c, k in man["keys"].items()}
    data["null_local"] = {"manifest": os.path.relpath(out, REPO), "manifest_sha256": WIN.file_sha(out),
                          "store": os.path.relpath(store, REPO), "records": recs, "n": len(recs),
                          "tester_head": t["head"], "command": " ".join(cmd[1:])}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"null-local: {len(recs)} records pinned into {os.path.relpath(path, REPO)}: commit it")


def phase_fit2_pick(args):
    """FR-009, fit 2's choice (research R10): a human cohort whose W1 record says the basin re-fit fired, priced from
    that record's timer (Mac per-core taken as WEXAC's [I]) before it runs. The registered rule took BASIN-J ids; in
    W1 the re-fit fired on none of them, so (Naveh, 2026-10-27) the cheapest VALID human panel cohort where it fired."""
    pan = panel()
    run = _run("fit2-pick", args, "0 fits; reads W1's pinned human records")
    data, mods, pins, lib, head = _context(1, args)
    run_j = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
    rows = {r["cohort_id"]: r for r in read_panel_rows()
            if r["calibrator"] == "profile_jax2" and ("BASIN-J" in r["strata"] or "LB-2" in r["strata"])}
    sets, man = _sets(mods, pins, run_j, list(rows))

    def fired(cs):
        return ((cs.diagnostics or {}).get("basin_fix") or {}).get("fired")
    basin_j = [cs.cohort_id for cs in sets if fired(cs) and "BASIN-J" in rows[cs.cohort_id]["strata"]]
    cands = [{"cohort_id": cs.cohort_id, "cpu_s": cs.cpu_seconds, "arm": rows[cs.cohort_id]["arm"],
              "valid": rows[cs.cohort_id]["valid"], "basin_j": "BASIN-J" in rows[cs.cohort_id]["strata"],
              "n_refit": ((cs.diagnostics or {}).get("basin_fix") or {}).get("n_refit")} for cs in sets if fired(cs)]
    cands.sort(key=lambda x: (not x["basin_j"], not x["valid"], x["cpu_s"] or 1e18, x["cohort_id"]))
    pick = cands[0] if cands else None
    run.write({"status": "picked" if pick else "none", "candidates": cands, "pick": pick,
               "basin_j_fired": basin_j,
               "deviation": None if basin_j else "no BASIN-J record fired the basin re-fit in W1; the cheapest valid "
                                                 "human cohort where it fired (Naveh, 2026-10-27)",
               "price_cpu_h": None if pick is None else pick["cpu_s"] / 3600.0,
               "price_basis": "the W1 record's cpu_seconds (WEXAC), Mac per-core taken as equal [I]",
               "tester_head": head})


def phase_fit2(args):
    """FR-009 fit 2: fit2-pick's cohort with the default settings into sraging's own store; its record compared."""
    import time
    import sraging
    p = json.load(open(os.path.join(RESULTS_DIR, "fit2_pick_003g.json")))
    if not p.get("pick"):
        raise Refused("REFUSED: fit2-pick found no basin re-fit among the BASIN-J records: nothing to fit")
    cid = p["pick"]["cohort_id"]
    run = _run("fit2", args, f"1 Mac fit of {cid}, ~{p['price_cpu_h']:.1f} CPU-h [I]; keep the lid open")
    c, _o, window = j_view(cid)          # J's view: a realfull window set aside, as the W1 record did
    c0 = time.process_time()
    r = sraging.fit_cohort(c, calibrator="profile_jax2", version="1.0.1")
    cpu = time.process_time() - c0
    store = os.path.join(REPO, ".cache", "store_003g")
    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, f"fit2_{cid}.json")
    with open(path, "w") as fh:
        fh.write(r.to_json())
    run.write({"status": "written", "cohort_id": cid, "cpu_s": cpu, "price_cpu_h": p["price_cpu_h"],
               "window_set_aside": window,
               "fit": _fit_summary(r), "store_file": os.path.relpath(path, REPO),
               "store_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
               "provenance": row_provenance(calibrator="profile_jax2", version="1.0.1")})


def phase_fit2_compare(args):
    """SC-007, fit 2 (reported): its claims against its W1 record (the parameters both claim), the record's basin
    re-fit against the Mac fit's, and the two log-likelihoods. Read only."""
    import window_003g as WIN
    run = _run("fit2-compare", args, "0 fits; reads fit 2's result and its pinned W1 record")
    f2 = json.load(open(os.path.join(RESULTS_DIR, "fit2_003g.json")))
    pan = panel()
    data, mods, pins, lib, head = _context(1, args)
    run_j = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
    cid = f2["cohort_id"]
    man = json.loads(pins.read_file(os.path.join("runs", f"{run_j}.json")))
    rec = json.loads(pins.read_record(run_j, cid, man["keys"][cid], mods["store.store"].ResultStore(STORE)))["claim_set"]
    cmp = WIN.claim_compare(list(f2["fit"]["claims"].values()), rec["claims"], lib.truth(cid))
    dg = rec.get("diagnostics") or {}
    run.write({"status": "reported", "cohort_id": cid, "compare": cmp,
               "basin_refit": {"record": (dg.get("basin_fix") or {}).get("fired"),
                               "record_n_refit": (dg.get("basin_fix") or {}).get("n_refit"),
                               "mac": f2["fit"]["basin_fix_fired"]},
               "loglik": {"record": dg.get("loglik"), "mac": f2["fit"]["loglik"],
                          "mac_minus_record": (f2["fit"]["loglik"] or 0) - (dg.get("loglik") or 0)},
               "cpu_s": {"record_wexac": rec.get("cpu_seconds"), "mac": f2["cpu_s"]},
               "record_sha256": data["records"][run_j][cid]["sha256"], "tester_head": head})


def phase_score_w2(args):
    """S7 / FR-008 (reported, not gating): `_h` 1.0.1 on BASIN-H and HZ-4 from W2's witness files and records; HZ-1
    (sharp hazard) has no `_h` record in any window. The numbers HAZARD_SHARP_LIMIT will quote (T046)."""
    import window_003g as WIN
    pan = panel()
    run = _run("score-w2", args, "0 fits; reads W2's pinned records and witness files")
    data, mods, pins, lib, head = _context(2, args)
    runs = args.runs.split(",") if args.runs else pan["windows"]["W2"]["runs"]
    run_h = [r for r in runs if not r.startswith("null_prior")][0]
    wname = pan["windows"]["W2"]["witness"]
    rows = [r for r in read_panel_rows() if r["calibrator"] == "profile_jax2_h"]
    out = {}
    for stratum in ("BASIN-H", "HZ-4", "HZ-1"):
        srows = [r for r in rows if stratum in r["strata"] and r["window"] == "W2"]
        if not srows:
            out[stratum] = {"status": "no _h record in any window" if stratum == "HZ-1" else "empty"}
            continue
        _sets_h, man = _sets(mods, pins, run_h, [r["cohort_id"] for r in srows])
        wr = WIN.witness_rows(srows, data, wname, man["keys"], witness_root("W2"))
        ws = [x["witness"]["W"] for x in wr if x["witness"]["W"] is not None]
        out[stratum] = {"n": len(srows), "n_witnessed": len(ws), "w_max": max(ws, default=None),
                        "n_over_1_92": sum(w > 1.92 for w in ws),
                        "verdict": DG().witness_verdict(stratum, wr).as_dict(),
                        "rows": [{"cohort_id": x["cohort_id"], "W": x["witness"]["W"],
                                  "reason": x["witness_detail"].get("reason")} for x in wr]}
    run.write({"status": "reported", "strata": out, "run": run_h, "tester_head": head})


def phase_predecessor(args):
    """S3 / PR-5 and PR-2 (after the seal): v0.3.3's v6 records, read only once [seal] names jax2's gate run.
    --pin-only pins the sealed run's manifest and records (then commit); the scoring run recovers v0.3.3's points
    (tools/v033_points.recover), takes W33 in 1.0.1's likelihood (witness_j; human cohorts: W4, robust when > 10 nats),
    the dt-ceiling stratum's timers, and PR-2 per stratum x arm through the tester's own comparison."""
    import v033_points as VP
    import window_003g as WIN
    from sraging.adapters import srbench as ADP
    pan = panel()
    evidence = require_unsealed(pan)
    sealed = pan["windows"]["sealed_runs"][0]
    if args.pin_only:
        path = _pins_path(args)
        data = json.load(open(path))
        head = data["tester"]["W1"]["head"]
        counts = WIN.pin_window(data, window="SEAL", runs=[sealed], witness_name=None, tester_root=TESTER,
                                mirror=WITNESS_MIRROR, library=args.library, head=head,
                                clone_rel=os.path.relpath(tester_clone(head), REPO))
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(json.dumps(counts, indent=1))
        return
    run = _run("predecessor", args, "0 fits; v0.3.3 point recovery + W33 on BASIN-J 32 + P-ROUNDOFF 324 "
                                    "(price from the 3-cohort pilot printed first); PR-2 comparisons")
    data, mods, pins, lib, head = _context(1, args)
    run_j = [r for r in pan["windows"]["W1"]["runs"] if not r.startswith("null_prior")][0]
    store = mods["store.store"].ResultStore(STORE)
    man33 = json.loads(pins.read_file(os.path.join("runs", f"{sealed}.json")))
    manj = json.loads(pins.read_file(os.path.join("runs", f"{run_j}.json")))
    rows = read_panel_rows()
    verdicts, pts = [], {}
    only = set(args.only.split(",")) if args.only else None
    for stratum in ("BASIN-J", "P-ROUNDOFF", "P-DTCEIL", "P-CENSORTIE"):
        ids = [r["cohort_id"] for r in rows if stratum in r["strata"] and r["half"] == "inside"
               and (only is None or r["cohort_id"] in only)]
        prow = []
        for cid in ids:
            if cid not in man33["keys"]:
                prow.append({"cohort_id": cid, "shown": None, "reason": "no v0.3.3 v6 record"})
                continue
            rec = json.loads(pins.read_record(sealed, cid, man33["keys"][cid], store))["claim_set"]
            if stratum == "P-DTCEIL":
                recj = json.loads(pins.read_record(run_j, cid, manj["keys"][cid], store))["claim_set"]
                prow.append({"cohort_id": cid, "cpu_s_v033": rec.get("cpu_seconds"),
                             "cpu_s_j": recj.get("cpu_seconds"),
                             "shown": (rec.get("cpu_seconds") or 0) > (recj.get("cpu_seconds") or 0)})
                continue
            if cid not in pts:
                c, o, window = j_view(cid, lib)              # J's view: a realfull window set aside, as J does
                dg = rec.get("diagnostics") or {}
                got = VP.recover(rec["claims"], dg["solver"], dg["loglik"], o.ages, o.events)
                w = VP.witness_j(o, lib.truth(cid), got["point"]) if got["kept"] else None
                human = __import__("witness_003g").is_human(cid)
                pts[cid] = {"cohort_id": cid, "source": got["source"], "recon_dll": got.get("recon_dll"),
                            "point": got.get("point"), "W33": None if w is None else w["W4"],
                            "human": human, "robust": None if w is None else (not human or w["W4"] > W_ROBUST),
                            "shown": None if w is None else w["shown"],
                            "record_sha256": data["records"][sealed][cid]["sha256"], "window_set_aside": window}
            prow.append(pts[cid])
        meta = {r["cohort_id"]: r for r in rows}
        if stratum != "P-DTCEIL":           # W33 is defined on valid informative cohorts only, as PR-1's W
            scored = [x for x in prow if meta[x["cohort_id"]]["valid"] and meta[x["cohort_id"]]["informative"]]
        else:
            scored = prow
        verdicts.append(DG().predecessor_verdict(stratum, scored, measure=pan["rules"]["pr5_measures"][stratum]))
        st = verdicts[-1].statistic
        st["rows"], st["excluded_not_valid_or_informative"] = scored, sorted(
            x["cohort_id"] for x in prow if x not in scored)
        st["shown_also_basin_j"] = sorted(x["cohort_id"] for x in scored if x.get("shown") is True
                                          and "BASIN-J" in meta[x["cohort_id"]]["strata"])
    def pr2(stratum, arm, ids):
        if not ids:
            return DG().coverage_inside(f"{stratum}/inside", arm, {})
        cmp = ADP.compare_v5(mods, {"run_id": run_j, "store_root": STORE}, {"run_id": sealed, "store_root": STORE},
                             None, library=args.library, pins=pins, tester_root=TESTER, cohort_ids=ids)
        axes = {f"{q}/{ax}": {"outcome": e["axes"][ax]["outcome"], "n": e["axes"][ax]["denominator"],
                              "statistic": e["axes"][ax]["statistic"]}
                for q, e in cmp["by_parameter"].items() for ax in ("calibration", "calibration_one_sided")}
        return DG().coverage_inside(f"{stratum}/inside", arm, axes)
    meta, wi = {r["cohort_id"]: r for r in rows}, []
    for stratum in _gating_strata(rows, "inside"):
        srows = [r for r in rows if stratum in r["strata"] and r["half"] == "inside"]
        for arm in sorted({r["arm"] for r in srows if r["valid"]}):
            ids = [r["cohort_id"] for r in srows if r["arm"] == arm and r["cohort_id"] in man33["keys"]]
            verdicts.append(pr2(stratum, arm, [c for c in ids if meta[c]["valid"]]))
            wi.append(pr2(stratum, arm, ids))                       # FR-005 (T068): INVALID included, reported
    run.write({"status": "written", "seal": evidence, "verdicts": [v.as_dict() for v in verdicts],
               "with_invalid": [v.as_dict() for v in wi], "tester_head": head, "n_points": len(pts)})


def phase_readme(args):
    """T053: tools/verify_readme.py recomputes every marked number of README.md from the stored results."""
    run = _run("readme", args, "seconds: renders the stored example results; 0 fits")
    p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "verify_readme.py")], capture_output=True, text=True,
                       cwd=REPO)
    run.write({"status": "pass" if p.returncode == 0 else "fail", "returncode": p.returncode,
               "output": (p.stdout + p.stderr)[-3000:]})


#: the results tests of every earlier spec (T057, SC-010); 003g's own is T058's
UNCHANGED_RESULTS = ["tests/acceptance/test_results.py", "tests/acceptance/test_results_002.py",
                     "tests/acceptance/test_results_003.py", "tests/acceptance/test_results_003b.py",
                     "tests/acceptance/test_results_003e.py", "tests/acceptance/test_results_003f.py"]


def phase_unchanged(args):
    """T057 (SC-010), as 003e's and 003f's: the unit and parity suites and every earlier spec's results test, each by
    pytest in its own process at the committed results; every group 0 failures and 0 errors, every skip named."""
    import tempfile
    import time
    import xml.etree.ElementTree as ET
    import acceptance_003e as E
    groups = {"unit": ["tests/unit"], "parity": ["tests/parity"], "results_001_to_003f": UNCHANGED_RESULTS}
    run = _run("unchanged", args, "pytest in 3 processes: the unit suite (~6 min), the parity suite and the 001-003f "
                                  "results tests; 0 calibrations")
    out = {}
    with tempfile.TemporaryDirectory(prefix="sraging003g_") as tmp:
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
    ok, unlisted = E.unchanged_verdict(out)
    run.write({"status": "PASS" if ok else "FAIL", "groups": out, "skips_unlisted": unlisted,
               "superseded": E.SUPERSEDED, "recorded_skips": E.RECORDED_SKIPS})
    if not ok:
        sys.exit(f"FAIL: {[(g, v['failed_list'][:3]) for g, v in out.items() if v['failures'] or v['errors']]}; "
                 f"skips not named: {unlisted}")


def read_panel_rows():
    import panel_003g as P
    return P.read_ids(os.path.join(REPO, IDS_PATH))


import lev_small_n_003g  # noqa: E402  (T087: lev-small-n)
PHASES = {"panel": phase_panel, "fit1": phase_fit1, "v033-points-verify": phase_v033_points_verify,
          "v033-mle-verify": phase_v033_mle_verify, "p15-v033": phase_p15_v033, "lev-pilot": phase_lev_pilot,
          "pin-w1": _pin_phase(1), "pin-w2": _pin_phase(2), "pin-w3": _pin_phase(3), "score-w1": phase_score_w1,
          "score-w3": phase_score_w3, "sut": phase_sut, "gate": phase_gate, "null-local": phase_null_local,
          "fit2-pick": phase_fit2_pick, "fit2": phase_fit2, "score-w2": phase_score_w2,
          "predecessor": phase_predecessor, "fit2-compare": phase_fit2_compare, "pr3-power": phase_pr3_power,
          "readme": phase_readme, "unchanged": phase_unchanged}
PHASES["lev-small-n"] = lev_small_n_003g.phase_lev_small_n
for _name in ("v033-points-verify", "v033-mle-verify", "lev-pilot", "p15-v033", "fit1", "null-local", "pin-w1",
              "pin-w2", "pin-w3", "sut", "score-w1", "fit2-pick", "fit2", "predecessor", "score-w2", "score-w3",
              "gate", "readme", "unchanged"):
    PHASES.setdefault(_name, _todo(_name))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--head", help="pin-wN: the tester commit jax2 named for the window")
    ap.add_argument("--outside-window", action="store_true", help="pin-wN: jax2 has closed the window (its M7)")
    ap.add_argument("--witness-sums", help="pin-wN: the witness export's SHA256SUMS sha256 from jax2's WINDOW ENDS")
    ap.add_argument("--store-fingerprint", help="pin-wN: the store fingerprint jax2's WINDOW ENDS message carries")
    ap.add_argument("--window-ends", help="pin-wN: the sesync id of jax2's WINDOW ENDS message")
    ap.add_argument("--runs", help="pin-wN, development only: run names instead of the panel's [windows]")
    ap.add_argument("--pins", help="development only: another pins file (never the committed one)")
    ap.add_argument("--library", default="v6", help="development only: another library (a dry run on v5)")
    ap.add_argument("--pin-only", action="store_true", help="predecessor: pin the sealed run, then commit")
    ap.add_argument("--only", help="predecessor: restrict to these cohort ids (the price pilot; never counts)")
    args = ap.parse_args(argv)
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
