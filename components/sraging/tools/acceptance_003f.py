"""Spec 003f's pilot and acceptance phases (specs/003f-leverage-check/contracts/acceptance.md). Nothing runs on WEXAC.

    .venv/bin/python tools/acceptance_003f.py p1              # P1: statistics and influence on ITP+age0 (minutes)
    .venv/bin/python tools/acceptance_003f.py p2              # P2: the MLE-only drop-one against the stored fit
    .venv/bin/python tools/acceptance_003f.py p3-panel        # names P3's cohorts into panel_003f.toml (no fits)
    .venv/bin/python tools/acceptance_003f.py p3              # P3: the null and stratified panel (~3-4 CPU-h)
    .venv/bin/python tools/acceptance_003f.py itp             # SC-001, SC-002
    .venv/bin/python tools/acceptance_003f.py panel           # SC-003, SC-004
    .venv/bin/python tools/acceptance_003f.py readme-example  # SC-005
    .venv/bin/python tools/acceptance_003f.py age0            # SC-006
    .venv/bin/python tools/acceptance_003f.py unchanged       # SC-007

Every phase runs through spec 003's `acceptance_003.Run` (suffix "003f"): a clean committed tree, its price printed
first, its result in tests/acceptance/results/<phase>_003f.json with the leverage provenance (calibrator and version,
host, CPU, library, the rule in force and the commit of panel_003f.toml).
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

PANEL_PATH = "tests/acceptance/panel_003f.toml"
RESULTS_DIR = os.path.join(REPO, "tests", "acceptance", "results")
C3 = os.path.join(REPO, "tests", "acceptance", "results", "checks_003.json")
ITP_AGE0 = os.path.join(REPO, "tests", "data", "itp_plus_age0.csv")
CALIBRATOR = "profile_jax2"


def panel() -> dict:
    return tomllib.load(open(os.path.join(REPO, PANEL_PATH), "rb"))


def _git(*a):
    return subprocess.run(["git", "--no-optional-locks", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def cpu_model() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return platform.processor()


def provenance(library=None) -> dict:
    """The leverage provenance of FR-010 plus the rule in force and where it was committed."""
    from sraging import registry
    from sraging.leverage_rule import RULE
    spec = registry.get(CALIBRATOR)
    return {"calibrator": CALIBRATOR, "version": spec.version, "host": platform.node(),
            "machine": platform.machine(), "cpu": cpu_model(), "library": library, "rule": dict(RULE),
            "panel_commit": _git("log", "-1", "--format=%H", "--", PANEL_PATH)}


def require_rule_matches_panel():
    """The library's rule and the committed panel's [rule] must agree before any phase reads either."""
    from sraging.leverage_rule import RULE
    want = {k: v for k, v in panel()["rule"].items()}
    if want != dict(RULE):
        sys.exit(f"REFUSED: sraging.leverage_rule.RULE {dict(RULE)} differs from {PANEL_PATH} [rule] {want}")


def _run(phase, args, cost):
    import acceptance_003 as A
    return A.Run(phase, manifests=[PANEL_PATH], dev=args.dev or bool(getattr(args, "only", None)), suffix="003f",
                 cost=cost)


# ------------------------------------------------------------------------------------------------- the pilots
def _itp(path=ITP_AGE0):
    from sraging import Cohort
    return Cohort.from_csv(path, delta=1.0, time_unit="days")


def _c3():
    a = json.load(open(C3))["itp_age0"]["absorbed"]
    return a["theta_hat_with"], a["theta_hat_without"]


def _ranked(LEV, sh, set_name, ages, events, k=5):
    r = LEV.ranking(sh, set_name)
    order = [int(j) for j in __import__("numpy").argsort(-r["score"], kind="stable")[:k]]
    return [{"row": j + 1, "age": float(ages[j]), "event": int(events[j]), "score": float(r["score"][j]),
             "shift_dex": {q: float(sh[q][j]) / 2.302585092994046 for q in LEV.QUANTITIES}} for j in order]


def p1(args):
    """FR-009 P1 (research R10): the free statistics and the influence ranking at the stored theta-hat (C3: the 1.0.0
    port's fits on this Mac), on ITP with its added age-0 death and on the README example without it; the Hessian by
    central differences of the port's gradient against forward-over-forward differentiation. Design gate: on
    ITP+age0 the influence under the rule's quantities picks data row 1 (age 0)."""
    import time
    import numpy as np
    from sraging import leverage as LEV
    run = _run("p1", args, "2 cohorts (n 305, 304): terms, their Jacobian, two Hessians each; under a minute; 0 fits")
    with_, without = _c3()
    out, ok = {}, True
    for tag, path, th in (("itp_age0", ITP_AGE0, with_),
                          ("itp_example", os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv"),
                           without)):
        c0 = time.process_time()
        c = _itp(path)
        z = LEV._z(th, False)
        td = LEV.terms_and_derivatives(c, CALIBRATOR, z)
        ages, events = np.asarray(c.ages, float), np.asarray(c.events, int)
        lo, hi = LEV._bounds(td["inputs"]["box"], False)
        inf = LEV.influence(td["G"], td["H"], z, lo, hi)
        sh = LEV.shifts(inf["DZ"], z)
        Hf = LEV.hessian_forward(c, CALIBRATOR, z)
        rel = float(np.max(np.abs(td["H"] - Hf)) / np.max(np.abs(Hf)))
        out[tag] = {"n": int(ages.size), "theta_source": "checks_003.json (the 1.0.0 port, this Mac)",
                    "gap_sd": LEV.gap_sd(ages, events), "share": LEV.share(td["terms"], ages, events),
                    "hessian": {k: inf[k] for k in ("free", "active", "eigenvalues", "min_eigenvalue", "min_relative",
                                                    "n_negative", "condition", "n_cut", "n_kept", "rel_cut")},
                    "hessian_fd_vs_forward_max_rel": rel,
                    "ranking": {s: _ranked(LEV, sh, s, ages, events) for s in LEV.QUANTITY_SETS},
                    "cpu_s": time.process_time() - c0}
        ok = ok and rel < 1e-6
    from sraging.leverage_rule import RULE
    picks = {s: out["itp_age0"]["ranking"][s][0]["row"] for s in LEV.QUANTITY_SETS}
    age0_row = int(np.flatnonzero(np.asarray(_itp().ages, float) == 0.0)[0]) + 1
    gate = picks[RULE["quantities"]] == age0_row
    run.write({"status": "PASS" if (gate and ok) else "FAIL", "design_gate": {
        "rule": f"on ITP+age0 the influence under the rule's quantities ({RULE['quantities']}) picks the age-0 death",
        "age0_row": age0_row, "picks": picks, "passed": gate}, "hessian_agreement": ok, "cohorts": out,
        "provenance": provenance()})
    if not (gate and ok):
        sys.exit(f"FAIL: picks {picks} (age-0 row {age0_row}); Hessian agreement {ok}")


def p2(args):
    """FR-009 P2: the check itself (sraging.leverage.check) on ITP+age0 at C3's theta-hat, dropping the age-0 death.
    Its MLE-stage refit must equal C3's theta-hat without the death (the calibrator's own path); its warm start, from
    C3's theta-hat with the death, must end no lower than that start on the reduced objective; which start stood is
    recorded. The full data's MLE stage against C3's theta-hat with the death says whether the basin re-fit moved
    theta-hat there (recorded; the check compares with the reported theta-hat either way)."""
    import numpy as np
    from sraging import leverage as LEV
    run = _run("p2", args, "the check on ITP+age0: 1 drop with 2 refits (MLE stage, warm start) + 1 full-data MLE stage "
               "+ evaluations; ~0.1-0.3 CPU-h; no full fit")
    with_, without = _c3()
    c = _itp()
    row0 = int(np.flatnonzero(np.asarray(c.ages, float) == 0.0)[0])
    res = LEV.check(c, CALIBRATOR, with_, {"basin_fix": {"n_refit": 0}}, drop_rows={"top": row0})
    full = LEV.refit_mle(c, CALIBRATOR, None)
    zw = np.log([without[p] for p in LEV.PARAMS])
    rm = ((res.get("refits") or {}).get("mle_stage") or {}).get("theta")
    zr = np.log(rm) if rm else None
    zf = np.log(full["theta"]) if full.get("theta") else None
    zh = np.log([with_[p] for p in LEV.PARAMS])
    d_without = None if zr is None else [float(x) for x in zr - zw]
    d_with = None if zf is None else [float(x) for x in zf - zh]
    eq_without = d_without is not None and max(abs(x) for x in d_without) < 1e-9
    eq_with = d_with is not None and max(abs(x) for x in d_with) < 1e-9
    by = res.get("ll_by_start") or {}
    warm_not_below = "warm" in by and by["warm"] >= res["ll_ref"]
    ok = eq_without and warm_not_below and res["dropped"]["row"] == row0 + 1
    run.write({"status": "PASS" if ok else "FAIL",
               "refit_vs_stored_without": {"log_diff": d_without, "equal_to_round_off": eq_without},
               "warm_start": {"ll_ref": res.get("ll_ref"), "ll_by_start": by, "not_below_its_start": warm_not_below,
                              "start_that_stood": (res.get("refit") or {}).get("start")},
               "full_mle_stage_vs_stored_with": {"log_diff": d_with, "equal_to_round_off": eq_with,
                                                 "reads_as": "the basin re-fit did not move theta-hat" if eq_with
                                                 else "theta-hat is not the MLE stage's (recorded only)"},
               "check": res, "stored": {"theta_hat_with": with_, "theta_hat_without": without,
                                        "shift_dex": json.load(open(C3))["itp_age0"]["absorbed"]["shift_dex"]},
               "full_mle_stage": full, "provenance": provenance()})
    if not ok:
        sys.exit(f"FAIL: MLE-stage refit equal {eq_without}, warm start not below its start {warm_not_below}, "
                 f"dropped {res.get('dropped')}")


# ---------------------------------------------------------------------------------------------- P3's panel (T010)
def _tester():
    """The pinned tester clone's modules, the content pins and the tester's data root (spec 003e)."""
    import acceptance_003e as E
    return E, E._mods(), E._pins()


def _terms_at_truth(cohort, cid, truth):
    """The port's per-individual terms at the library truth on the fit's own grid (profile_jax2_h on hazard cohorts)."""
    import math
    import numpy as np
    import jax.numpy as jnp
    from sraging import leverage as LEV
    from sraging.calibrators import profile_jax2 as PJ
    name = "profile_jax2_h" if cid.startswith("hazard-") else CALIBRATOR
    z = [math.log(truth[p]) for p in LEV.PARAMS]
    if name == "profile_jax2_h":
        z.append(math.log(truth["external_hazard"]))
    _f, _vg, tf = LEV._loglik(PJ.fit_inputs(cohort, name), with_terms=True)
    return np.asarray(tf(jnp.asarray(z))["term"], float), name


def p3_panel(args):
    """T010: P3's cohorts by the gate's C8, in v6 library order, read from the pinned v6 manifest and the pinned J v5
    records; no fit. Writes the result first, then panel_003f.toml [p3] (commit it before p3 runs)."""
    import csv
    import glob
    import hashlib
    import re
    import numpy as np
    from sraging.adapters import srbench as ADP
    from sraging.adapters.observation import cohort_from_observation
    run = _run("p3-panel", args, "manifests and ~100 store records read; terms at the truth on the panel's v6 cohorts "
               "with one age-0 death; minutes; 0 fits")
    E, mods, pins = _tester()
    T = E.TESTER
    man = json.loads(pins.read_file(os.path.join("data", "library", "v6", "manifest.json")))
    specs = {sp["cohort_id"]: sp for sp in sorted(man["specs"], key=lambda sp: sp["position"])}
    order = list(specs)
    usable = lambda c: specs[c]["group"] != "sharp" and specs[c].get("window") is None  # noqa: E731
    basin = [r["cohort_id"] for r in csv.DictReader(open(os.path.join(T, "reports", "basin", "scope",
                                                                       "label_final.csv")))
             if r["fail_mle"] == "True" and "ecoli" not in r["cohort_id"] and r["real_name"]]
    ref = E._side("reference", pins)
    store = mods["store.store"].ResultStore(ref["store_root"])
    fired = {}
    for c in basin:
        if c in ref["manifest"]["keys"]:
            rec = json.loads(pins.read_record(ref["run_id"], c, ref["manifest"]["keys"][c], store))
            fired[c] = bool(((rec["claim_set"].get("diagnostics") or {}).get("basin_fix") or {}).get("fired"))
    human_no = [c for c in order if c in fired and not fired[c] and usable(c)][:10]
    human_bf = [c for c in order if fired.get(c) and usable(c)][:3]
    pure_main = [c for c in order if specs[c]["arm"] == "pure" and specs[c]["group"] == "main"
                 and specs[c]["n"] == 1000 and not specs[c].get("real_name") and usable(c)][:5]
    censor_main = [c for c in order if specs[c]["arm"] == "censor" and specs[c]["group"] == "main"
                   and not specs[c].get("real_name") and usable(c)][:5]
    panel_ids = set()
    for f in glob.glob(os.path.join(REPO, "tests", "acceptance", "panel*.toml")):
        if not f.endswith("panel_003f.toml"):
            panel_ids |= set(re.findall(r'"((?:pure|censor|hazard|window|het|realfull|nlad)-[A-Za-z0-9_.\-]+)"',
                                        open(f).read()))
    lib = mods["library.load"].Library(root=ADP.library_root("v6", T))
    cand = [c for c in order if c in panel_ids and usable(c)]
    truths = lib.truths(cand)
    age0, scanned = [], []
    for c in cand:
        obs = lib.observation(c)
        a, e = np.asarray(obs.ages, float), np.asarray(obs.events, int)
        k = int(np.sum((a == 0) & (e == 1)))
        if k != 1:
            continue
        terms, name = _terms_at_truth(cohort_from_observation(obs), c, truths[c])
        i0 = int(np.flatnonzero((a == 0) & (e == 1))[0])
        below = bool(terms[i0] < np.min(np.delete(terms, i0)))
        scanned.append({"cohort": c, "age0_term": float(terms[i0]), "min_other": float(np.min(np.delete(terms, i0))),
                        "calibrator_for_terms": name, "below_all_others": below})
        if below:
            age0.append(c)
    strata = {"human_no_basin_fix": human_no, "human_basin_fix": human_bf, "main_pure_n1000": pure_main,
              "main_censor": censor_main, "v6_single_age0": age0}
    ids = [c for v in strata.values() for c in v]
    arrays = {c: hashlib.sha256(open(os.path.join(T, "data", "library", "v6", "cohorts", f"{c}.npz"),
                                     "rb").read()).hexdigest() for c in ids}
    res = run.write({"status": "written", "strata": strata, "n": len(ids), "age0_scan": scanned,
                     "panel_ids_scanned": len(cand), "v6_manifest_sha256": pins.data["libraries"]["v6"]["sha256"],
                     "arrays_sha256": arrays, "provenance": provenance("v6")}, name="p3_panel")
    text = open(os.path.join(REPO, PANEL_PATH)).read()
    block = "\n".join([f"{k} = {json.dumps(v)}" for k, v in strata.items()]
                      + [f"v6_manifest_sha256 = {json.dumps(pins.data['libraries']['v6']['sha256'])}",
                         "[p3.arrays_sha256]"] + [f"{json.dumps(c)} = {json.dumps(h)}" for c, h in arrays.items()])
    head, _sep, _old = text.partition("[p3]")
    rule_text = re.search(r'rule = """.*?"""', text, re.S).group(0)
    open(os.path.join(REPO, PANEL_PATH), "w").write(head + "[p3]\n" + rule_text + "\n" + block + "\n")
    print(f"[p3-panel] {len(ids)} cohorts: " + ", ".join(f"{k} {len(v)}" for k, v in strata.items()))
    return res


# ------------------------------------------------------------------------------------------------------- P3 (T011)
def wilson(k, n, z=1.959963984540054):
    """The Wilson 95 % interval of k successes in n."""
    import math
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def _p3_worker(item):
    """One P3 cohort: the production fit (profile_jax2, combinations off; it runs the check as users get it), then
    the check again with the drops of the top- and the median-influence individual under the rule's quantities (one
    drop per distinct row; the top repeats the production drop, a determinism check), each against the reported
    theta-hat. Every drop's R is recorded for every quantity set."""
    import hashlib
    import time
    cid, stratum, want_sha = item
    t0 = time.time()
    E, mods, pins = _tester()
    import sraging
    from sraging import leverage as LEV
    from sraging.adapters import srbench as ADP
    from sraging.adapters.observation import cohort_from_observation
    path = os.path.join(E.TESTER, "data", "library", "v6", "cohorts", f"{cid}.npz")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != want_sha:
        return {"cohort_id": cid, "status": "refused", "reason": f"array sha256 {sha[:16]} is not the panel's"}
    lib = mods["library.load"].Library(root=ADP.library_root("v6", E.TESTER))
    c = cohort_from_observation(lib.observation(cid))
    r = sraging.fit_cohort(c, calibrator=CALIBRATOR, combinations=False)
    if r.status != "ok":
        return {"cohort_id": cid, "stratum": stratum, "status": f"fit:{r.status}", "wall_s": time.time() - t0}
    fit_cpu = r.timing.get("cpu_s")
    prod = r.diagnostics.get("leverage") or {}
    from sraging.leverage_rule import RULE
    q = RULE["quantities"]
    keys = {}
    for kind in ("top", "median"):
        j = (prod.get("influence") or {}).get(q, {}).get(kind, {}).get("row")
        if j is not None:
            keys.setdefault(j - 1, f"{kind}_{q}")
    aliases = {f"{kind}_{q}": keys.get(((prod.get("influence") or {}).get(q, {}).get(kind, {}).get("row") or 0) - 1)
               for kind in ("top", "median")}
    extra = LEV.check(c, CALIBRATOR, r.theta_hat, r.diagnostics, library="v6", fit_cpu_s=fit_cpu,
                      drop_rows={k: j for j, k in keys.items()})
    bf = r.diagnostics.get("basin_fix") or {}
    return {"cohort_id": cid, "stratum": stratum, "status": "ok", "n": c.n, "n_deaths": c.n_deaths,
            "fit_cpu_s": fit_cpu, "basin_fix": {"fired": bool(bf.get("fired")), "n_refit": bf.get("n_refit")},
            "production": dict({k: prod.get(k) for k in ("label", "reason", "dropped", "R", "delta_total", "delta_nats",
                                                         "labels", "ll_ref", "ll_new", "ll_by_start", "n_tied_top",
                                                         "basin_fix_fired", "at_theta_hat", "cost", "gap_sd",
                                                         "share")},
                               refit_start=(prod.get("refit") or {}).get("start"),
                               provenance=dict(prod.get("provenance") or {}, library="v6",
                                               library_source="the P3 tool (a calibrator cannot know the library)")),
            "influence": prod.get("influence"), "drops": extra.get("drops"), "aliases": aliases,
            "extra_label": extra.get("label"), "extra_reason": extra.get("reason"),
            "warnings": [w.code for w in r.warnings], "wall_s": time.time() - t0}


def p3_summary(rows):
    """Labels over the median- and the top-influence drops (selected under the rule's quantities) under every quantity
    set, flags by stratum and pooled with the Wilson 95 % interval (reported, Naveh 2026-10-19), which start stood, and
    the production check's overhead as a share of the fit."""
    import collections
    import numpy as np
    from sraging import leverage as LEV
    from sraging.leverage_rule import RULE
    q = RULE["quantities"]
    ok = [r for r in rows if r.get("status") == "ok"]
    out = {"drops_selected_by": q}
    for s_ in LEV.QUANTITY_SETS:
        per, pooled, med_labels, tops = {}, [0, 0], collections.Counter(), collections.Counter()
        for r in ok:
            d = (r.get("drops") or {}).get(r["aliases"].get(f"median_{q}")) or {}
            lab = (d.get("labels") or {}).get(s_, "failed" if d else None)
            if lab is None:
                continue
            med_labels[lab] += 1
            st = per.setdefault(r["stratum"], [0, 0])
            st[0] += lab == "flag"
            st[1] += 1
            pooled[0] += lab == "flag"
            pooled[1] += 1
            dt = (r.get("drops") or {}).get(r["aliases"].get(f"top_{q}")) or {}
            tops[(dt.get("labels") or {}).get(s_, "failed" if dt else None)] += 1
        out[s_] = {"median_false_flags": {k: {"flags": v[0], "n": v[1], "wilson95": wilson(*v)} for k, v in per.items()},
                   "median_pooled": {"flags": pooled[0], "n": pooled[1], "wilson95": wilson(*pooled)},
                   "median_labels": dict(med_labels), "top_labels": {str(k): v for k, v in tops.items()}}
    starts = collections.Counter((d.get("refit") or {}).get("start") for r in ok for d in (r.get("drops") or {}).values())
    out["start_that_stood"] = {str(k): v for k, v in starts.items()}
    out["production_labels"] = dict(collections.Counter(r["production"].get("label") for r in ok))
    out["top_repeats_production"] = {r["cohort_id"]: ((r.get("drops") or {}).get(r["aliases"].get(f"top_{q}")) or {})
                                     .get("ll_new") == r["production"].get("ll_new") for r in ok}
    shares = [r["production"]["cost"]["share_of_fit"] for r in ok
              if (r["production"].get("cost") or {}).get("share_of_fit") is not None]
    out["overhead_share_of_fit"] = ({"n": len(shares), "median": float(np.median(shares)),
                                     "p90": float(np.quantile(shares, 0.9)), "max": float(max(shares))}
                                    if shares else {"n": 0})
    out["basin_fix_fired_on_v6"] = [r["cohort_id"] for r in ok if r["basin_fix"]["fired"]]
    return out


def p3(args):
    """FR-009 P3 (T011): the null and stratified panel of panel_003f.toml [p3], on this Mac, workers in parallel."""
    import acceptance as A1
    pn = panel()["p3"]
    strata = [k for k in ("human_no_basin_fix", "human_basin_fix", "main_pure_n1000", "main_censor", "v6_single_age0")]
    items = [(c, k, pn["arrays_sha256"][c]) for k in strata for c in pn[k] if not args.only or c in args.only]
    run = _run("p3", args, f"{len(items)} cohorts: the production fit (combinations off; its check: 1 drop, 2 refits) "
               f"plus the check again with 2 drops; ~8-9 Mac CPU-h (P3 of 2026-10-18: fits 7.0 CPU-h, refits ~45 "
               f"CPU-s each), {args.workers} workers; keep the lid open; 0 WEXAC")
    rows = A1.run_pool(_p3_worker, items, args.workers)
    summ = p3_summary(rows)
    bad = [r for r in rows if r.get("status") != "ok"]
    run.write({"status": "written" if not bad else "incomplete", "n": len(items), "not_ok": bad,
               "summary": summ, "rows": rows, "provenance": provenance("v6")})


# ------------------------------------------------------------------------------------------------- acceptance
P3_FUNCS = ("_p3_worker", "p3_summary", "wilson", "p3", "_tester")
P3_EXCEPTION = "src/sraging/leverage_rule.py"     # holds only the rule, which selects among P3's recorded options


def p3_still_valid() -> list:
    """Why P3's stored drops can no longer be re-scored ([] = still valid): any change since P3 ran to src/sraging
    other than the rule module, or to the P3 functions of this tool (compared by source, not by file)."""
    import ast
    d = json.load(open(os.path.join(RESULTS_DIR, "p3_003f.json")))
    head = d["repo_head"]
    problems = [f for f in _git("diff", "--name-only", head, "HEAD", "--", "src/sraging").split() if f != P3_EXCEPTION]
    then = subprocess.run(["git", "-C", REPO, "show", f"{head}:tools/acceptance_003f.py"], capture_output=True,
                          text=True).stdout
    now = open(os.path.join(REPO, "tools", "acceptance_003f.py")).read()

    def funcs(src):
        t = ast.parse(src)
        return {n.name: ast.get_source_segment(src, n) for n in t.body if isinstance(n, ast.FunctionDef)}
    a, b = funcs(then), funcs(now)
    problems += [f"tools/acceptance_003f.py:{f}" for f in P3_FUNCS if a.get(f) != b.get(f)]
    return problems


def _rescore(rows, set_name, kind, select=None):
    """Labels of the `kind` ("top"/"median") drops (selected under `select`, default `set_name`) under `set_name`,
    from P3's stored drops; a drop with no labels (both refits failed) is `failed`."""
    out = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        dd = (r.get("drops") or {}).get(r["aliases"].get(f"{kind}_{select or set_name}")) or {}
        out[r["cohort_id"]] = {"stratum": r["stratum"], "label": (dd.get("labels") or {}).get(
                                   set_name, "failed" if dd else None),
                               "R": (dd.get("R") or {}).get(set_name), "delta_total": dd.get("delta_total"),
                               "start": (dd.get("refit") or {}).get("start"), "row": dd.get("row"), "age": dd.get("age")}
    return out


def itp(args):
    """SC-001, SC-002 and FR-014 on this fit: profile_jax2.fit on ITP+age0 (the check inside it); the check picks the
    age-0 death and flags under the rule; its MLE-stage refit equals C3's fit without the death (which start stood is
    recorded); theta-hat and loglik equal C3's fit with it (the check changes nothing)."""
    import numpy as np
    import sraging
    from sraging.cli import _leverage_line
    from sraging.leverage_rule import RULE
    run = _run("itp", args, "one profile_jax2 fit of ITP+age0 with the check (~1,600 CPU-s, combinations off); 0 WEXAC")
    with_, without = _c3()
    a = json.load(open(C3))["itp_age0"]["absorbed"]
    c = _itp()
    r = sraging.fit_cohort(c, calibrator=CALIBRATOR, combinations=False)
    lev = r.diagnostics.get("leverage") or {}
    age0_row = int(np.flatnonzero(np.asarray(c.ages, float) == 0.0)[0]) + 1
    th = r.theta_hat.as_dict()
    same_theta = all(th[p] == with_[p] for p in ("eta", "beta", "epsilon", "kappa"))
    same_ll = r.loglik == a["loglik_with"]
    rm = ((lev.get("refits") or {}).get("mle_stage") or {}).get("theta")
    zr = np.log(rm) if rm else None
    zw = np.log([without[p] for p in ("eta", "beta", "epsilon", "kappa")])
    refit_eq = zr is not None and float(np.max(np.abs(zr - zw))) < 1e-9
    sc001 = (lev.get("dropped") or {}).get("row") == age0_row
    sc002 = lev.get("label") == "flag" and refit_eq and "LEVERAGE" in [w.code for w in r.warnings]
    ok = sc001 and sc002 and same_theta and same_ll
    run.write({"status": "PASS" if ok else "FAIL", "SC-001": sc001, "SC-002": sc002,
               "FR-014": {"theta_hat_equal_to_C3": same_theta, "loglik_equal_to_C3": same_ll},
               "age0_row": age0_row, "rule": dict(RULE), "leverage": lev, "printed": _leverage_line(r),
               "warnings": [w.code for w in r.warnings], "stored_shift_dex": a["shift_dex"],
               "refit_vs_stored_without_max_log": None if zr is None else float(np.max(np.abs(zr - zw))),
               "start_that_stood": (lev.get("refit") or {}).get("start"),
               "fit_cpu_s": r.timing.get("cpu_s"), "provenance": provenance()})
    if not ok:
        sys.exit(f"FAIL: SC-001 {sc001}, SC-002 {sc002}, theta {same_theta}, loglik {same_ll}")


def panel_phase(args):
    """SC-003 (REPORTED, Naveh 2026-10-19: its ceiling was set after P3 and P3 shaped the rule again; the held-out and
    small-n nulls run in 003b AC11) and SC-004 (gate): P3's stored drops under the rule in force (valid while P3's code
    is unchanged)."""
    from sraging.leverage_rule import RULE
    run = _run("panel", args, "re-scores p3_003f.json; seconds; 0 fits")
    stale = p3_still_valid()
    if stale:
        sys.exit(f"REFUSED: P3's code changed since it ran ({stale}): re-run p3 first")
    d = json.load(open(os.path.join(RESULTS_DIR, "p3_003f.json")))
    gate = panel()["gate"]
    q = RULE["quantities"]
    med = _rescore(d["rows"], q, "median")
    flags = sum(v["label"] == "flag" for v in med.values())
    n = sum(v["label"] is not None for v in med.values())
    w = wilson(flags, n)
    per = {}
    for v in med.values():
        x = per.setdefault(v["stratum"], [0, 0])
        x[0] += v["label"] == "flag"
        x[1] += 1
    over = d["summary"]["overhead_share_of_fit"]
    small = {k: v for k, v in med.items() if next(r["n"] for r in d["rows"] if r["cohort_id"] == k) <= 10000}
    sc004 = over["n"] == d["n"] and over["p90"] <= gate["overhead_p90_ceiling"]
    run.write({"status": "PASS" if sc004 else "FAIL", "SC-003": "REPORTED", "SC-004": sc004, "rule": dict(RULE),
               "gate": gate, "median_drops": {"flags": flags, "n": n, "wilson95": w,
               "labels": {lab: sum(v["label"] == lab for v in med.values()) for lab in
                          sorted({v["label"] for v in med.values()}, key=str)},
               "by_stratum": {k: {"flags": a, "n": b, "wilson95": wilson(a, b)} for k, (a, b) in per.items()},
               "effective_null": {"n_le_10000": len(small), "flags": sum(v["label"] == "flag" for v in small.values()),
                                  "wilson95": wilson(sum(v["label"] == "flag" for v in small.values()), len(small))},
               "n_range": [min(r["n"] for r in d["rows"] if r.get("status") == "ok"),
                           max(r["n"] for r in d["rows"] if r.get("status") == "ok")]},
               "top_drops": _rescore(d["rows"], q, "top"), "median_detail": med, "overhead": over,
               "other_sets_for_the_record": {s_: d["summary"][s_] for s_ in d["summary"]
                                             if s_ not in (q,) and isinstance(d["summary"][s_], dict)
                                             and "median_pooled" in d["summary"][s_]},
               "p3": {"repo_head": d["repo_head"], "written_at": d["written_at"]}, "provenance": provenance("v6")})
    if not sc004:
        sys.exit(f"FAIL: SC-004 {sc004} ({over})")


def age0(args):
    """SC-006 (reported): the check on the v6 cohorts with a single age-0 death below their other terms (P3's
    stratum), under the rule in force."""
    from sraging.leverage_rule import RULE
    run = _run("age0", args, "reads p3_003f.json; seconds; 0 fits")
    stale = p3_still_valid()
    if stale:
        sys.exit(f"REFUSED: P3's code changed since it ran ({stale}): re-run p3 first")
    d = json.load(open(os.path.join(RESULTS_DIR, "p3_003f.json")))
    rows = [r for r in d["rows"] if r.get("stratum") == "v6_single_age0"]
    rep = {r["cohort_id"]: {"production": r["production"], "influence_top": (r.get("influence") or {}).get(
        RULE["quantities"], {}).get("top"), "warnings": r.get("warnings")} for r in rows}
    run.write({"status": "REPORTED", "n": len(rows), "cohorts": rep, "rule": dict(RULE),
               "p3": {"repo_head": d["repo_head"]}, "provenance": provenance("v6")})


def readme_example(args):
    """SC-005 (reported): the README example under profile_jax2 with the check (the README's own command), beside
    v0.3.3's measurement of its three early deaths; the stored example is regenerated (written after the result)."""
    import shutil
    import tempfile
    run = _run("readme-example", args, "one profile_jax2 fit of the README example, lead combinations (~1,600 CPU-s)")
    stored_path = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")
    with tempfile.TemporaryDirectory(prefix="sraging003f_") as tmp:
        out = os.path.join(tmp, "fit.json")
        p = subprocess.run([os.path.join(REPO, ".venv", "bin", "sraging"), "fit-cohort", "--csv",
                            os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv"), "--delta", "1",
                            "--time-unit", "days", "--calibrator", CALIBRATOR, "--json", out],
                           capture_output=True, text=True, cwd=REPO)
        if p.returncode not in (0, 1) or not os.path.exists(out):
            sys.exit(f"FAIL: fit-cohort exited {p.returncode}: {p.stderr[-1500:]}")
        new_d = json.load(open(out))
        old_d = json.load(open(stored_path))
        same_claims = new_d["claims"] == old_d["claims"]
        same_theta = new_d["theta_hat"] == old_d["theta_hat"]
        lev = new_d["diagnostics"].get("leverage") or {}
        ev = json.load(open(os.path.join(REPO, "specs", "001-reference-fit", "review", "evidence",
                                         "itp_early_deaths_leverage.json")))
        res = run.write({"status": "REPORTED", "claims_equal_to_stored": same_claims,
                         "theta_hat_equal_to_stored": same_theta, "leverage": lev, "printed": p.stdout.strip()
                         .split("\n"), "v033_early_deaths_evidence": {k: ev[k] for k in list(ev)[:8]},
                         "provenance": provenance()})
        if res["status_for_acceptance"] == "acceptance" and same_claims and same_theta:
            shutil.copyfile(out, stored_path)
    if not (same_claims and same_theta):
        sys.exit(f"FAIL: the refit's claims ({same_claims}) or theta-hat ({same_theta}) differ from the stored example")


def unchanged(args):
    """SC-007, as 003e's: the unit and parity suites and the results tests of 001-003e, every skip named."""
    import acceptance_003e as E
    import time
    import xml.etree.ElementTree as ET
    groups = dict(E.UNCHANGED_GROUPS)
    groups["results_001_002_003"] = groups["results_001_002_003"] + ["tests/acceptance/test_results_003e.py"]
    run = _run("unchanged", args, "pytest in 3 processes: the unit suite (~2.5 min), the parity suite and the 001-003e "
               "result tests; several minutes; 0 calibrations")
    out = {}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="sraging003f_") as tmp:
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
               "superseded": E.SUPERSEDED, "recorded_skips": E.RECORDED_SKIPS, "provenance": provenance()})
    if not ok:
        sys.exit(f"FAIL: {[(g, v['failed_list'][:3]) for g, v in out.items() if v['failures'] or v['errors']]}; "
                 f"skips not named: {unlisted}")


PHASES = {"p1": p1, "p2": p2, "p3-panel": p3_panel, "p3": p3, "itp": itp, "panel": panel_phase,
          "readme-example": readme_example, "age0": age0, "unchanged": unchanged}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="p3: restrict to these cohorts (a smoke run; never counts)")
    args = ap.parse_args(argv)
    require_rule_matches_panel()
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
