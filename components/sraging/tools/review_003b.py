"""Evidence for spec 003b's review (specs/003b-jax2-101-port/review/sr-naveh_003b.md; Naveh's decisions of
2026-10-21), derived from the committed results so the README and the results test can cite them. Read-only on every
result; writes only specs/003b-jax2-101-port/review/evidence/<name>.json.

    .venv/bin/python tools/review_003b.py provenance     # finding 2: per-phase and per-row library, manifest and lock
    .venv/bin/python tools/review_003b.py ac11           # findings 4, 6, 7: what AC11's flags moved, where it sampled
    .venv/bin/python tools/review_003b.py census-scope   # finding 19: DT_RULE's in-scope J rows by arm and window
    .venv/bin/python tools/review_003b.py lnsrch         # finding 5: L-BFGS-B from each pilot fit's theta-hat (fits none)
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
OUT = os.path.join(REPO, "specs", "003b-jax2-101-port", "review", "evidence")
PANEL = os.path.join(REPO, "tests", "acceptance", "panel_003b.toml")


def load(name):
    return json.load(open(os.path.join(RESULTS, f"{name}_003b.json")))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def write(name, payload, cmd):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as fh:
        json.dump({"tool": "tools/review_003b.py " + cmd, **payload}, fh, indent=1)
        fh.write("\n")
    print("wrote", os.path.relpath(path, REPO))


def provenance():
    """Finding 2 (Naveh: a provenance supplement, no re-runs): what each multi-library or two-port phase ran on, row by
    row, with the manifest each library had and the pins the phase checked before running."""
    import oracle_jax2 as OJ
    import oracle_jax2_101 as OJ1
    pins = tomllib.load(open(PANEL, "rb"))["pins"]
    man = {"v5": sha(os.path.join(OJ1.LIBRARIES["v5"], "manifest.json")),
           "v6": sha(os.path.join(OJ1.LIBRARIES["v6"], "manifest.json")),
           "v2": sha(os.path.join(OJ.LIBRARY_V2, "manifest.json"))}
    locks = {"1.0.0": sha(os.path.join(REPO, "tests", "oracle", "jax2_3bc5775", "mle_profile_jax2", "VERSION_LOCK.json")),
             "1.0.1": sha(os.path.join(REPO, "tests", "oracle", "jax2_6884fc8", "mle_profile_jax2", "VERSION_LOCK.json"))}
    p = load("parity")
    parity_rows = sorted({(r["cohort_id"], r["name"], r["library"]) for r in p["rows"]})
    b = load("bytes")
    pb = tomllib.load(open(PANEL, "rb"))["parity"]["bytes"]
    io = load("identity_off")
    out = {
        "manifests_now": man,
        "manifest_pins_checked_before_running": {
            "parity": {"v5": pins["library_v5_manifest"], "v6": pins["library_v6_manifest"],
                       "code": "tools/acceptance_003b.py phase_parity: refuses a library whose manifest is not the pin, "
                               "and an array whose sha256 is not the panel's"},
            "lev_measure": {"v6": "399c9c555782b4271f0e790099893180a2c5101e2e56318eb87d9c25785f7d2f (each array checked "
                                  "against 003f's panel_003f.toml [p3.arrays_sha256])"},
            "ac11": {"v6": pins["library_v6_manifest"]}, "census": {"v6": pins["library_v6_manifest"]},
            "identity_off": {"v2": None, "note": "not pinned by 003b: G1's lines read through spec 003's 1.0.0 oracle "
                                                 "loader (oracle_jax2.LIBRARY_V2); its manifest's sha256 now is "
                                                 "manifests_now.v2"},
            "bytes": {"v5": pins["library_v5_manifest"], "arrays": {e["cid"]: e["sha256"] for e in pb}}},
        "parity_rows": [{"cohort_id": c, "calibrator": n, "library": lib,
                         "manifest_sha256": pins[f"library_{lib}_manifest"]} for c, n, lib in parity_rows],
        "bytes_items": [{"cohort_id": e["cid"], "name": e["name"], "library": e["library"],
                         "manifest_sha256": pins[f"library_{e['library']}_manifest"]} for e in pb],
        "identity_off": {"library": "v2", "manifest_sha256_now": man["v2"], "ports": locks,
                         "calibrators": sorted({r["name"] for r in io["rows"]})},
        "locks": locks,
        "checks": {"v5_now_equals_pin": man["v5"] == pins["library_v5_manifest"],
                   "v6_now_equals_pin": man["v6"] == pins["library_v6_manifest"],
                   "bytes_rows_are_the_panels": [x["cid"] for x in b["bytes_item"]] == [e["cid"] for e in pb]}}
    write("provenance_003b", out, "provenance")


def ac11():
    """Findings 4, 6, 7: which arm each AC11 flag or non-flag moved, and where the panel sampled (the library spec's
    rho_beta / rho_eta for the library cohorts; the registered points for the planted ones)."""
    a = load("ac11")
    reg = tomllib.load(open(PANEL, "rb"))["ac11"]
    import oracle_jax2_101 as OJ1
    specs = {s["cohort_id"]: s for s in json.load(open(os.path.join(OJ1.LIBRARY_V6, "manifest.json")))["specs"]}
    rows = a["rows"]
    nulls = [r for r in rows if r["mode"] in ("held_out", "small_n")]
    twins = [r for r in rows if r["mode"] == "planted" and r["planted_row"] is None]
    late = [r for r in rows if r["mode"] == "planted" and r["cohort_id"].endswith("-late")
            and r["label"] == "flag" and r["planted_dropped"]]
    age0 = [r for r in rows if r["mode"] == "planted" and r["cohort_id"].endswith("-age0")
            and r["label"] == "flag" and r["planted_dropped"]]
    ratio = {c: specs[c]["rho_beta"] / specs[c]["rho_eta"] for c in (r["cohort_id"] for r in nulls)}
    pts = [p["spec"]["rho_beta"] / p["spec"]["rho_eta"] for p in reg["planted"]]
    out = {
        "nulls": {"n": len(nulls), "R_all4_above_1p5": sum(r["R"]["all4"] > 1.5 for r in nulls),
                  "small_n_flat_valley_slide": sum(r["label"] == "flat_valley_slide" for r in nulls
                                                   if r["mode"] == "small_n"),
                  "small_n": sum(r["mode"] == "small_n" for r in nulls),
                  "max_delta": {m: max(r["delta_total"] for r in rows if r["mode"] == m)
                                for m in ("held_out", "small_n")},
                  "rho_beta_over_rho_eta_below_0p1": sum(v < 0.1 for v in ratio.values()),
                  "rho_beta_over_rho_eta_range": [min(ratio.values()), max(ratio.values())]},
        "twins": {"n": len(twins), "R_all4_above_1p5": sum(r["R"]["all4"] > 1.5 for r in twins),
                  "max_delta": max(r["delta_total"] for r in twins)},
        "planted_points": {"rho_beta_over_rho_eta_range": [min(pts), max(pts)], "n_points": len(pts)},
        "late_detected": {"n": len(late), "R_raw_below_1p5": sum(r["R"]["raw"] < 1.5 for r in late),
                          "R_lead_pair_above_1e6_among_those": sum(r["R"]["raw"] < 1.5 and r["R"]["lead_pair"] > 1e6
                                                                    for r in late)},
        "age0_detected": {"n": len(age0), "R_raw_min": min(r["R"]["raw"] for r in age0),
                          "R_raw_max": max(r["R"]["raw"] for r in age0)},
        "wrong_row_flag": [{"cohort_id": r["cohort_id"], "dropped": r["dropped"], "planted_row": r["planted_row"],
                            "delta_total": r["delta_total"], "warnings": r["warnings"]}
                           for r in rows if r["mode"] == "planted" and r["planted_row"] is not None
                           and r["label"] == "flag" and not r["planted_dropped"]],
        "floored_late_n1000": sum("FLOORED_DEATHS" in r["warnings"] for r in rows
                                  if r["cohort_id"].endswith("-n1000-late")),
        "late_n1000": sum(r["cohort_id"].endswith("-n1000-late") for r in rows)}
    write("ac11_reading", out, "ac11")


def census_scope():
    """Finding 19: DT_RULE's in-scope J rows of the v6 census by arm, and how many are windowed (refused by sraging)."""
    import acceptance_003b as ACC
    pins = tomllib.load(open(PANEL, "rb"))["pins"]
    rows = ACC.census_rows(ACC.CENSUS, pins["census_rows_v6"])
    j = [r for r in rows if r["view"] == "J" and r["in5"]]
    by_arm = collections.Counter(r["arm"] for r in j)
    write("census_scope", {"in_scope_J": len(j), "by_arm": dict(sorted(by_arm.items())),
                           "windowed": sum(bool(r["windowed"]) for r in j),
                           "unwindowed": sum(not r["windowed"] for r in j)}, "census-scope")


def lnsrch():
    """Finding 5: the estimator's own L-BFGS-B (_LBFGS_MLE) from each pilot fit's theta-hat on the full data -- its
    termination, iterations and evaluations, and the gradient norm at theta-hat. No fit; seconds per cohort."""
    import numpy as np
    from scipy.optimize import minimize
    import sraging.compat.jax2_101  # noqa: F401
    import oracle_jax2_101 as OJ1
    from sraging import Cohort
    from sraging import leverage as LEV
    from sraging.adapters.observation import cohort_from_observation
    from sraging.calibrators import profile_jax2 as PJ
    os.environ.setdefault("PJRT_NPROC", "1")
    os.environ.setdefault("SRB_STRICT", "1")
    P1 = PJ.PORTS["1.0.1"]
    est = P1.estimator
    out = []
    for row in load("lev_measure")["rows"]:
        cid = row["cohort_id"]
        c = (Cohort.from_csv(os.path.join(REPO, "tests", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
             if cid == "ITP+age0" else cohort_from_observation(OJ1.observation(cid, "v6")))
        inp = PJ.fit_inputs(LEV._without(c, None), "profile_jax2", P1)
        obj = est._Objective(P1.fp_jax, inp["t"], inp["e"], inp["L"], inp["kw"], inp["box"], hazard=inp["hazard"],
                             window=inp["window"])
        bounds = [(math.log(inp["box"][p][0]), math.log(inp["box"][p][1])) for p in obj.params]
        z = LEV._z(row["theta_hat"], False)
        v0, g = obj.nll_grad(z)
        r = minimize(obj.nll_grad, z, jac=True, method="L-BFGS-B", bounds=bounds, options=est._LBFGS_MLE)
        out.append({"cohort_id": cid, "grad_norm": float(np.linalg.norm(g)), "message": str(r.message),
                    "nit": int(r.nit), "nfev": int(r.nfev), "moved": float(np.max(np.abs(r.x - z)))})
    human = [x for x in out if x["cohort_id"] != "ITP+age0"]
    starts = [abs(r["ll_by_start"]["mle_stage"] - r["ll_by_start"]["warm"]) for r in load("lev_measure")["rows"]
              if r["cohort_id"] != "ITP+age0" and len(r.get("ll_by_start") or {}) == 2]
    write("pilot_lnsrch", {"options": {k: v for k, v in est._LBFGS_MLE.items()}, "rows": out,
                           "itp_grad_norm": next(x["grad_norm"] for x in out if x["cohort_id"] == "ITP+age0"),
                           "human_grad_norm_range": [min(x["grad_norm"] for x in human),
                                                     max(x["grad_norm"] for x in human)],
                           "human_refit_starts_max_disagreement_nats": max(starts),
                           "abnormal_at_nit0": sum(x["nit"] == 0 and x["message"].startswith("ABNORMAL") for x in out),
                           "n": len(out)}, "lnsrch")


if __name__ == "__main__":
    {"provenance": provenance, "ac11": ac11, "census-scope": census_scope, "lnsrch": lnsrch}[sys.argv[1]]()
