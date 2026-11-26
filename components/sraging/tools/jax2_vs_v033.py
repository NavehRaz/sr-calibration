"""Read-only: the COMBINED shift of jax2 1.0.0 (mle_profile_jax2, mle_profile_jax2_h) against v0.3.3 (mle_profile_jax)
on cohorts where both have a stored record. Reads stored JSON only; runs no fit; writes only
tests/acceptance/results/jax2_vs_v033_003.json. Sources are never mixed: "mac" = the J2 tester's stored Mac fits of
jax2 1.0.0's runtime (freeze v100on, dev6on = the same runtime under the dev6 version string, R3d6/L5), "wexac" =
the Phase C store under reports/rc2/C/cluster_home/data/store (cluster, revision 3bc5775). v0.3.3 = the tester's
runs/mle_profile_jax-full.json reference (Mac, library v1). Nothing is evaluated on a common likelihood: fix B changes
it, so log-likelihoods are not compared.
Usage: python tools/jax2_vs_v033.py
HISTORICAL (spec 003e FR-002): pinned to libraries v1 and v2 and to spec 003's stores; it is not moved to the library
setting. For the current reference on v5, use tools/acceptance_003e.py (jcross)."""
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset"   # release 0.1.1: the lab's checkouts
J2 = os.environ.get("SRAGING_J2_ROOT", os.path.join(A, "sr_calibration_tester_jax2"))
T = os.environ.get("SRAGING_TESTER_ROOT", os.path.join(A, "sr_calibration_tester"))
OUT = os.path.join(REPO, "tests", "acceptance", "results", "jax2_vs_v033_003.json")
MAC_DIRS = ("reports/rc2/C/freeze_1.0.0/identity", "reports/rc2/dev6/identity", "reports/rc2/R3d6/L5")
WEXAC_STORE = "reports/rc2/C/cluster_home/data/store"
V033_MANIFEST = os.path.join(T, "runs", "mle_profile_jax-full.json")
PARAMS = ("eta", "beta", "epsilon", "kappa")
NAMES = {"mle_profile_jax2": "jax2", "jax2": "jax2", "mle_profile_jax2_h": "jax2_h", "jax2_h": "jax2_h"}
FILES = {}   # path -> md5 of every file read


def read(p):
    b = open(p, "rb").read()
    FILES[p] = hashlib.md5(b).hexdigest()
    return json.loads(b)


def git(root, *a):
    try:
        return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def claims_by_param(claims):
    return {c["parameter"]: c for c in (claims or []) if "parameter" in c}


def stratum(cid):
    p = cid.split("-")
    return "-".join(p[:2]) if p[1].startswith("k") else "-".join(p[:2])  # arm-kappa or arm-real


def listing_md5(paths):
    return hashlib.md5("\n".join(sorted(paths)).encode()).hexdigest()


# ------------------------------------------------------------------ price (before reading anything heavy)
def price():
    mac = [os.path.join(J2, d, f) for d in MAC_DIRS if os.path.isdir(os.path.join(J2, d))
           for f in sorted(os.listdir(os.path.join(J2, d))) if f.endswith(".json")]
    wx = []
    for dp, _, fs in os.walk(os.path.join(J2, WEXAC_STORE)):
        wx += [os.path.join(dp, f) for f in fs if f.endswith(".json")]
    print(f"PRICE: reads about {len(mac)} Mac JSON + {len(wx)} WEXAC-store JSON + up to ~1150 v0.3.3 store JSON; "
          f"no fit, no write outside {os.path.relpath(OUT, REPO)}; expected seconds to a minute.", flush=True)
    return mac, wx


def mac_records(mac_files):
    """{(cid,name): rec} of jax2 1.0.0's runtime on the Mac; first found wins (STORED_FITS order, as acceptance_003)."""
    out = {}
    for p in mac_files:
        f = os.path.basename(p)
        parts = f[:-5].split("__")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in ("v100on", "dev6on")):
            continue
        try:
            r = read(p)
        except (OSError, ValueError):
            continue
        cid, name = r.get("cid") or r.get("cohort"), NAMES.get(r.get("name"))
        if cid and name and r.get("claims") and (r.get("result") or {}).get("theta"):
            out.setdefault((cid, name), {"claims": r["claims"], "theta": r["result"]["theta"], "loglik": r["result"].get("loglik"),
                                         "path": p, "digest": None, "revision": None, "host": (r.get("meta") or {}).get("host")})
    return out


def wexac_records(wx_files):
    out = {}
    for p in sorted(wx_files):
        r = read(p)
        k, cs = r["key"], r["claim_set"]
        name = NAMES.get(k["method_name"])
        if name and k["method_version"] == "1.0.0" and r.get("status") == "ok":
            out[(k["cohort_id"], name)] = {"claims": cs["claims"], "theta": None, "loglik": None, "path": p,
                                           "digest": k.get("observation_digest"),
                                           "revision": (r.get("code_provenance") or {}).get("revision"), "host": "wexac"}
    return out


def v033_for(cids):
    man = read(V033_MANIFEST)
    out = {}
    for c in cids:
        d = man["keys"].get(c)
        if not d:
            continue
        p = os.path.join(T, "data", "store", d[:2], d[2:4], f"{d}.json")
        if not os.path.exists(p):
            continue
        r = read(p)
        if r["key"].get("method_version") != "0.3.3" or r.get("status") != "ok":
            continue
        out[c] = {"claims": r["claim_set"]["claims"], "digest": r["key"].get("observation_digest"), "path": p,
                  "revision": (r.get("code_provenance") or {}).get("revision")}
    return out


def row(cid, name, a, b, source, obs_note):
    ca, cb = claims_by_param(a["claims"]), claims_by_param(b["claims"])   # a = jax2, b = v0.3.3
    pr = {}
    for q in PARAMS:
        x, y = ca.get(q, {}), cb.get(q, {})
        pa, pb = x.get("point"), y.get("point")
        dex = math.log10(pa / pb) if pa and pb and pa > 0 and pb > 0 else None
        pr[q] = {"jax2_shape": x.get("shape"), "v033_shape": y.get("shape"),
                 "shape_changed": x.get("shape") != y.get("shape"),
                 "jax2_point": pa, "v033_point": pb, "shift_dex": dex}
    return {"cohort": cid, "name": name, "arm": cid.split("-")[0], "stratum": stratum(cid), "source": source,
            "jax2_record": a["path"], "v033_record": b["path"], "observation_check": obs_note,
            "jax2_loglik": a.get("loglik"), "loglik_diff": "different likelihoods, not comparable", "params": pr}


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    i = (len(v) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def summarise(rows):
    s = {"n_cohorts": len(rows), "params": {}}
    for q in PARAMS:
        sh = [abs(r["params"][q]["shift_dex"]) for r in rows if r["params"][q]["shift_dex"] is not None]
        sg = [r["params"][q]["shift_dex"] for r in rows if r["params"][q]["shift_dex"] is not None]
        ch = Counter(f'{r["params"][q]["v033_shape"]} -> {r["params"][q]["jax2_shape"]}' for r in rows
                     if r["params"][q]["shape_changed"])
        top = sorted((r for r in rows if r["params"][q]["shift_dex"] is not None),
                     key=lambda r: -abs(r["params"][q]["shift_dex"]))[:5]
        s["params"][q] = {"n_with_both_points": len(sh), "median_abs_shift_dex": statistics.median(sh) if sh else None,
                          "p90_abs_shift_dex": pct(sh, 0.9), "median_signed_shift_dex": statistics.median(sg) if sg else None,
                          "n_shape_changed": sum(ch.values()), "shape_changes": dict(ch),
                          "top5": [{"cohort": r["cohort"], "shift_dex": r["params"][q]["shift_dex"],
                                    "v033_shape": r["params"][q]["v033_shape"], "jax2_shape": r["params"][q]["jax2_shape"]}
                                   for r in top]}
    return s


def main():
    mac_files, wx_files = price()
    t0 = time.time()
    sources = {"mac": mac_records(mac_files), "wexac": wexac_records(wx_files)}
    meta = {"mac": {"runtime": "Mac (host MC06268844), jax2 1.0.0 runtime (freeze v100on; dev6on = same runtime under the dev6 "
                               "version string; R3d6/L5 plain records), library v2", "dirs": [os.path.join(J2, d) for d in MAC_DIRS],
                    "listing_md5": listing_md5(mac_files)},
            "wexac": {"runtime": "WEXAC cluster, Phase C store, jax2 1.0.0 at revision 3bc5775, library v2",
                      "dir": os.path.join(J2, WEXAC_STORE), "listing_md5": listing_md5(wx_files)}}
    wex_digest = {c: r["digest"] for (c, _), r in sources["wexac"].items() if r["digest"]}
    allc = {c for s in sources.values() for (c, _) in s}
    v033 = v033_for(allc)
    result = {"status": None, "script": "tools/jax2_vs_v033.py", "date": time.strftime("%Y-%m-%d %H:%M"),
              "v033": {"runtime": "Mac, mle_profile_jax 0.3.3, library v1, mle_profile_jax-full manifest", "manifest": V033_MANIFEST,
                       "n_records_read": len(v033)},
              "sources": meta, "loglik": "different likelihoods (fix B changes the likelihood), not comparable; no difference computed",
              "git": {"repo_HEAD": git(REPO, "rev-parse", "HEAD"), "J2_HEAD": git(J2, "rev-parse", "HEAD"),
                      "v033_tester_HEAD": git(T, "rev-parse", "HEAD")},
              "rows": [], "excluded": [], "summary": {}}
    for src, recs in sources.items():
        for (cid, name), a in sorted(recs.items()):
            b = v033.get(cid)
            if b is None:
                continue
            da = a["digest"] or wex_digest.get(cid)
            if da and b["digest"] and da != b["digest"]:
                result["excluded"].append({"cohort": cid, "name": name, "source": src,
                                           "reason": "observation_digest differs between jax2 (library v2) and v0.3.3 (library v1)"})
                continue
            note = "digest equal" if da and b["digest"] else "unchecked (no digest on the record or the cohort is not in the WEXAC store)"
            result["rows"].append(row(cid, name, a, b, src, note))
    rows = result["rows"]
    for src in ("mac", "wexac"):
        result["sources"][src]["n_jax2_records"] = {n: sum(1 for (c, m) in sources[src] if m == n) for n in ("jax2", "jax2_h")}
    if not rows:
        result["status"] = "not_applicable"
        result["reason"] = "no cohort has both a stored jax2 1.0.0 record and a stored v0.3.3 record"
    else:
        result["status"] = "written"
        for src in ("mac", "wexac"):
            for n in ("jax2", "jax2_h"):
                sel = [r for r in rows if r["source"] == src and r["name"] == n]
                if not sel:
                    continue
                result["summary"][f"{src}/{n}"] = {"overall": summarise(sel),
                                                   "by_arm": {a: summarise([r for r in sel if r["arm"] == a])
                                                              for a in sorted({r["arm"] for r in sel})},
                                                   "by_stratum": {a: summarise([r for r in sel if r["stratum"] == a])
                                                                  for a in sorted({r["stratum"] for r in sel})}}
    # extra (D-211): jax2_h against jax2 of the same source and cohort, h being the only difference between the names
    hv = {}
    for src, recs in sources.items():
        pairs = [(c, recs[(c, "jax2")], recs[(c, "jax2_h")]) for (c, n) in sorted(recs) if n == "jax2_h" and (c, "jax2") in recs]
        rr = [row(c, "jax2_h_vs_jax2", h, j, src, "same source") for c, h, j in pairs]
        if rr:
            hv[src] = {"overall": summarise(rr), "by_arm": {a: summarise([r for r in rr if r["arm"] == a])
                                                           for a in sorted({r["arm"] for r in rr})}}
    result["jax2_h_vs_jax2"] = hv
    result["files_read_md5"] = FILES
    result["files_read_listing_md5"] = listing_md5([f"{h} {p}" for p, h in FILES.items()])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(f"{result['status']}: {len(rows)} rows, {len(result['excluded'])} excluded, {len(FILES)} files read, "
          f"{time.time() - t0:.1f}s -> {os.path.relpath(OUT, REPO)}")
    for k, v in result["summary"].items():
        o = v["overall"]
        print(f"\n== {k}: N={o['n_cohorts']}")
        for q in PARAMS:
            p = o["params"][q]
            f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
            print(f"  {q:8s} n={p['n_with_both_points']:3d} med|d|={f(p['median_abs_shift_dex'])} p90={f(p['p90_abs_shift_dex'])} "
                  f"shape changes={p['n_shape_changed']} {p['shape_changes']}")


if __name__ == "__main__":
    sys.exit(main())
