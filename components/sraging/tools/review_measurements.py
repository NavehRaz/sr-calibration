"""Measurements behind the spec 001 review fixes (specs/001-reference-fit/review/), stored as JSON so the
README and the register cite numbers, not recollection. Reads the tester's stored records (read-only).

  host       the 10 named panel cohorts: this host's oracle claims (pilot.json) vs the stored v0.3.3
             reference records (computed on WEXAC): largest |log10 ratio| of a reported value, shape flips
  box_edge   stored reference records with the beta point on the 1e3 box edge and a closed eta claim:
             where the true eta lies relative to the interval
  hazard     the first README example's (Drosophila, now review evidence) life-table hazard by age window

    .venv/bin/python tools/review_measurements.py
"""
import csv
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "src")]
OUT = os.path.join(REPO, "specs", "001-reference-fit", "review", "measurements.json")
EXAMPLE = os.path.join(REPO, "specs/001-reference-fit/review/evidence/drosophila", "drosophila_simons2024_25C_2pct_batch1.csv")
WINDOWS = [(20, 30), (30, 36), (36, 42), (42, 48), (48, 54), (54, 60), (60, 66), (66, 72), (72, 84)]


def stored_records():
    import oracle_v033 as O
    O.load()
    from sraging.adapters import srbench as AD
    t = O.TESTER                                   # SRAGING_TESTER_ROOT: a checkout at the pinned revision
    mods = AD.connect(t)
    ref = json.load(open(os.path.join(t, "runs", f"{AD.REFERENCE_RUN}.json"), encoding="utf-8"))
    store = mods["store.store"].ResultStore(os.path.join(t, "data", "store"))
    recs = {cs.cohort_id: cs for cs in mods["run.execute"].load_results(ref, store)}
    lib = mods["library.load"].Library(root=os.path.join(t, "data", "library", "v1"))
    return recs, lib


def host(recs):
    pilot = json.load(open(os.path.join(REPO, "tests", "acceptance", "results", "pilot.json")))
    named = {n["cohort_id"]: n["role"] for n in pilot["named"]}
    rows = []
    for r in pilot["results"]:
        cid = r["cohort_id"]
        if cid not in named:
            continue
        here = {c["parameter"]: c for c in r["oracle_claims"]}
        st = {c.parameter: c for c in recs[cid].claims}
        dex, flips = 0.0, []
        for p in ("eta", "beta", "epsilon", "kappa"):
            a, b = here[p], st[p]
            if a["shape"] != b.shape:
                flips.append(f"{p}: {b.shape} -> {a['shape']}")
            for k in ("point", "lower", "upper"):
                x, y = a.get(k), getattr(b, k, None)
                if x and y and x > 0 and y > 0:
                    dex = max(dex, abs(math.log10(x / y)))
        rows.append({"cohort_id": cid, "role": named[cid], "max_abs_dex": dex, "shape_flips": flips})
    basin = [r for r in rows if r["role"] == "basin_fires"]
    other = [r for r in rows if r["role"] != "basin_fires"]
    return {"cohorts": rows, "n": len(rows), "n_differ": sum(r["max_abs_dex"] > 0 for r in rows),
            "max_abs_dex": max(r["max_abs_dex"] for r in rows),
            "n_basin": len(basin), "max_abs_dex_basin": max(r["max_abs_dex"] for r in basin),
            "n_other": len(other), "max_abs_dex_other": max(r["max_abs_dex"] for r in other),
            "n_shape_flips": sum(bool(r["shape_flips"]) for r in rows)}


def box_edge(recs, lib):
    truths = lib.truths(list(recs))
    out = {"selected": 0, "below": 0, "inside": 0, "above": 0, "below_by_arm": {}}
    for cid, cs in recs.items():
        cl = {c.parameter: c for c in cs.claims}
        b, e = cl.get("beta"), cl.get("eta")
        if not (b and e and b.point and abs(b.point - 1e3) / 1e3 < 1e-6 and e.shape == "closed"):
            continue
        tr = truths[cid]
        te = tr["eta"] if isinstance(tr, dict) else tr.eta
        out["selected"] += 1
        where = "below" if te < e.lower else "above" if te > e.upper else "inside"
        out[where] += 1
        if where == "below":
            arm = cid.split("-")[0]
            out["below_by_arm"][arm] = out["below_by_arm"].get(arm, 0) + 1
    return out


def box_edge_rule(recs, lib, rtol=1e-6):
    """The BOX_EDGE rule (FR-017b: theta-hat on an UPPER box edge) scored on the stored v0.3.3 records. Only
    claim points are stored, so a parameter whose claim is `none` has no theta-hat here: a record that
    stays quiet with such a parameter is counted as `quiet_unscorable`. Also: eta coverage by outcome."""
    from sraging.compat.v033 import estimator
    box = estimator.box_for(300.0)
    truths = lib.truths(list(recs))
    out = {"N": len(recs), "rtol": rtol, "fires": 0, "quiet_scorable": 0, "quiet_unscorable": 0,
           "fires_by_arm": {}, "fires_by_parameter": {},
           "eta_closed_covered": {"fires": [0, 0], "quiet": [0, 0], "quiet_scorable": [0, 0],
                                  "quiet_unscorable": [0, 0]},
           "eta_closed_covered_by_arm": {}, "all_starts_failed": 0,
           "closed_profile_end_at_box": 0, "eta_closed_covered_profile_end_at_box": [0, 0]}
    for cid, cs in recs.items():
        if (cs.diagnostics or {}).get("error") == "all starts failed":
            out["all_starts_failed"] += 1
        cl = {c.parameter: c for c in cs.claims}
        points = {p: (cl[p].point if p in cl else None) for p in box}
        edge = [p for p, (_, hi) in box.items() if points[p] is not None and points[p] >= hi * (1 - rtol)]
        grid = (cs.diagnostics or {}).get("grid") or {}
        at_box = any(cl[p].shape == "closed" and "box_edge" in (grid.get(p, {}).get("up"), grid.get(p, {}).get("down"))
                     for p in box if p in cl)
        out["closed_profile_end_at_box"] += int(at_box)
        if edge:
            out["fires"] += 1
            arm = cid.split("-")[0]
            out["fires_by_arm"][arm] = out["fires_by_arm"].get(arm, 0) + 1
            for p in edge:
                out["fires_by_parameter"][p] = out["fires_by_parameter"].get(p, 0) + 1
            key = "fires"
        else:
            sub = "quiet_unscorable" if any(v is None for v in points.values()) else "quiet_scorable"
            out[sub] += 1
            key = "quiet"
        e = cl.get("eta")
        if e is not None and e.shape == "closed":
            tr = truths[cid]
            te = tr["eta"] if isinstance(tr, dict) else tr.eta
            hit = int(e.lower <= te <= e.upper)
            keys = [key] + ([sub] if key == "quiet" else [])
            arm = cid.split("-")[0]
            by = out["eta_closed_covered_by_arm"].setdefault(arm, {"fires": [0, 0], "quiet": [0, 0]})
            by[key][0] += hit; by[key][1] += 1
            for k in keys:
                out["eta_closed_covered"][k][0] += hit
                out["eta_closed_covered"][k][1] += 1
            if at_box:
                out["eta_closed_covered_profile_end_at_box"][0] += hit
                out["eta_closed_covered_profile_end_at_box"][1] += 1
    return out


def ulp_noise(csv_path, result_path):
    """max |loglik(z +- 1 ulp) - loglik(z)| over the 8 one-ulp moves of z = log(theta-hat), port value function."""
    import numpy as np
    from sraging import Cohort
    from sraging.compat.v033 import estimator as E
    from sraging.compat.v033 import fp_jax as fj
    r = json.load(open(result_path))
    c = Cohort.from_csv(csv_path, delta=r["cohort"]["delta"])
    t, e = np.asarray(c.ages, float), np.asarray(c.events, int)
    L, kw, _ = E.solver_grid(t, e, r["cohort"]["delta"])
    f, _ = fj.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                          sampling_dt=kw.get("sampling_dt"))
    z = np.log([r["theta_hat"][p] for p in ("eta", "beta", "epsilon", "kappa")])
    f0 = float(f(z))
    moves = []
    for i in range(4):
        for d in (np.inf, -np.inf):
            z1 = z.copy()
            z1[i] = np.nextafter(z1[i], d)
            moves.append(abs(float(f(z1)) - f0))
    return {"cohort": os.path.relpath(csv_path, REPO), "loglik": f0, "max_abs_change": max(moves), "moves": moves}


def hazard():
    rows = list(csv.DictReader(open(EXAMPLE)))
    a = [float(r["death times"]) for r in rows]
    e = [int(r["events"]) for r in rows]

    def window(lo, hi):
        exposure = sum(max(0.0, min(x, hi) - lo) for x in a if x > lo)
        d = sum(1 for x, ev in zip(a, e) if ev and lo <= x < hi)
        return d, exposure
    win = [{"from": lo, "to": hi, "deaths": d, "exposure_days": t, "hazard_per_day": d / t}
           for lo, hi in WINDOWS for d, t in [window(lo, hi)]]
    (d1, t1), (d2, t2) = window(36, 42), window(42, 54)
    z = (math.log(d1 / t1) - math.log(d2 / t2)) / math.sqrt(1 / d1 + 1 / d2)
    return {"windows": win, "ratio_36_42_over_42_54": (d1 / t1) / (d2 / t2), "z": z,
            "p_two_sided": math.erfc(abs(z) / math.sqrt(2))}


def main():
    recs, lib = stored_records()
    ev = os.path.join(REPO, "specs", "001-reference-fit", "review", "evidence", "drosophila")
    out = {"host": host(recs), "box_edge": box_edge(recs, lib), "box_edge_rule": box_edge_rule(recs, lib),
           "box_edge_rule_rtol_1e-3": box_edge_rule(recs, lib, 1e-3),
           "hazard": hazard(), "stored_reference_records": len(recs),
           "ulp_noise": [ulp_noise(os.path.join(ev, "drosophila_simons2024_25C_2pct_batch1.csv"),
                                   os.path.join(ev, "drosophila_simons2024_25C_2pct_batch1.json"))]}
    itp = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female.json")
    if os.path.exists(itp):
        out["ulp_noise"].append(ulp_noise(os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv"), itp))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    h, b, z = out["host"], out["box_edge"], out["hazard"]
    print(f"host: {h['n_differ']}/{h['n']} named cohorts differ, max {h['max_abs_dex']:.2f} dex, "
          f"{h['n_shape_flips']} with shape flips")
    print(f"box edge: {b['selected']} records; true eta below {b['below']}, inside {b['inside']}, "
          f"above {b['above']}; below by arm {b['below_by_arm']}")
    print(f"hazard dip: ratio {z['ratio_36_42_over_42_54']:.2f}, p {z['p_two_sided']:.1e}")
    q = out["box_edge_rule"]
    print(f"BOX_EDGE rule on {q['N']} stored records: fires {q['fires']} {q['fires_by_arm']} {q['fires_by_parameter']}; "
          f"quiet scorable {q['quiet_scorable']}, quiet unscorable {q['quiet_unscorable']}; eta closed covered "
          f"(fires) {q['eta_closed_covered']['fires']} (quiet) {q['eta_closed_covered']['quiet']}; "
          f"all starts failed {q['all_starts_failed']}")
    for u in out["ulp_noise"]:
        print(f"ulp noise {u['cohort']}: max |d loglik| {u['max_abs_change']:.3g} nats")
    print(f"-> {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
