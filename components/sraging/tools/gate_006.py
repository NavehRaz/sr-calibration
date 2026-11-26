"""Spec 006's gate: the scoring of W on v6's window arm (spec.md rules R-1 to R-4; research R11-R13; tasks T025).

The counts come from the tester's own scorer (`score.claims.coverage`, shape class "closed": declines leave the
denominator, failures count) on ClaimSets, through 003g's pinned clone and `Pins`; this module judges them:

- R-1: coverage per gated parameter (eta, epsilon) against a cut computed from its frozen formula at that parameter's
  own number of closed claims -- the midpoint of the Wilson lower bound at a true p_good and the Wilson upper bound at
  a true p_bad, at n_eff = n_closed / D. beta and kappa are reported with their denominators, never gated. Since
  2026-11-11 (Naveh, on sr-naveh's review): A1 gates; A2, W's data-determined one-sided claims and the lead pair on
  Dg's cohorts are judged against a collapse (p_bad 0.70); closed and one-sided coverage, closure and cap rates,
  misses by side and truncation above are reported.
- R-2: W's widths against H's, paired over cohorts closed in both, a truncated-above claim counted as open (the gate,
  at most +0.1 dex); against the null's box (reported: it cannot fail).
- R-3: the truth witness, ll(truth) - ll(theta-hat) <= 1.92 on every gated cohort, each failure named; a missing truth
  is a failure; 2(ll-hat - ll(truth)) reported against chi-square.
- R-4: attribution, exact: H's log-likelihood at (theta-hat_W, h of H's lowest-h stored node) within 0.1 nats below
  that node, on every cohort.
"""
from __future__ import annotations

import ast
import json
import math
import os
import statistics
import sys
import time

GATED = ("eta", "epsilon")
REPORTED = ("beta", "kappa")


def wilson(p: float, n: float, z: float) -> tuple[float, float]:
    """The Wilson score interval (lower, upper) for a proportion p on n trials (n may be an effective, real n)."""
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return centre - half, centre + half


def r1_cut(n_closed: int, *, D: float, p_good: float, p_bad: float, z: float) -> float:
    """R-1's cut at n_closed closed claims (research R12): the midpoint of wilson(p_good)'s lower and wilson(p_bad)'s
    upper bound at n_eff = n_closed / D."""
    n_eff = n_closed / D
    return 0.5 * (wilson(p_good, n_eff, z)[0] + wilson(p_bad, n_eff, z)[1])


def r1_verdict(hits: int, claimed: int, rule: dict) -> dict:
    """One gated parameter: its coverage on `claimed` closed claims against the cut at that n. No closed claim never
    passes."""
    if not claimed:
        return {"hits": hits, "claimed": claimed, "coverage": None, "cut": None, "passes": False}
    cut = r1_cut(claimed, D=rule["D"], p_good=rule["p_good"], p_bad=rule["p_bad"], z=rule["z"])
    cov = hits / claimed
    return {"hits": hits, "claimed": claimed, "coverage": cov, "cut": cut, "passes": cov >= cut}


def r1_table(counts: dict, rule: dict) -> dict:
    """{param: (hits, claimed)} -> the gated verdicts (rule["gated"]), the reported coverages (rule["reported"]) and
    the overall pass (every gated parameter passes)."""
    gated = {q: r1_verdict(*counts[q], rule) for q in rule["gated"]}
    reported = {q: {"hits": counts[q][0], "claimed": counts[q][1],
                    "coverage": counts[q][0] / counts[q][1] if counts[q][1] else None} for q in rule["reported"]}
    return {"gated": gated, "reported": reported, "passes": all(v["passes"] for v in gated.values())}


def coverage_counts(CL, sets: list, truths: dict, params=GATED + REPORTED) -> dict:
    """{param: (hits, claimed)}: the tester's own closed-claim coverage (score.claims.coverage) on `sets`."""
    out = {}
    for q in params:
        c = CL.coverage(sets, truths, q, "closed").to_dict()
        out[q] = (int(c["hits"]), int(c["claimed"]))
    return out


def closed_bounds(sets, quantity: str) -> dict:
    """{cohort_id: (lower, upper)} of each set's closed claim on `quantity` with lower > 0, else None."""
    out = {}
    for cs in sets:
        c = cs.claim_for(quantity) if cs.ok else None
        out[cs.cohort_id] = ((c.lower, c.upper) if c is not None and c.is_claim and c.shape == "closed"
                             and c.lower and c.lower > 0 else None)
    return out


def paired_log10_ratio(a: dict, b: dict) -> dict:
    """Over cohorts closed in both: the median of log10(width_a / width_b), width = upper / lower -- the difference of
    the two dex widths, log10(upper / lower), as the tester's information axis measures a width (research R11)."""
    r = [math.log10(a[k][1] / a[k][0]) - math.log10(b[k][1] / b[k][0]) for k in a if a.get(k) and b.get(k)]
    return {"n_paired": len(r), "median_dex": statistics.median(r) if r else None}


def r2_gate(w: dict, null: dict) -> dict:
    """R-2's gate: W narrower than the null box -- the median paired log10 width ratio below 0."""
    p = paired_log10_ratio(w, null)
    return {**p, "passes": p["median_dex"] is not None and p["median_dex"] < 0.0}


#: chi-square 95 % points for R-3's report: 2(ll-hat - ll(truth)) with W's 4 free parameters (3 when kappa is flat)
CHI2_95 = {3: 7.814727903251178, 4: 9.487729036781154}


def r3(rows: list, max_gap: float) -> dict:
    """R-3: the cohorts whose truth's log-likelihood exceeds theta-hat's by more than max_gap (a search failure); a row
    with no truth log-likelihood (its objective was not captured) is a failure by name, never skipped (sr-naveh F9).
    Reported beside it: 2(ll-hat - ll(truth)) against chi-square (Wilks), where a wrong conditioning term would show
    before it shows in coverage."""
    fails, missing, w = [], [], []
    for r in rows:
        if r.get("loglik_truth") is None or r.get("loglik_hat") is None:
            missing.append(r["cohort_id"])
        elif r["loglik_truth"] - r["loglik_hat"] > max_gap:
            fails.append(r["cohort_id"])
        else:
            w.append(2.0 * (r["loglik_hat"] - r["loglik_truth"]))
    wilks = {"n": len(w), "mean": statistics.fmean(w) if w else None, "max": max(w) if w else None,
             "above_chi2_95": {k: sum(1 for x in w if x > v) for k, v in CHI2_95.items()}}
    return {"n": len(rows), "failures": fails + missing, "missing_truth": missing, "passes": not (fails or missing),
            "wilks": wilks}


def r4(rows: list, tol: float) -> dict:
    """R-4, exact (Naveh, 2026-11-11): on every cohort, H's log-likelihood at (theta-hat_W, h of H's lowest-h stored
    profile node) -- the element's `loglik_H_at_theta_w` -- is at most tol below that node's stored log-likelihood. W's
    model is H's at h = 0, so more than tol below means W's search missed what H found. A row without the evaluation
    fails by name. A failure stops W, as R-3's does."""
    fails, missing, gaps = [], [], {}
    for r in rows:
        node, got = (r.get("h_node") or {}).get("loglik"), r.get("loglik_H_at_theta_w")
        if node is None or got is None:
            missing.append(r["cohort_id"])
            continue
        gaps[r["cohort_id"]] = node - got
        if node - got > tol:
            fails.append(r["cohort_id"])
    return {"n": len(rows), "failures": fails + missing, "not_evaluated": missing, "passes": not (fails or missing),
            "max_gap": max(gaps.values()) if gaps else None, "gaps": gaps}


# ------------------------------------------ the table of 2026-11-11 (spec.md Clarifications; sr-naveh_cuts_006.md)
TRUNCATED_DEX = 0.3            # W's upper end this far below H's (both closed) is truncated above (sr-naveh F7)
ONE_SIDED_MIN = 20             # the collapse gate on one-sided claims applies from this many claims; below, reported


def collapse(rule: dict) -> dict:
    """R-1's formula against a collapse: p_bad = rule["collapse_p_bad"] (0.70). For A2, W's data-determined one-sided
    claims and the lead pair on Dg's cohorts, where too few claims separate 0.95 from 0.88 (sr-naveh F1)."""
    return {**rule, "p_bad": rule["collapse_p_bad"]}


def upper_open(claim: dict) -> str | None:
    """Why a W claim's upper end is not the data's: "box_limited" (set at the box: claims/shapes.py, always the upper
    side) or "widened" ("edge not found"), else None. label_info keeps only the first kind, so the text is read too."""
    from acceptance_003_coverage import label_kinds
    kinds = label_kinds(claim)
    return "box_limited" if "box_limited" in kinds else "widened" if "widened" in kinds else None


def claim_of(row: dict, q: str) -> dict | None:
    return next((c for c in row.get("claims") or [] if c["parameter"] == q), None)


def truncated_above(row: dict, q: str, h_bounds: dict) -> str | None:
    """Why W's claim on q is truncated above, else None (sr-naveh F7): box-limited or widened above; more than
    TRUNCATED_DEX below H's upper end, both closed; or contradicted above by the lead pair's own walk."""
    c = claim_of(row, q)
    if c is None or c.get("upper") is None:
        return None
    why = upper_open(c)
    if why:
        return why
    hb = h_bounds.get(row["cohort_id"])
    if c["shape"] == "closed" and hb and math.log10(hb[1] / c["upper"]) > TRUNCATED_DEX:
        return f"{math.log10(hb[1] / c['upper']):.2f} dex below H's upper end"
    for k in (row.get("combinations") or {}).get("param_claims_contradicted") or []:
        if k["parameter"] == q and k["side"] == "upper":
            return f"contradicted above by the lead pair's walk ({k['dex']:.2f} dex)"
    return None


def truncation(rows: list, h_bounds_by_q: dict, params=GATED) -> dict:
    """Per parameter: W's closed claims, those truncated above (each named with its reason) and their rate."""
    out = {}
    for q in params:
        closed = [r for r in rows if (claim_of(r, q) or {}).get("shape") == "closed"]
        why = {r["cohort_id"]: truncated_above(r, q, h_bounds_by_q.get(q, {})) for r in closed}
        named = {k: v for k, v in why.items() if v}
        out[q] = {"closed": len(closed), "truncated": len(named), "rate": len(named) / len(closed) if closed else None,
                  "named": named}
    return out


def w_closed_bounds(rows: list, q: str, h_bounds: dict) -> dict:
    """W's closed (lower, upper) on q per cohort, a truncated-above claim counted as open (None): R-2's widths."""
    out = {}
    for r in rows:
        c = claim_of(r, q)
        ok = (c is not None and c["shape"] == "closed" and c.get("lower") and c["lower"] > 0
              and truncated_above(r, q, h_bounds) is None)
        out[r["cohort_id"]] = (c["lower"], c["upper"]) if ok else None
    return out


def r2_vs_h(w: dict, h: dict, max_dex: float) -> dict:
    """R-2 (Naveh, 2026-11-11): the paired median of log10(width_W / width_H) over cohorts closed in both is at most
    max_dex (W has one nuisance fewer than H, so W wider than H by more than that is information lost)."""
    p = paired_log10_ratio(w, h)
    return {**p, "max_dex": max_dex, "passes": p["median_dex"] is not None and p["median_dex"] <= max_dex}


def data_determined_one_sided(claim: dict) -> bool:
    """A one-sided claim whose finite end the data set: a lower bound (lower ends are never box-limited), or an upper
    bound neither box-limited nor widened."""
    if claim is None or claim["shape"] not in ("lower_bound", "upper_bound"):
        return False
    return claim["shape"] == "lower_bound" or upper_open(claim) is None


def row_hits(rows: list, truths: dict, q: str, keep) -> tuple[int, int]:
    """(hits, claimed) over the rows' claims on q that `keep(claim)` selects; truths per cohort, combinations by
    sraging.claims.combinations (acceptance_003_coverage.hit / truth_value: spec 003's scoring of the same dicts)."""
    from acceptance_003_coverage import hit, truth_value
    hits = claimed = 0
    for r in rows:
        c = claim_of(r, q)
        if c is None or not keep(c):
            continue
        h = hit(c, truth_value(q, truths[r["cohort_id"]]))
        if h is None:
            continue
        claimed += 1
        hits += bool(h)
    return hits, claimed


def one_sided_gate(rows: list, truths: dict, rule: dict, params=GATED) -> dict:
    """W's data-determined one-sided coverage per parameter under the collapse rule (Naveh, 2026-11-11), gated from
    ONE_SIDED_MIN claims; below that, reported and not gated."""
    out = {}
    for q in params:
        hits, n = row_hits(rows, truths, q, data_determined_one_sided)
        v = r1_verdict(hits, n, collapse(rule)) if n >= ONE_SIDED_MIN else {
            "hits": hits, "claimed": n, "coverage": hits / n if n else None, "cut": None, "passes": None,
            "note": f"fewer than {ONE_SIDED_MIN} claims: reported, not gated"}
        out[q] = v
    return {"by_param": out, "passes": all(v["passes"] is not False for v in out.values())}


def shape_report(rows: list, truths: dict, params=GATED + REPORTED) -> dict:
    """Closed and one-sided coverage with their denominators, closure rates, and the rate at which the data cap each
    parameter from above (closed, and its upper end neither box-limited nor widened); one-sided caps split into
    box-limited and data-determined (sr-naveh F2)."""
    out = {}
    n = len(rows)
    for q in params:
        cs = [claim_of(r, q) for r in rows]
        closed = [c for c in cs if c and c["shape"] == "closed"]
        out[q] = {"n": n, "closed": len(closed), "closure_rate": len(closed) / n if n else None,
                  "capped_by_data": sum(1 for c in closed if upper_open(c) is None),
                  "one_sided": sum(1 for c in cs if c and c["shape"] in ("lower_bound", "upper_bound")),
                  "one_sided_data_determined": sum(1 for c in cs if data_determined_one_sided(c)),
                  "coverage_closed": row_hits(rows, truths, q, lambda c: c["shape"] == "closed"),
                  "coverage_one_sided": row_hits(rows, truths, q, lambda c: c["shape"] in ("lower_bound",
                                                                                          "upper_bound")),
                  "coverage_one_sided_data_determined": row_hits(rows, truths, q, data_determined_one_sided)}
    return out


def misses(rows: list, truths: dict, clusters: dict, params=GATED) -> dict:
    """Closed-claim misses per parameter by side (truth below the lower end, above the upper end), each with the
    truth's margin in dex, and the clusters with 2 or more misses (sr-naveh F4, F5)."""
    out = {}
    for q in params:
        below, above, per = {}, {}, {}
        for r in rows:
            c = claim_of(r, q)
            if not c or c["shape"] != "closed" or not c.get("lower") or not c.get("upper"):
                continue
            t = truths[r["cohort_id"]][q]
            if t < c["lower"]:
                below[r["cohort_id"]] = math.log10(c["lower"] / t)
            elif t > c["upper"]:
                above[r["cohort_id"]] = math.log10(t / c["upper"])
            else:
                continue
            k = clusters.get(r["cohort_id"])
            per[k] = per.get(k, 0) + 1
        out[q] = {"below": below, "above": above, "clusters_with_2_or_more": sorted(k for k, v in per.items() if v >= 2)}
    return out


def fr022(rows: list, max_rate: float) -> dict:
    """FR-022 (Naveh, 2026-11-11): W's leverage flags (the label `flag` only) among the assessed fits (flag, no_flag,
    flat_valley_slide) at most max_rate needs no new acceptance; each flag is named with the dropped row's age against
    the window; not_assessed, failed and slides are counted."""
    labels = [str(r.get("leverage_flag")) for r in rows]
    assessed = sum(1 for x in labels if x in ("flag", "no_flag", "flat_valley_slide"))
    flags = {r["cohort_id"]: {"dropped": (r.get("leverage") or {}).get("dropped"),
                              "window": ((r.get("window") or {}).get("window_eff"))}
             for r in rows if r.get("leverage_flag") == "flag"}
    allowed = math.floor(max_rate * assessed + 1e-9)
    return {"leverage_flagged": len(flags), "assessed": assessed, "allowed": allowed, "max_rate": max_rate,
            "passes": len(flags) <= allowed, "flags": flags,
            "leverage_labels": {lab: labels.count(lab) for lab in sorted(set(labels))}, "n": len(rows)}


LEAD_PAIR = ("eta_over_eps", "rbar_over_eta")


def lead_pair_report(rows: list, truths: dict, rule: dict) -> dict:
    """Dg (Naveh, 2026-11-11): on the rows that ran the lead pair, its closed coverage under the collapse rule, its
    one-sided coverage reported, the count of claim ends its walk contradicted, and the `negligible` rate where the
    label can fire (FR-022: the label needs the combination claims)."""
    lead = [r for r in rows if (r.get("combinations") or {}).get("on")]
    out = {"n": len(lead), "by_quantity": {}}
    for q in LEAD_PAIR:
        hits, n = row_hits(lead, truths, q, lambda c: c["shape"] == "closed")
        out["by_quantity"][q] = {**r1_verdict(hits, n, collapse(rule)),
                                 "one_sided": row_hits(lead, truths, q, lambda c: c["shape"] in ("lower_bound",
                                                                                                 "upper_bound"))}
    out["contradicted_ends"] = {r["cohort_id"]: (r["combinations"].get("param_claims_contradicted") or [])
                                + (r["combinations"].get("combination_claims_contradicted") or [])
                                for r in lead if (r["combinations"].get("param_claims_contradicted")
                                                  or r["combinations"].get("combination_claims_contradicted"))}
    out["negligible"] = sum(1 for r in lead if r.get("negligible_labels"))
    out["passes"] = bool(lead) and all(v["passes"] for v in out["by_quantity"].values())
    return out


# ------------------------------------------------------------------------------------- the stored records (FR-018)
def stored_summary(counts: dict, bounds: dict, clusters: dict, rules: dict) -> dict:
    """R-1 and R-2 on stored records at 0 fits (FR-018): `counts` {run label: {param: (hits, claimed)}}, `bounds`
    {label: {param: {cohort: (lo, hi) | None}}}, `clusters` {cohort: cluster}. Every label gets R-1's table (with the
    clusters its closed claims span) and R-2 against the null; J against H is paired and reported. The rules
    discriminate as designed when J's window records fail R-1 and H's pass it."""
    r1, r2 = rules["R-1"], {}
    out = {}
    for label, c in counts.items():
        tab = r1_table(c, r1)
        for q, v in {**tab["gated"], **tab["reported"]}.items():
            v["n_clusters"] = len({clusters[k] for k, b in bounds[label].get(q, {}).items() if b})
        out[label] = {"R-1": tab}
    for label in counts:
        if label == "null":
            continue
        r2[label] = {q: r2_gate(bounds[label][q], bounds["null"][q]) for q in r1["gated"]}
        out[label]["R-2_vs_null"] = r2[label]
    if "J" in counts and "H" in counts:
        out["J_vs_H"] = {q: paired_log10_ratio(bounds["J"][q], bounds["H"][q]) for q in r1["gated"]}
    disc = (not out["J"]["R-1"]["passes"]) and out["H"]["R-1"]["passes"]
    return {"runs": out, "discriminates": disc,
            "expected": "J's window records fail R-1 and H's pass it (spec.md US2 scenario 1)"}


def design_effect(bounds: dict, truths: dict, clusters: dict, q: str) -> dict:
    """The design effect of closed-claim coverage over clusters, by the ratio estimator: the cluster-robust variance
    of the hit rate over its binomial variance (research R12's measurement; sr-naveh F4 measured 3.35 and 3.12 on J's
    window records, where misses cluster)."""
    y, m, per = {}, {}, {}
    for c, b in bounds.items():
        if not b:
            continue
        k = clusters[c]
        m[k] = m.get(k, 0) + 1
        miss = not (b[0] <= truths[c][q] <= b[1])
        y[k] = y.get(k, 0) + (not miss)
        per[k] = per.get(k, 0) + miss
    M, K = sum(m.values()), len(m)
    if K < 2 or not M:
        return {"coverage": None, "claims": M, "clusters": K, "D": None, "clusters_with_2_or_more_misses": []}
    p = sum(y.values()) / M
    vc = K / (K - 1) * sum((y[k] - p * m[k]) ** 2 for k in m) / M ** 2
    vs = p * (1 - p) / M
    return {"coverage": p, "claims": M, "clusters": K, "D": vc / vs if vs > 0 else None,
            "clusters_with_2_or_more_misses": sorted(k for k, v in per.items() if v >= 2)}


def shrunk_hits(bounds: dict, truths: dict, q: str, s: float) -> tuple[int, int]:
    """(hits, claimed) of the closed claims shrunk about their log-midpoints by the factor s (s = 1: as stored)."""
    hits = n = 0
    for c, b in bounds.items():
        if not b:
            continue
        mid, half = 0.5 * (math.log(b[0]) + math.log(b[1])), 0.5 * (math.log(b[1]) - math.log(b[0]))
        n += 1
        hits += math.exp(mid - s * half) <= truths[c][q] <= math.exp(mid + s * half)
    return hits, n


def planted_shrink(bounds: dict, truths: dict, q: str, rule: dict, target: float = 0.88) -> dict:
    """sr-naveh F3's planted defect at 0 fits: H's stored closed claims shrunk in log space, by the largest factor whose
    coverage is at most `target` (bisection; coverage only rises with the factor). R-1 must fail it at its N."""
    lo, hi = 0.0, 1.0
    if shrunk_hits(bounds, truths, q, 1.0)[0] <= target * shrunk_hits(bounds, truths, q, 1.0)[1]:
        lo = 1.0
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            h, n = shrunk_hits(bounds, truths, q, mid)
            lo, hi = (mid, hi) if h <= target * n else (lo, mid)
    h, n = shrunk_hits(bounds, truths, q, lo)
    return {"factor": lo, **r1_verdict(h, n, rule)}


# --------------------------------------------------------------------------------------- the element (T027, T028)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_BUILD = os.path.join(REPO, "tests", "oracle", "tester_f94ec2b", "srbench", "library", "build.py")
W = "profile_jax2_w"
ROW_FIELDS = ("cohort_id", "cell", "cluster", "n", "kappa_true", "status", "claims", "theta_hat", "loglik_hat",
              "loglik_truth", "window", "leverage_flag", "leverage", "negligible_labels", "warnings", "combinations",
              "h_node", "loglik_H_at_theta_w", "fit_cpu_s", "element_cpu_s", "host", "cpu_model", "lsf_jobid",
              "lsf_index", "head")
LEVERAGE_KEEP = ("label", "reason", "dropped", "R", "delta_total", "delta_nats", "cost")


def tester_cohort_digest():
    """The tester's own `cohort_digest`, compiled verbatim from the snapshot's build.py (no srbench import)."""
    import numpy as np
    tree = ast.parse(open(SNAPSHOT_BUILD).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cohort_digest")
    ns = {"hashlib": __import__("hashlib"), "np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SNAPSHOT_BUILD, "exec"), ns)
    return ns["cohort_digest"]


def load_checked(cid: str, entry: dict, v6_dir: str):
    """(ages, events) of one v6 cohort from its .npz, refused unless the tester's digest equals the manifest's."""
    import numpy as np
    with np.load(os.path.join(v6_dir, "cohorts", f"{cid}.npz")) as z:
        ages, events = np.asarray(z["ages"], dtype=float), np.asarray(z["events"], dtype=int)
    got = tester_cohort_digest()(ages, events)
    if got != entry["digest"]:
        raise ValueError(f"{cid}: the arrays' digest {got[:12]} is not the manifest's {entry['digest'][:12]}")
    return ages, events


def cpu_self_and_children() -> float:
    import resource
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def cpu_model() -> str | None:
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or None


def h_loglik_at(cohort, theta_hat: dict, h: float) -> float:
    """R-4, exact (Naveh, 2026-11-11): H's log-likelihood (profile_jax2_h 1.0.1) at (theta-hat_W, h), on the prepared
    data and grid H's own fit of `cohort` uses (its window path and fit_inputs), by the value march (leverage._loglik:
    the leverage check's evaluation of the same objective)."""
    import jax.numpy as jnp
    from sraging import leverage as LEV
    from sraging.calibrators import profile_jax2 as PJ
    port = PJ.PORTS["1.0.1"]
    wc, _rec, _rows = PJ.window_path(cohort, port.classes["profile_jax2_h"], port)
    f = LEV._loglik(PJ.fit_inputs(wc, "profile_jax2_h", port), port=port)[0]
    return float(f(jnp.asarray([math.log(theta_hat[p]) for p in PJ.PARAMS] + [math.log(h)])))


def element_row(cid: str, entry: dict, *, cell: str, head: str, v6_dir: str, lead_pair: bool,
                calibrator: str = W) -> dict:
    """One gate row (data-model.md "Gate rows"): W's fit of the v6 cohort as a user gives it (its raw arrays and its
    window; combinations off, or the lead pair where the panel marks the cohort, which does not change the four
    claims: T004), every claim, the truth witness on the fit's own main objective, the leverage block, the negligible
    labels, R-4's evaluation of H at (theta-hat, the h of H's lowest node, from the panel) and the CPU. A failure is a
    row with its error, never a silent gap."""
    import socket
    import traceback
    import numpy as np
    row = {"cohort_id": cid, "cell": cell, "cluster": entry["cluster"], "n": entry["n"], "kappa_true": entry["kappa"],
           "head": head, "host": socket.gethostname(), "cpu_model": cpu_model(),
           "lsf_jobid": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX")}
    try:
        import sraging
        from sraging import Cohort
        from sraging.calibrators import profile_jax2 as PJ
        ages, events = load_checked(cid, entry, v6_dir)
        if calibrator == "profile_v033":            # spec 006 R-9: v033's own leverage check on an unwindowed cohort
            r = sraging.fit_cohort(Cohort.from_arrays(ages, events, delta=entry["delta"], name=cid),
                                   calibrator="profile_v033")
            lev = (r.diagnostics or {}).get("leverage") or {}
            row.update({"status": r.status, "calibrator": calibrator, "loglik_hat": r.loglik,
                        "leverage_flag": lev.get("label"), "leverage_reason": lev.get("reason"),
                        "dropped": lev.get("dropped"), "leverage_cpu_s": (lev.get("cost") or {}).get("cpu_s"),
                        "warnings": [w.code for w in r.warnings], "fit_cpu_s": (r.timing or {}).get("cpu_s")})
            row["element_cpu_s"] = cpu_self_and_children()
            return row
        cohort = Cohort.from_arrays(ages, events, delta=entry["delta"], window=tuple(entry["window"]), name=cid)
        lead_pair = bool(entry.get("lead_pair", lead_pair))
        with PJ.capture_objectives(PJ.PORTS["1.0.1"]) as objs:      # W's port shares 1.0.1's modules
            r = sraging.fit_cohort(cohort, calibrator=W, combinations="lead" if lead_pair else False)
        obj = next((o for o in objs if o.NP == len(PJ.PARAMS)), None)   # the fit's main objective, created first
        truth = [float(entry["truth"][p]) for p in PJ.PARAMS]
        row.update({
            "status": r.status, "claims": [c.to_dict() for c in r.claims],
            "theta_hat": None if r.theta_hat is None else r.theta_hat.as_dict(), "loglik_hat": r.loglik,
            "loglik_truth": None if obj is None else -float(obj.nll(np.log(truth))),
            "window": (r.diagnostics or {}).get("window"),
            "leverage_flag": ((r.diagnostics or {}).get("leverage") or {}).get("label"),
            "leverage": {k: v for k, v in ((r.diagnostics or {}).get("leverage") or {}).items() if k in LEVERAGE_KEEP},
            "negligible_labels": [c.parameter for c in r.claims if (c.label_info or {}).get("kind") == "negligible"],
            "warnings": [w.code for w in r.warnings],
            "combinations": {k: v for k, v in (r.combinations or {}).items() if k != "walks"},
            "fit_cpu_s": (r.timing or {}).get("cpu_s")})
        row.update({"h_node": entry.get("h_node"), "loglik_H_at_theta_w": None})
        if entry.get("h_node") and r.theta_hat is not None:
            try:
                row["loglik_H_at_theta_w"] = h_loglik_at(cohort, r.theta_hat.as_dict(), entry["h_node"]["value"])
            except Exception as ex:  # noqa: BLE001 -- R-4 then fails by name, the fit is kept
                row.update({"loglik_H_at_theta_w": None, "h_eval_error": f"{type(ex).__name__}: {ex}"})
    except Exception as ex:  # noqa: BLE001 -- a failed fit is a row
        row.update({"status": "error", "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-2000:]})
    row["element_cpu_s"] = cpu_self_and_children()
    return row


def write_atomic(path: str, payload: dict):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, allow_nan=True)
        fh.write("\n")
    os.replace(tmp, path)


def element_main(argv=None):
    """On a WEXAC node: `gate_006.py element --panel PANEL.json --wave WAVE.txt --index I --v6 DIR --out DIR`. The
    I-th cohort of the wave file is fitted and its row written to OUT/<cohort>.json, unless it is there already (the
    same head: a resubmitted wave skips what finished)."""
    import argparse
    ap = argparse.ArgumentParser(prog="gate_006.py element")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--wave", required=True)
    ap.add_argument("--index", type=int, default=None, help="1-based (default: $LSB_JOBINDEX)")
    ap.add_argument("--v6", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    panel = json.load(open(a.panel))
    ids = [ln.strip() for ln in open(a.wave) if ln.strip()]
    cid = ids[(a.index if a.index is not None else int(os.environ["LSB_JOBINDEX"])) - 1]
    path = os.path.join(a.out, f"{cid}.json")
    if os.path.exists(path):
        print(f"{cid}: already written; skipped")
        return
    os.makedirs(a.out, exist_ok=True)
    row = element_row(cid, panel["cohorts"][cid], cell=panel["cell"], head=panel["head"], v6_dir=a.v6,
                      lead_pair=bool(panel.get("lead_pair")), calibrator=panel.get("calibrator", W))
    write_atomic(path, row)
    print(f"{cid}: {row.get('status')} in {row['element_cpu_s']:.0f} CPU-s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "element":
        sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "tools")]
        element_main(sys.argv[2:])
    else:
        sys.exit("usage: gate_006.py element --panel P --wave W [--index I] --v6 DIR --out DIR")
