"""The external hazard h, REPORTED as a nuisance -- read from the would-be claim the method stores.

`mle_profile_jax2_h` (H) fits a constant external hazard h beside the four SR parameters but never
CLAIMS it: the tester scores the four, and h is a nuisance. Every H record nevertheless carries the
claim H would make, `diagnostics.hazard.h_claim` -- its final claim, widened by the same
self-consistency step as the four -- "so h can be scored later without a re-fit". Naveh,
2026-10-28: report it, zero refits, unpaired, NOT in the verdict.

What this is, and what it is not:
  * the figures are about H's WOULD-BE claim, not a claim H returned; the paired verdict's
    `external_hazard` row stays "not measured", and nothing here compares two methods;
  * every figure is per group and never pooled. Heterogeneity is kept apart and labelled -- H has no
    heterogeneity term, so a hazard "found" on a heterogeneous cohort is a statement about H's
    model, not about what survival curves contain -- and `realfull` is kept apart from the hazard arm;
  * declines (shape `none`) are counted, never scored;
  * the box null is H's OWN per-cohort search box, [1e-6/L, 10/L] with L the cohort's Kaplan-Meier
    mean -- data-informed, not data-ignoring. It covers the truth essentially always, so what H earns
    on h is WIDTH, not coverage; the width comparison is the tester's addition, not part of HB8.

Definitions are jax2's registered ones (`sr_calibration_tester_jax2/phaseC/h_estimate.py`, by sha256
in PREREG_C_ADD1), with jax2's port notes of 2026-10-29:
  HB4  false detection on an h = 0 cohort: the claim is closed or a lower bound AND its lower edge
       exceeds 10 x h_lo, with h_lo = `diagnostics.hazard.box[0]` -- "the claim rules out h ~ 0";
  HB8  on h-planted cohorts: closed and one-sided coverage, availability, and |log10(h_hat/h)| with
       h_hat = `diagnostics.hazard.h`, the UNCLIPPED MLE (the claim point is clipped to the box);
       h_true = exp(-ExtH) in the cohort's own age unit -- the library already converted ExtH, so no
       365.25 is ever applied here.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import replace

from ..contract.types import Claim

SOURCE = "diagnostics.hazard.h_claim"
QUANTITY = "external_hazard"

G_HAZARD = "hazard arm (h planted)"
#: Heterogeneous groups name their arm: realfull-type cohorts can carry heterogeneity too.
G_PLANTED_HET = "{arm} + heterogeneity, h planted (misspecified: H has no heterogeneity term)"
G_ZERO_HET = "{arm}, heterogeneous, no h (misspecified: H has no heterogeneity term)"

#: HB4's bar: a lower edge above this many times the per-cohort h floor. With the floor at hL = 1e-6
#: that excludes only hL < 1e-5 -- three decades below any h that matters -- so the report also counts
#: detections whose lower edge clears NEGLIGIBLE_HL (sr-naveh review, 2026-10-29).
HB4_FLOOR_MULTIPLE = 10.0
#: hL below which an external hazard is physically negligible: about 1% of a lifespan's deaths.
NEGLIGIBLE_HL = 0.01
#: "Can the data cap h?": the share of h = 0 cohorts whose claim's upper edge lies at or below these hL.
CAP_HL = (0.03, 0.1, 0.3)
#: True hL bands for the descriptive strata (hL = h x the cohort's KM mean lifespan).
HL_EDGES = (0.1, 0.3, 1.0)
#: Sample-size bands: every real anchor has its own n, so per-n rows would be two cohorts each.
N_EDGES = (1000, 10000, 50000)
SHAPES = ("closed", "upper_bound", "lower_bound", "none")


def _hz(cs) -> dict | None:
    hz = (getattr(cs, "diagnostics", None) or {}).get("hazard")
    return hz if isinstance(hz, dict) and isinstance(hz.get("h_claim"), dict) else None


def hazard_claim(cs) -> Claim | None:
    """H's would-be claim on h as a `Claim` on `external_hazard`, or None if the record has none."""
    hz = _hz(cs)
    if hz is None:
        return None
    return Claim.from_dict(dict(hz["h_claim"], parameter=QUANTITY))


def with_hazard_claims(claim_sets) -> list:
    """COPIES of the claim sets with the would-be h claim appended where a record carries one and no
    real `external_hazard` claim exists. The stored objects are never mutated."""
    out = []
    for cs in claim_sets:
        c = hazard_claim(cs) if cs.claim_for(QUANTITY) is None else None
        out.append(replace(cs, claims=tuple(cs.claims) + (c,)) if c is not None else replace(cs))
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Wilson score 95% interval for k of n. None when n is 0."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def false_detection(claim: Claim | None, box) -> bool:
    """HB4: closed or lower_bound, with a lower edge above 10 x the cohort's h floor."""
    if claim is None or claim.shape not in ("closed", "lower_bound") or claim.lower is None:
        return False
    return claim.lower > HB4_FLOOR_MULTIPLE * float(box[0])


def group_of(spec, planted: bool) -> str:
    het = (getattr(spec, "het_var", 0.0) or 0.0) > 0
    if planted:
        if het:
            return G_PLANTED_HET.format(arm=spec.arm)
        return G_HAZARD if spec.arm == "hazard" else f"{spec.arm} (h planted)"
    return G_ZERO_HET.format(arm=spec.arm) if het else f"{spec.arm} (no h)"


def _L(hz):
    """The method's L: the Kaplan-Meier mean of the OBSERVED cohort (external deaths included; on a
    windowed cohort, from left-truncated data) -- not the SR lifespan."""
    if hz.get("L"):
        return float(hz["L"])
    h = hz.get("h")
    return (hz["hL"] / h) if (h and hz.get("hL") is not None) else None


def _at_floor(hz) -> bool:
    if hz.get("at_floor") in (True, "true"):
        return True
    bh, hl = hz.get("box_hL"), hz.get("hL")
    return bool(bh and hl is not None and hl <= float(bh[0]) * (1 + 1e-6))


def _at_start(hz) -> bool:
    """The MLE is exactly a ladder START value (e.g. 'ladder:hL=0.01'): where the profile in h is flat,
    the optimiser never left it."""
    w = str(((hz.get("mle") or {}).get("winner")) or "")
    hl = hz.get("hL")
    if not w.startswith("ladder:hL=") or not hl:
        return False
    try:
        x = float(w.split("=", 1)[1])
    except ValueError:
        return False
    return x > 0 and abs(math.log10(hl / x)) < 1e-6


def _p90(v):
    """HB8's p90: numpy's default (linear) quantile, without importing numpy."""
    s = sorted(v)
    if not s:
        return None
    pos = 0.9 * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _banded(x, edges, fmt="g"):
    """(order, label) of `x` in `edges`: '< e0', 'e0-e1', ..., '>= eN'."""
    for i, e in enumerate(edges):
        if x < e:
            return i, (f"< {e:{fmt}}" if i == 0 else f"{edges[i - 1]:{fmt}}-{e:{fmt}}")
    return len(edges), f">= {edges[-1]:{fmt}}"


def _planted_row(group, recs):
    shapes = Counter(r["claim"].shape for r in recs)
    closed = [r for r in recs if r["claim"].shape == "closed"]
    one = [r for r in recs if r["claim"].shape in ("upper_bound", "lower_bound")]
    k_closed = sum(1 for r in closed if r["claim"].contains(r["t"]))
    k_one = sum(1 for r in one if r["claim"].contains(r["t"]))
    # HB8's error: every cohort's MLE, declines included (registered). Where the claim is one-sided the
    # profile in h is flat and the MLE is wherever the optimiser stopped -- the box floor or a start
    # value -- so the error on CLOSED claims is reported beside it, with the parked counts.
    errs = [math.log10(r["h"] / r["t"]) for r in recs if r["h"] and r["h"] > 0]
    errs_cl = [abs(math.log10(r["h"] / r["t"])) for r in closed if r["h"] and r["h"] > 0]
    widths = [r["claim"].width_dex() for r in closed if r["claim"].width_dex() is not None]
    boxw = [math.log10(r["box"][1] / r["box"][0]) for r in recs if r["box"][0] > 0]
    missed = {r["spec"].cluster_key for r in closed if not r["claim"].contains(r["t"])}
    return {
        "group": group, "n": len(recs),
        "clusters": len({r["spec"].cluster_key for r in recs}),
        "closed_clusters": len({r["spec"].cluster_key for r in closed}),
        "clusters_with_a_closed_miss": len(missed),
        "shapes": {s: shapes.get(s, 0) for s in SHAPES},
        "declined": shapes.get("none", 0),
        "closed": {"hits": k_closed, "n": len(closed), "wilson": wilson(k_closed, len(closed))},
        "one_sided": {"hits": k_one, "n": len(one)},
        "availability_closed": (len(closed) / len(recs)) if recs else None,
        "accuracy": {"n": len(errs),
                     "median": statistics.median([abs(e) for e in errs]) if errs else None,
                     "p90": _p90([abs(e) for e in errs]),
                     "signed_median": statistics.median(errs) if errs else None},
        "accuracy_closed": {"n": len(errs_cl),
                            "median": statistics.median(errs_cl) if errs_cl else None,
                            "p90": _p90(errs_cl)},
        "mle_at_floor": sum(1 for r in recs if r["at_floor"]),
        "mle_at_start": sum(1 for r in recs if r["at_start"]),
        # A start value on a CLOSED claim is a different statement: the profile is not flat there,
        # the optimiser simply did not refine h.
        "mle_at_start_closed": sum(1 for r in closed if r["at_start"]),
        "width": {"claim_median_dex": statistics.median(widths) if widths else None,
                  "n_closed": len(widths),
                  "box_median_dex": statistics.median(boxw) if boxw else None},
        "box_null": {"hits": sum(1 for r in recs if r["box"][0] <= r["t"] <= r["box"][1]),
                     "n": len(recs)},
    }


def _zero_row(group, recs):
    shapes = Counter(r["claim"].shape for r in recs)
    det = [r for r in recs if false_detection(r["claim"], r["box"])]
    k_neg = sum(1 for r in det if r["L"] and r["claim"].lower * r["L"] > NEGLIGIBLE_HL)
    # Can the data CAP h? Every cohort whose claim has an upper edge (closed or upper bound), in hL.
    up = [r["claim"].upper * r["L"] for r in recs
          if r["claim"].shape in ("closed", "upper_bound") and r["claim"].upper is not None and r["L"]]
    return {"group": group, "n": len(recs),
            "clusters": len({r["spec"].cluster_key for r in recs}),
            "shapes": {s: shapes.get(s, 0) for s in SHAPES},
            "false_detection": {"k": len(det), "n": len(recs), "wilson": wilson(len(det), len(recs))},
            "detections_above_negligible": k_neg,
            "cap": [{"hL": c, "k": sum(1 for u in up if u <= c), "n": len(recs)} for c in CAP_HL],
            "upper_hL_median": statistics.median(up) if up else None}


def hazard_report(method_sets, specs: dict, truths: dict, not_in_run: dict | None = None,
                  context: dict | None = None) -> dict | None:
    """The nuisance-h section as DATA, or None when no record of the run carries a would-be h claim.

    `specs` and `truths` are keyed by cohort id; `not_in_run` maps a library group to the number of
    h-planted cohorts of the library this run does not contain; `context` names the method and
    library for the limitations.
    """
    recs = {"planted": {}, "zero": {}}
    failed = Counter()
    any_h = False
    box_hl = set()
    for cs in method_sets:
        sp = specs.get(cs.cohort_id)
        if sp is None:
            continue
        t = (truths.get(cs.cohort_id) or {}).get(QUANTITY)
        planted = t is not None
        g = group_of(sp, planted)
        if not cs.ok:
            failed[g] += 1
            continue
        hz = _hz(cs)
        if hz is None:
            continue
        any_h = True
        rec = {"cid": cs.cohort_id, "claim": hazard_claim(cs), "t": t, "h": hz.get("h"),
               "L": _L(hz), "at_floor": _at_floor(hz), "at_start": _at_start(hz),
               "box": tuple(float(b) for b in hz.get("box") or (0.0, math.inf)), "spec": sp}
        if hz.get("box_hL"):
            box_hl.add(tuple(float(f"{float(b):.9g}") for b in hz["box_hL"]))   # ulps differ
        recs["planted" if planted else "zero"].setdefault(g, []).append(rec)
    if not any_h:
        return None

    def order(d, first):
        return sorted(d, key=lambda g: (g != first, g))

    planted = [dict(_planted_row(g, recs["planted"][g]), failed=failed.get(g, 0))
               for g in order(recs["planted"], G_HAZARD)]
    zero = [dict(_zero_row(g, recs["zero"][g]), failed=failed.get(g, 0))
            for g in order(recs["zero"], "pure (no h)")]

    from ..library.order import MIN_CELLS_PER_STRATUM
    strata = {"by_n": [], "by_hL": []}
    hz_recs = recs["planted"].get(G_HAZARD, [])
    def by_n(r):
        o, lab = _banded(r["spec"].n, N_EDGES, "d")
        return o, f"n {lab}"

    def by_hl(r):
        if not r["L"]:
            return 99, "true hL unknown"
        o, lab = _banded(r["t"] * r["L"], HL_EDGES)
        return o, f"true hL {lab}"

    for key, label_of in (("by_n", by_n), ("by_hL", by_hl)):
        cells, rank = {}, {}
        for r in hz_recs:
            o, lab = label_of(r)
            cells.setdefault(lab, []).append(r)
            rank[lab] = o
        for lab in sorted(cells, key=rank.get):
            cl = [r for r in cells[lab] if r["claim"].shape == "closed"]
            strata[key].append({"label": lab, "n": len(cells[lab]),
                                "closed_hits": sum(1 for r in cl if r["claim"].contains(r["t"])),
                                "closed_n": len(cl),
                                "thin": len(cells[lab]) < MIN_CELLS_PER_STRATUM})

    ctx = context or {}
    lim = [f"Only {ctx.get('method', 'this method')} v{ctx.get('method_version', '?')} on library "
           f"{ctx.get('library_version', '?')}, and only the cohorts in this run."]
    if not_in_run:
        lim.append("h-planted cohorts of the library NOT in this run: " +
                   ", ".join(f"{n} in group `{g}`" for g, n in sorted(not_in_run.items())) +
                   ". Nothing here describes them; the sharp hazard cohorts are where H's "
                   "likelihood is known to be off at the truth ([T-H-HAZARD-SHARP]).")
    lim.append("Nothing here says how h absorbing heterogeneity moves H's four SR claims.")
    lim.append("On the heterogeneity rows the truth is the generator's h, not the value H's misspecified "
               "model would converge to; on `realfull (h planted)` H's model matches the generator "
               "(SR + h + window + recording bins).")
    lim.append("The Wilson intervals treat cohorts as independent. Cohorts at one parameter point (or one "
               "real anchor) are not, so read each interval beside its cluster count.")
    lim.append("The n and hL strata are descriptive only: these data have been looked at before, "
               "so no stratum is a finding.")
    return {"source": SOURCE, "quantity": QUANTITY, "planted": planted, "zero": zero,
            "box_hL": (list(next(iter(box_hl))) if len(box_hl) == 1 else None),
            "negligible_hL": NEGLIGIBLE_HL, "cap_hL": list(CAP_HL),
            "strata": strata, "not_in_run": dict(not_in_run or {}), "limitations": lim,
            "hb4_floor_multiple": HB4_FLOOR_MULTIPLE}
