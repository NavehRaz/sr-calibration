"""Deliverable A: the five-axis comparison against the current reference.

RUN TIME IS A FIRST-CLASS RESULT HERE, not a footnote. It is measured per calibration, aggregated,
and reported for BOTH the method and the reference, next to the accuracy axes. It never decides:
a method is not refused for being expensive nor accepted for being cheap. But a tenfold cost for a
hundredth of coverage is a different trade from a tenfold saving, and the reader is entitled to see
which one they are being offered.

EVERY FIGURE CARRIES its denominator, its cohort set, whether it is paired, and the aggregation
convention. Every coverage appears with its decline rate and beside the data-ignoring null. The
reference is named with its version and run id, because once the reference can be re-pointed two
documents with the same title can rest on different baselines.

The document is GENERATED from stored results. `report/verify.py` recomputes every number in it
independently and shares no code with this module, because a verifier built on the generator's
helpers re-tests the generator against itself.
"""
from __future__ import annotations

import json
import math
import os
import time

from ..contract.types import PARAMETERS
from ..library import order as O
from ..score import claims as CL
from ..score import floors as F
from ..score import nuisance as NU
from ..score import verdict as V


def build_comparison(library, method_sets, reference_sets, run_manifest, ref_manifest,
                     reference_pointer=None, null_sets=None) -> dict:
    """Compute the whole comparison as DATA. Rendering is separate and derives from this."""
    ids = sorted({cs.cohort_id for cs in method_sets} & {cs.cohort_id for cs in reference_sets})
    truths = library.truths(ids)
    clusters = library.cluster_map(ids)
    specs = [library.spec(c) for c in ids]
    strength = O.claim_strength(specs)
    a = [cs for cs in method_sets if cs.cohort_id in set(ids)]
    b = [cs for cs in reference_sets if cs.cohort_id in set(ids)]
    nulls = [cs for cs in (null_sets or []) if cs.cohort_id in set(ids)]

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": {"name": run_manifest["method"], "version": run_manifest["method_version"],
                   "run_id": run_manifest.get("run_id", "")},
        "reference": {"name": ref_manifest["method"], "version": ref_manifest["method_version"],
                      "run_id": ref_manifest.get("run_id", ""),
                      "pointer": reference_pointer.to_dict() if reference_pointer else None},
        "library_version": run_manifest["library_version"],
        "contract_version": run_manifest["contract_version"],
        "cohort_set": {"paired": len(ids), "method_total": len(method_sets),
                       "reference_total": len(reference_sets)},
        "claim_strength": strength,
        "by_parameter": {},
        "runtime": {},
    }

    for p in CL.scored_quantities(truths):
        axes = {
            "calibration": V.calibration_axis(a, b, truths, p, clusters,
                                              cohort_set=f"{len(ids)} paired"),
            # one-sided bounds, scored separately and never pooled (added 2026-09-25)
            "calibration_one_sided": V.calibration_one_sided_axis(a, b, truths, p, clusters,
                                                                  f"{len(ids)} paired"),
            "availability": V.availability_axis(a, b, p, f"{len(ids)} paired", truths),
            "information": V.information_axis(a, b, p, f"{len(ids)} paired"),
            # floors applied (they were `{}` until 2026-09-25 -- see score/floors.cohort_floors)
            "accuracy": V.accuracy_axis(a, b, truths, p,
                                        F.cohort_floors(dict(zip(ids, specs)), p),
                                        f"{len(ids)} paired"),
            "cost": V.cost_axis(a, b, f"{len(ids)} paired"),
        }
        overall, reasons = V.decide(axes)
        entry = {
            "overall": overall, "reasons": reasons,
            "axes": {k: v.to_dict() for k, v in axes.items()},
            "coverage": {
                "method": CL.coverage(a, truths, p, "closed", f"{len(ids)} paired").to_dict(),
                "reference": CL.coverage(b, truths, p, "closed", f"{len(ids)} paired").to_dict(),
                "method_one_sided": CL.coverage(a, truths, p, "one_sided").to_dict(),
                "reference_one_sided": CL.coverage(b, truths, p, "one_sided").to_dict(),
            },
            # The null accompanies EVERY coverage figure. A coverage it matches is not evidence.
            "null": (CL.coverage(nulls, truths, p, "closed", f"{len(ids)} paired").to_dict()
                     if nulls else None),
            "null_one_sided": (CL.coverage(nulls, truths, p, "one_sided").to_dict()
                               if nulls else None),
        }
        out["by_parameter"][p] = entry

    # Run time, as a result rather than a footnote.
    def _times(sets):
        w = sorted(cs.wall_clock_seconds for cs in sets if cs.wall_clock_seconds)
        c = sorted(cs.cpu_seconds for cs in sets if cs.cpu_seconds)
        med = (lambda x: None if not x else
               (x[len(x) // 2] if len(x) % 2 else 0.5 * (x[len(x) // 2 - 1] + x[len(x) // 2])))
        return {"median_wall_s": med(w), "min_wall_s": w[0] if w else None,
                "max_wall_s": w[-1] if w else None, "median_cpu_s": med(c), "n": len(w),
                "total_cpu_hours": sum(c) / 3600.0 if c else None}

    tm, tr = _times(a), _times(b)
    out["runtime"] = {
        "method": tm, "reference": tr,
        "ratio_median": (tm["median_wall_s"] / tr["median_wall_s"])
        if (tm["median_wall_s"] and tr["median_wall_s"]) else None,
        "note": "reported as a result; never decides on its own",
    }

    # The external hazard as a NUISANCE (score/nuisance.py): the method's own WOULD-BE claim, over
    # the method's whole run -- unpaired, never in `by_parameter`, never in a verdict. None when the
    # method stores no would-be h claim (every method but mle_profile_jax2_h today), so this key
    # is the only change to the returned shape.
    out["nuisance_hazard"] = None
    if any(NU.hazard_claim(cs) is not None for cs in method_sets):
        m_ids = [cs.cohort_id for cs in method_sets]
        in_run = set(m_ids)
        not_in = {}
        for sp in library.specs:
            if sp.external_hazard is not None and sp.cohort_id not in in_run:
                not_in[sp.group] = not_in.get(sp.group, 0) + 1
        out["nuisance_hazard"] = NU.hazard_report(
            method_sets, {c: library.spec(c) for c in m_ids}, library.truths(m_ids), not_in,
            {"method": run_manifest["method"], "method_version": run_manifest["method_version"],
             "library_version": run_manifest["library_version"]})
    return out


def render_markdown(cmp: dict) -> str:
    """The human-readable document. Every number here is read from `cmp` at render time."""
    m, r = cmp["method"], cmp["reference"]
    L = [f"# {m['name']} v{m['version']} against {r['name']} v{r['version']}", ""]
    L.append(f"**Generated** {cmp['generated_at']}  ")
    L.append(f"**Reference** {r['name']} v{r['version']} (run `{r['run_id']}`)  ")
    L.append(f"**Library** {cmp['library_version']} · **Contract** {cmp['contract_version']}  ")
    L.append(f"**Paired cohorts** {cmp['cohort_set']['paired']} "
             f"(method scored {cmp['cohort_set']['method_total']}, "
             f"reference {cmp['cohort_set']['reference_total']})")
    L.append("")

    cs = cmp["claim_strength"]
    if cs["arms_absent"]:
        L.append(f"> **UNTESTED REGIMES in this run: {', '.join(cs['arms_absent'])}.** "
                 "Nothing below says anything about them.")
        L.append("")
    if not cs["supports_per_arm"]:
        L.append("> This size supports a **pooled verdict only** — no per-arm and no per-stratum "
                 "claim is made.")
        L.append("")

    L += ["## Verdict by parameter", ""]
    L.append("| parameter | verdict | calibration | one-sided calibration | availability "
             "| information | accuracy |")
    L.append("|---|---|---|---|---|---|---|")
    for p, e in cmp["by_parameter"].items():
        ax = e["axes"]
        def _f(k):
            if k not in ax:                      # comparisons built before the one-sided axis
                return "n/a"
            v = ax[k]["statistic"]
            return "n/a" if v is None else f"{v:+.4f} ({ax[k]['outcome']})"
        L.append(f"| `{p}` | **{e['overall']}** | {_f('calibration')} "
                 f"| {_f('calibration_one_sided')} | {_f('availability')} "
                 f"| {_f('information')} | {_f('accuracy')} |")
    L.append("")
    if cmp.get("nuisance_hazard") and "external_hazard" in cmp["by_parameter"]:
        L.append("`external_hazard` reads as above because no method RETURNED a claim on h. "
                 f"{m['name']}'s would-be claim is reported apart, unpaired and outside "
                 "every verdict, in the last section.")
        L.append("")

    L += ["## Coverage, with its denominator, decline rate and the null", ""]
    L.append("| parameter | method | n | declined | reference | n | declined | null |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p, e in cmp["by_parameter"].items():
        cm, cr, cn = e["coverage"]["method"], e["coverage"]["reference"], e["null"]
        def _c(d):
            return "n/a" if d["coverage"] is None else f"{d['coverage']:.3f}"

        def _d(d):
            # `decline_rate` is None whenever the denominator is 0 -- which happens for a quantity
            # no cohort in this run carries. Formatting it unguarded raised TypeError and killed
            # render_markdown AFTER comparison.json was written, leaving a half-written report
            # directory. The verifier already prints "n/a" here, so this also makes the two agree.
            return "n/a" if d.get("decline_rate") is None else f"{d['decline_rate']:.3f}"

        L.append(f"| `{p}` | {_c(cm)} | {cm['claimed']} | {_d(cm)} "
                 f"| {_c(cr)} | {cr['claimed']} | {_d(cr)} "
                 f"| {('n/a' if not cn else _c(cn))} |")
    L.append("")
    L.append("Two-sided claims only; one-sided claims are scored separately and never pooled. "
             "A coverage the null matches or beats is not evidence of anything.")
    L.append("")

    L += ["## One-sided bounds, scored separately", ""]
    L.append("| parameter | method | n | reference | n | null |")
    L.append("|---|---|---|---|---|---|")
    for p, e in cmp["by_parameter"].items():
        cm, cr = e["coverage"]["method_one_sided"], e["coverage"]["reference_one_sided"]
        cn = e.get("null_one_sided")
        def _c1(d):
            return "n/a" if not d or d["coverage"] is None else f"{d['coverage']:.3f}"
        L.append(f"| `{p}` (one-sided) | {_c1(cm)} | {cm['claimed']} | {_c1(cr)} | {cr['claimed']} "
                 f"| {_c1(cn)} |")
    L.append("")
    L.append("A bound (lower or upper) is the surviving edge of a 95% two-sided likelihood-ratio "
             "interval, so its nominal coverage is at least 0.95. Declines are not claims and are "
             "not in these denominators.")
    L.append("")

    rt = cmp["runtime"]
    L += ["## Run time", ""]
    L.append("| | median wall | min | max | median CPU | total CPU-h | n |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, d in (("method", rt["method"]), ("reference", rt["reference"])):
        def _s(v):
            return "n/a" if v is None else f"{v:.4g}"
        L.append(f"| {lab} | {_s(d['median_wall_s'])} | {_s(d['min_wall_s'])} "
                 f"| {_s(d['max_wall_s'])} | {_s(d['median_cpu_s'])} "
                 f"| {_s(d['total_cpu_hours'])} | {d['n']} |")
    if rt["ratio_median"]:
        L.append("")
        L.append(f"Median cost ratio **{rt['ratio_median']:.3g}x** the reference. "
                 "Reported as a result; it never decides on its own.")
    L.append("")

    L += ["## Why each verdict", ""]
    for p, e in cmp["by_parameter"].items():
        L.append(f"**`{p}` — {e['overall']}**")
        for reason in e["reasons"]:
            L.append(f"- {reason}")
        L.append("")
    if cmp.get("nuisance_hazard"):
        L += _render_nuisance_hazard(cmp["nuisance_hazard"], m)
    return "\n".join(L)


def _render_nuisance_hazard(nh: dict, m: dict) -> list[str]:
    """The nuisance-h section. Plain-text group labels in the first column, never a backticked
    parameter, so `verify`'s coverage-table parser cannot mistake these rows for scored ones.

    Revised after the sr-naveh review of 2026-10-29: every statistic jax2 registered is anchored to the
    method's box floor (hL = 1e-6), not to what is physically negligible, so each one is printed beside
    the figure that answers the physical question -- the error on closed claims, how many MLEs were
    parked at the floor or a start value, detections that clear negligible h, and whether the data
    CAP h at all."""
    def kn(k, n, w=None):
        base = f"{k}/{n}"
        return base if w is None else f"{base} ({w[0]:.3f}-{w[1]:.3f})"

    def x(v, fmt=".3f"):
        return "n/a" if v is None else format(v, fmt)
    neg = nh.get("negligible_hL", 0.01)
    bh = nh.get("box_hL")
    L = ["## External hazard h — a NUISANCE: reported, unpaired, not in the verdict", ""]
    L.append(f"h is the constant external hazard per unit of the cohort's own age (per day, per year, ...); "
             "the generator stores it as ExtH = -ln h. Every figure here is about "
             f"{m['name']}'s **would-be claim** on h, read from `{nh['source']}`: the claim it would make, "
             "**not returned by the method**, which scores h as a nuisance. One method, no comparator -- "
             "nothing in this section compares methods, and nothing here enters a verdict. Groups are "
             "never pooled: heterogeneous cohorts are apart because the method has no heterogeneity term, "
             "and `realfull` is apart from the hazard arm. Declines are counted and never scored for "
             "coverage. L below is the Kaplan-Meier mean of the OBSERVED cohort (external deaths "
             "included; on a windowed cohort, from left-truncated data), not the SR lifespan, and hL = h x L.")
    L.append("")
    L.append("| group (h planted) | cohorts (clusters) | closed: covered/claims (Wilson 95%) "
             "| clusters with a closed miss | one-sided: covered/claims | declined "
             "| claim width dex (median; n closed) | box null covers truth "
             "| error on closed claims: median · p90 | error, all cohorts (HB8): median · p90 · signed "
             "| MLE at box floor / at a start value (of which closed) | failed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in nh["planted"]:
        a, ac = r["accuracy"], r["accuracy_closed"]
        L.append(f"| {r['group']} | {r['n']} ({r['clusters']}) "
                 f"| {kn(r['closed']['hits'], r['closed']['n'], r['closed']['wilson'])} "
                 f"| {r['clusters_with_a_closed_miss']} of {r['closed_clusters']} "
                 f"| {kn(r['one_sided']['hits'], r['one_sided']['n'])} | {r['declined']} "
                 f"| {x(r['width']['claim_median_dex'], '.2f')} ({r['width']['n_closed']}) "
                 f"| {kn(r['box_null']['hits'], r['box_null']['n'])} "
                 f"| {x(ac['median'])} · {x(ac['p90'])} "
                 f"| {x(a['median'])} · {x(a['p90'])} · {x(a['signed_median'], '+.3f')} "
                 f"| {r['mle_at_floor']} / {r['mle_at_start']} ({r['mle_at_start_closed']}) "
                 f"| {r['failed']} |")
    L.append("")
    floor_n = sum(r["mle_at_floor"] for r in nh["planted"])
    start_cl = sum(r["mle_at_start_closed"] for r in nh["planted"])
    start_n = sum(r["mle_at_start"] for r in nh["planted"]) - start_cl
    L.append("**The error.** The MLE (`diagnostics.hazard.h`) is not clipped to the claim but is confined "
             "to the box. Where the claim is one-sided the likelihood is flat in h and the MLE is wherever "
             f"the optimiser stopped -- the box floor on {floor_n} h-planted cohorts and a start value on "
             f"{start_n} more -- so HB8's all-cohort error there measures the box, not the method. The error "
             "on closed claims is the figure that describes the estimate."
             + (f" On {start_cl} CLOSED claims the MLE is also exactly a start value: the profile is not "
                "flat there, the optimiser simply did not refine h." if start_cl else ""))
    L.append("")
    if bh:
        below = math.log10(neg / bh[0]) if bh[0] > 0 else None
        L.append(f"**The box null** is the method's own search box, fixed at [{bh[0]:g}, {bh[1]:g}]/L for "
                 f"every cohort, so in units of 1/L it is the same {math.log10(bh[1] / bh[0]):.2f}-dex box "
                 "everywhere. It contains the planted h by construction, so an h coverage is not evidence on "
                 "its own: what the method earns is the WIDTH, and only where coverage holds. "
                 + (f"{below:.0f} of the box's {math.log10(bh[1] / bh[0]):.0f} decades lie below hL = "
                    f"{neg:g}, where h is indistinguishable from zero. " if below else "")
                 + "Comparing the claim's width with the box is the tester's addition, not part of jax2's HB8.")
        L.append("")
    else:
        L.append("**The box null** is the method's own per-cohort search box (`diagnostics.hazard.box`). It "
                 "contains the planted h by construction, so an h coverage is not evidence on its own: what "
                 "the method earns is the WIDTH, and only where coverage holds. Comparing the claim's width "
                 "with the box is the tester's addition, not part of jax2's HB8.")
        L.append("")
    if nh["zero"]:
        caps = nh.get("cap_hL") or []
        L.append("| group (no h planted) | cohorts (clusters) | false detection, HB4: detections/cohorts "
                 f"(Wilson 95%) | of which lower edge hL > {neg:g} | shapes closed/upper/lower/none | "
                 + " | ".join(f"upper edge hL <= {c:g}" for c in caps) + " | failed |")
        L.append("|---|---|---|---|---|" + "---|" * len(caps) + "---|")
        for r in nh["zero"]:
            fd, sh = r["false_detection"], r["shapes"]
            L.append(f"| {r['group']} | {r['n']} ({r['clusters']}) | {kn(fd['k'], fd['n'], fd['wilson'])} "
                     f"| {r['detections_above_negligible']} "
                     f"| {sh['closed']}/{sh['upper_bound']}/{sh['lower_bound']}/{sh['none']} | "
                     + " | ".join(kn(c["k"], c["n"]) for c in r["cap"]) + f" | {r['failed']} |")
        L.append("")
        L.append(f"HB4 counts a claim that is closed or a lower bound with its lower edge above "
                 f"{nh['hb4_floor_multiple']:g} x the cohort's h floor (`box[0]`). With the floor at "
                 + (f"hL = {bh[0]:g} that excludes only hL < {nh['hb4_floor_multiple'] * bh[0]:g}" if bh
                    else "its per-cohort value that excludes only an h far below anything that matters")
                 + f" -- a lower edge that low still includes a physically negligible h, so the next column "
                 f"counts the detections whose lower edge clears hL = {neg:g}. A detection in a "
                 "heterogeneity row means the method's model, which has no heterogeneity term, explains the "
                 "curve's extra early deaths as h > 0. Whether survival curves can separate threshold "
                 "heterogeneity from a constant hazard at all is not measured here; that needs a method "
                 "with both terms.")
        L.append("")
        cap_note = "; ".join(
            f"{r['group']} {r['cap'][1]['k']}/{r['cap'][1]['n']}" for r in nh["zero"] if len(r["cap"]) > 1)
        if len(caps) > 1:
            L.append(f"**Can a survival curve cap h?** The upper-edge columns answer it: the share of cohorts "
                     f"with no planted h whose claim bounds h at or below hL = {caps[1]:g} is {cap_note}. A "
                     "truncated fit window removes the young ages where h is the only hazard, and inside a "
                     "mid-life window a constant extra hazard trades off against the level of the SR hazard -- "
                     "a property of the data, not of the method.")
            L.append("")
    for key, title in (("by_n", "sample size"), ("by_hL", "true hL")):
        rows = nh["strata"].get(key) or []
        if not rows:
            continue
        L.append(f"| hazard arm by {title} (descriptive only) | cohorts "
                 "| closed: covered/claims | note |")
        L.append("|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['label']} | {r['n']} | {kn(r['closed_hits'], r['closed_n'])} "
                     f"| {'thin (fewer than 10 cohorts)' if r['thin'] else ''} |")
        L.append("")
    L.append("**Limitations**")
    L.append("")
    L += [f"- {t}" for t in nh["limitations"]]
    L.append("")
    L.append("Definitions: HB4 and HB8 of jax2's registered `phaseC/h_estimate.py` "
             "(PREREG_C_ADD1), with jax2's port notes of 2026-10-29; the closed-claim error, the parked "
             "counts, the negligible-h column, the cap columns and the box comparison are the tester's "
             "additions.")
    L.append("")
    return L


def write(cmp: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "comparison.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(cmp, fh, indent=1)
    mp = os.path.join(out_dir, "comparison.md")
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(cmp))
    return jp, mp
