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
import os
import time

from ..contract.types import PARAMETERS
from ..library import order as O
from ..score import claims as CL
from ..score import floors as F
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
    return "\n".join(L)


def write(cmp: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "comparison.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(cmp, fh, indent=1)
    mp = os.path.join(out_dir, "comparison.md")
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(cmp))
    return jp, mp
