"""The explorer's data payload: columnar, versioned, and emitted as its own file.

WHY COLUMNAR, MEASURED. The predecessor's `mle_report/explorer.html` is 2,484,416 bytes, of which
**2,444,512 (98.4%) is the data and 39,862 (1.6%) is all the HTML, CSS and view code combined**.
That ratio is the whole reason a new view is cheap: the expensive artifact is the data, and the
view is nearly free to replace. One array per field, all of equal length, is what keeps it that
small.

WHY IT IS ALSO WRITTEN AS A SEPARATE FILE. Embedding alone means a new view must regenerate the
whole explorer to reach the data, and a third party cannot build one at all. The sidecar makes the
data reusable on its own -- which is what "adding a view costs zero calibrations" actually requires.

WHAT THE PAYLOAD CARRIES THAT A VIEW CANNOT OMIT. Denominators, the decline counts, the reference
identity and the arms present and absent all travel IN the payload, so a view physically cannot
render a figure without them. That is the difference between a rule and an enforced rule.
"""
from __future__ import annotations

import json
import math
import os
import time

from ..contract.types import (CAPABILITIES, OPTIONAL_QUANTITIES, PARAMETERS,
                              SHAPE_CODE)
from ..score import claims as CL
from ..score import combos as CB
from ..score import floors as FL

#: 2 (2026-09-27): the four raw parameters gained a FLOORED error and a real `clamped` flag, the
#: ten combinations gained an error at all, and the cohort block gained the rho coordinates, the
#: observable CV and the regime axis. Schema 1 payloads are missing those keys, not wrong in them.
PAYLOAD_SCHEMA_VERSION = "2"

#: Which observation-model feature each arm plants, and the capability that would model it.
#: `pure` plants nothing; `realfull` plants more than one (`library/spec.py:arm_for`).
ARM_FEATURE = {
    "pure": (None, None),
    "hazard": ("an age-independent external hazard added to SR mortality", "external_hazard"),
    "window": ("observation restricted to a time window", "fit_window"),
    "het": ("between-individual heterogeneity in one parameter", "heterogeneity"),
    "censor": ("a fraction of individuals right-censored", "censoring"),
    "realfull": ("more than one feature at once -- what a real cohort looks like", None),
}

#: What the method is actually SHOWN. `Observation` carries no truth, so a feature is invisible
#: unless it survives into one of these fields (`contract/types.py:Observation`).
FEATURE_VISIBLE_VIA = {
    "external_hazard": None,          # nothing in the observation reveals it
    "heterogeneity": None,            # nor it
    "censoring": "the `events` flags",
    "fit_window": "`observation.window`",
}


def cohort_features(spec) -> list[str]:
    """The observation-model features actually present in ONE cohort, as capability names.

    Per cohort rather than per arm, because `realfull` is defined as carrying more than one and a
    single arm label cannot say which. Measured on library v2: 68 cohorts carry two or three.
    """
    out = []
    if getattr(spec, "external_hazard", None) is not None:
        out.append("external_hazard")
    if getattr(spec, "window", None) is not None:
        out.append("fit_window")
    if getattr(spec, "het_param", None) is not None:
        out.append("heterogeneity")
    if getattr(spec, "censor_frac", 0):
        out.append("censoring")
    return out


def _well_specified(features, capabilities) -> int:
    """Is the model the method fits the model that generated this cohort?

    True only when EVERY feature present is one the method declares it models. A cohort with a
    feature the method does not model is a misspecification measurement -- still worth having, and
    the whole point of the feature arms, but it is not a calibration result and pooling the two
    silently answers a different question than the reader thinks.

    `capabilities is None` means the method could not be loaded, so this cannot be decided; such a
    cohort is left OUT of the well-specified set rather than assumed into it.
    """
    if capabilities is None:
        return 0
    return int(all(f in capabilities for f in features))


def _was_clamped(estimate, truth, floor) -> int:
    """Did the negligibility floor do the work in this cell?

    The column exists so a reader can separate "the method got this right" from "nobody could have
    measured this". It was declared in schema 1 and written as a literal 0 for every cell, so the
    table columns that depend on it (`unclamped`, `clamp%`) had nothing to read.
    """
    if floor is None or estimate is None or truth is None:
        return 0
    try:
        return int(float(estimate) <= float(floor) or float(truth) <= float(floor))
    except (TypeError, ValueError):
        return 0


def _regime_axis(spec) -> float | None:
    """`log10( rho_beta / ((kappa+1) * rho_eta) )` -- ballistic below 0, quasi-steady above."""
    try:
        v = spec.rho_beta / ((spec.kappa + 1.0) * spec.rho_eta)
        return math.log10(v) if v > 0 else None
    except (ZeroDivisionError, ValueError, TypeError):
        return None


def _observed_cv(library, cohort_id: str) -> float | None:
    """CV of the cohort's OBSERVED deaths -- the manifold's second coordinate.

    Measured rather than looked up, because `cv` is not on `CohortSpec`: it is computed when the
    library is planned and only its band survives, inside the `stratum` string ("cv-sharp" and so
    on). A band cannot be a plot axis.

    Censored records are excluded: a censoring time is not a lifespan, and pooling the two would
    make the censor arm's CV a statement about the censoring rule rather than about the cohort.
    Returns None rather than a guess when the cohort cannot be loaded -- the manifold view then
    says so instead of drawing a point at the wrong height.
    """
    try:
        obs = library.observation(cohort_id)
        deaths = [float(a) for a, e in zip(obs.ages, obs.events) if int(e) == 1]
        if len(deaths) < 2:
            return None
        mean = sum(deaths) / len(deaths)
        if mean <= 0:
            return None
        var = sum((a - mean) ** 2 for a in deaths) / (len(deaths) - 1)
        return math.sqrt(var) / mean
    except Exception:                      # noqa: BLE001 - a missing array must not fail emission
        return None


def _method_capabilities(method_name):
    """What the method declares it models, or None if the method is not importable here.

    None is distinct from `()`: "we could not ask" must not render as "it models nothing". The
    lookup is guarded because scoring stored results is meant to work without the method installed
    -- that standalone property is worth more than this panel.
    """
    if not method_name:
        return None
    try:
        from ..contract import method as M
        return tuple(M.get(method_name).descriptor.capabilities or ())
    except Exception:                      # noqa: BLE001
        return None


def _arm_row(arm: str, n: int, capabilities) -> dict:
    """One row of the arms panel: what this arm plants, and what the method did about it.

    THREE STATES, NOT TWO -- the distinction Naveh asked for on 2026-09-27, and the reason a flat
    "not fitted" was wrong:

      * invisible  the feature is not a field on `Observation` at all, so the cohort is fitted
                   under the assumption that the feature is ABSENT. This is misspecification
                   (`hazard`, `het`).
      * ignored    the observation carries it and the method does not read it (`window`).
                   Actionable in a way that an invisible feature is not.
      * handled    it reaches the likelihood and is treated correctly (`censor`: censored records
                   contribute log-survival, not log-density).

    No method is ever HANDED a feature's true value: `Observation` deliberately contains no truth.
    """
    feature, capability = ARM_FEATURE.get(arm, (None, None))
    row = {"arm": arm, "n": n, "feature": feature, "capability": capability}
    if arm == "pure":
        row.update(state="none", detail="nothing added -- SR exactly as specified")
        return row
    if arm == "realfull":
        row.update(state="mixed",
                   detail="several features at once; each follows its own row above")
        return row
    if capabilities is None:
        row.update(state="unknown",
                   detail="the method could not be loaded here, so what it models is unknown")
        return row
    via = FEATURE_VISIBLE_VIA.get(capability)
    if capability in capabilities:
        row.update(state="handled", detail=f"declared as modelled, and visible via {via}")
    elif via is None:
        row.update(state="invisible",
                   detail=("nothing in the observation reveals it, so the cohort is fitted under "
                           "the assumption that it is absent -- misspecification, not calibration"))
    else:
        # NOT asserted as misspecification. The observation carries the feature, so a method MAY
        # be handling it without declaring the capability. The capability list is the only thing
        # the harness can know for an arbitrary method, so the panel reports the declaration and
        # says plainly that it cannot see inside the estimator.
        #
        # CENSORING IS THE CAUTIONARY CASE, and an earlier version of this comment got it wrong.
        # The censored TERM is correct in both shipped methods -- a censored record contributes
        # log-survival at its recorded age -- but on THIS library that term meets a generation
        # defect: the cohort is censored at a sample quantile c and the ages are floored to delta
        # only afterwards, so a death recorded at b = floor(c) actually died in [b, c) while the
        # likelihood scores it over [b, b+delta). The implied total probability exceeds 1.
        # Measured on library v2: 287 of 304 censor-bearing cohorts carry such ties (median 1.2%
        # of n, max 33%), and paired against their pure twins eta coverage falls 0.843 -> 0.640.
        #
        # So "the term is right" and "these cohorts are well specified" are DIFFERENT claims, and
        # only the first is true. Declaring the capability would move all of them into the default
        # correctly-specified view and contaminate exactly the figure that view exists to keep
        # clean. Tracked as [T-TESTER-DEFECTS] 7.
        row.update(state="undeclared",
                   detail=(f"the observation carries it (in {via}), but the method does not "
                           "declare modelling it -- so whether it is handled cannot be read off "
                           "the contract"))
    return row


def build_payload(library, claim_sets, run_manifest: dict, reference_pointer=None,
                  reference_sets=None) -> dict:
    """Assemble the payload from stored results. Performs NO calibration."""
    by_id = {cs.cohort_id: cs for cs in claim_sets}
    ref_by_id = {cs.cohort_id: cs for cs in (reference_sets or [])}
    ids = [cid for cid in run_manifest["cohort_ids"] if cid in by_id]
    specs = [library.spec(cid) for cid in ids]

    capabilities = _method_capabilities(run_manifest.get("method"))
    feats = [cohort_features(sp) for sp in specs]

    cohort = {
        "id": ids,
        "position": [s.position for s in specs],
        "n": [s.n for s in specs],
        "arm": [s.arm for s in specs],
        "stratum": [s.stratum for s in specs],
        "cluster_key": [s.cluster_key for s in specs],
        "kappa": [s.kappa for s in specs],
        "log_ratio": [s.log_ratio for s in specs],
        "delta_over_L": [(s.delta / s.L_phys) if s.delta else 0.0 for s in specs],
        "real_name": [s.real_name or "" for s in specs],
        # The dimensionless truth coordinates. A cube or a regime axis is not derivable without
        # them, and they are pure metadata -- no calibration, no observation load.
        "rho_eta": [s.rho_eta for s in specs],
        "rho_beta": [s.rho_beta for s in specs],
        "rho_eps": [s.rho_eps for s in specs],
        # The regime axis: removal at the damage threshold over production at the lifespan scale.
        # The L is already folded in by the rho convention, which is what makes the threshold at 1
        # unit-free -- beta/((kappa+1)*eta) on its own is a time, so its sign against 1 would
        # depend on the recording unit.
        "qs": [_regime_axis(s) for s in specs],
        # The manifold's second coordinate. NOT stored on the spec (only its band survives, in
        # `stratum`), so it is measured here from the cohort's own observed deaths.
        "cv": [_observed_cv(library, cid) for cid in ids],
        # Which observation-model features this cohort carries, and whether the method models all
        # of them. This is what lets a figure separate "how well does it calibrate" from "what
        # does misspecification cost", instead of pooling the two.
        "features": feats,
        "well_specified": [_well_specified(f, capabilities) for f in feats],
        "group": [s.group for s in specs],
    }

    # Three classes of quantity, and the distinction matters downstream:
    #   PARAMETERS        always scored, always claimable
    #   optional          scored and claimable, but only where the cohort set carries a truth
    #   combinations      accuracy only -- a point estimate, never an interval
    truths_all = library.truths(ids)
    optional = [q for q in OPTIONAL_QUANTITIES
                if any(t.get(q) is not None for t in truths_all.values())]
    quantities = list(PARAMETERS) + optional + list(CB.COMBO_NAMES)
    # `applicable` is 1 where the quantity EXISTS for that cohort. Always 1 for the four
    # parameters; 0 for e.g. external_hazard on a cohort that carries no hazard.
    #
    # Without it a view cannot tell "the method declined" from "there was nothing to claim", and
    # external_hazard reads as 100% declined across a pure-arm run. The scorer applies exactly
    # this filter via the truth lookup; this is the same filter, made explicit for the views.
    result = {"hit": [], "err": [], "width": [], "shape": [], "clamped": [], "applicable": []}
    for q in quantities:
        hit, err, wid, shp, clm, app = [], [], [], [], [], []
        # By NAME, not by exclusion: an optional quantity is claimable and interval-bearing like
        # a parameter, so treating "not a parameter" as "a combination" would strip its coverage.
        is_combo = q in CB.COMBO_NAMES
        for cid, s in zip(ids, specs):
            cs = by_id[cid]
            theta_true = s.physical_truth()

            if is_combo:
                # A COMBINATION IS SCORED FOR ACCURACY ONLY, and that is not an omission.
                #
                # The method claims intervals on the four parameters; nothing in the harness
                # derives an interval for a function of them, and inventing one would be the very
                # thing `mle_profile` refuses to do -- "inventing a bound the profile did not
                # actually produce is how a decline turns into a false measurement". So `hit`,
                # `width` and `shape` stay empty here and every coverage view says so, rather than
                # quoting a number nobody computed.
                #
                # Before this, combinations carried NO column at all: ten of the fourteen
                # selectable quantities rendered a blank panel.
                # Substitute NaN for a parameter the method declined, then keep the result only if
                # it came out finite. NaN propagates through whichever parameters the formula
                # actually reads, so a combination is available exactly when ITS OWN inputs are --
                # no dependency table to keep in step with `COMBOS`.
                #
                # Requiring all four instead cost real data: kappa is declined on most cohorts, so
                # `eta_over_eps`, which does not involve kappa at all, was available on only 497 of
                # 1,990 cohorts rather than on every cohort that estimated eta and epsilon.
                pts = [cs.claim_for(p) for p in PARAMETERS] if cs.ok else []
                theta_est = [(c.point if (c is not None and c.point is not None) else float("nan"))
                             for c in pts] if cs.ok else []
                truth = CB.combos([theta_true[p] for p in PARAMETERS]).get(q) if theta_true else None
                est = CB.combos(theta_est).get(q) if theta_est else None
                if est is not None and not math.isfinite(est):
                    est = None
                bf = FL.cohort_floor("beta", s)
                floor = CB.combo_floor(q, [theta_true[p] for p in PARAMETERS], bf) \
                    if theta_true else None
                e = CL.log_error(est, truth, floor) if (est and truth and truth > 0) else None
                err.append(e)
                clm.append(_was_clamped(est, truth, floor))
                hit.append(None); wid.append(None); shp.append(SHAPE_CODE["none"])
                app.append(1 if truth else 0)
                continue

            truth = theta_true.get(q)
            app.append(0 if truth is None else 1)
            c = cs.claim_for(q) if cs.ok else None
            if c is None:
                hit.append(None); err.append(None); wid.append(None)
                shp.append(SHAPE_CODE["none"]); clm.append(0)
                continue
            shp.append(c.shape_code)
            # None means NO CLAIM -- never counted as either a hit or a miss.
            h = c.contains(truth) if truth is not None else None
            hit.append(None if h is None else int(h))
            # FLOORED, as the report floors it (`report/comparison.py`). Passing None here meant
            # the explorer plotted an error the comparison no longer treats as real: on the medium
            # comparison, pooled beta read +1.21 dex unfloored against +0.062 dex floored, because
            # 182 of 350 beta truths lie below their own cohort's floor. A panel disagreeing with
            # the report it ships beside is worse than a missing panel.
            floor = FL.cohort_floor(q, s)
            err.append(CL.log_error(c.point, truth, floor) if truth else None)
            wid.append(c.width_dex())          # None for anything not closed
            clm.append(_was_clamped(c.point, truth, floor))
        result["hit"].append(hit)
        result["err"].append(err)
        result["width"].append(wid)
        result["shape"].append(shp)
        result["clamped"].append(clm)
        result["applicable"].append(app)

    timing = {
        "wall_clock_seconds": [by_id[c].wall_clock_seconds for c in ids],
        "cpu_seconds": [by_id[c].cpu_seconds for c in ids],
        "ref_wall_clock_seconds": [ref_by_id[c].wall_clock_seconds if c in ref_by_id else None
                                   for c in ids],
        "ref_cpu_seconds": [ref_by_id[c].cpu_seconds if c in ref_by_id else None for c in ids],
    }

    denominators = {}
    truths = library.truths(ids)
    sets = [by_id[c] for c in ids]
    for q in list(PARAMETERS) + optional:
        closed = CL.coverage(sets, truths, q, "closed")
        one = CL.coverage(sets, truths, q, "one_sided")
        denominators[q] = {"closed": closed.to_dict(), "one_sided": one.to_dict()}

    arms_present = sorted({s.arm for s in specs})
    all_arms = ("pure", "hazard", "window", "het", "censor", "realfull")

    # The nominal level, READ FROM THE CLAIMS rather than assumed. A coverage figure without it is
    # unreadable: "0.823" only means something against the level it is coverage of.
    level = None
    for cs in sets:
        for c in cs.claims:
            if getattr(c, "level", None) is not None:
                level = float(c.level)
                break
        if level is not None:
            break

    counts = {a: sum(1 for s in specs if s.arm == a) for a in arms_present}
    arms = [_arm_row(a, counts[a], capabilities) for a in arms_present]

    return {
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "provenance": {
            "library_version": run_manifest["library_version"],
            "contract_version": run_manifest["contract_version"],
            "method_name": run_manifest["method"],
            "method_version": run_manifest["method_version"],
            "run_id": run_manifest.get("run_id", ""),
            "reference_method": getattr(reference_pointer, "method_name", None),
            "reference_version": getattr(reference_pointer, "method_version", None),
            "reference_run_id": getattr(reference_pointer, "run_id", None),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arms_present": arms_present,
            "arms_absent": [a for a in all_arms if a not in arms_present],
            #: What the method declares it models. None = the method was not importable here.
            "method_capabilities": list(capabilities) if capabilities is not None else None,
            "all_capabilities": list(CAPABILITIES),
            #: The nominal level every coverage figure is coverage OF.
            "level": level,
        },
        "arms": arms,
        #: Combinations carry an error but no interval -- see the combo branch in the result loop.
        "accuracy_only_quantities": list(CB.COMBO_NAMES),
        "quantities": quantities,
        "n_cohorts": len(ids),
        "cohort": cohort,
        "result": result,
        "timing": timing,
        "denominators": denominators,
    }


def validate_payload(p: dict) -> None:
    """Equal-length columns, a declared schema version, and the provenance a view cannot omit.

    A length mismatch fails emission rather than shipping a silently misaligned column -- which
    would attribute every value to the wrong cohort and look entirely plausible.
    """
    n = p["n_cohorts"]
    for group in ("cohort", "timing"):
        for k, v in p[group].items():
            if len(v) != n:
                raise ValueError(f"{group}.{k} has {len(v)} entries against n_cohorts={n}")
    nq = len(p["quantities"])
    for k, v in p["result"].items():
        if len(v) != nq:
            raise ValueError(f"result.{k} has {len(v)} rows against {nq} quantities")
        for i, col in enumerate(v):
            if len(col) != n:
                raise ValueError(f"result.{k}[{i}] has {len(col)} entries against n_cohorts={n}")
    if not p.get("payload_schema_version"):
        raise ValueError("payload has no schema version; a view written today must keep working")
    for required in ("library_version", "arms_present", "arms_absent"):
        if required not in p["provenance"]:
            raise ValueError(f"payload provenance is missing {required!r}")
    if not p.get("denominators"):
        raise ValueError("payload carries no denominators, so a view could render a figure "
                         "without one")


def write(payload: dict, out_dir: str) -> str:
    validate_payload(payload)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "explorer_data.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return p
