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
from ..score import nuisance as NU
from ..score import ridge as RG

#: 2 (2026-09-27): the four raw parameters gained a FLOORED error and a real `clamped` flag, the
#: ten combinations gained an error at all, and the cohort block gained the rho coordinates, the
#: observable CV and the regime axis. Schema 1 payloads are missing those keys, not wrong in them.
#: 3 (2026-10-29): combinations gained `result.spread` (how tightly the data pin them),
#: `result.err_theta` / `result.theta_fill` / `result.clamped_theta` (the error at the fitted point,
#: what it filled, and whether the floor set it), the
#: cohort block `fit_stale` / `fit_gap_nats` / `theta_at_edge`, and the top-level
#: `combo_resolution`. `result.err` is unchanged.
PAYLOAD_SCHEMA_VERSION = "3"

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


#: Where a validity audit lives if nobody says otherwise. The audit is produced OUTSIDE this
#: repository (the jax2 session owns the tool); the file is taken in as data, with its sha256.
#: The v1 audit is `validity.csv`; every later library names its version. Kept as data rather
#: than a rule so the v1 exception cannot be "simplified" away.
VALIDITY_BY_LIBRARY = {"v1": "validity.csv"}


def validity_for(library_version: str | None) -> str:
    """The validity audit that matches a LIBRARY, not a module-level default.

    A default made this a silent mismatch: the explorer pairs a run's cohorts with whatever audit
    the constant named, so a v5 run took validity_v6.csv's flags (or, before 2026-10-27, a v6 run
    took v5's) and the page said nothing. The flags then describe different cohorts from the ones
    drawn, which looks like data rather than a mix-up.
    """
    ver = (library_version or "").strip()
    if not ver:
        return os.path.join("reports", "validity", "validity.csv")
    return os.path.join("reports", "validity",
                        VALIDITY_BY_LIBRARY.get(ver, f"validity_{ver}.csv"))


#: Only for callers with no library to hand. Anything holding a library must use validity_for().
DEFAULT_VALIDITY = validity_for("v6")


def _sha256(path: str) -> str | None:
    """So a figure can name the exact audit it was filtered by."""
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def load_validity(path: str | None = None, library_version: str | None = None) -> dict:
    """`{cohort_id: {"valid": 0|1, "reason": str}}` from a validity audit CSV, or {} if absent.

    VALIDITY IS NOT THE SAME THING AS CORRECT SPECIFICATION, and the explorer needs both.
    `well_specified` asks whether the METHOD fits the model that generated the cohort. Validity
    asks whether the COHORT'S OWN DATA is trustworthy -- on library v5, 1,047 of 2,410 are not,
    because the backward-Euler generator inflates lifespan CV (`cv_bias`) or leaves a death with
    zero probability mass. A coverage figure computed on those is partly measuring a generator
    defect, which is exactly the class of error this project keeps finding.

    Returned empty when the file is missing, and the views then say validity is unavailable rather
    than quietly treating every cohort as fine.
    """
    path = path or validity_for(library_version)
    if not os.path.exists(path):
        return {}
    import csv
    out = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cid = r.get("cohort_id")
            if not cid:
                continue
            bad = str(r.get("flag_invalid", "")).strip().lower() in ("true", "1", "yes")
            out[cid] = {"valid": 0 if bad else 1, "reason": (r.get("flag_reason") or "") if bad else ""}
    return out


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


def _method_bounds(method_name):
    """The method's declared parameter bounds, or None if it is not importable here -- guarded for
    the same reason as `_method_capabilities`. Used to form a filled point exactly as the method
    forms a claim point (theta_hat clipped to the bounds) and to say which sit on a box edge."""
    if not method_name:
        return None
    try:
        from ..contract import method as M
        return dict(M.get(method_name).descriptor.bounds or {}) or None
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
                  reference_sets=None, validity_path=None, reference_manifest=None) -> dict:
    """Assemble the payload from stored results. Performs NO calibration.

    The reference is stamped from `reference_manifest` -- the run whose results are
    `reference_sets` -- when given. `explorer --reference X` loaded X but stamped the pointer's
    run, so the reference timing columns and their label disagreed (measured by the
    mle_profile_jax2 session, 2026-10-28; the same defect 4fe9714 fixed in `score`). The pointer
    stamps only for callers that loaded the reference FROM it, as the autobuild does.
    """
    # The external hazard as a NUISANCE (`score/nuisance.py`): where a record stores the claim the
    # method WOULD make on h, the views show it -- on copies, labelled as not returned by the method
    # (`nuisance_quantities` below). A method that stores none is unaffected.
    n_lifted = sum(1 for cs in claim_sets
                   if cs.claim_for(NU.QUANTITY) is None and NU.hazard_claim(cs) is not None)
    if n_lifted:
        claim_sets = NU.with_hazard_claims(claim_sets)
    by_id = {cs.cohort_id: cs for cs in claim_sets}
    ref_by_id = {cs.cohort_id: cs for cs in (reference_sets or [])}
    ids = [cid for cid in run_manifest["cohort_ids"] if cid in by_id]
    specs = [library.spec(cid) for cid in ids]

    capabilities = _method_capabilities(run_manifest.get("method"))
    feats = [cohort_features(sp) for sp in specs]
    validity = load_validity(validity_path, getattr(library, "version", None))

    # What each stored fit says about its own ridge -- read once per cohort, used by every
    # combination below (`score/ridge.py`). Absent for methods that do not store it.
    bounds = _method_bounds(run_manifest.get("method"))
    fit = {}
    for cid in ids:
        dg = by_id[cid].diagnostics if by_id[cid].ok else {}
        th = RG.fitted_theta(dg, bounds)
        fit[cid] = {"theta": th, "near": RG.near_optimal_thetas(dg),
                    "stale": RG.fit_staleness(dg),
                    "edge": RG.at_box_edge(th, bounds,
                                           ((dg or {}).get("self_consistency") or {}).get("edge_dex"))}

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
        # 1 valid, 0 invalid, null when the audit does not cover this cohort. NULL IS NOT VALID:
        # a cohort the audit never saw must not be silently counted as sound.
        "valid": [(validity.get(cid) or {}).get("valid") for cid in ids],
        "invalid_reason": [(validity.get(cid) or {}).get("reason", "") for cid in ids],
        # How the fitted point relates to the best point the fit evaluated (the method's own class:
        # none <= 0.01 < minor <= thr/2 < major) and which fitted parameters sit on a box edge.
        # Shown beside every error at the fitted point, so a reader can drop what they distrust.
        "fit_stale": [fit[c]["stale"][0] for c in ids],
        "fit_gap_nats": [fit[c]["stale"][1] for c in ids],
        "theta_at_edge": [fit[c]["edge"] for c in ids],
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
    result = {"hit": [], "err": [], "width": [], "shape": [], "clamped": [], "applicable": [],
              "spread": [], "err_theta": [], "theta_fill": [], "clamped_theta": []}
    for q in quantities:
        hit, err, wid, shp, clm, app = [], [], [], [], [], []
        spr, eth, fil, clt = [], [], [], []
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

                # NOT "the MLE of g is g(theta_hat)". That invariance holds for the exact maximum;
                # theta_hat is not the best point the fit evaluated on 489 of 1,776 J v6 records,
                # and kappa_hat sits on a box edge on 383 of the 1,314 where kappa was declined.
                # More to the point, on most beta-combinations the fit's own near-optimal points
                # span > 1 dex, so g(theta_hat) is mostly the place the fit stopped. Hence two rows
                # beside `err`, which stays as it was: how tightly the data pin g (`spread`), and
                # the error at the fitted point with what it filled (`err_theta`, `theta_fill`).
                # The explorer toggles between them; neither is a claim or a coverage.
                f = fit[cid]
                spr.append(RG.combo_spread(f["near"], q, floor) if f["near"] else None)
                cp = {p: (c.point if c is not None else None) for p, c in zip(PARAMETERS, pts)} \
                    if cs.ok else {}
                th_c, filled = RG.combination_inputs(cp, f["theta"]) if cs.ok else ([], [])
                g = CB.combos(th_c).get(q) if th_c else None
                g = g if (g is not None and math.isfinite(g)) else None
                eth.append(CL.log_error(g, truth, floor) if (g and truth and truth > 0) else None)
                clt.append(_was_clamped(g, truth, floor) if eth[-1] is not None else None)
                # Filled only if THIS combination reads a filled input: NaN there propagates.
                probe = CB.combos([float("nan") if p in filled else v
                                   for p, v in zip(PARAMETERS, th_c)]).get(q) if filled else None
                uses = bool(filled) and not (probe is not None and math.isfinite(probe))
                fil.append(None if eth[-1] is None else int(uses))
                continue

            spr.append(None); eth.append(None); fil.append(None); clt.append(None)
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
        result["spread"].append(spr)
        result["err_theta"].append(eth)
        result["theta_fill"].append(fil)
        result["clamped_theta"].append(clt)

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

    if reference_manifest is not None:
        ref_stamp = (reference_manifest.get("method"), reference_manifest.get("method_version"),
                     reference_manifest.get("run_id"))
    else:
        ref_stamp = tuple(getattr(reference_pointer, a, None)
                          for a in ("method_name", "method_version", "run_id"))

    return {
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "provenance": {
            "library_version": run_manifest["library_version"],
            "contract_version": run_manifest["contract_version"],
            "method_name": run_manifest["method"],
            "method_version": run_manifest["method_version"],
            "run_id": run_manifest.get("run_id", ""),
            "reference_method": ref_stamp[0],
            "reference_version": ref_stamp[1],
            "reference_run_id": ref_stamp[2],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arms_present": arms_present,
            "arms_absent": [a for a in all_arms if a not in arms_present],
            #: What the method declares it models. None = the method was not importable here.
            "method_capabilities": list(capabilities) if capabilities is not None else None,
            "all_capabilities": list(CAPABILITIES),
            #: The nominal level every coverage figure is coverage OF.
            "level": level,
            #: Where the validity flags came from, and how much of this run they cover. A figure
            #: filtered by an audit that only saw half the cohorts would otherwise look authoritative.
            "validity_source": (validity_path or validity_for(getattr(library, "version", None))) if validity else None,
            "validity_sha256": _sha256(validity_path or validity_for(getattr(library, "version", None))) if validity else None,
            "validity_coverage": (round(sum(1 for c in ids if c in validity) / len(ids), 4)
                                  if ids and validity else 0.0),
        },
        "arms": arms,
        #: Combinations carry an error but no interval -- see the combo branch in the result loop.
        "accuracy_only_quantities": list(CB.COMBO_NAMES),
        #: Quantities whose claims are the method's WOULD-BE claims, read from its diagnostics and
        #: not returned by it. Every view of one says so; none enters a verdict.
        "nuisance_quantities": ({NU.QUANTITY: {"source": NU.SOURCE, "n_cohorts": n_lifted}}
                                if n_lifted else {}),
        #: How the views decide whether a combination is pinned by the data (`score/ridge.py`).
        "combo_resolution": {
            "resolved_dex": RG.RESOLVED_DEX,
            "unresolved_dex": RG.UNRESOLVED_DEX,
            "spread": ("log10 range of the combination over the fit's own near-optimal points "
                       "(within half_thr of the best log-likelihood), floored like the error: a "
                       "LOWER bound on what the data leave undetermined; not a claim"),
        },
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
