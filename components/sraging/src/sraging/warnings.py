"""The seven truth-free checks run after every fit (T049, T070; FR-016, FR-017, FR-017a, FR-017b, FR-018; spec 002
FR-026 and FR-031).

Conditions and thresholds are VERBATIM from the reference entry point
`tests/oracle/v033/sr_calibration_tester/tools/fit_cohort_jax.py`: `convergence_warnings` (gain > 1.92 nats
on a recentred profile; a closed point outside its own interval beyond a 1e-9 relative tolerance),
`censoring_tie_warning` (delta given, follow-up ending in censoring, tied deaths >= 0.5 % of n), and the
off-grid delta check of `main()` (lines 108-111). `tests/parity/test_warnings_parity.py` proves the
outcomes equal the reference functions' on the stored records. BOX_EDGE (FR-017b) is sraging's own: the
reference has no such check, nor FLOORED_DEATHS or AGE0_DEATHS (spec 002). Warnings never change a number.
"""
from __future__ import annotations

import numpy as np

from sraging.result import CHECK_CODES, CheckRecord, Warning

_GAIN_NATS = 1.92          # fit_cohort_jax.py: half the 95% deviance threshold
_TIE_FRACTION = 0.005      # fit_cohort_jax.py
_EDGE_RTOL = 1e-3          # BOX_EDGE: theta-hat within 0.1 % of an upper edge. 1e-6 missed 22 stored v0.3.3 fits
                           # at beta = 999.0-999.998 with the same failure (eta covered in 3 of 22; review/naveh2.md N1)


def _basin_gain(diagnostics):
    grid = (diagnostics or {}).get("grid") or {}
    gains = {p: g.get("gain_nats", 0.0) for p, g in grid.items() if g.get("recentred")}
    worst = max(gains.values(), default=0.0)
    if worst > _GAIN_NATS:
        ps = [p for p, g in gains.items() if g > _GAIN_NATS]
        return [Warning("BASIN_GAIN",
                        f"the profile walk found a point {worst:.1f} nats better than the reported best fit "
                        f"({', '.join(ps)}). The point is not the maximum and the intervals are centred "
                        f"wrongly -- do not use these numbers ([T-JAX-BASIN]).",
                        {"gain_nats": worst, "parameters": ps})]
    return []


def _basin_outside(claims):
    out = []
    for c in claims:
        if (c.shape == "closed" and None not in (c.point, c.lower, c.upper)
                and not c.lower * (1 - 1e-9) <= c.point <= c.upper * (1 + 1e-9)):
            out.append(Warning("BASIN_OUTSIDE",
                               f"the {c.parameter} point {c.point:.5g} lies outside its own interval "
                               f"[{c.lower:.5g}, {c.upper:.5g}] ([T-JAX-BASIN]).",
                               {"parameter": c.parameter, "point": c.point, "lower": c.lower, "upper": c.upper}))
    return out


def _censor_tie(ages, events, delta, relabel=None):
    """`relabel` (spec 003, jax2 fits): the observation model's censor-tie record. Where jax2 relabelled the tied
    deaths the bias is handled and the check is quiet; where it did not, the rule runs and says why it did not."""
    if relabel is not None and relabel.get("applied"):
        return []
    if not delta or np.all(events == 1):
        return []
    tc = float(np.max(ages[events == 0]))
    if tc < float(np.max(ages)):
        return []                              # censoring is not at the end of follow-up
    tied = int(np.sum((events == 1) & np.isclose(ages, tc)))
    if tied < _TIE_FRACTION * len(ages):
        return []
    return [Warning("CENSOR_TIE",
                    f"{tied} deaths ({tied / len(ages):.1%} of the cohort) are recorded at the censoring age "
                    f"{tc:g}. They are scored as dying anywhere in [{tc:g}, {tc + delta:g}) although they died "
                    f"before follow-up ended, which biases eta upward. If follow-up ended inside that bin, "
                    f"refit with those deaths relabelled as censored at {tc:g} ([T-TESTER-DEFECTS] 7)."
                    + ("" if relabel is None else
                       f" jax2's censor-tie relabel did not apply here: {relabel.get('reason') or 'its trigger did not hold'}"
                       f" (it fires only when every censored record shares one age, the maximum; register D-205)."),
                    {"tied": tied, "fraction": tied / len(ages), "censoring_age": tc})]


def _delta_grid(ages, delta):
    if not delta:
        return []
    off = np.abs(ages / delta - np.round(ages / delta))
    frac = float(np.mean(off > 1e-6))
    if frac > 0:
        return [Warning("DELTA_GRID",
                        f"{frac:.1%} of ages are not multiples of delta {delta:g}; check the recording "
                        f"resolution.", {"fraction": frac, "delta": delta})]
    return []


def _box_edge(theta_hat, box):
    """theta-hat on an UPPER edge of the prior box (D-018). Lower edges (kappa = 1e-3, beta ~ 0) are values the
    data allow, not the likelihood running out of the box, so they are not flagged (Naveh, 2026-10-03)."""
    edge = [p for p, (_, hi) in box.items() if theta_hat.get(p) is not None and theta_hat[p] >= hi * (1 - _EDGE_RTOL)]
    if not edge:
        return []
    where = ", ".join(f"{p} = {theta_hat[p]:.6g} (box upper edge {box[p][1]:g})" for p in edge)
    return [Warning("BOX_EDGE",
                    f"theta-hat is on the prior box's upper edge: {where}. The likelihood kept rising out of the "
                    f"box, so the eta, beta and epsilon intervals end where the box does, not where the data "
                    f"do: they are not usable, even when labelled closed (D-018).",
                    {"parameters": edge, "theta_hat": {p: theta_hat[p] for p in edge},
                     "upper_edges": {p: box[p][1] for p in edge}})]


_AGE0_FEW = (5, 0.01)       # AGE0_DEATHS: at most max(5, 1 % of the deaths) in the first interval (Naveh, 2026-10-04)


def _age0_deaths(ages, events, delta):
    """A few deaths in the first recording interval [0, delta) (spec 002 FR-031). The SR model gives deaths that
    early essentially no probability, so a handful of them can move the whole fit; many of them (infant
    mortality in a life table) are a different matter, which the README covers."""
    if delta is None or (np.ndim(delta) == 0 and not delta):
        return []
    d_row = np.broadcast_to(np.asarray(delta, float), ages.shape)
    delta = float(np.min(d_row))
    died = events == 1
    k = int(np.sum(died & (ages < d_row)))
    n_d = int(np.sum(died))
    if not 1 <= k <= max(_AGE0_FEW[0], _AGE0_FEW[1] * n_d):
        return []
    return [Warning("AGE0_DEATHS",
                    f"{k} death(s) are recorded in the first interval [0, {delta:g}), {k / n_d:.1%} of the "
                    f"deaths. The SR model gives deaths this early essentially no probability, so a few of them "
                    f"can move the whole fit: on the README's ITP example one added age-0 death moved eta-hat "
                    f"about 5-fold and epsilon-hat about 10-fold and made the basin warnings fire. Check those "
                    f"rows; if they are real deaths the model does not describe, refit without them (a cut-off at "
                    f"{delta:g}; state it -- fit-cohort cannot fit a window until spec 006) and compare the fits with "
                    f"and without them (register D-105).",
                    {"deaths_first_interval": k, "deaths": n_d, "delta": float(delta), "cutoff": float(delta)})]


def _floored_deaths(floor_report):
    if not floor_report or not floor_report.get("deaths"):
        return []
    from sraging.floors import message
    return [Warning("FLOORED_DEATHS", message(floor_report), floor_report)]


def _archive_gap(gap_nats, diagnostics):
    """Spec 003 (FR-012, FR-017): a jax2 fit's archive gap -- the best point it evaluated (the profile walks, and the
    combination walks) minus theta-hat. Fires above 1.92 nats unless the recentred profile gain already did."""
    if gap_nats is None or gap_nats <= _GAIN_NATS or _basin_gain(diagnostics):
        return []
    return [Warning("BASIN_GAIN",
                    f"the fit evaluated a point {gap_nats:.1f} nats better than the reported best fit (its archive of "
                    f"evaluated points). theta-hat is the port's, unchanged; the intervals around it may be centred "
                    f"wrongly ([T-JAX-BASIN]).",
                    {"gain_nats": gap_nats, "parameters": [], "source": "archive"})]


def evaluate(cohort, claims, diagnostics, *, theta_hat=None, box=None, floor_report=None, gap_nats=None,
             relabel=None):
    """(warnings, checks) for one fit. The first four checks and AGE0_DEATHS always run; BOX_EDGE runs when the calibrator
    passes its theta-hat and its prior box, FLOORED_DEATHS when it passes the floor report (spec 002;
    sraging.floors) -- otherwise each is recorded as NOT RUN, never as quiet."""
    ages = np.asarray(cohort.ages, float)
    events = np.asarray(cohort.events, int)
    found = {
        "BASIN_GAIN": _basin_gain(diagnostics) + _archive_gap(gap_nats, diagnostics),
        "BASIN_OUTSIDE": _basin_outside(claims),
        "CENSOR_TIE": _censor_tie(ages, events, cohort.delta, relabel),
        "DELTA_GRID": _delta_grid(ages, cohort.delta),
        "AGE0_DEATHS": _age0_deaths(ages, events, cohort.delta),
    }
    ran = {code: True for code in found}
    ran["BOX_EDGE"] = theta_hat is not None and box is not None
    found["BOX_EDGE"] = _box_edge(theta_hat, box) if ran["BOX_EDGE"] else []
    ran["FLOORED_DEATHS"] = floor_report is not None
    found["FLOORED_DEATHS"] = _floored_deaths(floor_report)
    warnings = tuple(w for code in CHECK_CODES for w in found[code])
    checks = tuple(CheckRecord(code, ran[code], bool(found[code])) for code in CHECK_CODES)
    return warnings, checks


# ------------------------------------------------------------------------------------------- spec 003f's warnings
def _fmt3(x):
    return "?" if x is None else f"{x:.3g}"


#: spec 006 (Naveh 2026-11-15, sr-naveh S7): profile_v033's check runs since spec 006, but it failed its validation
V033_LEVERAGE_CAVEAT = "profile_v033's check failed its validation, spec 006 R-9"


def leverage_warnings(block, caveat: str | None = None) -> tuple:
    """`LEVERAGE` when the check flagged, could not assess the fit (`not_assessable`; spec 003g's `not_assessed`),
    failed, or did not run (spec 003f FR-008; constitution V: none of these is a pass); nothing for `no_flag` and
    `flat_valley_slide`. `caveat`, when given, closes a flag's message (profile_v033's: V033_LEVERAGE_CAVEAT)."""
    if not block or block.get("label") in ("no_flag", "flat_valley_slide"):
        return ()
    lab = block.get("label")
    if lab == "flag":
        d, r = block.get("dropped") or {}, block.get("ratios") or {}
        kind = "death" if d.get("event") == 1 else "censored"
        rr = lambda q: _fmt3((r.get(q) or {}).get("ratio"))  # noqa: E731
        msg = (f"the estimate rests on one individual: without row {d.get('row')} (age {d.get('age'):g}, {kind}) "
               f"eta/eps moves {rr('eta_over_eps')}x, rbar/eta {rr('rbar_over_eta')}x, eta {rr('eta')}x and eps "
               f"{rr('epsilon')}x, and the refit gains {_fmt3(block.get('delta_total'))} nats: read before using "
               f"these numbers") + ("" if caveat is None else f" ({caveat})")
    elif lab == "not_assessed":                       # spec 003g FR-011 (Naveh Da; DV-3)
        gi = (block.get("at_theta_hat") or {}).get("free_gradient_per_individual")
        msg = (f"the leverage check could not assess this fit: the optimiser could not move from theta-hat (gradient "
               f"{_fmt3(gi)} per individual), so a single influential death would not be seen")
    elif lab == "not_assessable":
        msg = f"the leverage check could not assess this fit: {block.get('reason')}"
    elif lab == "failed":
        msg = f"the leverage check's refit failed: {block.get('reason')}"
    else:
        msg = f"the leverage check did not run: {block.get('reason')}"
    data = {k: block.get(k) for k in ("label", "dropped", "R", "delta_total", "reason") if block.get(k) is not None}
    return (Warning("LEVERAGE", msg, data),)


def h_not_usable(version: str = "1.0.0") -> Warning | None:
    """On every profile_jax2_h 1.0.0 fit (spec 003f FR-011); None at 1.0.1, which is usable with the
    HAZARD_SHARP_LIMIT note (spec 003b FR-011: keyed on the version, the call site passes the calibrator's)."""
    if version != "1.0.0":
        return None
    return Warning("H_NOT_USABLE", "profile_jax2_h 1.0.0 is not usable: the jax2 session reports it not validated on "
                                   "library v5; 1.0.1 arrives with spec 003b", {"version": "1.0.0"})


#: spec 003b FR-011 (research R12): every profile_jax2_h 1.0.1 result carries this note in diagnostics["notes"]. A
#: note, not a warning (constitution V: a warning that fires on every fit of a name is not a check), so a clean fit
#: stays clean. Its numbers are jax2's defect register REV 6 (DEFECT_REGISTER.txt sha256 20ba3bc9...): 56 sharp hazard
#: cohorts EXACT on v6; the size measured on 3 of them (v5 data: +1.57, +1.18, +13.64 nats at the truth); beyond the
#: cut 1 MEASURED + 69 INFERRED on v6. Spec 003g (T046, T066; jax2's W2 witness files): of those 69, 36 are valid on v6.
#: The 34 human ones (all valid) were scored on a reference grid (W_REF): none more than 1.92 nats below the truth, max
#: -0.366. The other 35 (2 valid) have only W4, on the fit's own grid, where a grid error common to the truth and
#: theta-hat cancels, so W4 cannot see this defect. The 56 have no v6 _h record in any window.
#: Spec 003g N2 (Naveh 2026-10-29): what was measured where DT_RULE fires (CV_c < 0.05), on the 227 witnessed valid
#: sharp cohorts of SRK-1 (library v6, W3), printed on every profile_jax2 1.0.1 fit in that regime. The numbers are
#: checked against tests/acceptance/results/score_w3_003g.json by tests/acceptance/test_results_003g.py.
SHARP_COHORT_LIMIT = {
    "code": "SHARP_COHORT_LIMIT",
    "message": "measured on 227 benchmark cohorts in this regime (library v6; spec 003g): closed eta intervals "
               "covered the truth on 132 of 133 pure and 43 of 45 censored cohorts, epsilon on 156 of 157 and 45 of "
               "47; eta had no upper end on 49 of 227 (about 1 in 5; each of those lower bounds covered); median closed "
               "widths are 1.5-1.7 dex for eta and 2.4-2.5 dex for epsilon; raw beta was closed on 6 of 227 and its "
               "upper end is the kappa box, so removal is not determined: read r-bar/eta, which is not truth-scored "
               "on 1.0.1. Below CV_c 0.01 only 2 cohorts were witnessed, and no censored cohort below 0.018",
    "data": {"n_cohorts": 227, "n_pure": 171, "n_censor": 56,
             "eta_closed_covered": {"pure": [132, 133], "censor": [43, 45]},
             "epsilon_closed_covered": {"pure": [156, 157], "censor": [45, 47]},
             "eta_open_above": {"pure": 38, "censor": 11}, "eta_lower_bounds_covered": [49, 49],
             "median_closed_dex": {"eta": {"pure": 1.5, "censor": 1.67}, "epsilon": {"pure": 2.42, "censor": 2.47}},
             "beta_closed": {"pure": 4, "censor": 2}, "below_cv_c_0_01_witnessed": 2, "censor_min_cv_c": 0.018,
             "source": "tests/acceptance/results/score_w3_003g.json (SRK-1/outside), census_003b.json (CV_c)"},
}

HAZARD_SHARP_LIMIT = {
    "code": "HAZARD_SHARP_LIMIT",
    "message": "profile_jax2_h 1.0.1 under-resolves sharp-SR hazard cohorts: up to 13.6 nats at the truth where "
               "measured, on 3 of the 56 registered sharp hazard cohorts and on library v5 data only (none of the 56 "
               "has a v6 _h record). Of the 69 cohorts the register INFERRED, 36 are valid on v6: on the 34 human ones "
               "jax2's witness scored the truth on a reference grid, none more than 1.92 nats below it; the other 35 "
               "were scored only on the fit's own grid, which cannot see this defect ([T-H-HAZARD-SHARP]; jax2 defect "
               "register REV 6; spec 003g)",
    "data": {"defect": "T-H-HAZARD-SHARP", "max_measured_nats": 13.64, "registered_v6": 56, "measured": 3,
             "measured_on": "library v5 data only", "inferred_v6": 69, "register": "REV 6",
             "v6_inferred_valid": 36, "v6_inferred_ref_grid": 34, "v6_inferred_own_grid_only": 35,
             "v6_inferred_over_1_92": 0, "v6_inferred_ref_grid_w_max": -0.36643059199835987,
             "v6_source": "tests/acceptance/results/score_w2_003g.json (jax2's W2 witness, export_v6/W2)",
             "register_sha256": "20ba3bc9bba5c71e53630a2dd1a828abff8445c08b6250a984dc5ef631d01a94"},
}



# ------------------------------------------------------------------------------------------- spec 004's warnings
#: Spec 004 (tasks T008; contracts/python-api.md §2, §5, §6; data-model.md): the heterogeneous mixture's, the Langevin
#: engine's and the engine switch's. Each is a warning, not a note: in a result's warnings it makes the result not clean
#: (a MixtureResult, a LangevinSample and a Population are clean iff they carry none). None is one of the seven fit
#: checks (CHECK_CODES). `sampler_warning` fills a template from its data.
SAMPLER_WARNINGS = {
    "HET_UNCONVERGED": "the mixture did not converge by M = {M_check}: |ℓ({M}) − ℓ({M_check})| = {abs_dll:.3g} nats "
                       "≥ {threshold:g}",
    "HET_NOT_CHECKED": "M = {M} with check=False: the mixture's convergence at this M was not checked, so the value "
                       "is not clean (M=\"auto\", or a checked fixed M, checks it)",
    "HET_OUTSIDE_MEASURED_RANGE": "sd = {sd:g} is outside the range spec 004 measured (sd {lo:g}-{hi:g}, its "
                                  "acceptance panel): the checked M and the stated cost are not validated there",
    "LANGEVIN_STEP_BIAS_NOT_MEASURED": "the step bias at dt = {dt:g} is NOT MEASURED (error=\"ks\"): the KS test at "
                                       "n = {n:,} does not see the bias of the step, so this sample is not clean (the "
                                       "default error=\"ladder\" measures it)",
    "LANGEVIN_INCONSISTENT_AT_N": "the sample is inconsistent with the FP reference at n = {n:,}: KS D = {ks_D:.4g} > "
                                  "crit_95 = {crit_95:.4g} (p = {p_value:.3g})",
    # Naveh 2026-11-05 (T141, sr-naveh_004.md finding 1): at small n the KS has no power to see the step's bias.
    "LANGEVIN_STEP_BIAS_ABOVE_CRIT": "the stated step bias b̂(dt) = {bias_at_dt:.4g} at dt = {dt:g} exceeds the KS "
                                     "crit_95 = {crit_95:.4g} at n = {n:,}, so this sample is not clean whatever the "
                                     "KS reads: halve dt, or use wall=\"reflect\" for a population near the wall",
    "LANGEVIN_SURVIVORS_AT_TMAX": "{n_survived:,} of {n:,} individuals were alive at t_max = {t_max:g}; they are "
                                  "censored there",
    "LANGEVIN_NO_FP_REFERENCE": "no FP reference: {reason}. The sample's agreement with the model is not measured, so "
                                "it is not clean",
    "LANGEVIN_PATHS_INCONSISTENT_AT_N": "the damage paths at age {age:g} are inconsistent with the FP damage density "
                                        "among the n_alive = {n_alive:,} living ({wall}, {statistic} statistic): KS "
                                        "D = {ks_D:.4g} > crit = {crit:.4g}",
    "LANGEVIN_PATHS_WALL_ATOM": "under wall=\"clamp\", a fraction {p0:.3g} of the living have damage exactly 0 at age "
                                "{age:g}, above crit/10 = {limit:.3g}: damage below x_c = {x_c:.3g} is not checked "
                                "there (wall=\"reflect\" has no lump)",
    "ENGINE_AUTO_LANGEVIN": "engine=\"auto\" drew this sample with the Langevin engine, not FP or gen-1: {reason}",
}


def sampler_warning(code: str, **data) -> Warning:
    """One of spec 004's warnings, its message filled from `data`, which it keeps whole. Refused: an unknown code, or a
    template field missing from `data` (named)."""
    if code not in SAMPLER_WARNINGS:
        raise ValueError(f"unknown sampler warning {code!r}; one of {sorted(SAMPLER_WARNINGS)}")
    try:
        message = SAMPLER_WARNINGS[code].format(**data)
    except KeyError as ex:
        raise ValueError(f"{code} needs its data field {ex.args[0]!r}") from None
    return Warning(code, message, dict(data))


# ------------------------------------------------------------------------------------------- spec 006's warnings
#: Spec 006 (research R7, R8; data-model.md "Warnings"): the windowed fit's. WINDOW_S_LO_CLIPPED: W's or H's S(lo) at
#: theta-hat is below the frozen code's clip (fp_jax.py:850), so the conditioning term there is the clip's (FR-010).
#: ENGINE_S_LO_CLIPPED: the engine's numerics "v1" clipped S(t_lo) (D-108 narrowed, Naveh's Db). W_COVERAGE: W's
#: measured coverage on v6's window arm missed R-1's cut while its truth witness passed (Naveh's Dd, FR-021).
#: W_LARGE_N: W's windowed fit at n >= 25,000, where A2 found more than 10 % of its closed claims truncated above
#: (Naveh 2026-11-11, A2's rule; tasks T067); worded on 2026-11-15 (Naveh, sr-naveh M1): the variable is the life
#: table, and 25,000 is the panel's split.
WINDOW_WARNINGS = {
    "WINDOW_S_LO_CLIPPED": "S(lo) at theta-hat is {S_lo_min:.3g}, below the frozen code's clip of 1e-12 at lo = {lo:g} "
                           "({calibrator}): the conditioning on survival to lo is the clip's there, not the model's, "
                           "so the windowed fit does not describe the cohort's survival to lo",
    "ENGINE_S_LO_CLIPPED": "the engine clipped S(t_lo) at 1e-12 (smallest {S_lo_min:.3g}) in numerics \"{numerics}\": "
                           "the windowed value there is the clip's, not the model's (D-108)",
    "W_COVERAGE": "profile_jax2_w's 95 % intervals on windowed data missed spec 006's coverage cut ({cut:.3f}) on v6's "
                  "window arm: {detail}. Treat its windowed intervals as narrower than calibrated",
    "W_LARGE_N": "profile_jax2_w on windowed data at n = {n} (n >= {n_min}): on spec 006's A2 (44 cohorts planted at "
                 "human life tables, n >= {n_min}) the upper ends of its closed 95 % intervals were not set by the "
                 "data on {detail}. The same holds below {n_min} on cohorts planted at real life tables ({below}): "
                 "{n_min} is the panel's split, not a measured change. An end at the box would not move with more "
                 "data, and a widened end ('edge not found') can sit below the truth. Read each claim's label, and "
                 "read such an upper end as 'at least this', not as the interval's end",
}


def window_warning(code: str, **data) -> Warning:
    """One of spec 006's warnings, its message filled from `data`, which it keeps whole. Refused: an unknown code, or a
    template field missing from `data` (named)."""
    if code not in WINDOW_WARNINGS:
        raise ValueError(f"unknown window warning {code!r}; one of {sorted(WINDOW_WARNINGS)}")
    try:
        message = WINDOW_WARNINGS[code].format(**data)
    except KeyError as ex:
        raise ValueError(f"{code} needs its data field {ex.args[0]!r}") from None
    return Warning(code, message, dict(data))
