"""Fit ONE real cohort from a CSV, by the recommended protocol. The entry point that was missing.

A usability review put it plainly: the benchmark documents are a rigorous comparison of methods and
contain **zero lines a new user can paste into a terminal**. `mle_profile` -- the recommended method --
takes an internal benchmark `Cohort` object with `.death_times`, `.events` and `.spec.delta`, which is
described nowhere; `coverage.py` and `emcee_production.py` only run over the synthetic library and have
no `--data` flag; and the one CLI that does accept a real cohort
(`Baysian03_kappa/run_file_mcmc_excel.py`) runs the configuration the recommendation explicitly warns
against. So there was no path from "I have a CSV of death times" to "I ran the recommended fit".

This is that path.

    python -m SRK.fit_cohort --csv my_flies.csv --delta 1.0
    python -m SRK.fit_cohort --csv Mice_F_ds.csv --delta 1.0 --json out.json

INPUT. A CSV with a header and two columns, `death times,events` -- the format the existing datasets
use (`Baysian03_kappa/datasets/Mice_F_ds.csv`), which was itself undocumented. `events` is 1 for an
observed death and 0 for a right-censored record. One row per individual.

`--delta` IS NOT OPTIONAL IN PRACTICE, and getting it wrong is the most expensive mistake available
here. It is your RECORDING RESOLUTION in the same time unit as the ages in the CSV: 1.0 if ages are
whole days, 0.5 for half-days, 182.625 for half-years. Omitting it fits interval-censored data with a
density, which was measured to cost 0.15-0.20 of coverage on cohorts with a rounded grid. If your ages
really are continuous, pass `--delta 0` and say so.

WHAT IT REPORTS, and why it is not just a parameter table:
  * `eta` and `epsilon` as intervals -- these are the two the data determines.
  * `beta` and `kappa` as whatever the profile actually supports: a closed interval, a one-sided
    bound, or NOT DETERMINED. A decline is the measurement, not a failure; it is not retried.
  * the width of every interval as a fraction of its prior box, with the one-division rule attached,
    because an interval that fills its box is the prior handed back.
  * the negligibility floor for this cohort's n, so "beta is indistinguishable from zero" is
    distinguishable from "beta is small".

WHAT IT DOES NOT DO. It does not run the MCMC (that needs a cluster; see `emcee_production`), it does
not produce cap probabilities, and it does not validate the observation model -- if your cohort has an
external hazard, a truncated window or a heterogeneous sub-population, NOTHING in this benchmark tests
that and the intervals below are optimistic. That is a scope limit, not a caveat.
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import os
import sys

import numpy as np


def read_csv(path):
    """`(ages, events)` from a two-column CSV. Tolerant of column order and header spelling."""
    ages, events = [], []
    with open(path, newline="") as fh:
        rd = _csv.reader(fh)
        head = next(rd, None)
        icol, ecol = 0, 1
        if head:
            low = [h.strip().lower() for h in head]
            # FIRST match wins, and the two columns may not collide. Without the break, a header like
            # `time,death` set icol=0 on "time" and then OVERWROTE it to 1 on "death", so the event
            # column was read as ages and the fit silently ran on 1/1/0. Same for `death_day,
            # death_event` and `age,age_at_censor`.
            icol = ecol = None
            for i, h in enumerate(low):
                if icol is None and ("death" in h or "time" in h or "age" in h):
                    icol = i
            for i, h in enumerate(low):
                if ecol is None and i != icol and ("event" in h or "status" in h
                                                   or "censor" in h):
                    ecol = i
            icol = 0 if icol is None else icol
            ecol = (1 if icol != 1 else 0) if ecol is None else ecol
            if icol == ecol:
                raise SystemExit(f"{path}: could not tell the age column from the event column "
                                 f"(header {head}). Rename them, e.g. 'death times,events'.")
            # a header row that is actually data (no letters) must not be discarded
            if all(not any(c.isalpha() for c in h) for h in head):
                rd = [head] + list(rd)
        for row in rd:
            if not row or len(row) <= max(icol, ecol):
                continue
            try:
                a = float(row[icol])
                e = float(row[ecol])
            except ValueError:
                continue
            if not np.isfinite(a) or a <= 0:
                continue          # a literal 'nan' cell passes float() and would enter death_times
            ages.append(a)
            events.append(1 if e else 0)
    if not ages:
        raise SystemExit(f"{path}: no usable rows. Expected a header plus two columns, "
                         f"'death times,events'.")
    return np.asarray(ages, float), np.asarray(events, int)


class _Spec:
    """The three attributes `mle_profile` reaches for on `cohort.spec`."""

    def __init__(self, delta, name):
        self.delta = delta
        self.name = name


class _Cohort:
    """The duck type `mle_profile` expects: `.death_times`, `.events`, `.spec.delta`.

    Written out explicitly because "construct the cohort object" was the exact step at which a reader
    of the protocol got stuck, and the required shape appears in no document.
    """

    def __init__(self, ages, events, delta, name="cohort"):
        self.death_times = np.asarray(ages, float)
        self.events = np.asarray(events, int)
        self.spec = _Spec(delta if delta else None, name)


def fit(ages, events, delta=None, name="cohort", levels=(0.95,), n_profile=11, kappa_max=None):
    """Run the recommended fit. `kappa_max` MUST be applied before `methods` is imported.

    `methods.py` evaluates `KAPPA_MAX = os.environ.get("SRK_KAPPA_MAX", 300)` and builds `BOX` at
    IMPORT time, and `BOX` is what bounds the profile walk. An earlier version of this function set the
    environment variable AFTER importing `methods`, so `--kappa-max` was a silent no-op: the box stayed
    at its default, the user's chosen prior never reached the fit, and the printed box disagreed with
    the flag. That flag is in the one command the protocol asks people to paste.
    """
    if kappa_max:
        os.environ["SRK_KAPPA_MAX"] = str(float(kappa_max))
    if "SRK.benchmark.methods" in sys.modules:      # a prior import would have frozen the old box
        import importlib
        importlib.reload(sys.modules["SRK.benchmark.methods"])
    from .benchmark import methods as M
    if kappa_max and abs(M.BOX["kappa"][1] - float(kappa_max)) > 1e-9:
        raise SystemExit(f"--kappa-max {kappa_max:g} did not take effect (box is "
                         f"{M.BOX['kappa']}). Refusing to fit under a prior you did not choose.")
    co = _Cohort(ages, events, delta, name)

    # SOLVER-STEP GUARD. `auto_grid` sizes dt from the LIFESPAN scale (dt = max(2e-3,
    # tau_max/1200)), which knows nothing about the recording resolution. When delta/L falls below
    # one dt, the interval likelihood log(S(t) - S(t+delta)) is being asked for a difference across
    # LESS THAN ONE STEP of the solver -- a subtraction of two nearly identical interpolated values.
    #
    # MEASURED on the 19 benchmark cohorts where this happens (delta/L from 9.5e-5 to 1.3e-3, i.e.
    # 4.7% to 64% of a step), comparing the loglik at the fitted theta under the default dt against
    # dt = (delta/L)/3:
    #
    #     median |change| 0.066 nats  --  negligible on 18 of 19
    #     max     |change| 1.711 nats --  `real_ecoli_s4`, which is 89% of the chi2_1(0.95)/2 = 1.921
    #                                     nat threshold that SETS an interval edge
    #
    # and the sign varies cohort to cohort, which is the signature of numerical error rather than a
    # bias. One cohort in nineteen can therefore have an interval edge moved by most of a threshold
    # for a purely numerical reason. So the ceiling is applied automatically, but ONLY in the
    # regime that needs it -- outside it, dt is unchanged and results stay comparable with every
    # run already computed.
    _restore = os.environ.get("SRK_DT_CEILING")
    _applied = None
    if delta:
        try:
            _L = float(M.kaplan_meier(co.death_times, co.events)[2])
            _tm, _dt = M._fpl.auto_grid(co.death_times, _L)
            _dl = float(delta) / _L
            if _dl > 0 and _dl < _dt:
                _applied = _dl / 3.0
                os.environ["SRK_DT_CEILING"] = repr(_applied)
                print(f"[fit_cohort] delta/L = {_dl:.3g} is below one solver step ({_dt:.3g}); "
                      f"capping dt at {_applied:.3g} (= (delta/L)/3). Slower, but the interval "
                      f"likelihood is no longer a sub-step difference.")
        except Exception as _ex:
            print(f"[fit_cohort] warning: could not check the solver step ({_ex}); "
                  f"proceeding at the default dt")
    try:
        res = M.mle_profile(co, levels=tuple(levels), n_profile=int(n_profile))
    finally:
        if _applied is not None:
            if _restore is None:
                os.environ.pop("SRK_DT_CEILING", None)
            else:
                os.environ["SRK_DT_CEILING"] = _restore
    if _applied is not None:
        res = dict(res) if isinstance(res, dict) else res
        try:
            res["dt_ceiling"] = _applied
        except Exception:
            pass
    return res, co


def _floor(n, param, eta=None, L=None):
    """The negligibility floor in PHYSICAL units: below it, "indistinguishable from zero" is honest.

    THE CONVENTION, and getting it wrong is easy. From the calibration in `negligible.py`, KL is
    exactly quadratic in the perturbation so detectability scales as 1/sqrt(n) and the floors are
    closed-form -- but they are floors on DIFFERENT things:

        beta_neg / eta   = 3.03  / sqrt(n_eff)      -> beta's floor is a RATIO TO ETA
        kappa_neg / xc   = 0.364 / sqrt(n_eff)      -> kappa is dimensionless, so this is direct

    and **`eta` and `epsilon` have no floor at all** -- there is no "negligible eta", since eta is the
    damage production rate that makes the cohort die in the first place.

    An earlier version of this function returned `3.03/sqrt(n)` for BOTH `beta` and `eta` and compared
    it against the raw physical value. That flagged `eta = 0.0033` as "negligible" against a floor of
    0.152 on a real cohort -- a confident, wrong, user-facing verdict on the one parameter the data
    determines best. Caught by running the tool on real data and reading its own output.
    """
    # ONE implementation, in `space_search`, next to the coefficients it uses. Three copies of this
    # arithmetic existed and two of them were wrong by a factor of L in opposite directions.
    from .benchmark.space_search import beta_floor_physical, kappa_floor
    if param == "kappa":
        return kappa_floor(n)
    if param == "beta":
        if eta is None or not np.isfinite(eta) or eta <= 0:
            return float("nan")
        # THE FLOOR IS ON THE DIMENSIONLESS RATIO rho_beta/rho_eta = beta/(L*eta), which is what
        # `negligible.py` calibrated (its sweep ran at tau-mean = 1, i.e. L = 1) and what
        # `report.py` and `beta_slice.py` both use. beta/eta has units of TIME, so comparing it to a
        # dimensionless 3.03/sqrt(n) is only correct at L = 1. Omitting L made the floor 926x too
        # strict on a mouse cohort (L = 926 d) and the "negligible" verdict effectively unreachable
        # on any real cohort. Same bug class as the eta floor caught earlier, one parameter over.
        return beta_floor_physical(n, eta, L)
    return float("nan")           # eta, epsilon: no negligibility convention exists


def _delta_audit(ages, delta):
    """Is the `--delta` you passed actually the grid your ages sit on?

    Omitting `--delta` already warns. Passing the WRONG one did not: it produces a plausible, wrong
    answer with no diagnostic, which is the failure mode this project keeps paying for. Weekly census
    data handed `--delta 1.0` is fitted with an interval likelihood a factor of seven too narrow.

    Returns (verdict, message). `verdict` is "ok", "off-grid" or "coarser".
    """
    a = np.asarray(ages, float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 3:
        return "ok", ""
    u = np.unique(a)
    # the grid the DATA appears to sit on: the smallest positive gap between distinct ages
    g = float(np.min(np.diff(u))) if u.size > 1 else 0.0
    if delta:
        d = float(delta)
        r = a / d
        off = float(np.max(np.abs(r - np.round(r))))
        if off > 1e-6 * max(1.0, float(np.max(r))):
            return ("off-grid",
                    f"the ages are NOT multiples of --delta {d:g} (worst residual {off:.3g} of a "
                    f"step). The interval likelihood assumes ages were rounded DOWN onto that grid. "
                    f"Smallest gap between distinct ages is {g:.6g}"
                    + (f", so --delta {g:.6g} may be what you meant." if g > 0 else "."))
        if g > 0 and g > 1.5 * d:
            return ("coarser",
                    f"--delta {d:g} was passed, but the smallest gap between distinct ages is "
                    f"{g:.6g} -- {g / d:.1f}x coarser. If the cohort was censused every {g:.6g}, "
                    f"pass that instead: too small a delta narrows the interval likelihood and the "
                    f"intervals with it.")
        return "ok", ""
    if g > 0 and u.size < a.size:
        return ("coarser",
                f"--delta was NOT passed, but the ages are not distinct (smallest gap {g:.6g} over "
                f"{u.size} distinct values in {a.size} individuals), so they look ROUNDED. "
                f"Pass --delta {g:.6g}.")
    return "ok", ""


def report(res, co, level=0.95, kappa_max=None):
    from .benchmark import methods as M
    # the SAME L the fitter used: the Kaplan-Meier mean lifetime, not the median of the ages
    try:
        _L = float(M.kaplan_meier(co.death_times, co.events)[2])
    except Exception:
        _L = 1.0
    P4 = ("eta", "beta", "epsilon", "kappa")
    th = res.get("theta") or []
    ci = res.get("ci") or {}
    L = []
    A = L.append
    n = int(co.death_times.size)
    d = co.spec.delta
    # WHICH SRK IS THIS? `pip install srtools-aging` can resolve to a checkout that is not the one
    # you edited -- on this machine it resolves to a cluster mount whose copy lacked the negligibility
    # -floor fix entirely. A fit run against the wrong copy is indistinguishable from a fit run
    # against the right one unless the tool says which it was.
    A(f"CODE: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    A(f"COHORT {co.spec.name}: n = {n}, "
      + (f"ages recorded to {d:g} (delta/median-lifetime = {d / float(np.median(co.death_times)):.2e})"
         if d else "ages treated as EXACT (delta not set)"))
    A(f"  observed deaths {int(np.sum(co.events))}, censored {n - int(np.sum(co.events))}, "
      f"median age {float(np.median(co.death_times)):.4g}")
    _dv, _dm = _delta_audit(co.death_times, d)
    if _dv != "ok":
        A(f"  ** CHECK --delta: {_dm}")
    _fb = _floor(n, "beta", eta=(th[0] if th else None), L=_L)
    from .benchmark.space_search import beta_floor_regime_band
    _bl, _bh = beta_floor_regime_band(n)
    A(f"  negligibility floors at this n: beta {(_fb if np.isfinite(_fb) else float('nan')):.4g} "
      f"(= 3.03/sqrt(n_eff) x eta x L, L={_L:.4g}), kappa {_floor(n, 'kappa'):.4g}. `eta` and "
      f"`epsilon` have no floor -- there is no 'negligible eta'.")
    if np.isfinite(_fb) and th:
        # THE FLOOR IS NOT A THRESHOLD. Its coefficient was fitted across four regimes whose own
        # values span 0.93 to >16.8, so any verdict within that band is unresolved by the
        # calibration rather than decided by it. Printing the band stops a reader treating a
        # near-floor beta as a classification.
        _rlo, _rhi = _bl * float(th[0]) * _L, _bh * float(th[0]) * _L
        A(f"  ** the beta floor is UNCERTAIN BY REGIME: the calibrated coefficient ranges 0.93-16.8")
        A(f"     across the four swept regimes against the 3.03 used here, so on this cohort the")
        A(f"     floor could be anywhere in [{_rlo:.4g}, {_rhi:.4g}]. A 'negligible' verdict inside")
        A(f"     that band is NOT resolved by the calibration. Your beta estimate is "
          f"{float(th[1]):.4g}: "
          + ("SAFELY negligible (below the whole band)." if float(th[1]) < _rlo else
             "SAFELY detectable (above the whole band)." if float(th[1]) > _rhi else
             "INSIDE the band -- report it as a bound and say the negligibility call is uncertain."))
    # THE REMOVAL TIMESCALE, in lifespans: rho_beta/rho_eta = beta/(eta*L).
    #
    # This is printed because the verdict column below and the performance map used to state DIFFERENT
    # tests for beta and the tool computed only one of them. `width/box` says whether the interval is
    # narrow; the map's rule is "report beta as a value only where removal is STRONG (ratio >= 3
    # lifespans), where its coverage reaches 0.948". A reader could not apply the map's rule because
    # nothing here printed the ratio. Both are now shown, and they answer different questions:
    # width/box is about THIS cohort's interval, the ratio is about which REGIME the cohort sits in.
    _ratio = None
    if th and len(th) > 1 and th[0] and th[0] > 0 and np.isfinite(_L) and _L > 0:
        _ratio = float(th[1]) / (float(th[0]) * float(_L))
        _band = ("STRONG (>= 3 lifespans): beta's coverage reaches 0.948 in this regime, so a beta "
                 "VALUE is defensible" if _ratio >= 3.0 else
                 "moderate (1-3 lifespans): report beta as a bound" if _ratio >= 1.0 else
                 "WEAK (< 1 lifespan): report beta as a bound, and expect kappa to be undetermined "
                 "-- kappa's coverage is 0.231 below 0.1 lifespans")
        A(f"  removal timescale beta/(eta*L) = {_ratio:.3g} lifespans -- {_band}")
    A("")
    A("PARAMETER ESTIMATES -- read the VERDICT column, not the point estimate.")
    _lp = f"{100 * float(level):g}%"
    A(f"{'param':<9s}{'point':>13s}{(_lp + ' lower'):>14s}{(_lp + ' upper'):>14s}"
      f"{'width/box':>11s}  verdict")
    box = M.BOX
    out_rows = []
    for i, p in enumerate(P4):
        e = ci.get(f"{p}@{level}")
        lo, hi = (e[0], e[1]) if (e and len(e) > 1) else (None, None)
        fin = lambda v: v is not None and np.isfinite(v)
        blo, bhi = box[p]
        bw = math.log10(bhi / blo)
        if fin(lo) and fin(hi) and lo > 0:
            w = math.log10(hi / lo)
            frac = w / bw
            verdict = ("MEASURED" if frac <= 0.05 else
                       "weakly bounded -- quote as a bound" if frac < 0.5 else
                       "NOT DETERMINED -- do not quote as an interval")
            wtxt = f"{frac:.2f}"
        elif fin(hi):
            verdict = f"UPPER BOUND only: {p} < {hi:.4g}"
            wtxt = "—"
        elif fin(lo):
            verdict = f"LOWER BOUND only: {p} > {lo:.4g} (NO upper cap)"
            wtxt = "—"
        else:
            verdict = "NOT DETERMINED -- the profile never crossed the threshold in the box"
            wtxt = "—"
        fl = _floor(n, p, eta=(th[0] if th else None), L=_L)
        if np.isfinite(fl) and th and i < len(th) and th[i] < fl:
            verdict += f"  [below the {fl:.3g} negligibility floor: report as NEGLIGIBLE]"
        A(f"{p:<9s}{(th[i] if i < len(th) else float('nan')):>13.5g}"
          + (f"{lo:>14.5g}" if fin(lo) else f"{'—':>14s}")
          + (f"{hi:>14.5g}" if fin(hi) else f"{'—':>14s}")
          + f"{wtxt:>11s}  {verdict}")
        out_rows.append(dict(param=p, point=(th[i] if i < len(th) else None),
                             lo=(lo if fin(lo) else None), hi=(hi if fin(hi) else None),
                             width_over_box=(None if wtxt == "—" else frac),
                             verdict=verdict))
    A("")
    A("HOW TO READ width/box (both in decades):  <=0.05 a real measurement  |  ~0.2 a bound  |")
    A("                                          >=0.5 NOT DETERMINED, do not quote as an interval")
    A("")
    A("MANDATORY CAVEATS -- these are measured, not boilerplate:")
    if n >= 5000:
        A(f"  * ** YOUR n IS {n}, AND THE NOMINAL LEVEL IS NOT THE REAL ONE. ** Measured on 2,373")
        A(f"    MLE cohorts, coverage FALLS as n rises: eta 0.894 -> 0.754, beta 0.983 -> 0.817,")
        A(f"    epsilon 0.888 -> 0.726, kappa 0.846 -> 0.584 from n=150 to n=10,000. The interval")
        A(f"    shrinks faster than the error does, because the residual error at large n is BIAS")
        A(f"    that more individuals cannot remove while the interval, built from curvature, keeps")
        A(f"    shrinking as if it were all noise. At n >= 5000 the {_lp} interval above behaves like")
        A(f"    a 75-85% interval. Report it as such, or widen it. Note the point estimates for eta")
        A(f"    and epsilon are BETTER at your n (0.073 and 0.080 dex) -- it is only the interval")
        A(f"    that is optimistic.")
    A("  * A cap or bound reported here at nominal 95% is NOT calibrated. On the benchmark's")
    A("    `real` cohorts -- the ones planted at parameters actually fitted to real data, which is")
    A("    the row that describes you -- it was correct on **0.50 / 0.61 / 0.50 / 0.74** of cohorts")
    A("    for eta / beta / epsilon / kappa. (The pooled 0.64-0.85 often quoted is higher because it")
    A("    averages in extrapolation regimes that must not be pooled with real ones.) Report the")
    A("    bound; call the probability NOMINAL. If you need an honest 95%, pass --levels 0.99 and say")
    A("    which you used -- that substitution is untested.")
    A("  * kappa is not identifiable from a normalised survival curve. If it comes back NOT")
    A("    DETERMINED, that is the expected and correct answer, not a failed fit. Do not retry it,")
    A("    do not fill it in from the MCMC, and do not quote a point estimate for it.")
    A("  * NOTHING in this benchmark tests an external hazard, a truncated fit window, a")
    A("    heterogeneous sub-population or censoring: all 462 benchmark cohorts are pure-SR, full")
    A("    window, homogeneous, uncensored. If your cohort is not, these intervals are optimistic")
    A("    by an unmeasured amount.")
    A("  * Coverage depends on n and on your recording resolution more than on the method: it runs")
    A("    from 0.84 (n=400, exact ages) to 0.14 (n=100,000, ages on a grid). Find the row matching")
    A("    your cohort in the report's fair-comparison section rather than quoting a pooled number.")
    diag = (res.get("diag") or res.get("diagnostics") or {})
    if diag.get("verdicts"):
        A("")
        A(f"profile verdicts (from the fitter): {diag['verdicts']}")
    A("")
    A(f"loglik at the MLE {res.get('loglik', float('nan')):.4f}   "
      f"wall {res.get('wall_s', float('nan')):.1f} s   "
      f"kappa box [{box['kappa'][0]:g}, {box['kappa'][1]:g}]")
    A("")
    A("kappa box: set with --kappa-max. It is a PRIOR, not a data constraint -- restricting it is")
    A("nearly costless in deviance, so whatever you choose is an assumption you are making. The")
    A("benchmark used 300 because its library was built around 27 already-fitted cohorts; that is")
    A("not a recommendation for your cohort. Choose from what is physically plausible for your")
    A("system and report the choice alongside the result.")
    return "\n".join(L), out_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="two columns: 'death times,events'")
    ap.add_argument("--delta", type=float, default=None,
                    help="RECORDING RESOLUTION in the same unit as the ages (1.0 = whole days). "
                         "Pass 0 only if ages are genuinely continuous.")
    ap.add_argument("--name", default=None)
    ap.add_argument("--levels", default="0.95", help="comma-separated, e.g. 0.95 or 0.95,0.99")
    ap.add_argument("--n-profile", type=int, default=11)
    ap.add_argument("--kappa-max", type=float, default=None,
                    help="upper edge of the kappa prior box (a PRIOR: state it in your write-up)")
    ap.add_argument("--json", default=None, help="also write the machine-readable result here")
    a = ap.parse_args(argv)
    ages, events = read_csv(a.csv)
    if a.delta is None:
        print("WARNING: --delta not given. If your ages are rounded onto a grid (whole days, "
              "half-years, ...) omitting it fits interval-censored data with a density, measured to "
              "cost 0.15-0.20 of coverage. Pass --delta 0 to assert the ages are exact.",
              file=sys.stderr)
    levels = tuple(float(x) for x in a.levels.split(",") if x.strip())
    name = a.name or os.path.splitext(os.path.basename(a.csv))[0]
    res, co = fit(ages, events, delta=a.delta, name=name, levels=levels,
                  n_profile=a.n_profile, kappa_max=a.kappa_max)
    txt, rows = report(res, co, level=levels[0], kappa_max=a.kappa_max)
    print(txt)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(cohort=name, n=int(co.death_times.size), delta=co.spec.delta,
                           levels=list(levels), theta=res.get("theta"),
                           loglik=res.get("loglik"), ci=res.get("ci"), params=rows), fh,
                      indent=1, default=float)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
