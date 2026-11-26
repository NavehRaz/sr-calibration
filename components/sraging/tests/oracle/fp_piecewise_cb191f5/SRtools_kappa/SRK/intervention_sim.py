"""Windowed interventions: the Langevin evaluator behind the FP framework's documented seam.

`SRK.intervention.metrics` already defines `Intervention(param, effect, window)` and already refuses
a windowed one:

    fp_const supports only chronic (whole-life) interventions; got windowed ...
    Time-dependent interventions need a 'piecewise_fp'/'sim' evaluator (future work).

The simulator has implemented exactly that case for years (`SRK.intervention_SR`), so the gap was
never capability -- it was that the two halves did not meet. This module is the join: it maps an
`Intervention` onto `getISR`'s per-parameter argument blocks, runs the cohort at an automatically
chosen resolution, and returns THE SAME SIX KEYS as `fp_metrics`, so every downstream consumer
(`param_plane`, `best_pair`, the per-condition tables) works unchanged.

WHY IT LIVES HERE AND NOT IN `SRK/intervention/`
A concurrent session owns that package right now. Rather than edit it, this module carries a small
registry plus mirrors that emit the same DataFrame schemas. When the package is free, `fp_metrics`
can grow a three-line lookup into `EVALUATORS` and these mirrors become thin aliases; nothing about
the call sites changes either way. Monkeypatching was rejected: `param_plane` does
`from .metrics import fp_metrics` at import time, so patching the name would silently not reach it.

THE MAPPING IS FULL OF SHARP EDGES -- all of these are real, and all of them fail silently:
  * ``window=None`` must become ``[0, t_end]``. getISR's default duration is ``[0, 0]`` and the
    effect functions return 0 outside the window, so a "chronic" intervention passed naively is
    active nowhere and you get the baseline cohort back.
  * ``intervention_time`` is ignored for a 2-element duration (`_process_duration`), so always pass
    explicit ``[start, stop]`` pairs and never rely on it.
  * a rebound (`*_exit_alpha`) is SILENTLY IGNORED unless the method name contains ``_exit``
    (`intervention_SR.py:49`), so the method is auto-upgraded here.
  * ``time_dep_value > 100`` makes ``(1 - p/100)`` negative and its non-integer power NaN inside a
    nopython kernel -- rejected up front.
  * ``'A'`` (the internal ageing-clock rate) is SIM-ONLY. Chronic A is exactly equivalent to chronic
    eta, but WINDOWED A is not: it permanently shifts the accumulated clock, so its effect persists
    after the window closes. It is the one direction here with memory, and it must never be handed
    to `posterior_ratio` (it has no posterior).
  * external hazard is an additive LEVEL (rate ``exp(-level)``), not a multiplicative factor, so it
    gets its own :class:`ExtHazard` spec rather than being forced into `Intervention`.
"""
import warnings

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from . import curves as _curves

# `SRK.intervention.metrics` is clean and uses only absolute imports, but importing it pulls the
# package __init__, which a concurrent session may have mid-edit. Fall back to loading the file
# directly (the pattern fp_curvefit/fp_gof already use for fp_likelihood) so this module cannot be
# taken down by someone else's unfinished commit.
try:
    from .intervention.metrics import (Intervention, PARAM_IDX, PARAM_NAMES, SWEEP_NAMES,
                                       apply_chronic, data_metrics, fp_metrics, split_xc)
except Exception as _exc:                                        # pragma: no cover - safety net
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location(
        "_srk_intervention_metrics",
        _os.path.join(_os.path.dirname(__file__), "intervention", "metrics.py"))
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    Intervention, PARAM_IDX, PARAM_NAMES = _m.Intervention, _m.PARAM_IDX, _m.PARAM_NAMES
    SWEEP_NAMES, apply_chronic, split_xc = _m.SWEEP_NAMES, _m.apply_chronic, _m.split_xc
    fp_metrics, data_metrics = _m.fp_metrics, _m.data_metrics
    warnings.warn(f"loaded intervention.metrics by path ({_exc})", RuntimeWarning)

__all__ = ["ExtHazard", "SIM_SWEEP_NAMES", "EVALUATORS", "register_evaluator", "sr_metrics",
           "sim_metrics", "sim_scan", "to_getisr_kwargs", "worst_case_theta",
           "theta_from_posterior", "intervention_lines_sim", "Intervention"]

# 'A' scales the ageing clock and exists only in the simulator; see the module docstring.
SIM_SWEEP_NAMES = tuple(SWEEP_NAMES) + ("A",)

# Intervention.param -> the prefix getISR uses for that parameter's argument block. Note the capital
# X: the kwargs are Xc_effect / Xc_duration / ..., while the metrics API spells the parameter 'xc'.
_BLOCK = {"eta": "eta", "beta": "beta", "epsilon": "epsilon", "kappa": "kappa",
          "xc": "Xc", "A": "A"}


class ExtHazard:
    """A windowed external-hazard LEVEL (the hazard rate is ``exp(-level)``).

    Deliberately not an `Intervention`: that class means ``param *= (1 + effect)``, while
    `exth_level` is an absolute override of the constant background hazard. Forcing one into the
    other would make ``effect=0`` mean "hazard rate 1" instead of "no change".
    """

    __slots__ = ("level", "window")

    def __init__(self, level, window=None):
        self.level = float(level)
        self.window = None if window is None else (float(window[0]), float(window[1]))

    def is_chronic(self):
        return self.window is None

    def __repr__(self):
        return f"ExtHazard(level={self.level:g}, window={self.window})"


def _is_intervention(obj):
    """Duck-typed: the fallback loader above can yield a second, distinct Intervention class."""
    return hasattr(obj, "param") and hasattr(obj, "effect") and hasattr(obj, "window")


# --------------------------------------------------------------------------- mapping
def to_getisr_kwargs(interventions, t_end, method="brownian_bridge", decays=None, rebounds=None):
    """Translate `Intervention` / `ExtHazard` specs into `getISR` keyword arguments.

    Returns ``(kwargs, method)`` -- the method may be upgraded to its ``_exit`` variant, because a
    rebound amplitude is otherwise ignored without any error.

    ``decays`` / ``rebounds`` are optional ``{param: value}`` / ``{param: (alpha, tau)}`` maps for the
    within-window decay and the post-window rebound, which have no slot in the `Intervention` spec.
    """
    kw, needs_exit = {}, False
    seen = set()
    for iv in interventions:
        if isinstance(iv, ExtHazard):
            kw["exth_level"] = iv.level
            kw["exth_duration"] = [0.0, float(t_end)] if iv.is_chronic() else list(iv.window)
            continue
        if not _is_intervention(iv):
            raise TypeError(f"expected Intervention or ExtHazard, got {type(iv).__name__}")
        if iv.param not in _BLOCK:
            raise KeyError(f"unknown intervention parameter {iv.param!r}; "
                           f"expected one of {SIM_SWEEP_NAMES}")
        if iv.param in seen:
            raise ValueError(f"two interventions on {iv.param!r}: getISR has ONE argument block per "
                             f"parameter, so the second would silently overwrite the first")
        seen.add(iv.param)
        b = _BLOCK[iv.param]
        kw[f"{b}_effect"] = float(iv.effect)
        # window=None means CHRONIC, which in getISR's language is the whole run. The default [0, 0]
        # would switch the effect off everywhere except exactly t=0.
        kw[f"{b}_duration"] = [0.0, float(t_end)] if iv.is_chronic() else list(iv.window)

        dv = float((decays or {}).get(iv.param, 0.0))
        if not 0.0 <= dv <= 100.0:
            raise ValueError(
                f"{iv.param}: time_dep_value must lie in [0, 100] (it is a percent-per-unit-time "
                f"decay); {dv} makes (1 - p/100) negative and its fractional power NaN inside the "
                f"jitted kernel, with no error raised")
        kw[f"{b}_time_dep_value"] = dv

        alpha, tau = (rebounds or {}).get(iv.param, (0.0, 1.0))
        if float(alpha) != 0.0:
            needs_exit = True
            if float(tau) <= 0:
                raise ValueError(f"{iv.param}: exit_tau must be > 0 (tau <= 0 makes the rebound a "
                                 f"permanent -alpha offset rather than a decaying one)")
        kw[f"{b}_exit_alpha"] = float(alpha)
        kw[f"{b}_exit_tau"] = float(tau)

    if needs_exit and "_exit" not in method:
        if "euler" in method:
            raise ValueError("a rebound (exit_alpha) needs a Brownian-bridge engine: the euler "
                             "kernels have no _exit variant. Use method='brownian_bridge'.")
        method = method + "_exit"
    return kw, method


def worst_case_theta(theta, interventions, decays=None, rebounds=None):
    """The stiffest parameter set the run will pass through, for sizing the time step.

    The resolution must survive the whole timeline, not just the baseline: an intervention that
    scales beta up mid-run needs a finer substep than the unintervened cohort does. Bounds use each
    effect's PEAK magnitude, which is conservative for decaying and windowed forms since those only
    reduce |effect| inside the window.
    """
    th = _curves.Theta.coerce(theta)
    peak = {}
    for iv in interventions:
        if isinstance(iv, ExtHazard) or not _is_intervention(iv):
            continue
        a = abs(float((rebounds or {}).get(iv.param, (0.0, 1.0))[0]))
        peak[iv.param] = float(iv.effect) + a
    return th.replace(
        beta=th.beta * max(1.0, 1.0 + peak.get("beta", 0.0)),
        epsilon=th.epsilon * max(1.0, 1.0 + peak.get("epsilon", 0.0)),
        kappa=th.kappa * max(1e-6, min(1.0, 1.0 + peak.get("kappa", 0.0))),
        eta=th.eta * max(1.0, 1.0 + max(peak.get("eta", 0.0), peak.get("A", 0.0))),
    )


# --------------------------------------------------------------------------- evaluators
def sim_metrics(theta, interventions=(), xc=1.0, t_end=None, n=25000, nsteps=6000,
                resolution="auto", method="brownian_bridge", hetro=False, het_var=0.2,
                external_hazard=np.inf, seed=None, parallel=False, steepness="inverseCV",
                decays=None, rebounds=None, replicates=1, return_sim=False, return_curves=False,
                **kw):
    """Longevity + steepness for an SR theta under interventions, by simulation.

    Returns the same keys as `fp_metrics` -- ``median, mean, cv, steepness, tail, absorbed`` -- so
    the two evaluators are interchangeable, plus ``n_died``, ``mc_se_*`` and ``resolution``.

    The metrics are computed from the CURVE (integrated survival), not from the Dataset's own
    estimators, because `Dataset.getMeanLifetime`/`getCV` use died-only individuals while the FP
    moments cover the whole first-passage distribution. Mixing the two definitions would make a
    cross-engine comparison meaningless.
    """
    from .intervention_SR import getISR                          # lazy: numba kernels

    th = _curves.Theta.coerce(theta).replace(xc=xc)
    if t_end is None:
        raise ValueError("sim_metrics needs t_end: the simulator has no infinite horizon "
                         "(fp_metrics does, which is why it does not take one)")
    kwargs, method = to_getisr_kwargs(interventions, t_end, method=method, decays=decays,
                                      rebounds=rebounds)

    # A window shorter than a few grid steps cannot be represented at all: the kernels evaluate every
    # intervention effect at the OUTER grid time t[j], outside the substep loop, so refining
    # time_step_multiplier improves the SDE and does nothing for the timeline. This floor guards that
    # structural failure.
    #
    # HONEST SCOPE: the constant (20 points per window) is a rule of thumb, NOT calibrated. The
    # obvious candidate measurement does not support one: mice_F with beta x0.4 on [0, 60] reads
    # KS 0.0181 at nsteps=5000 and 0.0066 at 20000 against the exact piecewise-FP curve, which looks
    # like a window effect but is not -- at n=12000 the finite-sample floor is 0.0124, so what is
    # being revealed there is ordinary dt discretisation error that a bigger sample made visible.
    # Those windows (60 units, ~1400 grid points) were never edge-limited. Calibrating this would
    # need a sweep over window LENGTH at fixed dt.
    from . import stepsize as _ss
    _wins = [iv.window for iv in interventions
             if _is_intervention(iv) and not iv.is_chronic()]
    _floor_n = _ss.window_nsteps_floor(t_end, _wins)
    if _floor_n > nsteps:
        nsteps = int(_floor_n)

    # Size the step on the worst case the timeline reaches, not on the baseline theta.
    res_opts = resolution
    if resolution == "auto":
        wc = worst_case_theta(th, interventions, decays=decays, rebounds=rebounds)
        nsteps, tsm, report = _ss.auto_resolution(
            wc.eta, wc.beta, wc.epsilon, wc.kappa, t_end=t_end, nsteps=nsteps, xc=wc.xc,
            base_tsm=kw.pop("time_step_multiplier", 1), npeople=n, hetro=hetro, het_var=het_var)
        res_opts = report
    else:
        tsm = kw.pop("time_step_multiplier", 1)
        report = None

    out, sims = [], []
    for r in range(max(1, int(replicates))):
        if seed is not None:
            _curves.seed_simulator(int(seed) + r)
            parallel = False
        sim = getISR(th.to_sim_vector(), n=n, nsteps=nsteps, t_end=t_end,
                     time_step_multiplier=tsm, hetro=hetro, xc_var=het_var,
                     external_hazard=external_hazard, method=method, parallel=parallel,
                     **kwargs, **kw)
        cs = _curves.from_dataset(sim, engine="sim")
        m = cs.metrics(steepness=steepness)
        m["n_died"] = int(np.asarray(sim.events, int).sum())
        out.append(m)
        sims.append((sim, cs))

    keys = ("median", "mean", "cv", "steepness", "tail", "median_sensitivity")
    res = {k: float(np.mean([o[k] for o in out])) for k in keys}
    if len(out) > 1:
        for k in ("median", "steepness"):
            res[f"mc_se_{k}"] = float(np.std([o[k] for o in out], ddof=1) / np.sqrt(len(out)))
    res["absorbed"] = bool(res["tail"] < 1e-2)
    res["n_died"] = int(np.mean([o["n_died"] for o in out]))
    res["resolution"] = report
    res["evaluator"] = "sim"
    res["nsteps"] = int(nsteps)
    res["window_nsteps_floor"] = int(_floor_n)
    if return_sim:
        res["sim"] = sims[0][0]
    if return_curves:
        res["curves"] = sims[0][1]
    return res


def _fp_const(theta, interventions, **kw):
    kw.pop("t_end", None)
    for drop in ("n", "nsteps", "resolution", "method", "seed", "parallel", "replicates",
                 "decays", "rebounds", "external_hazard", "hetro", "het_var", "return_sim",
                 "return_curves"):
        kw.pop(drop, None)
    return fp_metrics(_curves.Theta.coerce(theta).to_fit_vector(), interventions, **kw)


EVALUATORS = {"fp_const": _fp_const, "sim": sim_metrics}


def register_evaluator(name, fn):
    """Add an evaluator (e.g. the piecewise-FP prototype) to the dispatch table."""
    EVALUATORS[str(name)] = fn
    return fn


def sr_metrics(theta, interventions=(), evaluator="fp_const", **kw):
    """Dispatch to an evaluator by name -- the single entry point.

    ``'fp_const'`` delegates to the unchanged `fp_metrics` (chronic only, analytic);
    ``'sim'`` handles windowed / decaying / periodic cases via the Langevin engine;
    ``'piecewise_fp'`` appears once `SRK.fp_piecewise` is imported.
    """
    try:
        fn = EVALUATORS[evaluator]
    except KeyError:
        raise NotImplementedError(
            f"unknown evaluator {evaluator!r}; available: {sorted(EVALUATORS)}") from None
    return fn(theta, interventions, **kw)


# --------------------------------------------------------------------------- consumers
def theta_from_posterior(thetas, how="median", xc=1.0, seed=0):
    """``(M, 4)`` physical rows from `intervention.io.load_posterior` -> one `Theta`.

    Closes the worst footgun in this area: those rows are ``[eta, beta, epsilon, kappa]``, and handed
    straight to a simulation factory the kappa lands in ``xc`` while kappa itself reverts to a
    default -- two errors, no message.
    """
    arr = np.atleast_2d(np.asarray(thetas, float))
    if arr.shape[1] < 4:
        raise ValueError(f"expected (M, 4) physical posterior rows; got shape {arr.shape}")
    if how == "median":
        row = np.median(arr[:, :4], axis=0)
    elif how == "mean":
        row = np.mean(arr[:, :4], axis=0)
    elif how == "draw":
        row = arr[np.random.default_rng(seed).integers(len(arr))][:4]
    else:
        raise ValueError("how must be 'median', 'mean' or 'draw'")
    return _curves.Theta.from_fit_vector(row, xc=xc)


def intervention_lines_sim(theta, params=SIM_SWEEP_NAMES, effects=(-0.5, -0.25, 0.25, 0.5),
                           window=None, t_end=None, xc=1.0, **kw):
    """Fractional-change effect lines in the steepness-longevity plane, via the sim evaluator.

    Emits the same columns as `intervention.param_plane.intervention_lines`
    (``param, effect, median, steepness, d_median, d_steepness``) so `plot_param_plane` consumes it
    unchanged. With ``window`` set, these are the WINDOWED lines the FP evaluator cannot produce.
    """
    import pandas as pd

    th = _curves.Theta.coerce(theta).replace(xc=xc)
    base = sim_metrics(th, (), t_end=t_end, **kw)
    rows = []
    for p in params:
        for e in effects:
            m = sim_metrics(th, [Intervention(p, float(e), window)], t_end=t_end, **kw)
            rows.append({"param": p, "effect": float(e),
                         "median": m["median"], "steepness": m["steepness"],
                         "d_median": m["median"] / base["median"] - 1.0,
                         "d_steepness": m["steepness"] / base["steepness"] - 1.0})
    return pd.DataFrame(rows)


def sim_scan(control_theta, cond_dt, cond_ev, params=SIM_SWEEP_NAMES,
             effects=(0.5, 0.75, 1.5, 2.0), window=None, t_end=None, xc=1.0,
             n=5000, nsteps=6000, seed=0, min_margin_se=3.0, **kw):
    """Which parameter change best explains a condition -- the windowed analogue of `fp_scan`.

    Scores with the same KDE likelihood the Langevin pipeline uses. Unlike `fp_scan` this is a NOISY
    objective, so it reports the Monte-Carlo standard error and REFUSES to name a winner whose margin
    is below ``min_margin_se`` standard errors -- the project has previously decided contrasts inside
    that noise.
    """
    import pandas as pd

    from . import distance_metrics as dm
    from . import deathTimesDataSet as dtds

    obs = dtds.Dataset(np.asarray(cond_dt, float), np.asarray(cond_ev, int))
    th = _curves.Theta.coerce(control_theta).replace(xc=xc)

    def _score(ivs, rep_seed):
        r = sim_metrics(th, ivs, t_end=t_end, n=n, nsteps=nsteps, seed=rep_seed,
                        return_sim=True, **kw)
        return float(dm.baysianDistance(obs, r["sim"])), r

    base_ll, _ = _score((), seed)
    rows = []
    for p in params:
        for e in effects:
            ivs = [Intervention(p, float(e) - 1.0, window)]
            lls = [_score(ivs, seed + k)[0] for k in range(2)]
            rows.append({"param": p, "effect": float(e), "ll": float(np.mean(lls)),
                         "ll_se": float(np.std(lls, ddof=1) / np.sqrt(len(lls))),
                         "ll_gain": float(np.mean(lls)) - base_ll})
    df = pd.DataFrame(rows).sort_values("ll_gain", ascending=False).reset_index(drop=True)

    best = None
    if len(df) >= 2:
        margin = df.loc[0, "ll_gain"] - df.loc[1, "ll_gain"]
        noise = min_margin_se * max(df.loc[0, "ll_se"], 1e-12)
        best = (f"{df.loc[0, 'param']} x{df.loc[0, 'effect']:g}" if margin > noise else None)
    return {"baseline_ll": base_ll, "singles": df, "best_single": best,
            "margin_ok": best is not None}
