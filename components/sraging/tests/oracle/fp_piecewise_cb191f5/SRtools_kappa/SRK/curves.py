"""One way to get a model curve, from either engine: ``SRK.curves``.

    from SRK import curves
    cs = curves.get_curves(eta=..., beta=..., epsilon=..., kappa=..., t_end=110)   # FP, ~60 ms
    cs.plot_survival(); print(cs.median, cs.steepness)

Until now "plot the model" meant running a Langevin cohort, because plotting is a method on the
simulated object (`Dataset -> SR -> SR_lf -> SR_Hetro`). This module separates the CURVE from the
ENGINE that produced it, so the exact FP solver becomes usable for everything the simulator was used
for, and the simulator is reserved for what genuinely needs individual trajectories.

WHAT ONLY THE FP ENGINE MAKES AFFORDABLE
    `curves_band_from_posterior` -- a credible band on the model curve, from n posterior draws.
    200 FP solves ~ 30 s; the same band by simulation is ~27 min, which is why it has never existed.
    Without it you cannot tell whether a data-model gap is larger than the posterior's own width.

ENGINE SELECTION follows `SRK.config` unless you pass ``engine=`` explicitly, so a project moves to
the FP default with one line (``config.use_profile('modern')``) rather than an edit sweep.

WHAT STAYS ON THE SIMULATOR (a density solver cannot produce these):
individual trajectories X_i(t); occupation-time statistics (healthspan/time-above-threshold);
simultaneous heterogeneity in more than one parameter; per-individual heterogeneity draws; and, until
the piecewise-FP work lands, windowed/periodic interventions and the joint (disease, death)
distribution. `engine='auto'` routes to the simulator exactly when one of those is requested.
"""
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from . import config as _config
from . import fp_curves as _fpc
from . import stepsize as _ss
from . import theta_utils as _thu
from .fp_curves import (CurveSet, auto_L, disease_onset_curves, fp_curves, fp_survival_hazard,
                        moments_from_survival)
from .stepsize import safe_time_step_multiplier
from .theta_utils import CANONICAL_ORDER

__all__ = ["CurveSet", "Theta", "get_curves", "from_dataset", "sim_curves", "fp_curves",
           "fp_survival_hazard", "auto_L", "disease_onset_curves", "moments_from_survival",
           "safe_time_step_multiplier", "plot_fit", "curves_from_posterior",
           "curves_band_from_posterior", "seed_simulator", "CANONICAL_ORDER"]

_PARAMS = ("eta", "beta", "epsilon", "kappa")


# --------------------------------------------------------------------------- theta
class Theta:
    """A labelled SR parameter vector, so no call site has to guess an array's layout.

    A bare 5-element array is ambiguous -- ``[eta, beta, epsilon, xc, kappa]`` (presets, the
    simulator) or ``[eta, beta, epsilon, kappa, ExtH]`` (`sr_mcmc.inv_transform`, the fit) -- and
    picking wrong swaps kappa with xc, which yields a plausible-looking wrong curve rather than an
    error. Build one of these from a labelled source and the question never arises.
    """

    __slots__ = ("eta", "beta", "epsilon", "kappa", "xc", "external_hazard")

    def __init__(self, eta, beta, epsilon, kappa, xc=1.0, external_hazard=None):
        self.eta, self.beta, self.epsilon = float(eta), float(beta), float(epsilon)
        self.kappa, self.xc = float(kappa), float(xc)
        self.external_hazard = None if external_hazard is None else float(external_hazard)

    # -- constructors from labelled sources
    @classmethod
    def from_sim_vector(cls, theta, external_hazard=None):
        """``[eta, beta, epsilon, xc, kappa]`` (+ ExtH) -- the `getSrHetro` / `presets` layout."""
        t = np.asarray(theta, float)
        kappa = _thu.kappa_from_theta(t, source="Theta.from_sim_vector", warn=False)
        eh = external_hazard if external_hazard is not None else (t[5] if t.size > 5 else None)
        return cls(t[0], t[1], t[2], kappa, xc=t[3], external_hazard=eh)

    @classmethod
    def from_fit_vector(cls, theta, xc=1.0):
        """``[eta, beta, epsilon, kappa]`` (+ ExtH) -- what `inv_transform`/`load_posterior` give."""
        t = np.asarray(theta, float)
        if t.size not in (4, 5):
            raise ValueError(f"a fit vector has 4 or 5 elements; got {t.size}")
        return cls(t[0], t[1], t[2], t[3], xc=xc,
                   external_hazard=(t[4] if t.size == 5 else None))

    @classmethod
    def from_preset(cls, name, type="max_likelihood", **kw):
        from . import presets                                  # lazy: presets reads CSVs
        ext = kw.pop("ExtH", False)
        vec = presets.getTheta(preset_name=name, type=type, ExtH=ext, **kw)
        return cls.from_sim_vector(vec)

    @classmethod
    def coerce(cls, obj, order=None, xc=1.0):
        """Accept a Theta, a mapping, or an array whose layout is named by ``order``.

        ``order=None`` consults ``config['theta_order']``: 'strict' refuses an unlabelled 5-vector
        (it is genuinely ambiguous), 'canonical' reads it as :data:`CANONICAL_ORDER`.
        """
        if isinstance(obj, cls):
            return obj
        if hasattr(obj, "keys"):
            d = dict(obj)
            return cls(d["eta"], d["beta"], d["epsilon"], d["kappa"],
                       xc=d.get("xc", xc), external_hazard=d.get("external_hazard"))
        t = np.asarray(obj, float)
        if order in ("sim", "canonical"):
            return cls.from_sim_vector(t)
        if order == "fit":
            return cls.from_fit_vector(t, xc=xc)
        if order is not None:
            raise ValueError(f"order must be 'sim'/'canonical', 'fit' or None; got {order!r}")
        if t.size == 4:
            return cls.from_fit_vector(t, xc=xc)               # unambiguous
        if t.size == 6:
            return cls.from_sim_vector(t)                      # unambiguous
        mode = _config.get("theta_order")
        if mode == "canonical":
            return cls.from_sim_vector(t)
        raise ValueError(
            f"a {t.size}-element theta is ambiguous: [eta, beta, epsilon, xc, kappa] (the simulator "
            f"and presets) or [eta, beta, epsilon, kappa, ExtH] (inv_transform / load_posterior)? "
            f"Pass order='sim' or order='fit', build a Theta from a labelled source "
            f"(Theta.from_preset / from_fit_vector), or set "
            f"SRK.config.set(theta_order='canonical') to read it as {CANONICAL_ORDER[:5]}.")

    # -- conversions
    def to_sim_vector(self, with_ext=False):
        v = [self.eta, self.beta, self.epsilon, self.xc, self.kappa]
        if with_ext:
            v.append(np.inf if self.external_hazard is None else self.external_hazard)
        return np.asarray(v, float)

    def to_fit_vector(self):
        return np.asarray([self.eta, self.beta, self.epsilon, self.kappa], float)

    def as_dict(self):
        return {k: getattr(self, k) for k in ("eta", "beta", "epsilon", "kappa", "xc",
                                              "external_hazard")}

    def replace(self, **kw):
        d = self.as_dict()
        d.update(kw)
        return Theta(**d)

    def scaled(self, **factors):
        """Multiplicative change on named parameters -- the chronic-intervention operation."""
        d = self.as_dict()
        for k, f in factors.items():
            if k not in d:
                raise KeyError(f"unknown parameter {k!r}")
            d[k] = d[k] * float(f)
        return Theta(**d)

    def __repr__(self):
        eh = "" if self.external_hazard is None else f", ExtH={self.external_hazard:.4g}"
        return (f"Theta(eta={self.eta:.6g}, beta={self.beta:.6g}, epsilon={self.epsilon:.6g}, "
                f"kappa={self.kappa:.6g}, xc={self.xc:g}{eh})")

    def __eq__(self, other):
        return isinstance(other, Theta) and self.as_dict() == other.as_dict()


# --------------------------------------------------------------------------- simulator seeding
def seed_simulator(seed):
    """Make the Langevin engine reproducible.

    ``np.random.seed()`` does NOT reach the jitted kernels -- numba keeps its own RNG state, so two
    identical `getSrHetro` calls after seeding give different death times (measured). Seeding must
    happen inside an ``@njit`` function. Also requires ``parallel=False``: the joblib workers re-seed
    nondeterministically under spawn and identically under fork, and identical streams across chunks
    are worse than no seeding at all.
    """
    from numba import njit                                     # lazy: numba costs ~1 s to import

    global _numba_seed
    if _numba_seed is None:
        @njit(cache=False)
        def _s(x):
            np.random.seed(x)
        _numba_seed = _s
    _numba_seed(int(seed))
    np.random.seed(int(seed))


_numba_seed = None


# --------------------------------------------------------------------------- engines
def from_dataset(ds, t_grid=None, time_range=None, engine="data"):
    """Wrap a `Dataset` (real data, or a finished simulation) as a :class:`CurveSet`.

    With ``t_grid=None`` the curve IS the dataset's own Kaplan-Meier timeline, so ``survival`` is
    bit-identical to ``ds.getSurvival()[1]``; hazard is the Nelson-Aalen smoothed hazard interpolated
    onto that timeline. ``native`` is kept, so plotting delegates to the dataset's own methods by
    default and lifelines confidence bands survive.
    """
    t_km, S_km = (np.asarray(a, float) for a in ds.getSurvival())
    t_h, h_km = (np.asarray(a, float) for a in ds.getHazard())
    t = t_km if t_grid is None else np.asarray(t_grid, float)
    S = S_km if t_grid is None else np.interp(t, t_km, S_km)
    h = np.interp(t, t_h, h_km)
    meta = {"profile": _config.active_profile(), "engine_detail": engine,
            "time_range": None, "drawstyle": "steps-post", "warnings": []}
    cs = CurveSet(t=t, survival=S, hazard=h, pdf=None, engine=engine, meta=meta,
                  t_full=t, S_full=S, p_full=None, h_full=h, native=ds)
    return cs.trim(time_range) if time_range is not None else cs


def sim_curves(theta=None, *, t_end, nsteps=6000, npeople=25000, time_step_multiplier=1,
               hetro=False, het_var=0.2, het_param="xc", external_hazard=None, method="brownian_bridge",
               parallel=False, seed=None, t_grid=None, time_range=None,
               resolution=_config.INHERIT, sim=None, **theta_kw):
    """A :class:`CurveSet` from the Langevin simulator (needed when trajectories matter).

    ``resolution`` (see `SRK.config`): ``None`` leaves ``(nsteps, time_step_multiplier)`` exactly as
    given -- today's behaviour; ``'rule'``/``'auto'`` refine the substep count using
    `SRK.stepsize` and can only make it finer, never coarser. The step size shipped in a preset was
    tuned around that organism's SEED theta, so replaying a FITTED theta at it is the documented way
    to get a badly wrong curve.
    """
    th = Theta.coerce(theta, **{}) if theta is not None else Theta(**theta_kw)
    if external_hazard is None:
        external_hazard = th.external_hazard
    resolution = _config.resolve("resolution", resolution)

    tsm = int(time_step_multiplier)
    notes, report = [], None
    if resolution is not None:
        opts = dict(resolution) if isinstance(resolution, dict) else {}
        mode = opts.pop("mode", None) or (resolution if isinstance(resolution, str) else "auto")
        if mode not in ("rule", "auto"):
            raise ValueError(f"resolution must be None, 'rule', 'auto' or a dict; got {resolution!r}")
        common = dict(t_end=t_end, nsteps=nsteps, xc=th.xc, base_tsm=tsm)
        if mode == "rule":
            # L1 only: the timescale rule, no solver and no simulation. Microseconds.
            nsteps, tsm = _ss.required_resolution(
                th.eta, th.beta, th.epsilon, th.kappa, **common, **opts)
        else:
            # L2: measure agreement against the exact FP curve and let that decide.
            nsteps, tsm, report = _ss.auto_resolution(
                th.eta, th.beta, th.epsilon, th.kappa, **common, npeople=npeople,
                hetro=hetro, het_var=het_var, het_param=het_param,
                external_hazard=external_hazard, **opts)
        if tsm != int(time_step_multiplier):
            notes.append(f"time_step_multiplier {int(time_step_multiplier)} -> {tsm}")
        # Warn only for verdicts that signal a PROBLEM. 'rule_only' means the caller passed
        # verify=False, i.e. they chose to skip the measurement -- warning about it on every call
        # would be noise, and noisy warnings are how real ones get filtered out.
        if report is not None and report.status in ("floor", "capped", "refused_cost"):
            # Do not silently hand back a curve the chooser does not stand behind. The caller asked
            # for a simulation, so still run one -- at the affordable/best resolution -- but make the
            # verdict impossible to miss and attach the full report.
            import warnings as _w
            _w.warn(f"simulated curve is not verified as converged (status={report.status!r}).\n"
                    f"{report}", RuntimeWarning, stacklevel=2)
            notes.append(f"resolution status={report.status}")

    if sim is None:
        from .SR_hetro import getSrHetro                       # lazy: pulls numba kernels
        if seed is not None:
            seed_simulator(seed)
            parallel = False                                   # parallel breaks the seeding
        kw = {f"{het_param}_var": het_var} if hetro else {}
        sim = getSrHetro(th.to_sim_vector(), npeople=npeople, nsteps=nsteps, t_end=t_end,
                         time_step_multiplier=tsm, external_hazard=external_hazard,
                         hetro=hetro, parallel=parallel, method=method, **kw)

    cs = from_dataset(sim, t_grid=t_grid, engine="sim")
    cs.meta.update({"params": th.as_dict(), "hetro": bool(hetro),
                    "het_param": het_param if hetro else None,
                    "het_var": het_var if hetro else None,
                    "external_hazard": external_hazard, "seed": seed,
                    "engine_detail": (f"langevin/{method}/n={npeople}/nsteps={nsteps}/tsm={tsm}"),
                    "nsteps": nsteps, "time_step_multiplier": tsm, "npeople": npeople,
                    "resolution": resolution, "resolution_report": report, "warnings": notes})
    if report is not None:
        # Also on the simulation object itself, so `presets.getSim(..., resolution='auto')` can
        # expose `sim.resolution_report` without the caller going through a CurveSet.
        try:
            sim.resolution_report = report
        except AttributeError:
            pass
    return cs.trim(time_range) if time_range is not None else cs


def get_curves(theta=None, *, engine=_config.INHERIT, t_end=None, needs=(), **kw):
    """Model curves for a theta, from whichever engine the request needs.

    engine='fp'   exact Fokker-Planck (no simulation).
    engine='sim'  Langevin cohort.
    engine='auto' 'sim' only if ``needs`` asks for something a density solver cannot produce
                  ('trajectories', 'damage', 'occupation_time', 'windowed_intervention',
                  'multi_het', 'joint_disease'); otherwise 'fp'.
    Omitted: taken from `SRK.config` (profile 'legacy' -> 'sim', 'modern' -> 'fp').
    """
    engine = _config.resolve("curve_engine", engine)
    needs = {needs} if isinstance(needs, str) else set(needs)
    sim_only = {"trajectories", "damage", "occupation_time", "windowed_intervention",
                "multi_het", "joint_disease"}

    if engine == "auto":
        engine = "sim" if (needs & sim_only) else "fp"
    elif engine == "fp" and (needs & sim_only):
        raise ValueError(
            f"engine='fp' cannot provide {sorted(needs & sim_only)}: the FP solver evolves a density, "
            f"so it has no individual realisations, no occupation-time functionals and (until the "
            f"piecewise-FP evaluator lands) no time-dependent parameters. Use engine='sim'.")

    th = Theta.coerce(theta) if theta is not None else None
    if engine == "sim":
        if t_end is None:
            raise ValueError("engine='sim' needs t_end (the simulator has no infinite horizon)")
        return sim_curves(th, t_end=t_end, **kw)
    if engine == "fp":
        if th is None:
            cs = fp_curves(t_end=t_end, **kw)
        else:
            kw.setdefault("external_hazard", th.external_hazard)
            cs = fp_curves(eta=th.eta, beta=th.beta, epsilon=th.epsilon, kappa=th.kappa,
                           xc=th.xc, t_end=t_end, **kw)
        cs.meta.setdefault("profile", _config.active_profile())
        return cs
    raise ValueError(f"engine must be 'fp', 'sim' or 'auto'; got {engine!r}")


# --------------------------------------------------------------------------- posterior curves
def _physical_thetas(post, n=None, which=None, set_params=None, rng=None):
    """Physical [eta, beta, epsilon, kappa(, ExtH)] rows from a Posterior or a raw array."""
    if isinstance(post, np.ndarray) or (hasattr(post, "__len__") and not hasattr(post, "samples")):
        arr = np.atleast_2d(np.asarray(post, float))           # already physical (load_posterior)
        if n is not None and n < len(arr):
            rng = np.random.default_rng(0) if rng is None else rng
            arr = arr[rng.choice(len(arr), size=n, replace=False)]
        return arr
    from . import sr_mcmc as srmc                              # lazy
    sp = {"xc": 1.0} if set_params is None else set_params      # inv_transform requires xc here
    if which == "best":
        rows = [post.best_raw_sample()]
    elif which == "mode":
        rows = [post.get_best_sample_in_mode()]
    else:
        samples = np.atleast_2d(np.asarray(post.samples, float))
        rng = np.random.default_rng(0) if rng is None else rng
        k = min(n or len(samples), len(samples))
        rows = samples[rng.choice(len(samples), size=k, replace=False)]
    return np.atleast_2d([srmc.inv_transform(r, set_params=sp) for r in rows])


def curves_from_posterior(post, which="best", ds=None, engine=_config.INHERIT, set_params=None,
                          **kw):
    """Curves for a posterior's best-fit or modal theta.

    Replaces the four hand-written cells the analysis notebooks use (``best_raw_sample`` ->
    ``inv_transform(set_params={'xc': 1})`` -> insert xc=1 -> ``getSrHetro`` -> plot), and inherits
    ``L``, ``time_range`` and ``external_hazard`` from ``ds`` so the plotted curve sits on the same
    footing as the likelihood that produced the theta.
    """
    row = _physical_thetas(post, which=which, set_params=set_params)[0]
    th = Theta.from_fit_vector(row)
    if ds is not None:
        kw.setdefault("data", ds)
    return get_curves(th, engine=engine, **kw)


def curves_band_from_posterior(post, n=200, q=(0.16, 0.5, 0.84), ds=None, seed=0,
                               set_params=None, **kw):
    """Per-age posterior quantiles of S(t) and h(t) -- a credible band on the model curve.

    THE capability the FP engine unlocks: 200 solves take ~30 s, where 200 Langevin cohorts take
    ~27 min. Without a band you cannot tell whether a data-model gap exceeds the posterior's own
    width, which is exactly the question a hazard misfit raises.

    Returns
    -------
    dict with ``t``, ``survival`` and ``hazard`` arrays of shape (len(q), len(t)), ``quantiles``,
    ``n_used``, and ``thetas`` (the draws actually solved).
    """
    rng = np.random.default_rng(seed)
    rows = _physical_thetas(post, n=n, set_params=set_params, rng=rng)
    if ds is not None:
        kw.setdefault("data", ds)
    kw.setdefault("engine", "fp")

    S_list, h_list, t_ref, used = [], [], None, []
    for row in rows:
        th = Theta.from_fit_vector(row)
        cs = get_curves(th, **kw)
        if t_ref is None:
            t_ref = cs.t
        S_list.append(np.interp(t_ref, cs.t, cs.survival))
        h_list.append(np.interp(t_ref, cs.t, cs.hazard))
        used.append(th.to_fit_vector())
    qs = np.asarray(q, float)
    return {"t": t_ref,
            "survival": np.quantile(np.vstack(S_list), qs, axis=0),
            "hazard": np.quantile(np.vstack(h_list), qs, axis=0),
            "quantiles": qs, "n_used": len(used), "thetas": np.vstack(used)}


# --------------------------------------------------------------------------- plotting
def plot_fit(curves, data=None, axes=None, time_range=None, panels=("survival", "hazard"),
             labels=None, time_unit="", band=None, logy_hazard=True):
    """Data-vs-model overlay: the two panels the analysis notebooks build by hand.

    ``time_range`` is applied to SURVIVAL only. `Dataset.plotSurvival` clips and renormalises to 1 at
    the window start, so a model curve plotted as raw unconditional survival next to it is on an
    inconsistent baseline -- which once looked exactly like a bad fit. Hazard needs no such treatment
    (it is invariant under conditioning), and `plotHazard` is never called with a window anywhere in
    the pipeline.
    """
    import matplotlib.pyplot as plt                            # lazy

    cs_list = [curves] if isinstance(curves, CurveSet) else list(curves)
    if axes is None:
        _fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.5))
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        for i, cs in enumerate(cs_list):
            lbl = labels[i] if labels else f"{cs.engine} fit"
            if panel == "survival":
                (cs.trim(time_range) if time_range is not None else cs).plot_survival(
                    ax=ax, native=False, label=lbl)
            elif panel == "hazard":
                cs.plot_hazard(ax=ax, native=False, logy=logy_hazard, label=lbl)
            else:
                raise ValueError(f"unknown panel {panel!r}")
        if band is not None and panel in ("survival", "hazard"):
            y = band[panel]
            if len(y) >= 3:
                ax.fill_between(band["t"], y[0], y[-1], alpha=0.20, lw=0,
                                label=f"{int(100*(band['quantiles'][-1]-band['quantiles'][0]))}% CI")
        if data is not None:
            if panel == "survival":
                data.plotSurvival(ax=ax, time_range=time_range, label="data", linestyle="--")
            else:
                data.plotHazard(ax=ax, label="data", linestyle="--")
        ax.set_xlabel(f"Age [{time_unit}]" if time_unit else "Age")
        ax.set_title(f"{panel.capitalize()} ({cs_list[0].meta.get('engine_detail', cs_list[0].engine)})")
        ax.legend()
    return axes[0].figure, axes
