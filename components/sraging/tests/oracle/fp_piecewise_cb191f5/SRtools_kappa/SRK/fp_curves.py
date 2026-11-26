"""Survival / hazard / density CURVES for an SR theta from the exact Fokker-Planck solver.

This is the plotting-side counterpart of `fp_likelihood`'s fitting-side API: the FP engine has been
used only to FIT since it landed, and nothing in the library would hand you a curve from it. Getting
one meant running a Langevin Monte-Carlo cohort (1.4-8 s, and biased at the shipped step size for
stiff organisms) instead of one deterministic PDE solve (~60 ms, exact).

CANONICAL HOME OF ``fp_survival_hazard``, which until now existed only as a copy-pasted cell in 26
notebooks under ``Baysian03_kappa/analysis_fp/``. The name, the positional signature and the
arithmetic are deliberately IDENTICAL to that cell, so a notebook can replace its ``def`` with an
import and every downstream cell keeps working and keeps producing the same numbers.

SOLVER DEFAULTS ARE NOT RESTATED HERE, ON PURPOSE. The notebook cell calls `fpt_full_density` /
`fp_mixture_curves_physical` with no solver arguments, so it inherits whatever those functions
default to. Restating them (N=900, dt=1e-3, ...) would silently diverge the moment those defaults
change -- and `fp_mixture_curves_physical` does not even have its own; it forwards ``**solver_kw``.
So this module forwards NOTHING unless the caller explicitly asks. Opt-in solver options
(``scheme='bdf2'``, ``absorb_flux``, a coarser grid) go through ``**solver_kw``.

Self-contained by design: numpy + `SRK.fp_likelihood` only. No matplotlib, no lifelines, no numba at
import time, so this module loads on a cluster node, under the `sr_fp_fit_share` importlib shims, and
in a test that must not spend 5 s compiling kernels. `CurveSet`'s plotting methods import matplotlib
lazily, and its `trim` imports `deathTimesDataSet` lazily.
"""
import inspect
from dataclasses import dataclass, field

import numpy as np

if not hasattr(np, "trapz"):                 # numpy >= 2 removed the alias the FP modules use
    np.trapz = np.trapezoid

from . import fp_likelihood as fpl

__all__ = ["fp_survival_hazard", "auto_L", "fp_curves", "disease_onset_curves", "CurveSet",
           "moments_from_survival", "median_sensitivity", "effective_solver_kw"]

# _spread_proxy's signature is narrower than the solver's: passing Gtol or scheme to it raises
# TypeError. Keep the accepted set explicit rather than discovering it at runtime.
_SPREAD_KEYS = ("N", "dt", "tau_max")

_S_FLOOR = 1e-300            # matches the notebook helper's np.clip(S, 1e-300, None)


def effective_solver_kw(**solver_kw):
    """The solver settings a call will ACTUALLY run with: `fpt_full_density`'s own defaults + overrides.

    Read from the signature rather than restated, for two reasons. (1) The defaults differ between
    entry points -- `fpt_full` uses N=3000/tau_max=40/Gtol=1e-9 while `fpt_full_density` uses
    N=900/tau_max=20/Gtol=1e-11 -- and `fp_mixture_curves_physical` has none of its own (it forwards
    ``**solver_kw`` to `fpt_full_density`), so BOTH branches of `fp_survival_hazard` inherit the
    density solver's. Hardcoding one set here would be wrong for the other. (2) The fitting side is
    under active development; a curve should keep matching whatever the likelihood currently uses,
    and the way to stay safe about that is to RECORD the effective values, not to freeze them.

    Every consumer that needs to know the resolution (the short-horizon check, the provenance stamped
    into ``CurveSet.meta``) goes through here, so no number is written down twice.
    """
    eff = {k: v.default for k, v in inspect.signature(fpl.fpt_full_density).parameters.items()
           if v.default is not inspect.Parameter.empty}
    eff.update(solver_kw)
    return eff


def fp_survival_hazard(eta, beta, epsilon, kappa, t_grid, L, hetro, het_var=0.2,
                       external_hazard=None, *, het_param="xc", M=None, xc=1.0, **solver_kw):
    """Exact survival S(t) and hazard(t) on ``t_grid`` from the FP solver (xc fixed=1), no simulation.

    Mirrors `SRK.fp_model.fp_model`'s homogeneous / heterogeneous branches exactly, so a plotted
    curve is the same object the likelihood scored.

    The first nine parameters are positionally identical to the notebook helper this replaces;
    everything added is keyword-only, so ``fp_survival_hazard(eta, beta, eps, kappa, t, L, hetro)``
    behaves exactly as before.

    Parameters
    ----------
    eta, beta, epsilon, kappa : float
        Physical SR parameters.
    t_grid : array-like
        Ages at which to evaluate. Physical units (the same units as ``L``).
    L : float
        Lifetime scale. ``tau = t / L``. Use the data's Kaplan-Meier mean (what the fit used); see
        :func:`auto_L` when there is no data. **L=1 is a trap** for long-lived organisms: the solver
        integrates only to its ``tau_max`` (20 by default), so the tail is silently right-clamped.
    hetro : bool
        Heterogeneous (mixture over ``het_param``) or homogeneous.
    het_var, het_param : float, str
        Heterogeneity strength and which parameter carries it. Defaults match the fit (xc, 0.2).
    external_hazard : float or None
        SRK convention: the constant hazard RATE is ``exp(-external_hazard)``; None / inf = off.
    M : int or None
        Mixture node count. None reproduces the notebook: auto-M from ``_spread_proxy``/``choose_M``
        AT THIS THETA. Note the FIT chose M once at the SEED theta and cached it on
        ``dataSet._fp_M``, so a plotted auto-M can differ from the fitted one -- pass ``M=`` (or
        ``M=ds._fp_M``) to match the fit exactly.
    xc : float
        Death threshold. 1.0 under the kappa convention. Also the hook that makes
        :func:`disease_onset_curves` a two-line wrapper: first passage to a lower threshold is the
        same problem with a different absorbing level.

    Returns
    -------
    (S, hazard) : both shaped like ``t_grid``.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    L = float(L)
    if not np.isfinite(L) or L <= 0:
        raise ValueError(f"L must be finite and positive; got {L!r}")
    tau_eval = t_grid / L

    if hetro:
        if M is None:
            probe = {k: v for k, v in solver_kw.items() if k in _SPREAD_KEYS}
            spread = fpl._spread_proxy(eta, beta, epsilon, xc, kappa, L, het_param, het_var, **probe)
            M = fpl.choose_M(spread)
        S, p = fpl.fp_mixture_curves_physical(eta, beta, epsilon, xc, kappa, L, tau_eval,
                                              het_param=het_param, het_var=het_var, M=M,
                                              **solver_kw)
    else:
        # to_dimensionless returns (rho_beta, rho_eta, rho_eps, kappa) -- BETA FIRST. fpt_full takes
        # rho_eta first. Getting this backwards produces a plausible-looking wrong curve.
        rho_beta, rho_eta, rho_eps, kappa_dim = fpl.to_dimensionless(
            eta, beta, epsilon, xc, L, k=kappa)
        tau, S_full, p_full, *_ = fpl.fpt_full_density(rho_eta, rho_beta, rho_eps, kappa_dim,
                                                       **solver_kw)
        S, p = fpl._interp_S_p(tau_eval, tau, S_full, p_full)

    p_phys = p / L
    if external_hazard is not None and np.isfinite(external_hazard):
        h_rate = np.exp(-external_hazard)
        p_phys = (p_phys + h_rate * S) * np.exp(-h_rate * t_grid)
        S = S * np.exp(-h_rate * t_grid)
    hazard = p_phys / np.clip(S, _S_FLOOR, None)
    return S, hazard


def auto_L(eta, beta, epsilon, kappa, xc=1.0, t_ref=None, *, probe_N=300, probe_dt=4e-3,
           probe_tau_max=20.0, probe_Gtol=1e-6, max_tries=12, **solver_kw):
    """A numerically sane lifetime scale WITHOUT data: probe cheaply, then set L = the FP mean.

    Needed because presets carry parameters but no death times, and because ``L=1`` silently
    truncates. The solver's horizon is ``L * tau_max``, so if the cohort has not absorbed we must
    GROW L to see further -- hence the multiplicative retry.

    Returns
    -------
    float : a lifetime scale in the same units as the parameters.
    """
    L = float(t_ref) if t_ref else 1.0
    probe = dict(N=probe_N, dt=probe_dt, tau_max=probe_tau_max, Gtol=probe_Gtol)
    probe.update({k: v for k, v in solver_kw.items() if k in ("scheme", "nstart")})
    for _ in range(max_tries):
        rho_beta, rho_eta, rho_eps, kap = fpl.to_dimensionless(eta, beta, epsilon, xc, L, k=kappa)
        mean, _cv, s_end = fpl.fpt_full(rho_eta, rho_beta, rho_eps, kap, **probe)
        if s_end < 1e-3 and np.isfinite(mean) and mean > 0:
            return float(L * mean)               # `mean` is trustworthy once the cohort absorbed
        L *= 4.0
    raise RuntimeError(
        "auto_L: the cohort never absorbs within the probe horizon, so no lifetime scale can be "
        "inferred. Pass L= explicitly (e.g. fp_likelihood.kaplan_meier_mean(death_times, events)) "
        "or t_ref= as a rough guess of the mean lifetime.")


def moments_from_survival(t, S):
    """(mean, cv, e2) of a non-negative lifetime from its survival curve, by integration.

    ``mean = int_0^inf S dt`` and ``E[T^2] = 2 int_0^inf t S dt``. Applied identically to both the
    homogeneous and the mixture branch so the two are never compared under different definitions
    (the homogeneous solver also returns its own mean/cv, but the mixture one returns none).

    The integral is truncated at ``t[-1]``; report ``S[-1]`` next to any moment so a heavy remaining
    tail is visible rather than silently biasing the mean low.
    """
    t = np.asarray(t, float)
    S = np.clip(np.asarray(S, float), 0.0, 1.0)
    if t.size < 2:
        return float("nan"), float("nan"), float("nan")
    if t[0] > 0:                                  # extend to 0; S(0) = 1 by definition
        t = np.concatenate(([0.0], t))
        S = np.concatenate(([1.0], S))
    mean = float(np.trapz(S, t))
    e2 = float(2.0 * np.trapz(t * S, t))
    var = e2 - mean ** 2
    cv = float(np.sqrt(var) / mean) if (var > 0 and mean > 0) else float("nan")
    return mean, cv, e2


def median_sensitivity(t, S):
    """Time units the median moves per unit of survival error: ``1 / |dS/dt|`` at S = 0.5.

    Worth reporting next to any median, because it is not a constant of the problem. A windowed
    intervention can flatten the survival curve exactly where it crosses 0.5 -- measured 0.0067 per
    unit time on a mice_F cohort with beta x0.4 over [0, 60], i.e. a survival error of 0.01 (about
    the Monte-Carlo floor at n=12000) displaces the median by 1.5 time units. Two engines can then
    look 3% apart in median while their CURVES agree to within sampling noise. Large values mean
    "compare curves, not medians".
    """
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    if S.min() > 0.5 or t.size < 3:
        return float("nan")
    i = int(np.argmax(S <= 0.5))
    lo, hi = max(i - 2, 0), min(i + 3, t.size)
    if hi - lo < 2:
        return float("nan")
    slope = np.gradient(S[lo:hi], t[lo:hi])[min(i - lo, hi - lo - 1)]
    return float("inf") if slope == 0 else float(1.0 / abs(slope))


def _median_from_survival(t, S):
    """Age at which S crosses 0.5, linearly interpolated (not grid-snapped)."""
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    if S.min() > 0.5:
        return float("nan")                       # more than half still alive at t[-1]
    i = int(np.argmax(S <= 0.5))                  # first index at or below 0.5
    if i == 0:
        return float(t[0])
    s_hi, s_lo = S[i - 1], S[i]
    if s_hi == s_lo:
        return float(t[i])
    w = (s_hi - 0.5) / (s_hi - s_lo)
    return float(t[i - 1] + w * (t[i] - t[i - 1]))


@dataclass
class CurveSet:
    """Survival / hazard / density for one theta, from either engine, with its provenance.

    Carries BOTH the plotted arrays and the untrimmed originals. That is not redundancy: when
    ``time_range`` is set, ``survival`` is the CONDITIONAL curve (clipped and renormalised to 1 at
    the window start, the convention `Dataset.getSurvival` uses), so computing a mean from it would
    silently give "mean given survival to t_lo". ``metrics()`` therefore reads the full curve unless
    a window is asked for explicitly.
    """
    t: np.ndarray
    survival: np.ndarray
    hazard: np.ndarray
    pdf: np.ndarray = None
    engine: str = "fp"
    meta: dict = field(default_factory=dict)
    t_full: np.ndarray = None
    S_full: np.ndarray = None
    p_full: np.ndarray = None
    h_full: np.ndarray = None
    native: object = None                         # the Dataset / SR_Hetro when engine == 'sim'

    def __post_init__(self):
        if self.t_full is None:
            self.t_full, self.S_full, self.p_full = self.t, self.survival, self.pdf
        if self.h_full is None:
            # Must be kept alongside S_full/p_full: `t` can be a coarse plotting grid while `t_full`
            # is the dense solved one, so interpolating the plot-resolution hazard against t_full
            # would be a length mismatch (or, worse, silently wrong if the lengths happened to match).
            self.h_full = self.hazard if np.size(self.hazard) == np.size(self.t_full) else None

    # ---------------------------------------------------------------- metrics
    def metrics(self, time_range=None, source="curve", steepness="inverseCV"):
        """median / mean / cv / steepness / tail, from the curve (default) or the native object.

        source='curve'  -- integrate this CurveSet's survival. Comparable across engines.
        source='native' -- delegate to the simulated Dataset's own estimators. NOT comparable to
                           'curve': `Dataset.getMeanLifetime`/`getCV` use DIED individuals only,
                           while the curve moments cover the full first-passage distribution
                           truncated at t[-1].
        """
        if source == "native":
            if self.native is None:
                raise ValueError("source='native' needs a simulation-backed CurveSet")
            out = {"median": float(self.native.getMedianLifetime()),
                   "mean": float(self.native.getMeanLifetime()),
                   "cv": float(self.native.getCV()),
                   "tail": float("nan"), "source": "native"}
            out["steepness"] = 1.0 / out["cv"] if out["cv"] > 0 else float("nan")
            return out

        if time_range is None:
            t, S = self.t_full, self.S_full
        else:
            t, S = self._clip(self.t_full, self.S_full, time_range, renormalize=True)
        mean, cv, _e2 = moments_from_survival(t, S)
        steep = (1.0 / cv) if (steepness == "inverseCV" and np.isfinite(cv) and cv > 0) else float("nan")
        return {"median": _median_from_survival(t, S), "mean": mean, "cv": cv,
                "steepness": steep, "tail": float(np.asarray(S, float)[-1]), "source": "curve",
                "median_sensitivity": median_sensitivity(t, S)}

    @property
    def median(self):
        return self.metrics()["median"]

    @property
    def mean(self):
        return self.metrics()["mean"]

    @property
    def cv(self):
        return self.metrics()["cv"]

    @property
    def steepness(self):
        return self.metrics()["steepness"]

    @property
    def tail(self):
        return float(np.asarray(self.S_full, float)[-1])

    # ---------------------------------------------------------------- reshaping
    @staticmethod
    def _clip(t, vals, time_range, renormalize):
        """Delegate to the library's own windowing so the anchor and the end bound match the data.

        `trim_to_range` picks indices by argmin|t - bound| and slices [i_start:i_end] (end
        EXCLUSIVE). A hand-rolled boolean mask gives a different length and a different
        renormalisation anchor -- exactly the inconsistency this layer exists to remove.
        """
        from .deathTimesDataSet import trim_to_range       # lazy: pulls lifelines
        return trim_to_range(np.asarray(t, float), np.asarray(vals, float), time_range,
                             renormalize_survival=renormalize)

    def trim(self, time_range):
        """A new CurveSet restricted to ``time_range``; survival renormalised, hazard not.

        Hazard is invariant under conditioning (h = -d log S / dt), so it must NOT be renormalised;
        only the survival baseline moves.
        """
        t, S = self._clip(self.t_full, self.S_full, time_range, renormalize=True)
        hz = None
        if self.h_full is not None:
            _th, hz = self._clip(self.t_full, self.h_full, time_range, renormalize=False)
        pdf = None
        if self.p_full is not None:
            _tp, pdf = self._clip(self.t_full, self.p_full, time_range, renormalize=False)
        meta = dict(self.meta, time_range=list(time_range))
        return CurveSet(t=t, survival=S, hazard=hz, pdf=pdf, engine=self.engine, meta=meta,
                        t_full=self.t_full, S_full=self.S_full, p_full=self.p_full,
                        h_full=self.h_full, native=self.native)

    def at(self, times):
        """(S, hazard, pdf) interpolated at arbitrary ages, from the untrimmed curve."""
        times = np.asarray(times, float)
        S = np.interp(times, self.t_full, self.S_full)
        hz = (np.interp(times, self.t_full, self.h_full) if self.h_full is not None
              else np.interp(times, self.t, self.hazard))
        pdf = None if self.p_full is None else np.interp(times, self.t_full, self.p_full)
        return S, hz, pdf

    def remaining_lifetime(self, age, kind="median"):
        """Expected remaining lifetime at ``age`` -- exact, and a pure functional of S.

        The conditional survival is ``S(age + s) / S(age)``, so this needs no simulation and carries
        no Monte-Carlo noise. `Dataset.remaining_lifetime_at_age` is the empirical counterpart.
        """
        t = np.asarray(self.t_full, float)
        S = np.asarray(self.S_full, float)
        S_age = float(np.interp(age, t, S))
        if not S_age > 0:
            return float("nan")
        keep = t >= age
        ts, Ss = t[keep] - age, S[keep] / S_age
        if ts.size < 2:
            return float("nan")
        if kind == "median":
            return _median_from_survival(ts, Ss)
        if kind == "mean":
            return moments_from_survival(ts, Ss)[0]
        raise ValueError("kind must be 'median' or 'mean'")

    def to_frame(self):
        import pandas as pd                                    # lazy
        d = {"t": self.t, "survival": self.survival, "hazard": self.hazard}
        if self.pdf is not None:
            d["pdf"] = self.pdf
        return pd.DataFrame(d)

    # ---------------------------------------------------------------- plotting
    def _ax(self, ax):
        if ax is not None:
            return ax
        import matplotlib.pyplot as plt                         # lazy
        return plt.gca()

    def _use_native(self, native):
        """Simulation-backed sets plot through the Dataset by default, to keep lifelines CI bands."""
        if native is None:
            return self.engine == "sim" and self.native is not None
        return bool(native) and self.native is not None

    def plot_survival(self, ax=None, native=None, **kw):
        ax = self._ax(ax)
        if self._use_native(native):
            tr = self.meta.get("time_range")
            return self.native.plotSurvival(ax=ax, time_range=tr, **kw)
        kw.setdefault("drawstyle", self.meta.get("drawstyle", "default"))
        ax.plot(self.t, self.survival, **kw)
        ax.set_ylabel("Survival")
        return ax

    def plot_hazard(self, ax=None, native=None, logy=True, **kw):
        ax = self._ax(ax)
        if self._use_native(native):
            out = self.native.plotHazard(ax=ax, **kw)           # never called with time_range
        else:
            kw.setdefault("drawstyle", self.meta.get("drawstyle", "default"))
            ax.plot(self.t, self.hazard, **kw)
            out = ax
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel("Hazard")
        return out

    def bin_masses(self, bin_dt, t_start=None, t_stop=None):
        """Exact probability of death in each NON-OVERLAPPING bin of width ``bin_dt``.

        ``S(t) - S(t + dt)``, not ``pdf * dt``: exact for any bin width, and it is what
        `Dataset.plotDeathTimesDistribution` estimates (``np.histogram(...)/n``, i.e. probability per
        bin, NOT a density -- overlaying a raw pdf on those axes is wrong by the bin width).

        The bins must be non-overlapping or the masses do not sum to 1: evaluating a width-5 bin at
        every point of a 1-unit grid counts each death five times.

        Returns
        -------
        (edges_left, mass) : mass[i] = P(t in [edges_left[i], edges_left[i] + bin_dt))
        """
        bin_dt = float(bin_dt)
        if not bin_dt > 0:
            raise ValueError(f"bin_dt must be positive; got {bin_dt}")
        lo = float(self.t_full[0] if t_start is None else t_start)
        hi = float(self.t_full[-1] if t_stop is None else t_stop)
        edges = np.arange(lo, hi + 0.5 * bin_dt, bin_dt)
        S_edges = np.interp(edges, self.t_full, self.S_full)
        return edges[:-1], -np.diff(S_edges)

    def plot_pdf(self, ax=None, as_probability=False, bin_dt=None, **kw):
        """Death-time density, or probability-per-bin when ``as_probability`` (see :meth:`bin_masses`).

        ``as_probability=True`` plots on its own ``bin_dt``-spaced grid, because that is the only
        version comparable to a histogram: the bins have to be non-overlapping.
        """
        ax = self._ax(ax)
        if as_probability:
            step = float(bin_dt) if bin_dt else float(np.median(np.diff(self.t)))
            left, mass = self.bin_masses(step)
            kw.setdefault("drawstyle", "steps-post")
            ax.plot(left, mass, **kw)
            ax.set_ylabel("Probability of death")
            return ax
        if self.pdf is None:
            raise ValueError("this CurveSet carries no density")
        ax.plot(self.t, np.asarray(self.pdf, float), **kw)
        ax.set_ylabel("Death-time density")
        return ax

    def __repr__(self):
        m = self.metrics()
        return (f"CurveSet(engine={self.engine!r}, n={np.size(self.t)}, "
                f"median={m['median']:.4g}, steepness={m['steepness']:.4g}, "
                f"tail={m['tail']:.3g}, profile={self.meta.get('profile', '?')})")


def fp_curves(*, eta, beta, epsilon, kappa, xc=1.0, t_grid=None, t_end=None, n_grid=400,
              t_start=1e-3, L=None, data=None, hetro=False, het_var=0.2, het_param="xc", M=None,
              external_hazard=None, time_range=None, n_moment_grid=800,
              on_short_horizon="warn", meta=None, **solver_kw):
    """A :class:`CurveSet` from the exact FP solver.

    L resolution order, documented because it is the one number that can silently ruin a curve:
      1. ``L=<float>``               -- explicit, wins.
      2. ``data=<Dataset | (dt, ev)>`` -- L = Kaplan-Meier mean, the scale the likelihood used.
      3. neither                     -- :func:`auto_L` (data-free; for presets).

    ``on_short_horizon`` controls what happens when the requested grid runs past what the solver
    integrated (``L * tau_max``), where S is right-clamped and the tail is fiction: 'warn' (default,
    numbers unchanged), 'raise', or 'ignore'.
    """
    # ---- L
    if L is None and data is not None:
        if hasattr(data, "getDeathTimes"):
            dts = np.asarray(data.getDeathTimes(), float)
            evs = np.asarray(getattr(data, "events", None), int) if getattr(
                data, "events", None) is not None else None
        else:
            dts, evs = (np.asarray(a) for a in data)
        L = float(fpl.kaplan_meier_mean(dts, evs))
    if L is None:
        L = auto_L(eta, beta, epsilon, kappa, xc=xc, t_ref=t_end, **solver_kw)
    L = float(L)

    # ---- grid
    if t_grid is None:
        if t_end is None:
            t_end = 3.0 * L                      # ~3 mean lifetimes covers essentially any cohort
        t_grid = np.linspace(float(t_start), float(t_end), int(n_grid))
    t_grid = np.asarray(t_grid, float)

    # Moments need a grid that starts near 0 and is dense; the caller's plotting grid may be neither.
    # Appending points to tau_eval cannot change the values at the existing points (both branches
    # interpolate a solved curve), so this is exact and costs no extra PDE solve.
    t_mom = np.linspace(0.0, float(t_grid.max()), int(n_moment_grid))
    t_all = np.unique(np.concatenate([t_grid, t_mom]))

    S_all, hz_all = fp_survival_hazard(
        eta, beta, epsilon, kappa, t_all, L, hetro, het_var=het_var,
        external_hazard=external_hazard, het_param=het_param, M=M, xc=xc, **solver_kw)

    # p = h * S, reconstructed rather than returned, so pdf/hazard/survival stay mutually consistent.
    p_all = np.asarray(hz_all, float) * np.asarray(S_all, float)

    # ---- horizon check: how far did the solver actually integrate?
    eff = effective_solver_kw(**solver_kw)
    t_covered = L * float(eff["tau_max"])
    warnings_list = []
    if t_all.max() > t_covered * (1 + 1e-9):
        msg = (f"requested ages up to {t_all.max():.4g} but the FP solver integrated only to "
               f"L*tau_max = {t_covered:.4g}; beyond that S is right-clamped and the tail is not "
               f"real. Raise tau_max, lower L, or shorten the grid.")
        if on_short_horizon == "raise":
            raise ValueError(msg)
        if on_short_horizon == "warn":
            import warnings as _w
            _w.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_list.append(msg)

    idx = np.searchsorted(t_all, t_grid)
    md = {"params": dict(eta=eta, beta=beta, epsilon=epsilon, kappa=kappa, xc=xc),
          "L": L, "hetro": bool(hetro), "het_param": het_param if hetro else None,
          "het_var": het_var if hetro else None, "M": M,
          "external_hazard": external_hazard, "time_range": None,
          # The EFFECTIVE settings, not just the overrides: a stored CurveSet must be able to say
          # what resolution produced it even when the caller passed nothing. `solver_requested`
          # keeps the overrides separately so it is still visible what was asked for vs inherited.
          "solver": eff, "solver_requested": dict(solver_kw), "t_covered": t_covered,
          "engine_detail": f"fp/{eff['scheme']}/N{eff['N']}/dt{eff['dt']:g}",
          "warnings": warnings_list}
    if meta:
        md.update(meta)

    cs = CurveSet(t=t_grid, survival=S_all[idx], hazard=hz_all[idx], pdf=p_all[idx],
                  engine="fp", meta=md, t_full=t_all, S_full=S_all, p_full=p_all, h_full=hz_all)
    return cs.trim(time_range) if time_range is not None else cs


def disease_onset_curves(*, eta, beta, epsilon, kappa, xd, xc=1.0, **kw):
    """Exact FP curves for DISEASE ONSET: first passage to a lower threshold ``xd < xc``.

    Same first-passage problem as death with a different absorbing level, so it is one solve -- no
    simulation and no Monte-Carlo noise, where `SR_ddd` needs a full cohort. What FP cannot give from
    marginal solves is the JOINT (onset, death) distribution: after crossing xd the walker keeps
    going, so sick-span needs the density carried across the crossing (see SRK/fp_piecewise.py).

    ``xd`` is absolute damage; under the kappa convention xc == 1, so it is also the fraction of xc.
    """
    xd = float(xd)
    if not 0 < xd < float(xc):
        raise ValueError(f"xd must satisfy 0 < xd < xc (={xc}); got {xd}")
    cs = fp_curves(eta=eta, beta=beta, epsilon=epsilon, kappa=kappa, xc=xd, **kw)
    cs.meta["readout"] = "disease_onset"
    cs.meta["xd"] = xd
    cs.meta["xc"] = float(xc)
    return cs
