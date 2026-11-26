"""Choosing the Langevin simulator's time discretisation.

The SR simulator resolves ``[t_start, t_end]`` with ``nsteps`` grid points, each subdivided into
``time_step_multiplier`` Euler substeps, so the substep is ``dt / time_step_multiplier``. Getting that
wrong is not a small quantitative error: a step size tuned around a SEED theta, then reused to replay
an FP-FITTED theta (which typically has a much larger beta), makes the cohort die almost immediately
-- the documented case is a simulated median of 24 days against a true 69 years.

LAYERS (this module grows downward; only L0 exists so far)
    L0  arithmetic and the historical heuristics, no solver, no simulation:
        `safe_time_step_multiplier`, `step_size_to_grid`, `grid_dt`.
    L1  a physics rule from the model's own timescales (kappa- and epsilon-aware).   [phase 3]
    L2  `auto_resolution`: refine until the simulated curve agrees with the EXACT FP curve, with a
        cost gate that refuses (and says what would be required) instead of running for hours. [phase 3]

Deliberately dependency-free (numpy only) so it can be imported from a notebook, a cluster node, or
`SRK.curves` without pulling numba/lifelines/matplotlib.
"""
import numpy as np

__all__ = ["safe_time_step_multiplier", "step_size_to_grid", "grid_dt", "NSTEPS_CAP",
           "sr_timescales", "required_step", "required_resolution",
           "fp_reference", "auto_resolution", "ResolutionReport", "mc_floor", "clear_cache",
           "window_nsteps_floor"]

NSTEPS_CAP = 6000          # the cap every duplicated `step_size` block in this package uses


def safe_time_step_multiplier(beta_phys, t_end, nsteps, base_tsm, safety=10, cap=200):
    """Smallest ``time_step_multiplier >= base_tsm`` such that ``dt <= 1/(safety*beta_phys)``.

    Never returns less than ``base_tsm`` (only makes the simulation finer, never coarser).

    PROMOTED VERBATIM from the analysis notebooks, where it exists as a copy-pasted cell in 28 files
    under ``Baysian03_kappa/analysis_fp/``. Same name, same positional signature, same defaults, so a
    notebook can swap its ``def`` for an import and change nothing else -- and
    tests/test_stepsize.py pins it against the notebook copy.

    KNOWN LIMITATIONS, quantified, because this is a heuristic and not the criterion:
      * it ignores kappa. The Euler stability limit of the removal term
        ``d/dX [beta*X/(X+kappa)] = beta*kappa/(X+kappa)^2`` is ``beta/kappa`` as X -> 0, so with
        xc = 1 (hence kappa <= ~1) the correct requirement is *stricter* than ``1/beta`` by ``1/kappa``.
      * it ignores epsilon, which sets the one-step jump probability that actually produces the
        "everybody dies instantly" artifact.
      * ``cap=200`` is not enough for the stiffest fitted human cohorts; those saturate it and remain
        under-resolved, which is why those curves should come from the FP solver instead.
    L1/L2 (phase 3) supersede it. It is kept because 28 notebooks and their recorded numbers depend
    on exactly this arithmetic.
    """
    if not np.isfinite(beta_phys) or beta_phys <= 0:
        return base_tsm
    dt_now = t_end / (nsteps * base_tsm)
    dt_needed = 1.0 / (safety * beta_phys)
    if dt_needed >= dt_now:
        return base_tsm
    return int(min(cap, np.ceil(base_tsm * dt_now / dt_needed)))


def step_size_to_grid(t_end, step_size, nsteps_cap=NSTEPS_CAP):
    """Convert a requested substep size into ``(nsteps, time_step_multiplier)``.

    The canonical form of a block that is duplicated verbatim in six places (`SR_hetro.getSrHetro`,
    `SRmodellib_lifelines.getSr`, `sr_mcmc.getSr`, `getISR`, `getISR_periodic`, `getSrDDD`). Note it
    is a REQUEST, not an accuracy criterion: it only spreads the requested resolution across the two
    knobs while keeping ``nsteps <= nsteps_cap``.
    """
    total_steps = int(np.ceil(t_end / step_size))
    if total_steps <= nsteps_cap:
        return total_steps, 1
    tsm = int(np.ceil(total_steps / nsteps_cap))
    nsteps = int(np.ceil(total_steps / tsm))
    return max(1, nsteps), max(1, tsm)


def grid_dt(t_end, nsteps, time_step_multiplier=1, t_start=0.0, convention="kernel"):
    """The substep actually integrated, or the one the legacy formulas assume.

    There are genuinely two conventions in the codebase and they differ by ``nsteps/(nsteps-1)``:

    'kernel' -- ``(t_end - t_start) / ((nsteps - 1) * tsm)``. What the simulator really uses: the
        kernels take ``dt = t[1] - t[0]`` from ``np.linspace(t_start, t_end, nsteps)``.
    'legacy' -- ``t_end / (nsteps * tsm)``. What the ``step_size`` blocks and the ``1/beta < dt``
        rejector use.

    0.02 % apart at nsteps=6000, so it never mattered -- but making the discrepancy explicit and
    testable is better than each call site picking one silently.
    """
    if convention == "kernel":
        n = max(int(nsteps) - 1, 1)
        return (float(t_end) - float(t_start)) / (n * int(time_step_multiplier))
    if convention == "legacy":
        return float(t_end) / (int(nsteps) * int(time_step_multiplier))
    raise ValueError(f"convention must be 'kernel' or 'legacy'; got {convention!r}")


# =============================================================================================
# L1 -- the model's own timescales, and a first guess. Microseconds, no solver, no simulation.
# =============================================================================================
#
# MEASURED, 2026-08-05, six thetas (five fitted presets + one deliberately stiff), doubling ladder
# on time_step_multiplier, KS = max|S_sim - S_FP| against the exact solver at npeople=2000
# (MC floor 1.358/sqrt(2000) = 0.030). "needs" = the first rung at or below the floor:
#
#   theta          beta    kappa     eps    needs |  1/(10b)  kappa/b  kappa^2/2eps  4*0.02*eps/v_c^2
#   mice_F         0.106  0.00526  0.0089      1  |       1        1        28              1
#   ecoli          7.47   0.113    0.485       2  |       3        2         3             35
#   Jack_Russell   587    1.13     8.74       32  |      29        3         1            522
#   Staffy         9.68   0.0170   0.808     1(*) |       1        3        27              7
#   cats_vp_M      6.87   0.0270   0.725       8  |       1        2        10              4
#   stiff synth    1321   0.191   45.3    ~128    |     291      153        55           7470
#   (*) Staffy never reaches the floor: it plateaus at KS ~ 0.035 at every tsm -- see the floor note.
#
# Conclusions that shape the code below, and they are NOT what the textbook derivation predicts:
#
#  * `1/(10*beta)` -- the heuristic already in the notebooks -- is the best single predictor
#    (1, 3, 29, 1, 1, 291 against 1, 2, 32, 1, 8, ~128). It is the FIRST GUESS here.
#  * the barrier-Peclet requirement `4*delta*eps/v_c^2`, which a stability analysis makes look like
#    the binding term, OVER-predicts by 10-100x (Jack_Russell 522 vs 32 measured; ecoli 35 vs 2).
#    Shipping it as a requirement would have wasted ~16x the compute on every stiff cohort. It is
#    kept as a DIAGNOSTIC only.
#  * `kappa^2/(2*eps)` (resolving the saturation scale against the per-step noise) also over-predicts
#    where it is large (mice_F 28 vs 1, Staffy 27 vs 1). Diagnostic only.
#  * no closed form got cats_vp_M right (rule 1, measured 8) -- although that case sits only 1.7x
#    above the MC floor, so its "needs 8" is itself noisy. This is exactly why the authority is the
#    MEASURED ladder in L2 and the rule is only a starting point.
#
# HETEROGENEITY (measured separately, 2026-08-06 -- the table above is all homogeneous, which was a
# gap worth closing since the human cohorts are precisely the xc-mixtures):
#
#   cohort                shipped tsm | homogeneous needs | heterogeneous (het_var=0.2) needs | M
#   Denmark_M_1900_hetro       7      |        7          |            7                     | 11
#   Denmark_F_1890_hetro       7      |        7          |            7                     | 14
#   Sweden_F_1910_hetro        7      |       40          |           10                     | 19
#
# Heterogeneity never made the requirement WORSE, and on Sweden_F it made it 4x cheaper: mixing over
# xc smooths the survival curve, so discretisation error has less sharp structure to distort. The
# shipped tsm=7 is adequate for all three human cohorts. Practical consequence: sizing the step from
# the homogeneous case is the conservative choice, so hetro=False is a safe fallback when in doubt.
# (The heterogeneous FP reference costs M solves rather than one, M = 11-19 here.)

_SAFETY = 10.0          # the factor in the legacy rule; kept so the first guess matches the notebooks


def sr_timescales(eta, beta, epsilon, kappa, xc=1.0, t=None):
    """The timescales the integrator has to resolve, for diagnostics and for the first guess.

    The removal term's local rate is ``d/dX [beta*X/(X+kappa)] = beta*kappa/(X+kappa)^2``, so it is
    fastest at the reflecting wall (``lam0 = beta/kappa``) and slowest at the barrier
    (``lam_c = beta*kappa/(xc+kappa)^2``); their ratio is ``((xc+kappa)/kappa)^2``, which is ~1e5 for
    the smallest fitted kappas. Which one matters depends on where the probability mass sits, which
    is why this returns all of them rather than picking.
    """
    eta, beta, epsilon, kappa, xc = (float(v) for v in (eta, beta, epsilon, kappa, xc))
    out = {
        "lam0": beta / kappa,                                  # removal relaxation at X -> 0
        "lam_c": beta * kappa / (xc + kappa) ** 2,              # ... at the absorbing barrier
        "v_c": beta * xc / (xc + kappa),                        # inward drift speed at the barrier
        "t_diff_kappa": kappa ** 2 / (2 * epsilon),             # diffusion across the saturation scale
        "t_diff_xc": xc ** 2 / (2 * epsilon),                   # ... across the whole domain
        "t_legacy": 1.0 / (_SAFETY * beta),                     # the notebooks' rule
    }
    if t is not None and eta * t < beta:                        # quasi-stationary well position
        x_star = kappa * eta * t / (beta - eta * t)
        out["x_star"] = x_star
        out["lam_star"] = beta * kappa / (x_star + kappa) ** 2
    return out


def required_step(eta, beta, epsilon, kappa, xc=1.0, safety=_SAFETY, target_ks=0.02):
    """First-guess substep ``ndt``, plus every candidate term and how it has been measured to behave.

    Returns a dict with ``ndt`` (the recommendation), ``binding`` (which term set it), and ``terms``
    (all candidates). See the measurement table above: the recommendation is the legacy
    ``1/(safety*beta)`` rule because it is the only term that tracked the measured requirement across
    four orders of magnitude in beta; the others are reported so a surprising case can be diagnosed,
    and are explicitly NOT used as bounds.
    """
    ts = sr_timescales(eta, beta, epsilon, kappa, xc)
    terms = {
        "legacy_1_over_safety_beta": 1.0 / (safety * float(beta)),
        "stability_lam0": 1.0 / ts["lam0"],
        "stability_lam_c": 1.0 / ts["lam_c"],
        "wall_kappa2_over_2eps": ts["t_diff_kappa"],
        "jump_xc2_over_2eps": ts["t_diff_xc"] / (5.6 ** 2),
        "barrier_peclet": 4.0 * float(target_ks) * float(epsilon) / max(ts["v_c"] ** 2, 1e-300),
    }
    return {"ndt": terms["legacy_1_over_safety_beta"],
            "binding": "legacy_1_over_safety_beta",
            "terms": terms,
            "diagnostic_only": ("wall_kappa2_over_2eps", "jump_xc2_over_2eps", "barrier_peclet",
                                "stability_lam0", "stability_lam_c"),
            "sigma_step_over_kappa": None}


def required_resolution(eta, beta, epsilon, kappa, t_end, nsteps, xc=1.0, base_tsm=1,
                        safety=_SAFETY, cap=None):
    """First-guess ``(nsteps, time_step_multiplier)``; never coarser than ``base_tsm``."""
    ndt = required_step(eta, beta, epsilon, kappa, xc, safety=safety)["ndt"]
    dt = grid_dt(t_end, nsteps, base_tsm)
    if ndt >= dt:
        return int(nsteps), int(base_tsm)
    tsm = int(np.ceil(base_tsm * dt / ndt))
    if cap is not None:
        tsm = min(int(cap), tsm)
    return int(nsteps), max(int(base_tsm), tsm)


def mc_floor(npeople, level=0.95):
    """Kolmogorov distance a FINITE SAMPLE from the exact distribution shows anyway.

    One-sample KS against an exact reference: ``E[D_n] ~ 0.8687/sqrt(n)`` and the 95th percentile is
    ``1.358/sqrt(n)``. Refining the time step below this buys nothing measurable, so it is the
    stopping threshold. (The 0.9/sqrt(n) figure often quoted is the MEAN, not a 95% bound -- using it
    as a threshold would stop the ladder about 1.5x too early.)
    """
    q = {0.90: 1.224, 0.95: 1.358, 0.99: 1.628}.get(level)
    if q is None:
        raise ValueError("level must be 0.90, 0.95 or 0.99")
    return q / np.sqrt(float(npeople))


# =============================================================================================
# L2 -- the authority: measure agreement against the EXACT FP curve.
# =============================================================================================
#
# The rule above is a good first guess but it demonstrably misses cases (cats_vp_M: rule 1, measured
# 8), and no closed form tracked all six test thetas. So the rule proposes and a measurement decides.
# The exact solver makes that affordable: it is the reference, it costs ~60 ms, and it turns "did I
# resolve this?" from a judgement call into a number.
#
# THREE WAYS THE LADDER MUST BE ABLE TO END, because only one of them is "fine":
#   'ok'            KS reached the finite-sample floor -- refining further is unmeasurable.
#   'floor'         KS stopped improving ABOVE the floor. Measured on Staffy: KS ~ 0.035 at every tsm
#                   from 1 to 8. Refining forever would not fix it; the residual is a property of the
#                   scheme (the X=0 boundary is handled by clamping, x = max(x_new, 0), which is only
#                   O(sqrt(dt)) accurate), so the honest answer is to report it and point at FP.
#   'refused_cost'  the required resolution would take longer than the budget. Then the report says
#                   what it would cost and how to get the same curve exactly from the FP solver.

_MAX_LADDER_RUNGS = 8
_STALL_LIMIT = 2        # consecutive non-improving rungs required before calling it a floor
_cache = {}


def clear_cache():
    """Forget memoised resolutions (they are keyed on theta, grid and probe size)."""
    _cache.clear()


class ResolutionReport:
    """What was decided, why, what it cost, and -- when refusing -- what would be required.

    ``str(report)`` is meant to be pasted into a lab notebook: it names the binding term, the measured
    agreement, the finite-sample floor it was compared against, and the exact FP call that gives the
    same curve without the simulation.
    """

    __slots__ = ("status", "nsteps", "time_step_multiplier", "ndt", "requested_tsm", "rule_tsm",
                 "measured_ks", "mc_floor", "n_probe", "ladder", "est_seconds", "budget_s",
                 "npeople", "use_fp", "sigma_step_over_kappa", "fp_self_ks", "terms", "params",
                 "notes")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.notes = kw.get("notes") or []
        self.ladder = kw.get("ladder") or []

    @property
    def ok(self):
        return self.status in ("ok", "rule_only")

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def _fp_call(self):
        p = self.params or {}
        return ("curves.get_curves(engine='fp', "
                + ", ".join(f"{k}={p.get(k):.6g}" for k in ("eta", "beta", "epsilon", "kappa")
                            if p.get(k) is not None)
                + ")")

    def __str__(self):
        p = self.params or {}
        head = {
            "ok": "[stepsize] OK",
            "rule_only": "[stepsize] rule only (not verified against the FP solver)",
            "floor": "[stepsize] FLOOR: refining the time step stopped helping",
            "capped": "[stepsize] CAPPED: hit the requested cap before converging",
            "refused_cost": "[stepsize] REFUSED: this cohort is not affordable at the resolution "
                            "it needs",
        }.get(self.status, f"[stepsize] {self.status}")
        lines = [head]
        if p:
            lines.append("  theta      " + " ".join(
                f"{k}={p[k]:.6g}" for k in ("eta", "beta", "epsilon", "kappa", "xc") if k in p))
        lines.append(f"  chose      nsteps={self.nsteps} time_step_multiplier="
                     f"{self.time_step_multiplier} (requested {self.requested_tsm}, "
                     f"rule suggested {self.rule_tsm})")
        if self.measured_ks is not None:
            # The floor belongs to the PROBE size, not to the run being configured: at n_probe=2000
            # it is 0.030, at npeople=25000 it is 0.0086. Labelling it with the wrong n would invite
            # the reader to conclude their full-size run is converged to 3.5x better than measured.
            lines.append(f"  measured   sup|dS| = {self.measured_ks:.4f} vs finite-sample floor "
                         f"{self.mc_floor:.4f} at the probe size npeople={self.n_probe} "
                         f"(1.358/sqrt(n), 95%)")
        if self.ladder:
            lines.append("  ladder     " + "  ".join(
                f"tsm={t}:{k:.3f}" for t, k, _s in self.ladder))
        if self.est_seconds is not None:
            lines.append(f"  cost       ~{self.est_seconds:.1f} s for the requested run"
                         + (f" (budget {self.budget_s:.0f} s)" if self.budget_s else ""))
        if self.status == "refused_cost":
            lines.append(f"  required   time_step_multiplier={self.rule_tsm} "
                         f"-> ~{self.est_seconds:.0f} s = {self.est_seconds / 3600:.2f} core-hours "
                         f"at npeople={self.npeople}")
            lines.append(f"  use FP     {self._fp_call()}   # exact, no simulation")
            lines.append("  override   resolution={'mode': 'auto', 'budget_s': <bigger>} to run it "
                         "anyway")
        if self.status == "floor" and self.sigma_step_over_kappa is not None:
            lines.append(f"  cause      per-substep noise / kappa = "
                         f"{self.sigma_step_over_kappa:.2f}; the X=0 boundary is handled by "
                         f"clamping (O(sqrt(dt))), so this residual is a scheme property, not a "
                         f"step-size one. For better than this, use the FP solver:")
            lines.append(f"             {self._fp_call()}")
        for n in self.notes:
            lines.append(f"  note       {n}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"ResolutionReport(status={self.status!r}, nsteps={self.nsteps}, "
                f"tsm={self.time_step_multiplier}, ks={self.measured_ks})")


def fp_reference(eta, beta, epsilon, kappa, t_end, xc=1.0, hetro=False, het_var=0.2,
                 het_param="xc", n_grid=1200, tol_ratio=0.1, target_ks=None):
    """Exact survival on a grid over ``[0, t_end]``, plus proof the SOLVER itself has converged.

    Returns ``(t, S, L, self_ks)``. ``self_ks`` compares the production grid against a finer, second
    -order one; if it is not far below the accuracy being asked of the simulator then the "Langevin
    error" being measured would partly be FP discretisation error. Measured on six thetas: 2e-6 to
    1e-3, i.e. 30-1000x below a 0.03 floor.
    """
    from . import fp_curves as fpc                            # lazy: keeps L0/L1 dependency-free

    L = fpc.auto_L(eta, beta, epsilon, kappa, xc=xc, t_ref=t_end)
    t = np.linspace(1e-9, float(t_end), int(n_grid))
    kw = dict(het_var=het_var, het_param=het_param, xc=xc)
    S_fast, _h = fpc.fp_survival_hazard(eta, beta, epsilon, kappa, t, L, hetro, **kw)
    S_acc, _h = fpc.fp_survival_hazard(eta, beta, epsilon, kappa, t, L, hetro,
                                       N=1200, dt=2.5e-4, scheme="bdf2", **kw)
    self_ks = float(np.max(np.abs(S_acc - S_fast)))
    if target_ks is not None and self_ks > tol_ratio * target_ks:
        raise RuntimeError(
            f"the FP reference is not accurate enough to judge the simulator: its own grid-to-grid "
            f"difference is {self_ks:.2e}, not << the target {target_ks:.2e}. Tighten the reference "
            f"(smaller dt, scheme='bdf2') before comparing.")
    return t, S_acc, L, self_ks


def _ks_vs_reference(sim, t_ref, S_ref):
    t_km, S_km = (np.asarray(a, float) for a in sim.getSurvival())
    S_i = np.interp(t_ref, t_km, S_km, left=1.0, right=S_km[-1])
    return float(np.max(np.abs(S_i - S_ref)))


def auto_resolution(eta, beta, epsilon, kappa, t_end, nsteps, xc=1.0, base_tsm=1, npeople=25000,
                    hetro=False, het_var=0.2, het_param="xc", external_hazard=None,
                    target_level=0.95, n_probe=2000, verify=True, cap=None, budget_s=900.0,
                    seed=17, plateau_ratio=0.75, use_cache=True):
    """Choose ``(nsteps, time_step_multiplier)`` and prove the choice against the exact FP curve.

    Starts from the L1 rule, then measures. Costs one cheap probe simulation (``n_probe``) per rung,
    and usually stops after one because the rule is a good guess. ``verify=False`` skips the
    measurement entirely and returns the rule's answer with ``status='rule_only'``.

    Never returns a resolution coarser than ``base_tsm``: refining can only improve an existing curve,
    so an automatic chooser must not be able to make one worse.

    Returns ``(nsteps, time_step_multiplier, ResolutionReport)``.
    """
    params = dict(eta=float(eta), beta=float(beta), epsilon=float(epsilon), kappa=float(kappa),
                  xc=float(xc))
    key = None
    if use_cache:
        key = (tuple(round(v, 10) for v in params.values()), float(t_end), int(nsteps),
               int(base_tsm), bool(hetro), float(het_var), het_param, int(n_probe), bool(verify),
               None if cap is None else int(cap), int(npeople), float(budget_s))
        if key in _cache:
            return _cache[key]

    floor = mc_floor(n_probe, target_level)
    rq = required_step(**params)
    _ns, rule_tsm = required_resolution(t_end=t_end, nsteps=nsteps, base_tsm=base_tsm, **params)
    notes = []

    def _sigma_over_kappa(tsm):
        ndt = grid_dt(t_end, nsteps, tsm)
        return float(np.sqrt(2.0 * params["epsilon"] * ndt) / params["kappa"])

    def _finish(status, tsm, ks=None, ladder=(), est=None, extra=()):
        rep = ResolutionReport(
            status=status, nsteps=int(nsteps), time_step_multiplier=int(tsm),
            ndt=grid_dt(t_end, nsteps, tsm), requested_tsm=int(base_tsm), rule_tsm=int(rule_tsm),
            measured_ks=ks, mc_floor=floor, n_probe=int(n_probe), ladder=list(ladder),
            est_seconds=est,
            budget_s=budget_s, npeople=int(npeople), use_fp=(status in ("refused_cost", "floor")),
            sigma_step_over_kappa=_sigma_over_kappa(tsm), fp_self_ks=None,
            terms=rq["terms"], params=params, notes=list(notes) + list(extra))
        out = (int(nsteps), int(tsm), rep)
        if key is not None:
            _cache[key] = out
        return out

    if not verify:
        notes.append("not verified against the FP solver (verify=False)")
        return _finish("rule_only", rule_tsm)

    # ---- cost gate BEFORE any expensive simulation: refuse cheaply, with a recipe.
    # Throughput is measured from the probe rather than assumed, but we need an estimate first, so
    # start from a conservative constant and correct it once a rung has actually run.
    ns_per_substep_person = 30e-9
    def _est(tsm, n):
        return nsteps * tsm * n * ns_per_substep_person

    if _est(rule_tsm, npeople) > budget_s:
        notes.append("the rule's resolution already exceeds the budget, so no simulation was run")
        return _finish("refused_cost", min(rule_tsm, cap or rule_tsm),
                       est=_est(rule_tsm, npeople))

    from .SR_hetro import getSrHetro                          # lazy: numba
    from . import curves as _crv                              # lazy: for the njit seeder

    t_ref, S_ref, _L, self_ks = fp_reference(
        params["eta"], params["beta"], params["epsilon"], params["kappa"], t_end, xc=xc,
        hetro=hetro, het_var=het_var, het_param=het_param, target_ks=floor)

    theta_sim = [params["eta"], params["beta"], params["epsilon"], params["xc"], params["kappa"]]
    hkw = {f"{het_param}_var": het_var} if hetro else {}
    ladder, tsm, prev_ks, best = [], max(int(rule_tsm), int(base_tsm)), None, None
    stalls = 0          # consecutive rungs that failed to improve; see _STALL_LIMIT

    for _rung in range(_MAX_LADDER_RUNGS):
        if cap is not None and tsm > cap:
            tsm = int(cap)
        import time as _time
        _crv.seed_simulator(seed)
        t0 = _time.perf_counter()
        sim = getSrHetro(theta_sim, npeople=n_probe, nsteps=nsteps, t_end=t_end,
                         time_step_multiplier=tsm, external_hazard=external_hazard,
                         hetro=hetro, parallel=False, **hkw)
        elapsed = _time.perf_counter() - t0
        ks = _ks_vs_reference(sim, t_ref, S_ref)
        ladder.append((int(tsm), ks, elapsed))
        # correct the throughput estimate from what actually ran
        ns_per_substep_person = max(elapsed / (nsteps * tsm * n_probe), 1e-12)
        if best is None or ks < best[1]:
            best = (tsm, ks)

        if ks <= floor:
            est = _est(tsm, npeople)
            if est > budget_s:
                notes.append("converged, but the full-size run exceeds the budget")
                return _finish("refused_cost", tsm, ks=ks, ladder=ladder, est=est)
            return _finish("ok", tsm, ks=ks, ladder=ladder, est=est)

        if prev_ks is not None and ks > plateau_ratio * prev_ks:
            stalls += 1
            # ONE stalled rung is not evidence of a floor. Measured: cats_vp_M reads
            # 0.050 / 0.039 / 0.041 / 0.029 at tsm 1/2/4/8, so a single-stall rule declares a floor
            # at tsm=4 and misses that tsm=8 actually converges. Seed noise near the floor is of the
            # same size as the improvement being tested, so require two in a row.
            if stalls >= _STALL_LIMIT:
                return _finish("floor", best[0], ks=best[1], ladder=ladder,
                               est=_est(best[0], npeople),
                               extra=[f"KS failed to improve by {100*(1-plateau_ratio):.0f}% on "
                                      f"{_STALL_LIMIT} consecutive rungs (last {prev_ks:.4f} -> "
                                      f"{ks:.4f}); further refinement is not the fix"])
        else:
            stalls = 0
        if cap is not None and tsm >= cap:
            return _finish("capped", tsm, ks=ks, ladder=ladder, est=_est(tsm, npeople))

        prev_ks = ks
        nxt = tsm * 2
        if _est(nxt, npeople) > budget_s:
            notes.append(f"the next rung (tsm={nxt}) would exceed the budget")
            return _finish("refused_cost", tsm, ks=ks, ladder=ladder, est=_est(nxt, npeople))
        tsm = nxt

    return _finish("capped", best[0], ks=best[1], ladder=ladder, est=_est(best[0], npeople),
                   extra=[f"stopped after {_MAX_LADDER_RUNGS} rungs"])


# =============================================================================================
# Window resolution -- a floor on `nsteps`, which `time_step_multiplier` cannot substitute for.
# =============================================================================================

def window_nsteps_floor(t_end, windows=(), period=None, points_per_window=20,
                        points_per_period=40, t_start=0.0):
    """Minimum ``nsteps`` needed to RESOLVE intervention windows (not the SDE).

    `intervention_SR`'s kernels set ``current_time = t[j]`` OUTSIDE the substep loop, so every
    intervention effect -- window edges, the within-window decay, the periodic square wave -- is
    sampled on the OUTER grid. Refining ``time_step_multiplier`` therefore improves the SDE and does
    nothing at all for the intervention timeline; only ``nsteps`` does.

    The constant is a RULE OF THUMB and is not calibrated -- stated plainly because the obvious
    supporting measurement turns out not to support it. mice_F with beta x0.4 on [0, 60] reads
    KS 0.0181 at nsteps=5000 and 0.0066 at nsteps=20000 against the exact piecewise-FP curve, which
    looks like a window effect; but at n=12000 the finite-sample floor is 0.0124, so that is ordinary
    dt discretisation error made visible by the larger sample, on a window spanning ~1400 grid points.
    What this floor genuinely prevents is the structural case -- a window comparable to dt, which no
    amount of substepping can represent. Calibrating the constant needs a sweep over window LENGTH at
    fixed dt.

    A caution that came out of the same measurement: do NOT judge such a run by its median. A window
    can leave the survival curve nearly flat where it crosses 0.5 -- |dS/dt| was 0.0067 per unit time
    there, so a survival error of 0.01 moves the median by 1.5 time units. Compare curves (KS), and
    see :func:`SRK.fp_curves.median_sensitivity`.

    Parameters
    ----------
    windows : iterable of (start, stop)
    period : float, optional
        For the periodic engine. Also requires the half-window to be resolved, and the grid should
        divide it evenly or the realised duty cycle drifts from cycle to cycle.
    """
    span = float(t_end) - float(t_start)
    need = 0
    for w in windows or ():
        if w is None:
            continue
        length = abs(float(w[1]) - float(w[0]))
        if length <= 0:
            continue
        need = max(need, int(np.ceil(span / (length / float(points_per_window)))))
    if period:
        need = max(need, int(np.ceil(float(points_per_period) * span / float(period))))
    return int(need)
