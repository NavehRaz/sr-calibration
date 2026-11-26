"""PROTOTYPE: exact Fokker-Planck curves for WINDOWED interventions.

    >>> import SRK.fp_piecewise            # registers evaluator='piecewise_fp'
    >>> from SRK.intervention_sim import sr_metrics, Intervention
    >>> sr_metrics(theta, [Intervention('beta', -0.5, (60, 120))], evaluator='piecewise_fp',
    ...            L=124.0, t_end=214.0)

WHY THIS IS POSSIBLE AT ALL
`fpt_full` already rebuilds its operator on every time step, because the drift carries ``rho_eta *
tau`` and is therefore time-dependent already. A piecewise-constant parameter change is a small
extension of that: march the density with one parameter set, and at a window edge rebuild the
operator with another and keep going. The state that has to survive the boundary is just ``p``.

WHY IT IS A PROTOTYPE AND NOT A PATCH TO `fp_likelihood`
`fpt_full` never exposes ``p``; it returns only moments and the survival trace. Doing this properly
means adding a state hand-off there (``fpt_full(..., return_state=True)`` or an explicit stepper
object) -- and that file is the hub every fitting path imports, currently owned by a concurrent
session. So the discretisation below is DUPLICATED from `fpt_full`, deliberately and visibly.

    *** MERGE-BACK OWED ***
    When `SRK/fp_likelihood.py` is free, factor its stepper out (build() + the march) and have BOTH
    `fpt_full` and this module call it. The duplication is the whole cost of this file, and it is a
    real cost: any fix to the discretisation has to be made twice until then. The same hand-off also
    unlocks the joint (disease, death) distribution, damage distributions p(x, age) and the F(t)
    decline statistics, which are the readouts `SR_ddd` currently needs a Langevin cohort for.

`test_fp_piecewise.py` asserts this module reproduces `fpt_full` BIT-IDENTICALLY on a single
segment, which is what keeps the duplicate honest: if the two discretisations ever drift apart, that
test fails rather than the numbers quietly disagreeing.

WINDOWED ``'A'`` IS SUPPORTED, AND IT IS EXACT (added 2026-08-13)
`A` scales the ageing clock, and in the simulator (`intervention_SR.py`, the `y` variable:
``x=0, y=0, a=1.0`` at birth, then ``y += a*(1+A_eff)*dt``) that clock is a **deterministic scalar
shared by every trajectory** -- not per-particle state. So the drift's time argument becomes a
piecewise-linear, continuous ``y(tau)`` instead of ``tau``, and nothing else changes: no extra
density dimension, no extra cost. Windowed A does outlive its window (that is the whole point --
it is the only direction with memory), and that is exactly what an accumulated clock reproduces.
When no A intervention is present the march is BIT-IDENTICAL to the pre-A code (asserted).

WINDOWED EXTERNAL HAZARD IS SUPPORTED via ``return_exth=True``, applied ANALYTICALLY outside the
solver as the survival factor ``exp(-H(t))`` -- it is a multiplicative factor, so it never needs to
enter the PDE. Semantics follow the simulator: a windowed level OVERRIDES the constant background
inside its window, it does not add to it.

WINDOWED ``xc`` IS SUPPORTED, AND IT NEEDS NO REMAP (added 2026-08-13)
The docstring used to say moving the threshold mid-run means "remapping the density onto a new
domain". That is only true if you insist the grid END at the threshold. It need not: nondimensionalise
EVERY segment by ``xc_ref = max_k xc_k`` and the grid is fixed once and for all, while the absorbing
boundary becomes a moving CELL INDEX ``M_k = round(N * xc_k / xc_ref)``. Particles never move, ``h``
never changes, and nothing is interpolated -- the operator is simply built on the first ``M_k`` cells.
Both directions are then physically right for free: raising the threshold exposes empty cells (nobody
was there, and the already-dead stay dead), and lowering it absorbs whatever sits above the new
threshold immediately, which is exactly what lowering a death threshold does.
Cost: the active domain holds ``N * xc_k / xc_ref`` cells rather than ``N``, so a windowed-xc run is
slightly coarser than a plain one at the same ``N``; raise ``N`` if the window is a deep cut. When
every segment shares one threshold, ``M_k == N`` for all k and the march is bit-identical to before.
"""
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from scipy.linalg import solve_banded

from . import curves as _curves
from . import fp_likelihood as _fpl
from .fp_likelihood import (_bernoulli, _LOG_FLOOR, _P_FLOOR, _interp_S_p, kaplan_meier,
                            to_dimensionless)
from .intervention_sim import ExtHazard, _is_intervention, register_evaluator

__all__ = ["piecewise_survival", "piecewise_metrics", "build_segments", "piecewise_curves",
           "piecewise_loglik", "piecewise_loglik_terms"]


def _operator(a_face, N, h, re):
    """Scharfetter-Gummel tridiagonal operator. VERBATIM from `fpt_full.build` -- see MERGE-BACK."""
    P = a_face * h / re
    Bp = _bernoulli(P)
    Bm = _bernoulli(-P)
    M = N
    lo = np.zeros(M)
    di = np.zeros(M)
    up = np.zeros(M)
    i = np.arange(1, M - 1)
    lo[i] = (re / h ** 2) * Bm[i - 1]
    di[i] = -(re / h ** 2) * (Bm[i] + Bp[i - 1])
    up[i] = (re / h ** 2) * Bp[i]
    di[0] = -(re / h ** 2) * Bm[0]
    up[0] = (re / h ** 2) * Bp[0]
    lo[M - 1] = (re / h ** 2) * Bm[M - 2]
    di[M - 1] = -(re / h ** 2) * (Bm[M - 1] + Bp[M - 2])
    return lo, di, up


def piecewise_survival(segments, N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11, scheme="be",
                       nstart=20):
    """March the density through piecewise-constant parameter segments; return (tau, S).

    Parameters
    ----------
    segments : list of ``(tau_start, tau_stop, rho_eta, rho_beta, rho_eps, kappa)``, optionally
        with a 7th element ``a`` = the AGEING-CLOCK RATE on that segment (default 1.0) and an 8th
        ``xc_frac`` = that segment's absorbing threshold as a FRACTION of the grid's top (default
        1.0; see the module docstring on windowed xc).
        Contiguous and ascending in dimensionless time. The LAST segment's ``tau_stop`` may be
        ``inf`` to run to ``tau_max``.

    Notes
    -----
    The drift uses ABSOLUTE accumulated age throughout, so age keeps accumulating across a window
    edge; only the parameters switch. Restarting the clock at each boundary would be a different --
    and wrong -- model.

    With every ``a == 1.0`` the accumulated age IS ``tau`` and the expression evaluated is literally
    ``rho_eta * tau_next``, i.e. bit-identical to the pre-clock code. Only when some segment carries
    ``a != 1.0`` does the piecewise-linear clock ``y(tau)`` replace it. That branch is what keeps
    `test_single_segment_is_bit_identical_to_fpt_full` meaningful.
    """
    segments = [tuple(map(float, s)) for s in segments]
    if not segments:
        raise ValueError("need at least one segment")
    # Validate the ARITY before padding. `s + (1.0,)*(8-len(s))` pads a SHORT tuple silently, so a
    # 5-tuple used to run happily with kappa defaulted to 1.0 -- a completely different model, no
    # message. A 9-tuple padded by (1.0,)*-1 == () and then died inside `_seg_consts` with
    # "too many values to unpack", hundreds of lines from the caller's mistake.
    for s in segments:
        if not (6 <= len(s) <= 8):
            raise ValueError(
                f"each segment must be a 6-, 7- or 8-tuple (tau_start, tau_stop, rho_eta, "
                f"rho_beta, rho_eps, kappa[, a][, xc_frac]); got {len(s)} values: {s}")
    segments = [s + (1.0,) * (8 - len(s)) for s in segments]
    for a, b in zip(segments, segments[1:]):
        if not np.isclose(a[1], b[0]):
            raise ValueError(f"segments must be contiguous: {a[1]} != {b[0]}")

    # The ageing clock. y0[i] is the accumulated age at segment i's START; within a segment the
    # clock runs at rate a_i, so y(tau) = y0[i] + a_i * (tau - tau_start_i) -- continuous by
    # construction, which is what makes a windowed A outlive its window.
    use_clock = any(s[6] != 1.0 for s in segments)
    y0 = [segments[0][0]]
    for s in segments[:-1]:
        y0.append(y0[-1] + s[6] * (s[1] - s[0]))

    # The absorbing boundary as a cell INDEX, so a windowed xc never remaps the density (see the
    # module docstring). M_of[i] == N for every segment unless some segment carries xc_frac < 1.
    M_of = [max(2, min(N, int(round(N * s[7])))) for s in segments]

    h = 1.0 / N
    yf = (np.arange(N) + 0.5) * h
    M = M_of[0]
    p = np.zeros(N)
    p[0] = 1.0 / h
    p_prev = None
    nsteps = int(tau_max / dt)
    S = np.empty(nsteps + 1)
    tl = np.empty(nsteps + 1)
    S[0] = p.sum() * h
    tl[0] = 0.0
    last = 0
    use_bdf2 = (str(scheme).lower() == "bdf2")

    seg_i = 0
    # `rem` depends on rho_beta and kappa, so unlike fpt_full it CANNOT be hoisted out of the whole
    # march -- only out of each segment. That is precisely the assumption a windowed intervention
    # breaks, and the reason this cannot be expressed as a single fpt_full call.
    def _seg_consts(i):
        _s, _e, r_eta, r_beta, r_eps, kap, _a, _xf = segments[i]
        return r_eta, r_eps, r_beta * yf / (yf + kap)

    r_eta, r_eps, rem = _seg_consts(seg_i)

    for n in range(nsteps):
        tau_next = (n + 1) * dt
        while seg_i + 1 < len(segments) and tau_next > segments[seg_i][1]:
            seg_i += 1
            r_eta, r_eps, rem = _seg_consts(seg_i)
            if M_of[seg_i] != M:
                # The threshold moved. Nobody moves with it: LOWERING it absorbs whatever now sits
                # above it (those individuals have already crossed), RAISING it exposes empty cells.
                if M_of[seg_i] < M:
                    p[M_of[seg_i]:] = 0.0
                M = M_of[seg_i]
                p_prev = None                      # the BDF2 history spans a different domain now

        age = (y0[seg_i] + segments[seg_i][6] * (tau_next - segments[seg_i][0])
               if use_clock else tau_next)
        lo1, di1, up1 = _operator(r_eta * age - rem, M, h, r_eps)
        ab = np.zeros((3, M))
        ab[0, 1:] = -dt * up1[:-1]
        ab[2, :-1] = -dt * lo1[1:]
        if use_bdf2 and n >= nstart and p_prev is not None:
            ab[1, :] = 1.5 - dt * di1
            rhs = 2.0 * p[:M] - 0.5 * p_prev[:M]
        else:
            ab[1, :] = 1.0 - dt * di1
            rhs = p[:M]
        p_new = solve_banded((1, 1), ab, rhs)
        p_new[p_new < 0] = 0.0
        p_prev = p
        if M == N:
            p = p_new                              # the common path: no copy, as before
        else:
            p = np.zeros(N)
            p[:M] = p_new
        Sv = p[:M].sum() * h
        S[n + 1] = Sv
        tl[n + 1] = tau_next
        last = n + 1
        if Sv < Gtol and tau_next > 0.3:
            break
    return tl[:last + 1], S[:last + 1]


def build_segments(theta, interventions, L, t_end, xc=1.0, return_exth=False):
    """Physical theta + windowed interventions -> dimensionless piecewise segments.

    Every segment is converted with the SAME ``L`` and ``xc``, so they share one dimensionless time
    axis ``tau = t / L``. Segments are 7-tuples ``(tau_lo, tau_hi, rho_eta, rho_beta, rho_eps,
    kappa, a)``; ``a`` is the ageing-clock rate and is 1.0 unless an ``'A'`` intervention is active.

    return_exth : bool
        False (default) keeps the historical contract: ``(segments, xc)``, and an `ExtHazard` in the
        list raises, because a caller that gets only segments back would silently drop it.
        True returns ``(segments, xc, exth)`` -- the hazard specs are handed back for the caller to
        apply as the analytic survival factor ``exp(-H(t))`` (see `piecewise_curves`).
    """
    th = _curves.Theta.coerce(theta)
    ivs, xc_scale, exth = [], 1.0, []
    for iv in interventions:
        if isinstance(iv, ExtHazard):
            if not return_exth:
                raise NotImplementedError(
                    "piecewise_fp does not fold a windowed external hazard into the SEGMENTS; it "
                    "is a multiplicative survival factor applied outside the solver. Call with "
                    "return_exth=True (or use piecewise_curves / piecewise_loglik, which do), or "
                    "use evaluator='sim'.")
            exth.append(iv)
            continue
        if not _is_intervention(iv):
            raise TypeError(f"expected Intervention, got {type(iv).__name__}")
        if iv.param == "xc" and iv.is_chronic():
            xc_scale *= iv.scale                   # a chronic threshold rescales once, up front
            continue
        ivs.append(iv)                             # windowed xc stays: it becomes a moving boundary

    xc = float(xc) * xc_scale
    edges = {0.0, float(t_end)}
    for iv in ivs:
        if not iv.is_chronic():
            edges.update((float(iv.window[0]), float(iv.window[1])))
    edges = sorted(e for e in edges if 0.0 <= e <= float(t_end))

    # The threshold per segment, and the reference the whole grid is nondimensionalised by. Using
    # the MAX means every segment's boundary lands at or inside the grid top, so the domain is fixed
    # once and the boundary is a cell index -- no density remap. See the module docstring.
    spans = [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo]
    xc_seg = []
    for lo, hi in spans:
        mid = 0.5 * (lo + hi)
        s = 1.0
        for iv in ivs:
            if iv.param == "xc" and (iv.window[0] <= mid <= iv.window[1]):
                s *= iv.scale
        xc_seg.append(xc * s)
    xc_ref = max(xc_seg) if xc_seg else xc

    segments = []
    for (lo, hi), xc_k in zip(spans, xc_seg):
        mid = 0.5 * (lo + hi)
        eta, beta, eps, kap = th.eta, th.beta, th.epsilon, th.kappa
        a_rate = 1.0
        for iv in ivs:
            active = iv.is_chronic() or (iv.window[0] <= mid <= iv.window[1])
            if not active:
                continue
            s = iv.scale
            if iv.param == "eta":
                eta *= s
            elif iv.param == "beta":
                beta *= s
            elif iv.param == "epsilon":
                eps *= s
            elif iv.param == "kappa":
                kap *= s
            elif iv.param == "A":
                # the clock rate, NOT a parameter: it multiplies dy/dtau, so its effect on the
                # drift persists after the window through the accumulated y. Chronic A is
                # algebraically the same as chronic eta (eta*(a*tau) == (eta*a)*tau).
                a_rate *= s
            elif iv.param == "xc":
                pass                               # already folded into xc_k / the moving boundary
            else:
                raise KeyError(f"unknown parameter {iv.param!r}")
        # Nondimensionalise EVERY segment by the same xc_ref so they share one grid; the segment's
        # own threshold then enters only as the fraction xc_k / xc_ref of that grid.
        r_beta, r_eta, r_eps, kap_dl = to_dimensionless(eta, beta, eps, xc_ref, L, k=kap)
        segments.append((lo / L, hi / L, r_eta, r_beta, r_eps, kap_dl, a_rate, xc_k / xc_ref))
    # let the final segment run to tau_max so the tail is not truncated at t_end
    if segments:
        s = list(segments[-1])
        s[1] = np.inf
        segments[-1] = tuple(s)
    return (segments, xc_ref, exth) if return_exth else (segments, xc_ref)


# --------------------------------------------------------------------------- external hazard
def _cum_ext_hazard(t, exth=(), external_hazard=np.inf):
    """Cumulative external hazard ``H(t) = int_0^t h(s) ds`` for a piecewise-constant h.

    Semantics copied from the simulator's kernels (`intervention_SR.py`: inside the window
    ``chance_to_die_externally = exth_chance``, outside ``base_chance_exth``): a windowed level
    **OVERRIDES** the constant background inside its window rather than adding to it. A level is an
    additive LEVEL, so the rate is ``exp(-level)`` and ``level = inf`` means no hazard.
    """
    t = np.asarray(t, float)
    h_base = 0.0 if not np.isfinite(external_hazard) else float(np.exp(-float(external_hazard)))
    chronic = [e for e in exth if e.window is None]
    for e in chronic:                       # a chronic spec replaces the background outright
        h_base = 0.0 if not np.isfinite(e.level) else float(np.exp(-float(e.level)))
    windowed = [e for e in exth if e.window is not None]
    H = h_base * t
    if len(windowed) <= 1:
        # UNCHANGED fast path: with at most one window there is nothing to overlap, and this stays
        # bit-for-bit what it always produced.
        for e in windowed:
            h_in = 0.0 if not np.isfinite(e.level) else float(np.exp(-float(e.level)))
            lo, hi = float(e.window[0]), float(e.window[1])
            overlap = np.clip(np.maximum(np.clip(t, None, hi) - lo, 0.0), 0.0, max(hi - lo, 0.0))
            H = H + (h_in - h_base) * overlap
        return H
    # TWO OR MORE windows can OVERLAP, and `_hazard_rate` resolves an overlap by LAST-WINS (the
    # simulator's `chance_to_die_externally = exth_chance` assignment is an override, not a sum).
    # Adding each window's excess independently instead SUMS the levels in the overlap, so H stops
    # being the integral of h -- measured H(300) = 2.035 against a true 1.718 for two windows
    # overlapping on [100, 120]. Integrate the last-wins rate exactly on the breakpoint partition.
    bps = np.array(sorted({0.0}.union(float(w) for e in windowed for w in e.window)), float)
    bps = bps[bps >= 0.0]
    if bps.size < 2:
        return H
    mids = 0.5 * (bps[:-1] + bps[1:])
    Hb = np.concatenate([[0.0], np.cumsum(_hazard_rate(mids, exth, external_hazard) * np.diff(bps))])
    # H is piecewise-linear in t: interpolate inside the partition, extend at h_base past its end
    # (every window is closed there, so the background is all that is left).
    return (np.interp(np.clip(t, 0.0, bps[-1]), bps, Hb)
            + h_base * np.maximum(t - bps[-1], 0.0))


def _hazard_rate(t, exth=(), external_hazard=np.inf):
    """The instantaneous h(t) matching `_cum_ext_hazard` -- needed only for the exact-time density."""
    t = np.asarray(t, float)
    h_base = 0.0 if not np.isfinite(external_hazard) else float(np.exp(-float(external_hazard)))
    for e in exth:
        if e.window is None:
            h_base = 0.0 if not np.isfinite(e.level) else float(np.exp(-float(e.level)))
    h = np.full(t.shape, h_base, float)
    for e in exth:
        if e.window is None:
            continue
        h_in = 0.0 if not np.isfinite(e.level) else float(np.exp(-float(e.level)))
        inside = (t >= float(e.window[0])) & (t <= float(e.window[1]))
        h[inside] = h_in
    return h


_SOLVER_KW = dict(N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11)     # fp_loglik's own defaults


def _solve(theta, interventions, xc, L, t_end, external_hazard, solver_kw):
    """(tau, S, p, L, xc_eff, exth) on the DIMENSIONLESS grid, before any hazard is folded in.

    S and p are assembled exactly as `fp_likelihood.fpt_full_density` assembles them -- S forced
    monotone, then ``p = -dS/dtau`` clipped at 0 -- which is what lets the no-intervention case
    reproduce the canonical likelihood bit-for-bit rather than merely closely.
    """
    from . import fp_curves as _fpc
    th = _curves.Theta.coerce(theta)
    if L is None:
        L = _fpc.auto_L(th.eta, th.beta, th.epsilon, th.kappa, xc=xc, t_ref=t_end)
    L = float(L)
    segments, xc_eff, exth = build_segments(th, interventions, L, t_end, xc=xc, return_exth=True)
    kw = {**_SOLVER_KW, **{k: v for k, v in solver_kw.items()
                           if k in ("N", "dt", "tau_max", "Gtol", "scheme", "nstart")}}
    tau, S = piecewise_survival(segments, **kw)
    tau = np.asarray(tau, float)
    S = np.clip(np.minimum.accumulate(np.asarray(S, float)), 0.0, 1.0)
    p = np.clip(-np.gradient(S, tau), 0.0, None)
    return tau, S, p, L, float(xc_eff), exth


def piecewise_curves(theta, interventions=(), xc=1.0, L=None, t_end=None,
                     external_hazard=np.inf, **solver_kw):
    """Physical (t, S, p) under windowed interventions, external hazard already folded in.

    Returns a dict with ``t``, ``S``, ``p``, ``tau``, ``L``, ``xc``, ``n_segments``.
    """
    if t_end is None:
        raise ValueError("piecewise_curves needs t_end to place the intervention windows in time")
    tau, S, p, L, xc_eff, exth = _solve(theta, interventions, xc, L, t_end, external_hazard,
                                        solver_kw)
    t = tau * L
    S_phys, p_phys = S, p / L
    H = _cum_ext_hazard(t, exth, external_hazard)
    if np.any(H > 0.0):
        emht = np.exp(-H)
        p_phys = (p_phys + _hazard_rate(t, exth, external_hazard) * S_phys) * emht
        S_phys = S_phys * emht
    return {"t": t, "S": S_phys, "p": p_phys, "tau": tau, "L": L, "xc": xc_eff,
            "n_segments": len(build_segments(_curves.Theta.coerce(theta), interventions, L, t_end,
                                             xc=xc, return_exth=True)[0])}


def piecewise_loglik_terms(death_times, events, theta, interventions=(), xc=1.0, L=None,
                           t_end=None, external_hazard=np.inf, sampling_dt=1.0, **solver_kw):
    """Per-individual log-likelihood contributions under WINDOWED interventions.

    INTERVAL-CENSORED BY DEFAULT (``sampling_dt=1.0``). A death recorded at age ``t`` did not occur
    AT ``t``; it occurred somewhere in ``[t, t+delta)``, so its weight is ``log[S(t) - S(t+delta)]``
    rather than ``log p(t)``. That is the correct likelihood for age-to-the-day data, it handles
    ties properly (many individuals share a recorded day), and it makes log-likelihoods comparable
    across cohorts recorded at different resolutions. Pass ``sampling_dt=None`` for the exact-time
    density form.

    The interval form also suits this solver better than the density form does: the marcher produces
    ``S``, and ``p`` has to be recovered from it by differencing, which the interval form skips
    entirely.

    Returns an array aligned with ``death_times`` -- ``sum()`` is the log-likelihood. Exposing the
    terms is what makes the bootstrap over individuals free (no refitting), the same way
    `SRK.intervention.approximation_quality` uses `loglik_terms`.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if L is None:
        _, _, L = kaplan_meier(t, e)
    if t_end is None:
        t_end = float(np.max(t))
    tau, S, p, L, _xc_eff, exth = _solve(theta, interventions, xc, L, t_end, external_hazard,
                                         solver_kw)

    S_i, p_i = _interp_S_p(t / L, tau, S, p)
    S_phys, p_phys = S_i, p_i / L
    windowed = [x for x in exth if x.window is not None]
    h_scalar = float(_hazard_rate(np.zeros(1), exth, external_hazard)[0])
    if windowed:
        H = _cum_ext_hazard(t, exth, external_hazard)
        emht = np.exp(-H)
        p_phys = (p_phys + _hazard_rate(t, exth, external_hazard) * S_phys) * emht
        S_phys = S_phys * emht
    elif h_scalar > 0.0:
        emht = np.exp(-h_scalar * t)
        p_phys = (p_phys + h_scalar * S_phys) * emht
        S_phys = S_phys * emht

    # Normalise through the CANONICAL helper rather than `if sampling_dt: float(sampling_dt)`. That
    # spelling silently accepted a NEGATIVE delta (truthy) and scored log[S(t) - S(t-|d|)] <= 0,
    # which the clip turned into the -700 floor for every death (measured -29404 against fp_loglik's
    # -239 on the same 50-individual cohort); and it rejected the per-observation ARRAY form that
    # `fp_loglik` supports for cohorts with a missed scoring round. A positive scalar broadcasts to
    # a constant array, so the scalar path stays bit-for-bit what it was.
    _sdt = _fpl._norm_sampling_dt(sampling_dt, t.size)
    if _sdt is not None:
        d = _sdt
        if windowed:                       # general path: S(t+d) needs its own cumulative hazard
            S_hi = np.interp((t + d) / L, tau, S, left=1.0, right=S[-1])
            S_hi = S_hi * np.exp(-_cum_ext_hazard(t + d, exth, external_hazard))
            p_phys = np.clip(S_phys - S_hi, 0.0, None)
        else:                              # reuse the canonical helper, so the terms are identical
            p_phys = _fpl._interval_mass((t + d) / L, tau, S, S_phys, h_scalar, t, d)

    died = e == 1
    terms = np.empty(t.size, float)
    terms[died] = np.maximum(np.log(np.clip(p_phys[died], _P_FLOOR, None)), _LOG_FLOOR)
    terms[~died] = np.maximum(np.log(np.clip(S_phys[~died], _P_FLOOR, None)), _LOG_FLOOR)
    return terms


def piecewise_loglik(death_times, events, theta, interventions=(), xc=1.0, L=None, t_end=None,
                     external_hazard=np.inf, sampling_dt=1.0, **solver_kw):
    """Total log-likelihood under windowed interventions. See `piecewise_loglik_terms`.

    Returns ``-inf`` for a parameter set the solver cannot evaluate, matching `fp_loglik`.
    """
    th = _curves.Theta.coerce(theta)
    if not (np.isfinite([th.eta, th.beta, th.epsilon, th.kappa]).all()
            and th.epsilon > 0 and th.kappa > 0):
        return -np.inf
    terms = piecewise_loglik_terms(death_times, events, theta, interventions, xc=xc, L=L,
                                   t_end=t_end, external_hazard=external_hazard,
                                   sampling_dt=sampling_dt, **solver_kw)
    # Sum deaths and censored as two partial sums, exactly as fp_loglik does. Floating-point
    # addition is not associative, so summing the interleaved array instead lands one ulp away and
    # the no-intervention case stops being BIT-identical to the canonical likelihood -- which is the
    # property the gate exists to check. (Measured: 4.5e-13 on a 400-individual cohort.)
    e = (np.ones(terms.size, int) if events is None
         else np.asarray(events, int)[:terms.size])
    died = e == 1
    ll = float(terms[died].sum() + (terms[~died].sum() if (~died).any() else 0.0))
    return ll if np.isfinite(ll) else -np.inf


def piecewise_metrics(theta, interventions=(), xc=1.0, L=None, t_end=None, steepness="inverseCV",
                      N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11, scheme="be",
                      external_hazard=np.inf, **kw):
    """Longevity + steepness under WINDOWED interventions, exactly and without simulation.

    Same six keys as `fp_metrics` / `sim_metrics`.
    """
    from . import fp_curves as _fpc

    th = _curves.Theta.coerce(theta)
    if t_end is None:
        raise ValueError("piecewise_metrics needs t_end to place the intervention windows in time")
    if L is None:
        L = _fpc.auto_L(th.eta, th.beta, th.epsilon, th.kappa, xc=xc, t_ref=t_end)
    segments, xc_eff, exth = build_segments(th, interventions, L, t_end, xc=xc, return_exth=True)
    tau, S = piecewise_survival(segments, N=N, dt=dt, tau_max=tau_max, Gtol=Gtol, scheme=scheme)

    t = np.asarray(tau, float) * float(L)
    S = np.clip(np.asarray(S, float), 0.0, 1.0)
    H = _cum_ext_hazard(t, exth, external_hazard)
    if np.any(H > 0.0):
        S = S * np.exp(-H)
    mean, cv, _e2 = _fpc.moments_from_survival(t, S)
    median = float("nan")
    if S.min() <= 0.5:
        i = int(np.argmax(S <= 0.5))
        if i > 0 and S[i - 1] != S[i]:
            w = (S[i - 1] - 0.5) / (S[i - 1] - S[i])
            median = float(t[i - 1] + w * (t[i] - t[i - 1]))
        else:
            median = float(t[i])
    steep = (1.0 / cv) if (steepness == "inverseCV" and np.isfinite(cv) and cv > 0) else float("nan")
    # median_sensitivity travels WITH the median, always. A window can flatten the survival curve
    # exactly where it crosses 0.5, and then a median comparison is meaningless without it -- which is
    # how a windowed result once looked several percent off while the curves agreed within noise.
    # `sim_metrics` and `CurveSet.metrics` report it too. (`fp_const` does not: it lives in
    # SRK/intervention/, which this session does not own, and it never returns its curve.)
    return {"median": median, "mean": mean, "cv": cv, "steepness": steep,
            "median_sensitivity": _fpc.median_sensitivity(t, S),
            "tail": float(S[-1]), "absorbed": bool(S[-1] < 1e-2),
            "evaluator": "piecewise_fp", "n_segments": len(segments), "L": float(L),
            "xc": float(xc_eff)}


def _evaluator(theta, interventions, **kw):
    # NOTE `external_hazard` is deliberately NOT dropped: `piecewise_metrics` implements it, and
    # dropping it made `sr_metrics(..., evaluator='piecewise_fp', external_hazard=4.0)` silently
    # return the hazard-free answer (median 126.0 instead of 37.7) while `evaluator='sim'` honoured
    # it -- i.e. swapping evaluators silently swapped the model. `_fp_const` still drops it because
    # `fp_metrics` genuinely cannot represent it.
    for drop in ("n", "nsteps", "resolution", "method", "seed", "parallel", "replicates",
                 "decays", "rebounds", "hetro", "het_var", "return_sim", "return_curves",
                 "time_step_multiplier"):
        kw.pop(drop, None)
    return piecewise_metrics(theta, interventions, **kw)


register_evaluator("piecewise_fp", _evaluator)
