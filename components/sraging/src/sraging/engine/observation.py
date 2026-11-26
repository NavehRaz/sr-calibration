"""The observation model: from the march's survival to per-individual log-likelihood terms
(research R2, R7, R8; spec FR-011 to FR-013).

Numerics "v033" is the legacy formulas, feature by feature, so that T1 against SRK is like for like:
- the pure path is v0.3.3's assembly (`sraging.compat.v033.fp_jax.make_loglik`) operation for
  operation: S and the density interpolated at the ages (S = 1 left of the grid, S_end right of it; the
  density 0 outside and beyond the march's stop), a death recorded with resolution delta scored by the
  interval mass S(t) - S(t+delta) clipped at 0, an exact-time death by the density, a censored record by
  S, every term floored at -700, one reduction in input order;
- a constant external hazard rate h as SRK's `fp_loglik`: S_obs = S e^{-ht}, f_obs = (f + h S) e^{-ht},
  interval mass S_obs(t) - S(t+delta) e^{-h(t+delta)} clipped at 0 (a death beyond the march keeps the
  hazard-only mass: register D-115);
- a common fit window as SRK's `_windowed_ll`: rows with t < t_lo excluded (counted), deaths with
  t <= t_hi keep their whole interval, rows with t > t_hi censored at t_hi with log(max(S_hi, 1e-300)),
  every at-risk row conditioned on log(clip(S_lo, 1e-12, 1)) -- in both numerics since spec 006 (Naveh's Db; register
  D-108, retired), the clip reported in aux["lo_clipped"]. With t_hi = inf no S_hi term is assembled (register D-601):
  no row reaches past it, and interpolating at inf gives a NaN gradient.
Per-individual entry has no legacy formula; both numerics condition each row on log+ S_obs(e_i), and a
death whose recording interval contains its entry is scored over [e, t + delta) (research R8).

Every feature is switched on the host (static), so an absent feature adds no traced operation.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from sraging.engine import stepper as ST
from sraging.engine import stepper_gen as SG


#: the window's conditioning clip on S(t_lo): SRK's _windowed_ll, v0.3.3 and jax2 (register D-108)
S_LO_CLIP = 1e-12


def _logp(x):
    """log+ x: clip at e^-700 before the log, floor at -700 after (SRK's and v0.3.3's order)."""
    return jnp.maximum(jnp.log(jnp.clip(x, ST._P_FLOOR, None)), ST._LOG_FLOOR)


class Assembly:
    """The cohort's constants (closed over by the traced function, as v0.3.3 does) and the term rules."""

    def __init__(self, cohort, grid, *, numerics="v033", hazard=False, nodes=None, hazard_spans=None):
        """`nodes`: the general march's mesh (segments; User Story 4), else v0.3.3's uniform tau.
        `hazard_spans`: (starts, widths) in physical time when hazard segments make the rate piecewise."""
        t = np.asarray(cohort.ages, float)
        e = np.asarray(cohort.events, int)
        L = float(grid.L)
        self.numerics, self.hazard = numerics, bool(hazard)
        self.L, self.N, self.n = L, grid.N, t.size
        self.Gtol = grid.Gtol
        self.general = nodes is not None
        if self.general:
            self.nsteps, self.dt = len(nodes) - 1, None
            self.tau = jnp.asarray(np.asarray(nodes, float))
        else:
            self.nsteps, self.dt = grid.nsteps, grid.dt
            self.tau = jnp.arange(self.nsteps + 1) * self.dt            # exactly as fpt_full / v0.3.3
        self.hz = None
        if hazard_spans is not None:
            a, w = hazard_spans
            self.hz = (jnp.asarray(np.asarray(a, float)), jnp.asarray(np.asarray(w, float)),
                       np.asarray(a, float))
        self.teval = jnp.asarray(t / L)
        self.died = jnp.asarray(e == 1)
        d = cohort.delta
        self.interval = d is not None
        if self.interval:
            dd = np.broadcast_to(np.asarray(d, float), t.shape)
            self.thi = jnp.asarray((t + dd) / L)
            self.dl = jnp.asarray(dd)                    # the interval's length (v1 hazard composition)
        if self.hazard:
            self.t_phys = jnp.asarray(t)
            if self.interval:
                self.thi_phys = jnp.asarray(t + dd)
        # -- per-individual entry (left truncation) ------------------------------------------------
        self.entry = cohort.entry is not None
        if self.entry:
            en = np.asarray(cohort.entry, float)
            self.e_eval = jnp.asarray(en / L)
            self.inside = bool(np.any(en > t))           # an entry inside a death's recording interval
            if self.inside:
                ts = np.maximum(t, en)
                self.ts_eval = jnp.asarray(ts / L)
                if self.hazard:
                    self.ts_phys = jnp.asarray(ts)
                    if self.interval:
                        self.dl = jnp.asarray(t + dd - ts)
            if self.hazard:
                self.e_phys = jnp.asarray(en)
        # -- common fit window -------------------------------------------------------------------
        self.window = cohort.window is not None
        if self.window:
            lo, hi = cohort.window
            self.lo, self.hi = float(lo), float(hi)
            self.hi_finite = bool(np.isfinite(self.hi))      # register D-601: no S_hi term at hi = inf
            excluded = t < lo
            if numerics == "v033":                        # SRK's _windowed_ll
                beyond = (~excluded) & (t > hi)
            else:                                         # register D-107: anything reaching past t_hi
                reach = t + (np.broadcast_to(np.asarray(d, float), t.shape) if self.interval else 0.0)
                beyond = (~excluded) & ((t > hi) | ((e == 1) & (reach > hi)))
            self.excluded = jnp.asarray(excluded)
            self.beyond = jnp.asarray(beyond)
            self.at_risk = jnp.asarray(~excluded)
            self.n_excluded = int(excluded.sum())

    # -- the external hazard: constant (v0.3.3/SRK's exp(-h t)) or piecewise by segment -----------------
    def _E(self, x, h):
        """e^{-H(x)}."""
        if self.hz is None:
            return jnp.exp(-h * x)
        return jnp.exp(-self._H(x, h))

    def _H(self, x, r):
        a, w, _ = self.hz
        x = jnp.asarray(x)
        return jnp.sum(r * jnp.clip(x[..., None] - a, 0.0, w), axis=-1)

    def _rate(self, x, h):
        if self.hz is None:
            return h
        a_np = self.hz[2]
        return h[jnp.searchsorted(jnp.asarray(a_np), jnp.asarray(x), side="right") - 1]

    def _dH(self, x0, x1, h):
        """The hazard integrated over [x0, x1] (v1's interval composition)."""
        if self.hz is None:
            return h * self.dl
        a, w, _ = self.hz
        b = a + w
        ov = jnp.clip(jnp.minimum(x1[..., None], b) - jnp.maximum(x0[..., None], a), 0.0, None)
        return jnp.sum(h * ov, axis=-1)

    # -- parameters --------------------------------------------------------------------------------
    def theta(self, vals):
        """(rho_eta, rho_eps, rem) for the march from named physical values, as v0.3.3 (xc = 1). A
        chronic xc != 1 is reduced exactly by the symmetry (xc, eta, beta, eps, kappa) ->
        (c xc, c eta, c beta, c^2 eps, c kappa): to_dimensionless divides by xc (constitution VIII)."""
        L, N = self.L, self.N
        beta, eta, eps, kap = vals["beta"], vals["eta"], vals["epsilon"], vals["kappa"]
        xc = vals.get("xc", 1.0)
        if isinstance(xc, float) and xc == 1.0:
            rb, rh, re = L * beta, L ** 2 * eta, L * eps           # to_dimensionless, xc = 1
        else:
            rb, rh, re = L * beta / xc, L ** 2 * eta / xc, L * eps / xc ** 2
            kap = kap / xc
        h = 1.0 / N
        yf = (jnp.arange(N) + 0.5) * h
        rem = rb * yf / (yf + kap)                              # time-invariant removal term
        return (rh, re, rem)

    # -- terms -------------------------------------------------------------------------------------
    def points(self, march_out, diagnostics=False):
        """Stage 1: one march's survival quantities at the observation ages (no hazard, no logs). Linear in
        the march's S and F, so a heterogeneity mixture averages these (research R6)."""
        v1 = self.numerics == "v1"
        S_raw, F_raw = march_out if v1 else (march_out, None)
        if self.general:
            S, p, k = SG.truncate_gen(S_raw, self.tau, self.Gtol, self.nsteps)
        else:
            S, p, k = ST.truncate(S_raw, self.tau, self.dt, self.Gtol, self.nsteps)
        pts = {"k": k}
        pts["S_i"] = jnp.interp(self.teval, self.tau, S, left=1.0, right=S[-1])
        use_F = None
        if v1 and not self.interval:                     # register D-106: the density from F
            if self.general:
                p, use_F = SG.density_F_gen(F_raw, S, p, k, self.tau, self.nsteps)
            else:
                p, use_F = ST.density_F(F_raw, S, p, k, self.dt, self.nsteps)
        if not self.interval:
            p_i = jnp.interp(self.teval, self.tau, p, left=0.0, right=0.0)
            pts["p_phys"] = jnp.where(self.teval > self.tau[k], 0.0, p_i) / self.L
        else:
            inside = self.entry and self.inside
            if v1:                                       # register D-004: F at the interval's ends (linear)
                F = SG.truncate_F_gen(F_raw, k, self.nsteps) if self.general else ST.truncate_F(F_raw, k, self.nsteps)
                start = self.ts_eval if inside else self.teval
                pts["S_st"] = pts["S_i"] if not inside else jnp.interp(start, self.tau, S, left=1.0, right=S[-1])
                pts["S_hi"] = jnp.interp(self.thi, self.tau, S, left=1.0, right=S[-1])
                pts["F_st"] = jnp.interp(start, self.tau, F, left=0.0, right=F[-1])
                pts["F_hi"] = jnp.interp(self.thi, self.tau, F, left=0.0, right=F[-1])
            else:
                if inside:
                    pts["S_st"] = jnp.interp(self.ts_eval, self.tau, S, left=1.0, right=S[-1])
                pts["S_hi"] = jnp.interp(self.thi, self.tau, S, left=1.0, right=S[-1])
        if diagnostics:                     # only in the terms program: the value program stays v0.3.3's
            pts["beyond_march"] = self.teval >= self.tau[k]
            if use_F is not None and not self.interval:
                pts["use_F"] = jnp.interp(self.teval, self.tau, use_F * 1.0) > 0
        if self.entry:
            pts["S_e"] = jnp.interp(self.e_eval, self.tau, S, left=1.0, right=S[-1])
        if self.window and self.hi_finite:
            edge = jnp.interp(jnp.asarray([self.lo, self.hi]) / self.L, self.tau, S, left=1.0, right=S[-1])
            pts["S_lo_w"], pts["S_hi_w"] = edge[0], edge[1]
        elif self.window:                                   # register D-601: hi = inf, the edge is lo alone
            pts["S_lo_w"] = jnp.interp(jnp.asarray(self.lo) / self.L, self.tau, S, left=1.0, right=S[-1])
        return pts

    def assemble(self, pts, vals=None):
        """Stage 2: the external hazard, the death weights, the -700 floor, entry and the window."""
        v1 = self.numerics == "v1"
        h = (vals["h_rates"] if self.hz is not None else vals["h_ext"]) if self.hazard else None
        S_i = pts["S_i"]
        if not self.interval:
            p_phys = pts["p_phys"]
        if self.hazard:
            emht = self._E(self.t_phys, h)
            if not self.interval:
                p_phys = (p_phys + self._rate(self.t_phys, h) * S_i) * emht
            S_obs = S_i * emht
        else:
            S_obs = S_i
        inside = self.entry and self.inside
        if self.interval:
            S_lo_d = S_obs
            if inside:                                    # a death scored over [max(t, e), t + delta)
                S_st = pts["S_st"]
                if self.hazard:
                    S_st = S_st * self._E(self.ts_phys, h)
                S_lo_d = S_st
            S_hi = pts["S_hi"]
            if self.hazard:
                S_hi = S_hi * self._E(self.thi_phys, h)
            p_phys = jnp.clip(S_lo_d - S_hi, 0.0, None)   # v0.3.3's death mass
            if v1:                              # register D-004: the F route, only where use_F holds
                # the difference whose operands are smaller has the smaller ulp; decided after any mixing,
                # so a mixture chooses on its own S and F (research R6), and the guard can recompute it
                use_F = pts["F_hi"] < pts["S_st"]
                m = jnp.clip(pts["F_hi"] - pts["F_st"], 0.0, None)
                if self.hazard:                 # e^{-H(ts)} [m + S(t+delta)(1 - e^{-h dl})]: no cancellation
                    ts_phys = self.ts_phys if inside else self.t_phys
                    p_F = (m + pts["S_hi"] * (-jnp.expm1(-self._dH(ts_phys, self.thi_phys, h)))) * self._E(ts_phys, h)
                else:
                    p_F = m
                p_phys = jnp.where(use_F, p_F, p_phys)
                pts["use_F"] = use_F
        logp = _logp(p_phys)
        logS = _logp(S_obs)
        terms = jnp.where(self.died, logp, logS)
        aux = dict(S_i=S_i, mass=p_phys, k=pts["k"])
        for key in ("beyond_march", "use_F", "S_hi", "S_st"):
            if key in pts:
                aux[key] = pts[key]
        if self.entry:
            S_e = pts["S_e"]
            if self.hazard:
                S_e = S_e * self._E(self.e_phys, h)
            terms = terms - _logp(S_e)
        if self.window:
            S_lo_w = pts["S_lo_w"]
            if self.hazard:
                S_lo_w = S_lo_w * self._E(self.lo, h)
            if self.hi_finite:
                S_hi_w = pts["S_hi_w"] * self._E(self.hi, h) if self.hazard else pts["S_hi_w"]
                log_hi = jnp.maximum(jnp.log(jnp.maximum(S_hi_w, 1e-300)), ST._LOG_FLOOR)   # SRK, both numerics
                terms = jnp.where(self.beyond, log_hi, terms)
            # both numerics: SRK's clip (register D-108, retired by spec 006: v1's log+ floor alone gave a spurious
            # maximum where S(t_lo) underflows)
            log_lo = jnp.log(jnp.minimum(jnp.maximum(S_lo_w, S_LO_CLIP), 1.0))
            terms = jnp.where(self.excluded, 0.0, terms - log_lo)
            aux["log_S_lo"] = log_lo
            aux["S_lo"] = S_lo_w
            aux["lo_clipped"] = S_lo_w < S_LO_CLIP
        return terms, aux

    def terms(self, march_out, vals=None, diagnostics=False):
        """Per-individual log terms. Returns (terms, aux); excluded rows (before a window) contribute 0.
        `march_out` is S_raw ("v033") or (S_raw, F_raw) ("v1")."""
        return self.assemble(self.points(march_out, diagnostics), vals)

    @staticmethod
    def total(terms):
        """One reduction in input order, as v0.3.3 (SRK forms partial sums; the difference is round-off,
        inside the T1 tolerance)."""
        ll = jnp.sum(terms)
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
