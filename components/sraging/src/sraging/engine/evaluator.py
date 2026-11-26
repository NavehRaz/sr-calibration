"""Evaluator: one cohort, one configuration, compiled once (contracts/python-api.md; research R1).

The jitted functions take the log-parameter vector in the order of the evaluator's `ParamLayout`; with
the default layout (eta, beta, epsilon, kappa), every feature off, numerics "v033", grid "v033" and
scheme "bdf2" they are v0.3.3's `make_loglik` programs (tests/parity/test_engine_t0.py). The public
methods take named parameters (`SRParams`) and return gradients by name.

Model inputs that are not free are FIXED for the evaluator's life (compiled in as constants, like the
cohort's data): `fixed` gives their values; xc defaults to 1 and h_ext to 0. A call whose parameters
disagree with a fixed value is refused, never silently overridden (constitution VIII).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np

import jax
import jax.numpy as jnp

from sraging.engine import mixture as MIX
from sraging.engine import stepper as ST
from sraging.engine import stepper_gen as SG
from sraging.engine.grid import insert_edges
from sraging.engine.inputs import AgeSegments, Heterogeneity, refuse_zero_spread
from sraging.engine.grid import SCHEMES, GridPolicy
from sraging.engine.layout import DEFAULT_FREE, ParamLayout
from sraging.engine.observation import Assembly
from sraging.engine.program import canonical_text
from sraging.errors import CohortDataError, UnsupportedOptionError

NUMERICS = ("v033", "v1")
_IMPLEMENTED = ("v033", "v1")
_PHYS = ("eta", "beta", "epsilon", "kappa", "xc", "h_ext")
_MIX_KEYS = ("S_i", "p_phys", "S_hi", "S_st", "F_st", "F_hi", "S_e", "S_lo_w", "S_hi_w")   # linear in S and F
_DEFAULT_FIXED = {"xc": 1.0, "h_ext": 0.0}


class Evaluator:
    def __init__(self, cohort, *, free=DEFAULT_FREE, fixed=None, numerics="v1", grid=None, scheme="bdf2",
                 q_budget_bytes=None, segments=None, heterogeneity=None):
        if numerics not in NUMERICS:
            raise UnsupportedOptionError(numerics, f"unknown numerics {numerics!r}; versions are {NUMERICS}")
        if numerics not in _IMPLEMENTED:
            raise UnsupportedOptionError(numerics, f"numerics {numerics!r} is not implemented yet")
        if scheme not in SCHEMES:
            raise UnsupportedOptionError(scheme, f"unknown scheme {scheme!r}; schemes are {SCHEMES}")
        # what a worker process rebuilds this evaluator from, and what a NodePool is bound to (spec 004 research R6)
        self._construction = {"cohort": cohort, "free": tuple(free), "fixed": dict(fixed or {}), "numerics": numerics,
                              "grid": grid, "scheme": scheme, "q_budget_bytes": q_budget_bytes, "segments": segments,
                              "heterogeneity": heterogeneity}
        self.cohort = cohort
        self.numerics = numerics
        self.layout = ParamLayout(names=tuple(free))
        if segments is not None and not isinstance(segments, AgeSegments):
            segments = AgeSegments(*segments)
        self.segments = segments
        for n in self.layout.names:
            if n.startswith("seg:"):
                sid = n[4:]
                if segments is None or sid not in {x.id for x in segments}:
                    raise UnsupportedOptionError(n, f"no segment {sid!r} to make free")
                if segments.by_id(sid).parameter == "xc":
                    raise UnsupportedOptionError(n, f"segment {sid!r} moves xc: a threshold is a cell index on "
                                                 f"the damage grid and has no gradient (FR-008)")
            elif n == "het_v":
                if heterogeneity is None:
                    raise UnsupportedOptionError(n, "het_v is free only with a heterogeneity spec")
            elif n not in _PHYS:
                raise UnsupportedOptionError(n, f"unknown free parameter {n!r}")
        if isinstance(heterogeneity, (tuple, list)):
            raise UnsupportedOptionError("heterogeneity", "a tuple of heterogeneities (several parameters at once) "
                                         "needs the Langevin engine (spec 004 FR-023); the FP engine takes one")
        if heterogeneity is not None and not isinstance(heterogeneity, Heterogeneity):
            raise UnsupportedOptionError("heterogeneity", "pass a sraging.engine.Heterogeneity")
        refuse_zero_spread(heterogeneity, "Evaluator")
        self.het = heterogeneity
        if self.het is not None:
            if self.het.parameter == "xc" and segments is not None:
                raise UnsupportedOptionError("heterogeneity", "heterogeneity in xc together with age segments is "
                                             "not in spec 002 (the thresholds would move per node)")
            if self.het.definition == "normal_frac_sd":
                from numpy.polynomial.hermite_e import hermegauss
                z, w = hermegauss(self.het.M)
                self._het_z, self._het_w = z, w / np.sqrt(2.0 * np.pi)
            elif "het_v" in self.layout.names:
                raise UnsupportedOptionError("het_v", "het_v is 'normal_frac_sd''s spread; the truncated normal's sd "
                                             "is not a free parameter in spec 004 (fitting sd is spec 006)")
        self.fixed = self._fixed(fixed)
        self.grid = (grid or GridPolicy("v033")).grid(cohort, scheme)
        if cohort.window is not None and not np.any(np.asarray(cohort.ages) >= cohort.window[0]):
            raise CohortDataError(f"nobody is at risk in the window {cohort.window}")
        seg_h = segments is not None and any(x.parameter == "h_ext" for x in segments)
        self.hazard = "h_ext" in self.layout.names or self.fixed["h_ext"] > 0.0 or seg_h
        if segments is None and self.grid.uniform:
            self._spec = ST.spec_for(self.grid, carry_F=(numerics == "v1"))
            self._asm = Assembly(cohort, self.grid, numerics=numerics, hazard=self.hazard)
            self._value_march = ST.make_march(self._spec, "value")
            self._grad_march = ST.make_march(self._spec, ST.grad_mode(self._spec, q_budget_bytes))
        else:
            self._setup_segments(scheme, seg_h)
        self._value = jax.jit(self._build(self._value_march))
        self._vg = jax.jit(jax.value_and_grad(self._build(self._grad_march)))
        self._terms = jax.jit(self._build_terms(self._value_march))
        self._wgrad = jax.jit(jax.grad(self._build_weighted(self._grad_march)))
        self._surv = jax.jit(self._build_survival(self._value_march))

    def _fixed(self, fixed):
        out = {k: v for k, v in _DEFAULT_FIXED.items() if k not in self.layout.names}
        for k, v in (fixed or {}).items():
            if k not in _PHYS:
                raise UnsupportedOptionError(k, f"unknown fixed input {k!r}; inputs are {_PHYS}")
            if k in self.layout.names:
                raise UnsupportedOptionError(k, f"{k!r} is free; it cannot also be fixed")
            v = float(v)
            if not math.isfinite(v) or v < 0 or (k != "h_ext" and v == 0):
                raise ValueError(f"fixed {k} must be finite and > 0 (h_ext >= 0), got {v!r}")
            out[k] = v
        missing = [k for k in _PHYS if k not in self.layout.names and k not in out]
        if missing:
            raise UnsupportedOptionError(missing[0], f"{missing} are neither free nor fixed")
        return out

    def _setup_segments(self, scheme, seg_h):
        """Spans, threshold cells, the mesh with the edges as nodes, and the general march (User Story 4). Also
        a graded mesh without segments (FR-020; review T064 N4): one span, the grid's own nodes."""
        g, segs = self.grid, self.segments
        if segs is None:
            from sraging.engine.inputs import AgeSegments
            segs = AgeSegments()
        if "xc" in self.layout.names and any(x.parameter == "xc" for x in segs):
            raise UnsupportedOptionError("xc", "a free xc together with an xc segment would move the threshold "
                                         "cells with xc; fix xc or drop the segment")
        L, N = g.L, g.N
        spans = segs.spans()
        self._act = [segs.active(sp) for sp in spans]
        xc0 = self.fixed.get("xc", 1.0)
        xc_k = [xc0 * (a["xc"].value if "xc" in a else 1.0) for a in self._act]
        self._x_ref = max(xc_k) if "xc" not in self.layout.names else None
        M = []
        for k, xk in enumerate(xc_k):
            m = int(round(N * xk / max(xc_k)))
            if m < 2:
                raise UnsupportedOptionError("xc", f"span {spans[k]} lowers the threshold to {xk!r}, below two "
                                             f"damage cells of {max(xc_k) / N!r} (register D-114: SRK's "
                                             f"fp_piecewise served it silently at 2/N); refine N or raise it")
            M.append(min(m, N))
        nodes0 = g.tau()
        nodes, restart = insert_edges(nodes0, segs.edges() / L)
        bounds = [sp[0] / L for sp in spans]
        nstart = g.nstart if scheme == "bdf2" else len(nodes) + 2
        self._gspec = SG.build_spec(N=N, tau=nodes, span_bounds=bounds, M_by_span=M,
                                    clock=any("A" in a for a in self._act), Gtol=g.Gtol, nstart=nstart,
                                    carry_F=(self.numerics == "v1"), extra_restart=restart)
        hz = None
        if seg_h:
            hspans = segs.spans(hazard=True)
            self._hact = [segs.active(sp, hazard=True) for sp in hspans]
            hz = (np.array([sp[0] for sp in hspans]), np.array([sp[1] - sp[0] for sp in hspans]))
        self._asm = Assembly(self.cohort, g, numerics=self.numerics, hazard=self.hazard, nodes=nodes,
                             hazard_spans=hz)
        self._asm.theta = self._theta_gen
        self._asm.h_rates = self._h_rates if seg_h else None
        self._value_march = SG.make_march(self._gspec, "value")
        self._grad_march = SG.make_march(self._gspec, "grad")
        self._spec = self._gspec

    def _seg_value(self, vals, seg):
        key = "seg:" + seg.id
        return vals[key] if key in vals else seg.value

    def _theta_gen(self, vals):
        """Per-span (rho_eta, rho_eps, rem, alpha) on the common x_ref; per-span hazard rates in vals."""
        L, N = self.grid.L, self.grid.N
        xr = self._x_ref if self._x_ref is not None else vals["xc"]
        h = 1.0 / N
        yf = (jnp.arange(N) + 0.5) * h
        re_, rh_, rem_, al_, hz_ = [], [], [], [], []
        for a in self._act:
            mult = {p: (self._seg_value(vals, a[p]) if p in a else 1.0)
                    for p in ("eta", "beta", "epsilon", "kappa", "A")}
            rb = L * vals["beta"] * mult["beta"] / xr
            rh_.append(L ** 2 * vals["eta"] * mult["eta"] / xr)
            re_.append(L * vals["epsilon"] * mult["epsilon"] / xr ** 2)
            kap = vals["kappa"] * mult["kappa"] / xr
            rem_.append(rb * yf / (yf + kap))
            al_.append(mult["A"] * jnp.ones(()))
        return {"rho_eta": jnp.stack(rh_), "rho_eps": jnp.stack(re_), "rem": jnp.stack(rem_),
                "alpha": jnp.stack(al_)}

    def _h_rates(self, vals):
        """Per hazard span: the segment's rate where one is active, else the base h_ext."""
        return jnp.stack([self._seg_value(vals, a["h_ext"]) * jnp.ones(()) if "h_ext" in a
                          else vals["h_ext"] * jnp.ones(()) for a in self._hact])

    def stiffness(self, params, segment_values=None, het_v=None) -> float:
        """The solve's largest diffusion number rho_eps * dtau * N^2 over heterogeneity nodes and age spans.
        The S route's mass residual grows with it (register D-004's condition; review sr-naveh T064 N2)."""
        z = self._z(params, segment_values, het_v)
        vals = {k: float(v) for k, v in self._vals(jnp.asarray(z)).items()}
        N = self.grid.N
        dtau = float(np.max(np.diff(np.asarray(self._asm.tau, float))))
        if self.segments is not None:
            xr = self._x_ref if self._x_ref is not None else vals["xc"]
            eps = max(vals["epsilon"] * (float(self._seg_value(vals, a["epsilon"])) if "epsilon" in a else 1.0)
                      for a in self._act)
        else:
            xr, eps = vals["xc"], vals["epsilon"]
        if self.het is not None:
            if self.het.definition == "normal_frac_sd":
                sc = 1.0 + vals.get("het_v", self.het.v) * np.asarray(self._het_z)
                sc = sc[sc > 0.0]
            else:                       # the fixed M's nodes, or the cap's for M="auto" (the widest range of s)
                sc = MIX.Q.truncated_normal_rule(self.het.sd, self.het.M if self.het.M != "auto" else MIX.CAP).s
            if self.het.parameter == "epsilon":
                eps *= float(sc.max())
            elif self.het.parameter == "xc":
                xr *= float(sc.min())
        return float(self.grid.L * eps / xr ** 2 * dtau * N ** 2)

    # -- heterogeneity (User Story 5; research R6) --------------------------------------------------
    def nodes_dropped(self, v=None) -> int:
        """Gauss-Hermite nodes with a non-positive parameter value at spread v (register D-116). The truncated normal
        drops none: its measure is truncated, not its nodes."""
        if self.het is None or self.het.definition != "normal_frac_sd":
            return 0
        v = self.het.v if v is None else float(v)
        return int(np.sum(1.0 + v * self._het_z <= 0.0))

    def _node_scales(self, vals):
        v = vals["het_v"] if "het_v" in self.layout.names else self.het.v
        sc = 1.0 + v * jnp.asarray(self._het_z)
        valid = sc > 0.0
        wt = jnp.where(valid, jnp.asarray(self._het_w), 0.0)
        wt = wt / jnp.sum(wt)
        return jnp.where(valid, sc, 1.0), wt          # a dropped node is marched at scale 1 and weighted 0

    def _points(self, march, vals, diagnostics=False):
        asm = self._asm
        if self.het is None:
            return asm.points(march(asm.theta(vals)), diagnostics)
        par = self.het.parameter
        sc, wt = self._node_scales(vals)

        def one(si):
            vk = dict(vals)
            vk[par] = (vals.get(par, 1.0) if par == "xc" else vals[par]) * si
            return asm.points(march(asm.theta(vk)), diagnostics)

        P = jax.vmap(one)(sc)
        mix = {k: jnp.tensordot(wt, P[k], axes=1) for k in _MIX_KEYS if k in P}
        mix["k"] = P["k"]
        if "beyond_march" in P:
            mix["beyond_march"] = jnp.all(P["beyond_march"], axis=0)
        if "use_F" in P:
            mix["use_F"] = jnp.any(P["use_F"], axis=0)
        return mix

    def _terms_of(self, march, vals, diagnostics=False):
        if self._asm.hz is not None:
            vals["h_rates"] = self._h_rates(vals)
        return self._asm.assemble(self._points(march, vals, diagnostics), vals)

    # -- traced functions ------------------------------------------------------------------------
    def _vals(self, z):
        # index then exp, one name at a time: v0.3.3's order (jnp.exp(z[0]), jnp.exp(z[1]), ...)
        vals = {n: jnp.exp(z[i]) for i, n in enumerate(self.layout.names)}
        vals.update(self.fixed)
        return vals

    def _build(self, march):
        asm = self._asm

        def ll_of_z(z):
            terms, _ = self._terms_of(march, self._vals(z))
            return asm.total(terms)
        return ll_of_z

    def _build_terms(self, march):
        def terms_of_z(z):
            return self._terms_of(march, self._vals(z), diagnostics=True)
        return terms_of_z

    def _build_weighted(self, march):
        def weighted(z, w):
            terms, _ = self._terms_of(march, self._vals(z))
            return jnp.sum(w * terms)
        return weighted

    def _build_survival(self, march):
        asm = self._asm

        def surv(z):
            vals = self._vals(z)
            out = march(asm.theta(vals))
            S_raw = out[0] if self._spec.carry_F else out
            if asm.general:
                return SG.truncate_gen(S_raw, asm.tau, asm.Gtol, asm.nsteps)
            return ST.truncate(S_raw, asm.tau, asm.dt, asm.Gtol, asm.nsteps)
        return surv

    # -- public, named -----------------------------------------------------------------------------
    def _z(self, params, segment_values=None, het_v=None, grad=False) -> np.ndarray:
        if params.time_unit != self.cohort.time_unit:
            raise UnsupportedOptionError("time_unit", f"the parameters are in {params.time_unit!r} but the "
                                         f"cohort is in {self.cohort.time_unit!r}")
        d = params.as_dict()
        for k, v in self.fixed.items():
            if d[k] != v:
                raise UnsupportedOptionError(k, f"{k} is fixed at {v!r} in this evaluator; the parameters "
                                             f"give {d[k]!r} (build another evaluator, or make it free)")
        if self.segments is not None:
            sv = dict(segment_values or {})
            unknown = set(sv) - {x.id for x in self.segments}
            if unknown:
                raise UnsupportedOptionError(sorted(unknown)[0], f"no segments {sorted(unknown)}")
            for x in self.segments:
                key = "seg:" + x.id
                if key in self.layout.names:
                    d[key] = sv.get(x.id, x.value)
                elif x.id in sv and sv[x.id] != x.value:
                    raise UnsupportedOptionError(key, f"segment {x.id!r} is fixed at {x.value!r} in this "
                                                 f"evaluator; make it free to vary it")
        elif segment_values:
            raise UnsupportedOptionError("segment_values", "this evaluator has no segments")
        if "het_v" in self.layout.names:
            d["het_v"] = self.het.v if het_v is None else float(het_v)
            if grad and self.nodes_dropped(d["het_v"]):
                raise UnsupportedOptionError("het_v", f"at v = {d['het_v']!r}, {self.nodes_dropped(d['het_v'])} "
                                             f"Gauss-Hermite node(s) have a non-positive {self.het.parameter}: they "
                                             f"are dropped and the likelihood jumps in v there, so no gradient in v "
                                             f"is given (register D-116)")
        elif het_v is not None:
            raise UnsupportedOptionError("het_v", "het_v is not free in this evaluator")
        return self.layout.pack(d)

    # -- the truncated normal (spec 004; contracts/python-api.md §2): the per-node mixture, not one program ---------
    @property
    def _per_node(self) -> bool:
        return self.het is not None and self.het.definition == "truncated_normal"

    def _fixed_m_only(self, method, segment_values, het_v):
        """The plain methods of a truncated-normal evaluator: a fixed M, evaluated unchecked by the per-node mixture
        (what mixture_value returns as .value, bit for bit); automatic M is refused."""
        if self.het.M == "auto":
            raise UnsupportedOptionError("M", f"{method}: automatic M returns a record: use mixture_value (the value, "
                                         f"its M and its check verdict)")
        if het_v is not None:
            raise UnsupportedOptionError("het_v", "het_v is 'normal_frac_sd''s spread")
        if segment_values:
            raise UnsupportedOptionError("segment_values", f"{method} of a truncated-normal evaluator takes the "
                                         f"segments' fixed values")

    def mixture_value(self, params, *, grad=False, het_sd_grad=False, pool=None):
        """The truncated-normal mixture with its checked M: a MixtureResult (value, grad by name with `het_sd` when
        asked, node and cost records, warnings, provenance, clean). sraging.engine.mixture.mixture_value."""
        return MIX.mixture_value(self, params, grad=grad, het_sd_grad=het_sd_grad, pool=pool)

    def objective(self):
        """(value_fn, value_and_grad_fn) of the log vector in `self.layout` order."""
        if self._per_node:
            raise UnsupportedOptionError("objective", "a truncated-normal mixture is not one jitted program (per-node "
                                         "programs folded on the host); use mixture_value. Spec 006 decides its "
                                         "calibrator interface")
        return self._value, self._vg

    def value(self, params, segment_values=None, het_v=None) -> float:
        if self._per_node:
            self._fixed_m_only("value", segment_values, het_v)
            return MIX.fixed_value(self, params)
        return float(self._value(self._z(params, segment_values, het_v)))

    def value_and_grad(self, params, segment_values=None, het_v=None):
        if self._per_node:
            self._fixed_m_only("value_and_grad", segment_values, het_v)
            v, gz = MIX.fixed_value_and_grad(self, params)
            return v, self.layout.grad_dict(gz)
        v, g = self._vg(self._z(params, segment_values, het_v, grad=True))
        return float(v), self.layout.grad_dict(g)

    def terms(self, params, segment_values=None, het_v=None):
        if self._per_node:
            self._fixed_m_only("terms", segment_values, het_v)
            return MIX.fixed_terms(self, params)
        terms, aux = self._terms(self._z(params, segment_values, het_v))
        return np.asarray(terms), {k: np.asarray(v) for k, v in aux.items()}

    def window_report(self, params, segment_values=None, het_v=None) -> dict:
        """Spec 006 (FR-015; register D-108): S(t_lo) at `params` and whether the 1e-12 clip of the window's conditioning
        binds there, with the ENGINE_S_LO_CLIPPED warning when it does. Refused for a cohort with no window."""
        if self.cohort.window is None:
            raise UnsupportedOptionError("window", "window_report needs a cohort with a fit window")
        _terms, aux = self.terms(params, segment_values, het_v)
        s_lo = float(np.min(aux["S_lo"]))
        clipped = bool(np.any(aux["lo_clipped"]))
        warning = None
        if clipped:
            from sraging.warnings import window_warning
            warning = window_warning("ENGINE_S_LO_CLIPPED", S_lo_min=s_lo, numerics=self.numerics)
        return {"S_lo_min": s_lo, "lo_clipped": clipped, "warning": warning}

    def weighted_grad(self, params, weights, segment_values=None, het_v=None) -> dict:
        w = np.asarray(weights, float)
        if w.shape != (self.cohort.n,):
            raise ValueError(f"weights must have one value per row ({self.cohort.n}), got {w.shape}")
        if self._per_node:
            self._fixed_m_only("weighted_grad", segment_values, het_v)
            return self.layout.grad_dict(MIX.fixed_value_and_grad(self, params, wts=w)[1])
        return self.layout.grad_dict(self._wgrad(self._z(params, segment_values, het_v, grad=True), w))

    def survival_on_grid(self, params, segment_values=None):
        """(tau, S, f, k) on the march's own grid: SRK's fpt_full_density outputs, in units of L (the SR
        survival, before any hazard)."""
        S, p, k = self._surv(self._z(params, segment_values))
        return np.asarray(self._asm.tau), np.asarray(S), np.asarray(p), int(k)

    def m_report(self, params, *, extra=4, early_S=0.999, segment_values=None, het_v=None) -> dict:
        """FR-018: how much each early-death term moves between M and M + extra Gauss-Hermite nodes, so a
        user sees when the quadrature, not the data, sets the term (sr-naveh seed review, finding 7). Early
        deaths: those whose mixture survival at their age exceeds `early_S`.

        Spec 004 FR-004 adds `whole`: {M, M_more, ll_M, ll_more, abs_dll, threshold, verdict}, the whole likelihood's
        change. For the truncated normal, M is the checked M (mixture_value's) and M_more its check rung, from the
        same per-node programs (no second evaluator; `extra` does not apply); the early-death fields compare the
        terms at those two M."""
        if self.het is None:
            raise UnsupportedOptionError("heterogeneity", "m_report needs a heterogeneity spec")
        if self._per_node:
            return self._m_report_tn(params, early_S, segment_values, het_v)
        more = Heterogeneity(parameter=self.het.parameter, definition="normal_frac_sd", v=self.het.v,
                             M=self.het.M + extra)
        other = Evaluator(self.cohort, free=self.layout.names, fixed=self.fixed, numerics=self.numerics,
                          grid=GridPolicy("fixed", N=self.grid.N, dt=self.grid.dt, tau_max=self.grid.tau_max,
                                          Gtol=self.grid.Gtol, L=self.grid.L),
                          scheme=self.grid.scheme, segments=self.segments, heterogeneity=more)
        t0, a0 = self.terms(params, segment_values, het_v)
        t1, _ = other.terms(params, segment_values, het_v)
        ll_M, ll_more = self.value(params, segment_values, het_v), other.value(params, segment_values, het_v)
        whole = {"M": self.het.M, "M_more": more.M, "ll_M": ll_M, "ll_more": ll_more, "abs_dll": abs(ll_M - ll_more),
                 "threshold": MIX.THRESHOLD,
                 "verdict": "converged" if abs(ll_M - ll_more) < MIX.THRESHOLD else "unconverged"}
        return self._early_deaths(self.het.M, more.M, early_S, t0, t1, a0, whole)

    def _early_deaths(self, M, M_more, early_S, t0, t1, a0, whole) -> dict:
        early = (np.asarray(self.cohort.events) == 1) & (np.asarray(a0["S_i"]) > early_S)
        rows = np.flatnonzero(early)
        return {"M": M, "M_more": M_more, "early_S": early_S,
                "rows": [int(i) + 1 for i in rows], "ages": [float(self.cohort.ages[i]) for i in rows],
                "dterm": [float(t1[i] - t0[i]) for i in rows],
                "max_abs_dterm": float(np.max(np.abs(t1[rows] - t0[rows]))) if rows.size else 0.0,
                "whole": whole}

    def _m_report_tn(self, params, early_S, segment_values, het_v) -> dict:
        if het_v is not None or segment_values:
            raise UnsupportedOptionError("het_v" if het_v is not None else "segment_values",
                                         "m_report of a truncated-normal evaluator takes neither")
        het = self.het if self.het.check else Heterogeneity(parameter=self.het.parameter, sd=self.het.sd,
                                                             M=self.het.M)
        c = MIX.mixture_value(self, params, het=het).node_record["check"]
        t0, a0 = MIX.fixed_terms(self, params, c["M"])
        t1, _ = MIX.fixed_terms(self, params, c["M_check"])
        whole = {k: c[k] for k in ("M", "ll_M", "abs_dll", "threshold", "verdict")}
        whole.update(M_more=c["M_check"], ll_more=c["ll_check"])
        return self._early_deaths(c["M"], c["M_check"], early_S, t0, t1, a0, whole)

    def program_text(self, kind: str = "value") -> str:
        if self._per_node:
            raise UnsupportedOptionError("program_text", "a truncated-normal mixture is not one jitted program "
                                         "(sraging.engine.mixture.Programs holds its per-node programs)")
        z = jnp.zeros(self.layout.size)
        fn = {"value": self._value, "value_and_grad": self._vg}[kind]
        return canonical_text(fn.lower(z))

    def construction(self) -> dict:
        """The keyword arguments this evaluator was built from: `Evaluator(**ev.construction())` builds the same
        configuration (a NodePool worker does, research R6)."""
        return dict(self._construction)

    def digest(self) -> str:
        """sha256 of the construction arguments in a canonical form (arrays by dtype, shape and bytes): two evaluators
        with the same digest compile the same programs on the same host. A NodePool is bound to one."""
        return hashlib.sha256(json.dumps(_canon(self._construction), sort_keys=True).encode()).hexdigest()

    def series(self, policy=None, *, pool=None, start=None):
        """A series of evaluations at this sd with adaptive M (spec 004 research R7): explore at M_low, the checked
        ladder afresh at each final point. sraging.engine.series.Series."""
        from sraging.engine.series import AdaptivePolicy, Series
        return Series(self, policy or AdaptivePolicy(), pool=pool, start=start)

    def price(self, params, *, M, mode="value", workers=1):
        """The stated cost of one mixture evaluation at M nodes (spec 004 FR-008; research R8): M x t_hom x f_mode,
        with t_hom timed here on this evaluator's homogeneous node (three runs after a warm-up) and f_mode the frozen
        factor (or the unfrozen one before freeze-H). No mixture is evaluated. A CostStatement."""
        from sraging.engine import cost as COST
        MIX.require_truncated_normal(self)
        if isinstance(M, bool) or not isinstance(M, int) or not 1 <= M <= MIX.CAP:
            raise UnsupportedOptionError("M", f"M is an integer in [1, {MIX.CAP}], got {M!r}")
        t_hom, _ = MIX.time_hom(MIX.programs(self), self._z(params))
        return COST.state(mode, M, t_hom, workers=workers, phase="price")

    def provenance(self) -> dict:
        return {"engine": "fp", "numerics": self.numerics, "grid": self.grid.summary(),
                "free": list(self.layout.names), "fixed": dict(self.fixed)}


def evaluator(cohort, **kw) -> Evaluator:
    return Evaluator(cohort, **kw)


def _canon(x):
    """A JSON-able canonical form of construction arguments: dataclasses by class and fields, arrays by dtype, shape
    and the sha256 of their bytes, floats by repr, dicts sorted."""
    if isinstance(x, np.ndarray):
        return {"ndarray": str(x.dtype), "shape": list(x.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"class": type(x).__name__, **{f.name: _canon(getattr(x, f.name)) for f in dataclasses.fields(x)}}
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float):
        return repr(x)
    return x
