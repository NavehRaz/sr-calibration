"""The truncated-normal heterogeneity mixture: per-node programs, the checked ladder of node counts, and the two-pass
gradient (spec 004 FR-001 to FR-004; research R3-R5; contracts/python-api.md §2; data-model "Ladder and check",
"Mixture result").

**The programs** (research R4). An evaluator's mixture is three compiled programs, the same for every M, so the ladder
never recompiles:
- `node_points(z, s)`: one node's linear survival quantities at the observation ages (`Assembly.points`), with the
  parameters scaled by s, five factors (eta, beta, epsilon, kappa, xc); a single-parameter mixture passes 1.0 in the
  other four, which is exact;
- its VJP;
- `assemble(mixed, z)`: `Assembly.assemble` and `Assembly.total` on the mixture.
The mixture is sum_k w_k P_k, folded on the host in float64 in ascending node index, so its bits do not depend on who
computed each node (the worker pool of US2 keeps that order).

**The ladder** (research R3). RUNGS = 192 / 1.5^j rounded. The start is the smallest rung >= C_R r g(n), with r from
the model at (theta, sd) (OD-10): r = sqrt(CV_mix^2 - CV_c^2) / CV_c, CV_c from the homogeneous march, CV_mix from the
GUIDE_M-node rule. The walk stops at the first rung M with |l(M) - l(next)| < THRESHOLD and returns l(M); at the cap
the pair (128, 192) still failing returns l(192) with HET_UNCONVERGED, never clean, with no fallback. A fixed M is
checked against the smallest rung >= ceil(1.5 M) and returns l(M); its check failing gives `unconverged` with the same
warning. check=False gives `not_checked`, never clean.

**The gradient** (research R5): pass 1 computes and mixes the points; the parent runs jax.vjp(assemble) at the mixture,
giving the mixture cotangent c and the direct z-cotangent; pass 2 runs each node's VJP with w_k c and sums the z parts
in node order. The sd term is sum_k [c.P_k dw_k/dsd + w_k (c.dP_k/ds)(z_k + sd dz_k/dsd)], c.P_k in the parent. The
value never depends on whether a gradient was asked for.
"""
from __future__ import annotations

import copy
import functools
import math
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

from sraging.engine import quadrature as Q
from sraging.engine import stepper as ST
from sraging.engine import stepper_gen as SG
from sraging.engine.inputs import Heterogeneity
from sraging.errors import UnsupportedOptionError

RUNGS = (5, 8, 11, 17, 25, 38, 57, 85, 128, 192)        # frozen in panel_004.toml [ladder] at freeze-H
THRESHOLD = 0.05                                         # nats: |l(M) - l(next)| below it passes (Naveh 2026-10-29)
CAP = 192                                                # no node count above it is ever evaluated
C_R, G_N0, G_EXP = 10.0, 10_000, 0.3                     # the guide: start >= C_R r g(n), g = (n/G_N0)^G_EXP > G_N0
GUIDE_M = 5                                              # CV_mix's rule (OD-10)
SCALED = ("eta", "beta", "epsilon", "kappa", "xc")       # node_points' scale vector, in this order
VERDICTS = ("converged", "unconverged_at_cap", "unconverged", "not_checked")
MIX_KEYS = ("S_i", "p_phys", "S_hi", "S_st", "F_st", "F_hi", "S_e", "S_lo_w", "S_hi_w")   # linear in S and F


class NonFiniteNode(ValueError):
    """A node returned a non-finite point: refused, never dropped (research R1). Names the node, its s and the keys."""

    def __init__(self, k, s, parameter, keys):
        self.k, self.s, self.parameter, self.keys = int(k), float(s), parameter, tuple(keys)
        super().__init__(f"node k = {self.k} at s = {self.s!r} ({parameter} x s) returned a non-finite point in "
                         f"{', '.join(self.keys)}: refused, never dropped from the mixture (research R1)")


@dataclass(frozen=True, eq=False)
class MixtureResult:
    """data-model "Mixture result". `grad` is by name in log parameters (and `het_sd`, d/dsd, when asked), or None."""

    value: float
    grad: dict | None
    node_record: dict
    cost_record: dict
    warnings: tuple
    provenance: dict
    clean: bool


# ------------------------------------------------------------------------------------------------------ the guide
def g(n) -> float:
    return 1.0 if n <= G_N0 else (n / G_N0) ** G_EXP


def start_rung(r, n) -> int:
    """The smallest rung >= C_R r g(n); the cap when the guide is above it."""
    x = C_R * float(r) * g(n)
    return next((m for m in RUNGS if m >= x), CAP)


def check_rung(M) -> int:
    """A fixed M's check: the smallest rung >= ceil(1.5 M) (a checked M is at most 128, so this is at most the cap)."""
    need = math.ceil(1.5 * M)
    return next(m for m in RUNGS if m >= need)


def _cv(m1, m2) -> float:
    return math.sqrt(max(m2 - m1 * m1, 0.0)) / m1


# --------------------------------------------------------------------------------------------------- the programs
class Programs:
    """One evaluator's per-node programs (research R4), compiled ahead of time on first use with their CPU recorded
    apart (`compile_cpu_s`). Holds no state that changes a number: a program returns what a fresh compilation does."""

    def __init__(self, ev):
        self.ev = ev
        self.layout_size = ev.layout.size
        self.compile_cpu_s: dict = {}
        self._c: dict = {}
        asm = ev._asm

        def node(march, z, s):
            vals = ev._vals(z)
            vk = dict(vals)
            for i, p in enumerate(SCALED):
                vk[p] = vals[p] * s[i]
            pts = asm.points(march(asm.theta(vk)))
            return {key: pts[key] for key in MIX_KEYS if key in pts}, pts["k"]

        def vals_of(z):
            vals = ev._vals(z)
            if asm.hz is not None:
                vals["h_rates"] = ev._h_rates(vals)
            return vals

        def terms(lin, k, z):
            return asm.assemble({**lin, "k": k}, vals_of(z))

        def total(lin, k, z):
            return asm.total(terms(lin, k, z)[0])

        def weighted(lin, k, z, wts):
            return jnp.sum(wts * terms(lin, k, z)[0])

        def cot(fn):
            def run(lin, k, z, *extra):
                out, f = jax.vjp(lambda a, b: fn(a, k, b, *extra), lin, z)
                return f(jnp.ones_like(out))
            return run

        def node_vjp(z, s, c, with_points):
            P, f = jax.vjp(lambda a, b: node(ev._grad_march, a, b)[0], z, s)
            cz, cs = f(c)
            return (P, cz, cs) if with_points else (cz, cs)

        def survival(z, s):
            vals = ev._vals(z)
            vk = dict(vals)
            for i, p in enumerate(SCALED):
                vk[p] = vals[p] * s[i]
            out = ev._value_march(asm.theta(vk))
            S_raw = out[0] if ev._spec.carry_F else out
            if asm.general:
                return SG.truncate_gen(S_raw, asm.tau, asm.Gtol, asm.nsteps)[0]
            return ST.truncate(S_raw, asm.tau, asm.dt, asm.Gtol, asm.nsteps)[0]

        self._fns = {"points": lambda z, s: node(ev._value_march, z, s), "total": total, "terms": terms,
                     "total_cot": cot(total), "weighted_cot": cot(weighted),
                     "node_vjp": lambda z, s, c: node_vjp(z, s, c, False),
                     "node_vjp_points": lambda z, s, c: node_vjp(z, s, c, True), "survival": survival}

    def _call(self, name, *args):
        c = self._c.get(name)
        if c is None:
            t0 = time.process_time()
            c = jax.jit(self._fns[name]).lower(*args).compile()
            self.compile_cpu_s[name] = time.process_time() - t0
            self._c[name] = c
        return c(*args)

    # -- the node programs: numpy in, numpy out (the fold is on the host) ------------------------------------------
    def points(self, z, s) -> dict:
        lin, k = self._call("points", jnp.asarray(z), jnp.asarray(s))
        out = {key: np.asarray(v) for key, v in lin.items()}
        out["k"] = int(k)
        return out

    def node_vjp(self, z, s, c, *, with_points=False):
        args = (jnp.asarray(z), jnp.asarray(s), {key: jnp.asarray(v) for key, v in c.items()})
        if with_points:
            P, cz, cs = self._call("node_vjp_points", *args)
            return {key: np.asarray(v) for key, v in P.items()}, np.asarray(cz), np.asarray(cs)
        cz, cs = self._call("node_vjp", *args)
        return np.asarray(cz), np.asarray(cs)

    def survival(self, z, s) -> np.ndarray:
        return np.asarray(self._call("survival", jnp.asarray(z), jnp.asarray(s)))

    # -- the assembly: the parent's programs ----------------------------------------------------------------------
    def _lin(self, mixed):
        return {key: jnp.asarray(v) for key, v in mixed.items() if key != "k"}, jnp.asarray(mixed["k"])

    def total(self, mixed, z) -> float:
        lin, k = self._lin(mixed)
        return float(self._call("total", lin, k, jnp.asarray(z)))

    def terms(self, mixed, z):
        lin, k = self._lin(mixed)
        return self._call("terms", lin, k, jnp.asarray(z))

    def total_cot(self, mixed, z):
        lin, k = self._lin(mixed)
        c, cz = self._call("total_cot", lin, k, jnp.asarray(z))
        return {key: np.asarray(v) for key, v in c.items()}, np.asarray(cz)

    def weighted_cot(self, mixed, z, wts):
        lin, k = self._lin(mixed)
        c, cz = self._call("weighted_cot", lin, k, jnp.asarray(z), jnp.asarray(wts))
        return {key: np.asarray(v) for key, v in c.items()}, np.asarray(cz)


def programs(ev) -> Programs:
    """The evaluator's programs, built once per evaluator (the evaluator's construction arguments fix them)."""
    p = getattr(ev, "_mix_programs", None)
    if p is None:
        p = ev._mix_programs = Programs(ev)
    return p


# ----------------------------------------------------------------------------------------------- pass 1, the fold
def _scales(parameter, s_k) -> np.ndarray:
    s = np.ones(len(SCALED))
    s[SCALED.index(parameter)] = s_k
    return s


def _node_stream(progs, z, rule, parameter, pool):
    """Pass 1's nodes as (k, P_k, worker cpu_s): in process in k order, or from the pool's workers in any order."""
    nodes = [(k, _scales(parameter, float(rule.s[k]))) for k in range(rule.M)]
    if pool is None:
        return ((k, progs.points(z, s), 0.0) for k, s in nodes)
    return pool.points(z, nodes)


def _fold_stream(stream, rule, parameter, M=None):
    """Pass 1's fold: sum_k w_k P_k strictly in k = 0..M-1 (float64, host), early arrivals buffered and the indices
    checked (pool.in_order). Returns (mixed, the nodes' march stop indices, the workers' CPU); a non-finite point is
    refused, naming its node."""
    from sraging.engine.pool import in_order
    acc, ks, wcpu = None, [], 0.0
    for k, P, cpu in in_order(stream, rule.M if M is None else M):
        s_k = float(rule.s[k])
        bad = [key for key, v in P.items() if key != "k" and not np.all(np.isfinite(v))]
        if bad:
            raise NonFiniteNode(k, s_k, parameter, bad)
        P = dict(P)
        ks.append(P.pop("k"))
        wcpu += cpu
        wk = float(rule.w[k])
        if acc is None:
            acc = {key: wk * v for key, v in P.items()}
        else:
            for key, v in P.items():
                acc[key] = acc[key] + wk * v
    acc["k"] = np.asarray(max(ks))               # only in the terms' aux; the per-node ks are reported apart
    return acc, ks, wcpu


def _fold(progs, z, rule, parameter, pool=None):
    return _fold_stream(_node_stream(progs, z, rule, parameter, pool), rule, parameter)


def _ll(progs, z, rule, parameter, pool=None):
    """(l, the mixture's points, the workers' CPU)."""
    mixed, _, wcpu = _fold(progs, z, rule, parameter, pool)
    return progs.total(mixed, z), mixed, wcpu


# ----------------------------------------------------------------------------------------------- pass 2, gradient
def _vjp_stream(progs, z, c, rule, parameter, pool, with_points):
    nodes = [(k, _scales(parameter, float(rule.s[k])), float(rule.w[k])) for k in range(rule.M)]
    if pool is None:
        return ((k, progs.node_vjp(z, s, {key: wk * v for key, v in c.items()}, with_points=with_points), 0.0)
                for k, s, wk in nodes)
    return pool.vjp(z, c, nodes, with_points=with_points)


def _pass2(progs, z, rule, parameter, mixed, *, sd_term, wts=None, pool=None):
    """(d/dz by layout, d/dsd or None, the workers' CPU): the parent's VJP at the mixture, then each node's VJP with
    w_k c (in process, or on the worker that computed node k), summed in node order after the direct z part."""
    from sraging.engine.pool import in_order
    c, gz = progs.total_cot(mixed, z) if wts is None else progs.weighted_cot(mixed, z, wts)
    j = SCALED.index(parameter)
    gsd = 0.0 if sd_term else None
    wcpu = 0.0
    for k, out, cpu in in_order(_vjp_stream(progs, z, c, rule, parameter, pool, sd_term), rule.M):
        wcpu += cpu
        if sd_term:
            P, cz, cs = out
            cP = sum(float(np.vdot(c[key], P[key])) for key in c)
            ds_k = float(rule.z[k]) + rule.spread * float(rule.dz_dsd[k])          # d s_k / d sd
            gsd = gsd + cP * float(rule.dw_dsd[k]) + float(cs[j]) * ds_k
        else:
            cz, cs = out
        gz = gz + cz
    return gz, gsd, wcpu


# ------------------------------------------------------------------------------------------------ the ladder
def guide(progs, z, sd, n) -> dict:
    """OD-10: r from the model at (theta, sd), and the start rung. Six node marches: the homogeneous one and GUIDE_M.

    r = sqrt(cv_mix^2 - cv_c^2) / cv_c is undefined where the old code divided by zero: cv_c = 0 (the homogeneous
    curve's m2 - m1^2 <= 0, as at a degenerate fitted point; hr3, Naveh 2026-10-30), or a zero first moment. There the
    start is the LOWEST rung, with r and guide None and `undefined` naming why; the checked ladder still chooses M.
    Every other point returns exactly the dict it returned before."""
    ev = progs.ev
    tau = np.asarray(ev._asm.tau, float)
    par = ev.het.parameter

    def moments(s):
        S = progs.survival(z, s)
        return float(np.trapezoid(S, tau)), float(np.trapezoid(2.0 * tau * S, tau))
    m0 = moments(np.ones(len(SCALED)))
    rule = Q.truncated_normal_rule(sd, GUIDE_M)
    mk = np.array([moments(_scales(par, float(sk))) for sk in rule.s])
    m1, m2 = float(rule.w @ mk[:, 0]), float(rule.w @ mk[:, 1])
    cv_c = _cv(*m0) if m0[0] != 0 else None
    cv_mix = _cv(m1, m2) if m1 != 0 else None
    if cv_c is None or cv_c == 0 or cv_mix is None:
        what = ("the homogeneous curve's first moment is 0" if cv_c is None else
                "the mixture's first moment is 0" if cv_mix is None else
                "cv_c = 0 (the homogeneous curve's m2 - m1^2 <= 0)")
        return {"rung": RUNGS[0], "guide": None, "r": None, "n": int(n), "cv_c": cv_c, "cv_mix": cv_mix,
                "undefined": f"{what}: r = sqrt(cv_mix^2 - cv_c^2) / cv_c is undefined, so the ladder starts at the "
                             f"lowest rung"}
    r = math.sqrt(max(cv_mix ** 2 - cv_c ** 2, 0.0)) / cv_c
    return {"rung": start_rung(r, n), "guide": C_R * r * g(n), "r": r, "n": int(n), "cv_c": cv_c, "cv_mix": cv_mix}


def _check_record(M, M_check, ll_M, ll_check, verdict, ladder, start):
    abs_dll = None if ll_check is None else abs(ll_M - ll_check)
    return {"M": M, "M_check": M_check, "ll_M": ll_M, "ll_check": ll_check, "abs_dll": abs_dll,
            "threshold": THRESHOLD, "verdict": verdict, "ladder": ladder, "start": start}


# ------------------------------------------------------------------------------------------------ the stated cost
def time_hom(progs, z, reps=3):
    """t_hom (research R8): the CPU of one homogeneous value through the per-node programs (one node at s = 1 and the
    assembly), the median of `reps` runs after a warm-up (which absorbs any compilation). Returns (t_hom, the CPU the
    measurement took)."""
    c0 = time.process_time()
    one = np.ones(len(SCALED))

    def run():
        P = progs.points(z, one)
        return progs.total(P, z)
    run()
    t = []
    for _ in range(reps):
        c1 = time.process_time()
        run()
        t.append(time.process_time() - c1)
    return float(np.median(t)), time.process_time() - c0


class _Ledger:
    """One evaluation's cost statements (research R8): each stated before its work (COST.state) and measured after:
    the parent's CPU less any compilation, plus the workers' CPU, and the wall time. Compilation is kept apart."""

    def __init__(self, progs, t_hom, pool):
        from sraging.engine import cost as COST
        self.COST, self.progs, self.t_hom, self.pool = COST, progs, t_hom, pool
        self.workers = pool.workers if pool is not None else 1
        self.f = COST.factors()
        self.statements, self.compile_cpu_s, self.worker_cpu_s = [], 0.0, 0.0

    def _compiled(self):
        return sum(self.progs.compile_cpu_s.values())

    def unit(self, phase, mode, M, work):
        """Runs work() -> (result, worker_cpu_s) under one statement; returns the result."""
        st = self.COST.state(mode, M, self.t_hom, factors=self.f, workers=self.workers, phase=phase)
        c0, w0, comp0 = time.process_time(), time.perf_counter(), self._compiled()
        out, wcpu = work()
        comp = self._compiled() - comp0
        self.compile_cpu_s += comp
        self.worker_cpu_s += wcpu
        self.statements.append(st.measured(time.process_time() - c0 - comp + wcpu, time.perf_counter() - w0).as_dict())
        return out

    def record(self, c0, t0, t_measure):
        pool = self.pool
        return {"workers": self.workers, "t_hom_cpu_s": self.t_hom, "t_hom_measure_cpu_s": t_measure,
                "factor": {"value": self.f["value"], "value_and_grad": self.f["value_and_grad"],
                           "source": self.f["source"]},
                "compile_cpu_s": self.compile_cpu_s,
                "compile_cpu_s_workers": dict(pool.compile_cpu_s) if pool is not None else {},
                "statements": self.statements, "measured_cpu_s": sum(x["measured_cpu_s"] for x in self.statements),
                "worker_cpu_s": self.worker_cpu_s, "total_cpu_s": time.process_time() - c0 + self.worker_cpu_s,
                "wall_s": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------------ the ladder
def _rung(ledger, progs, z, sd, M, parameter, pool):
    """l at one rung under its statement: (l, the mixture's points, the measured CPU)."""
    def work():
        r = _ll(progs, z, Q.truncated_normal_rule(sd, M), parameter, pool)
        return r, r[2]
    ll, mixed, _ = ledger.unit("rung", "value", M, work)
    return ll, mixed, ledger.statements[-1]["measured_cpu_s"]


def _walk(ledger, progs, z, het, n, pool):
    """The checked ladder from the guide (M="auto"). Returns (check record, M_used, value, mixed at M_used)."""
    start = ledger.unit("guide", "value", GUIDE_M + 1, lambda: (guide(progs, z, het.sd, n), 0.0))
    j = min(RUNGS.index(start["rung"]), len(RUNGS) - 2)       # the top check is (128, 192)
    ladder, mixed = [], {}

    def at(M):
        ll, mx, cpu = _rung(ledger, progs, z, het.sd, M, het.parameter, pool)
        ladder.append({"M": M, "ll": ll, "cpu_s": cpu})
        mixed[M] = mx
        return ll

    ll_lo = at(RUNGS[j])
    while True:
        lo, hi = RUNGS[j], RUNGS[j + 1]
        ll_hi = at(hi)
        if abs(ll_lo - ll_hi) < THRESHOLD:
            for key in [m for m in mixed if m != lo]:
                del mixed[key]
            return _check_record(lo, hi, ll_lo, ll_hi, "converged", ladder, start), lo, ll_lo, mixed[lo]
        if hi == CAP:
            return _check_record(lo, hi, ll_lo, ll_hi, "unconverged_at_cap", ladder, start), CAP, ll_hi, mixed[CAP]
        mixed.pop(lo)
        j, ll_lo = j + 1, ll_hi


def _fixed(ledger, progs, z, het, pool, explore=False):
    """A fixed M: checked against the smallest rung >= ceil(1.5 M), or not checked."""
    ll, mixed, cpu = _rung(ledger, progs, z, het.sd, het.M, het.parameter, pool)
    ladder = [{"M": het.M, "ll": ll, "cpu_s": cpu}]
    if not het.check:
        verdict = "not_checked: explore" if explore else "not_checked"
        return _check_record(het.M, None, ll, None, verdict, ladder, None), het.M, ll, mixed
    Mc = check_rung(het.M)
    llc, _, cpu_c = _rung(ledger, progs, z, het.sd, Mc, het.parameter, pool)
    ladder.append({"M": Mc, "ll": llc, "cpu_s": cpu_c})
    verdict = "converged" if abs(ll - llc) < THRESHOLD else "unconverged"
    return _check_record(het.M, Mc, ll, llc, verdict, ladder, None), het.M, ll, mixed


def _warnings(het, check) -> tuple:
    from sraging.warnings import sampler_warning
    out = list(het.warnings)
    if check["verdict"] in ("unconverged_at_cap", "unconverged"):
        out.append(sampler_warning("HET_UNCONVERGED", M=check["M"], M_check=check["M_check"],
                                   abs_dll=check["abs_dll"], threshold=check["threshold"]))
    elif check["verdict"].startswith("not_checked"):
        out.append(sampler_warning("HET_NOT_CHECKED", M=check["M"]))
    return tuple(out)


# ----------------------------------------------------------------------------------------------- the entry point
def require_truncated_normal(ev, het=None) -> Heterogeneity:
    """The heterogeneity this evaluation uses: the evaluator's, or `het` (same parameter, truncated_normal) for a
    caller that needs another sd or M on the same compiled programs (HR-2's sd grid, the reference's M = 128, 192)."""
    if ev.het is None:
        raise UnsupportedOptionError("heterogeneity", "mixture_value needs a heterogeneity spec")
    if ev.het.definition != "truncated_normal":
        raise UnsupportedOptionError("heterogeneity", "mixture_value is the truncated_normal per-node mixture; a "
                                     "'normal_frac_sd' evaluator keeps spec 002's fused program (value, "
                                     "value_and_grad)")
    het = ev.het if het is None else het
    if not isinstance(het, Heterogeneity) or het.definition != "truncated_normal" or het.parameter != ev.het.parameter:
        raise UnsupportedOptionError("het", f"an override must be a truncated_normal Heterogeneity in the evaluator's "
                                     f"parameter {ev.het.parameter!r}, got {het!r}")
    return het


def mixture_value(ev, params, *, grad=False, het_sd_grad=False, pool=None, het=None, t_hom=None,
                  explore=False) -> MixtureResult:
    """contracts/python-api.md §2. `het`: an override of the evaluator's heterogeneity (same parameter). `pool`: a
    NodePool bound to this evaluator's configuration. `t_hom`: a series' own (else timed here, research R8).
    `explore`: a series' explore point (an unchecked M; its verdict reads "not_checked: explore")."""
    het = require_truncated_normal(ev, het)
    if pool is not None:
        from sraging.engine.pool import NodePool
        if not isinstance(pool, NodePool):
            raise UnsupportedOptionError("pool", f"pool must be a sraging.engine.NodePool, got {type(pool).__name__}")
        pool.require(ev)
    if explore and (het.M == "auto" or het.check):
        raise UnsupportedOptionError("explore", "an explore point is an unchecked fixed M")
    c0, t0 = time.process_time(), time.perf_counter()
    z = ev._z(params)
    progs = programs(ev)
    t_measure = 0.0
    if t_hom is None:
        t_hom, t_measure = time_hom(progs, z)
    ledger = _Ledger(progs, t_hom, pool)
    if het.M == "auto":
        check, M_used, value, mixed = _walk(ledger, progs, z, het, ev.cohort.n, pool)
    else:
        check, M_used, value, mixed = _fixed(ledger, progs, z, het, pool, explore)
    rule = Q.truncated_normal_rule(het.sd, M_used)
    out_grad = None
    if grad or het_sd_grad:
        def work():
            r = _pass2(progs, z, rule, het.parameter, mixed, sd_term=het_sd_grad, pool=pool)
            return r[:2], r[2]
        gz, gsd = ledger.unit("pass 2", "pass2", M_used, work)
        out_grad = ev.layout.grad_dict(gz) if grad else {}
        if het_sd_grad:
            out_grad["het_sd"] = float(gsd)
    warnings = _warnings(het, check)
    node_record = {"definition": "truncated_normal", "parameter": het.parameter, "sd": het.sd, "M_used": M_used,
                   "rule_sha256": rule.sha256, "s_min": rule.s_min, "check": check}
    return MixtureResult(value=value, grad=out_grad, node_record=node_record,
                         cost_record=ledger.record(c0, t0, t_measure), warnings=warnings,
                         provenance=provenance(ev, rule, ledger.workers), clean=not warnings)


@functools.lru_cache(maxsize=1)
def _process_provenance() -> dict:
    """The package (version, git sha and dirty flag), libraries and host, read once per process: a git call per
    evaluation would cost more than a node at small n. A run that spans a commit is caught by the acceptance run's own
    envelope (tree_changed_during_run)."""
    from sraging import provenance as PV
    return {"package": {"version": PV._version("sraging"), **PV._git()},
            "libraries": {"python": sys.version.split()[0],
                          **{d: PV._version(d) for d in ("numpy", "scipy", "jax", "jaxlib")}},
            "host": {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node()}}


def provenance(ev, rule, workers=1) -> dict:
    return {"engine": "fp", "numerics": ev.numerics, "grid": ev.grid.summary(), "definition": rule.definition,
            "rule_sha256": rule.sha256, "workers": workers, **copy.deepcopy(_process_provenance())}


def value_at_rule(ev, params, rule) -> float:
    """The mixture at an explicit rule (any nodes z, weights w and scales s), unchecked and with no record: the
    reference's independent composite cross-check (quadrature.composite_rule) and FR-009's gen-1 law
    (generator._het_nodes). The evaluator's heterogeneity names the parameter the scales multiply."""
    require_truncated_normal(ev)
    if not (np.all(np.isfinite(rule.s)) and np.all(rule.s > 0) and abs(float(np.sum(rule.w)) - 1.0) < 1e-12):
        raise ValueError(f"a rule needs finite scales s > 0 and weights summing to 1 ({rule.definition}, M = {rule.M})")
    return _ll(programs(ev), ev._z(params), rule, ev.het.parameter)[0]


# ------------------------------------------------------------------------- a fixed M: the plain methods' route
def fixed_value(ev, params) -> float:
    het = ev.het
    return _ll(programs(ev), ev._z(params), Q.truncated_normal_rule(het.sd, het.M), het.parameter)[0]


def fixed_value_and_grad(ev, params, wts=None):
    """(value, d/dz by layout) at the evaluator's fixed M, unchecked; `wts` gives the weighted sum's gradient."""
    het, progs = ev.het, programs(ev)
    z = ev._z(params, grad=True)
    rule = Q.truncated_normal_rule(het.sd, het.M)
    ll, mixed, _ = _ll(progs, z, rule, het.parameter)
    gz, _, _ = _pass2(progs, z, rule, het.parameter, mixed, sd_term=False, wts=wts)
    return ll, gz


def fixed_terms(ev, params, M=None):
    """(terms, aux) of the mixture at a fixed M (the evaluator's, or M for m_report), with aux["k"] the node ks."""
    het, progs = ev.het, programs(ev)
    z = ev._z(params)
    mixed, ks, _ = _fold(progs, z, Q.truncated_normal_rule(het.sd, het.M if M is None else M), het.parameter)
    terms, aux = progs.terms(mixed, z)
    aux = {k: np.asarray(v) for k, v in aux.items()}
    aux["k"] = np.asarray(ks)
    return np.asarray(terms), aux
