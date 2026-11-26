"""Adaptive M inside a series: maximisations and profile points at fixed sd (spec 004 FR-006; research R7; data-model
"Series and maximisation"; contracts/python-api.md §4).

    s = ev.series(policy=AdaptivePolicy())          # m_low: two rungs below the start rung (floor 5), or a rung
    s.evaluate(params, phase="explore")             # at M_low, unchecked: never clean on its own
    s.evaluate(params, phase="final")               # the checked ladder afresh from the guide at this point
    maximise_fixed_sd(ev, start, fixed={"epsilon": v})   # a profile point

**The series** spends few nodes far from the end and the checked M at its final points. A final value never depends on
the call history: it runs the ladder afresh. M_low is set from the guide at the series' first point (or `start`) and
recorded; t_hom (the cost statements' unit, research R8) is timed once at the series' start.

**maximise_fixed_sd** (research R7): A, L-BFGS-B in log parameters at M_low from `start`; B, the checked ladder at
theta_A gives M_c; C, L-BFGS-B at M_c from theta_A; D, the check at theta-hat (l(M_c) against the next rung). If D
fails, M_c is raised one rung and C repeats, up to the cap. The full-M arm (HR-3's comparator) runs B at `start`,
then C and D. The optimiser is pre-registered (OPTIMISER). sd is never free: fitting it is spec 006.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from sraging.engine import mixture as MIX
from sraging.engine.inputs import Heterogeneity
from sraging.errors import UnsupportedOptionError

#: research R7, pre-registered: scipy L-BFGS-B in log parameters, bounds log(start) +- 5
OPTIMISER = {"method": "L-BFGS-B", "bounds_log": 5.0, "maxiter": 500, "ftol": 1e-12, "gtol": 1e-6}
SD_NAMES = ("sd", "het_sd", "het_v")
PHASES = ("explore", "final")


@dataclass(frozen=True)
class AdaptivePolicy:
    """`m_low`: an explicit rung, or None for the rung two below the guide's start rung (floor 5). `raise_M=False` is
    HR-3's never-raise plant: phase A only, at M_low, never clean."""

    m_low: int | None = None
    raise_M: bool = True

    def __post_init__(self):
        if self.m_low is not None and (isinstance(self.m_low, bool) or self.m_low not in MIX.RUNGS):
            raise UnsupportedOptionError("m_low", f"m_low is a rung of {MIX.RUNGS}, got {self.m_low!r}")
        if not isinstance(self.raise_M, bool):
            raise UnsupportedOptionError("raise_M", f"raise_M is True or False, got {self.raise_M!r}")

    def low_rung(self, start_rung: int) -> int:
        if self.m_low is not None:
            return self.m_low
        return MIX.RUNGS[max(MIX.RUNGS.index(start_rung) - 2, 0)]


@dataclass(frozen=True)
class SeriesRecord:
    phases: tuple                         # ({name, M, n_evals, cpu_s, wall_s}, ...) in order of first use
    M_low: int | None
    M_final_by_point: tuple


@dataclass(frozen=True, eq=False)
class MaximisationResult:
    """data-model "Series and maximisation". `phases`: ({name, M, n_evals, cpu_s, wall_s, stated_cpu_s, termination
    {status, nit, message}, ll}, ...); `check`: the final check record; `termination`: {phase name: its last}."""

    theta_hat: object
    ll_final: float
    M_low: int | None
    M_final: int
    phases: tuple
    n_evals: int
    termination: dict
    check: dict
    cost_record: dict
    free: tuple
    fixed: dict
    warnings: tuple = field(default=())
    clean: bool = False


class Series:
    def __init__(self, ev, policy, *, pool=None, start=None):
        self.het = MIX.require_truncated_normal(ev)
        if not isinstance(policy, AdaptivePolicy):
            raise UnsupportedOptionError("policy", "pass a sraging.engine.series.AdaptivePolicy")
        self.ev, self.policy, self.pool = ev, policy, pool
        self.t_hom = None
        self.M_low = None
        self._phases, self._finals = {}, []
        if start is not None:
            self._start(start)

    def _start(self, params):
        """t_hom once, at the series' start, and M_low from the guide there (unless the policy names a rung)."""
        c0, w0 = time.process_time(), time.perf_counter()
        progs = MIX.programs(self.ev)
        z = self.ev._z(params)
        if self.t_hom is None:
            self.t_hom, _ = MIX.time_hom(progs, z)
        if self.M_low is None:
            self.M_low = self.policy.low_rung(MIX.guide(progs, z, self.het.sd, self.ev.cohort.n)["rung"]) \
                if self.policy.m_low is None else self.policy.m_low
        return time.process_time() - c0, time.perf_counter() - w0

    def _count(self, name, M, cpu, wall, n=1):
        ph = self._phases.setdefault(name, {"name": name, "M": M, "n_evals": 0, "cpu_s": 0.0, "wall_s": 0.0})
        ph["M"], ph["n_evals"] = M, ph["n_evals"] + n
        ph["cpu_s"] += cpu
        ph["wall_s"] += wall

    def evaluate(self, params, phase="explore", *, grad=False):
        """A MixtureResult: at M_low ("explore", verdict "not_checked: explore", never clean) or the checked ladder
        afresh ("final")."""
        if phase not in PHASES:
            raise UnsupportedOptionError("phase", f"phase is one of {PHASES}, got {phase!r}")
        cpu0, wall0 = self._start(params)
        if phase == "explore":
            het = Heterogeneity(parameter=self.het.parameter, sd=self.het.sd, M=self.M_low, check=False)
            r = MIX.mixture_value(self.ev, params, grad=grad, pool=self.pool, het=het, t_hom=self.t_hom, explore=True)
        else:
            het = Heterogeneity(parameter=self.het.parameter, sd=self.het.sd)
            r = MIX.mixture_value(self.ev, params, grad=grad, pool=self.pool, het=het, t_hom=self.t_hom)
            self._finals.append(r.node_record["M_used"])
        self._count(phase, r.node_record["M_used"], cpu0 + r.cost_record["total_cpu_s"], wall0 + r.cost_record["wall_s"])
        return r

    def record(self) -> SeriesRecord:
        return SeriesRecord(phases=tuple(dict(p) for p in self._phases.values()), M_low=self.M_low,
                            M_final_by_point=tuple(self._finals))


# -------------------------------------------------------------------------------------------- maximise_fixed_sd
def _names(ev, free, fixed):
    names = ev.layout.names
    fixed = dict(fixed or {})
    free = tuple(free) if free is not None else tuple(n for n in names if n not in fixed)
    sd = [n for n in free if n in SD_NAMES]
    if sd:
        raise UnsupportedOptionError("free", f"{sd} in free: sd is fixed in a series; fitting it is spec 006")
    for n in free:
        if n not in names or n.startswith("seg:"):
            raise UnsupportedOptionError("free", f"{n!r} is not a free model parameter of this evaluator "
                                         f"({names}; segment values are spec 006's)")
    for n in fixed:
        if n not in names:
            raise UnsupportedOptionError("fixed", f"{n!r} is not free in this evaluator, so it cannot be held for a "
                                         f"profile point (build the evaluator with it free)")
    if set(free) & set(fixed) or set(free) | set(fixed) != set(names):
        raise UnsupportedOptionError("free", f"every free parameter of the evaluator ({names}) is in exactly one of "
                                     f"free {free} and fixed {sorted(fixed)}")
    return free, fixed


def maximise_fixed_sd(ev, start, *, policy=None, full_M=False, fixed=None, free=None, pool=None) -> MaximisationResult:
    """research R7's A-D (or B-D with full_M=True) at the evaluator's sd; `fixed` holds parameters for a profile
    point. `start`: SRParams."""
    from scipy.optimize import minimize
    from sraging.params import SRParams
    policy = policy or AdaptivePolicy()
    het = MIX.require_truncated_normal(ev)
    free, fixed = _names(ev, free, fixed)
    base = {**start.as_dict(), **{k: float(v) for k, v in fixed.items()}}
    c_all, w_all = time.process_time(), time.perf_counter()
    ser = Series(ev, policy, pool=pool, start=SRParams(**base))
    t_hom = ser.t_hom
    x0 = np.log([base[n] for n in free])
    bounds = [(x - OPTIMISER["bounds_log"], x + OPTIMISER["bounds_log"]) for x in x0]
    phases, wcpu_all = [], [0.0]

    def params_of(x):
        return SRParams(**{**base, **{n: float(math.exp(v)) for n, v in zip(free, x)}})

    def H(M, check=False):
        return Heterogeneity(parameter=het.parameter, sd=het.sd, M=M, check=check)

    def value(p, M, grad, explore=False):
        r = MIX.mixture_value(ev, p, grad=grad, pool=pool, het=H(M), t_hom=t_hom, explore=explore)
        wcpu_all[0] += r.cost_record["worker_cpu_s"]
        return r

    def run_phase(name, M, work):
        c0, w0, k0 = time.process_time(), time.perf_counter(), wcpu_all[0]
        out, n, stated, term, ll = work()
        phases.append({"name": name, "M": M, "n_evals": n, "cpu_s": time.process_time() - c0 + wcpu_all[0] - k0,
                       "wall_s": time.perf_counter() - w0, "stated_cpu_s": stated, "termination": term, "ll": ll})
        return out

    def optimise(M, x_start, explore):
        def work():
            n, stated = [0], [0.0]

            def f(x):
                r = value(params_of(x), M, True, explore)
                n[0] += 1
                stated[0] += sum(s["stated_cpu_s"] for s in r.cost_record["statements"])
                return -r.value, -np.array([r.grad[k] for k in free])
            res = minimize(f, x_start, jac=True, method=OPTIMISER["method"], bounds=bounds,
                           options={"maxiter": OPTIMISER["maxiter"], "ftol": OPTIMISER["ftol"],
                                    "gtol": OPTIMISER["gtol"]})
            term = {"status": "converged" if res.success else "not_converged", "nit": int(res.nit),
                    "message": str(res.message)}
            return res.x, n[0], stated[0], term, float(-res.fun)
        return work

    def ladder_at(x):
        def work():
            r = MIX.mixture_value(ev, params_of(x), pool=pool, het=Heterogeneity(parameter=het.parameter, sd=het.sd),
                                  t_hom=t_hom)
            wcpu_all[0] += r.cost_record["worker_cpu_s"]
            c = r.node_record["check"]
            return (r.node_record["M_used"], c), 1, sum(s["stated_cpu_s"] for s in r.cost_record["statements"]), \
                {"status": c["verdict"], "nit": len(c["ladder"]), "message": f"M = {r.node_record['M_used']}"}, r.value
        return work

    def check_at(x, M):
        """D: l(M) against the next rung (at the cap, the pair (128, 192), as the ladder's top check)."""
        def work():
            lo, hi = (M, MIX.RUNGS[MIX.RUNGS.index(M) + 1]) if M < MIX.CAP else (MIX.RUNGS[-2], MIX.CAP)
            rl, rh = value(params_of(x), lo, False), value(params_of(x), hi, False)
            ok = abs(rl.value - rh.value) < MIX.THRESHOLD
            verdict = "converged" if ok else ("unconverged_at_cap" if hi == MIX.CAP else "unconverged")
            chk = MIX._check_record(lo, hi, rl.value, rh.value, verdict,
                                    [{"M": lo, "ll": rl.value, "cpu_s": rl.cost_record["total_cpu_s"]},
                                     {"M": hi, "ll": rh.value, "cpu_s": rh.cost_record["total_cpu_s"]}], None)
            stated = sum(s["stated_cpu_s"] for r in (rl, rh) for s in r.cost_record["statements"])
            return chk, 2, stated, {"status": verdict, "nit": 1, "message": f"|l({lo}) - l({hi})| = "
                                                                             f"{chk['abs_dll']:.3g}"}, rl.value
        return work

    M_low = None
    if full_M:
        x = x0
    else:
        M_low = ser.M_low
        x = run_phase("A explore", M_low, optimise(M_low, x0, True))
        if not policy.raise_M:                                                   # HR-3's never-raise plant
            r = value(params_of(x), M_low, False, True)
            chk = r.node_record["check"]
            return _result(ev, params_of(x), r.value, M_low, M_low, phases, chk, het, free, fixed, t_hom, c_all,
                           w_all, wcpu_all[0], pool)
    M_c, _ = run_phase("B check", None, ladder_at(x))
    phases[-1]["M"] = M_c
    while True:
        x = run_phase("C polish", M_c, optimise(M_c, x, False))
        chk = run_phase("D check", M_c, check_at(x, M_c))
        if chk["verdict"] == "converged" or M_c == MIX.CAP:
            break
        M_c = MIX.RUNGS[MIX.RUNGS.index(M_c) + 1]
    ll = chk["ll_check"] if chk["M_check"] == M_c else chk["ll_M"]
    return _result(ev, params_of(x), ll, M_low, M_c, phases, chk, het, free, fixed, t_hom, c_all, w_all, wcpu_all[0],
                   pool)


def _result(ev, theta, ll, M_low, M_final, phases, chk, het, free, fixed, t_hom, c0, w0, wcpu, pool):
    from sraging.warnings import sampler_warning
    warnings = list(het.warnings)
    if chk["verdict"] in ("unconverged", "unconverged_at_cap"):
        warnings.append(sampler_warning("HET_UNCONVERGED", M=chk["M"], M_check=chk["M_check"],
                                        abs_dll=chk["abs_dll"], threshold=chk["threshold"]))
    elif chk["verdict"].startswith("not_checked"):
        warnings.append(sampler_warning("HET_NOT_CHECKED", M=chk["M"]))
    cost = {"workers": pool.workers if pool is not None else 1, "t_hom_cpu_s": t_hom,
            "statements": [{"phase": p["name"], "M": p["M"], "n_evals": p["n_evals"], "stated_cpu_s": p["stated_cpu_s"],
                            "measured_cpu_s": p["cpu_s"]} for p in phases],
            "measured_cpu_s": sum(p["cpu_s"] for p in phases), "worker_cpu_s": wcpu,
            "total_cpu_s": time.process_time() - c0 + wcpu, "wall_s": time.perf_counter() - w0}
    return MaximisationResult(theta_hat=theta, ll_final=float(ll), M_low=M_low, M_final=M_final, phases=tuple(phases),
                              n_evals=sum(p["n_evals"] for p in phases),
                              termination={p["name"]: p["termination"] for p in phases}, check=chk,
                              cost_record=cost, free=free, fixed=dict(fixed), warnings=tuple(warnings),
                              clean=not warnings)
