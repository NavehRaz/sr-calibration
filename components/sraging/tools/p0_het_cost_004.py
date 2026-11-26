"""Spec 004's pilot P0, the heterogeneity cost (research R8, R19; contracts/acceptance.md §1; tasks T036, T039). Its
result freezes the cost factors at freeze-H (panel_004.toml [cost]) and prices the heterogeneity phases.

    .venv/bin/python tools/acceptance_004.py p0-het-cost       # the RUN (tasks T039); this module is its work

On 6 panel-external cohorts (COHORTS: the het-tradeoff pilot's, which the panel holds out), broad to human, at sd 0.2
and M = 31 (research R4's probe point), per cohort:
- **t_hom**, the cost statements' unit (mixture.time_hom), and the per-node CPU of a value (pass 1) and of a value and
  gradient (both passes), over REPEATS and GRAD_REPEATS runs, compilation apart: f_value = per-node value / t_hom,
  f_grad = per-node value-and-gradient / t_hom;
- **the compilation** of the evaluator's per-node programs;
- **the timing noise**: the sd of log CPU over REPEATS value runs;
- **the gradients against central differences**: the per-node two-pass gradient and today's vmapped one (spec 002's
  program with the same truncated-normal nodes), each against central differences in every log parameter, then
  against each other (research R4's 2.1e-7 on Sweden);
- **node finiteness at M = 192**: every node of the cap's rule finite (a non-finite one is refused by name).
Then a W in {1, 2, 3} identity smoke on the smallest cohort: pooled value and gradient bytes equal in-process.

The result carries what `acceptance_004.p0_per_node()` reads: per_node_cpu_s {value, value_and_grad} (medians over
the cohorts) and compile_cpu_s (the median per evaluator), with f_value, f_grad and the noise.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: the het-tradeoff pilot's cohorts, held out of the panel (panel_004.HELD_OUT): broad to human
COHORTS = ("het-real-Guiniea_pig_VC-c0-s0", "het-k0.01-p03-c1-s0", "het-k0.01-p00-c1-s0",
           "het-real-Jack_Russell_vetCompass-c0-s0", "het-real-combined_human_F-c2-s0",
           "het-real-Sweden_F_1910_homo-c0-s0")
SD, M, M_CAP = 0.2, 31, 192
REPEATS, GRAD_REPEATS = 20, 5
CD_STEP = 1e-5
SMOKE_W = (1, 2, 3)
FREE = ("eta", "beta", "epsilon", "kappa")


def _het(M_, check=False):
    from sraging.engine.inputs import Heterogeneity
    return Heterogeneity(parameter="xc", sd=SD, M=M_, check=check)


def _cpu(r, phases=("rung", "pass 2")) -> float:
    return sum(s["measured_cpu_s"] for s in r.cost_record["statements"] if s["phase"] in phases)


def cd_grad(value_of, theta, h=CD_STEP) -> dict:
    """Central differences of value_of(SRParams) in each log parameter."""
    from sraging import SRParams
    d = theta.as_dict()
    out = {}
    for n in FREE:
        up = SRParams(**{**d, n: d[n] * math.exp(h)})
        dn = SRParams(**{**d, n: d[n] * math.exp(-h)})
        out[n] = (value_of(up) - value_of(dn)) / (2 * h)
    return out


def _rel(a: dict, b: dict) -> float:
    return max(abs(a[k] - b[k]) / max(1.0, abs(b[k])) for k in a)


def measure(cid, man, specs) -> dict:
    """One cohort's P0 row."""
    import acceptance_004 as ACC
    from sraging.engine import mixture as MIX
    from sraging.engine import quadrature as Q
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.inputs import Heterogeneity
    coh, theta, h = ACC.v6_case(cid, man, specs)
    ev = ACC.evaluator_for(coh, h)
    progs = MIX.programs(ev)
    c0 = time.process_time()
    MIX.mixture_value(ev, theta, grad=True, het=_het(M))                      # compiles the per-node programs
    first_cpu = time.process_time() - c0
    compile_cpu = sum(progs.compile_cpu_s.values())
    z = ev._z(theta)
    t_hom, _ = MIX.time_hom(progs, z)
    val = [_cpu(MIX.mixture_value(ev, theta, het=_het(M), t_hom=t_hom)) for _ in range(REPEATS)]
    vg = [_cpu(MIX.mixture_value(ev, theta, grad=True, het=_het(M), t_hom=t_hom)) for _ in range(GRAD_REPEATS)]
    node_v, node_vg = float(np.median(val)) / M, float(np.median(vg)) / M
    # the gradients: per node (two-pass) and vmapped (spec 002's program, the same nodes), each against CD first
    r = MIX.mixture_value(ev, theta, grad=True, het=_het(M), t_hom=t_hom)
    g_node = {k: r.grad[k] for k in FREE}
    cd_node = cd_grad(lambda p: MIX.mixture_value(ev, p, het=_het(M), t_hom=t_hom).value, theta)
    rule = Q.truncated_normal_rule(SD, M)
    old = Evaluator(coh, numerics="v1", fixed={"h_ext": h},
                    heterogeneity=Heterogeneity(parameter="xc", definition="normal_frac_sd", v=SD, M=M))
    old._het_z, old._het_w = np.asarray(rule.z), np.asarray(rule.w)
    c1 = time.process_time()
    v_old, g_old = old.value_and_grad(theta)
    vmap_vg_cpu = time.process_time() - c1
    cd_old = cd_grad(old.value, theta)
    # node finiteness at the cap
    rcap = MIX.mixture_value(ev, theta, het=_het(M_CAP), t_hom=t_hom)
    return {"cohort_id": cid, "n": coh.n, "sd": SD, "M": M, "t_hom_cpu_s": t_hom, "first_call_cpu_s": first_cpu,
            "compile_cpu_s": compile_cpu, "compile_by_program": dict(progs.compile_cpu_s),
            "value_cpu_s": val, "value_and_grad_cpu_s": vg, "per_node_value_cpu_s": node_v,
            "per_node_value_and_grad_cpu_s": node_vg, "f_value": node_v / t_hom, "f_grad": node_vg / t_hom,
            "log_sd_value": float(np.std(np.log(val), ddof=1)),
            "gradients": {"per_node": g_node, "per_node_cd": cd_node, "per_node_rel_vs_cd": _rel(g_node, cd_node),
                          "vmapped": g_old, "vmapped_cd": cd_old, "vmapped_rel_vs_cd": _rel(g_old, cd_old),
                          "per_node_vs_vmapped_rel": _rel(g_node, g_old),
                          "value_per_node": r.value, "value_vmapped": v_old, "vmapped_vg_cpu_s": vmap_vg_cpu},
            "finite_at_192": {"value": rcap.value, "finite": bool(np.isfinite(rcap.value)),
                              "s_min": rcap.node_record["s_min"]}}


def identity_smoke(cid, man, specs, workers=SMOKE_W) -> list:
    """Pooled value and gradient bytes against in-process, M = 31, at each W."""
    import acceptance_004 as ACC
    from sraging.engine import mixture as MIX
    from sraging.engine.pool import NodePool
    coh, theta, h = ACC.v6_case(cid, man, specs)
    ev = ACC.evaluator_for(coh, h)
    r0 = MIX.mixture_value(ev, theta, grad=True, het=_het(M))
    out = []
    for W in workers:
        with NodePool(ev, workers=W) as pool:
            r = MIX.mixture_value(ev, theta, grad=True, het=_het(M), pool=pool)
            out.append({"W": W, "value_equal": r.value.hex() == r0.value.hex(),
                        "grad_equal": {k: r.grad[k].hex() == r0.grad[k].hex() for k in r.grad},
                        "worker_cpu_s": r.cost_record["worker_cpu_s"],
                        "compile_cpu_s_workers": dict(pool.compile_cpu_s)})
    return out


def summarise(rows, smoke) -> dict:
    """The result's summary: what acceptance_004.p0_per_node() reads, and the factors freeze-H freezes."""
    med = lambda k: float(np.median([r[k] for r in rows]))                  # noqa: E731
    return {"per_node_cpu_s": {"value": med("per_node_value_cpu_s"), "value_and_grad": med("per_node_value_and_grad_cpu_s")},
            "compile_cpu_s": med("compile_cpu_s"), "f_value": med("f_value"), "f_grad": med("f_grad"),
            "f_value_range": [min(r["f_value"] for r in rows), max(r["f_value"] for r in rows)],
            "f_grad_range": [min(r["f_grad"] for r in rows), max(r["f_grad"] for r in rows)],
            "noise_log_sd": {"max": max(r["log_sd_value"] for r in rows), "median": med("log_sd_value")},
            "gradients_rel_vs_cd_max": {"per_node": max(r["gradients"]["per_node_rel_vs_cd"] for r in rows),
                                        "vmapped": max(r["gradients"]["vmapped_rel_vs_cd"] for r in rows)},
            "all_finite_at_192": all(r["finite_at_192"]["finite"] for r in rows),
            "identity_smoke_all_equal": bool(smoke) and all(x["value_equal"] and all(x["grad_equal"].values())
                                                            for x in smoke)}


def projection_cpu_h(probe: dict, n_cohorts=len(COHORTS)) -> float:
    """P0's price from research R4's probe (specs/004-samplers/evidence/cost_probe.json), per cohort: the first calls
    (compilation), REPEATS + 3 values and GRAD_REPEATS + 1 values-and-gradients at M = 31, 8 central-difference values
    on each path, the vmapped value and gradient, M = 192 once; and the smoke's 6 workers' compilation."""
    per = []
    for c in probe["cohorts"].values():
        first = sum(c[k]["first_cpu_s"] for k in ("mixture_M31_per_node_value", "mixture_M31_per_node_value_and_grad",
                                                    "mixture_M31_vmap_value", "mixture_M31_vmap_value_and_grad"))
        v, vg = c["mixture_M31_per_node_value"]["median_cpu_s"], c["mixture_M31_per_node_value_and_grad"]["median_cpu_s"]
        vv, vvg = c["mixture_M31_vmap_value"]["median_cpu_s"], c["mixture_M31_vmap_value_and_grad"]["median_cpu_s"]
        per.append(first + (REPEATS + 3 + 8) * v + (GRAD_REPEATS + 1) * vg + 8 * vv + vvg + v * M_CAP / M)
    smoke = sum(SMOKE_W) * float(np.mean([c["mixture_M31_per_node_value_and_grad"]["first_cpu_s"]
                                          for c in probe["cohorts"].values()]))
    return (n_cohorts * float(np.mean(per)) + smoke) / 3600.0
