"""Spec 004's S4, FP against Langevin (tasks T077, T079; research R18; spec FR-015, HR-6). The phase `s4` in
tools/acceptance_004.py runs it under SR (cap 35 CPU-h).

**The grid** is frozen at freeze-L (`langevin_004.toml [s4]`): per heterogeneous parameter (eta, beta, epsilon, kappa,
xc) x cv for each of its presets, plus v = 0 cells for Jack Russell and E. coli (C13), at n = 25,000, on the default
clamp; reflect on P1's 8 cells (each preset, homogeneous and xc sd 0.2). Each row starts at P1's step for its preset,
heterogeneity class (hom, or het for any cv > 0) and wall, and its ladder is capped at m_max = 2 m (gate C2). A row
without a P1 step is refused.

**E. coli** has no clamp row (`clamp_excluded`; Naveh 2026-11-03): the default clamp is not shown to agree with FP for
it at any affordable step; its reflect rows stay.

**A cell at a frozen step** (`run_frozen`; research R18's branch, amendment A1, Naveh 2026-11-03): P1's noise floor
exceeds crit/30, so each step is frozen from p1-pooled and not re-measured: the row runs once, KS judged on its dt/2
rung against the FP reference that meets R26's curve check; its assessability is the frozen cell's pooled bias, and a
borrowed row (any (parameter, cv) other than xc 0.2) is guarded by its own D against 3x its cell's mean per-key D.
`run_cell`, the coupled ladder to b(dt/2) <= crit/10 or m_max, stays for steps that are not frozen. **The plant:** on its 2 cells (one homogeneous, one xc sd 0.2), Langevin at eta(1 + delta) against
FP at eta, delta sized so the predicted sup|dS| is 2 x 0.0165 (`plant_delta`).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CRIT = {"clamp": 1.973, "reflect": 1.698}          # sraging.sampler_rules.HR6_CRIT
PLANT_TARGET = 2 * 0.0165                          # the plant's predicted sup|dS| (gate C2)
REFLECT_P1 = (("hom", None, 0.0), ("het", "xc", 0.2))


class Refused(Exception):
    """A grid S4 cannot be built from."""


def crit(wall, n) -> float:
    return CRIT[wall] / math.sqrt(n)


def step_key(preset, het, wall) -> str:
    """P1's step's key: preset | hom or het | wall (P1 measures each preset homogeneous and at xc sd 0.2)."""
    return f"{preset}|{het}|{wall}"


def _row(rid, preset, parameter, cv, wall, n, p1):
    k = step_key(preset, "het" if cv > 0 else "hom", wall)
    if k not in p1["steps"]:
        raise Refused(f"row {rid}: no P1 step for {k} (S4 starts each row at P1's step)")
    st = p1["steps"][k]
    return {"id": rid, "preset": preset, "parameter": parameter, "cv": float(cv), "wall": wall, "n": int(n),
            "m": int(st["m"]), "dt": float(st["dt"]), "m_max": 2 * int(st["m"]), "p1_step": k}


def build_grid(spec: dict, p1: dict) -> dict:
    """{"clamp": rows, "reflect": rows} from the frozen [s4] spec and P1's steps. A preset in `clamp_excluded` (E.
    coli, Naveh 2026-11-03: the default clamp is not shown to agree with FP for it at any affordable step) has no
    clamp row; its reflect rows stay."""
    n = int(spec["n"])
    out_ = set(spec.get("clamp_excluded") or ())
    clamp = [_row(f"{p}/{q}/cv{cv:g}/clamp", p, q, cv, "clamp", n, p1)
             for p in spec["het_presets"] if p not in out_ for q in spec["parameters"] for cv in spec["cvs"]]
    clamp += [_row(f"{p}/v0/clamp", p, None, 0.0, "clamp", n, p1) for p in spec["v0_presets"] if p not in out_]
    presets = list(dict.fromkeys(c["preset"] for c in p1["cells"]))
    reflect = [_row(f"{p}/{q or 'v0'}/cv{cv:g}/reflect", p, q, cv, "reflect", n, p1) for p in presets
               for _, q, cv in REFLECT_P1]
    return {"clamp": clamp, "reflect": reflect}


def _het(parameter, cv):
    from sraging.engine.inputs import Heterogeneity
    return None if not cv else Heterogeneity(parameter=parameter, sd=float(cv))


def plant_delta(params, het=None, target=PLANT_TARGET, lo=0.0, hi=0.5, tol=1e-4) -> dict:
    """delta with the predicted sup_t |S_obs(eta (1 + delta)) - S_obs(eta)| = target on the FP model grid (bisection;
    one evaluator, its node count by R26's curve check at the target's scale)."""
    from sraging import SRParams
    from sraging.langevin.sample import LangevinConfig, Reference
    base = Reference(LangevinConfig(params=params, n=25_000, key=0, dt=1.0, heterogeneity=het, error="ks"))
    M = None
    if het is not None:
        tau, _ = base.survival_grid(5)
        t = tau * base.L
        M = base.curve_check(t[base.S_obs(t, 5) > 1e-12], target / 100)["M"]
    tau, S0 = base.survival_grid(M)
    t = tau * base.L

    def sup(delta):
        alt = Reference.at(base, SRParams(**{**params.as_dict(), "eta": params.eta * (1.0 + delta)}))
        return float(np.max(np.abs(alt.S_obs(t, M) - base.S_obs(t, M))))
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sup(mid) < target else (lo, mid)
    d = 0.5 * (lo + hi)
    return {"delta": d, "predicted_sup_dS": sup(d), "target": target, "M": M}


def run_cell(row, params, key, p_hat, *, het=None) -> dict:
    """One row: the coupled ladder from P1's step to b(dt/2) <= crit/10 or m_max; KS judged on the dt/2 rung."""
    from sraging.langevin import simulate
    cr = crit(row["wall"], row["n"])
    m, dt, ladder = row["m"], row["dt"], []
    p = min(max(float(p_hat), 0.5), 1.0)
    while True:
        s = simulate(params, n=row["n"], key=key, dt=dt, heterogeneity=het, wall=row["wall"], error="ladder")
        b = s.error.step_bias
        jb = b["D_dt_vs_dt2"] / (2.0 ** p - 1.0)
        ladder.append({"m": m, "dt": dt, "D": b["D_dt_vs_dt2"], "judged_bias": jb, "ks_D_half": b["ks_D_half"],
                       "p_value_half": b["p_value_half"], "ks_D_dt": s.error.ks_D, "cpu_s": s.cost_record["cpu_s"],
                       "M_ref": s.error.fp_reference["M"]})
        if jb <= cr / 10 or 2 * m > row["m_max"]:
            break
        m, dt = 2 * m, dt / 2
    last = ladder[-1]
    return {"id": row["id"], "wall": row["wall"], "n": row["n"], "D": last["D"], "ks_D": last["ks_D_half"],
            "p_hat": p, "m": m, "m_max": row["m_max"], "crit": cr, "ladder": ladder, "key": key}


def borrowed(row) -> bool:
    """A heterogeneous row whose (parameter, cv) is not its step cell's measured one (xc, 0.2) borrows that step."""
    return row["cv"] > 0 and (row["parameter"], row["cv"]) != ("xc", 0.2)


def run_frozen(row, params, key, frozen, *, het=None) -> dict:
    """R18's branch (amendment A1, Naveh 2026-11-03): the row ONCE at its frozen step (no per-row ladder, no
    doubling); KS judged on its dt/2 rung; its assessability is the frozen cell's pooled bias (`frozen_b`), and a
    borrowed row is guarded by its own coupled D against 3x its cell's mean per-key D (sampler_rules.frozen_bias)."""
    from sraging.langevin import simulate
    s = simulate(params, n=row["n"], key=key, dt=row["dt"], heterogeneity=het, wall=row["wall"], error="ladder")
    b = s.error.step_bias
    return {"id": row["id"], "wall": row["wall"], "n": row["n"], "ks_D": b["ks_D_half"], "p_value": b["p_value_half"],
            "ks_D_dt": s.error.ks_D, "D_own": b["D_dt_vs_dt2"], "D": b["D_dt_vs_dt2"], "p_hat": frozen["p_hat"],
            "frozen_b": frozen["pooled_b"], "D_key_mean": frozen["D_key_mean"], "borrowed": borrowed(row),
            "m": row["m"], "m_max": row["m_max"], "dt": row["dt"], "step": row["p1_step"], "key": key,
            "cpu_s": s.cost_record["cpu_s"], "M_ref": s.error.fp_reference["M"]}


def run_plant(row, params, key, delta, *, het=None) -> dict:
    """The eta(1 + delta) plant: a Langevin sample at eta (1 + delta), its raw times against FP at eta."""
    from sraging import SRParams
    from sraging.langevin import simulate
    from sraging.langevin.sample import LangevinConfig, Reference, ks_one_sample
    p = SRParams(**{**params.as_dict(), "eta": params.eta * (1.0 + delta)})
    s = simulate(p, n=row["n"], key=key, dt=row["dt"], heterogeneity=het, wall=row["wall"], error="ks")
    ref = Reference(LangevinConfig(params=params, n=row["n"], key=key, dt=row["dt"], heterogeneity=het, error="ks"))
    M = None if het is None else ref.curve_check(s.times, crit("clamp", row["n"]) / 10)["M"]
    kinds = (s.times < s.provenance["t_max"]).astype(np.int8)
    D, pv = ks_one_sample(s.times, kinds, lambda t: ref.S_obs(t, M))
    return {"id": row["id"] + "/plant", "n": row["n"], "ks_D": D, "p_value": pv, "delta": delta, "dt": row["dt"]}
