"""Spec 004's HR-11, heterogeneity in more than one parameter (tasks T109, T126; research R27-R29; contracts/
acceptance.md §4). The phase `multi` in tools/acceptance_004.py runs it under SR (cap 4 CPU-h).

The configuration is frozen at freeze-L (`langevin_004.toml [multi]`): the preset, the 6 cells (xc+eta, xc+epsilon,
eta+beta, epsilon+kappa, xc+eta+epsilon, eta+beta+kappa; sd 0.2 each), n = 25,000, G = 6, the reduction cases, the
dropped-spread plants with their predicted sizes, the composite cell and the above-budget configuration.
- **(i) The reductions** (`reductions`): each case, a tuple with every other member at sd 0, against its
  single-parameter draw, bytes of (times, ages, events), under both walls; the shared-stream plant
  (`sample.draw_parameters_shared_stream`) must break the identity.
- **(ii) A cell** (`run_cell`): the default clamp at P1's heterogeneous step for the preset; the coupled ladder halves
  dt until the judged dt/2 sample's bias D / (2^p - 1) is <= crit/10 (crit = 1.655 / sqrt(n)) or m reaches m_max
  = 2 m; the KS of the dt/2 rung against the tensor reference (R26 per dimension). **The plants** (`plant_size`,
  `run_plant`): a draw with one member's spread dropped, against the full tensor reference; its predicted
  sup|dS| between the full and the dropped tensor references is computed first. **The composite** (`composite`):
  Gauss-Legendre panels per dimension (quadrature.composite_rule) refined until two refinements agree, against the
  tensor reference: sup|dS| <= 1e-4.
- **(iii) The above-budget statement** (`above_budget`): it must read "no FP reference", not clean.
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

import s4_fp_langevin_004 as S4  # noqa: E402

G = 6
WALLS = ("clamp", "reflect")


class Refused(Exception):
    """A configuration HR-11 cannot be built from."""


def crit(n) -> float:
    from sraging.sampler_rules import HR11_CRIT
    return HR11_CRIT / math.sqrt(n)


def members(d: dict) -> tuple:
    """{parameter: sd} -> the canonical tuple of truncated normals (sd 0: Heterogeneity.zero)."""
    from sraging.engine.inputs import Heterogeneity
    return tuple(Heterogeneity(parameter=p, sd=float(sd)) if sd > 0 else Heterogeneity.zero(p)
                 for p, sd in sorted(d.items()))


def build_cells(spec: dict, p1: dict) -> list:
    """The 6 cells on the default clamp, each at P1's heterogeneous clamp step for the preset (m_max = 2 m)."""
    k = S4.step_key(spec["preset"], "het", "clamp")
    if k not in p1["steps"]:
        raise Refused(f"no P1 step for {k} (HR-11 starts each cell at P1's heterogeneous step)")
    if len(spec["cells"]) != G or int(spec.get("G", G)) != G:
        raise Refused(f"[multi] has {len(spec['cells'])} cells; HR-11's family is G = {G}")
    st = p1["steps"][k]
    rows = []
    for c in spec["cells"]:
        if len(c["members"]) > 3:
            raise Refused(f"{c['id']}: {len(c['members'])} members; the tensor reference takes d <= 3")
        rows.append({"id": c["id"], "preset": spec["preset"], "members": dict(c["members"]), "wall": "clamp",
                     "n": int(spec["n"]), "m": int(st["m"]), "dt": float(st["dt"]), "m_max": 2 * int(st["m"]),
                     "p1_step": k})
    return rows


# --------------------------------------------------------------------------------------------------------- (i)
def reductions(params, cases, *, n, key, dt, t_max) -> list:
    """Each case {"single": {p: sd}, "zero": [q, ...]} under both walls: the tuple (with the zero members) against the
    single-parameter draw, bytes; and the shared-stream plant on the first case."""
    from sraging.engine.inputs import Heterogeneity
    from sraging.langevin import sample as LS
    from sraging.langevin import simulate
    rows = []
    for case in cases:
        single = members(case["single"])
        tup = tuple(sorted(single + tuple(Heterogeneity.zero(q) for q in case["zero"]), key=lambda h: h.parameter))
        one = single[0] if len(single) == 1 else single
        cid = "+".join(sorted(case["single"])) + "+" + "+".join(f"{q}0" for q in case["zero"])
        for w in WALLS:
            kw = dict(n=n, key=key, dt=dt, wall=w, error="ks", t_max=t_max)
            a, b = simulate(params, heterogeneity=tup, **kw), simulate(params, heterogeneity=one, **kw)
            rows.append({"id": f"{cid}/{w}", "kind": "case", "wall": w,
                         "equal": all(getattr(a, f).tobytes() == getattr(b, f).tobytes()
                                      for f in ("times", "ages", "events"))})
    case = cases[0]
    single = members(case["single"])
    tup = single + tuple(Heterogeneity.zero(q) for q in case["zero"])
    base = dict(params=params, n=n, key=key, dt=dt, error="ks", t_max=t_max)
    pa = LS.draw_parameters_shared_stream(LS.LangevinConfig(heterogeneity=tup, **base))
    pb = LS.draw_parameters_shared_stream(LS.LangevinConfig(heterogeneity=single, **base))
    rows.append({"id": "shared-stream", "kind": "plant", "breaks": any(pa[k].tobytes() != pb[k].tobytes() for k in pa)})
    return rows


# -------------------------------------------------------------------------------------------------------- (ii)
def _tensor(params, d: dict, bar, **kw):
    from sraging.engine.tensor_reference import TensorReference
    return TensorReference(params, members(d), bar=bar, **kw)


def plant_size(params, d: dict, drop: str, *, bar=1e-4) -> dict:
    """The dropped-spread plant's predicted sup_t |S_obs(full) - S_obs(drop's sd = 0)|, both at the full reference's
    node counts (the dropped dimension one node)."""
    full = _tensor(params, d, bar)
    if full.kind != "tensor":
        raise Refused(f"the full tensor reference reads {full.kind}")
    dropped = _tensor(params, {**d, drop: 0.0}, bar, M=full.M, ev=full._ev)
    t = full.tau * full.L
    return {"drop": drop, "predicted_sup_dS": float(np.max(np.abs(full.S_obs(t) - dropped.S_obs(t)))),
            "M": dict(full.M), "bar": bar}


def run_cell(row, params, key, p_hat) -> dict:
    """One cell: the coupled ladder from P1's step to b(dt/2) <= crit/10 or m_max; KS judged on the dt/2 rung against
    the tensor reference."""
    from sraging.langevin import simulate
    cr = crit(row["n"])
    p = min(max(float(p_hat), 0.5), 1.0)
    m, dt, ladder = row["m"], row["dt"], []
    het = members(row["members"])
    while True:
        s = simulate(params, n=row["n"], key=key, dt=dt, heterogeneity=het, wall=row["wall"], error="ladder")
        b = s.error.step_bias
        jb = b["D_dt_vs_dt2"] / (2.0 ** p - 1.0)
        ref = s.error.fp_reference
        ladder.append({"m": m, "dt": dt, "D_bias": b["D_dt_vs_dt2"], "judged_bias": jb, "ks_D_half": b["ks_D_half"],
                       "ks_D_dt": s.error.ks_D, "cpu_s": s.cost_record["cpu_s"], "reference": ref})
        if jb <= cr / 10 or 2 * m > row["m_max"]:
            break
        m, dt = 2 * m, dt / 2
    ref = ladder[-1]["reference"]
    verdict = (ref.get("curve_check") or {}).get("verdict")
    return {"id": row["id"], "n": row["n"], "D": ladder[-1]["ks_D_half"], "D_bias": ladder[-1]["D_bias"],
            "p_hat": p, "m": m, "m_max": row["m_max"], "crit": cr, "reference_kind": ref["kind"],
            "reference_verdict": verdict, "M": ref.get("M"), "ladder": ladder, "key": key}


def run_frozen(row, params, key, frozen) -> dict:
    """R18's branch for HR-11 (amendment A1, Naveh 2026-11-03): the cell ONCE at the preset's frozen heterogeneous
    clamp step (a borrowed step: the multi-parameter cell was not measured), KS judged on the dt/2 rung against the
    tensor reference; assessability from the frozen cell's pooled bias, checked against HR-11's own bar (crit 1.655 /
    sqrt(n) / 10) by the rule, and the borrowed-row guard (its own D against 3x the cell's mean per-key D)."""
    from sraging.langevin import simulate
    s = simulate(params, n=row["n"], key=key, dt=row["dt"], heterogeneity=members(row["members"]), wall=row["wall"],
                 error="ladder")
    b, ref = s.error.step_bias, s.error.fp_reference
    return {"id": row["id"], "n": row["n"], "D": b["ks_D_half"], "D_own": b["D_dt_vs_dt2"],
            "D_bias": b["D_dt_vs_dt2"], "p_hat": float(frozen["p_hat"]), "frozen_b": frozen["pooled_b"],
            "D_key_mean": frozen["D_key_mean"], "borrowed": True, "m": row["m"], "m_max": row["m_max"],
            "crit": crit(row["n"]), "reference_kind": ref["kind"],
            "reference_verdict": (ref.get("curve_check") or {}).get("verdict"), "M": ref.get("M"), "key": key,
            "dt": row["dt"], "step": row["p1_step"], "cpu_s": s.cost_record["cpu_s"]}


def run_plant(row, params, drop, key, predicted) -> dict:
    """The dropped-spread draws (drop's sd at 0) against the full tensor reference, the KS at the cell's step."""
    from sraging.langevin import sample as LS
    from sraging.langevin import simulate
    d = {**row["members"], drop: 0.0}
    s = simulate(params, n=row["n"], key=key, dt=row["dt"], heterogeneity=members(d), wall=row["wall"], error="ks")
    full = _tensor(params, row["members"], crit(row["n"]) / 10)
    kinds = (s.times < s.provenance["t_max"]).astype(np.int8)
    D, pv = LS.ks_one_sample(s.times, kinds, full.S_obs)
    return {"id": f"{row['id']}/drop-{drop}", "n": row["n"], "D": D, "p_value": pv, "predicted_sup_dS": predicted,
            "drop": drop, "M": dict(full.M), "dt": row["dt"], "key": key}


def composite(params, d: dict, *, panels=(16, 32, 64), order=8, agree=1e-4, M=None, bar=1e-4) -> dict:
    """The tensor reference (at M, or its own check at `bar`) against Gauss-Legendre panels per dimension refined
    through `panels` until two successive refinements agree to `agree`: sup|dS| over the march's grid."""
    from sraging.engine import mixture as MIX
    from sraging.engine import quadrature as Q
    from sraging.engine import tensor_reference as TR
    tr = _tensor(params, d, bar, M=M)
    if tr.kind != "tensor":
        raise Refused(f"the tensor reference reads {tr.kind}")
    progs = MIX.programs(tr._ev)
    z = tr._ev._z(tr.p0)
    prev, out = None, None
    for k in panels:
        rules = []
        for p in tr.parameters:
            r = Q.composite_rule(tr.sd[p], int(k), int(order))
            rules.append((np.asarray(r.s, float), np.asarray(r.w, float)))
        S, _ = TR.fold(progs, z, tr.parameters, rules)
        step = None if prev is None else float(np.max(np.abs(S - prev)))
        out = {"panels": int(k), "nodes": int(math.prod(r[0].size for r in rules)), "refinement_sup_dS": step}
        prev = S
        if step is not None and step <= agree:
            break
    m = tr.S > TR.S_FLOOR
    return {**out, "converged": out["refinement_sup_dS"] is not None and out["refinement_sup_dS"] <= agree,
            "sup_dS": float(np.max(np.abs(tr.S[m] - prev[m]))), "M": dict(tr.M), "order": int(order), "agree": agree}


# ------------------------------------------------------------------------------------------------------- (iii)
def above_budget(params, d: dict, *, n, key, dt, t_max) -> dict:
    """The statement of a configuration whose tensor product is above the budget: kind, clean and its reason."""
    from sraging.langevin import simulate
    s = simulate(params, n=n, key=key, dt=dt, heterogeneity=members(d), error="ks", t_max=t_max)
    ref = s.error.fp_reference
    return {"kind": ref["kind"], "clean": s.clean, "reason": ref.get("reason"), "verdict": s.error.verdict,
            "warnings": [w.code for w in s.warnings]}


def projected_cpu_h(rows, draw_cpu_s, spec) -> float:
    """Each cell once at its frozen step (its draw CPU), one draw per plant, the reductions at their n, the composite's
    and the tensor references' estimates."""
    cells = sum(float(draw_cpu_s[r["p1_step"]]) for r in rows)
    one = max((float(draw_cpu_s[r["p1_step"]]) for r in rows), default=0.0)
    plants = len(spec.get("plants", [])) * one
    red = len(spec.get("reductions", [])) * 2 * len(WALLS) * one * float(spec.get("reduction_n", spec["n"])) / spec["n"]
    extra = float(spec.get("composite_cpu_s", 420.0)) + len(rows) * float(spec.get("tensor_cpu_s_per_cell", 200.0))
    return (cells + plants + red + extra + one) / 3600.0
