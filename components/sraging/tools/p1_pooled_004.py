"""Spec 004's pilot p1-pooled: each Langevin cell's step frozen from a measurement at pooled n (tasks T111b; spec.md
Clarifications 2026-11-01, 8f0f269; the pre-registered rule of research gate C1 and spec "The noise floor": "if the
floor exceeds crit/30, each cell's step is frozen from P1, measured at pooled n"). The phase `p1-pooled` in
tools/acceptance_004.py runs it after P1 and before P2 and P3 (cap 45 CPU-h, Naveh 2026-11-01).

**Why:** P1's coupled D at n = 25,000 has a noise floor of ~0.0017, above the crit/10 bar (0.00125 clamp, 0.00107
reflect), so P1 met the bar on 1 of 16 cells.

**The cells** (`cells`): P1's 16 (preset x hom | het x wall) and S5's configuration x {clamp, reflect} (ROLES["s5"]:
score_het_002's pure-k0.01-p00-c0-s0 at theta0, xc sd 0.2; its base step S5_BASE_DT).
**The start step** (`extrapolate`): m* = m_f max(1, b_f / bar)^(1 / p), b_f = D_f / (2^p - 1) on the cell's finest rung
(P1's ladder with its extension; for S5's cells, a short ladder m in {1, 2, 4, 8, 16} at n 25,000 and one key run
here first, `s5_ladder`), bar = crit/10 at n = 25,000 (P1's crit), m* rounded up to an integer; dt = Delta t / m.
**At a step** (`pooled_at`): K = 18 independent keys, each a coupled dt / dt-2 pair at n = 25,000 (the keys from
(cell, k) alone, so nothing depends on the worker count); the K samples pooled per rung (n = 450,000) and b =
D_pooled / (2^p - 1) judged against the n = 25,000 bar. The pooled noise floor is recorded as mean_k(D_k) / sqrt(K):
the per-key D measures the noise at n = 25,000 where the bias is small, and a sup of two empirical CDFs shrinks like
1 / sqrt(n); sd_k(D_k) / sqrt(K) is recorded beside it.
**The doubling** (`run_cell`): over the bar, m doubles, at most twice; then the cell reads "not met", never a silent
pass. The frozen step is the first m that meets the bar.
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

import s4_fp_langevin_004 as S4  # noqa: E402

K = 18                              # keys pooled per step (n = 450,000)
N = 25_000                          # per key
MAX_DOUBLINGS = 2
BAR = 10.0                          # the judged bias <= crit / BAR, crit = S4.CRIT[wall] / sqrt(N) (P1's)
S5_LADDER = (1, 2, 4, 8, 16)
#: S5's base step (days): spec 002's score harness's step before its /8 refinement (tools/score_test.py SIM_DT x 8,
#: where the pure control's score bias was ~5 SE at R = 200, shrinking ~0.39x per halving)
S5_BASE_DT = 0.0125
S5_KEYS = ("s5|het|clamp", "s5|het|reflect")
KEY_BASE = 90_000


def crit(wall, n=N) -> float:
    return S4.CRIT[wall] / math.sqrt(n)


def bar(wall, n=N) -> float:
    return crit(wall, n) / BAR


def judged(D, p_hat) -> float:
    return float(D) / (2.0 ** float(p_hat) - 1.0)


def extrapolate(rungs, p_hat, wall, n=N) -> dict:
    """m* = m_f max(1, b_f / bar)^(1 / p), rounded up, from the finest rung."""
    f = max(rungs, key=lambda r: r["m"])
    b = judged(f["D"], p_hat)
    x = f["m"] * max(1.0, b / bar(wall, n)) ** (1.0 / float(p_hat))
    return {"m_start": int(math.ceil(x - 1e-9)), "m_f": int(f["m"]), "D_f": float(f["D"]), "b_f": b,
            "bar": bar(wall, n), "p_hat": float(p_hat), "raw": x}


# ------------------------------------------------------------------------------------------------- the draws
def coupled_pair(task) -> dict:
    """One key's coupled dt / dt-2 pair (worker side): the raw times of both rungs, its CPU and steps per individual.
    `task`: {params (dict), het (None or (parameter, sd)), wall, dt, t_max, n, key}."""
    from sraging import SRParams
    from sraging.engine.inputs import Heterogeneity
    from sraging.langevin import sample as LS
    c0 = time.process_time()
    het = None if task["het"] is None else Heterogeneity(parameter=task["het"][0], sd=float(task["het"][1]))
    cfg = LS.LangevinConfig(params=SRParams(**task["params"]), n=int(task["n"]), key=int(task["key"]),
                            dt=float(task["dt"]), heterogeneity=het, wall=task["wall"], t_max=float(task["t_max"]),
                            error="ks")
    arrays = LS.draw_parameters(cfg)
    t1 = LS.run_rung(cfg, arrays, cfg.t_max, 1)[0]
    t2 = LS.run_rung(cfg, arrays, cfg.t_max, 2)[0]
    return {"key": int(task["key"]), "t1": t1, "t2": t2, "cpu_s": time.process_time() - c0,
            "steps_per_individual": float(np.sum(np.ceil(t1 / cfg.dt - 1e-9))) / cfg.n}


class Pool:
    """W worker processes (spawned) over the coupled pairs, or in process for W = 1; outputs in task order."""

    def __init__(self, workers: int):
        self.workers, self._ex = int(workers), None
        if self.workers > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self._ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))

    def map(self, tasks) -> list:
        if self._ex is None:
            return [coupled_pair(t) for t in tasks]
        return list(self._ex.map(coupled_pair, tasks))

    def close(self):
        if self._ex is not None:
            self._ex.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _task(cell, dt, key):
    return {"params": cell["params"], "het": cell["het_spec"], "wall": cell["wall"], "dt": dt, "t_max": cell["t_max"],
            "n": cell.get("n", N), "key": key}


def pooled_at(cell, m, pool, k=K) -> dict:
    """K coupled pairs at dt = Delta t / m, pooled: D_pooled, b, the pooled noise floor, met; and the worker CPU."""
    from sraging.langevin.sample import ecdf_sup
    dt = cell["base_dt"] / m
    outs = pool.map([_task(cell, dt, cell["key"] + j) for j in range(int(k))])
    Dk = [ecdf_sup(o["t1"], o["t2"]) for o in outs]
    D = ecdf_sup(np.concatenate([o["t1"] for o in outs]), np.concatenate([o["t2"] for o in outs]))
    b = judged(D, cell["p_hat"])
    br = bar(cell["wall"], cell.get("n", N))
    sq = math.sqrt(len(outs))
    return {"m": int(m), "dt": dt, "K": len(outs), "n_pooled": int(sum(o["t1"].size for o in outs)), "D_pooled": D,
            "b": b, "bar": br, "met": b <= br, "D_per_key": Dk, "noise_floor": float(np.mean(Dk)) / sq,
            "noise_spread": float(np.std(Dk, ddof=1)) / sq if len(Dk) > 1 else None,
            "draw_cpu_s": float(np.median([o["cpu_s"] for o in outs])),
            "steps_per_individual": float(np.median([o["steps_per_individual"] for o in outs])),
            "worker_cpu_s": float(sum(o["cpu_s"] for o in outs))}


def run_cell(cell, pool, k=K, max_doublings=MAX_DOUBLINGS) -> tuple:
    """(the cell's record, its worker CPU): the start step, then doublings while b > bar, at most `max_doublings`."""
    m, tried, cpu = int(cell["m_start"]), [], 0.0
    for _ in range(int(max_doublings) + 1):
        r = pooled_at(cell, m, pool, k)
        cpu += r.pop("worker_cpu_s")
        tried.append(r)
        if r["met"]:
            break
        m *= 2
    last = tried[-1]
    frozen = ({"m": last["m"], "dt": last["dt"], "steps_per_individual": last["steps_per_individual"],
               "draw_cpu_s": last["draw_cpu_s"]} if last["met"] else None)
    rec = {k_: cell[k_] for k_ in ("id", "preset", "het", "wall", "base_dt", "p_hat", "m_start", "t_max", "key")}
    rec.update({"extrapolation": cell.get("extrapolation"), "tried": tried, "met": last["met"], "frozen": frozen,
                "status": "met" if last["met"] else "not met", "noise_floor": last["noise_floor"],
                "p_hat_source": cell.get("p_hat_source"), "ladder": cell.get("ladder"), "fit": cell.get("fit")})
    return rec, cpu


# ------------------------------------------------------------------------------------------------- the cells
def _t_max(params, het, n=N) -> float:
    """The default t_max (S_obs < 0.01 / n) at n = 25,000, as P1's draws had it."""
    from sraging.langevin import sample as LS
    return float(LS.prepare(LS.LangevinConfig(params=params, n=n, key=0, dt=1.0, heterogeneity=het, error="ks"))
                 ["t_max"])


def p1_cells(p1) -> list:
    """P1's 16 cells with their start steps, extrapolated from P1's own ladders (extension included)."""
    import presets_004 as PR
    out = []
    for i, c in enumerate(p1["cells"]):
        rungs = c["ladder"] + c["extension"]
        p_hat = float(c["fit"]["p_hat"])
        ex = extrapolate(rungs, p_hat, c["wall"])
        params = PR.preset(c["preset"])
        out.append({"id": c["id"], "preset": c["preset"], "het": c["het"], "wall": c["wall"],
                    "base_dt": float(c["base_dt"]), "p_hat": p_hat, "m_start": ex["m_start"], "extrapolation": ex,
                    "params": {k: v for k, v in params.as_dict().items() if k != "time_unit"},
                    "het_spec": None if c["het"] == "hom" else ("xc", 0.2), "key": KEY_BASE + 1000 * i,
                    "p_hat_source": "P1", "finest": max(rungs, key=lambda r: r["m"])})
    return out


def s5_params(cfg) -> tuple:
    """(SRParams, Heterogeneity) of ROLES["s5"]'s gen-1 spec."""
    import s5_gen1_004 as S5
    from sraging import generator as GEN
    return S5.params_of(GEN.CohortSpec(**cfg["spec"]))


def s5_ladder(cfg, wall, pool, rungs=S5_LADDER, n=N) -> dict:
    """S5's short ladder (no P1 ladder): one key, m in `rungs` at n 25,000, P1's way; then p and the start step."""
    import p1_langevin_004 as P1
    from sraging.langevin.sample import ecdf_sup
    params, het = s5_params(cfg)
    base = {"params": {k: v for k, v in params.as_dict().items() if k != "time_unit"},
            "het_spec": (het.parameter, het.sd) if het is not None else None, "wall": wall, "base_dt": S5_BASE_DT,
            "t_max": _t_max(params, het, n), "n": n}
    key = KEY_BASE + 50_000 + (0 if wall == "clamp" else 1000)
    outs = pool.map([_task(base, S5_BASE_DT / m, key) for m in rungs])
    ladder = [{"m": m, "dt": S5_BASE_DT / m, "D": ecdf_sup(o["t1"], o["t2"]), "draw_cpu_s": o["cpu_s"],
               "steps_per_individual": o["steps_per_individual"]} for m, o in zip(rungs, outs)]
    fit = P1.fit_p_hat(ladder)
    ex = extrapolate(ladder, fit["p_hat"], wall, n)
    return {"id": f"s5|het|{wall}", "preset": "s5", "het": "het", "wall": wall, "base_dt": S5_BASE_DT,
            "p_hat": fit["p_hat"], "fit": fit, "m_start": ex["m_start"], "extrapolation": ex, "ladder": ladder,
            "params": base["params"], "het_spec": base["het_spec"], "t_max": base["t_max"], "n": n,
            "key": key + 100, "p_hat_source": "its own short ladder", "worker_cpu_s": sum(o["cpu_s"] for o in outs)}


def s5_probe(cfg, wall, n=1_000) -> dict:
    """One coupled pair of S5's configuration at m = 1 and n = 1,000: the per-key CPU at n = 25,000 and m = 1."""
    params, het = s5_params(cfg)
    t_max = _t_max(params, het)
    o = coupled_pair({"params": {k: v for k, v in params.as_dict().items() if k != "time_unit"},
                      "het": (het.parameter, het.sd) if het is not None else None, "wall": wall, "dt": S5_BASE_DT,
                      "t_max": t_max, "n": n, "key": KEY_BASE + 99_000})
    return {"per_key_m1_s": o["cpu_s"] * N / n, "steps_per_individual_m1": o["steps_per_individual"], "t_max": t_max}


def s5_projected_cpu_h(probe, m_nominal=16, k=K, overhead=1.0) -> dict:
    """S5's price before its ladder: the ladder (sum of its m) and K keys at a nominal start m (its ladder's finest;
    the real start is extrapolated after the ladder, and the cap's stop rule holds either way)."""
    ladder = sum(S5_LADDER) * probe["per_key_m1_s"] / 3600.0
    pooled = k * m_nominal * probe["per_key_m1_s"] * float(overhead) / 3600.0
    return {"ladder_cpu_h": ladder, "pooled_cpu_h": pooled, "m_nominal": m_nominal, "cpu_h": ladder + pooled}


def with_t_max(cell) -> dict:
    from sraging import SRParams
    from sraging.engine.inputs import Heterogeneity
    if "t_max" in cell:
        return cell
    het = None if cell["het_spec"] is None else Heterogeneity(parameter=cell["het_spec"][0], sd=cell["het_spec"][1])
    return {**cell, "t_max": _t_max(SRParams(**cell["params"]), het, cell.get("n", N))}


# --------------------------------------------------------------------------------------------- price, result
def projected_cpu_h(cells, k=K, overhead=1.0) -> dict:
    """K coupled pairs per cell at its start step, scaled from P1's finest rung (its dt rung's and dt/2 rung's
    measured CPU, x m_start / m_f); S5's cells from their ladder's finest rung likewise. No doubling counted.
    `overhead`: a measured factor on the per-key CPU (p1-pooled-b passes p1-pooled's; p1-pooled ran with 1)."""
    rows = {}
    for c in cells:
        f = c.get("finest")
        if f is not None:
            per = (float(f["kernel_cpu_s"]) + float(f["rung_cpu_s"])) * c["m_start"] / float(f["m"])
        else:
            lf = max(c["ladder"], key=lambda r: r["m"])
            per = float(lf["draw_cpu_s"]) * c["m_start"] / float(lf["m"])
        per *= float(overhead)
        rows[c["id"]] = {"m_start": c["m_start"], "per_key_s": per, "cpu_h": k * per / 3600.0}
    return {"cells": rows, "total_cpu_h": sum(r["cpu_h"] for r in rows.values()), "K": k, "overhead": overhead}


def result(records) -> dict:
    """{"cells": {id: record}, "frozen": {id: {m, dt, steps_per_individual, draw_cpu_s, p_hat, met}}}: what the
    accessor (acceptance_004.frozen_steps) reads."""
    ok = [r for r in records if not r.get("error")]
    frozen = {r["id"]: {"m": (r["frozen"] or {}).get("m"), "dt": (r["frozen"] or {}).get("dt"),
                        "steps_per_individual": (r["frozen"] or {}).get("steps_per_individual"),
                        "draw_cpu_s": (r["frozen"] or {}).get("draw_cpu_s"), "p_hat": r["p_hat"], "met": r["met"]}
              for r in ok}
    return {"cells": {r["id"]: r for r in ok}, "frozen": frozen,
            "not_met": sorted(r["id"] for r in ok if not r["met"]),
            "cell_errors": [{"id": r.get("id"), **r["error"]} for r in records if r.get("error")]}
