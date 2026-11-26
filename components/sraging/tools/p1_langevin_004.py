"""Spec 004's pilot P1, the Langevin price and step bias (tasks T072, T111; research R16, R18, R19, R25, R30; gate C1,
C4; contracts/acceptance.md §1 `p1-langevin`). The phase `p1-langevin` in tools/acceptance_004.py runs it (cap 6
CPU-h).

**The grid** (`cells`): the 4 presets (tools/presets_004.py) x homogeneous and xc sd 0.2 x both walls; per cell the
rungs m in {1, 2, 4, 8, 16}, dt = Delta t / m with Delta t the preset's base step (SRK's effective step), n = 25,000,
one key per cell.
**A rung** (`run_rung`): `simulate(..., error="ladder", record_ages=the cell's two HR-10 ages)`: the KS against FP,
the coupled D = sup|S_dt - S_dt/2| and the dt/2 rung's KS; the CPU, ns per individual-step and steps per individual;
at each HR-10 age the clamp's lump p0 (the living's fraction at exactly 0) and the paths' step bias (the coupled
sup|F_dt - F_dt/2| on the judged range).
**Per cell** (`fit_p_hat`, `choose_step`): the weak order p from log D against log dt on its rungs, taken in [1/2, 1]
(gate C1); the step: the largest dt whose judged bias b(dt/2) = D / (2^p - 1) is <= crit/10, crit = S4's family bound
for the wall / sqrt(n). **The extension** (gate C4): a clamp cell that misses the bar at 16 adds m = 32 and 64.
**Beside the grid:** D's noise floor on 2 cells x 3 keys at m = 16 (`noise_floor`); the per-individual streams'
overhead, Dd (`dd_overhead`); B1's predicted lag f_max Delta t / 2 per preset (`b1_lag`); SRK's 7.2 s / 0.45 s
re-measured (`srk_remeasure`: n 15,000 Langevin and an M = 9 FP mixture at xc sd 0.2, SRK n_jobs = 1, host and cores;
spec.md Clarifications, Naveh 2026-10-29).
**The output** (`result`): the keys acceptance_004.p1_result() documents ("steps", "cells", "draw_cpu_s",
"ns_per_individual_step", "steps_per_individual"), "p_hat_cells" for sraging.langevin.sample.p_hat, and the rest.
"""
from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import presets_004 as PR  # noqa: E402
import s4_fp_langevin_004 as S4  # noqa: E402

N = 25_000
M_RUNGS = (1, 2, 4, 8, 16)
M_EXTEND = (32, 64)
WALLS = ("clamp", "reflect")
CLASSES = S4.REFLECT_P1                     # (hom, None, 0.0), (het, xc, 0.2)
BAR = 10.0                                  # the judged bias <= crit / BAR
NOISE_CELLS = (("mice_F", "hom", "clamp"), ("human", "het", "clamp"))
NOISE_KEYS = 3
LAG_TARGET = 0.02                           # B1's plant: the predicted lag f_max Delta t / 2 >= 0.02 (gate C3)
SRK_N, SRK_M, SRK_SD = 15_000, 9, 0.2       # SRK's 7.2 s / 0.45 s, re-measured (Naveh 2026-10-29)


def cells(presets=PR.NAMES) -> list:
    """The 16 cells (preset x hom | het x wall), each with its base step and its key in S4's step map."""
    out = []
    for p in presets:
        for het, q, sd in CLASSES:
            for w in WALLS:
                out.append({"id": S4.step_key(p, het, w), "preset": p, "het": het, "parameter": q, "sd": float(sd),
                            "wall": w, "base_dt": PR.base_dt(p), "n": N})
    return out


def hr10_ages(params, het, base_dt) -> list:
    """The two HR-10 ages on the base step's grid (S near 0.75 and 0.4): on every rung's grid too."""
    import paths_004 as PA
    return PA.pick_ages(params, het, base_dt)


def run_rung(cell, params, m, key, ages, *, n=None) -> dict:
    """One rung of a cell: the coupled draw at dt = Delta t / m with paths at the HR-10 ages."""
    from sraging.langevin import simulate
    n = n or cell["n"]
    dt = cell["base_dt"] / m
    s = simulate(params, n=n, key=key, dt=dt, heterogeneity=S4._het(cell["parameter"], cell["sd"]),
                 wall=cell["wall"], error="ladder", record_ages=ages)
    b, c = s.error.step_bias, s.cost_record
    kern = c["kernel_cpu_s"]
    paths = [{"age": r["age"], "n_alive": r["n_alive"], "p0": r["p0"], "x_c": r["x_c"],
              "path_D": (r.get("step_bias") or {}).get("D_dt_vs_dt2"), "ks_D_censored": r["ks_D_censored"],
              "ks_D_full": r["ks_D_full"], "reference_grid_error": r.get("reference_grid_error")}
             for r in s.error.paths or []]
    return {"m": m, "dt": dt, "n": n, "key": key, "ks_D": s.error.ks_D, "p_value": s.error.p_value,
            "crit_95": s.error.crit_95, "D": b["D_dt_vs_dt2"], "se": b["se"], "ks_D_half": b["ks_D_half"],
            "p_value_half": b["p_value_half"], "draw_cpu_s": c["cpu_s"], "kernel_cpu_s": kern,
            "rung_cpu_s": c["rung_cpu_s"], "individual_steps": c["individual_steps"],
            "steps_per_individual": c["individual_steps"] / n,
            "ns_per_individual_step": 1e9 * kern / max(c["individual_steps"], 1),
            "reference_M": s.error.fp_reference["M"], "paths": paths, "warnings": [w.code for w in s.warnings]}


def fit_p_hat(rungs) -> dict:
    """The weak order p from log D against log dt (least squares over the rungs with D > 0), taken in [1/2, 1]."""
    pts = [(math.log(r["dt"]), math.log(r["D"])) for r in rungs if r["D"] and r["D"] > 0]
    if len(pts) < 2:
        return {"p_raw": None, "p_hat": 0.5, "n_points": len(pts), "source": "the conservative 1/2: < 2 points"}
    x, y = np.array(pts).T
    p_raw = float(np.polyfit(x, y, 1)[0])
    return {"p_raw": p_raw, "p_hat": min(max(p_raw, 0.5), 1.0), "n_points": len(pts), "source": "fit"}


def crit(wall, n=N) -> float:
    return S4.CRIT[wall] / math.sqrt(n)


def choose_step(rungs, p_hat, wall, n=N) -> dict:
    """The largest dt (smallest m) whose judged bias D / (2^p - 1) <= crit / 10; None if no rung meets it."""
    cr = crit(wall, n)
    for r in sorted(rungs, key=lambda r: r["m"]):
        jb = r["D"] / (2.0 ** p_hat - 1.0)
        if jb <= cr / BAR:
            return {"m": r["m"], "dt": r["dt"], "judged_bias": jb, "crit": cr, "bar": cr / BAR, "met": True}
    last = max(rungs, key=lambda r: r["m"])
    return {"m": None, "dt": None, "judged_bias": last["D"] / (2.0 ** p_hat - 1.0), "crit": cr, "bar": cr / BAR,
            "met": False, "finest_m": last["m"]}


def run_cell(cell, params, key, *, rungs=M_RUNGS, extend=M_EXTEND, n=None) -> dict:
    """A cell's ladder, its p, its step, and (clamp only, gate C4) the extension when 16 misses the bar."""
    het = S4._het(cell["parameter"], cell["sd"])
    ages = hr10_ages(params, het, cell["base_dt"])
    ladder = [run_rung(cell, params, m, key, ages, n=n) for m in rungs]
    fit = fit_p_hat(ladder)
    step = choose_step(ladder, fit["p_hat"], cell["wall"], n or cell["n"])
    extended = []
    if not step["met"] and cell["wall"] == "clamp":
        for m in extend:
            extended.append(run_rung(cell, params, m, key, ages, n=n))
            step = choose_step(ladder + extended, fit["p_hat"], cell["wall"], n or cell["n"])
            if step["met"]:
                break
    return {**cell, "key": key, "ages": ages, "ladder": ladder, "extension": extended, "fit": fit, "step": step}


# --------------------------------------------------------------------------------------------- beside the grid
def noise_floor(cell, params, keys, m=16, n=None) -> dict:
    """D at one rung on several keys: its spread is D's noise floor at n (gate C1)."""
    het = S4._het(cell["parameter"], cell["sd"])
    ages = hr10_ages(params, het, cell["base_dt"])
    Ds = [run_rung(cell, params, m, k, ages, n=n)["D"] for k in keys]
    nn = n or cell["n"]
    floor = float(np.mean(Ds))
    return {"cell": cell["id"], "m": m, "keys": list(keys), "D": Ds, "floor": floor, "max": float(max(Ds)),
            "crit_over_30": crit(cell["wall"], nn) / 30.0, "above_crit_over_30": floor > crit(cell["wall"], nn) / 30.0}


def dd_overhead(params, dt, n=N, key=1) -> dict:
    """Dd's price: the kernel over the same individuals for one step (streams seeded, almost no path) against the
    full draw: their CPU ratio and the seeding CPU per individual."""
    from sraging.langevin import sample as LS
    cfg = LS.LangevinConfig(params=params, n=n, key=key, dt=dt, error="ks")
    st = LS.prepare(cfg)
    c0 = time.process_time()
    LS.run_rung(cfg, st["arrays"], st["t_max"], 1)
    full = time.process_time() - c0
    c1 = time.process_time()
    LS.run_rung(cfg, st["arrays"], 1.5 * dt, 1)
    one = time.process_time() - c1
    return {"full_cpu_s": full, "one_step_cpu_s": one, "overhead_fraction": one / full if full > 0 else None,
            "per_individual_ns": 1e9 * one / n, "n": n, "dt": dt}


def b1_lag(params, het, base_dt, target=LAG_TARGET) -> dict:
    """B1's predicted lag f_max Delta t / 2 (the frozen age's shift of S, research R14), f_max the FP observed
    density's maximum; and the Delta t at which the lag reaches `target`."""
    from sraging.langevin.sample import LangevinConfig, Reference
    ref = Reference(LangevinConfig(params=params, n=N, key=0, dt=base_dt, heterogeneity=het, error="ks"))
    M = None
    if het is not None:
        tau, _ = ref.survival_grid(5)
        t = tau * ref.L
        M = ref.curve_check(t[ref.S_obs(t, 5) > 1e-12], 1e-4)["M"]
    tau, _ = ref.survival_grid(M)
    t = tau * ref.L
    S = ref.S_obs(t, M)
    f_max = float(np.max(-np.diff(S) / np.diff(t)))
    return {"f_max": f_max, "base_dt": base_dt, "lag_at_base_dt": f_max * base_dt / 2.0,
            "dt_for_target": 2.0 * target / f_max, "target": target, "M": M}


def srk_remeasure(session, name, *, n=SRK_N, M=SRK_M, sd=SRK_SD, seed=7) -> dict:
    """SRK's Langevin (the bridge kernel, one process) at the preset's own configuration with xc_var = sd, and SRK's
    FP mixture likelihood of that sample at M nodes, each op's CPU (spec.md Clarifications, Naveh 2026-10-29)."""
    rec = PR.PRESETS[name]
    p, st = rec["params"], rec["srk_step"]
    dt = float(st["t_end"]) / (int(st["nsteps"]) - 1)
    kw = {"eta": p["eta"], "beta": p["beta"], "epsilon": p["epsilon"], "kappa": p["kappa"], "xc": p["xc"]}
    ext = {} if rec["ext_h"] is None else {"external_hazard": float(rec["ext_h"])}
    sim = session.call("simulate_sr_hetro", steps=int(st["nsteps"]), dt=dt, npeople=int(n), seed=int(seed),
                       xc_var=float(sd), time_step_multiplier=int(st["time_step_multiplier"]), **kw, **ext)
    fp = session.call("fp_loglik_het_physical", death_times=np.asarray(sim["death_times"], float),
                      events=np.asarray(sim["events"], int), het_param="xc", het_var=float(sd), M=int(M), **kw,
                      **ext)
    return {"preset": name, "n": int(n), "M": int(M), "sd": float(sd), "langevin_cpu_s": float(sim["cpu_s"]),
            "fp_mixture_cpu_s": float(fp["cpu_s"]), "fp_value": float(fp["value"]), "n_jobs": 1,
            "srk_step": dict(st), "against": {"langevin_s": 7.2, "fp_mixture_s": 0.45}}


def probe(params, dt, n=1_000, key=1) -> dict:
    """The price's probe: one homogeneous clamp draw at m = 1 without the coupled rung."""
    from sraging.langevin import simulate
    s = simulate(params, n=n, key=key, dt=dt, error="ks")
    c = s.cost_record
    return {"n": n, "dt": dt, "steps_per_individual": c["individual_steps"] / n,
            "ns_per_individual_step": 1e9 * c["kernel_cpu_s"] / max(c["individual_steps"], 1), "cpu_s": c["cpu_s"]}


def host() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node(),
            "cores": os.cpu_count()}


# --------------------------------------------------------------------------------------------------- the result
def result(cell_rows, *, noise=None, dd=None, b1=None, srk=None) -> dict:
    """The P1 result's keys: what acceptance_004.p1_result() documents, plus "p_hat_cells" for sample.p_hat."""
    from sraging.langevin.sample import LangevinConfig, config_key
    errors = [{"id": c.get("id"), **c["error"]} for c in cell_rows if c.get("error")]
    cell_rows = [c for c in cell_rows if not c.get("error")]          # a cell that raised has no step (S4 refuses it)
    steps, draw, spi, ns, p_hat_cells = {}, {}, {}, [], []
    for c in cell_rows:
        rungs = c["ladder"] + c["extension"]
        st = c["step"]
        steps[c["id"]] = {"m": st["m"], "dt": st["dt"], "judged_bias": st["judged_bias"], "crit": st["crit"],
                          "met": st["met"]}
        chosen = next((r for r in rungs if r["m"] == st["m"]), None) if st["met"] else max(rungs, key=lambda r: r["m"])
        draw[c["id"]] = chosen["draw_cpu_s"]
        spi[c["id"]] = chosen["steps_per_individual"]
        ns += [r["ns_per_individual_step"] for r in rungs]
        params = PR.preset(c["preset"])
        for r in rungs:
            cfg = LangevinConfig(params=params, n=r["n"], key=0, dt=r["dt"], heterogeneity=S4._het(c["parameter"],
                                                                                                   c["sd"]),
                                 wall=c["wall"], error="ks")
            p_hat_cells.append({"cell": c["id"], "m": r["m"], "config": config_key(cfg), "p_hat": c["fit"]["p_hat"]})
    return {"steps": steps, "cells": cell_rows, "draw_cpu_s": draw, "steps_per_individual": spi,
            "ns_per_individual_step": float(np.median(ns)) if ns else None, "p_hat_cells": p_hat_cells,
            "p_hat": {c["id"]: c["fit"]["p_hat"] for c in cell_rows},
            "cell_ages": {c["id"]: list(c["ages"]) for c in cell_rows}, "noise_floor": noise, "dd_overhead": dd,
            "b1": b1, "srk_remeasure": srk, "host": host(), "presets": PR.PRESETS, "cell_errors": errors}


def projected_cpu_h(probe_ns, probe_steps_per_individual, n=N) -> float:
    """The grid's CPU from a probe: per cell sum_m (1 + 2) m steps per individual at m = 1, both walls and classes
    (each rung is a dt draw plus its coupled dt/2 rung), plus the noise floor's 6 draws at m = 16."""
    tot = 0.0
    for name, spi in probe_steps_per_individual.items():
        per_m1 = n * spi * probe_ns * 1e-9
        tot += len(WALLS) * len(CLASSES) * sum(3 * m for m in M_RUNGS) * per_m1
    worst = max(probe_steps_per_individual.values()) * n * probe_ns * 1e-9 if probe_steps_per_individual else 0.0
    return (tot + len(NOISE_CELLS) * NOISE_KEYS * 3 * 16 * worst) / 3600.0
