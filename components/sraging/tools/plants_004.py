"""Spec 004's plants (tasks T074, T121; research R14; gate C3; register D-402 to D-411): each fix's planted case
against SRK's frozen code (the fp_piecewise snapshot through tools/oracle_fp_piecewise.py, read-only), beside its
twin. The phase `plants` in tools/acceptance_004.py runs them under SR (cap 3 CPU-h); `hr8` runs HR-8 (`hr8_rows`).

**Every plant** returns {id, plant, twin, sup_abs_dS, n, predicted, informative, status}: `status` is pass when the
plant shows the legacy defect (informative) and the twin does not, fail when the twin does, incomplete when the plant
does not (constitution V: a plant that does not fail is "not informative", never a pass).
- **D-402, B1, HR-9 like for like** (`hr9`): SRK at (Delta t, m = 16) against SRK at (Delta t / 16, m = 1), two-sample
  KS at alpha 0.05 (must reject); sraging at Delta t / 16 against SRK at (Delta t / 16, 1), two-sample at 0.025, and
  sraging at its frozen step (`fix_dt`; Naveh 2026-11-03) against FP observed through the Delta t recording,
  one-sample at 0.025 (both must not);
  every time rounded up to (outer step + 1) Delta t; the verdict is sampler_rules.hr9_verdict.
- **D-403, B2** (`b2`): SRK's realised step for a requested step_size (stepsize.step_size_to_grid, then its kernel
  convention) differs from it; sraging's realised step is the stated dt; sup|dS| between SRK at its realised grid and
  SRK at the requested step.
- **D-404, B4** (`b4`): SRK's parallel Euler path at n = 1,003 returns n_jobs floor(n / n_jobs) real people and pads
  the rest as censored at t_end; sraging returns all n. sup|dS| of the padded Kaplan-Meier against the real part's.
- **D-405, B11** (`b11`): SRK's unfiltered xc draws at sd 0.5 die at the first step (predicted Phi(-2) = 2.3 %); the
  twin, SRK at sd 0.2, has nothing to see (P = 2.9e-7). sraging's truncated normal draws no xc <= 0; its first-step
  fraction (the model's own: thresholds near the truncation die early) is reported beside it, not judged. sup|dS|
  SRK against sraging at sd 0.5.
- **D-406, seeding** (`seeding`): two SRK parallel runs with one seed differ; two sraging runs with one key, and
  W = 1 against W, are identical (no dS).
- **D-407, hazard** (`hazard`): SRK's per-substep kill h delta against the exact competing exponential (SRK's
  no-hazard draw with an exact exponential time at the end of its outer step, as SRK records a death), at h delta =
  0.05 (predicted size first) and the twin 1e-4.
- **D-408, recording** (`recording`): SRK records (j + 1) Delta t on the outer grid whatever m is; sraging floor(t /
  delta) delta. A coarse Delta t with delta < Delta t: SRK's ages all on the Delta t grid; the twin Delta t = delta.
- **D-409** (`d409`): getSrHetro's defaults (hetro=True, xc sd 0.2) against the named homogeneous call; sraging's
  default is homogeneous (identical bytes).
- **D-410** (`d410`, ported-as-is): the clamp leaves exact zeros at a young age, reflect none.
- **D-411** (`d411`): getISR's recorder writes 0 for the dead (the count of zeros among the dead, SRK > 0, sraging
  0) and sup|dF| of the living's X, SRK against sraging.
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

N = 25_000
HR9_ALPHA = {"plant": 0.05, "twin": 0.025, "fix": 0.025}
B11_PREDICTED = 0.5 * math.erfc(2.0 / math.sqrt(2.0))      # Phi(-2): P(1 + 0.5 z <= 0)


def _srk_kw(params) -> dict:
    p = params.as_dict() if hasattr(params, "as_dict") else dict(params)
    kw = {k: float(p[k]) for k in ("eta", "beta", "epsilon", "kappa", "xc")}
    if p.get("h_ext", 0.0) > 0:
        kw["external_hazard"] = -math.log(float(p["h_ext"]))
    return kw


def _ks2(a, b) -> dict:
    from scipy.stats import ks_2samp
    r = ks_2samp(a, b)
    return {"D": float(r.statistic), "p_value": float(r.pvalue), "n": int(min(a.size, b.size))}


def _times(r, horizon) -> np.ndarray:
    """SRK's death times, (j + 1) dt for a crossing in outer step j (SR_hetro.py:513: j advances after the substeps,
    so the time is already the outer step's end), with its survivors placed beyond every death (+inf)."""
    t = np.asarray(r["death_times"], float).copy()
    t[np.asarray(r["events"], int) == 0] = np.inf
    return t


def _status(informative, twin_ok) -> str:
    return "fail" if not twin_ok else "pass" if informative else "incomplete"


def _sup(a, b) -> float:
    """sup_t |F_a(t) - F_b(t)| over the finite values, each ECDF over its whole sample (survivors at +inf)."""
    grid = np.union1d(a[np.isfinite(a)], b[np.isfinite(b)])
    if grid.size == 0:
        return 0.0
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


# ------------------------------------------------------------------------------------------------- D-402, HR-9
def round_up(t, Dt) -> np.ndarray:
    """A death time (a step end, or an exact external time) onto the Delta t grid: the end of its outer step."""
    t = np.asarray(t, float)
    out = np.ceil(t / Dt - 1e-9) * Dt
    return np.where(np.isfinite(t), out, np.inf)


def hr9(srk, params, cfg, key) -> dict:
    """B1 like for like (gate C3). cfg: dt (Delta t), m (16), t_end (the horizon), n, alpha, predicted_lag."""
    from sraging import sampler_rules as R
    from sraging.langevin import simulate
    from sraging.langevin.sample import LangevinConfig, Reference
    from scipy.stats import kstwo
    Dt, m, n = float(cfg["dt"]), int(cfg.get("m", 16)), int(cfg.get("n", N))
    S = int(math.floor(float(cfg["t_end"]) / Dt)) + 1
    horizon = (S - 1) * Dt
    kw = _srk_kw(params)
    coarse = srk.call("simulate_sr_hetro", steps=S, dt=Dt, npeople=n, seed=int(key), time_step_multiplier=m, **kw)
    fine = srk.call("simulate_sr_hetro", steps=m * (S - 1) + 1, dt=Dt / m, npeople=n, seed=int(key) + 1,
                    time_step_multiplier=1, **kw)
    t_coarse = _times(coarse, horizon)                          # already (j + 1) Delta t
    t_fine = round_up(_times(fine, horizon), Dt)                 # (j' + 1) h -> the end of its outer step
    s = simulate(params, n=n, key=int(key) + 2, dt=Dt / m, t_max=horizon, error="ks")
    t_sra = round_up(np.where(s.times < horizon, s.times, np.inf), Dt)
    arms = {"plant": _ks2(t_coarse, t_fine), "twin": _ks2(t_sra, t_fine)}
    # the fix arm at sraging's frozen step (Naveh 2026-11-03, sr-naveh finding 3): at Delta t / 16 = 0.147 wk the
    # step bias alone rejects FP; recorded through the same Delta t grid
    fix_dt = float(cfg.get("fix_dt") or Dt / m)
    sf = s if fix_dt == Dt / m else simulate(params, n=n, key=int(key) + 3, dt=fix_dt, t_max=horizon, error="ks")
    t_fix = round_up(np.where(sf.times < horizon, sf.times, np.inf), Dt)
    ref = Reference(LangevinConfig(params=params, n=n, key=0, dt=fix_dt, error="ks"))
    grid = np.arange(1, S) * Dt
    F = 1.0 - ref.S_obs(grid)
    Fh = np.searchsorted(np.sort(t_fix), grid + 1e-9 * Dt, side="right") / n
    D = float(np.max(np.abs(Fh - F)))
    arms["fix"] = {"D": D, "p_value": float(kstwo.sf(D, n)), "n": n, "dt": fix_dt}
    v = R.hr9_verdict(arms, alpha=cfg.get("alpha", HR9_ALPHA))
    return {"id": "D-402", "plant": arms["plant"], "twin": arms["twin"], "fix": arms["fix"], "verdict": v.as_dict(),
            "sup_abs_dS": arms["plant"]["D"], "n": n, "dt": Dt, "m": m, "predicted": cfg.get("predicted_lag"),
            "informative": arms["plant"]["p_value"] < HR9_ALPHA["plant"], "status": v.status}


# ---------------------------------------------------------------------------------------------- D-403, B2
def b2(srk, params, cfg, key) -> dict:
    """B2: SRK's realised step for a requested step_size; sraging's realised step is the stated one."""
    from sraging.langevin.sample import LangevinConfig
    step, t_end, n = float(cfg["step_size"]), float(cfg["t_end"]), int(cfg.get("n", N))
    g = srk.call("step_size_to_grid", t_end=t_end, step_size=step)
    nsteps, tsm, realised = int(g["nsteps"]), int(g["time_step_multiplier"]), float(g["realised_step"])
    kw = _srk_kw(params)
    a = srk.call("simulate_sr_hetro", steps=nsteps, dt=realised * tsm, npeople=n, seed=int(key),
                 time_step_multiplier=tsm, **kw)
    b = srk.call("simulate_sr_hetro", steps=int(math.floor(t_end / step)) + 1, dt=step, npeople=n, seed=int(key) + 1,
                 time_step_multiplier=1, **kw)
    sra = LangevinConfig(params=params, n=10, key=0, dt=step, error="ks").dt
    plant = {"requested": step, "realised": realised, "relative_mismatch": realised / step - 1.0, "nsteps": nsteps,
             "time_step_multiplier": tsm}
    twin = {"stated": step, "realised": sra, "equal": sra == step}
    return {"id": "D-403", "plant": plant, "twin": twin, "n": n, "sup_abs_dS": _sup(_times(a, t_end), _times(b, t_end)),
            "informative": realised != step, "status": _status(realised != step, twin["equal"])}


# ---------------------------------------------------------------------------------------------- D-404, B4
def b4(srk, params, cfg, key) -> dict:
    """B4 at n = 1,003 with SRK's n_jobs = os.cpu_count()."""
    from sraging.langevin import simulate
    from sraging.population import kaplan_meier
    n, t_end, nsteps = int(cfg.get("n", 1_003)), float(cfg["t_end"]), int(cfg["nsteps"])
    dt = t_end / (nsteps - 1)
    r = srk.call("simulate_accelerator2", steps=nsteps, dt=dt, npeople=n, seed=int(key), t_end=t_end,
                 time_step_multiplier=int(cfg.get("time_step_multiplier", 1)), **_srk_kw(params))
    n_real, n_jobs = int(r["n_real"]), int(r["n_jobs"])
    T, E = np.asarray(r["death_times"], float), np.asarray(r["events"], int)
    at = np.unique(T[E == 1])
    dS = float(np.max(np.abs(kaplan_meier(T, E, at)["S"] - kaplan_meier(T[:n_real], E[:n_real], at)["S"]),
                      initial=0.0))
    s = simulate(params, n=n, key=int(key), dt=dt, t_max=t_end, error="ks")
    plant = {"n": n, "n_jobs": n_jobs, "n_real": n_real, "padded": n - n_real}
    twin = {"n": n, "present": int(s.times.size), "all_present": int(s.times.size) == n}
    return {"id": "D-404", "plant": plant, "twin": twin, "n": n, "sup_abs_dS": dS, "informative": n - n_real > 0,
            "predicted": {"padded": n % n_jobs}, "status": _status(n - n_real > 0, twin["all_present"])}


# --------------------------------------------------------------------------------------------- D-405, B11
def b11(srk, params, cfg, key) -> dict:
    """B11: first-step deaths at xc sd 0.5 (SRK) against sd 0.2, and sraging's truncated normal at 0.5."""
    from scipy.stats import binomtest
    from sraging.engine.inputs import Heterogeneity
    from sraging.langevin import simulate
    n, t_end, nsteps = int(cfg.get("n", N)), float(cfg["t_end"]), int(cfg["nsteps"])
    dt = t_end / (nsteps - 1)
    kw = _srk_kw(params)
    out = {}
    for sd in (float(cfg.get("sd_plant", 0.5)), float(cfg.get("sd_twin", 0.2))):
        r = srk.call("simulate_sr_hetro", steps=nsteps, dt=dt, npeople=n, seed=int(key), xc_var=sd,
                     time_step_multiplier=int(cfg.get("time_step_multiplier", 1)), **kw)
        t = _times(r, t_end)
        out[sd] = {"sd": sd, "first_step_fraction": float(np.mean(t <= dt)), "times": t}
    sp, st = out[float(cfg.get("sd_plant", 0.5))], out[float(cfg.get("sd_twin", 0.2))]
    s = simulate(params, n=n, key=int(key), dt=dt, t_max=t_end, error="ks",
                 heterogeneity=Heterogeneity(parameter="xc", sd=sp["sd"]))
    sra_first = float(np.mean(s.times <= dt))
    k = int(round(sp["first_step_fraction"] * n))
    informative = sp["first_step_fraction"] > B11_PREDICTED / 2 and binomtest(k, n, 1e-4, alternative="greater").pvalue < 0.05
    t_sra = np.where(s.times < t_end, s.times, np.inf)
    return {"id": "D-405", "n": n, "predicted": B11_PREDICTED,
            "plant": {k_: v for k_, v in sp.items() if k_ != "times"},
            "twin": {"srk_sd_twin": {k_: v for k_, v in st.items() if k_ != "times"}, "sraging_first_step": sra_first},
            "sup_abs_dS": _sup(sp["times"], t_sra), "informative": informative,
            "status": _status(informative, st["first_step_fraction"] < B11_PREDICTED / 10)}


# ------------------------------------------------------------------------------------------- D-406, seeding
def seeding(srk, params, cfg, key) -> dict:
    """Two SRK parallel runs with one seed differ; sraging with one key is identical, and at W = 1 and W."""
    from sraging.langevin import simulate
    n, t_end, nsteps = int(cfg.get("n", 2_000)), float(cfg["t_end"]), int(cfg["nsteps"])
    dt = t_end / (nsteps - 1)
    W = int(cfg.get("W", 7))
    kw = _srk_kw(params)
    a, b = (srk.call("simulate_sr_hetro_parallel", steps=nsteps, dt=dt, npeople=n, seed=int(key), n_jobs=W, **kw)
            for _ in range(2))
    differ = np.asarray(a["death_times"]).tobytes() != np.asarray(b["death_times"]).tobytes()
    s1, s2 = (simulate(params, n=n, key=int(key), dt=dt, t_max=t_end, error="ks") for _ in range(2))
    sw = simulate(params, n=n, key=int(key), dt=dt, t_max=t_end, error="ks", workers=W, chunk=max(1, n // (2 * W)))
    same = s1.times.tobytes() == s2.times.tobytes() == sw.times.tobytes()
    return {"id": "D-406", "n": n, "plant": {"srk_runs_differ": differ, "n_jobs": W},
            "twin": {"sraging_identical": same, "W": W}, "sup_abs_dS": None, "informative": differ,
            "status": _status(differ, same)}


# -------------------------------------------------------------------------------------------- D-407, hazard
def _hazard_arm(srk, params, nsteps, dt, tsm, n, key, h_delta):
    """SRK with its per-substep kill at h delta, and SRK's no-hazard draw with an exact exponential time placed at
    the end of its outer step, as SRK records a death: (two-sample KS, h, the predicted sup|dS|)."""
    from sraging.langevin.sample import LangevinConfig, Reference
    delta = dt / tsm
    h = h_delta / delta
    kw = {k: v for k, v in _srk_kw(params).items() if k != "external_hazard"}
    t_end = (nsteps - 1) * dt
    a = srk.call("simulate_sr_hetro", steps=nsteps, dt=dt, npeople=n, seed=int(key), time_step_multiplier=tsm,
                 external_hazard=-math.log(h), **kw)
    b = srk.call("simulate_sr_hetro", steps=nsteps, dt=dt, npeople=n, seed=int(key) + 1, time_step_multiplier=tsm,
                 **kw)
    E = np.random.default_rng(int(key)).exponential(1.0 / h, n)
    tb = np.minimum(_times(b, t_end), np.where(E <= t_end, round_up(E, dt), np.inf))     # SRK's (j + 1) dt
    ref = Reference(LangevinConfig(params=params.__class__(**{**params.as_dict(), "h_ext": 0.0}), n=n, key=0, dt=dt,
                                   error="ks"))
    t = np.arange(nsteps) * dt
    S = ref.S_obs(t)
    pred = float(np.max(S * np.abs((1.0 - h_delta) ** (t / delta) - np.exp(-h * t))))
    return _ks2(_times(a, t_end), tb), h, pred


def hazard(srk, params, cfg, key) -> dict:
    n, nsteps, dt, tsm = int(cfg.get("n", N)), int(cfg["nsteps"]), float(cfg["dt"]), int(cfg.get("time_step_multiplier", 1))
    plant, h_p, pred_p = _hazard_arm(srk, params, nsteps, dt, tsm, n, key, float(cfg.get("h_delta_plant", 0.05)))
    twin, h_t, pred_t = _hazard_arm(srk, params, nsteps, dt, tsm, n, int(key) + 10, float(cfg.get("h_delta_twin", 1e-4)))
    crit = 1.358 * math.sqrt(2.0 / n)
    informative = plant["p_value"] < 0.05
    return {"id": "D-407", "n": n, "plant": {**plant, "h": h_p, "predicted_sup_dS": pred_p},
            "twin": {**twin, "h": h_t, "predicted_sup_dS": pred_t}, "sup_abs_dS": plant["D"],
            "predicted": {"plant": pred_p, "detection_crit_2sample": crit, "plant_detectable": pred_p > crit},
            "informative": informative, "status": _status(informative, twin["p_value"] >= 0.025)}


# ----------------------------------------------------------------------------------------- D-408, recording
def recording(srk, params, cfg, key) -> dict:
    """SRK's outer-grid recording against gen-1's floor(t / delta) delta at a coarse Delta t with delta < Delta t."""
    from sraging.langevin import simulate
    n, Dt, m, delta = int(cfg.get("n", N)), float(cfg["dt"]), int(cfg.get("m", 1)), float(cfg["delta"])
    nsteps = int(cfg["nsteps"])
    t_end = (nsteps - 1) * Dt
    r = srk.call("simulate_sr_hetro", steps=nsteps, dt=Dt, npeople=n, seed=int(key), time_step_multiplier=m,
                 **_srk_kw(params))
    t_srk = _times(r, t_end)                                     # already (j + 1) Delta t
    s = simulate(params, n=n, key=int(key), dt=Dt / m, t_max=t_end, delta=delta, error="ks")
    ages = np.where(s.events == 1, s.ages, np.inf)
    on = lambda x, g: np.isclose(np.mod(x[np.isfinite(x)] + 1e-12, g), 0.0, atol=1e-9 * g) | np.isclose(  # noqa: E731
        np.mod(x[np.isfinite(x)] + 1e-12, g), g, atol=1e-9 * g)
    srk_on = float(np.mean(on(t_srk, Dt))) if np.isfinite(t_srk).any() else None
    sra_off = float(np.mean(~on(ages, Dt))) if np.isfinite(ages).any() else None
    informative = delta < Dt and srk_on == 1.0 and (sra_off or 0.0) > 0.0
    return {"id": "D-408", "n": n, "plant": {"Dt": Dt, "delta": delta, "srk_on_Dt_grid": srk_on,
                                             "sraging_off_Dt_grid": sra_off},
            "twin": {"note": "Delta t = delta: both on one grid"}, "sup_abs_dS": _sup(t_srk, ages),
            "informative": informative, "status": _status(informative, True)}


# ----------------------------------------------------------------------------------------------------- D-409
def d409(srk, params, cfg, key) -> dict:
    """getSrHetro's heterogeneous default against the named homogeneous call; sraging's default is homogeneous."""
    from sraging.langevin import simulate
    n, nsteps, t_end = int(cfg.get("n", N)), int(cfg["nsteps"]), float(cfg["t_end"])
    tsm = int(cfg.get("time_step_multiplier", 1))
    kw = _srk_kw(params)
    a = srk.call("get_sr_hetro", npeople=n, nsteps=nsteps, t_end=t_end, time_step_multiplier=tsm, seed=int(key), **kw)
    b = srk.call("get_sr_hetro", npeople=n, nsteps=nsteps, t_end=t_end, time_step_multiplier=tsm, seed=int(key) + 1,
                 hetro=False, **kw)
    plant = {**_ks2(_times(a, t_end), _times(b, t_end)), "default_xc_var": float(a["xc_var"])}
    dt = t_end / (nsteps - 1) / tsm
    s1 = simulate(params, n=2_000, key=int(key), dt=dt, t_max=t_end, error="ks")
    s2 = simulate(params, n=2_000, key=int(key), dt=dt, t_max=t_end, error="ks", heterogeneity=None)
    same = s1.times.tobytes() == s2.times.tobytes()
    informative = plant["p_value"] < 0.05 and plant["default_xc_var"] > 0
    return {"id": "D-409", "n": n, "plant": plant, "twin": {"sraging_default_homogeneous": same},
            "sup_abs_dS": plant["D"], "informative": informative, "status": _status(informative, same)}


# ----------------------------------------------------------------------------------------------------- D-410
def d410(params, cfg, key) -> dict:
    """The default clamp leaves exact zeros at a young age; reflect leaves none (ported-as-is: a measurement)."""
    from sraging.langevin import simulate
    n, dt, age = int(cfg.get("n", N)), float(cfg["dt"]), float(cfg["age"])
    out = {}
    for w in ("clamp", "reflect"):
        s = simulate(params, n=n, key=int(key), dt=dt, wall=w, record_ages=[age], error="ks", t_max=cfg.get("t_max"))
        x = s.paths.X[:, 0][np.isfinite(s.paths.X[:, 0])]
        out[w] = {"p0": float(np.mean(x == 0.0)), "n_alive": int(x.size)}
    informative = out["clamp"]["p0"] > 0
    return {"id": "D-410", "n": n, "plant": out["clamp"], "twin": out["reflect"], "sup_abs_dS": None, "age": age,
            "informative": informative, "status": _status(informative, out["reflect"]["p0"] == 0.0)}


# ----------------------------------------------------------------------------------------------------- D-411
def d411(srk, params, cfg, key) -> dict:
    """getISR's recorder (zeros for the dead) against sraging's (NaN), and sup|dF| of the living's X."""
    from sraging import sampler_rules as R
    from sraging.langevin import simulate
    n, dt, age = int(cfg.get("n", N)), float(cfg["dt"]), float(cfg["age"])
    t_end = math.ceil(1.25 * age / dt) * dt
    kw = _srk_kw(params)
    r = srk.call("simulate_isr_paths", npeople=n, nsteps=int(round(t_end / dt)) + 1, t_end=t_end, seed=int(key),
                 ages=[age], **kw)
    srk_check = R.hr10_nan_check(r["death_times"], np.asarray(r["ages"], float), r["trajectories"])
    s = simulate(params, n=n, key=int(key), dt=dt, record_ages=[age], error="ks", t_max=t_end)
    sra_check = R.hr10_nan_check(s.times, s.paths.ages, s.paths.X)
    alive_srk = np.asarray(r["death_times"], float) > age
    X_srk = np.asarray(r["trajectories"], float)[alive_srk, 0]
    X_sra = s.paths.X[:, 0][np.isfinite(s.paths.X[:, 0])]
    informative = srk_check["zeros_among_dead"] > 0
    return {"id": "D-411", "n": n, "age": age, "plant": srk_check, "twin": sra_check,
            "sup_abs_dS": _sup(X_srk, X_sra), "measure": "sup|dF| of the living's X at the age",
            "informative": informative, "status": _status(informative, sra_check["holds"])}


# ------------------------------------------------------------------------------------------------------ HR-8
def hr8_rows(params, cfg, key) -> tuple:
    """HR-8: bytes at each W against W = 1 (n = 25,013, not a multiple of the chunk or any W), and the B4 plant per
    W (n mod W dropped and padded)."""
    from sraging.langevin import simulate
    from sraging.langevin.pool import drop_remainder_times
    from sraging.langevin.sample import LangevinConfig
    n, dt = int(cfg.get("n", 25_013)), float(cfg["dt"])
    Ws = [int(w) for w in cfg.get("W", (1, 7, 15))]
    het = cfg.get("heterogeneity")
    base = dict(n=n, key=int(key), dt=dt, error="ks", heterogeneity=het)
    ref = simulate(params, **base, workers=1)
    rows, plants = [], []
    for W in Ws:
        s = ref if W == 1 else simulate(params, **base, workers=W)
        rows.append({"W": W, "n": n, "equal": all(getattr(s, f).tobytes() == getattr(ref, f).tobytes()
                                                   for f in ("times", "ages", "events"))})
        if W > 1:
            t = drop_remainder_times(LangevinConfig(params=params, **base), W)
            plants.append({"W": W, "dropped": n % W, "differs": t.tobytes() != ref.times.tobytes()})
    return rows, plants


# ---------------------------------------------------------------------------------------------- the phase's helpers
ORDER = ("hr9", "b2", "b4", "b11", "seeding", "hazard", "recording", "d409", "d410", "d411")
FUNCS = {"hr9": hr9, "b2": b2, "b4": b4, "b11": b11, "seeding": seeding, "hazard": hazard, "recording": recording,
         "d409": d409, "d411": d411}
#: SRK draws and sraging draws per plant (the projection's counts)
DRAWS = {"hr9": (2, 1), "b2": (2, 0), "b4": (1, 1), "b11": (2, 1), "seeding": (2, 3), "hazard": (4, 0),
         "recording": (1, 1), "d409": (2, 2), "d410": (0, 2), "d411": (1, 1)}


class CountingSession:
    """The oracle session, its workers' CPU (each reply's cpu_s) summed for SR's budget."""

    def __init__(self, srk):
        self.srk, self.cpu_s = srk, 0.0

    def call(self, op, **kw):
        r = self.srk.call(op, **kw)
        self.cpu_s += float(r.get("cpu_s", 0.0))
        return r


def run_plant(name, srk, params, cfg, key) -> dict:
    if name == "d410":
        return d410(params, cfg, key)
    return FUNCS[name](srk, params, cfg, key)


def plant_steps(name, cfg) -> tuple:
    """(SRK's step(s), sraging's step(s)) each plant draws at, one entry per draw (DRAWS' counts)."""
    t_end, nsteps = cfg.get("t_end"), cfg.get("nsteps")
    base = None if not (t_end and nsteps) else float(t_end) / (int(nsteps) - 1) / int(cfg.get("time_step_multiplier", 1))
    if name == "hr9":
        h = float(cfg["dt"]) / int(cfg.get("m", 16))
        return [h, h], [h, float(cfg.get("fix_dt") or h)]
    if name == "b2":
        return [float(cfg["step_size"])] * 2, []
    if name == "recording":
        h = float(cfg["dt"]) / int(cfg.get("m", 1))
        return [h], [h]
    if name == "hazard":
        return [float(cfg["dt"]) / int(cfg.get("time_step_multiplier", 1))] * 4, []
    if name in ("d410",):
        return [], [float(cfg["dt"])] * 2
    if name == "d411":
        return [float(cfg["dt"])], [float(cfg["dt"])]
    srk, sra = DRAWS[name]
    return [base] * srk, [base] * sra


def projected_cpu_h(spec, rates) -> float:
    """Each plant priced at its own steps (Naveh 2026-11-03): per draw, n x the individual's mean lifetime / the step
    x the CPU per individual-step, SRK's (from P1's re-measure) for an SRK draw, sraging's (P1's ns) for a sraging one.
    `rates`: {"lifetime": {preset: mean steps x step at P1's base}, "ns_srk": s, "ns_sraging": s}."""
    tot = 0.0
    for name in ORDER:
        cfg = spec[name]
        n = int(cfg.get("n", N))
        life = float(rates["lifetime"][cfg["preset"]])
        srk, sra = plant_steps(name, cfg)
        tot += sum(n * life / h * rates["ns_srk"] for h in srk) + sum(n * life / h * rates["ns_sraging"] for h in sra)
    return tot / 3600.0
