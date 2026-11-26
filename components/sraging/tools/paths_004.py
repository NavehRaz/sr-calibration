"""Spec 004's HR-10, damage paths against the FP damage density (tasks T098, T125; research R24, R25; contracts/
acceptance.md §4). The phase `paths` in tools/acceptance_004.py runs it under SR (cap 6 CPU-h).

**The configurations** are frozen at freeze-L (`langevin_004.toml [paths]`): each preset homogeneous and at xc sd 0.2
(S4's REFLECT_P1 classes, 8), at the two ages frozen for its preset (near S = 0.75 and 0.4 on the dt grid; `pick_ages`
proposes them at freeze-L), n = 25,000, under both walls, each at P1's step for (preset, hom | het, wall) with its
ladder capped at m_max = 2 m (gate C2). G = 16 tests per wall.

**A configuration** (`run_config`): the coupled ladder (simulate's error="ladder", record_ages) halves dt until every
age's path bias D / (2^p - 1) (p clipped to [1/2, 1]) is <= crit/10 at crit = 1.797 / sqrt(n_alive), or m reaches
m_max; its tests are read on the dt/2 rung (gate C1): n_alive, p0, u_c, x_c, D_c (censored) and D (full) against
`damage_density`, with the reference's grid error. The NaN check runs on the dt rung's paths and times.

**The plant** (`plant_delta`, `run_plant`): Langevin at eta (1 + delta) against the FP density at eta, at the frozen
plant age, delta sized so the predicted sup_x |dF| = 2 x 2.02 / sqrt(n_alive); its delta = 0 twin. Each wall judges
its own statistic (D_c under clamp, D under reflect) at P1's step.

**The deterministic checks** (`deterministic_checks`): times equal with and without record_ages; the paths' and
times' bytes equal at every W of [paths] W. **SRK's recorder** (`srk_recorder`, OFP `simulate_isr_paths`, read-only)
must fail the NaN check (D-411): zeros for the dead.
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

TARGETS = (0.75, 0.4)               # the two ages' observed survival
PLANT_C = 2.02                      # the 90 %-detection shift x sqrt(n_alive) at G = 16, censored (research R25)
CLASSES = S4.REFLECT_P1             # (hom, None, 0.0), (het, xc, 0.2)
WALLS = ("clamp", "reflect")
G = 16


class Refused(Exception):
    """A configuration HR-10 cannot be built from."""


def _p(p_hat) -> float:
    return min(max(float(p_hat), 0.5), 1.0)


def crit(n_alive) -> float:
    from sraging.sampler_rules import HR10_CRIT
    return HR10_CRIT / math.sqrt(n_alive)


def ages_of(spec, preset, het) -> list:
    """The configuration's HR-10 ages: keyed per preset and class ("mice_F|hom"; Naveh 2026-11-03), or per preset."""
    a = spec["ages"]
    return [float(x) for x in (a[f"{preset}|{het}"] if f"{preset}|{het}" in a else a[preset])]


def reflect_only(spec) -> set:
    """Amendment A2's clamp tests judged on reflect only: {(clamp cell, age index)}."""
    return {(r["cell"], int(i)) for r in spec.get("reflect_only") or () for i in r["ages"]}


def build_configs(spec: dict, p1: dict) -> list:
    """The rows per wall from the frozen [paths] spec and the frozen steps; refused without a step, or with ages off a
    row's dt grid. A preset in `clamp_excluded` (E. coli, Naveh 2026-11-03) has no clamp configuration; its reflect
    ones stay. G stays HR-10's 16 per wall (conservative where fewer clamp tests are judged)."""
    from sraging.errors import UnsupportedOptionError
    from sraging.langevin.sample import check_record_ages
    rows = []
    out_ = set(spec.get("clamp_excluded") or ())
    for wall in WALLS:
        for p in spec["presets"]:
            if wall == "clamp" and p in out_:
                continue
            for het, q, sd in CLASSES:
                k = S4.step_key(p, het, wall)
                if k not in p1["steps"]:
                    raise Refused(f"{p}/{het}/{wall}: no P1 step for {k} (HR-10 starts each row at P1's step)")
                st = p1["steps"][k]
                ages = ages_of(spec, p, het)
                try:
                    check_record_ages(ages, float(st["dt"]))
                except UnsupportedOptionError as e:
                    raise Refused(f"{p}/{het}/{wall}: {e}") from None
                rows.append({"id": f"{p}/{q or 'v0'}/sd{sd:g}/{wall}", "preset": p, "parameter": q, "sd": float(sd),
                             "wall": wall, "n": int(spec["n"]), "ages": ages, "m": int(st["m"]),
                             "dt": float(st["dt"]), "m_max": 2 * int(st["m"]), "p1_step": k})
    for wall in WALLS:
        per_wall = sum(len(r["ages"]) for r in rows if r["wall"] == wall)
        if per_wall > G or int(spec.get("G", G)) != G:
            raise Refused(f"[paths] has {per_wall} {wall} tests; HR-10's family is G = {G} (spec G = {spec.get('G')})")
    return rows


# ------------------------------------------------------------------------------------------------ ages and plant
def _reference(params, het):
    from sraging.langevin.sample import LangevinConfig, Reference
    ref = Reference(LangevinConfig(params=params, n=25_000, key=0, dt=1.0, heterogeneity=het, error="ks"))
    M = None
    if het is not None:
        tau, _ = ref.survival_grid(5)
        t = tau * ref.L
        M = ref.curve_check(t[ref.S_obs(t, 5) > 1e-12], 1e-4)["M"]
    return ref, M


def observed_survival(params, het, ages) -> np.ndarray:
    """The FP observed survival S_obs at `ages` (the hazard included: n_alive ~ n S_obs)."""
    ref, M = _reference(params, het)
    return ref.S_obs(np.asarray(ages, float), M)


def pick_ages(params, het, dt, targets=TARGETS) -> list:
    """The dt-grid ages whose S_obs is nearest each target (proposed at freeze-L)."""
    ref, M = _reference(params, het)
    tau, _ = ref.survival_grid(M)
    grid = np.arange(1, int(tau[-1] * ref.L / dt)) * dt
    S = ref.S_obs(grid, M)
    return [float(grid[int(np.argmin(np.abs(S - s)))]) for s in targets]


def plant_delta(params, age, n, het=None, c=PLANT_C, lo=0.0, hi=0.5, tol=1e-4) -> dict:
    """delta with the predicted sup_x |F_(eta (1 + delta))(x | alive, age) - F_eta(x | alive, age)| = 2 c / sqrt(
    n_alive), n_alive = n S_obs(age) (bisection, on the density's grid at eta with its node count fixed)."""
    from sraging import SRParams
    from sraging.engine.density import damage_density
    from sraging.engine.grid import GridPolicy
    from sraging.engine.inputs import Heterogeneity
    d0 = damage_density(params, [age], heterogeneity=None if het is None else
                        Heterogeneity(parameter=het.parameter, sd=het.sd))
    L0 = float(d0.grid["L"])
    grid = GridPolicy("fixed", N=int(d0.grid["N"]), dt=float(d0.grid["dt"]), tau_max=1.5 * age / L0 + 1e-6, L=L0)
    hfix = None if het is None else Heterogeneity(parameter=het.parameter, sd=het.sd, M=d0.M, check=False)
    S_obs = float(d0.S[0]) * math.exp(-float(params.h_ext) * age)
    target = 2.0 * c / math.sqrt(n * S_obs)
    x = np.linspace(0.0, float(np.max(d0.nodes["xc"])), 2001)
    F0 = d0.cdf_alive(x, age)

    def sup(delta):
        p = SRParams(**{**params.as_dict(), "eta": params.eta * (1.0 + delta)})
        return float(np.max(np.abs(damage_density(p, [age], heterogeneity=hfix, grid=grid).cdf_alive(x, age) - F0)))
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sup(mid) < target else (lo, mid)
    d = 0.5 * (lo + hi)
    return {"delta": d, "predicted_sup_dF": sup(d), "target": target, "S_obs": S_obs, "age": age, "M": d0.M}


# ---------------------------------------------------------------------------------------------- the runs
def _test_row(row, r, p, m, dt) -> dict:
    """One age of a sample's statement as HR-10's test row, read on the dt/2 rung."""
    base = {"id": f"{row['id']}/a{r['age']:g}", "age": r["age"], "wall": row["wall"], "p_hat": p, "m": m,
            "m_max": row["m_max"], "dt": dt, "reference_grid_error": r.get("reference_grid_error")}
    sb = r.get("step_bias")
    if not sb or "half" not in sb:
        return {**base, "n_alive": 0, "p0": None, "u_c": None, "x_c": None, "D_c": None, "D": None,
                "D_bias": math.nan, "reason": r.get("reason", "no dt/2 rung")}
    h = sb["half"]
    return {**base, "n_alive": h["n_alive"], "p0": h["p0"], "u_c": h["u_c"], "x_c": h["x_c"],
            "D_c": h["ks_D_censored"], "D": h["ks_D_full"], "D_bias": sb["D_dt_vs_dt2"],
            "judged_bias": sb["D_dt_vs_dt2"] / (2.0 ** p - 1.0), "D_dt": r["ks_D_full"], "D_c_dt": r["ks_D_censored"]}


def run_config(row, params, key, p_hat, *, het=None) -> dict:
    """One configuration: the coupled ladder from P1's step to every age's path bias <= crit/10 or m_max."""
    from sraging import sampler_rules as R
    from sraging.langevin import simulate
    p = _p(p_hat)
    m, dt, ladder = row["m"], row["dt"], []
    while True:
        s = simulate(params, n=row["n"], key=key, dt=dt, heterogeneity=het, wall=row["wall"], error="ladder",
                     record_ages=row["ages"])
        tests = [_test_row(row, r, p, m, dt) for r in s.error.paths]
        ok = all(t["n_alive"] and math.isfinite(t["D_bias"]) and t["judged_bias"] <= crit(t["n_alive"]) / 10
                 for t in tests)
        ladder.append({"m": m, "dt": dt, "tests": tests, "cpu_s": s.cost_record["cpu_s"],
                       "reference": s.error.paths_reference})
        if ok or 2 * m > row["m_max"]:
            break
        m, dt = 2 * m, dt / 2
    return {"id": row["id"], "wall": row["wall"], "n": row["n"], "m": m, "m_max": row["m_max"], "dt": dt,
            "tests": tests, "nan_check": R.hr10_nan_check(s.times, s.paths.ages, s.paths.X), "ladder": ladder,
            "key": key}


def run_frozen(row, params, key, frozen, *, het=None, skip_ages=()) -> dict:
    """Amendment A2 with R18's branch (Naveh 2026-11-03): the configuration ONCE at its frozen step (no ladder); its
    tests read on the dt/2 rung; each test's assessability is the frozen cell's pooled death-time bias (`frozen_b`),
    and the path bias (the coupled sup|F_dt - F_dt/2|) is reported, not gating. Ages in `skip_ages` (indices; amendment
    A2's reflect-only clamp tests) are drawn but not judged on this wall."""
    from sraging import sampler_rules as R
    from sraging.langevin import simulate
    p = float(frozen["p_hat"])
    s = simulate(params, n=row["n"], key=key, dt=row["dt"], heterogeneity=het, wall=row["wall"], error="ladder",
                 record_ages=row["ages"])
    tests, reported = [], []
    for j, r in enumerate(s.error.paths):
        t = _test_row(row, r, p, row["m"], row["dt"])
        t.update({"frozen_b": frozen["pooled_b"], "path_bias": t.pop("D_bias", None), "borrowed": False})
        t.pop("judged_bias", None)
        (reported if j in skip_ages else tests).append({**t, "reflect_only": j in skip_ages})
    return {"id": row["id"], "wall": row["wall"], "n": row["n"], "m": row["m"], "m_max": row["m_max"],
            "dt": row["dt"], "tests": tests, "not_judged_on_this_wall": reported,
            "nan_check": R.hr10_nan_check(s.times, s.paths.ages, s.paths.X), "key": key,
            "cpu_s": s.cost_record["cpu_s"]}


def run_plant(row, params, age, delta, key, *, het=None, kind="plant") -> dict:
    """The eta(1 + delta) draws' paths at `age` against the FP density at eta (kind "plant"; delta = 0 is the
    "twin"), the wall's statistic at P1's step."""
    from sraging import SRParams
    from sraging.engine.density import damage_density
    from sraging.engine.inputs import Heterogeneity
    from sraging.langevin import simulate
    from sraging.langevin import sample as LS
    p = SRParams(**{**params.as_dict(), "eta": params.eta * (1.0 + delta)})
    s = simulate(p, n=row["n"], key=key, dt=row["dt"], heterogeneity=het, wall=row["wall"], error="ks",
                 record_ages=[age])
    d = damage_density(params, [age], heterogeneity=None if het is None else
                       Heterogeneity(parameter=het.parameter, sd=het.sd))
    arrays = LS.draw_parameters(LS.LangevinConfig(params=p, n=row["n"], key=key, dt=row["dt"], heterogeneity=het,
                                                  error="ks"))
    x = s.paths.X[:, 0][np.isfinite(s.paths.X[:, 0])]
    t = LS.path_test(x, lambda z: d.cdf_alive(z, age), wall=row["wall"], step=row["dt"],
                     eps_max=float(np.max(arrays["epsilon"])),
                     k=1, x_max=max(float(np.max(arrays["xc"])), float(np.max(d.nodes["xc"]))))
    stat = t["ks_D_censored"] if row["wall"] == "clamp" else t["ks_D_full"]
    return {"id": f"{row['id']}/{kind}", "kind": kind, "wall": row["wall"], "delta": float(delta), "age": age,
            "statistic": stat, "n_alive": t["n_alive"], "D": t["ks_D_full"], "D_c": t["ks_D_censored"],
            "p0": t["p0"], "x_c": t["x_c"], "dt": row["dt"], "key": key}


def deterministic_checks(row, params, key, *, het=None, W_list=(1, 7, 15), chunk=1000) -> dict:
    """times equal with and without record_ages; the paths' and times' bytes at every W equal W = 1's."""
    from sraging.langevin import simulate
    base = dict(n=row["n"], key=key, dt=row["dt"], heterogeneity=het, wall=row["wall"], error="ks", chunk=chunk)
    a = simulate(params, **base)
    b = simulate(params, **base, record_ages=row["ages"])
    by_W = {}
    for W in W_list:
        c = b if int(W) == 1 else simulate(params, **base, record_ages=row["ages"], workers=int(W))
        by_W[str(int(W))] = c.paths.X.tobytes() == b.paths.X.tobytes() and c.times.tobytes() == b.times.tobytes()
    return {"times_equal": {"holds": a.times.tobytes() == b.times.tobytes(), "id": row["id"]},
            "bytes_by_W": {"holds": all(by_W.values()), "by_W": by_W, "id": row["id"], "chunk": chunk}}


def srk_recorder(session, params, ages, *, n, dt, seed) -> dict:
    """SRK's getISR path recorder (OFP simulate_isr_paths, read-only) through the NaN check: it writes 0 for the dead
    (D-411), so the check must fail."""
    from sraging import sampler_rules as R
    t_end = math.ceil(1.25 * max(ages) / dt) * dt
    kw = {"eta": params.eta, "beta": params.beta, "epsilon": params.epsilon, "kappa": params.kappa, "xc": params.xc,
          "npeople": int(n), "nsteps": int(round(t_end / dt)) + 1, "t_end": float(t_end), "seed": int(seed),
          "ages": [float(a) for a in ages]}
    if params.h_ext > 0:
        kw["external_hazard"] = -math.log(float(params.h_ext))
    r = session.call("simulate_isr_paths", **kw)
    return {"id": "srk_recorder", "kind": "srk_recorder", "n": int(n), "dt": dt, "t_end": t_end,
            "nan_check": R.hr10_nan_check(r["death_times"], np.asarray(r["ages"], float), r["trajectories"])}


def projected_cpu_h(rows, draw_cpu_s, spec) -> float:
    """Each configuration once at its frozen step (its draw CPU: p1-pooled's per-key coupled pair); the plant, twin
    and checks per wall at the plant configuration's draw CPU (2 + 2 + one per W), SRK's recorder at one draw."""
    per = {r["id"]: float(draw_cpu_s[r["p1_step"]]) for r in rows}
    pl = spec.get("plant") or {}
    k = S4.step_key(pl.get("preset"), pl.get("het", "hom"), "clamp")
    one = float(draw_cpu_s.get(k, max(per.values()) if per else 0.0))
    extra = len(WALLS) * (2 + 2 + len(spec.get("W", (1, 7, 15)))) * one + one
    return (sum(per.values()) + extra) / 3600.0
