"""Spec 002 T1 phases: `t1-pilot` (the low-mass cut), `t1` (SC-002) and `mpref` (low-mass and round-off terms
against mpmath). Run through tools/acceptance_002.py; see its docstring for the provenance rules.

T1 compares numerics "v033" with the old code like for like (same grid, same scheme):
- library cohorts with SRK's frozen NumPy likelihood (`fp_loglik_physical`; per term `piecewise_loglik_terms`
  at zero intervention when there is no window; S on the march grid from `fpt_full`);
- mixtures with `fp_loglik_het_physical` (total only: SRK returns no terms for a mixture);
- segment configurations with `fp_piecewise` (per term);
- staggered entry with tests/reference/numpy_scheme.py, whose terms are built here by definition.
Per evaluation: total |dl| <= 1e-5; every resolved term <= 1e-8; max |dS| <= 1e-8; the gradient within 1e-6
relative of autodiff through a FIXED-LENGTH march feeding the same assembly (uniform mesh); central
differences within the derived bound. Terms with mass below the frozen cut are routed to `mpref`; their
number and the totals with and without them are reported (sr-manager gate, condition 3). Every feature
configuration is also put through the CI guard's register-condition check, v1 against v033 (condition 4), and
the window cohorts' v1 - v033 totals are recorded for D-107's impact (condition 6).
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import acceptance_002 as A

PANEL_T1 = "tests/acceptance/panel_002_t1.toml"
PANEL_ROUNDOFF = "tests/acceptance/panel_002_roundoff.toml"
TOTAL, TERM, DS, GRAD_REL = 1e-5, 1e-8, 1e-8, 1e-6
CD_STEPS = (3e-3, 1e-3, 3e-4, 1e-4, 1e-5)       # four-point stencil, truncation O(h^4)
U = 2.220446049250313e-16
CUT_MARGIN = 4.0
THETA = {"truth": {}, "eta x1.5": {"eta": 1.5}, "beta x1.5": {"beta": 1.5}, "epsilon x1.5": {"epsilon": 1.5},
         "eta x10": {"eta": 10.0}}
# The cut's pilot: per arm the 5th medium-tier cohort in library order (outside the panel), plus one human
# life table outside every spec 002 panel (S ~ 1 for a long head, where the cut matters).
T1_PILOT = ["pure-k0.01-p01-c0-s0", "hazard-k0.01-p01-c0-s0", "het-k0.01-p01-c0-s0", "censor-k0.01-p01-c0-s0",
            "window-k0.01-p01-c0-s0", "realfull-real-mice_M-c0-s0", "realfull-real-Denmark_M_1900_homo-c0-s0"]
SEG = dict(L=124.0, N=300, dt=2.0 ** -8, tau_max=2.5, EXTH=6.0,
           TH=dict(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1))


# ------------------------------------------------------------------------------------------- helpers
def _scale(th, label):
    return {k: v * THETA[label].get(k, 1.0) for k, v in th.items()}


def library_case(cid, panel):
    """(ages, events, delta, window, truth, ExtH or None, het or None) for a library cohort."""
    O = A.O
    obs = O.observation(cid)
    t, e = np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int)
    row = _validity()[cid]
    ext = float(row["external_hazard"]) if row["external_hazard"] else None
    het = (row["het_param"], float(row["het_var"])) if row["het_param"] else None
    tr = dict(O.library().truth(cid))
    return dict(t=t, e=e, d=obs.delta, window=obs.window, truth={k: tr[k] for k in A.PARAMS}, ext=ext, het=het)


_VAL = {}


def _validity():
    if not _VAL:
        import csv
        _VAL.update({r["cohort_id"]: r for r in csv.DictReader(open(os.path.join(A.O.TESTER, "reports",
                                                                                   "validity", "validity.csv")))})
    return _VAL


def seg_cohort():
    rng = np.random.default_rng(7)
    t = np.sort(np.floor(rng.gamma(9.0, 14.0, 500)) + 1.0)
    e = (rng.random(500) < 0.9).astype(int)
    return t, e


def entry_ages(rule, t, e):
    if rule == "e_i = floor(0.5 t_i)":
        return np.floor(0.5 * t)
    if rule.startswith("e_i = 0.37 t_i"):
        en = 0.37 * t
        if "every 20th death row" in rule:
            d_rows = np.flatnonzero(e == 1)[::20]
            en[d_rows] = t[d_rows] + 0.5
        return en
    raise ValueError(f"unknown entry rule {rule!r}")


def fixed_length_march(spec):
    """V's march at fixed length: lax.scan over every step, no early stop, autodiff through the solve."""
    import jax
    import jax.numpy as jnp
    from sraging.engine import stepper as ST
    N, nsteps, dt, nstart = spec.N, spec.nsteps, spec.dt, spec.nstart
    h = 1.0 / N

    def march(theta):
        p0 = jnp.zeros(N).at[0].set(1.0 / h)

        def body(carry, n):
            p, pp = carry
            pn = jnp.maximum(ST._step_q(theta, n, p, pp, N, h, dt, nstart), 0.0)
            return (pn, p), pn.sum() * h
        _, S = jax.lax.scan(body, (p0, jnp.zeros(N)), jnp.arange(nsteps))
        return jnp.concatenate([jnp.array([p0.sum() * h]), S])
    return march


def grad_checks(ev, p, noise=0.0):
    """(value, grad dict, fixed-length relative error or None, central-difference worst ratio to its bound).
    `noise`: the S route's worst-case per-term round-off, recorded only: as a bound it made the check vacuous
    (hundreds of tiny-mass terms), while the measured estimates agreed to 1e-6; real noise shows instead as
    consecutive steps that disagree, and then the check is not applicable."""
    import jax
    v, g = ev.value_and_grad(p)
    z = ev._z(p)
    gv = np.array([g[n] for n in ev.layout.names])
    rel = None                      # None: no fixed-length reference (general march) or it is not finite
    if getattr(ev, "_gspec", None) is None:
        ref = jax.value_and_grad(ev._build(fixed_length_march(ev._spec)))
        vr, gr = ref(z)
        gr = np.asarray(gr)
        if np.all(np.isfinite(gr)):
            scale = max(1e-3 * float(np.max(np.abs(gr))), 1e-300)
            rel = float(np.max(np.abs(gv - gr) / np.maximum(np.abs(gr), scale)))
    f = ev.objective()[0]
    # The derived bound assumes the function is smooth at the scale of the step. Smooth at that scale is taken to
    # mean that the four-point estimates at two consecutive steps agree within the smaller step's bound, and the
    # check is then |CD - g| at the smaller step. It is informative only if that bound is below 1% of the largest
    # gradient component (else the function's own noise hides any error). If no consecutive pair is both, the
    # check is not applicable -- never a pass (BDF2's positivity clip at an extreme point; an early stop that
    # moves the value, D-115). A wrong gradient on a smooth function still fails: there the estimates agree with
    # each other and not with g.
    gmax = float(np.max(np.abs(gv)))
    prev = None
    for hstep in CD_STEPS:
        cds = []
        for i in range(z.size):
            dz = np.zeros(z.size)
            dz[i] = hstep
            fs = [float(f(z + k * dz)) for k in (1, -1, 2, -2)]
            cds.append((8 * (fs[0] - fs[1]) - (fs[2] - fs[3])) / (12 * hstep))
        cds = np.array(cds)
        bound = 50 * 2.2e-16 * max(abs(v), 1.0) / hstep + 1e-6 * gmax   # real noise shows as disagreement
        if prev is not None and np.max(np.abs(prev - cds)) <= bound:
            err = float(np.max(np.abs(cds - gv)))
            if err == 0.0:
                return v, g, rel, 0.0
            if bound <= 0.01 * gmax:
                return v, g, rel, err / bound
        prev = cds
    return v, g, rel, None


def guard_check(cohort, kw, p):
    """The CI guard's per-term register-condition check, v1 against v033, for a feature configuration."""
    from sraging import divergences
    from sraging.engine import guard
    from sraging.engine.evaluator import Evaluator
    conds = guard.conditions([{"id": e.id, "condition": e.condition} for e in divergences.load()])
    e33 = Evaluator(cohort, numerics="v033", **kw)
    e1 = Evaluator(cohort, numerics="v1", **kw)
    rep = guard.compare(e33, e1, p, conds)
    d107 = None
    if cohort.window is not None:                  # D-107's own rows: deaths with t <= t_hi < t + delta
        t33, _ = e33.terms(p)
        t1, _ = e1.terms(p)
        a, e_ = np.asarray(cohort.ages), np.asarray(cohort.events) == 1
        reach = a + (np.broadcast_to(np.asarray(cohort.delta, float), a.shape) if cohort.delta is not None else 0.0)
        rows = e_ & (a <= cohort.window[1]) & (reach > cohort.window[1])
        d107 = {"rows": int(rows.sum()), "dl": float(np.sum((t1 - t33)[rows]))}
    return {"ok": rep.ok, "covered": rep.covered, "n_violations": len(rep.violations), "d107": d107,
            "violations": rep.violations[:10], "grad_uncovered_rel": rep.grad_uncovered_rel,
            "v1_minus_v033": float(e1.value(p) - e33.value(p))}


def ns_terms(res, cohort, L, h):
    """Per-row terms by definition on the NumPy reference march (entry configurations): deaths
    log+ [S_obs(max(t, e)) - S_obs(t + delta)] (exact times: log+ of the observed density), censored
    log+ S_obs(t), each minus log+ S_obs(e); S linear on the mesh, 1 before it and S_end after it."""
    from reference import numpy_scheme as NS
    S, _ = NS.monotone(res)
    tau = res.tau

    def logp(x):
        return np.maximum(np.log(np.clip(x, math.exp(-700.0), None)), -700.0)

    def S_obs(x_phys):
        return np.interp(x_phys / L, tau, S, left=1.0, right=S[-1]) * np.exp(-h * x_phys)

    t = np.asarray(cohort.ages, float)
    died = np.asarray(cohort.events) == 1
    en = np.zeros_like(t) if cohort.entry is None else np.asarray(cohort.entry, float)
    if cohort.delta is not None:
        d = np.broadcast_to(np.asarray(cohort.delta, float), t.shape)
        ts = np.maximum(t, en)
        m = np.clip(S_obs(ts) - S_obs(t + d), 0.0, None)
        dead = logp(m)
    else:
        S_r, f = NS.density_route(tau, res.S_raw)
        f_i = np.interp(t / L, tau, f, left=0.0, right=0.0) / L
        S_i = np.interp(t / L, tau, S, left=1.0, right=S[-1])
        dead = logp((f_i + h * S_i) * np.exp(-h * t))
    terms = np.where(died, dead, logp(S_obs(t)))
    return terms - logp(S_obs(en))


def guard_feature(f, panel):
    """(cohort, evaluator keywords, truth theta, h_ext) for a guard-panel feature configuration."""
    from sraging import Cohort
    from sraging.engine.inputs import AgeSegments, Heterogeneity, Segment
    kind = f["kind"]
    if kind == "segments":
        sp = panel["segments"]
        cfg = next(x for x in sp["configs"] if x["name"] == f["config"])
        t, e = seg_cohort()
        win = tuple(cfg["window"]) if cfg.get("window") else None
        seg = Segment(id="s", parameter=cfg["parameter"], value=1.0 + cfg["effect"], window=win)
        from sraging.engine.grid import GridPolicy
        g = sp["grid"]
        h = math.exp(-sp["external_hazard_level"])
        kw = dict(grid=GridPolicy("fixed", N=g["N"], dt=g["dt"], tau_max=g["tau_max"], L=g["L"]),
                  segments=AgeSegments(seg), fixed={"h_ext": h})
        return Cohort.from_arrays(t, e, delta=1.0), kw, dict(sp["base_theta"]), h
    c = library_case(f["cohort"], panel)
    t, e, d = c["t"], c["e"], c["d"]
    h = 0.0 if c["ext"] is None else math.exp(-c["ext"])
    kw = {"fixed": {"h_ext": h}}
    if kind == "exact":
        coh = Cohort.from_arrays(t, e, continuous=True)
    elif kind == "entry":
        coh = Cohort.from_arrays(t, e, delta=d, entry=entry_ages(f["rule"], t, e))
    else:
        coh = Cohort.from_arrays(t, e, delta=d, window=c["window"])
    if kind == "het":
        kw["heterogeneity"] = Heterogeneity(parameter=c["het"][0], definition="normal_frac_sd", v=c["het"][1], M=f["M"])
    return coh, kw, c["truth"], h


# ---------------------------------------------------------------------------------------- the cases
def t1_cases(panel):
    """Every unit of work: (case id, kind, scheme)."""
    out = []
    for arm, ids in panel["arms"].items():
        out += [(cid, "library", s) for cid in ids for s in panel["schemes"]]
    for hw in panel["het_window"]:
        out += [(f"het_window:{hw['base']}:{par}", "het_window", s) for par in hw["parameters"]
                for s in panel["schemes"]]
    out += [(f"entry:{x['name']}", "entry", s) for x in panel["entry"] for s in panel["schemes"]]
    out += [(f"segments:{c['name']}", "segments", s) for c in panel["segments"]["configs"] for s in panel["schemes"]]
    return out


_W = {}


def _worker_init():
    sys.pycache_prefix = os.path.join(A.REPO, ".cache", "pycache")
    for p in (os.path.join(A.REPO, "tools"), os.path.join(A.REPO, "tests", "parity"), os.path.join(A.REPO, "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import sraging.compat.v033  # noqa: F401  -- PJRT_NPROC=1 and x64 before any JAX use


def _srk():
    import oracle_fp_piecewise as OP
    if "srk" not in _W:
        _W["srk"] = OP.Session()
    return _W["srk"]


def run_unit(unit, panel, cut):
    """All theta points of one (case, scheme). Returns a list of evaluation records."""
    try:
        return _run_unit(unit, panel, cut)
    except Exception as ex:                                   # noqa: BLE001 -- recorded, never a pass
        import traceback
        return [{"case": unit[0], "kind": unit[1], "scheme": unit[2], "ok": False,
                 "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-3000:]}]


def case_setup(case, kind, scheme, panel):
    """The cohort, the evaluator keywords, the base theta and the grid of a library, het_window or entry case,
    exactly as `t1` builds them (shared with `mpref`)."""
    from sraging import Cohort
    from sraging.engine.grid import GridPolicy
    from sraging.engine.inputs import Heterogeneity
    if kind in ("library", "het_window"):
        if kind == "library":
            cid = case
            c = library_case(cid, panel)
            win, het_par = c["window"], c["het"]
            v_het, M = (c["het"][1] if c["het"] else None), panel["het_M"]
        else:
            _, cid, het_par_name = case.split(":")
            hw = next(x for x in panel["het_window"] if x["base"] == cid)
            c = library_case(cid, panel)
            win, het_par, v_het, M = tuple(hw["window"]), (het_par_name, hw["v"]), hw["v"], hw["M"]
        t, e, d = c["t"], c["e"], c["d"]
        cont, en, rule = False, None, None
        g0 = GridPolicy("v033").grid(Cohort.from_arrays(t, e, delta=d))
        coh = Cohort.from_arrays(t, e, delta=d, window=win)
    elif kind == "entry":
        x = next(x for x in panel["entry"] if f"entry:{x['name']}" == case)
        c = library_case(x["base"], panel)
        t, e = c["t"], c["e"]
        rule = x["rule"]
        cont = "exact" in rule
        d = None if cont else c["d"]
        en = entry_ages(rule, t, e)
        win, het_par, v_het, M = None, None, None, None
        g0 = GridPolicy("v033").grid(Cohort.from_arrays(t, e, delta=d, continuous=cont))
        coh = Cohort.from_arrays(t, e, delta=d, continuous=cont, entry=en)
    else:
        raise ValueError(f"case_setup has no {kind!r} case")
    ext = c["ext"]
    h = 0.0 if ext is None else math.exp(-ext)
    grid = GridPolicy("fixed", N=g0.N, dt=g0.dt, tau_max=g0.tau_max, Gtol=g0.Gtol, L=g0.L)
    het = Heterogeneity(parameter=het_par[0], definition="normal_frac_sd", v=v_het, M=M) if het_par else None
    kw = dict(grid=grid, scheme=scheme, fixed={"h_ext": h}, heterogeneity=het)
    return dict(coh=coh, kw=kw, truth=c["truth"], h=h, ext=ext, g0=g0, L=g0.L, t=t, e=e, d=d, win=win, cont=cont,
                entry=en, rule=rule, het=(het_par[0], v_het, M) if het_par else None)


def _run_unit(unit, panel, cut):
    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.engine.inputs import AgeSegments, Heterogeneity, Segment
    A.O.load()
    case, kind, scheme = unit
    srk = _srk()
    recs = []
    if kind in ("library", "het_window"):
        cs = case_setup(case, kind, scheme, panel)
        c, coh, kw, h, ext, g0 = cs, cs["coh"], cs["kw"], cs["h"], cs["ext"], cs["g0"]
        t, e, d, win, het = cs["t"], cs["e"], cs["d"], cs["win"], kw["heterogeneity"]
        het_par, v_het, M = ((cs["het"][0], cs["het"][1]), cs["het"][1], cs["het"][2]) if cs["het"] else (None,) * 3
        ev = Evaluator(coh, numerics="v033", **kw)
        L = g0.L
        gk = dict(N=g0.N, dt=g0.dt, tau_max=g0.tau_max, Gtol=g0.Gtol, scheme=scheme)
        for label in panel["theta_labels"]:
            th = _scale(c["truth"], label)
            p = SRParams(**th, h_ext=h)
            t_c = time.process_time()
            terms, aux = ev.terms(p)
            r = {"case": case, "kind": kind, "scheme": scheme, "theta": label, "theta_values": th, "n": int(coh.n),
                 "grid": g0.summary(), "window": list(win) if win else None, "ext_hazard_level": ext,
                 "het": [het_par[0], v_het, M] if het_par else None}
            if het is None:
                ref = srk.call("fp_loglik_physical", death_times=t, events=e, xc=1.0, L=L,
                               external_hazard=math.inf if ext is None else ext,
                               time_range=list(win) if win else None, sampling_dt=d, **th, **gk)
            else:
                ref = srk.call("fp_loglik_het_physical", death_times=t, events=e, xc=1.0, L=L, het_param=het_par[0],
                               het_var=v_het, M=M, time_range=list(win) if win else None, sampling_dt=d,
                               **th, **gk)
            r["value_engine"], r["value_ref"] = float(np.sum(terms)), float(ref["value"])
            r["d_total"] = r["value_engine"] - r["value_ref"]
            fp = srk.call("fpt_full", rho_eta=L ** 2 * th["eta"], rho_beta=L * th["beta"], rho_eps=L * th["epsilon"],
                          kappa=th["kappa"], **gk)
            _, S, _, _ = ev.survival_on_grid(p)
            S_ref = _monotone(np.asarray(fp["S"]))
            r["dS"] = float(np.max(np.abs(S[:S_ref.size] - S_ref)))
            r["K"] = ev.stiffness(p)
            cut_e = _cut(r, cut)
            mass = np.where(e == 1, aux["mass"], aux["S_i"])
            routed = mass < cut_e
            r["n_routed"], r["routed_rows"] = int(routed.sum()), np.flatnonzero(routed).tolist()
            if win is None and het is None:
                rt = srk.call("piecewise_loglik_terms", death_times=t, events=e,
                              theta=dict(**th, xc=1.0), interventions=[], L=L, t_end=float(t.max()),
                              external_hazard=math.inf if ext is None else ext, sampling_dt=d, **gk)
                _per_term(r, terms, np.asarray(rt["terms"]), routed)
            _finish(r, ev, p, coh, kw, t_c, _noise(r, ev, p, mass, cut))
            recs.append(r)
        return recs
    if kind == "entry":
        cs = case_setup(case, kind, scheme, panel)
        c, coh, kw, h, ext, g0 = cs, cs["coh"], cs["kw"], cs["h"], cs["ext"], cs["g0"]
        t, e, d, cont = cs["t"], cs["e"], cs["d"], cs["cont"]
        x = {"rule": cs["rule"]}
        ev = Evaluator(coh, numerics="v033", **kw)
        from reference import numpy_scheme as NS
        for label in panel["theta_labels"]:
            th = _scale(c["truth"], label)
            p = SRParams(**th, h_ext=h)
            t_c = time.process_time()
            terms, aux = ev.terms(p)
            L = g0.L
            res = NS.march(g0.N, NS.uniform_mesh(g0.dt, g0.tau_max),
                           theta=(L ** 2 * th["eta"], L * th["beta"], L * th["epsilon"], th["kappa"]),
                           scheme=scheme, nstart=20, Gtol=g0.Gtol)
            ref_terms = ns_terms(res, coh, L, h)
            r = {"case": case, "kind": kind, "scheme": scheme, "theta": label, "theta_values": th, "n": int(coh.n),
                 "grid": g0.summary(), "ext_hazard_level": ext, "entry_rule": x["rule"], "continuous": cont}
            r["value_engine"], r["value_ref"] = float(np.sum(terms)), float(np.sum(ref_terms))
            r["d_total"] = r["value_engine"] - r["value_ref"]
            _, S, _, _ = ev.survival_on_grid(p)
            S_ref, _ = NS.monotone(res)
            n_ = min(S.size, S_ref.size)
            r["dS"] = float(np.max(np.abs(S[:n_] - S_ref[:n_])))
            r["K"] = ev.stiffness(p)
            cut_e = _cut(r, cut)
            mass = np.where(e == 1, aux["mass"] * (g0.dt * g0.L if cont else 1.0), aux["S_i"])
            routed = mass < cut_e
            r["n_routed"], r["routed_rows"] = int(routed.sum()), np.flatnonzero(routed).tolist()
            _per_term(r, terms, ref_terms, routed)
            _finish(r, ev, p, coh, kw, t_c, _noise(r, ev, p, mass, cut))
            recs.append(r)
        return recs
    if kind == "segments":
        sp = panel["segments"]
        cfg = next(x for x in sp["configs"] if f"segments:{x['name']}" == case)
        t, e = seg_cohort()
        par = cfg["parameter"]
        win = tuple(cfg["window"]) if cfg.get("window") else None
        if par == "h_ext":
            ivs = [{"kind": "ExtHazard", "level": cfg["level"], "window": list(win)}]
            segs = [Segment(id="s", parameter="h_ext", value=math.exp(-cfg["level"]), window=win)]
        else:
            ivs = [{"kind": "Intervention", "param": par, "effect": cfg["effect"], "window": list(win) if win else None}]
            segs = [Segment(id="s", parameter=par, value=1.0 + cfg["effect"], window=win)]
        g = sp["grid"]
        coh = Cohort.from_arrays(t, e, delta=1.0)
        h = math.exp(-sp["external_hazard_level"])
        kw = dict(grid=GridPolicy("fixed", N=g["N"], dt=g["dt"], tau_max=g["tau_max"], L=g["L"]), scheme=scheme,
                  segments=AgeSegments(*segs), fixed={"h_ext": h})
        ev = Evaluator(coh, numerics="v033", **kw)
        for label in panel["theta_labels"]:
            th = _scale(sp["base_theta"], label)
            p = SRParams(**th, h_ext=h)
            t_c = time.process_time()
            terms, aux = ev.terms(p)
            rt = srk.call("piecewise_loglik_terms", death_times=t, events=e, theta=dict(**th, xc=1.0), interventions=ivs,
                          L=g["L"], t_end=float(t.max()), external_hazard=sp["external_hazard_level"], sampling_dt=1.0,
                          N=g["N"], dt=g["dt"], tau_max=g["tau_max"], Gtol=1e-11, scheme=scheme)
            ref_terms = np.asarray(rt["terms"])
            r = {"case": case, "kind": kind, "scheme": scheme, "theta": label, "theta_values": th, "n": int(coh.n),
                 "segment": cfg}
            r["value_engine"], r["value_ref"] = float(np.sum(terms)), float(np.sum(ref_terms))
            r["d_total"] = r["value_engine"] - r["value_ref"]
            r["dS"] = None                                        # terms are the stronger check here
            r["K"] = ev.stiffness(p)
            cut_e = _cut(r, cut)
            mass = np.where(e == 1, aux["mass"], aux["S_i"])
            routed = mass < cut_e
            r["n_routed"], r["routed_rows"] = int(routed.sum()), np.flatnonzero(routed).tolist()
            _per_term(r, terms, ref_terms, routed)
            _finish(r, ev, p, coh, kw, t_c, _noise(r, ev, p, mass, cut))
            recs.append(r)
        return recs
    raise ValueError(f"unknown kind {kind!r}")


def _monotone(S):
    """SRK's survival as the likelihood uses it: running minimum, clipped to [0, 1] (BDF2 can exceed 1)."""
    return np.clip(np.minimum.accumulate(np.asarray(S, float)), 0.0, 1.0)


def _cut(r, rule):
    """This evaluation's low-mass cut (review T064, sr-naveh N2 and sr-manager M2): two S values each off by
    dS move a term of mass m by about 2 dS / m, so terms with m >= margin x 2 dS / 1e-8 are held to 1e-8.
    dS is this evaluation's measured |dS| where an S comparison exists; otherwise the pilot's
    stiffness-scaled bound u (c K + f). Both are floored at the pilot's floor."""
    pred = U * (rule["c"] * r["K"] + rule["f"])
    dS = pred if r.get("dS") is None else max(r["dS"], U * rule["f"])
    r["cut"], r["cut_source"] = CUT_MARGIN * 2.0 * dS / TERM, ("measured dS" if r.get("dS") is not None
                                                               else "pilot bound u (c K + f)")
    return r["cut"]


def _per_term(r, terms, ref_terms, routed):
    diff = np.abs(np.asarray(terms) - np.asarray(ref_terms))
    res = ~routed
    r["worst_resolved_term"] = float(diff[res].max()) if res.any() else 0.0
    r["worst_routed_term"] = float(diff[routed].max()) if routed.any() else 0.0
    r["d_total_resolved"] = float(np.sum((np.asarray(terms) - np.asarray(ref_terms))[res]))
    r["d_total_routed"] = float(np.sum((np.asarray(terms) - np.asarray(ref_terms))[routed]))


def _noise(r, ev, p, mass, rule):
    """The v033 objective's evaluation noise: the S route's round-off on each term, u (c K + f) / m_i (terms on
    the floor excluded)."""
    lim = U * (rule["c"] * r["K"] + rule["f"])
    m = np.asarray(mass, float)
    m = m[m > np.exp(-700.0)]
    return float(np.sum(np.log1p(lim / m)))           # a term moves by log(1 + dm/m) at most


def _finish(r, ev, p, coh, kw, t_c, noise=0.0):
    v, g, rel, cdw = grad_checks(ev, p, noise)
    r["cd_noise"] = noise
    r["grad"] = {k: float(x) for k, x in g.items()}
    r["grad_rel_fixed_length"], r["cd_worst_ratio"] = rel, cdw
    r["value_from_vg_rel"] = abs(v - r["value_engine"]) / max(abs(r["value_engine"]), 1e-300)
    r["guard"] = guard_check(coh, kw, p)
    if coh.window is not None:
        r["d107_rows"] = int(np.sum((np.asarray(coh.events) == 1) & (np.asarray(coh.ages) <= coh.window[1])
                                    & (np.asarray(coh.ages) + np.asarray(coh.delta, float) > coh.window[1])))
    fails = []
    if not abs(r["d_total"]) <= TOTAL:
        fails.append(f"total {r['d_total']:.3e}")
    if r.get("worst_resolved_term") is not None and not r["worst_resolved_term"] <= TERM:
        fails.append(f"resolved term {r['worst_resolved_term']:.3e}")
    if r.get("dS") is not None and not r["dS"] <= DS:
        fails.append(f"dS {r['dS']:.3e}")
    if rel is not None and not rel <= GRAD_REL:
        fails.append(f"grad vs fixed length {rel:.3e}")
    if cdw is None:
        r["cd_not_applicable"] = ("no two consecutive steps give estimates that agree within a bound below 1% of "
                                  "the gradient (the function is not smooth, or too noisy, at the scales tried)")
    elif not cdw <= 1.0:
        fails.append(f"central differences {cdw:.2f} x bound")
    if not r["value_from_vg_rel"] <= 1e-13:
        fails.append(f"value_and_grad value {r['value_from_vg_rel']:.1e} rel")
    if r["guard"].get("ok") is False:
        fails.append(f"guard: {r['guard']['n_violations']} violations, grad {r['guard']['grad_uncovered_rel']:.1e}")
    r["fails"], r["ok"] = fails, not fails
    r["cpu_s"] = time.process_time() - t_c


# ---------------------------------------------------------------------------------------------- phases
def phase_t1_pilot(args):
    """The low-mass cut's constants (spec SC-002; review T064, sr-naveh N2, sr-manager M2): the engine-vs-NumPy
    |dS| on the march grid at the truth, both schemes, on cohorts outside the T1 panel, as u (c K + f) with
    K = rho_eps dtau N^2: c = margin x the worst |dS|/(u K) at the stiff points (K >= 1e4), f = margin x the
    worst excess |dS|/u - (c/margin) K. A T1 evaluation's cut is margin x 2 x dS / 1e-8 with its own measured
    dS, or this bound where no S comparison exists (segments, mixtures)."""
    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    panel = A.load_toml(PANEL_T1)
    in_panel = {c for ids in panel["arms"].values() for c in ids}
    assert not set(T1_PILOT) & in_panel, "the pilot must be disjoint from the panel"
    run = A.Run("t1-pilot", [PANEL_T1], args.dev, f"~5 CPU-min ({len(T1_PILOT)} cohorts x 2 schemes)")
    A.preflight_or_exit(run)
    import oracle_fp_piecewise as OP
    rows = []
    with OP.session() as srk:
        for cid in T1_PILOT:
            c = library_case(cid, panel)
            g0 = GridPolicy("v033").grid(Cohort.from_arrays(c["t"], c["e"], delta=c["d"]))
            grid = GridPolicy("fixed", N=g0.N, dt=g0.dt, tau_max=g0.tau_max, Gtol=g0.Gtol, L=g0.L)
            L, th = g0.L, c["truth"]
            for scheme in panel["schemes"]:
                ev = Evaluator(Cohort.from_arrays(c["t"], c["e"], delta=c["d"]), numerics="v033", grid=grid,
                               scheme=scheme)
                p = SRParams(**th)
                _, S, _, k = ev.survival_on_grid(p)
                fp = srk.call("fpt_full", rho_eta=L ** 2 * th["eta"], rho_beta=L * th["beta"],
                              rho_eps=L * th["epsilon"], kappa=th["kappa"], N=g0.N, dt=g0.dt, tau_max=g0.tau_max,
                              Gtol=g0.Gtol, scheme=scheme)
                S_ref = _monotone(fp["S"])
                dS = float(np.max(np.abs(S[:S_ref.size] - S_ref)))
                K = ev.stiffness(p)
                rows.append({"cohort": cid, "scheme": scheme, "K": K, "max_dS": dS, "ulps": dS / U,
                             "ratio_uK": dS / (U * K)})
                print(f"  {cid:42s} {scheme:4s} K {K:9.3g} max|dS| {dS:.3e} = {dS / U:8.1f} u = {dS / (U * K):.3f} u K",
                      flush=True)
    stiff = [r for r in rows if r["K"] >= 1e4]
    b = max((r["ratio_uK"] for r in stiff), default=0.0)
    for r in rows:
        r["excess_ulps"] = max(r["ulps"] - b * r["K"], 0.0)
    c_, f_ = CUT_MARGIN * b, CUT_MARGIN * max(r["excess_ulps"] for r in rows)
    print(f"[t1-pilot] stiff points {len(stiff)}: worst {b:.4f} u K -> c = {c_:.4f}; worst excess "
          f"{f_ / CUT_MARGIN:.1f} u -> f = {f_:.1f}", flush=True)
    return run.write({"status": "measured", "cohorts": T1_PILOT, "margin": CUT_MARGIN, "c": c_, "f": f_,
                      "n_stiff": len(stiff), "rows": rows})


def phase_t1(args):
    panel = A.load_toml(PANEL_T1)
    cut = {k: panel["low_mass"].get(k) for k in ("c", "f")}
    units = t1_cases(panel)
    if args.only:
        units = [u for u in units if any(tok in u[0] for tok in args.only)]      # substrings of case names
    n_eval = len(units) * len(panel["theta_labels"])
    run = A.Run("t1", [PANEL_T1], args.dev, f"~1-2 CPU-h ({len(units)} case x scheme units, {n_eval} evaluations; "
                f"the oracle worker dominates); on {args.workers} worker(s)")
    if args.cut is not None:                       # development only (main() sets --dev): "c,f"
        cut = dict(zip(("c", "f"), (float(x) for x in str(args.cut).split(","))))
    if not all(isinstance(cut[k], float) for k in ("c", "f")):
        sys.exit("[t1] REFUSED: the panel's low-mass constants are not frozen yet (run t1-pilot, write c and f, "
                 "commit)")
    A.preflight_or_exit(run)
    recs = []
    if args.workers <= 1:
        _worker_init()
        for u in units:
            out = run_unit(u, panel, cut)
            recs += out
            _print(out)
    else:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_worker_init) as ex:
            futs = {ex.submit(run_unit, u, panel, cut): u for u in units}
            for f in cf.as_completed(futs):
                try:
                    out = f.result()
                except Exception as exn:                       # noqa: BLE001 -- a dead worker is a failure
                    u = futs[f]
                    out = [{"case": u[0], "kind": u[1], "scheme": u[2], "ok": False, "error": f"crash: {exn!r}"}]
                recs += out
                _print(out)
    recs.sort(key=lambda r: (r["case"], r["scheme"], r.get("theta", "")))
    summary = _summarize(recs, panel)
    status = "pass" if all(r.get("ok") for r in recs) and not args.only else ("smoke" if args.only else "fail")
    return run.write({"status": status, "cut": cut, "thresholds": {"total": TOTAL, "term": TERM, "dS": DS,
                                                                    "grad_rel": GRAD_REL},
                      "summary": summary, "results": recs})


def _print(out):
    for r in out:
        if "error" in r:
            print(f"  {r['case']:44s} {r['scheme']:4s} ERROR {r['error'][:200]}", flush=True)
            continue
        print(f"  {r['case'][:44]:44s} {r['scheme']:4s} {r['theta']:12s} dl {r['d_total']:+.2e} "
              f"term {r.get('worst_resolved_term', float('nan')):.1e} routed {r['n_routed']:5d} "
              f"dS {r['dS'] if r['dS'] is not None else float('nan'):.1e} g {r['grad_rel_fixed_length'] or 0:.1e} "
              f"cd {r['cd_worst_ratio'] if r['cd_worst_ratio'] is not None else float('nan'):.2f} guard {r['guard'].get('ok')} {'ok' if r['ok'] else 'FAIL ' + '; '.join(r['fails'])}",
              flush=True)


def _summarize(recs, panel):
    ok = [r for r in recs if r.get("ok")]
    by_kind = {}
    for r in recs:
        k = r.get("kind", "?")
        b = by_kind.setdefault(k, {"evaluations": 0, "passed": 0, "worst_d_total": 0.0, "worst_resolved_term": 0.0,
                                   "worst_dS": 0.0, "n_routed": 0})
        b["evaluations"] += 1
        b["passed"] += bool(r.get("ok"))
        if "error" in r:
            continue
        b["worst_d_total"] = max(b["worst_d_total"], abs(r["d_total"]))
        b["worst_resolved_term"] = max(b["worst_resolved_term"], r.get("worst_resolved_term") or 0.0)
        b["worst_dS"] = max(b["worst_dS"], r["dS"] or 0.0)
        b["n_routed"] += r["n_routed"]
    window_d107 = [{"case": r["case"], "scheme": r["scheme"], "theta": r["theta"], "d107_rows": r["d107_rows"],
                    "d107_dl": (r["guard"].get("d107") or {}).get("dl"),
                    "v1_minus_v033_all_terms": r["guard"].get("v1_minus_v033")}
                   for r in recs if "d107_rows" in r]
    n_by_arm = {arm: len(ids) for arm, ids in panel["arms"].items()}
    return {"evaluations": len(recs), "passed": len(ok), "by_kind": by_kind, "N_per_arm": n_by_arm,
            "largest_n": max((r.get("n", 0) for r in recs), default=0), "d107_window_cohorts": window_d107}


def phase_mpref(args):
    import acceptance_002_mpref as M
    return M.phase_mpref(args)
