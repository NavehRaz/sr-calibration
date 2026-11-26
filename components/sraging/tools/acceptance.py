"""Acceptance phases for spec 001 (tasks T041-T044, T050, T053; contracts/acceptance.md; research R6, R8).

    .venv/bin/python tools/acceptance.py pilot   [--workers 10] [--only ID,ID]
    .venv/bin/python tools/acceptance.py claims  [--workers 10] [--only ID,ID]
    .venv/bin/python tools/acceptance.py adapter [--only ID,ID]

pilot   oracle only: per panel cohort, theta points 2-5 and the named-role checks (+ substitution rule);
        writes tests/acceptance/panel_v1.toml and results/pilot.json. Commit the panel BEFORE `claims`.
claims  port and oracle side by side in one worker process per cohort (SC-001..003, SC-007).
adapter the tester's own comparison of the port's records against the stored reference (SC-004).

Every phase prints its estimated cost first (stored v0.3.3 CPU times), checks the parity preconditions
(NOT APPLICABLE -> exit 3, never a pass), and exits 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import oracle_v033 as O  # noqa: E402

RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
PANEL = os.path.join(REPO, "tests", "acceptance", "panel_v1.toml")
PARAMS = ("eta", "beta", "epsilon", "kappa")
BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
CORNER = {"eta": 1e-12, "beta": 1e3, "epsilon": 1e3, "kappa": 1e-3}
KAPPA_MAX = 300.0
ROLES = ("dt_ceiling", "basin_fires", "basin_blind", "censor_tie")
NAMED_INITIAL = [  # spec Clarifications 2026-10-03, Q2
    ("pure-k2-p04-c0-s0", "dt_ceiling"),
    ("realfull-real-mice_M-c1-s1", "dt_ceiling"),
    ("pure-real-Sweden_F_1910_homo-c0-s0", "basin_fires"),
    ("pure-real-Denmark_F_1900_hetro-c0-s0", "basin_fires"),
    ("censor-k0.0316-p05-c0-s1", "basin_fires"),
    ("het-k0.01-p01-c1-s0", "basin_fires"),
    ("pure-k0.01-p15-c1-s0", "basin_blind"),
    ("pure-k0.01-p15-c1-s1", "basin_blind"),
    ("censor-k0.01-p05-c0-s0", "censor_tie"),
    ("censor-k0.01-p05-c1-s1", "censor_tie"),
]
TIE_FRACTION = 0.005          # tools/fit_cohort_jax.py _TIE_FRACTION
GAIN_NATS = 1.92              # tools/fit_cohort_jax.py _GAIN_NATS


# ------------------------------------------------------------------------------ shared helpers
def now():
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head():
    r = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def stored_cpu(cid) -> float:
    try:
        return float(O.stored_reference_record(cid)["claim_set"].get("cpu_seconds") or 0.0)
    except (KeyError, OSError):
        return float("nan")


def print_cost(phase, cids, sides, workers):
    secs = [stored_cpu(c) for c in cids]
    known = [s for s in secs if math.isfinite(s)]
    total_h = sides * sum(known) / 3600.0
    wall_h = total_h / max(1, min(workers, len(cids)))
    print(f"[{phase}] estimated cost: {total_h:.2f} CPU-h over {len(cids)} cohorts x {sides} side(s) "
          f"(stored v0.3.3 CPU times: 30 quick-tier on the Mac, the rest on WEXAC; {len(cids) - len(known)} "
          f"unknown), roughly {wall_h:.2f} h wall on {workers} workers", flush=True)


def write_result(phase, payload):
    os.makedirs(RESULTS, exist_ok=True)
    tester_head = subprocess.run(["git", "-C", O.TESTER, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    payload = {"phase": phase, "written_at": now(), "repo_head": git_head(), "host": O.host_identity(),
               "tester": {"root": O.TESTER, "head": tester_head},
               **payload}
    with open(os.path.join(RESULTS, f"{phase}.json"), "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False, allow_nan=True)
        fh.write("\n")


def preflight_or_exit(phase):
    problems = O.preflight()
    if problems:
        write_result(phase, {"status": "not_applicable", "reasons": problems})
        print(f"[{phase}] NOT APPLICABLE:\n  " + "\n  ".join(problems), file=sys.stderr)
        raise SystemExit(3)


def _worker_init():
    sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import sraging.compat.v033  # noqa: F401  -- configures PJRT_NPROC=1 and x64 before any JAX use


def _guarded(fn, it):
    """One item in the worker. An exception becomes this item's recorded error: raised through the pool, an
    exception whose class does not unpickle in the parent (srbench's ContractError) breaks the whole pool
    and every pending item is reported crashed (2026-10-03)."""
    try:
        return fn(it)
    except Exception as e:  # noqa: BLE001 -- recorded with its traceback, never dropped
        import traceback
        return {"cohort_id": it if isinstance(it, str) else it[0], "status": f"error:{type(e).__name__}",
                "error": str(e)[:500], "traceback": traceback.format_exc()[-4000:]}


def run_pool(fn, items, workers):
    out, ctx = [], mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init) as ex:
        futs = {ex.submit(_guarded, fn, it): it for it in items}
        for k, f in enumerate(cf.as_completed(futs), 1):
            it = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 -- a worker crash is a recorded failure, never dropped
                r = {"cohort_id": it if isinstance(it, str) else it[0], "status": f"crash:{type(e).__name__}",
                     "error": str(e)[:500]}
            out.append(r)
            print(f"  [{k}/{len(items)}] {r.get('cohort_id')}: {r.get('status')} "
                  f"({r.get('wall_s', float('nan')):.0f} s)", flush=True)
    return out


# ------------------------------------------------------------------------------ per-individual terms
def per_individual_terms(fj, death_times, events, L, N, dt_, tau_max, Gtol, sampling_dt=None, nstart=20):
    """The per-individual log terms of `fp_jax.make_loglik`'s value function, unsummed.

    Mirrors fp_jax.py:324-369 (make_loglik -> build -> ll_of_z) line for line, returning
    `jnp.where(died, logp, logS)` instead of its sum. Used only to choose theta point 4 (a point where
    some term is at the -700 floor); the parity check itself then compares the oracle's own functions.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    L = float(L)
    nsteps = int(tau_max / dt_)
    tau = jnp.arange(nsteps + 1) * dt_
    teval = jnp.asarray(t / L)
    died = jnp.asarray(e == 1)
    if sampling_dt is not None and np.ndim(sampling_dt) == 0 and not (float(sampling_dt) > 0):
        sampling_dt = None
    if sampling_dt is not None:
        thi = jnp.asarray((t + np.broadcast_to(np.asarray(sampling_dt, float), t.shape)) / L)
    h = 1.0 / N
    march = fj._make_march(N, nsteps, dt_, nstart, Gtol, "value")

    def terms(z):
        eta, beta, eps, kap = jnp.exp(z[0]), jnp.exp(z[1]), jnp.exp(z[2]), jnp.exp(z[3])
        rb, rh, re = L * beta, L ** 2 * eta, L * eps
        yf = (jnp.arange(N) + 0.5) * h
        rem = rb * yf / (yf + kap)
        S_raw = march((rh, re, rem))
        S, p, k = fj._truncate(S_raw, tau, dt_, Gtol, nsteps)
        S_i = jnp.interp(teval, tau, S, left=1.0, right=S[-1])
        if sampling_dt is None:
            p_i = jnp.interp(teval, tau, p, left=0.0, right=0.0)
            p_phys = jnp.where(teval > tau[k], 0.0, p_i) / L
        else:
            S_hi = jnp.interp(thi, tau, S, left=1.0, right=S[-1])
            p_phys = jnp.clip(S_i - S_hi, 0.0, None)
        logp = jnp.maximum(jnp.log(jnp.clip(p_phys, fj._P_FLOOR, None)), fj._LOG_FLOOR)
        logS = jnp.maximum(jnp.log(jnp.clip(S_i, fj._P_FLOOR, None)), fj._LOG_FLOOR)
        return jnp.where(died, logp, logS)
    return jax.jit(terms)


# ------------------------------------------------------------------------------ pilot
def tie_fraction(ages, events, delta):
    """Fraction of deaths recorded at an end-of-study censoring age (the censor-tie rule's quantity)."""
    import numpy as np
    if not delta or np.all(events == 1):
        return 0.0
    tc = float(np.max(ages[events == 0]))
    if tc < float(np.max(ages)):
        return 0.0
    return float(np.sum((events == 1) & np.isclose(ages, tc))) / len(ages)


def oracle_claimset(o, cohort_id, res):
    """A reference srbench ClaimSet from estimator.fit's result, via the verbatim claim mapping."""
    from srbench.contract.types import Claim, ClaimSet
    from sraging.compat.v033 import claims as vclaims
    cl, diag = vclaims.build(res)
    return ClaimSet(cohort_id=cohort_id, method_name="mle_profile_jax",
                    claims=tuple(Claim(**{k: v for k, v in c.items()}) for c in cl), status="ok",
                    diagnostics=diag)


def pilot_worker(cid):
    import numpy as np
    t0 = time.perf_counter()
    o = O.load()
    lib = O.library()
    obs = O.observation(cid)
    ages, events = np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int)
    delta = obs.delta if obs.delta else None
    res = o.estimator.fit(o.fp_jax, list(obs.ages), list(obs.events), delta=obs.delta, levels=(0.95,),
                          kappa_max=KAPPA_MAX)
    cs = oracle_claimset(o, cid, res)
    basin_msgs = o.convergence_warnings(cs)
    theta_hat = [float(v) for v in res["theta"]]
    from SRK.fp_curvefit import fp_curvefit_seed       # the oracle's (snapshot) seed
    s = fp_curvefit_seed(ages, events, xc=1.0)
    seed = [float(s["eta"]), float(s["beta"]), float(s["epsilon"]), float(s["k"])]

    L, kw, ceiling = o.estimator.solver_grid(ages, events, delta)
    terms = per_individual_terms(o.fp_jax, ages, events, L, kw["N"], kw["dt"], kw["tau_max"], kw["Gtol"],
                                 kw.get("sampling_dt"))
    floor, rule = None, None
    for idx, pname in ((0, "eta"), (1, "beta")):
        for k in range(1, 7):
            th = list(theta_hat)
            th[idx] = min(th[idx] * 10.0 ** k, BOX[pname][1])
            vals = np.asarray(terms(np.log(np.array(th, float))))
            if np.any(vals == -700.0):
                floor, rule = th, f"optimum, {pname} x 10^{k} (clipped to box): {int(np.sum(vals == -700.0))} terms at -700"
                break
        if floor is not None:
            break
    if floor is None:
        rule = "no k in 1..6 for eta or beta reaches the -700 floor; point 4 = optimum, eta at box upper"
        floor = list(theta_hat)
        floor[0] = BOX["eta"][1]
    kmax = list(theta_hat)
    kmax[3] = KAPPA_MAX
    truth = lib.truth(cid)
    return {
        "cohort_id": cid, "status": "ok", "wall_s": time.perf_counter() - t0,
        "dt_ceiling": res.get("dt_ceiling"), "basin_messages": basin_msgs, "basin_fires": bool(basin_msgs),
        "tie_fraction": tie_fraction(ages, events, delta), "n": int(len(ages)),
        "points": [
            {"label": "truth", **{p: float(truth[p]) for p in PARAMS}, "rule": "library v1 physical_truth()"},
            {"label": "optimum", **dict(zip(PARAMS, theta_hat)), "rule": "pilot oracle theta-hat"},
            {"label": "seed", **dict(zip(PARAMS, seed)), "rule": "oracle fp_curvefit_seed(t, e, xc=1.0)"},
            {"label": "floor", **dict(zip(PARAMS, floor)), "rule": rule},
            {"label": "kmax", **dict(zip(PARAMS, kmax)), "rule": "optimum, kappa = 300"},
            {"label": "corner", **CORNER, "rule": "box corner (eta lo, beta hi, epsilon hi, kappa lo)"},
        ],
        "oracle_claims": [{k: getattr(c, k) for k in ("parameter", "shape", "lower", "upper", "point")}
                          for c in cs.claims],
        "loglik": res.get("loglik"),
    }


def role_ok(role, r):
    if r.get("status") != "ok":
        return False, f"pilot status {r.get('status')}"
    if role == "dt_ceiling":
        return r["dt_ceiling"] is not None, f"dt_ceiling={r['dt_ceiling']!r}"
    if role == "basin_fires":
        return r["basin_fires"], f"basin warnings on this host: {len(r['basin_messages'])}"
    if role == "basin_blind":
        return True, f"kept by rule; basin warning fired here: {r['basin_fires']}"
    if role == "censor_tie":
        return r["tie_fraction"] >= TIE_FRACTION, f"tie fraction {r['tie_fraction']:.4%}"
    raise ValueError(role)


def substitution_candidates(role, exclude):
    """Cohorts qualifying for `role` by the STORED v0.3.3 records (cheapest stored CPU first)."""
    import numpy as np
    man = json.load(open(os.path.join(O.TESTER, "runs", "mle_profile_jax-full.json")))
    rows = []
    lib = O.library()
    for cid in man["cohort_ids"]:
        if cid in exclude or getattr(lib.spec(cid), "window", None) is not None:
            continue                       # windowed observations are refused by the port (D-104)
        cs = O.stored_reference_record(cid)["claim_set"]
        d = cs.get("diagnostics") or {}
        if role == "dt_ceiling":
            ok = d.get("dt_ceiling") is not None
        elif role == "basin_fires":
            grid = d.get("grid") or {}
            gains = [g.get("gain_nats", 0.0) for g in grid.values() if isinstance(g, dict) and g.get("recentred")]
            outside = any(c.get("shape") == "closed" and None not in (c.get("point"), c.get("lower"), c.get("upper"))
                          and not c["lower"] * (1 - 1e-9) <= c["point"] <= c["upper"] * (1 + 1e-9)
                          for c in cs.get("claims", []))
            ok = max(gains, default=0.0) > GAIN_NATS or outside
        elif role == "censor_tie":
            if not cid.startswith("censor"):
                continue
            obs = O.observation(cid)
            ok = tie_fraction(np.asarray(obs.ages, float), np.asarray(obs.events, int), obs.delta) >= TIE_FRACTION
        else:
            ok = False
        if ok:
            rows.append((float(cs.get("cpu_seconds") or math.inf), cid))
    return [c for _, c in sorted(rows)]


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else ("nan" if math.isnan(v) else ("inf" if v > 0 else "-inf"))
    if isinstance(v, int):
        return str(v)
    if v is None:
        return '""'
    return json.dumps(str(v), ensure_ascii=False)


def write_panel(quick, named, results):
    lines = ["# Acceptance panel for spec 001 (FR-021). Written by `tools/acceptance.py pilot`;",
             "# COMMIT before `tools/acceptance.py claims` (it refuses an uncommitted panel).",
             f'library = "v1"', f'tester_head = "f8eaed2d56d7b237d01e38f140fe1d71a8c83717"',
             "quick = [" + ", ".join(toml_value(c) for c in quick) + "]", ""]
    for n in named:
        lines += ["[[named]]"] + [f"{k} = {toml_value(n.get(k, ''))}"
                                  for k in ("cohort_id", "role", "pilot_outcome", "substituted_for")] + [""]
    by = {r["cohort_id"]: r for r in results}
    for cid in quick + [n["cohort_id"] for n in named]:
        r = by[cid]
        lines += ["[[theta]]", f"cohort_id = {toml_value(cid)}", "points = ["]
        for p in r["points"]:
            lines.append("  { " + ", ".join(f"{k} = {toml_value(p[k])}" for k in ("label", *PARAMS, "rule")) + " },")
        lines += ["]", ""]
    lines += ["[pilot]", f"run_id = {toml_value('pilot-' + dt.datetime.now().strftime('%Y%m%d-%H%M'))}",
              f"host_uuid = {toml_value(O.host_identity()['uuid'])}", f"finished_at = {toml_value(now())}",
              f"cpu_hours = {toml_value(sum(r.get('wall_s', 0.0) for r in results) / 3600.0)}", ""]
    with open(PANEL, "w") as fh:
        fh.write("\n".join(lines))
    tomllib.load(open(PANEL, "rb"))                      # must parse


def phase_pilot(args):
    preflight_or_exit("pilot")
    quick = O.quick_tier()
    named = [{"cohort_id": c, "role": r, "substituted_for": ""} for c, r in NAMED_INITIAL]
    cids = quick + [n["cohort_id"] for n in named]
    if args.only:
        cids = [c for c in cids if c in args.only]
    print_cost("pilot", cids, 1, args.workers)
    results = run_pool(pilot_worker, cids, args.workers)
    by = {r["cohort_id"]: r for r in results}
    if not args.only:
        for n in named:
            ok, why = role_ok(n["role"], by[n["cohort_id"]])
            n["pilot_outcome"] = why
            tried = 0
            while not ok and tried < 5:
                on_panel = {m["cohort_id"] for m in named} | set(quick) | set(by)
                cands = substitution_candidates(n["role"], on_panel)
                if not cands:
                    break
                sub = cands[0]
                print(f"[pilot] {n['cohort_id']} fails role {n['role']} ({why}); trying {sub}", flush=True)
                r = run_pool(pilot_worker, [sub], 1)[0]
                by[sub] = r
                results.append(r)
                ok, why = role_ok(n["role"], r)
                tried += 1
                if ok:
                    n.update(substituted_for=n["cohort_id"], cohort_id=sub, pilot_outcome=why)
            n["role_ok"] = ok
        write_panel(quick, named, [by[c] for c in quick + [n["cohort_id"] for n in named]])
    status = "pass" if all(r.get("status") == "ok" for r in results) and all(n.get("role_ok", True) for n in named) \
        else "fail"
    write_result("pilot", {"status": status, "named": named, "results": results})
    print(f"[pilot] {status}: {sum(r.get('status') == 'ok' for r in results)}/{len(results)} cohorts ok"
          + (f"; roles {sum(n.get('role_ok', False) for n in named)}/{len(named)} confirmed" if not args.only
             else "; roles not evaluated (--only)")
          + ("" if args.only else f"; panel -> {os.path.relpath(PANEL, REPO)} (commit it before `claims`)"))
    return 0 if status == "pass" else 1


# ------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["pilot", "claims", "adapter"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", type=lambda s: [x for x in s.split(",") if x], default=None)
    ap.add_argument("--no-srbench", default=None, metavar="REASON",
                    help="claims: skip the srbench record check, recorded as not applicable with this reason")
    ap.add_argument("--result-name", default=None,
                    help="claims: write results/<name>.json instead of results/claims.json (spec 002's re-run)")
    args = ap.parse_args(argv)
    if args.phase == "pilot":
        return phase_pilot(args)
    if args.phase == "claims":
        from acceptance_claims import phase_claims
        return phase_claims(args)
    from acceptance_claims import phase_adapter
    return phase_adapter(args)


if __name__ == "__main__":
    raise SystemExit(main())
