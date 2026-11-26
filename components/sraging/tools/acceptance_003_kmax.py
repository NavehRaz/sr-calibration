"""Spec 003 `kmax` phase (FR-025, SC-007; research R15; task T039; sr-manager gate 4).

Are raw-beta upper bounds on ratio-lo cohorts set by kappa's box (kappa_max = 300) rather than by the data?
  1. confirm: walk the panel's seeded candidate list (panel_003.toml [kmax] draw: every candidate, in order) in
     batches of `workers`, fitting each with profile_jax2's statements at kappa_max = 300 and reading, at the node that
     sets beta's upper end (against the beta profile's own maximum, as `profile_ci` measures), whether kappa sits within
     0.01 dex of kappa_max; stop once KEEP are confirmed, keeping the first KEEP in list order (or the list ends: the
     result says so). The 3 negative controls get the same kappa_max = 300 fit, in the first batch;
  2. refit the KEEP and the negative controls with the port's own estimator at kappa_max = 3000 (a study setting,
     not a registered calibrator: the same run_fit statements with `kappa_max` overridden);
  3. rule: raw-beta caps are box-limited if the cap moves >= 0.5 dex on >= 8 of 10. A cap that is gone at 3000 (no
     finite upper end) has moved without limit (+inf), and is counted as moved;
  4. the caps on r-bar and r-bar/eta are walked in the same runs (the combination walker on each fit's objective);
     they should not move, which tests the combination choice.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVE_DEX, NEED, KEEP = 0.5, 8, 10
EDGE_DEX = 0.01
FIT_CPU_S = 400.0               # the stored Mac fits' order of magnitude (155-470 CPU-s), for the printed price


def _init():
    sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
    for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def fit_at(cohort, kappa_max):
    """run_fit's statements on the port with kappa_max overridden; returns beta's cap and the combination caps."""
    import numpy as np
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.claims import walker as W
    from sraging.compat.jax2 import claims as JC, estimator as JE, fp_jax as FJ, observation_model as OM
    cls = PJ.CLASSES["profile_jax2"]
    obs = PJ.observation(cohort)
    with PJ.capture_objectives() as objs, PJ.capture_nodes() as nodes:
        prep = OM.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=cls.WINDOW, relabel_on=cls.RELABEL)
        res = JE.fit(FJ, prep.ages, prep.events, delta=obs.delta, levels=(JC.LEVEL,), kappa_max=kappa_max,
                     hazard=False, window=prep.window_eff, exact_from_F=True, self_consistent=True,
                     floor_early=True, n_rule=True, early_all=True)
    lo, hi = res["ci"].get("beta@0.95", (float("nan"), float("nan")))
    verdict = (res["diagnostics"].get("verdicts") or {}).get("beta")
    obj = objs[0]
    z_hat = np.log(np.asarray(res["theta"], float))
    bounds = [(math.log(obj.box[p][0]), math.log(obj.box[p][1])) for p in obj.params]
    box4 = {p: obj.box[p] for p in ("eta", "beta", "epsilon", "kappa")}
    combos = {}
    for key in ("rbar", "rbar_over_eta"):
        w = W.walk(obj, key, z_hat, float(res["loglik"]), bounds=bounds, box4=box4,
                   archive=JE._sc_archive(obj))
        combos[key] = {"ci_lo": w["ci"]["ci_lo"], "ci_hi": w["ci"]["ci_hi"], "verdict": w["ci"]["verdict"]}
    beta_nodes = nodes.get(1, [])
    return {"beta_upper": hi, "beta_lower": lo, "beta_verdict": verdict, "loglik": res["loglik"],
            "theta": [float(x) for x in res["theta"]], "combos": combos, "beta_nodes": beta_nodes}


def kappa_at_cap(beta_nodes, loglik):
    """kappa at the node that sets beta's upper end, or None. Nodes are (value, ll, FULL log z) (profile_jax2
    `capture_nodes`), so z[3] is log kappa; the reference is the beta profile's own maximum (`profile_ci`'s)."""
    import numpy as np
    from sraging.claims import shapes as SH
    ns = [(v, ll, None if z is None else np.asarray(z)) for v, ll, z in beta_nodes if ll is not None and
          math.isfinite(ll)]
    if not ns:
        return None
    ll_ref = max([float(loglik)] + [n[1] for n in ns])
    node = SH.setting_node(ns, ll_ref, "upper")
    return None if node is None or node[2] is None else math.exp(float(node[2][3]))


def confirm_worker(cid):
    try:
        import oracle_jax2 as OJ
        from sraging.adapters.observation import cohort_from_observation
        c0 = time.process_time()
        cohort = cohort_from_observation(OJ.observation(cid))
        r = fit_at(cohort, 300.0)
        k_at = kappa_at_cap(r["beta_nodes"], r["loglik"])
        confirmed = (r["beta_verdict"] == "one-sided:<" and k_at is not None and k_at >= 300.0 / 10 ** EDGE_DEX)
        r.pop("beta_nodes")
        return {"cohort_id": cid, "confirmed": confirmed, "kappa_at_cap_node": k_at, "fit300": r,
                "cpu_s": time.process_time() - c0}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def refit_worker(cid):
    try:
        import oracle_jax2 as OJ
        from sraging.adapters.observation import cohort_from_observation
        c0 = time.process_time()
        cohort = cohort_from_observation(OJ.observation(cid))
        r = fit_at(cohort, 3000.0)
        r.pop("beta_nodes")
        return {"cohort_id": cid, "fit3000": r, "cpu_s": time.process_time() - c0}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def _pool(fn, ids, workers):
    out = {}
    if not ids:
        return out
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = {ex.submit(fn, c): c for c in ids}
        for f in cf.as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"cohort_id": futs[f], "error": f"crash: {e}"}
            out[r["cohort_id"]] = r
            print(f"  {r['cohort_id']}: {'ERROR ' + r['error'][:80] if 'error' in r else 'ok'}", flush=True)
    return out


def confirm_along(draw, neg, run_batch, batch, keep=KEEP):
    """Step 1: confirm along `draw` in order, `batch` candidates at a time (the negative controls ride in the first
    batch), until `keep` are confirmed or the list ends. `run_batch(ids)` -> {cid: confirm row}. Returns (kept: the
    first `keep` confirmed in list order, conf: every row, tried: the candidates fitted, ended: True if the list ran
    out first). Every candidate before the last kept one is fitted, so the kept set does not depend on timing."""
    conf, i = {}, 0
    while True:
        kept = [c for c in draw[:i] if conf.get(c, {}).get("confirmed")][:keep]
        if len(kept) == keep or i >= len(draw):
            return kept, conf, draw[:i], len(kept) < keep
        ids = draw[i:i + batch] + ([c for c in neg if c not in conf] if i == 0 else [])
        conf.update(run_batch(ids))
        i += batch


def _moved(a, b, refit_ok=True):
    """log10(b / a) for caps a (kappa_max 300) and b (3000); +inf when a is a finite cap and the refit (which ran)
    has none; None when a is not a finite positive cap or the refit failed."""
    if a is None or not math.isfinite(a) or a <= 0 or not refit_ok:
        return None
    if b is None or not math.isfinite(b):
        return math.inf
    return math.log10(b / a) if b > 0 else None


def _cap(fit, key=None):
    if fit is None:
        return None
    return fit.get("beta_upper") if key is None else ((fit.get("combos") or {}).get(key) or {}).get("ci_hi")


def phase_kmax(args):
    import acceptance_003 as A
    panel = A.load_toml(A.PANEL)["kmax"]
    draw, neg = list(panel["draw"]), list(panel["negative_controls"])
    if args.only:
        draw, neg = [c for c in draw if c in args.only], [c for c in neg if c in args.only]
    lo_fits = min(KEEP, len(draw)) + 2 * len(neg) + min(KEEP, len(draw))
    run = A.Run("kmax", manifests=[A.PANEL], dev=args.dev or bool(args.only),
                cost=f"confirm fits along the seeded list of {len(draw)} (at least {min(KEEP, len(draw))}, at most "
                     f"{len(draw)}; {len(neg)} negative controls with them), then {min(KEEP, len(draw)) + len(neg)} "
                     f"refits at kappa_max = 3000, each with two combination walks: at least {lo_fits} fits, about "
                     f"{lo_fits * FIT_CPU_S / 3600:.1f} CPU-h at {FIT_CPU_S:.0f} CPU-s a fit, {args.workers} workers")
    A.preflight_or_exit(run)
    kept, conf, tried, ended = confirm_along(draw, neg, lambda ids: _pool(confirm_worker, ids, args.workers),
                                             max(1, args.workers))
    ref = _pool(refit_worker, kept + neg, args.workers)
    rows = []
    for c in kept + neg:
        a, b = conf.get(c, {}).get("fit300"), ref.get(c, {}).get("fit3000")
        ok3000 = b is not None
        rows.append({"cohort_id": c, "negative_control": c in neg,
                     "beta_cap_moved_dex": _moved(_cap(a), _cap(b), ok3000),
                     "rbar_cap_moved_dex": _moved(_cap(a, "rbar"), _cap(b, "rbar"), ok3000),
                     "rbar_over_eta_cap_moved_dex": _moved(_cap(a, "rbar_over_eta"), _cap(b, "rbar_over_eta"), ok3000),
                     "fit300": a, "fit3000": b})
    n_moved = sum(1 for r in rows if not r["negative_control"] and (r["beta_cap_moved_dex"] or 0) >= MOVE_DEX)
    complete = len(kept) == KEEP and all(r["fit300"] is not None and r["fit3000"] is not None for r in rows)
    verdict = ("box-limited" if n_moved >= NEED else "not shown box-limited") if complete else \
        (f"inconclusive: the seeded list ended after {len(tried)} candidates with {len(kept)} of {KEEP} confirmed"
         if ended else "inconclusive: a kept cohort or negative control has no kappa_max 300 or 3000 fit")
    run.write({"status": "written" if complete else "incomplete", "verdict": verdict,
               "n_moved_ge_0p5_dex": n_moved, "kept": kept, "negative_controls": neg, "rule": {
                   "move_dex": MOVE_DEX, "need": NEED, "of": KEEP, "edge_dex": EDGE_DEX,
                   "vanished_cap": "+inf (counted as moved)"},
               "tried": tried, "list_ended": ended, "list_length": len(draw), "confirm": conf, "rows": rows,
               "errors": [r for r in list(conf.values()) + list(ref.values()) if "error" in r]})
