"""Spec 002 acceptance phases (specs/002-fp-engine/contracts/acceptance.md).

    .venv/bin/python tools/acceptance_002.py <phase> [--dev] [options]

Every phase prints its estimated cost first and writes tests/acceptance/results/<phase>_002.json with its
provenance: the repo HEAD, whether the tracked tree is dirty, the host, the tester revision. A phase REFUSES
to run on a dirty tracked tree or with a manifest that is not committed unchanged; `--dev` runs it anyway and
labels the result "development", which never counts (tests/acceptance/test_results_002.py refuses it). The
HEAD and the tree are checked again when the result is written: a run during which the tree changed is
labelled development too. Never edit a tracked file while a phase runs (spec 001's lesson); long phases run
from a separate git worktree at the committed HEAD (`git worktree add --detach .cache/wt/run HEAD`, then
`PYTHONPATH=$PWD/src` inside it) so editing can continue in the main checkout.

Phases: t0, guard-pilot, guard, t1-pilot, t1, mpref, early, anchor, speed, floors-expect, floors
(score: tools/score_test.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import tomllib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "tests", "parity"), os.path.join(REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import oracle_v033 as O  # noqa: E402

RESULTS = os.path.join(REPO, "tests", "acceptance", "results")
ACC = os.path.join(REPO, "tests", "acceptance")
PARAMS = ("eta", "beta", "epsilon", "kappa")
PANEL_V1 = "tests/acceptance/panel_v1.toml"
PANEL_GUARD = "tests/acceptance/panel_002_guard.toml"

# The C_S pilot (research R14): cohorts NOT in the guard panel (tests/acceptance/panel_002_guard.toml), at
# their six frozen theta points from spec 001's panel. Quick tier p02-p03 plus two named cohorts where
# v0.3.3's round-off is known (one basin human cohort, one heterogeneous synthetic).
GUARD_PILOT = ["pure-k0.01-p02-c0-s0", "pure-k0.01-p02-c0-s1", "pure-k0.01-p02-c1-s0", "pure-k0.01-p02-c1-s1",
               "pure-k0.01-p03-c0-s0", "pure-k0.01-p03-c0-s1", "pure-k0.01-p03-c1-s0", "pure-k0.01-p03-c1-s1",
               "pure-real-Denmark_F_1900_hetro-c0-s0", "het-k0.01-p01-c1-s0"]
C_S_MARGIN = 4.0


def now():
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args) -> str:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def tree_dirty() -> bool:
    """Tracked changes outside tests/acceptance/results/: a result file is an output, not an input of the code's
    provenance (a phase that overwrites a committed result must not make the next phase refuse)."""
    return bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)tests/acceptance/results"))


def committed(rel: str) -> bool:
    """True if `rel` is tracked and identical to HEAD."""
    return bool(git("ls-files", rel)) and not git("status", "--porcelain", "--", rel)


def last_commit(rel: str) -> str:
    return git("log", "-1", "--format=%H", "--", rel)


class Run:
    """One phase run: the provenance gate at the start, the re-check and the result file at the end."""

    def __init__(self, phase, manifests=(), dev=False, cost=""):
        self.phase, self.manifests, self.dev = phase, list(manifests), dev
        self.head0, self.dirty0 = git("rev-parse", "HEAD"), tree_dirty()
        self.uncommitted = [m for m in self.manifests if not committed(m)]
        print(f"[{phase}] estimated cost: {cost}", flush=True)
        if (self.dirty0 or self.uncommitted) and not dev:
            why = (["the tracked tree is dirty"] if self.dirty0 else []) + \
                  ([f"manifest(s) not committed unchanged: {', '.join(self.uncommitted)}"] if self.uncommitted else [])
            sys.exit(f"[{phase}] REFUSED: {'; '.join(why)} (commit first, or pass --dev for a development run that "
                     f"never counts)")
        self.t0, self.c0 = time.time(), time.process_time()

    def write(self, payload):
        os.makedirs(RESULTS, exist_ok=True)
        tester_head = subprocess.run(["git", "-C", O.TESTER, "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()
        head1, dirty1 = git("rev-parse", "HEAD"), tree_dirty()
        changed = head1 != self.head0 or dirty1 != self.dirty0
        dev = self.dev or self.dirty0 or dirty1 or bool(self.uncommitted) or changed
        out = {"phase": self.phase, "written_at": now(), "repo_head": self.head0, "tree_dirty": self.dirty0 or dirty1,
               "tree_changed_during_run": changed, "manifests": {m: last_commit(m) for m in self.manifests},
               "manifests_uncommitted": self.uncommitted,
               "status_for_acceptance": "development" if dev else "acceptance",
               "host": O.host_identity(), "tester": {"root": O.TESTER, "head": tester_head},
               "wall_s": time.time() - self.t0, "cpu_s": time.process_time() - self.c0, **payload}
        path = os.path.join(RESULTS, f"{self.phase.replace('-', '_')}_002.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, allow_nan=True, default=_jsonable)
            fh.write("\n")
        print(f"[{self.phase}] status {out.get('status')}; wrote {os.path.relpath(path, REPO)} "
              f"({out['status_for_acceptance']})", flush=True)
        return out


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def preflight_or_exit(run):
    try:
        O.load()
    except O.NotApplicable as ex:
        run.write({"status": "not_applicable", "reasons": ex.problems})
        sys.exit(3)


def load_toml(rel):
    return tomllib.load(open(os.path.join(REPO, rel), "rb"))


# ------------------------------------------------------------------------------------------------- t0
def phase_t0(args):
    """With every feature off (numerics "v033", grid "v033", scheme "bdf2") the engine IS v0.3.3: the same
    canonical program text as the port and as the frozen oracle, and the same value and gradient bytes, on all
    40 cohorts of spec 001's panel x their 6 frozen theta points (spec SC-001, FR-021)."""
    import jax.numpy as jnp
    from engine_cases import arrays, cohort, panel_ids, theta_points
    from parity_checks import bytes_diff
    from sraging.compat.v033 import estimator as port_est
    from sraging.compat.v033 import fp_jax as V
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.engine.program import canonical_text

    run = Run("t0", [PANEL_V1], args.dev, "~5 CPU-min (40 cohorts x 6 theta, compile dominated)")
    preflight_or_exit(run)
    ofj = O.load().fp_jax
    ids = panel_ids() if not args.only else args.only
    rows = []
    for cid in ids:
        t, e, d = arrays(cid)
        L, kw, _ = port_est.solver_grid(t, e, d)
        gk = dict(N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"], sampling_dt=kw.get("sampling_dt"))
        vf, vg = V.make_loglik(t, e, L, **gk)
        of, og = ofj.make_loglik(t, e, L, **gk)
        ev = Evaluator(cohort(cid), numerics="v033", grid=GridPolicy("v033"), scheme="bdf2")
        z0 = jnp.zeros(4)
        ptxt = {"value": ev.program_text("value"), "value_and_grad": ev.program_text("value_and_grad")}
        prog = {"port_value": canonical_text(vf.lower(z0)) == ptxt["value"],
                "port_value_and_grad": canonical_text(vg.lower(z0)) == ptxt["value_and_grad"],
                "oracle_value": canonical_text(of.lower(z0)) == ptxt["value"],
                "oracle_value_and_grad": canonical_text(og.lower(z0)) == ptxt["value_and_grad"]}
        ef, evg = ev.objective()
        pts = []
        for label, th in theta_points(cid):
            z = np.log(np.array([th[p] for p in PARAMS], float))
            ev_v, ev_g = ef(z), evg(z)
            diffs = {"port_value": bytes_diff(ev_v, vf(z), "value"),
                     "port_value_and_grad": bytes_diff(ev_g, vg(z), "value_and_grad"),
                     "oracle_value": bytes_diff(ev_v, of(z), "value"),
                     "oracle_value_and_grad": bytes_diff(ev_g, og(z), "value_and_grad")}
            pts.append({"theta": label, "value": float(ev_v), "diffs": diffs})
        ok = all(prog.values()) and all(not any(p["diffs"].values()) for p in pts)
        rows.append({"cohort_id": cid, "grid": ev.grid.summary(), "program_equal": prog, "points": pts, "ok": ok})
        print(f"  {cid:44s} program {'=' if all(prog.values()) else 'DIFFERS'}  bytes "
              f"{sum(not any(p['diffs'].values()) for p in pts)}/{len(pts)}", flush=True)
    n_pts = sum(len(r["points"]) for r in rows)
    checks = {"program_equal": {"passed": sum(all(r["program_equal"].values()) for r in rows), "total": len(rows)},
              "bytes": {"passed": sum(not any(p["diffs"].values()) for r in rows for p in r["points"]), "total": n_pts}}
    status = "pass" if all(r["ok"] for r in rows) and not args.only else ("smoke" if args.only else "fail")
    return run.write({"status": status, "checks": checks, "results": rows})


# ------------------------------------------------------------------------------------------ guard-pilot
K_STIFF = 1e4
# The stiffness sweep (review T064 N2): a human truth at several N and dtau, and a second human cohort, so that
# c does not rest on one point. Both cohorts are outside the guard panel.
SWEEP_COHORT = "pure-real-Denmark_F_1900_hetro-c0-s0"
SWEEP_GRIDS = [(N, dt) for N in (450, 900, 1800) for dt in (1e-3, 2e-3, 4e-3)]
SWEEP_EXTRA = "realfull-real-Sweden_M_1910_hetro-c0-s0"


def _pilot_point(e33, e1, p, guard):
    c = e1.cohort
    died = np.asarray(c.events) == 1
    t33, a33 = e33.terms(p)
    t1, a1 = e1.terms(p)
    K = e1.stiffness(p)
    uf = died & np.asarray(a1["use_F"], bool)
    dm = np.abs(guard.masses(e1, t1, a1) - guard.masses(e33, t33, a33))[uf]
    mx = float(dm.max()) if dm.size else 0.0
    return {"K": K, "n_use_F": int(uf.sum()), "max_abs_dmass": mx, "ulps": mx / guard.U,
            "ratio_uK": mx / (guard.U * K)}


def phase_guard_pilot(args):
    """The guard's constants (research R14 as revised at review T064, N2): |m_v1 - m_v033| <= c u K + f u over
    the use_F death terms, K = rho_eps dtau N^2 the solve's diffusion number. Measured on cohorts outside the
    guard panel at their six theta points, plus a stiffness sweep: b = the worst |dm|/(u K) at the stiff points
    (K >= K_STIFF), c = margin x b; f = margin x the worst excess |dm|/u - b K over all points (the part of the
    error the slope does not explain)."""
    from engine_cases import cohort, theta_points
    from sraging import SRParams
    from sraging.engine import guard
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    run = Run("guard-pilot", [], args.dev, f"~10 CPU-min ({len(GUARD_PILOT)} cohorts x 6 theta x 2 numerics, plus "
              f"a {len(SWEEP_GRIDS) + 1}-point stiffness sweep)")
    preflight_or_exit(run)
    panel = set(load_toml(PANEL_GUARD)["cohorts"])
    assert not (set(GUARD_PILOT) | {SWEEP_COHORT, SWEEP_EXTRA}) & panel, "the pilot must be disjoint from the panel"
    rows = []
    for cid in GUARD_PILOT:
        c = cohort(cid)
        e33, e1 = Evaluator(c, numerics="v033"), Evaluator(c, numerics="v1")
        for label, th in theta_points(cid):
            r = {"cohort": cid, "theta": label, "grid": "v033", **_pilot_point(e33, e1, SRParams(**th), guard)}
            rows.append(r)
            print(f"  {cid:40s} {label:8s} K {r['K']:9.3g} use_F {r['n_use_F']:6d} max|dm| {r['max_abs_dmass']:.3e} "
                  f"= {r['ulps']:7.1f} u = {r['ratio_uK']:.3f} u K", flush=True)
    truth = dict(O.library().truth(SWEEP_COHORT))
    p = SRParams(**{k: truth[k] for k in PARAMS})
    c = cohort(SWEEP_COHORT)
    g0 = GridPolicy("v033").grid(c)
    for N, dt in SWEEP_GRIDS:
        gp = GridPolicy("fixed", N=N, dt=dt, tau_max=g0.tau_max, Gtol=g0.Gtol, L=g0.L)
        e33, e1 = Evaluator(c, numerics="v033", grid=gp), Evaluator(c, numerics="v1", grid=gp)
        r = {"cohort": SWEEP_COHORT, "theta": "truth", "grid": f"N={N} dt={dt:g}", **_pilot_point(e33, e1, p, guard)}
        rows.append(r)
        print(f"  sweep {r['grid']:18s} K {r['K']:9.3g} max|dm| {r['max_abs_dmass']:.3e} = {r['ratio_uK']:.3f} u K",
              flush=True)
    from engine_cases import arrays
    t, e, d = arrays(SWEEP_EXTRA)
    from sraging import Cohort
    cx = Cohort.from_arrays(t, e, delta=d)
    tx = dict(O.library().truth(SWEEP_EXTRA))
    r = {"cohort": SWEEP_EXTRA, "theta": "truth", "grid": "v033",
         **_pilot_point(Evaluator(cx, numerics="v033"), Evaluator(cx, numerics="v1"),
                        SRParams(**{k: tx[k] for k in PARAMS}), guard)}
    rows.append(r)
    print(f"  {SWEEP_EXTRA} truth K {r['K']:9.3g} max|dm| {r['max_abs_dmass']:.3e} = {r['ratio_uK']:.3f} u K", flush=True)
    ordinary = [r for r in rows if r["K"] < K_STIFF and r["n_use_F"]]
    stiff = [r for r in rows if r["K"] >= K_STIFF and r["n_use_F"]]
    b = max(r["ratio_uK"] for r in stiff)                        # the slope, from the stiff points
    for r in rows:
        r["excess_ulps"] = max(r["ulps"] - b * r["K"], 0.0)       # what the slope does not explain
    c_stiff = C_S_MARGIN * b
    f_ulps = C_S_MARGIN * max(r["excess_ulps"] for r in rows if r["n_use_F"])
    from sraging import divergences
    frozen = {e.id: {k: (e.condition or {}).get(k) for k in ("c_stiff", "floor_ulps")}
              for e in divergences.load() if e.id in ("D-004", "D-106")}
    print(f"[guard-pilot] stiff points {len(stiff)}: worst {b:.4f} u K -> c_stiff = {c_stiff:.4f}; all points: worst "
          f"excess over b u K {f_ulps / C_S_MARGIN:.1f} u -> floor_ulps = {f_ulps:.1f}; register: {frozen}", flush=True)
    return run.write({"status": "measured", "cohorts": GUARD_PILOT, "sweep": {"cohort": SWEEP_COHORT,
                      "grids": SWEEP_GRIDS, "extra": SWEEP_EXTRA}, "K_stiff": K_STIFF, "margin": C_S_MARGIN,
                      "floor_ulps": f_ulps, "c_stiff": c_stiff, "n_ordinary": len(ordinary), "n_stiff": len(stiff),
                      "register": frozen, "rows": rows})


# ------------------------------------------------------------------------------------------------ guard
def _record_plant(plants, which, rep):
    named = sorted({v[0] for v in rep.violations})
    plants[which] = {"failed": not rep.ok, "n_violations": len(rep.violations), "rows_named": named[:20],
                     "checks_named": sorted({v[1] for v in rep.violations}),
                     "grad_uncovered_rel": rep.grad_uncovered_rel}
    print(f"  planted {which:15s} -> {'FAILS (good)' if not rep.ok else 'PASSES (BAD)'}; "
          f"{len(rep.violations)} violations named ({plants[which]['checks_named']})", flush=True)


def phase_guard(args):
    """The CI guard on its frozen panel x the six theta points: v1 differs from v033 only where a register
    condition says so; then each planted unregistered change must fail it and name the terms (SC-008)."""
    from engine_cases import cohort, theta_points
    from sraging import SRParams, divergences
    from sraging.engine import guard
    from sraging.engine import observation as OBS
    from sraging.engine.evaluator import Evaluator

    panel = load_toml(PANEL_GUARD)
    run = Run("guard", [PANEL_GUARD], args.dev, f"~20 CPU-min ({len(panel['cohorts'])} cohorts x 6 theta x 2 "
              f"numerics, {len(panel.get('feature', []))} feature configurations x 3 theta, 4 planted changes)")
    preflight_or_exit(run)
    conds = guard.conditions([{"id": e.id, "condition": e.condition} for e in divergences.load()])
    rows = []
    for cid in panel["cohorts"]:
        c = cohort(cid)
        e33, e1 = Evaluator(c, numerics="v033"), Evaluator(c, numerics="v1")
        for label, th in theta_points(cid):
            rep = guard.compare(e33, e1, SRParams(**th), conds, f"{cid}:{label}")
            rows.append({"cohort": cid, "theta": label, "ok": rep.ok, "n_terms": rep.n_terms, "covered": rep.covered,
                         "violations": rep.violations[:20], "n_violations": len(rep.violations),
                         "grad_uncovered_rel": rep.grad_uncovered_rel, "stiffness": rep.stiffness,
                         "grad_covered_rel": rep.grad_covered_rel, "rescued": rep.rescued,
                         "rescued_cd_ratio": rep.rescued_cd_ratio, "use_F_disagree": rep.use_F_disagree})
            print(f"  {cid:40s} {label:8s} {'ok' if rep.ok else 'FAIL'}  covered {rep.covered}  "
                  f"grad_rel {rep.grad_uncovered_rel:.1e}", flush=True)

    # feature configurations (review T064 N1, M4)
    import acceptance_002_t1 as T1
    t1_panel = load_toml(T1.PANEL_T1)
    for f in panel.get("feature", []):
        coh, kw, truth, h = T1.guard_feature(f, t1_panel)
        e33, e1 = Evaluator(coh, numerics="v033", **kw), Evaluator(coh, numerics="v1", **kw)
        for label in panel["feature_theta"]:
            th = T1._scale(truth, label)
            rep = guard.compare(e33, e1, SRParams(**th, h_ext=h), conds, f"{f['kind']}:{label}")
            rows.append({"cohort": f.get("cohort"), "feature": f["kind"], "theta": label, "ok": rep.ok,
                         "n_terms": rep.n_terms, "covered": rep.covered, "violations": rep.violations[:20],
                         "n_violations": len(rep.violations), "grad_uncovered_rel": rep.grad_uncovered_rel,
                         "stiffness": rep.stiffness, "grad_covered_rel": rep.grad_covered_rel,
                         "rescued": rep.rescued, "rescued_cd_ratio": rep.rescued_cd_ratio,
                         "use_F_disagree": rep.use_F_disagree})
            print(f"  feature {f['kind']:8s} {label:8s} {'ok' if rep.ok else 'FAIL'}  covered {rep.covered}  "
                  f"rescued {rep.rescued}  {rep.violations[:2]}", flush=True)

    # planted unregistered changes (constitution V): each must fail, naming the terms
    real = OBS.Assembly.assemble
    plants = {}
    # each plant on a panel cohort where it changes something (a censored plant needs censored rows)
    with_censored = next(x for x in panel["cohorts"] if np.any(np.asarray(cohort(x).events) == 0))
    for which in ("censored", "hazard", "use_F"):
        cid = with_censored if which == "censored" else panel["cohorts"][0]
        label, th = theta_points(cid)[0]
        def assemble(self, pts, vals=None, _w=which):
            t, aux = real(self, pts, vals)
            if self.numerics == "v1":
                died = np.asarray(self.died)
                if _w == "censored":
                    t = t + np.where(died, 0.0, 1e-12)
                elif _w == "hazard":
                    t = t * (1.0 + 1e-12)
                else:
                    t = t + np.where(died, 1e-3, 0.0)
            return t, aux
        OBS.Assembly.assemble = assemble
        try:
            c = cohort(cid)
            rep = guard.compare(Evaluator(c, numerics="v033"), Evaluator(c, numerics="v1"), SRParams(**th), conds,
                                f"{cid}:{label}:planted-{which}")
        finally:
            OBS.Assembly.assemble = real
        _record_plant(plants, which, rep)
    # a planted stop_gradient on the F route: the values are unchanged, the rescued terms' gradient is not (N3).
    # On the stiff human truth of the panel, where v1 rescues terms.
    real_pts = OBS.Assembly.points

    def points(self, march_out, diagnostics=False):
        import jax
        pts = real_pts(self, march_out, diagnostics)
        if self.numerics == "v1" and "F_hi" in pts:
            pts = {**pts, "F_hi": jax.lax.stop_gradient(pts["F_hi"]), "F_st": jax.lax.stop_gradient(pts["F_st"])}
        return pts
    sg_cid = next(x for x in panel["cohorts"] if "real" in x)
    sg_label, sg_th = theta_points(sg_cid)[0]
    OBS.Assembly.points = points
    try:
        c = cohort(sg_cid)
        rep = guard.compare(Evaluator(c, numerics="v033"), Evaluator(c, numerics="v1"), SRParams(**sg_th), conds,
                            f"{sg_cid}:{sg_label}:planted-stop_gradient")
    finally:
        OBS.Assembly.points = real_pts
    _record_plant(plants, "stop_gradient_F", rep)
    ok = all(r["ok"] for r in rows) and all(p["failed"] and p["n_violations"] > 0 for p in plants.values())
    return run.write({"status": "pass" if ok else "fail", "conditions": [eid for eid, _ in conds],
                      "checks": {"guard": {"passed": sum(r["ok"] for r in rows), "total": len(rows)},
                                 "planted_fail": {"passed": sum(p["failed"] for p in plants.values()),
                                                  "total": len(plants)}},
                      "results": rows, "planted": plants})


# ----------------------------------------------------------------------------------------------- main
def _phases():
    ph = {"t0": phase_t0, "guard-pilot": phase_guard_pilot, "guard": phase_guard}
    try:
        import acceptance_002_t1 as T1
        ph.update({"t1-pilot": T1.phase_t1_pilot, "t1": T1.phase_t1, "mpref": T1.phase_mpref})
    except ImportError:
        pass
    try:
        import acceptance_002_ref as R
        ph.update({"early": R.phase_early, "anchor": R.phase_anchor})
    except ImportError:
        pass
    try:
        import acceptance_002_speed as SP
        ph["speed"] = SP.phase_speed
    except ImportError:
        pass
    try:
        import score_test as SC
        ph["score"] = SC.phase_score
    except ImportError:
        pass
    try:
        import acceptance_002_floors as FL
        ph.update({"floors-expect": FL.phase_floors_expect, "floors": FL.phase_floors})
    except ImportError:
        pass
    return ph


def main(argv=None):
    ph = _phases()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=sorted(ph))
    ap.add_argument("--dev", action="store_true", help="run on a dirty tree / uncommitted manifest; never counts")
    ap.add_argument("--only", type=lambda s: [x for x in s.split(",") if x], default=None,
                    help="restrict to these cohort ids (a smoke run; never counts)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--features", type=lambda s: [x for x in s.split(",") if x], default=None,
                    help="score only: run these features (an acceptance run of a subset; the result names them)")
    ap.add_argument("--cut", default=None, help="t1 only, development runs only: provisional low-mass constants 'c,f'")
    args = ap.parse_args(argv)
    if args.only or args.cut is not None:
        args.dev = True
    ph[args.phase](args)


if __name__ == "__main__":
    main()
