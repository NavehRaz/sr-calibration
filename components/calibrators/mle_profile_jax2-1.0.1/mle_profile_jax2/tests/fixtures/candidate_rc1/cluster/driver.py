"""Run ONE (cohort, implementation) of the jax2 rc1 Phase-A test and write one JSON.

usage: python driver.py COHORT_ID IMPL OUTDIR

IMPL
  cand          mle_profile_jax2 1.0.0rc1 (code/adapter.py -> code/estimator.py + code/fp_jax.py), relabel OFF
  cand_tie      the same with the censoring-tie relabel ON
  frozen        mle_profile_jax v0.3.3 exactly (code/frozen/{estimator,fp_jax}.py = git show 8b4c2f1)
  probe_cand    no fit: one value and one value_and_grad on the candidate's grid at the seed (floor-bound
                cohorts: N=1800, dt=3e-4), with compile/eval times and peak RSS -- the memory/speed probe
  probe_frozen  the same on the production grid with the frozen likelihood

Layout: <root>/code (the candidate + code/frozen) and <root>/cluster (this file). Reads the library (v1)
and the SUT (SRK) read-only; writes ONLY OUTDIR: <cid>__<impl>.json on completion (atomic rename), a
<cid>__<impl>.partial.json after each re-fit search (progress only), <cid>__<impl>.error.json on failure.

Every JSON carries: claims (adapter mapping), theta, loglik, ci, verdicts, grid, the counters, solver,
dt_floor, censor_tie, basin_fix (with stock_claims), profile nodes with their nuisance z (for the
likelihood-level checks), host, CPU model, cpu_s, wall_s, peak RSS, md5 of every code file and of the
SRK modules, versions, and the claim set's contract validation.
"""
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PJRT_NPROC", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("SRK_CACHE_READONLY", "1")
os.environ["SRB_STRICT"] = "1"
import numpy as np  # noqa: E402
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = os.path.join(ROOT, "code")
IMPLS = ("cand", "cand_tie", "frozen", "probe_cand", "probe_frozen")
SRK_FILES = ("__init__.py", "fp_likelihood.py", "fp_gof.py", "fp_curvefit.py", "fp_model.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _md5(path):
    try:
        return hashlib.md5(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _peak_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform.startswith("linux") else ru / 1048576.0


def _cpu_model():
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def _write(outdir, name, obj):
    tmp = os.path.join(outdir, f".{name}.tmp")
    with open(tmp, "w") as fh:
        json.dump(_clean(obj), fh, indent=1, default=str)
    os.replace(tmp, os.path.join(outdir, name))


def _code_md5():
    files = {"adapter": os.path.join(CODE, "adapter.py"), "estimator": os.path.join(CODE, "estimator.py"),
             "fp_jax": os.path.join(CODE, "fp_jax.py"),
             "frozen_estimator": os.path.join(CODE, "frozen", "estimator.py"),
             "frozen_fp_jax": os.path.join(CODE, "frozen", "fp_jax.py"),
             "frozen_init": os.path.join(CODE, "frozen", "__init__.py"),
             "driver": os.path.abspath(__file__)}
    return {k: _md5(v) for k, v in files.items()}


def _srk_md5():
    import SRK
    d = os.path.dirname(SRK.__file__)
    return d, {f: _md5(os.path.join(d, f)) for f in SRK_FILES}


def _node_capture(est):
    """Wrap est._repair (called once or twice per parameter in EVERY profile stage, on that stage's node
    list) to keep a reference to each parameter's latest node list. Inputs and outputs unchanged. The lists
    are the ones the stage reads its rows from, so after the fit they hold the FINAL profiles' nodes."""
    latest = {}
    _rep = est._repair

    def _repair_logged(obj, theta, i, fidx, pts):
        latest[int(i)] = (list(fidx), pts)
        return _rep(obj, theta, i, fidx, pts)
    est._repair = _repair_logged
    return latest


def _nodes_out(latest, params):
    out = {}
    for i, (fidx, pts) in sorted(latest.items()):
        rows = []
        for v, l_, zz in pts:
            z = None
            if zz is not None:
                z = [0.0] * 4
                z[i] = math.log(float(v))
                for k, zk in zip(fidx, list(zz)):
                    z[k] = float(zk)
            rows.append(dict(value=float(v), loglik=(None if l_ is None else float(l_)), z=z))
        out[params[i]] = rows
    return out


def probe(cid, impl, outdir, lib):
    """One value and one value_and_grad at the seed, on the impl's grid; times and peak RSS."""
    import jax
    obs = lib.observation(cid)
    t = np.asarray(obs.ages, float)
    e = np.asarray(obs.events, int)
    d = obs.delta if obs.delta else None
    if impl == "probe_cand":
        est = _load("cand_probe_estimator", os.path.join(CODE, "estimator.py"))
        fj = _load("cand_probe_fp_jax", os.path.join(CODE, "fp_jax.py"))
        L, kw, ceiling, floor = est._grid(t, e, d)
    else:
        est = _load("frozen_probe_estimator", os.path.join(CODE, "frozen", "estimator.py"))
        fj = _load("frozen_probe_fp_jax", os.path.join(CODE, "frozen", "fp_jax.py"))
        L, kw, ceiling = est.solver_grid(t, e, d)
        floor = None
    from SRK.fp_curvefit import fp_curvefit_seed
    c0, w0 = time.process_time(), time.perf_counter()
    f, vg = fj.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                           sampling_dt=kw.get("sampling_dt"))
    s = fp_curvefit_seed(t, e, xc=1.0)
    z = np.log(np.clip([s["eta"], s["beta"], s["epsilon"], s["k"]], 1e-12, None))
    rss0 = _peak_mb()
    tc = time.perf_counter()
    v_f = float(f(z).block_until_ready())
    t_f_first = time.perf_counter() - tc
    tc = time.perf_counter()
    v_f2 = float(f(z + 1e-3).block_until_ready())
    t_f = time.perf_counter() - tc
    rss_f = _peak_mb()
    tc = time.perf_counter()
    v, g = vg(z)
    g = np.asarray(g)
    t_vg_first = time.perf_counter() - tc
    tc = time.perf_counter()
    v2, g2 = vg(z + 1e-3)
    np.asarray(g2)
    t_vg = time.perf_counter() - tc
    nsteps = int(kw["tau_max"] / kw["dt"])
    out = dict(cid=cid, impl=impl, host=socket.gethostname().split(".")[0], cpu_model=_cpu_model(),
               n=int(t.size), delta=d, L=L, solver=kw, dt_ceiling=ceiling, dt_floor=floor, nsteps=nsteps,
               q_bytes=(nsteps + 1) * kw["N"] * 8, checkpointed=bool((nsteps + 1) * kw["N"] * 8 > fj.Q_BUDGET_BYTES),
               z=list(z), value=v_f, value_vg=float(v), grad=list(g), value_shift=v_f2, value_vg_shift=float(v2),
               t_value_first_s=t_f_first, t_value_s=t_f, t_vg_first_s=t_vg_first, t_vg_s=t_vg,
               peak_rss_mb_before_eval=rss0, peak_rss_mb_after_value=rss_f, peak_rss_mb=_peak_mb(),
               cpu_s=time.process_time() - c0, wall_s=time.perf_counter() - w0,
               md5=_code_md5(), versions=dict(numpy=np.__version__, jax=jax.__version__,
                                               python=sys.version.split()[0]))
    _write(outdir, f"{cid}__{impl}.json", out)
    print(json.dumps(dict(cid=cid, impl=impl, N=kw["N"], dt=kw["dt"], nsteps=nsteps,
                          checkpointed=out["checkpointed"], t_value_s=round(t_f, 3), t_vg_s=round(t_vg, 3),
                          peak_rss_mb=round(out["peak_rss_mb"]))))


def main():
    cid, impl, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    if impl not in IMPLS:
        raise SystemExit(f"impl must be one of {IMPLS}")
    if os.environ.get("SRK_DT_CEILING"):
        raise SystemExit("REFUSING: SRK_DT_CEILING is set; it changes SRK.auto_grid under every grid rule")
    os.makedirs(outdir, exist_ok=True)
    import jax
    import scipy
    import SRK
    from srbench.library.load import Library
    lib = Library(version="v1")
    if impl.startswith("probe_"):
        return probe(cid, impl, outdir, lib)
    obs = lib.observation(cid)
    truth = lib.truth(cid)
    ad = _load("srbench_candidate.mle_profile_jax2_rc1.adapter", os.path.join(CODE, "adapter.py"))
    if impl in ("cand", "cand_tie"):
        est = ad._sibling("estimator")
        fj = ad._sibling("fp_jax")
        est_path, fj_path = est.__file__, fj.__file__
    else:
        est_path = os.path.join(CODE, "frozen", "estimator.py")
        fj_path = os.path.join(CODE, "frozen", "fp_jax.py")
        est = _load("srbench_candidate.frozen_8b4c2f1.estimator", est_path)
        fj = _load("srbench_candidate.frozen_8b4c2f1.fp_jax", fj_path)
    latest = _node_capture(est)
    calls = []
    if hasattr(est, "_basin_refit"):
        # PROGRESS ONLY (a walltime kill loses the final JSON): after each re-fit search, dump what it found.
        _refit = est._basin_refit

        def _refit_logged(obj, pts_all, ll, bounds4, nm_opts, tol):
            v, z, info = _refit(obj, pts_all, ll, bounds4, nm_opts, tol)
            calls.append(_clean(dict(ll_in=ll, ll_out=-v, info=info, cpu_s=time.process_time(),
                                     theta=None if z is None else np.exp(z))))
            _write(outdir, f"{cid}__{impl}.partial.json", calls)
            return v, z, info
        est._basin_refit = _refit_logged

    from srbench.contract.validate import validate_claimset
    from srbench.provenance import resolve_sut
    prov = resolve_sut()
    tie_on = impl == "cand_tie"
    c0, w0 = time.process_time(), time.perf_counter()
    try:
        if impl in ("cand", "cand_tie"):
            res, tie = ad.run_fit(obs, tie_relabel=tie_on)
        else:
            res = est.fit(fj, list(obs.ages), list(obs.events), delta=obs.delta, levels=(ad.LEVEL,),
                          kappa_max=ad.KAPPA_MAX)
            tie = ad.censor_tie_relabel(obs.ages, obs.events, obs.delta, enabled=False)[2]
    except Exception as ex:  # noqa: BLE001
        _write(outdir, f"{cid}__{impl}.error.json",
               dict(cid=cid, impl=impl, host=socket.gethostname().split(".")[0], error=repr(ex),
                    traceback=traceback.format_exc(), cpu_s=time.process_time() - c0,
                    peak_rss_mb=_peak_mb(), md5=_code_md5()))
        raise
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    peak_mb = _peak_mb()
    dg = dict(res.get("diagnostics") or {})
    profiles = dg.pop("profiles", None)
    bf = dg.pop("basin_fix", None)
    dt_floor = dg.pop("dt_floor", None)
    dg.pop("censor_tie", None)
    if impl == "frozen":
        dt_floor = dict(production_grid=True, candidate_floor=_load(
            "srbench_candidate.floorinfo.estimator", os.path.join(CODE, "estimator.py")).floor_info(
                obs.ages, obs.events, obs.delta))
    verdicts = dg.get("verdicts") or {}
    claims = ad.claim_kwargs(res["theta"], res["ci"], verdicts)
    # the adapter's own ClaimSet path, validated against the contract (cheap; no second fit)
    try:
        cs = ad.MleProfileJax2(tie_relabel=tie_on).claimset(obs, res, prov, w0, c0)
        validate_claimset(cs, ad.MleProfileJax2.descriptor, obs)
        contract = "ok"
    except Exception as ex:  # noqa: BLE001
        contract = f"FAILED: {ex!r}"
    srk_dir, srk_md5 = _srk_md5()
    out = dict(cid=cid, impl=impl, host=socket.gethostname().split(".")[0], cpu_model=_cpu_model(),
               cpu_s=cpu, wall_s=wall, fit_wall_s=res.get("wall_s"), peak_rss_mb=peak_mb,
               loglik=res.get("loglik"), theta=[float(x) for x in res["theta"]], claims=claims,
               ci={k: [float(x) for x in v] for k, v in res["ci"].items()}, verdicts=verdicts,
               diagnostics=dg, dt_ceiling=res.get("dt_ceiling"), dt_floor=dt_floor, censor_tie=tie,
               settings=(ad.MleProfileJax2(tie_relabel=tie_on).settings() if impl != "frozen" else None),
               profiles=profiles, profile_nodes=_nodes_out(latest, ("eta", "beta", "epsilon", "kappa")),
               contract=contract, truth=truth,
               observation=dict(n=len(obs.ages), n_censored=int(sum(1 for x in obs.events if int(x) == 0)),
                                delta=obs.delta, window=obs.window),
               md5=_code_md5(), srk=srk_dir, srk_md5=srk_md5, srk_dt_ceiling_env=os.environ.get("SRK_DT_CEILING"),
               code_provenance=prov.to_dict(),
               env={k: os.environ.get(k) for k in ("PJRT_NPROC", "XLA_FLAGS", "OMP_NUM_THREADS",
                                                    "SRK_CACHE_READONLY", "SRB_STRICT", "LSB_JOBID",
                                                    "LSB_JOBINDEX")},
               versions=dict(numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__,
                             python=sys.version.split()[0]))
    if impl == "frozen":
        out["basin_fix"] = None
    else:
        bf = dict(bf or {})
        st = bf.get("stock")
        if st is not None:
            bf["stock_claims"] = ad.claim_kwargs(st["theta"], st["ci"], st["verdicts"])
            bf["stock_is_final"] = False
        else:
            bf["stock_claims"] = claims          # nothing fired: the stock pass IS the final fit
            bf["stock_is_final"] = True
        out["basin_fix"] = bf
    _write(outdir, f"{cid}__{impl}.json", out)
    print(json.dumps(dict(cid=cid, impl=impl, host=out["host"], cpu_s=round(cpu, 1), peak_rss_mb=round(peak_mb),
                          loglik=out["loglik"], fired=(out["basin_fix"] or {}).get("fired"),
                          n_refit=(out["basin_fix"] or {}).get("n_refit"), floor=(dt_floor or {}).get("binds"),
                          tie=dict(detected=tie.get("detected"), applied=tie.get("applied"),
                                   n_relabelled=tie.get("n_relabelled"), dose=tie.get("dose")),
                          contract=contract, shapes={c["parameter"]: c["shape"] for c in claims})))


if __name__ == "__main__":
    main()
