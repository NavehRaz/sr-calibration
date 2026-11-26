"""IDENTITY HARNESS: mle_profile_jax2 with every switch off  vs  the frozen rc1 candidate + its cluster driver.

The pass condition (PHASE_B_DESIGN.txt L3, and B1): the WHOLE fit() output is byte-identical except timings,
counters included. Each side runs in its OWN process (no shared JIT cache, no shared module state):

  rc1  the fixture's driver (tests/fixtures/candidate_rc1/cluster/driver.py) is imported for its helpers and
       its environment; the rc1 adapter loads exactly as driver.main() loads it, and the fit is the
       driver's call, `ad.run_fit(obs, tie_relabel=...)` (impl `cand` = relabel off, `cand_tie` = on);
  new  this package's adapter, loaded under the key discovery uses, and its `run_fit(obs, relabel=...)`
       with hazard and window OFF.

Both sides wrap the SAME driver instrumentation (`_node_capture` -> every final profile node with its
nuisance z; the `_basin_refit` progress hook -> every re-fit search's in/out), and record the sha256 of the
data actually passed to `estimator.fit`. What is compared, as canonical JSON (sort_keys; floats by repr, so
byte-identity is bit-identity):

    result         estimator.fit's whole dict: theta, ci, loglik, dt_ceiling, diagnostics (verdicts,
                   profiles, grid, solver, basin_fix incl. every round, dt_floor, censor_tie, n_eval,
                   n_grad and every counter)
    claims         the claim mapping's output
    profile_nodes  the driver's node capture (value, loglik and full z of every final profile node)
    refit_calls    the driver's re-fit hook (ll in/out, info, theta)
    fit_inputs     sha256 of the ages / events given to estimator.fit, n, deaths, delta

minus TIMING_KEYS wherever they occur, and minus the three diagnostics keys only the new package writes
(NEW_ONLY_DIAG), each checked first: exc_counts == {}, observation_model says relabel as requested and
invariants ok, srk_sha has no file changed during the fit.

usage (PYTHONPATH = SRtools_kappa:<this checkout>; SRB_STRICT=1 is forced):
    python identity_rc1.py run {new|rc1} CID OUTDIR [--relabel on|off]
    python identity_rc1.py batch OUTDIR CID [CID ...] [--relabel on|off] [--jobs 2]
    python identity_rc1.py compare OUTDIR CID [CID ...] [--relabel on|off]
"""
from __future__ import annotations

import os
import sys

# The method's environment, before anything imports JAX (the rc1 driver sets the same defaults).
os.environ.setdefault("PJRT_NPROC", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("SRK_CACHE_READONLY", "1")
os.environ["SRB_STRICT"] = "1"

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

TESTS = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(TESTS)
ROOT = os.path.dirname(os.path.dirname(PKG))
FIXTURE = os.path.join(TESTS, "fixtures", "candidate_rc1")
DRIVER = os.path.join(FIXTURE, "cluster", "driver.py")
PARAMS4 = ("eta", "beta", "epsilon", "kappa")

TIMING_KEYS = frozenset({"wall_s", "compile_s", "seed_s", "mle_s", "cpu_s", "refit_cpu_s", "fit_wall_s"})
NEW_ONLY_DIAG = ("exc_counts", "observation_model", "srk_sha")
COMPARED = ("result", "claims", "profile_nodes", "refit_calls", "fit_inputs")


def _sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _load(key, path):
    spec = importlib.util.spec_from_file_location(key, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


def _fname(cid, impl, relabel):
    return f"{cid}__{impl}__relabel-{'on' if relabel else 'off'}.json"


def run_one(impl, cid, outdir, relabel):
    if "SRK_DT_CEILING" in os.environ:
        raise SystemExit("REFUSING: SRK_DT_CEILING is set")
    os.makedirs(outdir, exist_ok=True)
    drv = _load("jax2_identity.rc1_driver", DRIVER)          # its module top: env defaults + numpy shim
    import numpy as np
    from srbench.library.load import Library
    from srbench.provenance import resolve_sut
    prov = resolve_sut()
    lib = Library(version="v2")
    obs = lib.observation(cid)
    v1 = Library(version="v1")
    v1_same = None
    if any(s.cohort_id == cid for s in v1.specs):
        o1 = v1.observation(cid)
        v1_same = (o1.ages, o1.events, o1.delta, o1.window) == (obs.ages, obs.events, obs.delta, obs.window)
        if not v1_same:
            raise SystemExit(f"{cid}: library v1 and v2 observations differ")

    if impl == "rc1":
        ad = drv._load("srbench_candidate.mle_profile_jax2_rc1.adapter", os.path.join(drv.CODE, "adapter.py"))
    elif impl == "new":
        ad = _load("srbench_methods.mle_profile_jax2", os.path.join(PKG, "__init__.py"))
    else:
        raise SystemExit(f"impl must be new or rc1, not {impl!r}")
    est = ad._sibling("estimator")
    fj = ad._sibling("fp_jax")

    latest = drv._node_capture(est)                     # the driver's own instrumentation, both sides
    calls = []
    _refit = est._basin_refit

    def _refit_logged(obj, pts_all, ll, bounds4, nm_opts, tol):
        v, z, info = _refit(obj, pts_all, ll, bounds4, nm_opts, tol)
        calls.append(drv._clean(dict(ll_in=ll, ll_out=-v, info=info, theta=None if z is None else np.exp(z))))
        return v, z, info
    est._basin_refit = _refit_logged

    inputs = {}
    _fit = est.fit

    def _fit_logged(fj_, ages, events, **kw):
        a = np.asarray(ages, float)
        e = np.asarray(events, int)
        inputs.update(ages_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                      events_sha256=hashlib.sha256(e.astype(np.int64).tobytes()).hexdigest(),
                      n=int(a.size), n_deaths=int(e.sum()), delta=kw.get("delta"),
                      levels=list(kw.get("levels", ())), kappa_max=kw.get("kappa_max"),
                      types=[type(ages).__name__, type(events).__name__])
        extra = {k: v for k, v in kw.items() if k not in ("delta", "levels", "kappa_max")}
        if extra:
            inputs["extra_kwargs"] = {k: repr(v) for k, v in extra.items()}
        return _fit(fj_, ages, events, **kw)
    est.fit = _fit_logged

    c0, w0 = time.process_time(), time.perf_counter()
    if impl == "rc1":
        res, _tie = ad.run_fit(obs, tie_relabel=relabel)
    else:
        res = ad.run_fit(obs, relabel=relabel)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    verdicts = (res.get("diagnostics") or {}).get("verdicts") or {}
    mods = (ad, est, fj) + ((ad._sibling("observation_model"),) if impl == "new" else ())
    code = {os.path.relpath(m.__file__, ROOT): _sha256_file(m.__file__) for m in mods}
    code[os.path.relpath(DRIVER, ROOT)] = _sha256_file(DRIVER)
    import jax
    import scipy
    srk_dir, srk_md5 = drv._srk_md5()
    out = dict(cid=cid, impl=impl, relabel=bool(relabel),
               result=drv._clean(res),
               claims=drv._clean(ad.claim_kwargs(res["theta"], res["ci"], verdicts)),
               profile_nodes=drv._nodes_out(latest, PARAMS4),
               refit_calls=calls, fit_inputs=inputs,
               meta=dict(host=socket.gethostname().split(".")[0], cpu_model=drv._cpu_model(), cpu_s=cpu,
                         wall_s=wall, peak_rss_mb=drv._peak_mb(), code_sha256=code, srk=srk_dir,
                         srk_md5=srk_md5, provenance=prov.to_dict(), v1_obs_identical=v1_same,
                         library="v2", modules=[ad.__name__, est.__name__, fj.__name__],
                         env={k: os.environ.get(k) for k in ("PJRT_NPROC", "XLA_FLAGS", "SRB_STRICT",
                                                              "SRK_CACHE_READONLY", "SRK_DT_CEILING")},
                         versions=dict(numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__,
                                       python=sys.version.split()[0])))
    drv._write(outdir, _fname(cid, impl, relabel), out)
    print(json.dumps(dict(cid=cid, impl=impl, relabel=relabel, cpu_s=round(cpu, 1), loglik=out["result"]["loglik"],
                          n_grad=out["result"]["diagnostics"].get("n_grad"),
                          fired=out["result"]["diagnostics"].get("basin_fix", {}).get("fired"))))


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in TIMING_KEYS}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def _first_diff(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: only in {'rc1' if k in a else 'new'}"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if _canon(a) == _canon(b) else f"{path}: {a!r} vs {b!r}"


def compare_one(outdir, cid, relabel):
    pa = os.path.join(outdir, _fname(cid, "rc1", relabel))
    pb = os.path.join(outdir, _fname(cid, "new", relabel))
    a, b = json.load(open(pa)), json.load(open(pb))
    rep = dict(cid=cid, relabel=relabel, checks={}, sections={})
    dg = b["result"].get("diagnostics", {})
    om = dg.get("observation_model") or {}
    srk = dg.get("srk_sha") or {}
    rep["checks"] = dict(
        exc_counts_empty=(dg.get("exc_counts") == {}),
        relabel_as_requested=(om.get("relabel_on") is bool(relabel) and om.get("window_on") is False),
        invariants_ok=(om.get("invariants") == "ok"),
        srk_unchanged_during_fit=(srk.get("changed_during_fit") == [] and bool(srk.get("digest"))),
        no_extra_fit_kwargs_rc1=("extra_kwargs" not in a["fit_inputs"]),
        new_extra_fit_kwargs=b["fit_inputs"].get("extra_kwargs"),
        same_srk_md5=(a["meta"]["srk_md5"] == b["meta"]["srk_md5"]),
        same_host=(a["meta"]["host"] == b["meta"]["host"]),
        rc1_has_no_new_only_keys=not any(k in a["result"].get("diagnostics", {}) for k in NEW_ONLY_DIAG),
    )
    # the new side passes hazard=False, window=None explicitly: that is the switch-off path, not a difference
    nek = b["fit_inputs"].pop("extra_kwargs", None)
    rep["checks"]["new_extra_fit_kwargs_are_the_off_switches"] = (nek in (None, {"hazard": "False", "window": "None"}))
    for k in NEW_ONLY_DIAG:
        dg.pop(k, None)
    ok = all(v for k, v in rep["checks"].items() if k != "new_extra_fit_kwargs")
    for sec in COMPARED:
        sa, sb = _canon(_strip(a[sec])), _canon(_strip(b[sec]))
        same = sa == sb
        rep["sections"][sec] = dict(identical=same, bytes=len(sa), sha256_rc1=hashlib.sha256(sa.encode()).hexdigest(),
                                    sha256_new=hashlib.sha256(sb.encode()).hexdigest(),
                                    first_diff=None if same else _first_diff(_strip(a[sec]), _strip(b[sec])))
        ok = ok and same
    rep["identical"] = ok
    rep["cpu_s"] = dict(rc1=a["meta"]["cpu_s"], new=b["meta"]["cpu_s"])
    rep["n_grad"] = a["result"]["diagnostics"].get("n_grad")
    rep["fired"] = a["result"]["diagnostics"].get("basin_fix", {}).get("fired")
    return rep


def compare(outdir, cids, relabel):
    reps = [compare_one(outdir, c, relabel) for c in cids]
    tag = f"relabel-{'on' if relabel else 'off'}"
    with open(os.path.join(outdir, f"COMPARE_{tag}.json"), "w") as fh:
        json.dump(reps, fh, indent=1, sort_keys=True)
    lines = [f"identity_rc1 compare ({tag}), {time.strftime('%Y-%m-%d %H:%M:%S')}, outdir {outdir}",
             f"{'cohort':<28} {'identical':<10} {'bytes(result)':>13} {'n_grad':>7} {'fired':>6} "
             f"{'cpu rc1':>8} {'cpu new':>8}  first difference"]
    for r in reps:
        fd = next((s["first_diff"] for s in r["sections"].values() if s["first_diff"]), "")
        bad = [k for k, v in r["checks"].items() if k != "new_extra_fit_kwargs" and not v]
        lines.append(f"{r['cid']:<28} {str(r['identical']):<10} {r['sections']['result']['bytes']:>13} "
                     f"{r['n_grad']!s:>7} {r['fired']!s:>6} {r['cpu_s']['rc1']:>8.1f} {r['cpu_s']['new']:>8.1f}  "
                     f"{fd}{(' FAILED CHECKS ' + str(bad)) if bad else ''}")
    n_ok = sum(r["identical"] for r in reps)
    lines.append(f"IDENTICAL {n_ok}/{len(reps)}")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(outdir, f"COMPARE_{tag}.txt"), "w").write(txt)
    print(txt)
    return n_ok == len(reps)


def batch(outdir, cids, relabel, jobs):
    os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)
    todo = [(c, i) for c in cids for i in ("rc1", "new")
            if not os.path.exists(os.path.join(outdir, _fname(c, i, relabel)))]
    running = []
    failed = []
    while todo or running:
        while todo and len(running) < jobs:
            c, i = todo.pop(0)
            log = open(os.path.join(outdir, "logs", f"{c}__{i}__relabel-{'on' if relabel else 'off'}.log"), "w")
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "run", i, c, outdir,
                                  "--relabel", "on" if relabel else "off"], stdout=log, stderr=subprocess.STDOUT)
            running.append((p, c, i, log, time.time()))
            print(f"started {c} {i} (pid {p.pid})", flush=True)
        time.sleep(2.0)
        for item in list(running):
            p, c, i, log, t0 = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print(f"finished {c} {i} rc={p.returncode} wall={time.time() - t0:.0f}s", flush=True)
                if p.returncode != 0:
                    failed.append((c, i))
    if failed:
        print(f"FAILED runs: {failed}")
        return False
    return compare(outdir, cids, relabel)


def main(argv):
    if not argv or argv[0] not in ("run", "batch", "compare"):
        raise SystemExit(__doc__)
    relabel = True
    if "--relabel" in argv:
        j = argv.index("--relabel")
        relabel = {"on": True, "off": False}[argv[j + 1]]
        argv = argv[:j] + argv[j + 2:]
    jobs = 2
    if "--jobs" in argv:
        j = argv.index("--jobs")
        jobs = int(argv[j + 1])
        argv = argv[:j] + argv[j + 2:]
    if argv[0] == "run":
        _, impl, cid, outdir = argv
        run_one(impl, cid, os.path.abspath(outdir), relabel)
    elif argv[0] == "batch":
        ok = batch(os.path.abspath(argv[1]), argv[2:], relabel, jobs)
        raise SystemExit(0 if ok else 1)
    else:
        ok = compare(os.path.abspath(argv[1]), argv[2:], relabel)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
