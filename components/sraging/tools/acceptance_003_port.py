"""Spec 003 port parity phases: `bytes` (SC-001) and `fits` (SC-002) (tasks T018; research R4, R5).

Port and oracle run in the same spawn worker, per cohort. Workers set the single-thread XLA and BLAS flags the
tester's stored Mac fits ran with (reports/rc2/C/freeze_1.0.0/scripts/f1_fit.py) before JAX is imported, so the
port's outputs can also be compared with those stored fits.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREAD_ENV = {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
              "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
# run_fit result keys that are timing or tester-deployment provenance, never compared (FREEZE_NOTE.txt 3.2)
RES_SKIP = ("wall_s",)
DIAG_SKIP = ("compile_s", "seed_s", "mle_s", "srk_sha")


def _init():
    for k, v in THREAD_ENV.items():
        os.environ[k] = v
    sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
    for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import sraging.compat.jax2  # noqa: F401  (configures the runtime before the oracle's fp_jax)
    import oracle_jax2 as OJ
    OJ.load()


def _comparable(res):
    """A run_fit result without the tester's timing leaves (any depth) and without srk_sha (tester provenance)."""
    from sraging.calibrators.profile_jax2 import strip_timing
    from sraging.result import plain
    return strip_timing(plain(res), skip=("srk_sha",))


def port_observation(obs):
    """The oracle's observation as the port's `Observation`, field for field and window included: both sides fit the
    same data (a Cohort cannot carry a window, and jax2's `prepare` records the window even when windows are off)."""
    from sraging.calibrators import profile_jax2 as PJ
    return PJ.Observation(cohort_id=obs.cohort_id, ages=tuple(obs.ages), events=tuple(obs.events),
                          delta=obs.delta, window=None if obs.window is None else tuple(obs.window))


# ------------------------------------------------------------------------------------------------- bytes
def bytes_worker(cid, celegans_theta=None):
    try:
        import numpy as np
        import jax2_cases as C
        import oracle_jax2 as OJ
        from parity_checks import bytes_diff
        port, orc = C.port_modules(), C.oracle_modules()
        obs = OJ.observation(cid)
        if celegans_theta is not None:
            points = celegans_theta
        else:
            points = C.theta_points(cid)
        rows = []
        for name in ("jax2", "jax2_h"):
            po, pi = C.objective(port, obs, name)
            oo, oi = C.objective(orc, obs, name)
            grid_equal = pi == oi
            hs = [(None, None)] if name == "jax2" else C.h_points(pi)
            for label, th in points:
                for hl, h in hs:
                    z = C.z_of(th, h)
                    t0 = time.process_time()
                    d = bytes_diff(np.asarray(po.f(z)), np.asarray(oo.f(z)), "value")
                    (va, ga), (vb, gb) = po.vg(z), oo.vg(z)
                    dv = bytes_diff(np.asarray(va), np.asarray(vb), "vg.value") + \
                        bytes_diff(np.asarray(ga), np.asarray(gb), "vg.grad")
                    rows.append({"cohort_id": cid, "name": name, "theta": label, "h": hl, "grid_equal": grid_equal,
                                 "N": pi["N"], "floored": pi["binds"], "value_equal": d == [], "vg_equal": dv == [],
                                 "diffs": (d + dv)[:4], "cpu_s": time.process_time() - t0})
        return {"cohort_id": cid, "rows": rows}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-2000:]}


# ------------------------------------------------------------------------------------------------- fits
def fits_worker(cid, names, stored):
    """Port and oracle fits of one cohort under each name; compared with each other and with the stored Mac fit."""
    try:
        import oracle_jax2 as OJ
        from parity_checks import tree_diff
        from sraging.calibrators import profile_jax2 as PJ
        obs = OJ.observation(cid)
        out = []
        for name in names:
            cohort = port_observation(obs)
            c0 = time.process_time()
            port = PJ.run_port(cohort, f"profile_{name}", capture=True)
            port_cpu = time.process_time() - c0
            c0 = time.process_time()
            ores, oclaims, osc = OJ.fit(obs, name)
            orc_cpu = time.process_time() - c0
            pres, ores_c = _comparable(port["result"]), _comparable(ores)
            theta_equal = [float(x).hex() for x in port["result"]["theta"]] == [float(x).hex() for x in ores["theta"]]
            d_res = tree_diff(pres, ores_c)
            from sraging.calibrators.profile_jax2 import strip_timing
            d_claims = tree_diff(strip_timing(port["claims"]), strip_timing(oclaims))
            d_sc = tree_diff(strip_timing(port["sc_block"]), strip_timing(osc))
            row = {"cohort_id": cid, "name": name, "theta_bytes_equal": theta_equal,
                   "result_equal": d_res == [], "claims_equal": d_claims == [], "sc_equal": d_sc == [],
                   "diffs": (d_res + d_claims + d_sc)[:6], "port_cpu_s": port_cpu, "oracle_cpu_s": orc_cpu,
                   "mechanisms": PJ._mechanisms(port["result"], port["sc_block"]),
                   "port_claims": strip_timing(port["claims"]), "oracle_claims": strip_timing(oclaims)}
            st = stored.get(f"{cid}__{name}")
            if st is not None:
                row["stored"] = _stored_cmp(port, st)
            out.append(row)
        return {"cohort_id": cid, "rows": out}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-2000:]}


def _stored_cmp(port, st):
    """The port's fit against one stored Mac fit: result and claims equal, and the diff (all of it counted)."""
    from parity_checks import tree_diff
    from sraging.calibrators.profile_jax2 import strip_timing
    from sraging.result import canon
    pres, sres = _comparable(port["result"]), _comparable(st["result"])
    d = tree_diff(pres, sres)
    return {"path": st["_path"], "result_equal": canon(pres) == canon(sres),
            "claims_equal": canon(strip_timing(port["claims"])) == canon(strip_timing(st["claims"])),
            "n_diffs": len(d), "diffs": d[:8]}


def stored_worker(cid, names, stored):
    """`fits --stored-only`: the port's fit of one cohort under each name against the stored Mac fit only."""
    try:
        import oracle_jax2 as OJ
        from sraging.calibrators import profile_jax2 as PJ
        obs = OJ.observation(cid)
        out = []
        for name in names:
            c0 = time.process_time()
            port = PJ.run_port(port_observation(obs), f"profile_{name}", capture=True)
            out.append({"cohort_id": cid, "name": name, **_stored_cmp(port, stored[f"{cid}__{name}"]),
                        "port_cpu_s": time.process_time() - c0})
        return {"cohort_id": cid, "rows": out}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-2000:]}


def run_pool(fn, jobs, workers):
    ctx = mp.get_context("spawn")
    out = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 -- a crashed worker is a row, never a silent pass
                r = {"cohort_id": futs[f][0], "error": f"crash: {type(e).__name__}: {e}"}
            out.append(r)
            print(f"  [{i}/{len(jobs)}] {r['cohort_id']}: {'ERROR ' + r['error'][:80] if 'error' in r else 'ok'}",
                  flush=True)
    return sorted(out, key=lambda r: r["cohort_id"])
