"""Spec 003 `speed` phase (SC-010; research R17; task T019; sr-manager gate 7).

The port's process CPU per profile_jax2 fit against the frozen oracle's, on the panel's speed cohorts
(panel_003.toml [speed]), `replicates` rounds each, alternating which side goes first so drift cancels. A cohort's
ratio is the ratio of the sides' median CPU; the panel ratio is the geometric mean over cohorts with a 95%
bootstrap CI that resamples cohorts and, within each, rounds (tools/acceptance_002_speed.py `_boot`). Pass: the CI
includes 1 and its upper limit is <= 1.03. One process, sequential (no workers): concurrent fits would share cores.

A busy machine is NOT APPLICABLE, never a fail: the load average and the core counts are recorded at the start, after
each cohort and at the end. The phase refuses to start when the 1-minute load exceeds LOAD_MAX; the result is
not_applicable when, at any later sample, the 1-minute load less this run's own process (OWN_LOAD, one runnable
thread) exceeds LOAD_MAX. The measured ratio and the status it would have had are kept in the result.
"""
from __future__ import annotations

import os
import subprocess
import time

import numpy as np

UPPER = 1.03
LOAD_MAX = 5.0          # the 1-minute load above which the machine is busy: spec 002's bar (Naveh 2026-10-10,
                        # decisions_003.toml [speed_load]; was 1.0, the manager's run order "load below 1")
OWN_LOAD = 1.0          # this run's own contribution to the load while it fits (one sequential process)
THREAD_ENV = {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
              "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def cores() -> dict:
    """os.cpu_count and, on macOS, the physical cores per performance level (hw.perflevel<i>.{name,physicalcpu})."""
    out = {"logical": os.cpu_count(), "levels": []}
    def sysctl(k):
        r = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    n = sysctl("hw.nperflevels")
    for i in range(int(n) if n and n.isdigit() else 0):
        out["levels"].append({"name": sysctl(f"hw.perflevel{i}.name"),
                              "physical": int(sysctl(f"hw.perflevel{i}.physicalcpu") or 0)})
    return out


def sample(tag: str) -> dict:
    return {"at": tag, "time": time.time(), "load": list(os.getloadavg())}


def busy(samples, load_max=LOAD_MAX, own=OWN_LOAD) -> list[str]:
    """Reasons the machine was busy: the start sample's 1-minute load above load_max, or any later sample's above
    load_max + own (that later load includes this run's process)."""
    out = []
    for i, s in enumerate(samples):
        lim = load_max if i == 0 else load_max + own
        if s["load"][0] > lim:
            out.append(f"1-minute load {s['load'][0]:.2f} > {lim:g} at {s['at']}")
    return out


def phase_speed(args):
    for k, v in THREAD_ENV.items():          # before JAX is imported by anything below
        os.environ.setdefault(k, v)
    import sys
    import acceptance_003 as A
    from acceptance_002_speed import _boot
    panel = A.load_toml(A.PANEL)["speed"]
    ids, R = list(panel["cohorts"]), int(panel["replicates"])
    if args.only:
        ids = [c for c in ids if c in args.only]
    run = A.Run("speed", manifests=[A.PANEL], dev=args.dev or bool(args.only),
                cost=f"{len(ids)} cohorts x {R} replicates x 2 sides, sequential on one core: about "
                     f"{len(ids) * R * 2 * 250 / 3600:.1f} CPU-h (stored Mac fits: 155-290 CPU-s each)")
    host = {"cores": cores(), "load_max": LOAD_MAX, "own_load": OWN_LOAD}
    samples = [sample("start")]
    if busy(samples) and not args.dev:
        run.write({"status": "not_applicable", "reasons": busy(samples) + ["refused before any fit: a busy machine "
                                                                          "measures contention, not the port"],
                   "load_samples": samples, **host})
        sys.exit(3)
    A.preflight_or_exit(run)
    import oracle_jax2 as OJ
    from sraging.adapters.observation import cohort_from_observation
    from sraging.calibrators import profile_jax2 as PJ
    per, rows = [], []
    for cid in ids:
        obs = OJ.observation(cid)
        cohort = cohort_from_observation(obs)
        port_t, orc_t = [], []
        for r in range(R):
            order = ("port", "oracle") if r % 2 == 0 else ("oracle", "port")
            for side in order:
                c0 = time.process_time()
                if side == "port":
                    PJ.run_port(cohort, "profile_jax2", capture=True)
                    port_t.append(time.process_time() - c0)
                else:
                    OJ.fit(obs, "jax2")
                    orc_t.append(time.process_time() - c0)
        per.append((np.asarray(port_t), np.asarray(orc_t)))
        rows.append({"cohort_id": cid, "port_cpu_s": port_t, "oracle_cpu_s": orc_t,
                     "ratio": float(np.median(port_t) / np.median(orc_t))})
        samples.append(sample(cid))
        print(f"  {cid}: ratio {rows[-1]['ratio']:.4f} (load {samples[-1]['load'][0]:.2f})", flush=True)
    samples.append(sample("end"))
    host["cores_end"] = cores()
    est, lo, hi = _boot(per, 2000, np.random.default_rng(20261005))
    ok = lo <= 1.0 <= hi and hi <= UPPER
    measured = ("smoke-pass" if ok else "smoke-fail") if args.only else ("pass" if ok else "fail")
    reasons = busy(samples)
    run.write({"status": "not_applicable" if reasons else measured, "measured_status": measured, "reasons": reasons,
               "ratio": est, "ci95": [lo, hi], "upper_limit": UPPER, "rows": rows, "load_samples": samples,
               "load_before": samples[0]["load"], "load_after": samples[-1]["load"], "replicates": R, **host})
