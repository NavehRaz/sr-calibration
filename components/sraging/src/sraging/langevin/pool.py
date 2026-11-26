"""Multi-core Langevin (spec 004 FR-013; research R17): chunks of individuals over worker processes.

A sample depends only on (key, i) (rng.py), so the kernel over [i0, i1) returns the same bytes in any process, and the
parent places each chunk by its individuals' indices: the sample is bit-identical for any worker count and any chunk
size, and every one of n is present (B4, SRK's parallel path, dropped n mod n_jobs). `simulate_many` pools the chunks
of several configurations, each with its own key, and returns the samples in input order. `drop_remainder_times` is
HR-8's plant: B4 reproduced, n mod W individuals padded as censored at t_max, as SRK's SR_lf padded them.
"""
from __future__ import annotations

import numpy as np

from sraging.errors import UnsupportedOptionError


_NO_REC = np.zeros(0, dtype=np.int64)


def _chunks(n, chunk):
    return [(i0, min(i0 + chunk, n)) for i0 in range(0, n, chunk)]


def _task(job, i0, i1):
    """One (configuration, chunk) task: the kernel over individuals i0..i1-1 of the job, with their own slices."""
    from sraging.langevin import kernels as K
    a = job["arrays"]
    sl = slice(i0, i1)
    s, b = job["seeds"]
    out = K.run_chunk(0, i1 - i0, s[sl], b[sl], a["eta"][sl], a["beta"][sl], a["epsilon"][sl], a["kappa"][sl],
                      a["xc"][sl], a["t_ext"][sl], job["dt"], job["t_max"], job["wall"], job["bridge"], job["split"],
                      job.get("rec_steps", _NO_REC))
    return i0, i1, out


def _slice_job(job, i0, i1):
    """The task's own share of a job: the arrays cut to [i0, i1), so a worker receives no more than its chunk."""
    a = {k: v[i0:i1] for k, v in job["arrays"].items()}
    s, b = job["seeds"]
    return {**job, "arrays": a, "seeds": (s[i0:i1], b[i0:i1])}, 0, i1 - i0


def _run_tasks(tasks, workers):
    """[(job, i0, i1)] -> their outputs in task order, in process (workers = 1) or across spawned processes."""
    if workers == 1:
        return [_task(job, i0, i1) for job, i0, i1 in tasks]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_task, *_slice_job(job, i0, i1)) for job, i0, i1 in tasks]
        out = []
        for (job, i0, i1), f in zip(tasks, futs):
            _, _, r = f.result()
            out.append((i0, i1, r))
    return out


def _place(n, results):
    """The chunks' outputs placed by individual index (the paths too: the parent's (n, k) block); refused unless they
    cover range(n) exactly, each once."""
    k = results[0][2][3].shape[1] if results else 0
    times, kinds, x_end, X = np.empty(n), np.empty(n, dtype=np.int8), np.empty(n), np.empty((n, k))
    seen = np.zeros(n, dtype=bool)
    for i0, i1, (t, kd, x, xp) in results:
        if seen[i0:i1].any():
            raise ValueError(f"individuals {i0}..{i1 - 1} placed twice")
        times[i0:i1], kinds[i0:i1], x_end[i0:i1], X[i0:i1] = t, kd, x, xp
        seen[i0:i1] = True
    if not seen.all():
        raise ValueError(f"{int((~seen).sum())} individuals missing: refused (no silent loss)")
    return times, kinds, x_end, X


def run_chunks(job, n, chunk, workers):
    """(times, kinds, x_end, X) of one rung of one configuration over its chunks."""
    return _place(n, _run_tasks([(job, i0, i1) for i0, i1 in _chunks(n, chunk)], workers))


def simulate_many(configs, *, workers):
    """One sample per LangevinConfig, in input order, each with its own key; the (configuration, chunk) tasks of every
    rung share one pool of `workers` processes."""
    from sraging.langevin import sample as S
    from sraging.langevin.rng import hash64_array
    configs = list(configs)
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise UnsupportedOptionError("workers", f"workers is an integer >= 1, got {workers!r}")
    for c in configs:
        if not isinstance(c, S.LangevinConfig):
            raise UnsupportedOptionError("configs", "simulate_many takes LangevinConfig objects")
    S.require_numba()
    states, tasks, owners = [], [], []
    for j, cfg in enumerate(configs):
        st = S.prepare(cfg)
        states.append(st)
        idx = np.arange(cfg.n)
        seeds = (hash64_array(cfg.key, idx), hash64_array(cfg.key, idx, "bridge"))
        for split in ((1, 2) if cfg.error == "ladder" else (1,)):
            job = S.rung_job(cfg, st["arrays"], st["t_max"], split, seeds=seeds)
            for i0, i1 in _chunks(cfg.n, cfg.chunk):
                tasks.append((job, i0, i1))
                owners.append((j, split))
    results = _run_tasks(tasks, int(workers))
    out = []
    for j, cfg in enumerate(configs):
        rungs = {split: _place(cfg.n, [r for r, o in zip(results, owners) if o == (j, split)])
                 for split in ((1, 2) if cfg.error == "ladder" else (1,))}
        out.append(S.finish(cfg, states[j], rungs[1], rungs.get(2)))
    return out


def drop_remainder_times(cfg, W) -> np.ndarray:
    """HR-8's plant (B4): the raw times of the first W floor(n / W) individuals, the remaining n mod W padded as
    censored at t_max (SRK's SR_lf padding at t_end)."""
    from sraging.langevin import kernels as K
    from sraging.langevin import sample as S
    from sraging.langevin.rng import hash64_array
    st = S.prepare(cfg)
    keep = cfg.n - cfg.n % int(W)
    idx = np.arange(cfg.n)
    job = {"seeds": (hash64_array(cfg.key, idx), hash64_array(cfg.key, idx, "bridge")), "arrays": st["arrays"],
           "dt": cfg.dt, "t_max": st["t_max"], "wall": K.WALLS[cfg.wall], "bridge": cfg.kernel == "bridge", "split": 1}
    t = run_chunks(job, keep, cfg.chunk, 1)[0] if keep else np.empty(0)
    return np.concatenate([t, np.full(cfg.n - keep, st["t_max"])])
