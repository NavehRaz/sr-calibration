"""Nodes over worker processes, bit-identical for every worker count (spec 004 FR-007; research R6; data-model "Node
pool"; contracts/python-api.md §3).

    with NodePool(ev, workers=W) as pool:              # W in 1..os.cpu_count(); spawned processes
        r = ev.mixture_value(params, grad=True, pool=pool)

**Workers.** W `spawn` processes. Each imports sraging, so the engine's runtime (PJRT_NPROC=1, x64) is set before any
computation, and rebuilds the evaluator from its construction arguments (`Evaluator.construction()`), so it compiles
the same per-node programs. The pool records the evaluator's digest and refuses any other configuration.

**Assignment and fold.** Node k goes to worker k mod W. A worker returns (k, P_k, cpu_s) per node as it finishes; the
parent folds strictly in k = 0, 1, ..., M-1 (`in_order`), buffering early arrivals. Each P_k is the output of the same
compiled program on the same inputs and host, and the fold is the same sequence of float64 additions for every W, so
the value and gradient are byte-identical to the in-process ones, M not divisible by W included (HR-4 tests it).

**No silent loss.** The parent refuses an evaluation unless the node indices it received are exactly range(M), each
once (B4's failure mode, for nodes). `drop_remainder_value` is HR-4's plant: the mixture without the M mod W remainder.

**Two passes** go to the same workers: pass 2 sends the mixture cotangent once per worker, and each worker scales it by
its nodes' weights (the parent's own float64 products). CPU: each task reports its worker CPU; compilation is reported
apart, per worker.
"""
from __future__ import annotations

import os
import queue
import time

import numpy as np

from sraging.errors import UnsupportedOptionError

_START_TIMEOUT_S = 300.0          # a worker's import and evaluator build
_POLL_S = 1.0


class NodeIndicesError(ValueError):
    """The node indices received are not exactly range(M), each once: the evaluation is refused (research R6)."""


def in_order(stream, M):
    """Yields the stream's items (k, ...) in k = 0..M-1, buffering early arrivals; refuses an index outside range(M),
    one seen twice, and any missing at the end."""
    buf, seen, nxt = {}, set(), 0
    for item in stream:
        k = item[0]
        if not (isinstance(k, (int, np.integer)) and 0 <= k < M):
            raise NodeIndicesError(f"node index {k!r} is outside range({M}): the evaluation is refused")
        if k in seen:
            raise NodeIndicesError(f"node {k} twice: the indices received must be range({M}), each once")
        seen.add(k)
        buf[int(k)] = item
        while nxt in buf:
            yield buf.pop(nxt)
            nxt += 1
    if nxt != M:
        missing = sorted(set(range(M)) - seen)
        raise NodeIndicesError(f"nodes missing {missing[:10]}{' ...' if len(missing) > 10 else ''}: "
                               f"{len(seen)} of {M} received; the evaluation is refused (no silent loss)")


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:                                                    # a zombie still answers kill 0: reap it if it is ours
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


# ------------------------------------------------------------------------------------------------- the worker side
def _worker(w, construction, tasks, results):
    """One worker: build the evaluator, report the runtime, then serve tasks until None."""
    try:
        from sraging.engine import mixture as MIX
        from sraging.engine.evaluator import Evaluator
        import jax
        ev = Evaluator(**construction)
        progs = MIX.programs(ev)
        results.put(("ready", w, {"pid": os.getpid(), "PJRT_NPROC": os.environ.get("PJRT_NPROC"),
                                  "x64": bool(jax.config.jax_enable_x64), "digest": ev.digest()}))
    except BaseException as ex:                             # noqa: BLE001 - reported to the parent, which refuses
        results.put(("error", w, f"{type(ex).__name__}: {ex}"))
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        kind, job, z, payload = task
        try:
            if kind == "points":
                for k, s in payload:
                    c0, comp0 = time.process_time(), sum(progs.compile_cpu_s.values())
                    P = progs.points(z, s)
                    comp = sum(progs.compile_cpu_s.values()) - comp0
                    results.put(("node", job, k, P, time.process_time() - c0 - comp, comp, w))
            else:                                           # "vjp": (c, [(k, s, w_k)], with_points)
                c, nodes, with_points = payload
                for k, s, wk in nodes:
                    c0, comp0 = time.process_time(), sum(progs.compile_cpu_s.values())
                    ck = {key: wk * v for key, v in c.items()}
                    out = progs.node_vjp(z, s, ck, with_points=with_points)
                    comp = sum(progs.compile_cpu_s.values()) - comp0
                    results.put(("node", job, k, out, time.process_time() - c0 - comp, comp, w))
        except BaseException as ex:                         # noqa: BLE001
            results.put(("error", w, f"{type(ex).__name__}: {ex}"))
        results.put(("done", job, w))


# ------------------------------------------------------------------------------------------------- the parent side
class NodePool:
    """W worker processes bound to one evaluator configuration. A context manager: closing it ends the processes."""

    def __init__(self, ev, *, workers):
        if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
            raise UnsupportedOptionError("workers", f"workers is an integer >= 1, got {workers!r}")
        cores = os.cpu_count() or 1
        if workers > cores:
            raise UnsupportedOptionError("workers", f"workers = {workers} is above os.cpu_count() = {cores}")
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.workers = int(workers)
        self.evaluator_digest = ev.digest()
        self.closed = False
        self._job = 0
        self._tasks = [ctx.Queue() for _ in range(self.workers)]
        self._results = ctx.Queue()
        self._procs = [ctx.Process(target=_worker, args=(w, ev.construction(), self._tasks[w], self._results),
                                   daemon=True, name=f"sraging-node-{w}") for w in range(self.workers)]
        for p in self._procs:
            p.start()
        self.started = tuple(p.pid for p in self._procs)
        self.compile_cpu_s = {w: 0.0 for w in range(self.workers)}
        self.runtime = [None] * self.workers
        try:
            for _ in range(self.workers):
                msg = self._get(_START_TIMEOUT_S)
                if msg[0] == "error":
                    raise RuntimeError(f"worker {msg[1]} failed to start: {msg[2]}")
                _, w, info = msg
                if info["digest"] != self.evaluator_digest:
                    raise RuntimeError(f"worker {w} rebuilt another evaluator (digest {info['digest'][:12]}...)")
                self.runtime[w] = info
        except BaseException:
            self.close()
            raise

    # -- life ----------------------------------------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self.closed:
            return
        self.closed = True
        for q in self._tasks:
            try:
                q.put(None)
            except (OSError, ValueError):
                pass
        for p in self._procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for q in (*self._tasks, self._results):
            q.close()
            q.join_thread()

    def require(self, ev):
        """Refused: a closed pool, or one bound to another evaluator configuration."""
        if self.closed:
            raise UnsupportedOptionError("pool", "this NodePool is closed")
        if ev.digest() != self.evaluator_digest:
            raise UnsupportedOptionError("pool", f"this NodePool is bound to another evaluator configuration (digest "
                                         f"{self.evaluator_digest[:12]}..., the evaluator's {ev.digest()[:12]}...)")

    # -- one evaluation's tasks ------------------------------------------------------------------------------------
    def _get(self, timeout):
        t0 = time.monotonic()
        while True:
            try:
                return self._results.get(timeout=_POLL_S)
            except queue.Empty:
                dead = [w for w, p in enumerate(self._procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"NodePool worker(s) {dead} died") from None
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"NodePool: no reply in {timeout:g} s") from None

    def _run(self, kind, z, per_worker):
        """Sends each worker its tasks; yields (k, payload, cpu_s) as replies arrive (any order) until every worker
        reports done. Compilation is accumulated apart, per worker."""
        self._job += 1
        job = self._job
        for w in range(self.workers):
            self._tasks[w].put((kind, job, np.asarray(z), per_worker[w]))
        done = 0
        while done < self.workers:
            msg = self._get(_START_TIMEOUT_S)
            if msg[0] == "error":
                raise RuntimeError(f"NodePool worker {msg[1]}: {msg[2]}")
            if msg[1] != job:
                continue                                    # a stale reply of an abandoned evaluation
            if msg[0] == "done":
                done += 1
                continue
            _, _, k, payload, cpu, comp, w = msg
            self.compile_cpu_s[w] += comp
            yield k, payload, cpu

    def points(self, z, nodes):
        """nodes: [(k, s)] -> yields (k, P_k, cpu_s); node k on worker k mod W."""
        per = [[] for _ in range(self.workers)]
        for k, s in nodes:
            per[k % self.workers].append((k, np.asarray(s)))
        return self._run("points", z, per)

    def vjp(self, z, c, nodes, *, with_points=False):
        """nodes: [(k, s, w_k)] -> yields (k, (cz, cs) or (P, cz, cs), cpu_s); the cotangent c goes once to each
        worker, which scales it by w_k."""
        per = [[] for _ in range(self.workers)]
        for k, s, wk in nodes:
            per[k % self.workers].append((k, np.asarray(s), float(wk)))
        c = {key: np.asarray(v) for key, v in c.items()}
        return self._run("vjp", z, [(c, per[w], with_points) for w in range(self.workers)])


def drop_remainder_value(ev, params, rule, W) -> float:
    """HR-4's plant, B4's failure mode for nodes: the mixture over the first M - (M mod W) nodes only (a loop over full
    rounds of W), the dropped nodes' weight lost. In process; the pool itself refuses such a reply set."""
    from sraging.engine import mixture as MIX
    het = MIX.require_truncated_normal(ev)
    keep = rule.M - rule.M % int(W)
    progs = MIX.programs(ev)
    z = ev._z(params)
    stream = ((k, progs.points(z, MIX._scales(het.parameter, float(rule.s[k]))), 0.0) for k in range(keep))
    mixed, _, _ = MIX._fold_stream(stream, rule, het.parameter, keep)
    return progs.total(mixed, z)
