"""The frozen `fp_piecewise` oracle (spec 002, T015): SRK at cb191f5, used only through a worker process.

The snapshot `tests/oracle/fp_piecewise_cb191f5/` is imported as top-level `SRK`, exactly as v033's is, so
the two cannot share a process. Evaluations therefore run in a worker subprocess:

    problems = preflight()                      # [] or the reasons the oracle cannot be used validly
    out = call("fp_loglik_physical", death_times=t, events=e, eta=..., beta=..., epsilon=..., xc=1.0,
               kappa_phys=..., L=124.0, N=100, dt=1e-2, tau_max=3.0)       # -> dict of values + cpu_s
    with session() as s:                        # one long-lived worker: import cost paid once
        s.call(...); s.call(...)

Protocol. A request is JSON `{"op": ..., "args": {...}}`; numpy arrays go in a sibling `.npz` and are
referenced as `{"__array__": key}`. The worker writes a reply `.npz` (result arrays + `cpu_s`, the process CPU
time of the op, import excluded; or `error`). Single shot: `python tools/oracle_fp_piecewise.py worker
<request.json> <reply.npz>`. Session: `... serve`, one JSON line `{"request", "reply"}` in on stdin and one
JSON line `{"ok": bool, ...}` out on stdout per call. Files go in `var/oracle_io/` (gitignored). Nothing is
ever written into the snapshot: the worker redirects bytecode before it imports SRK.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYCACHE = os.path.join(REPO, ".cache", "pycache")
SNAP = os.path.join(REPO, "tests", "oracle", "fp_piecewise_cb191f5")
SNAP_SRK_ROOT = os.path.join(SNAP, "SRtools_kappa")
MANIFEST = os.path.join(SNAP, "MANIFEST.json")
IO_DIR = os.path.join(REPO, "var", "oracle_io")
NUMERICS_OVERRIDES = ("SRK_DT_CEILING", "SRK_KAPPA_MAX")
FREEZE_CRITICAL = ("numpy", "scipy", "numba")
THIS = os.path.abspath(__file__)


class NotApplicable(RuntimeError):
    """The oracle cannot be used validly here; never a pass."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("NOT APPLICABLE: " + "; ".join(self.problems))


class OracleError(RuntimeError):
    """The oracle's SRK raised (the exception type and message are in the text)."""


def _md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _read_manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def _freeze():
    import re
    out = {}
    for line in open(os.path.join(SNAP, "pip_freeze.txt")):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(.+)$", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def fingerprint(root=SNAP) -> dict:
    """{relative path: md5} of every file under `root`: a new, missing or changed file all show."""
    out = {}
    for d, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = _md5(p)
    return out


def preflight(worker: bool = False) -> list[str]:
    """Every reason the oracle would not be valid here (empty = valid). `worker=True` also checks that
    THIS process (the one about to import SRK) has its bytecode redirected."""
    problems = []
    if worker and not sys.pycache_prefix:
        problems.append("sys.pycache_prefix is not set in the worker: importing the oracle would write into tests/oracle")
    man = _read_manifest()
    for f in man["files"]:
        p = os.path.join(SNAP, f["repo"], f["path"])
        if not os.path.exists(p):
            problems.append(f"snapshot file missing: {f['repo']}/{f['path']}")
        elif _md5(p) != f["md5"]:
            problems.append(f"snapshot checksum mismatch: {f['repo']}/{f['path']}")
    freeze = _freeze()
    for pkg, want in man["environment_pins"].items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = "not installed"
        if got != want:
            problems.append(f"{pkg} {got} != pinned {want}")
    for pkg in FREEZE_CRITICAL:
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = "not installed"
        if got != freeze.get(pkg):
            problems.append(f"{pkg} {got} != pip_freeze {freeze.get(pkg)}")
    if sys.version.split()[0] != man["python"].split()[0]:
        problems.append(f"python {sys.version.split()[0]} != snapshot {man['python'].split()[0]}")
    for var in NUMERICS_OVERRIDES:
        if os.environ.get(var):
            problems.append(f"{var} is set ({os.environ[var]!r}): it changes the oracle's numerics")
    # hostname deliberately not compared (macOS assigns it from the network; see tools/oracle_v033.py)
    if platform.machine() != man["host"]["machine"] or platform.platform() != man["host"]["platform"]:
        problems.append(f"host {platform.machine()} {platform.platform()} is not the snapshot host "
                        f"{man['host']['machine']} {man['host']['platform']}")
    return problems


# ------------------------------------------------------------------------------------------ encoding
def _encode(obj, arrays):
    """JSON-able copy of `obj`; every ndarray moves to `arrays` and is replaced by a reference."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = obj
        return {"__array__": key}
    if isinstance(obj, dict):
        return {k: _encode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v, arrays) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _decode(obj, arrays):
    if isinstance(obj, dict):
        if set(obj) == {"__array__"}:
            return arrays[obj["__array__"]]
        return {k: _decode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    return obj


def _write_request(op, kwargs, req_json):
    import numpy as np
    arrays = {}
    body = {"op": op, "args": _encode(kwargs, arrays)}
    npz = req_json[:-5] + ".npz"
    with open(req_json, "w") as fh:
        json.dump(body, fh)
    np.savez(npz, **arrays)


def _read_reply(reply_npz):
    import numpy as np
    with np.load(reply_npz, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if "error" in out:
        raise OracleError(str(out["error"]))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in out.items()}


# --------------------------------------------------------------------------------------------- client
def _worker_env():
    keep = {k: v for k, v in os.environ.items()
            if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")}
    keep["NUMBA_CACHE_DIR"] = os.path.join(REPO, ".cache", "numba")
    return keep


def _check():
    problems = preflight()
    if problems:
        raise NotApplicable(problems)


def _tmp_paths():
    os.makedirs(IO_DIR, exist_ok=True)
    fd, req = tempfile.mkstemp(prefix="req_", suffix=".json", dir=IO_DIR)
    os.close(fd)
    return req, req[:-5] + "_reply.npz"


def _cleanup(*paths):
    for p in paths:
        with contextlib.suppress(OSError):
            os.remove(p)


def call(op: str, **kwargs) -> dict:
    """Run one op in a fresh worker (pays the import each time; use `session()` for many calls)."""
    _check()
    req, reply = _tmp_paths()
    try:
        _write_request(op, kwargs, req)
        r = subprocess.run([sys.executable, THIS, "worker", req, reply], capture_output=True, text=True,
                           env=_worker_env())
        if r.returncode == 3:
            raise NotApplicable([r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "worker preflight failed"])
        if r.returncode or not os.path.exists(reply):
            raise OracleError(f"worker exited {r.returncode}: {r.stderr[-2000:]}")
        return _read_reply(reply)
    finally:
        _cleanup(req, req[:-5] + ".npz", reply)


class Session:
    """One long-lived worker process serving many calls over a line protocol on stdin/stdout."""

    def __init__(self):
        _check()
        os.makedirs(IO_DIR, exist_ok=True)
        self._p = subprocess.Popen([sys.executable, THIS, "serve"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env=_worker_env(), bufsize=1)
        hello = self._p.stdout.readline()
        if not hello:
            err = self._p.stderr.read()
            self._p.wait()
            if self._p.returncode == 3:
                raise NotApplicable([err.strip().splitlines()[-1] if err.strip() else "worker preflight failed"])
            raise OracleError(f"worker did not start: {err[-2000:]}")
        self.ready = json.loads(hello)

    def call(self, op: str, **kwargs) -> dict:
        if self._p.poll() is not None:
            raise OracleError(f"worker has exited ({self._p.returncode}): {self._p.stderr.read()[-2000:]}")
        req, reply = _tmp_paths()
        try:
            _write_request(op, kwargs, req)
            self._p.stdin.write(json.dumps({"request": req, "reply": reply}) + "\n")
            self._p.stdin.flush()
            line = self._p.stdout.readline()
            if not line:
                self._p.wait()
                raise OracleError(f"worker died ({self._p.returncode}): {self._p.stderr.read()[-2000:]}")
            msg = json.loads(line)
            if not msg["ok"]:
                raise OracleError(msg["error"])
            return _read_reply(reply)
        finally:
            _cleanup(req, req[:-5] + ".npz", reply)

    def close(self):
        if self._p.poll() is None:
            with contextlib.suppress(OSError):
                self._p.stdin.close()
            try:
                self._p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._p.kill()
                self._p.wait()
        for s in (self._p.stdout, self._p.stderr):
            with contextlib.suppress(OSError):
                s.close()


@contextlib.contextmanager
def session():
    s = Session()
    try:
        yield s
    finally:
        s.close()


# ------------------------------------------------------------------------------------------- worker
_SRK = {}


def _load_snapshot():
    """Import SRK from the snapshot ONLY (worker side). Preflight first; bytecode redirected."""
    if _SRK:
        return _SRK
    problems = preflight(worker=True)
    if problems:
        sys.stderr.write("NOT APPLICABLE: " + "; ".join(problems) + "\n")
        sys.exit(3)
    sys.path.insert(0, SNAP_SRK_ROOT)
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    import SRK
    root = os.path.realpath(SNAP_SRK_ROOT) + os.sep
    assert os.path.realpath(SRK.__file__).startswith(root), f"SRK imported from {SRK.__file__}, not the snapshot"
    from SRK import curves, fp_likelihood, fp_piecewise, intervention_sim
    for m in (curves, fp_likelihood, fp_piecewise, intervention_sim):
        assert os.path.realpath(m.__file__).startswith(root), f"{m.__name__} from {m.__file__}"
    _SRK.update(np=np, curves=curves, fpl=fp_likelihood, fpp=fp_piecewise, isim=intervention_sim)
    return _SRK


def _interventions(specs):
    S = _SRK
    out = []
    for d in specs or ():
        w = d.get("window")
        w = None if w is None else (float(w[0]), float(w[1]))
        if d["kind"] == "Intervention":
            out.append(S["isim"].Intervention(d["param"], float(d["effect"]), w))
        elif d["kind"] == "ExtHazard":
            out.append(S["isim"].ExtHazard(float(d["level"]), w))
        else:
            raise ValueError(f"unknown intervention kind {d['kind']!r}")
    return out


def _theta(d):
    return _SRK["curves"].Theta(**d)


def _arr(x):
    return None if x is None else _SRK["np"].asarray(x, float)


def _data(a):
    a = dict(a)
    a["death_times"] = _arr(a["death_times"])
    if a.get("events") is not None:
        a["events"] = _SRK["np"].asarray(a["events"], int)
    return a


def _tr(a):
    if a.get("time_range") is not None:
        a["time_range"] = (float(a["time_range"][0]), float(a["time_range"][1]))
    if a.get("sampling_dt") is not None and not isinstance(a["sampling_dt"], (int, float)):
        a["sampling_dt"] = _arr(a["sampling_dt"])
    return a


def _op_fp_loglik_physical(a):
    a = _tr(_data(a))
    if "kappa" in a:                                  # alias: SRK's own name is kappa_phys
        a["kappa_phys"] = a.pop("kappa")
    r = _SRK["fpl"].fp_loglik_physical(**a)
    return r if isinstance(r, dict) else {"value": r}


def _op_fp_loglik_het_physical(a):
    a = _tr(_data(a))
    if "kappa" in a:
        a["kappa_phys"] = a.pop("kappa")
    return {"value": _SRK["fpl"].fp_loglik_het_physical(**a)}


def _op_fp_mixture_curves_physical(a):
    a = dict(a)
    if "kappa" in a:
        a["kappa_phys"] = a.pop("kappa")
    a["tau_eval"] = _arr(a["tau_eval"])
    S, p = _SRK["fpl"].fp_mixture_curves_physical(**a)
    return {"S_bar": S, "p_bar": p}


def _op_fpt_full(a):
    a = dict(a)
    a.setdefault("return_survival", True)
    r = _SRK["fpl"].fpt_full(**a)
    names = ["mean", "cv", "s_end"]
    if a["return_survival"]:
        names += ["tau", "S"] + (["flux"] if a.get("absorb_flux") else [])
    return dict(zip(names, r))


def _op_fpt_full_density(a):
    r = _SRK["fpl"].fpt_full_density(**a)
    return dict(zip(["tau", "S", "f", "mean", "cv", "s_end"], r))


def _pw(a):
    a = _tr(dict(a))
    if "death_times" in a:
        a = _data(a)
    a["theta"] = _theta(a["theta"])
    a["interventions"] = _interventions(a.get("interventions"))
    return a


def _op_piecewise_loglik_terms(a):
    return {"terms": _SRK["fpp"].piecewise_loglik_terms(**_pw(a))}


def _op_piecewise_loglik(a):
    return {"value": _SRK["fpp"].piecewise_loglik(**_pw(a))}


def _op_piecewise_curves(a):
    return dict(_SRK["fpp"].piecewise_curves(**_pw(a)))


def _op_build_segments(a):
    np = _SRK["np"]
    a = _pw(a)
    r = _SRK["fpp"].build_segments(**a)
    out = {"segments": np.array(r[0], float), "xc": r[1]}
    if a.get("return_exth"):
        out["exth_level"] = np.array([e.level for e in r[2]], float)
        out["exth_window"] = np.array([[np.nan, np.nan] if e.window is None else e.window for e in r[2]],
                                      float).reshape(-1, 2)
    return out


_SEEDER = {}


def _op_simulate_sr_hetro(a):
    """SR_hetro.death_times_euler_brownian_bridge (the snapshot's, read-only) with numba's generator seeded:
    times are j * dt (the step in which the crossing happened), events 1 for a crossing, 0 for survival to s."""
    np = _SRK["np"]
    from SRK import SR_hetro
    root = os.path.realpath(SNAP_SRK_ROOT) + os.sep
    assert os.path.realpath(SR_hetro.__file__).startswith(root), f"SR_hetro from {SR_hetro.__file__}"
    if "seed" not in _SEEDER:
        from numba import njit

        @njit
        def _seed(x):
            np.random.seed(x)
        _SEEDER["seed"] = _seed
    _SEEDER["seed"](int(a["seed"]))
    s, dt = int(a["steps"]), float(a["dt"])
    t = np.arange(s) * dt
    T, E = SR_hetro.death_times_euler_brownian_bridge(
        s, dt, t, float(a["eta"]), float(a.get("eta_var", 0.0)), float(a["beta"]), float(a.get("beta_var", 0.0)),
        float(a["kappa"]), float(a.get("kappa_var", 0.0)), float(a["epsilon"]), float(a.get("epsilon_var", 0.0)),
        float(a.get("xc", 1.0)), float(a.get("xc_var", 0.0)), float(np.sqrt(dt)), int(a["npeople"]),
        float(a.get("external_hazard", np.inf)), int(a.get("time_step_multiplier", 1)))
    return {"death_times": np.asarray(T, float), "events": np.asarray(E, int)}


def _snapshot_module(mod):
    """`mod` must come from the snapshot root (SRK's frozen code), as _op_simulate_sr_hetro asserts."""
    root = os.path.realpath(SNAP_SRK_ROOT) + os.sep
    assert os.path.realpath(mod.__file__).startswith(root), f"{mod.__name__} from {mod.__file__}"
    return mod


def _seed_numba(seed):
    np = _SRK["np"]
    if "seed" not in _SEEDER:
        from numba import njit

        @njit
        def _seed(x):
            np.random.seed(x)
        _SEEDER["seed"] = _seed
    _SEEDER["seed"](int(seed))


def _kernel_args(a):
    """SRK's kernel arguments (s, dt, t, eta, eta_var, beta, beta_var, kappa, kappa_var, epsilon, epsilon_var, xc,
    xc_var, sdt, npeople, external_hazard, time_step_multiplier) from named request fields."""
    np = _SRK["np"]
    s, dt = int(a["steps"]), float(a["dt"])
    return (s, dt, np.arange(s) * dt, float(a["eta"]), float(a.get("eta_var", 0.0)), float(a["beta"]),
            float(a.get("beta_var", 0.0)), float(a["kappa"]), float(a.get("kappa_var", 0.0)), float(a["epsilon"]),
            float(a.get("epsilon_var", 0.0)), float(a.get("xc", 1.0)), float(a.get("xc_var", 0.0)),
            float(np.sqrt(dt)), int(a["npeople"]), float(a.get("external_hazard", np.inf)),
            int(a.get("time_step_multiplier", 1)))


def _op_simulate_accelerator2(a):
    """B4 (spec 004 D-404; tasks T070): SR_hetro.death_times_accelerator2, SRK's parallel Euler path, read-only. It
    takes no job count: n_jobs = os.cpu_count() (recorded), and it simulates n_jobs * (npeople // n_jobs) people. Then
    SR_lf's padding (SRmodellib_lifelines.py:30-35, replicated): the rest at t_end, censored."""
    np = _SRK["np"]
    from SRK import SR_hetro
    _snapshot_module(SR_hetro)
    _seed_numba(a["seed"])
    T, E = SR_hetro.death_times_accelerator2(*_kernel_args(a))
    T, E = np.asarray(T, float), np.asarray(E, float)
    n, n_real = int(a["npeople"]), int(np.asarray(T).size)
    if n_real < n:                                         # SRmodellib_lifelines.SR_lf.calc_survival_and_hazard
        Tn, En = np.zeros(n), np.zeros(n)
        Tn[:n_real], En[:n_real] = T, E
        Tn[n_real:] = float(a["t_end"])
        T, E = Tn, En
    return {"death_times": T, "events": E.astype(int), "n_jobs": np.array(os.cpu_count()), "n_real": np.array(n_real)}


def _op_simulate_sr_hetro_parallel(a):
    """Seeding (spec 004 D-406; tasks T070): SR_hetro.death_times_euler_brownian_bridge_parallel, the joblib bridge
    wrapper, read-only. The parent's numba generator is seeded as asked; the joblib workers draw from their own
    generators, which that seed does not reach."""
    np = _SRK["np"]
    from SRK import SR_hetro
    _snapshot_module(SR_hetro)
    _seed_numba(a["seed"])
    T, E = SR_hetro.death_times_euler_brownian_bridge_parallel(*_kernel_args(a), n_jobs=int(a.get("n_jobs", -1)),
                                                              chunk_size=int(a.get("chunk_size", 1000)))
    return {"death_times": np.asarray(T, float), "events": np.asarray(E, int), "n_jobs": np.array(int(a.get("n_jobs", -1)))}


def _op_simulate_isr_paths(a):
    """D-411 (spec 004; tasks T070): intervention_SR.getISR with save_trajectory=True and no intervention, read-only:
    the recorder starts from zeros (intervention_SR.py:433) and writes x only while x < xc, at the outer step within
    dt/2 of each age (:461-463). theta = [eta, beta, epsilon, xc, kappa] (the kappa convention)."""
    np = _SRK["np"]
    from SRK import intervention_SR
    _snapshot_module(intervention_SR)
    _seed_numba(a["seed"])
    ages = np.asarray(a["ages"], float)
    theta = [float(a["eta"]), float(a["beta"]), float(a["epsilon"]), float(a.get("xc", 1.0)), float(a["kappa"])]
    sim = intervention_SR.getISR(theta, n=int(a["npeople"]), nsteps=int(a["nsteps"]), t_end=float(a["t_end"]),
                                 external_hazard=float(a.get("external_hazard", np.inf)),
                                 time_step_multiplier=int(a.get("time_step_multiplier", 1)), parallel=False,
                                 hetro=False, method=str(a.get("method", "brownian_bridge")), save_trajectory=True,
                                 traj_points=list(ages))
    if getattr(sim, "trajectories", None) is None:
        sim.calc_death_times()
    return {"death_times": np.asarray(sim.death_times, float), "events": np.asarray(sim.events, int),
            "trajectories": np.asarray(sim.trajectories, float), "ages": np.asarray(sim.traj_time_points, float)}


def _op_preset_theta(a):
    """Spec 004's presets (tasks T072; spec.md Clarifications, Naveh 2026-10-29), read-only: SRK.presets.getTheta(name,
    ExtH=True) at its defaults (type "mode_overall", time_unit "auto") and get_config_params(name) (nsteps,
    time_step_multiplier, npeople, t_end, in the preset's own time unit), with the time unit getTheta's auto picks."""
    np = _SRK["np"]
    from SRK import presets
    _snapshot_module(presets)
    th = np.asarray(presets.getTheta(a["name"], ExtH=True), float)
    cfg = presets.get_config_params(a["name"], config_params=["nsteps", "time_step_multiplier", "npeople", "t_end"],
                                    types=[int, int, int, int])
    low = presets.PRESET_ALIASES.get(a["name"], a["name"]).lower()
    unit = next((u for u, keys in (("years", ("human", "sweden", "denmark", "dog", "cat", "labrador", "staffy",
                                              "german", "jack", "pig", "guinea pigs")),
                                   ("hours", ("ecoli", "e. coli")), ("weeks", ("mice", "mouse")),
                                   ("generations", ("yeast",))) if any(k in low for k in keys)), "days")
    return {"theta": th, "nsteps": int(cfg["nsteps"]), "time_step_multiplier": int(cfg["time_step_multiplier"]),
            "npeople": int(cfg["npeople"]), "t_end": int(cfg["t_end"]), "time_unit": unit}


def _op_get_sr_hetro(a):
    """D-409 (spec 004; tasks T074): SR_hetro.getSrHetro, read-only, with SRK's own defaults unless named: without
    `hetro` in the request it is SRK's default (hetro=True, xc_var 0.2: B3/B21); `hetro=False` is the named
    homogeneous call. theta = [eta, beta, epsilon, xc, kappa]."""
    np = _SRK["np"]
    from SRK import SR_hetro
    _snapshot_module(SR_hetro)
    _seed_numba(a["seed"])
    theta = [float(a["eta"]), float(a["beta"]), float(a["epsilon"]), float(a.get("xc", 1.0)), float(a["kappa"])]
    kw = {"n": int(a["npeople"]), "nsteps": int(a["nsteps"]), "t_end": float(a["t_end"]),
          "time_step_multiplier": int(a.get("time_step_multiplier", 1))}
    if "external_hazard" in a:
        kw["external_hazard"] = float(a["external_hazard"])
    if "hetro" in a:
        kw["hetro"] = bool(a["hetro"])
    sim = SR_hetro.getSrHetro(theta, **kw)
    T, E = getattr(sim, "death_times", None), getattr(sim, "events", None)
    if T is None or E is None:
        T, E = sim.calc_death_times()
    return {"death_times": np.asarray(T, float), "events": np.asarray(E, int), "xc_var": np.asarray(sim.xc_var)}


def _op_step_size_to_grid(a):
    """D-403 (B2; tasks T074): stepsize.step_size_to_grid(t_end, step_size) and the step the kernels then realise,
    stepsize.grid_dt(..., convention="kernel"), read-only."""
    np = _SRK["np"]
    from SRK import stepsize
    _snapshot_module(stepsize)
    nsteps, tsm = stepsize.step_size_to_grid(float(a["t_end"]), float(a["step_size"]))
    realised = stepsize.grid_dt(float(a["t_end"]), nsteps, tsm, convention="kernel")
    return {"nsteps": np.asarray(int(nsteps)), "time_step_multiplier": np.asarray(int(tsm)),
            "realised_step": np.asarray(float(realised))}


OPS = {
    "ping": lambda a: {"value": 1},
    "get_sr_hetro": _op_get_sr_hetro,
    "step_size_to_grid": _op_step_size_to_grid,
    "preset_theta": _op_preset_theta,
    "simulate_sr_hetro": _op_simulate_sr_hetro,
    "simulate_accelerator2": _op_simulate_accelerator2,
    "simulate_sr_hetro_parallel": _op_simulate_sr_hetro_parallel,
    "simulate_isr_paths": _op_simulate_isr_paths,
    "fp_loglik_physical": _op_fp_loglik_physical,
    "fp_loglik_het_physical": _op_fp_loglik_het_physical,
    "fp_mixture_curves_physical": _op_fp_mixture_curves_physical,
    "fpt_full": _op_fpt_full,
    "fpt_full_density": _op_fpt_full_density,
    "piecewise_loglik_terms": _op_piecewise_loglik_terms,
    "piecewise_loglik": _op_piecewise_loglik,
    "piecewise_curves": _op_piecewise_curves,
    "build_segments": _op_build_segments,
}


def _serve_one(request_json, reply_npz):
    """Run one request file -> reply file. Returns None, or an error string (also written to the reply)."""
    import numpy as np
    S = _load_snapshot()
    try:
        with open(request_json) as fh:
            body = json.load(fh)
        with np.load(request_json[:-5] + ".npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        args = _decode(body["args"], arrays)
        if body["op"] not in OPS:
            raise ValueError(f"unknown op {body['op']!r}; known: {sorted(OPS)}")
        t0 = time.process_time()
        res = OPS[body["op"]](args)
        cpu = time.process_time() - t0
        res = {k: np.asarray(v) for k, v in res.items()}
        res["cpu_s"] = np.asarray(cpu)
        np.savez(reply_npz, **res)
        return None
    except Exception as ex:                                   # noqa: BLE001 -- relayed to the caller
        msg = f"{type(ex).__name__}: {ex}"
        np.savez(reply_npz, error=np.asarray(msg))
        return msg


def _worker_main(argv):
    sys.pycache_prefix = PYCACHE                              # BEFORE any import of SRK
    os.makedirs(PYCACHE, exist_ok=True)
    mode = argv[0]
    if mode == "worker":
        err = _serve_one(argv[1], argv[2])
        sys.exit(0)                                       # an op error is in the reply file
    if mode == "serve":
        proto = os.fdopen(os.dup(1), "w", buffering=1)        # keep SRK's prints off the protocol channel
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        _load_snapshot()
        proto.write(json.dumps({"ready": True, "pycache": sys.pycache_prefix, "srk": _SRK["curves"].__file__}) + "\n")
        for line in sys.stdin:
            if not line.strip():
                continue
            m = json.loads(line)
            err = _serve_one(m["request"], m["reply"])
            proto.write(json.dumps({"ok": err is None, "error": err}) + "\n")
        return
    sys.exit(f"usage: {sys.argv[0]} worker <request.json> <reply.npz> | serve")


if __name__ == "__main__":
    _worker_main(sys.argv[1:])
