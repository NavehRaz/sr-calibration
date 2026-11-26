"""Shared helpers for the mle_profile_jax2 package tests (not a test module; stdlib only at import).

The package's __init__.py REGISTERS two methods when discovery loads it (key `srbench_methods.<dir>`), and
srbench's `discover()` re-executes every adapter and refuses a second registration of a name. So these
tests never discover in the pytest process: anything that needs the registered adapter runs in a
SUBPROCESS (`run_py`). (pytest itself imports __init__.py as the plain package `mle_profile_jax2` to
collect these tests; under that name it registers nothing -- see the guard at its end.) The non-adapter
modules (fp_jax, estimator, observation_model) and the rc1 fixture load in-process under test-only keys.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(TESTS)
METHODS = os.path.dirname(PKG)
ROOT = os.path.dirname(METHODS)                       # the tester checkout this package lives in
FIXTURE = os.path.join(TESTS, "fixtures", "candidate_rc1")
OLD_PKG = os.path.join(METHODS, "mle_profile_jax")    # the FROZEN v0.3.3 adapter (never edited)


def srk_path():
    """The SUT checkout to put on PYTHONPATH: SRBENCH_SUT if set, else the SRK package's parent as
    already importable here (via PYTHONPATH), else None."""
    p = os.environ.get("SRBENCH_SUT")
    if p:
        return p
    for e in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if e and os.path.isdir(os.path.join(e, "SRK")):
            return e
    return None


def child_env(**extra):
    env = dict(os.environ)
    pp = [ROOT] + ([srk_path()] if srk_path() else [])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.setdefault("PJRT_NPROC", "1")
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    env.setdefault("SRK_CACHE_READONLY", "1")
    env["SRB_STRICT"] = "1"
    env.pop("SRK_DT_CEILING", None)
    for k, v in extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def run_py(code, timeout=600, **env):
    """Run `code` in a fresh interpreter (cwd = ROOT). The child prints ONE JSON line last; return it
    parsed, with the child's returncode/stderr under '_rc'/'_stderr'."""
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env(**env), capture_output=True,
                       text=True, timeout=timeout)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    out = {}
    if lines:
        try:
            out = json.loads(lines[-1])
        except json.JSONDecodeError:
            out = {"_stdout": r.stdout[-4000:]}
    out["_rc"] = r.returncode
    out["_stderr"] = r.stderr[-4000:]
    return out


def load(key, path):
    """Load a module by path under a test-only sys.modules key (cached)."""
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return mod


def pkg_module(name):
    """A NON-adapter module of the package (fp_jax / estimator / observation_model), in-process."""
    assert name != "__init__", "the adapter registers methods: load it in a subprocess (run_py)"
    return load(f"jax2_tests.pkg.{name}", os.path.join(PKG, f"{name}.py"))


def rc1_module(name):
    """A module of the frozen rc1 fixture (code/adapter.py is NOT registered, so it is safe in-process)."""
    return load(f"jax2_tests.rc1.{name}", os.path.join(FIXTURE, "code", f"{name}.py"))


def jax_env():
    """Single-threaded XLA, as the method runs; must happen before the first JAX import."""
    os.environ.setdefault("PJRT_NPROC", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    os.environ.setdefault("SRK_CACHE_READONLY", "1")
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
