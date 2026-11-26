"""The numerical runtime the v0.3.3 numerics require, set explicitly and checked (research R3).

The frozen oracle sets two things at import: `os.environ.setdefault("PJRT_NPROC", "1")` (fp_jax.py:62, pins
XLA's CPU client to one thread) and `jax.config.update("jax_enable_x64", True)` (fp_jax.py:69). Here they
are set explicitly -- PJRT_NPROC is OVERWRITTEN, not defaulted, so a user's shell variable cannot change the
runtime -- and `check()` refuses to fit when they cannot hold (constitution VIII: refuse rather than ignore).

PJRT_NPROC is read by the CPU client when the XLA backend initialises (on the first computation), so it is
effective if set before that; `jax._src.xla_bridge.backends_are_initialized()` (jax 0.11.1) tells whether
the backend was already up when `configure()` ran.
"""
from __future__ import annotations

import os

from sraging.errors import RuntimeConfigError

PJRT_NPROC = "1"

_STATE = {"configured": False, "backend_preinitialized": None, "pjrt_nproc_before": None}


def configure() -> None:
    """Set PJRT_NPROC=1 and float64 for JAX. Idempotent."""
    if _STATE["configured"]:
        return
    from jax._src import xla_bridge as _xb   # imports jax; the backend itself is not initialised by this

    _STATE["pjrt_nproc_before"] = os.environ.get("PJRT_NPROC")
    _STATE["backend_preinitialized"] = bool(_xb.backends_are_initialized())
    os.environ["PJRT_NPROC"] = PJRT_NPROC
    import jax

    jax.config.update("jax_enable_x64", True)
    _STATE["configured"] = True


def _x64() -> bool:
    import jax

    return bool(jax.config.read("jax_enable_x64"))


def check() -> None:
    """Raise RuntimeConfigError if the runtime cannot reproduce the v0.3.3 numerics."""
    configure()
    if not _x64():
        raise RuntimeConfigError(
            "jax_enable_x64 is off: profile_v033 needs float64 (the -700 log floor and Gtol=1e-11 "
            "require it). Do not disable it after importing sraging.")
    if _STATE["backend_preinitialized"] and _STATE["pjrt_nproc_before"] != PJRT_NPROC:
        raise RuntimeConfigError(
            f"the JAX CPU backend was initialised before sraging with PJRT_NPROC="
            f"{_STATE['pjrt_nproc_before']!r}; profile_v033 needs PJRT_NPROC=1. Import sraging before "
            f"running any JAX computation, in a fresh process.")


def state() -> dict:
    """The runtime facts recorded in provenance."""
    configure()
    return {"x64": _x64(), "pjrt_nproc": os.environ.get("PJRT_NPROC"),
            "pjrt_nproc_before": _STATE["pjrt_nproc_before"],
            "backend_preinitialized": _STATE["backend_preinitialized"]}
