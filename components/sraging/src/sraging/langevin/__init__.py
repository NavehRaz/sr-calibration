"""sraging.langevin: the SR model's Langevin engine (spec 004 FR-010 to FR-014; contracts/python-api.md §5).

    from sraging.langevin import simulate, simulate_many, LangevinConfig
    smp = simulate(params, n=25_000, key=7, dt=0.5)          # bridge kernel, wall="clamp", error="ladder"

numba is optional (`pip install sraging[langevin]`): this package imports it only when a sample is drawn, so
`import sraging.langevin` never loads it and the FP path never does (tests/unit/langevin/test_no_numba_on_fp_path.py).
Without numba, simulate refuses, naming the extra.
"""
from __future__ import annotations

_EXPORTS = {"simulate": "sample", "LangevinConfig": "sample", "LangevinSample": "sample", "ErrorStatement": "sample",
            "grid_dt": "sample", "simulate_many": "pool"}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(f"sraging.langevin.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module 'sraging.langevin' has no attribute {name!r}")
