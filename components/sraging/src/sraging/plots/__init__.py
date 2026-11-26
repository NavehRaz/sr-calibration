"""Plots in the lab's style (spec 005 US5; FR-018 to FR-020; research R7). Needs the `plots` extra
(`pip install 'sraging[plots]'`, matplotlib 3.10.8); without it every call is refused, naming the extra.

    fig, data = sraging.plots.survival(km_result)       # data["arrays"]: what was plotted; data["provenance"]

survival, hazard, death_times, set_survival (plots/cohort.py); curves (plots/curves.py); cleaning, k_sweep
(plots/cleaning.py); overlay (plots/overlay.py: a fit's data against its FP curves at both kappa ends); the style (plots/style.py: a context manager, never global).
"""
from __future__ import annotations

import importlib

from sraging.errors import UnsupportedOptionError


def _require():
    try:
        importlib.import_module("matplotlib")
    except ImportError:
        raise UnsupportedOptionError("plots", "sraging.plots needs the plots extra: pip install 'sraging[plots]' "
                                              "(matplotlib 3.10.8)") from None


def _call(module, name, args, kw):
    _require()
    fn = getattr(importlib.import_module(f"sraging.plots.{module}"), name)
    globals().update(_PUBLIC)          # importing plots.curves / plots.cleaning binds the module over the function
    return fn(*args, **kw)


def survival(*args, **kw):
    """plots.cohort.survival(km_or_cohort, *, ax=None) -> (fig, data)."""
    return _call("cohort", "survival", args, kw)


def hazard(*args, **kw):
    """plots.cohort.hazard(cohort, *, method="binned", bins, ax=None) -> (fig, data)."""
    return _call("cohort", "hazard", args, kw)


def death_times(*args, **kw):
    """plots.cohort.death_times(cohort, *, bins, ax=None) -> (fig, data)."""
    return _call("cohort", "death_times", args, kw)


def set_survival(*args, **kw):
    """plots.cohort.set_survival(cohort_set, *, ax=None) -> (fig, data)."""
    return _call("cohort", "set_survival", args, kw)


def curves(*args, **kw):
    """plots.curves.curves(params, ages, *, n=None, seed=None, delta=None, panels=("survival", "hazard"))."""
    return _call("curves", "curves", args, kw)


def cleaning(*args, **kw):
    """plots.cleaning.cleaning(report, *, cohort, ax=None) -> (fig, data)."""
    return _call("cleaning", "cleaning", args, kw)


def k_sweep(*args, **kw):
    """plots.cleaning.k_sweep(cage_set, *, by, ks, floors, ax=None) -> (fig, data)."""
    return _call("cleaning", "k_sweep", args, kw)


def overlay(*args, **kw):
    """plots.overlay.overlay(result, cohort, *, ax=None) -> (fig, data)."""
    return _call("overlay", "overlay", args, kw)


_PUBLIC = {f.__name__: f for f in (survival, hazard, death_times, set_survival, curves, cleaning, k_sweep, overlay)}
