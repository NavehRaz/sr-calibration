"""Explicit time units (spec 005 US4; FR-007 to FR-009; research R6; data-model "Unit table"; Naveh's Dc).

    p_weeks = convert(p_days, to="weeks")        # eta * 49, beta * 7, epsilon * 7, h_ext * 7; kappa, xc unchanged
    c_weeks = convert(cohort_days, to="weeks")   # ages, delta, entry, window / 7, rebuilt by Cohort.from_arrays
    c = with_unit(cohort, "days")                # labels an "input units" object; never converts

UNITS is the closed list of days per unit: hours 1/24, days 1, weeks 7, years 365, generations 1/8 (the factor table
SRK's getTheta uses, years 365 and generations 3/24). Any other unit, months included, is refused; nothing converts
unless `convert` is called. With s = UNITS[to] / UNITS[from], a rate scales by s and eta (time^-2) by s^2, so the
dimensionless groups at L are unchanged when L is converted with the ages (L / s).
"""
from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np

from sraging.cohort import Cohort
from sraging.errors import UnsupportedOptionError
from sraging.params import SRParams

UNITS = MappingProxyType({"hours": 1 / 24, "days": 1.0, "weeks": 7.0, "years": 365.0, "generations": 1 / 8})
INPUT_UNITS = "input units"


def _unit(unit, option):
    if not isinstance(unit, str) or unit not in UNITS:
        raise UnsupportedOptionError(option, f"{option} {unit!r} is not a unit sraging converts; the units are "
                                             f"{list(UNITS)} (days per unit {dict(UNITS)}); months are refused")
    return unit


def _source(obj):
    u = obj.time_unit
    if u == INPUT_UNITS:
        raise UnsupportedOptionError("time_unit", f"the object's time unit is {INPUT_UNITS!r}: name it first with "
                                                  f"with_unit(obj, unit), then convert")
    return _unit(u, "time_unit")


def factor(frm, to) -> float:
    """s = UNITS[to] / UNITS[frm]: a rate in `frm` units times s is the rate in `to` units."""
    return UNITS[_unit(to, "to")] / UNITS[_unit(frm, "from")]


def convert(obj, *, to):
    """`obj` (SRParams or Cohort) in unit `to`, a new object built by its own constructor (FR-008)."""
    _unit(to, "to")
    if not isinstance(obj, (SRParams, Cohort)):
        raise TypeError(f"convert takes an SRParams or Cohort, got {type(obj).__name__}")
    frm = _source(obj)
    d_from, d_to = UNITS[frm], UNITS[to]
    if isinstance(obj, SRParams):
        return SRParams(eta=obj.eta * d_to ** 2 / d_from ** 2, beta=obj.beta * d_to / d_from,
                        epsilon=obj.epsilon * d_to / d_from, kappa=obj.kappa, xc=obj.xc,
                        h_ext=obj.h_ext * d_to / d_from, time_unit=to)

    def age(x):
        return None if x is None else np.asarray(x, float) * d_from / d_to
    delta = obj.delta if obj.delta is None else (age(obj.delta) if np.ndim(obj.delta) == 1
                                                 else float(obj.delta) * d_from / d_to)
    window = None if obj.window is None else tuple(float(w) * d_from / d_to for w in obj.window)
    return Cohort.from_arrays(age(obj.ages), np.asarray(obj.events), delta=delta, continuous=obj.continuous,
                              name=obj.name, entry=age(obj.entry), window=window, time_unit=to)


def with_unit(obj, unit):
    """`obj` labelled with `unit`, unconverted; only from "input units" (an object with a unit is converted)."""
    _unit(unit, "unit")
    if obj.time_unit != INPUT_UNITS:
        raise UnsupportedOptionError("unit", f"the object is already in {obj.time_unit!r}; use convert(obj, "
                                             f"to={unit!r}) to change units")
    if isinstance(obj, SRParams):
        return dataclasses.replace(obj, time_unit=unit)
    if isinstance(obj, Cohort):
        return Cohort.from_arrays(obj.ages, obj.events, delta=obj.delta, continuous=obj.continuous, name=obj.name,
                                  entry=obj.entry, window=obj.window, time_unit=unit)
    raise TypeError(f"with_unit takes an SRParams or Cohort, got {type(obj).__name__}")
