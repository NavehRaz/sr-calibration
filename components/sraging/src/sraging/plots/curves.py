"""FP curves from parameters (spec 005 US5; FR-019; Naveh N62/N26; C6): exact (`n=None`, the engine's curves through
`sraging.population`) or a seeded sample of n individuals with its KM and band (`n=int`, population()'s own: gen-1
inside its domain, which needs a recording delta). Never a FitResult (the overlay needs theta at both kappa-profile
ends: spec 006, [T-FIT-OVERLAY-KAPPA-ENDS])."""
from __future__ import annotations

import numpy as np

from sraging.errors import UnsupportedOptionError
from sraging.params import SRParams
from sraging.plots.style import axes, finish, lab_style, result

PANELS = ("survival", "hazard", "density")


def curves(params, ages, *, n=None, seed=None, delta=None, panels=("survival", "hazard")):
    """Panels of S (and the hazard and density for exact curves) at `ages`. With n an int, `seed` (population()'s key)
    and `delta` are required, and only "survival" can be drawn (a sample has no hazard curve)."""
    from sraging.population import population
    if not isinstance(params, SRParams):
        raise UnsupportedOptionError("params", f"curves takes an SRParams (never a FitResult: C6), got "
                                               f"{type(params).__name__}")
    panels = tuple(panels)
    bad = [p for p in panels if p not in PANELS]
    if not panels or bad:
        raise UnsupportedOptionError("panels", f"panels are some of {PANELS}, got {panels!r}")
    ages = np.asarray(ages, float)
    if n is None:
        if seed is not None or delta is not None:
            raise UnsupportedOptionError("seed" if seed is not None else "delta",
                                         "the exact curves (n=None) draw nothing: seed and delta are a sample's")
        pop = population(params, ages)
    else:
        if seed is None:
            raise UnsupportedOptionError("seed", "curves(n=int) draws a sample: pass seed (population()'s key), so the "
                                                 "figure can be redrawn; it is recorded in the provenance")
        if delta is None:
            raise UnsupportedOptionError("delta", "curves(n=int) draws a sample: pass its recording resolution delta")
        if panels != ("survival",):
            raise UnsupportedOptionError("panels", f"a sample has no hazard or density curve; draw panels=('survival',)"
                                                   f" and plot the sample's binned hazard with plots.hazard (got "
                                                   f"{panels!r})")
        pop = population(params, ages, n=n, key=seed, delta=delta)
    unit = params.time_unit if params.time_unit != "input units" else None
    xlabel = f"age ({unit})" if unit else "age"
    arrays = {}
    with lab_style():
        fig, axs = axes(None, n=len(panels))
        for a, panel in zip(axs, panels):
            if panel == "survival":
                if pop.kind == "sample":
                    lo, hi = pop.bands["S_lo"], pop.bands["S_hi"]
                    a.fill_between(ages, lo, hi, alpha=0.25, color="C0", linewidth=0, gid="band")
                    a.plot(ages, lo, color="C0", linewidth=0.8, alpha=0.6, gid="S_lo")
                    a.plot(ages, hi, color="C0", linewidth=0.8, alpha=0.6, gid="S_hi")
                    arrays.update(S_lo=(ages, lo), S_hi=(ages, hi))
                a.plot(ages, pop.S, color="C0", linewidth=2.0, gid="S")
                arrays["S"] = (ages, pop.S)
                a.set_ylim(-0.02, 1.02)
                finish(a, xlabel=xlabel, ylabel="survival")
            elif panel == "hazard":
                a.plot(ages, pop.hazard, color="C1", linewidth=2.0, gid="hazard")
                arrays["hazard"] = (ages, pop.hazard)
                finish(a, xlabel=xlabel, ylabel="hazard", log_y=bool(np.all(np.asarray(pop.hazard) > 0)))
            else:
                a.plot(ages, pop.f, color="C2", linewidth=2.0, gid="density")
                arrays["density"] = (ages, pop.f)
                finish(a, xlabel=xlabel, ylabel="density")
    prov = {"params": params.as_dict(), "n": n, "seed": seed, "delta": delta, "engine": pop.engine_used,
            "kind": pop.kind, "reason": pop.reason}
    return result(fig, arrays, prov, band=None if pop.bands is None else pop.bands.get("method"))
