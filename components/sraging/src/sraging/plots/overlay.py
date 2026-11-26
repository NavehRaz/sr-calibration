"""The data-vs-fit overlay (spec 006 T044-T045; FR-026; data-model "kappa ends"): a cohort's KM with its band and the
FP survival curve at both ends of the kappa range a fit leaves open (`diagnostics["kappa_ends"]`), each end labelled with
its kappa, the definition of the end, and whether it sits at the calibrator's kappa box edge. Never refits."""
from __future__ import annotations

import numpy as np

from sraging.errors import UnsupportedOptionError
from sraging.params import SRParams
from sraging.plots.cohort import _unit_label, km_steps
from sraging.plots.style import axes, finish, lab_style, result, sha
from sraging.population import population
from sraging.survival import km

GRID = 200                                            # ages per FP curve, from 0 to the cohort's last age


def _fmt(k):
    return f"{k:.4g}"


def _params(theta, time_unit):
    """SRParams from [eta, beta, epsilon, kappa] or [..., h_ext] (the only place a theta list is unpacked)."""
    if len(theta) not in (4, 5):
        raise UnsupportedOptionError("kappa_ends", f"a kappa end's theta has 4 or 5 entries, got {len(theta)}")
    eta, beta, eps, kappa, *h = theta
    return SRParams(eta=eta, beta=beta, epsilon=eps, kappa=kappa, h_ext=h[0] if h else 0.0, time_unit=time_unit)


def overlay(fit, cohort, *, ax=None):
    """KM steps and band of `cohort` and the exact FP survival curve at each kappa end of `fit` (a FitResult)."""
    ends = (fit.diagnostics or {}).get("kappa_ends")
    if not isinstance(ends, dict):
        raise UnsupportedOptionError("kappa_ends", "this FitResult has no diagnostics['kappa_ends'] (the key is "
                                                   "recorded by fits made since spec 006): refit with a 1.0.1+ "
                                                   "calibrator, the overlay never refits")
    k = km(cohort)
    ages = np.linspace(0.0, float(max(k.last_age, np.max(cohort.ages))), GRID)
    arrays, thetas = {}, {}
    with lab_style():
        fig, (a,) = axes(ax, figsize=(7.2, 7.0))
        x, y = km_steps(k)
        _, lo = km_steps(k, k.S_lo)
        _, hi = km_steps(k, k.S_hi)
        a.fill_between(x, lo, hi, step="post", alpha=0.25, color="C0", linewidth=0, gid="km_band")
        a.step(x, y, where="post", color="C0", linewidth=2.0, label=f"KM (n = {k.n})", gid="km")
        arrays.update(km=(x, y), km_lo=(x, lo), km_hi=(x, hi))
        for i, end in enumerate(("lo", "hi")):
            th = [float(v) for v in ends[f"theta_{end}"]]
            thetas[end] = th
            params = _params(th, cohort.time_unit)
            S = np.asarray(population(params, ages).S, float)       # the exact FP curve, as plots.curves draws it
            label = f"κ = {_fmt(ends[f'kappa_{end}'])}"
            if ends.get(f"box_limited_{end}"):
                label += " (box-limited)"
            a.plot(ages, S, color=f"C{i + 1}", linewidth=2.0, linestyle="-" if end == "lo" else "--", label=label,
                   gid=f"fp_kappa_{end}")
            arrays[f"fp_kappa_{end}"] = (ages, S)
        a.set_ylim(-0.02, 1.02)
        finish(a, xlabel=_unit_label(cohort.time_unit), ylabel="survival")
        fig.legend(*a.get_legend_handles_labels(), title=f"κ ends: {ends['definition']}", title_fontsize=15,
                   fontsize=15, frameon=False, loc="outside lower center")   # below the axes: the definition is long
    prov = {"calibrator": (fit.provenance or {}).get("calibrator"), "definition": ends["definition"],
            "theta_lo": thetas["lo"], "theta_hi": thetas["hi"],
            "box_limited": {e: bool(ends.get(f"box_limited_{e}")) for e in ("lo", "hi")}, "n": k.n,
            "data_sha256": sha(*[v for xy in arrays.values() for v in xy])}
    return result(fig, arrays, prov, band=k.band, level=k.level)
