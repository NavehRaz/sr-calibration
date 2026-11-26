"""Two-factor parameter panels: how a fitted parameter moves along one experimental factor, with a
separate curve for each level of the other factor.

Built for factorial designs (e.g. C. elegans developmental x adulthood temperature): hold one factor,
sweep the other, one overlaid curve per held level, 95% error bars, and a reference line at 1 for
ratio-valued data. Organism-agnostic -- it only needs tidy rows.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Alon-lab styling: large fonts, no top/right spines.
STYLE = {"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
         "figure.dpi": 120, "savefig.bbox": "tight"}
_SERIES_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def series_panel(ax, df, x_col, series_col, value_col, lo_col=None, hi_col=None,
                 xlabel=None, ylabel=None, title=None, series_label="{}",
                 logy=True, ref_line=1.0, colors=None, marker="o", jitter=0.0,
                 anchor_col=None, legend_title=None):
    """Draw one panel: `value_col` vs `x_col`, one overlaid curve per level of `series_col`.

    lo_col/hi_col give asymmetric error bars (e.g. 95% CI bounds). `ref_line` draws a dashed
    horizontal reference (1.0 for ratios; pass None to skip). `anchor_col` marks the anchoring /
    reference point of each series with a hollow marker.
    """
    levels = sorted(df[series_col].unique())
    cols = colors or _SERIES_COLORS
    for i, lev in enumerate(levels):
        sub = df[df[series_col] == lev].sort_values(x_col)
        if sub.empty:
            continue
        c = cols[i % len(cols)]
        x = sub[x_col].to_numpy(float) + (i - (len(levels) - 1) / 2) * jitter
        y = sub[value_col].to_numpy(float)
        if lo_col and hi_col:
            yerr = np.vstack([np.maximum(y - sub[lo_col].to_numpy(float), 0),
                              np.maximum(sub[hi_col].to_numpy(float) - y, 0)])
            ax.errorbar(x, y, yerr=yerr, fmt=marker + "-", color=c, capsize=3, lw=1.8, ms=6,
                        label=series_label.format(lev))
        else:
            ax.plot(x, y, marker + "-", color=c, lw=1.8, ms=6, label=series_label.format(lev))
        if anchor_col is not None and anchor_col in sub.columns:
            m = sub[anchor_col].to_numpy(bool)
            if m.any():
                ax.plot(x[m], y[m], marker, mfc="white", mec=c, mew=2, ms=11, zorder=5)

    if ref_line is not None:
        ax.axhline(ref_line, color="k", ls="--", lw=1, alpha=0.5, zorder=0)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)
    _despine(ax)
    ax.legend(title=legend_title, fontsize=11)
    return ax


def panel_grid(specs, nrows, ncols, suptitle=None, figsize=None, **common):
    """Compose `nrows x ncols` `series_panel` specs (dicts of kwargs, each with a `df`) into a grid.

    Use for the full parameter x swept-factor matrix: rows = parameter, columns = which factor is
    swept. Returns (fig, axes_flat).
    """
    figsize = figsize or (7.5 * ncols, 5.5 * nrows)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axf = np.atleast_1d(axes).ravel()
        for ax, spec in zip(axf, specs):
            if spec is None:
                ax.axis("off")
                continue
            series_panel(ax, **{**common, **spec})
        for ax in axf[len(specs):]:
            ax.axis("off")
        if suptitle:
            fig.suptitle(suptitle, y=1.005)
        fig.tight_layout()
    return fig, axf


def two_factor_panels(panel_a, panel_b, suptitle=None, figsize=(15, 6), **common):
    """Compose two `series_panel` specs (dicts of kwargs, each with a `df`) into one 2-panel figure.

    Returns (fig, (ax_a, ax_b)).
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, spec in zip(axes, (panel_a, panel_b)):
            kw = {**common, **spec}
            series_panel(ax, **kw)
        if suptitle:
            fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
    return fig, axes
