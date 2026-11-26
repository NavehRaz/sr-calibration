"""The lab's plot style (spec 005 R7, FR-020; the alon-lab-plots skill: large fonts that do not overlap, no top or
right spines), applied per figure and never globally (constitution VIII).

Figures are `matplotlib.figure.Figure` objects on the Agg canvas, built inside `lab_style()`; pyplot's global state is
not touched. Every function in sraging.plots returns `(fig, data)`, `data` holding the plotted arrays by artist name,
the style and the inputs' provenance (`result`).
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np

FONT, LABEL, LEGEND, DPI = 16, 17, 15, 140
MIN_FONT = 14                     # the smallest text a figure may carry (tests/unit/test_plots.py)
STYLE = {"font.size": FONT, "axes.labelsize": LABEL, "axes.titlesize": LABEL, "xtick.labelsize": FONT,
         "ytick.labelsize": FONT, "legend.fontsize": LEGEND, "axes.spines.top": False,
         "axes.spines.right": False, "figure.dpi": DPI, "savefig.dpi": DPI}
STYLE_VERSION = (f"sraging lab style 1 (R7): fonts {FONT}, labels {LABEL}, legend {LEGEND}, no top/right spines, "
                 f"dpi {DPI}")


@contextmanager
def lab_style():
    """matplotlib's rc parameters set to STYLE inside the block only."""
    import matplotlib
    with matplotlib.rc_context(STYLE):
        yield


def axes(ax=None, *, n=1, figsize=None):
    """(fig, [axes]): `ax` and its figure, or a new Agg figure with `n` panels side by side (constrained layout)."""
    if ax is not None:
        return ax.figure, [ax]
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize or (6.4 * n, 4.8), dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    return fig, [fig.add_subplot(1, n, i + 1) for i in range(n)]


def finish(ax, *, xlabel=None, ylabel=None, legend=False, log_y=False):
    """The style on one axes, explicitly (it holds outside lab_style too): spines, labels, tick sizes and density."""
    from matplotlib.ticker import MaxNLocator
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=FONT)
    if legend:
        ax.legend(fontsize=LEGEND, frameon=False)


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    return h.hexdigest()


def result(fig, arrays: dict, provenance: dict, **extra):
    """(fig, data): data = {arrays: {name: {x, y}}, style, provenance, **extra}."""
    import matplotlib
    import sraging
    data = {"arrays": {k: {"x": np.asarray(v[0], float), "y": np.asarray(v[1], float)} for k, v in arrays.items()},
            "style": STYLE_VERSION,
            "provenance": {**provenance, "sraging": sraging.__version__, "matplotlib": matplotlib.__version__},
            **extra}
    return fig, data
