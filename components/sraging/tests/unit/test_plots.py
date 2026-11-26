"""Plots (spec 005 US5; FR-018 to FR-020; research R7; HR-5; tasks T027): each plotted array read back and compared with
its result object's (1e-12), a planted wrong array failing; the lab style (top and right spines off, every text >= 14
pt, no two text boxes overlapping, by the renderer's extents); curves(n=int) needs its seed, which the provenance
records; without matplotlib every call is refused, naming the `plots` extra. Agg only, no pyplot. Synthetic data."""
import dataclasses
import itertools
import sys

import numpy as np
import pytest

from sraging import SRParams
from sraging.cohort import Cohort
from sraging.cohorts import CohortSet
from sraging.errors import UnsupportedOptionError

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1, time_unit="days")
AGES = np.linspace(5.0, 150.0, 30)
RNG = np.random.default_rng(5)


def _cohort(scale=60.0, n=80, name="c"):
    a = np.round(scale * RNG.weibull(3.0, n)) + 1.0
    e = (RNG.random(n) < 0.85).astype(int)
    e[0] = 1
    return Cohort.from_arrays(a, e, delta=1.0, name=name, time_unit="days")


C = _cohort()


@pytest.fixture(scope="module")
def mpl():
    return pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def plots(mpl):
    import sraging.plots as PL
    return PL


def lines(fig):
    from matplotlib.lines import Line2D
    return {ln.get_gid(): ln for ax in fig.axes for ln in ax.get_children() if isinstance(ln, Line2D) and ln.get_gid()}


def xy(line):
    return np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)


def close(a, b, tol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.abs(a - b) <= tol) | (np.isnan(a) & np.isnan(b))))


def km_steps(k):
    """The KM's step vertices as plotted: (start, 1), each death time, carried to the last observed age."""
    x = np.concatenate([[k.start], k.times])
    y = np.concatenate([[1.0], k.S])
    if k.last_age > x[-1]:
        x, y = np.append(x, k.last_age), np.append(y, y[-1])
    return x, y


def visible_texts(fig):
    """Every drawn, non-empty text: titles, axis labels, legend texts and the tick labels inside each view."""
    fig.canvas.draw()
    out = [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:
        out += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        out += [t for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label)
                if t.get_visible() and t.get_text().strip()]
        leg = ax.get_legend()
        if leg is not None:
            out += [t for t in leg.get_texts() if t.get_text().strip()]
        for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lo, hi), max(lo, hi)
            for tick in axis.get_major_ticks():
                if tick.label1.get_visible() and tick.label1.get_text().strip() and \
                        lo - 1e-9 * abs(hi - lo) <= tick.get_loc() <= hi + 1e-9 * abs(hi - lo):
                    out.append(tick.label1)
    return out


def style_problems(fig):
    out = []
    for ax in fig.axes:
        for s in ("top", "right"):
            if ax.spines[s].get_visible():
                out.append(f"{s} spine visible")
    texts = visible_texts(fig)
    out += [f"font {t.get_fontsize()} < 14: {t.get_text()!r}" for t in texts if t.get_fontsize() < 14]
    r = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(r).expanded(0.98, 0.98)) for t in texts]
    out += [f"overlap {a!r} / {b!r}" for (a, ba), (b, bb) in itertools.combinations(boxes, 2) if ba.overlaps(bb)]
    return out


# ------------------------------------------------------------------------------------------------ read-back
def test_survival_plots_the_km_steps_and_band_and_a_planted_s_lo_fails(plots):
    from sraging.survival import km
    k = km(C)
    fig, data = plots.survival(k)
    L = lines(fig)
    want_x, want_y = km_steps(k)
    assert close(xy(L["S"])[0], want_x) and close(xy(L["S"])[1], want_y)
    lo_x, lo_y = km_steps(dataclasses.replace(k, S=k.S_lo))
    assert close(xy(L["S_lo"])[1], lo_y) and close(xy(L["S_hi"])[1], km_steps(dataclasses.replace(k, S=k.S_hi))[1])
    assert close(data["arrays"]["S"]["y"], want_y) and data["band"] == "log(-log) Greenwood"
    planted, _ = plots.survival(dataclasses.replace(k, S=k.S_lo))           # the wrong array plotted as S
    assert not close(xy(lines(planted)["S"])[1], want_y)
    fig2, _ = plots.survival(C)                                              # a cohort: its KM
    assert close(xy(lines(fig2)["S"])[1], want_y)
    assert style_problems(fig) == []


def test_hazard_and_death_times_plot_their_binned_arrays(plots):
    edges = np.arange(0.0, 130.0, 10.0)
    fig, data = plots.hazard(C, bins=edges)
    x, h = xy(lines(fig)["hazard"])
    a, e = np.asarray(C.ages), np.asarray(C.events)
    d = np.histogram(a[e == 1], bins=edges)[0]
    expo = np.array([np.sum(np.clip(np.minimum(a, b) - lo, 0.0, b - lo)) for lo, b in zip(edges[:-1], edges[1:])])
    keep = expo > 0
    assert close(x, ((edges[:-1] + edges[1:]) / 2)[keep]) and close(h, (d / np.where(keep, expo, 1.0))[keep])
    assert data["method"] == "binned" and style_problems(fig) == []
    with pytest.raises(UnsupportedOptionError, match="method"):
        plots.hazard(C, bins=edges, method="kernel")
    fig, data = plots.death_times(C, bins=edges)
    from matplotlib.patches import StepPatch
    (sp,) = [p for p in fig.axes[0].patches if isinstance(p, StepPatch)]
    vals, ed, _ = sp.get_data()
    assert close(ed, edges) and close(vals, d) and close(data["arrays"]["death_times"]["y"], d)
    assert style_problems(fig) == []


def test_set_survival_plots_each_members_km(plots):
    from sraging.survival import km
    s = CohortSet.from_members([(f"g/{i}", _cohort(50 + 10 * i, name=f"g/{i}"), {"cohort": "g", "cage": i})
                                for i in range(3)], time_unit="days")
    fig, data = plots.set_survival(s)
    L = lines(fig)
    assert set(L) >= set(s)
    for key in s:
        assert close(xy(L[key])[1], km_steps(km(s[key]))[1])
    assert style_problems(fig) == []


def test_curves_exact_are_the_engines_and_a_sample_is_its_km_with_the_seed_recorded(plots):
    from sraging.population import population
    fig, data = plots.curves(P, AGES)
    L, pop = lines(fig), population(P, AGES)
    assert close(xy(L["S"])[0], AGES) and close(xy(L["S"])[1], pop.S) and close(xy(L["hazard"])[1], pop.hazard)
    assert data["provenance"]["n"] is None and data["provenance"]["engine"] == "fp" and style_problems(fig) == []
    with pytest.raises(UnsupportedOptionError, match="seed"):
        plots.curves(P, AGES, n=300, delta=1.0)
    with pytest.raises(UnsupportedOptionError, match="delta"):
        plots.curves(P, AGES, n=300, seed=7)
    fig, data = plots.curves(P, AGES, n=300, seed=7, delta=1.0, panels=("survival",))
    samp = population(P, AGES, n=300, key=7, delta=1.0)
    L = lines(fig)
    assert close(xy(L["S"])[1], samp.S) and close(xy(L["S_lo"])[1], samp.bands["S_lo"])
    assert data["provenance"]["seed"] == 7 and data["provenance"]["n"] == 300
    assert not close(xy(L["S"])[1], pop.S)                         # planted: the FP curve is not the sample's KM
    with pytest.raises(UnsupportedOptionError, match="panels"):
        plots.curves(P, AGES, n=300, seed=7, delta=1.0, panels=("hazard",))
    assert style_problems(fig) == []


def _cleaning_report():
    from sraging.cleaning import clean
    rng = np.random.default_rng(3)
    members = []
    for i in range(6):
        scale = 60.0 if i < 5 else 90.0
        a = np.round(scale * (1 + 0.2 * rng.standard_normal(40)).clip(0.3)) + 1.0
        members.append((f"x/{i}", Cohort.from_arrays(a, np.ones(40, int), delta=1.0, time_unit="days"),
                        {"cohort": "x", "cage": i}))
    s = CohortSet.from_members(members, time_unit="days")
    return s, clean(s, by="cohort", k=2.5, floors=(0.05, 0.1), seed=1, noise_reps=100)


def test_cleaning_plots_the_cages_and_the_ellipse_each_was_judged_by(plots):
    s, rep = _cleaning_report()
    fig, data = plots.cleaning(rep, cohort="x")
    L = lines(fig)
    rows = [r for r in rep.cages if r.cohort == "x"]
    assert close(xy(L["cages"])[0], [r.median for r in rows]) and close(xy(L["cages"])[1], [r.steepness for r in rows])
    th = np.linspace(0.0, 2 * np.pi, 181)
    r0 = rows[0]
    ex, ey = xy(L["ellipse:0"])
    assert close(ex, r0.centre_ml + r0.radius_ml * np.cos(th)) and close(ey, r0.centre_st + r0.radius_st * np.sin(th))
    flagged = [r for r in rows if r.verdict == "flagged"]
    assert close(xy(L["flagged"])[0], [r.median for r in flagged]) and data["provenance"]["report_sha256"] == rep.sha256
    assert style_problems(fig) == []
    with pytest.raises(UnsupportedOptionError, match="'y'"):
        plots.cleaning(rep, cohort="y")


def test_k_sweep_counts_the_flags_at_each_k(plots):
    from sraging import cleaning as CL
    s, _ = _cleaning_report()
    ks = [1.5, 2.0, 2.5, 3.0]
    fig, data = plots.k_sweep(s, by="cohort", ks=ks, floors=(0.05, 0.1))
    sc = CL.cage_scores(s)
    ml, st = [sc[k][0] for k in s], [sc[k][1] for k in s]
    want = [int(CL.judge(ml, st, k=k, floors=(0.05, 0.1))["flagged"].sum()) for k in ks]
    assert close(xy(lines(fig)["flagged"])[0], ks) and close(xy(lines(fig)["flagged"])[1], want)
    assert style_problems(fig) == []


def test_an_existing_axes_is_used_and_styled(plots, mpl):
    from matplotlib.figure import Figure
    f = Figure()
    ax = f.add_subplot()
    fig, _ = plots.survival(C, ax=ax)
    assert fig is f and not ax.spines["top"].get_visible()


def test_the_style_check_catches_an_overlap_a_small_font_and_a_top_spine(plots):
    """The planted twins of the style checks: each fault is found."""
    fig, _ = plots.survival(C)
    ax = fig.axes[0]
    ax.xaxis.label.set_fontsize(10)
    assert any("font 10" in p for p in style_problems(fig))
    fig, _ = plots.survival(C)
    ax = fig.axes[0]
    ax.text(0.5, 0.5, "label A", transform=ax.transAxes, fontsize=16)
    ax.text(0.5, 0.5, "label B", transform=ax.transAxes, fontsize=16)
    assert any("overlap" in p for p in style_problems(fig))
    fig, _ = plots.survival(C)
    fig.axes[0].spines["top"].set_visible(True)
    assert style_problems(fig) == ["top spine visible"]


# ------------------------------------------------------------------------------------------- the refusal
def test_without_matplotlib_every_call_is_refused_naming_the_extra(monkeypatch):
    import sraging.plots as PL
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    for name in ("survival", "hazard", "death_times", "set_survival", "curves", "cleaning", "k_sweep"):
        with pytest.raises(UnsupportedOptionError, match="plots") as ex:
            getattr(PL, name)(C)
        assert "sraging[plots]" in str(ex.value) and ex.value.option == "plots"
