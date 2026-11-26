"""SRK intervention-vs-control analysis (FP framework).

Reusable, organism-agnostic tools to compare intervention conditions against control(s) using the
exact Fokker-Planck likelihood and the FP posteriors:

- metrics    : analytic FP longevity (median) + inverse-CV steepness, with an Intervention spec.
- param_plane: steepness-longevity fractional-change plane; single-parameter effect lines with
               posterior-bootstrap (and analytic linearized) error bands; condition points.
- best_pair  : which SR parameter(s) changed vs control -- FP-likelihood scan and/or direct
               posterior-ratio comparison (each runnable alone).
- two_population : is a cohort one population or two? discrete SR mixture vs continuous spread,
               compared by BIC, with the survival/density figure.
- per_file   : per-condition summary (fit overlay, param ratios vs control, delta longevity/steepness).
- io         : locate a condition's FP posterior, draw baseline theta; a thin analysis config.

The current evaluator handles chronic (whole-life) interventions analytically; time-dependent
(windowed) interventions are a documented future evaluator (see metrics.apply_chronic).
"""
from .metrics import (
    Intervention, fp_metrics, data_metrics, apply_chronic, split_xc,
    PARAM_NAMES, PARAM_IDX, SWEEP_NAMES,
)
from .param_plane import (
    intervention_lines, line_band_bootstrap, line_band_linearized, condition_points,
    plot_param_plane, DEFAULT_EFFECTS,
)
from .best_pair import fp_scan, posterior_ratio, DEFAULT_SCAN_EFFECTS, DEFAULT_PAIR_EFFECTS
from .constrained_fit import (
    constrained_fit, loglik_terms, check_terms, approximation_quality,
)
from .factor_plots import series_panel, two_factor_panels, panel_grid
from .per_file import per_file_table, plot_survival_overlay, fp_survival_curve
from .two_population import (
    describe_modes, fit_one_population, fit_heterogeneous, fit_two_population,
    compare_models, verdict, plot_two_population, MIN_WEIGHT, DBIC_BAR,
)
from .io import AnalysisConfig, load_posterior, find_h5dir

__all__ = [
    # metrics
    "Intervention", "fp_metrics", "data_metrics", "apply_chronic", "split_xc",
    "PARAM_NAMES", "PARAM_IDX", "SWEEP_NAMES",
    # param plane
    "intervention_lines", "line_band_bootstrap", "line_band_linearized", "condition_points",
    "plot_param_plane", "DEFAULT_EFFECTS",
    # best pair
    "fp_scan", "posterior_ratio", "DEFAULT_SCAN_EFFECTS", "DEFAULT_PAIR_EFFECTS",
    # constrained refits / restriction quality
    "constrained_fit", "loglik_terms", "check_terms", "approximation_quality",
    # two-factor panels
    "series_panel", "two_factor_panels", "panel_grid",
    # per file
    "per_file_table", "plot_survival_overlay", "fp_survival_curve",
    # one population or two
    "describe_modes", "fit_one_population", "fit_heterogeneous", "fit_two_population",
    "compare_models", "verdict", "plot_two_population", "MIN_WEIGHT", "DBIC_BAR",
    # io
    "AnalysisConfig", "load_posterior", "find_h5dir",
]
