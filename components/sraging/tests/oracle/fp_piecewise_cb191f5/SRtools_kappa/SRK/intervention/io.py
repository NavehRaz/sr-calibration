"""Loading FP posteriors + a thin analysis config for intervention-vs-control runs.

Reuses ``SRK.sr_mcmc.loadSamples`` + ``inv_transform`` (same convention as
``celegans_fp_fits/summarize_fp_fits.py``): read each chain, keep the best-lnprob samples, map the
transformed samples back to physical [eta, beta, epsilon, kappa] with xc fixed.
"""
from __future__ import annotations
import os
import glob
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from SRK import sr_mcmc as srmc
from .metrics import PARAM_NAMES


def find_h5dir(posterior_root, stem):
    """Most-recent ``<stem>_submit_*/h5_files`` under posterior_root, or None."""
    dirs = sorted(glob.glob(os.path.join(os.path.expanduser(posterior_root),
                                         f"{stem}_submit_*", "h5_files")))
    return dirs[-1] if dirs else None


def load_posterior(h5dir, n_per_file=800, ndims=4, xc=1.0):
    """Combine a condition's chains -> physical posterior samples (n_samples, 4).

    Keeps the best ``n_per_file`` samples per chain (by lnprob), inv_transforms to physical params
    (xc fixed), and drops non-finite / non-positive draws. Returns None if nothing loads.
    """
    files = sorted(glob.glob(os.path.join(os.path.expanduser(h5dir), "*.h5")))
    phys = []
    for f in files:
        try:
            s, l = srmc.loadSamples(f, flat=True)
        except Exception:
            continue
        if s is None or len(s) == 0:
            continue
        idx = np.argsort(l)[-n_per_file:]
        s = s[idx][:, :ndims]
        ph = np.array([srmc.inv_transform(r, set_params={"xc": xc}) for r in s])[:, :ndims]
        ok = np.all(np.isfinite(ph), axis=1) & np.all(ph > 0, axis=1)
        if ok.sum():
            phys.append(ph[ok])
    if not phys:
        return None
    return np.concatenate(phys)


@dataclass
class AnalysisConfig:
    """Points the reusable workflow at one use case.

    dataset_dir    : folder of per-condition survival CSVs (`death times`,`events`).
    posterior_root : folder holding `<stem>_submit_*/h5_files` per condition.
    conditions     : {label -> stem}. controls : list of labels that are controls.
    """
    dataset_dir: str
    posterior_root: str
    conditions: dict
    controls: Sequence[str]
    out_dir: str = "results/intervention"
    params: Sequence[str] = PARAM_NAMES
    xc: float = 1.0
    n_per_file: int = 800
    # effect grids / solver knobs (sensible defaults; override per use case)
    line_effects: Optional[Sequence[float]] = None
    N: int = 500
    n_line_draws: int = 150
    n_point_boot: int = 500

    def stem(self, label):
        return self.conditions[label]
