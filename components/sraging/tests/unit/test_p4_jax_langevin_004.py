"""P4's JAX prototype (tasks T073; OD-7 (a)): the numba kernel's scheme in lax.scan + vmap draws the same law (a
two-sample KS against the numba kernel); the paired ratio's upper 95 % bound and the decision at 1.2; compilation
apart; the prototype is never imported by `src`. Small n; P4 itself is never run in the unit suite."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

import p4_jax_langevin_004 as P4  # noqa: E402


def test_the_ratio_and_the_od7_decision():
    r = P4.ratio_ci([1.0] * 20, [1.0] * 19 + [1.0])
    assert r["ratio_geomean"] == pytest.approx(1.0) and r["n_pairs"] == 20
    rng = np.random.default_rng(1)
    jx = np.exp(rng.normal(math.log(1.1), 0.05, 20))
    up = P4.ratio_ci(jx, np.ones(20))
    assert up["upper_95_ratio"] > up["ratio_geomean"] and up["jax_spec_opens"] == (up["upper_95_ratio"] <= 1.2)
    assert not P4.ratio_ci(np.full(20, 1.5) * np.exp(rng.normal(0, 0.01, 20)), np.ones(20))["jax_spec_opens"]


@pytest.mark.parametrize("wall", ["clamp", "reflect"])
def test_the_jax_kernel_draws_the_numba_kernels_law(wall):
    from scipy.stats import ks_2samp
    cfg, st = P4.configuration("mice_F", 3_000, dt=0.5, wall=wall)
    nsteps = int(math.floor(st["t_max"] / cfg.dt + 1e-9))
    k = P4.make_kernel(nsteps, cfg.dt, st["t_max"], wall=wall)
    tj, kj, _ = P4.jax_draw(k, st, 5)
    tn, kn, _ = P4.numba_draw(cfg, st)
    assert tj.shape == tn.shape and set(np.unique(kj)) <= {0, 1, 2}
    assert np.all(tj <= st["t_max"] + 1e-9) and np.all(tj > 0)
    assert ks_2samp(tn, tj).pvalue > 1e-3                         # the same law (3,000 each; D within noise)


def test_compare_records_pairs_compilation_apart_and_the_ks():
    out = P4.compare("mice_F", 300, repeats=3, dt=1.0)
    assert len(out["cpu_numba_s"]) == len(out["cpu_jax_s"]) == 3 and out["compile_cpu_s"] > 0
    assert set(out) >= {"upper_95_ratio", "jax_spec_opens", "two_sample_ks", "nsteps", "t_max", "xla_flags"}


def test_the_prototype_is_never_imported_by_src():
    r = subprocess.run(["git", "-C", REPO, "grep", "-l", "p4_jax_langevin", "--", "src"], capture_output=True,
                       text=True)
    assert r.stdout.strip() == ""


def test_the_phase_is_registered():
    import acceptance_004 as ACC
    assert ACC.PHASES["p4-jax"] is ACC.phase_p4_jax and ACC.PILOT_CAPS["p4-jax"] == 1.0
