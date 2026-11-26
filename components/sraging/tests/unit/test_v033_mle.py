"""v0.3.3's MLE stage replayed from outside compat/v033 (spec 003g research R7; tasks T026): the replay's theta-hat and
log-likelihood equal a real v0.3.3 fit's bit for bit, and compat/v033 is untouched."""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import v033_mle as M  # noqa: E402


def test_compat_v033_is_untouched():
    r = subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", "src/sraging/compat/v033"])
    assert r.returncode == 0


@pytest.mark.slow
def test_the_replay_equals_a_real_fits_mle_bit_for_bit():
    rng = np.random.default_rng(7)
    ages, ev = np.round(rng.weibull(6.0, 200) * 100.0), np.ones(200, int)
    r, f = M.mle_stage(ages, ev, 1.0), M.full_fit_theta(ages, ev, 1.0)
    assert r["status"] == "ok" and r["theta"] == f["theta"] and r["loglik"] == f["loglik"]
