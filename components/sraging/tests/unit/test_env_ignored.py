"""No environment variable changes the numerics (T026; FR-012, register D-103).

The reference reads SRK_DT_CEILING (auto grid), SRK_KAPPA_MAX and defaults PJRT_NPROC. The port reads
none of them: likelihood value and gradient bytes (and the grid) are identical with them set or unset.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")

PROBE = f"""
import sys; sys.pycache_prefix = {os.path.join(REPO, '.cache', 'pycache')!r}
import numpy as np
from sraging.cohort import Cohort
from sraging.compat.v033 import fp_jax, estimator
c = Cohort.from_csv({CSV!r}, delta=1.0)
L, kw, ceil = estimator.solver_grid(c.ages, c.events, c.delta)
f, vg = fp_jax.make_loglik(c.ages, c.events, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"],
                           Gtol=kw["Gtol"], sampling_dt=kw.get("sampling_dt"))
z = np.log(np.array([1e-3, 0.02, 2e-3, 1.0]))
v = np.asarray(f(z)); w, g = vg(z)
print(repr((float(L), kw["dt"], kw["tau_max"], ceil)))
print(v.tobytes().hex(), np.asarray(w).tobytes().hex(), np.asarray(g).tobytes().hex())
"""


def run(env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("SRK_DT_CEILING", "SRK_KAPPA_MAX", "PJRT_NPROC")}
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_overrides_do_not_change_bytes():
    clean = run({})
    noisy = run({"SRK_DT_CEILING": "1e-9", "SRK_KAPPA_MAX": "5", "PJRT_NPROC": "8"})
    assert clean == noisy
