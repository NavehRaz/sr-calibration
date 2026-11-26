"""A real fit on the small synthetic cohort (T027; FR-001, FR-004, FR-009, SC-009, SC-010). Slow: opt-in.

    .venv/bin/pytest -m slow tests/unit/test_fit_small.py

Runs one fit through the Python API and one through the CLI, each in a subprocess with PYTHONPATH
cleared, and checks: every field present; the two results identical apart from timing/created_at
(determinism and CLI == Python); no SRK/srbench module was imported.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")
SRAGING = os.path.join(os.path.dirname(sys.executable), "sraging")

PY = f"""
import sys, json; sys.pycache_prefix = {os.path.join(REPO, '.cache', 'pycache')!r}
import sraging
r = sraging.fit_cohort(sraging.Cohort.from_csv({CSV!r}, delta=1.0), calibrator="profile_v033")
leaked = sorted(m for m in sys.modules if m.split('.')[0] in ('SRK', 'srbench', 'srbench_methods'))
print(json.dumps({{"result": json.loads(r.to_json()), "leaked": leaked}}))
"""


def clean_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SRBENCH_SUT")}
    return env


def strip(d):
    """Drop the timing fields (FR-009): `timing`, the reference's diagnostic timings, created_at."""
    d = json.loads(json.dumps(d))
    d.pop("timing", None)
    for k in ("compile_s", "seed_s", "mle_s"):
        d["diagnostics"].pop(k, None)
    d["provenance"].pop("created_at", None)
    return d


@pytest.mark.slow
def test_small_fit_python_and_cli(tmp_path):
    r1 = subprocess.run([sys.executable, "-c", PY], capture_output=True, text=True, env=clean_env(),
                        cwd=tmp_path, timeout=3600)
    assert r1.returncode == 0, r1.stderr
    out = json.loads(r1.stdout.strip().splitlines()[-1])
    py = out["result"]
    assert out["leaked"] == []                                          # SC-010
    assert py["status"] == "ok"
    th = py["theta_hat"]
    assert all(isinstance(th[p], float) for p in ("eta", "beta", "epsilon", "kappa"))
    assert th["xc"] == 1.0 and th["xc_fixed"] is True
    assert [c["parameter"] for c in py["claims"]] == ["eta", "beta", "epsilon", "kappa"]
    assert [c["code"] for c in py["checks"]] == ["BASIN_GAIN", "BASIN_OUTSIDE", "CENSOR_TIE", "DELTA_GRID", "BOX_EDGE",
                                               "FLOORED_DEATHS", "AGE0_DEATHS"]
    assert py["provenance"]["numerics"] == "v033" and py["provenance"]["engine"] == "fp"

    js = tmp_path / "cli.json"
    r2 = subprocess.run([SRAGING, "fit-cohort", "--csv", CSV, "--delta", "1", "--calibrator", "profile_v033", "--json",
                         str(js)],
                        capture_output=True, text=True, env=clean_env(), cwd=tmp_path, timeout=3600)
    assert r2.returncode == 0, r2.stderr
    cli = json.loads(js.read_text())
    from parity_checks import tree_diff
    assert tree_diff(strip(cli), strip(py)) == []                       # FR-001, FR-009, SC-009
