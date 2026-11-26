"""The NOT APPLICABLE guard: a parity session that cannot run validly never ends green (T036; constitution V)."""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_override_set_makes_parity_not_applicable_and_the_session_fail():
    env = dict(os.environ, SRK_DT_CEILING="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/parity/test_cohort_arrays.py"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0, r.stdout
    assert "NOT APPLICABLE" in r.stdout and "this session is NOT a pass" in r.stdout
    assert "SRK_DT_CEILING" in r.stdout
