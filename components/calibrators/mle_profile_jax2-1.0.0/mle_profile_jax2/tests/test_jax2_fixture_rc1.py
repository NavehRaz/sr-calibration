"""L0: the rc1 identity oracle (tests/fixtures/candidate_rc1/) is intact: its SHA256SUMS hold, its md5s are
the ones rc1 was mirrored to the cluster with (code/MD5SUMS.txt), and -- where the Phase A candidate tree
exists next to this checkout -- it is byte-identical to it."""
from __future__ import annotations

import hashlib
import os

import pytest

import _jax2_testlib as T

FILES = ("code/adapter.py", "code/estimator.py", "code/fp_jax.py", "cluster/driver.py")


def _h(path, algo):
    return hashlib.new(algo, open(path, "rb").read()).hexdigest()


def test_sha256sums_hold():
    lines = [ln.split() for ln in open(os.path.join(T.FIXTURE, "SHA256SUMS.txt")) if ln.strip()]
    assert len(lines) == 5
    for sha, rel in lines:
        assert _h(os.path.join(T.FIXTURE, rel), "sha256") == sha, rel


def test_md5_matches_the_rc1_mirror_record():
    want = {}
    for ln in open(os.path.join(T.FIXTURE, "code", "MD5SUMS.txt")):
        if ln.strip() and not ln.startswith("#"):
            md5, rel = ln.split(None, 1)
            want[rel.strip()] = md5
    for rel in FILES:
        assert _h(os.path.join(T.FIXTURE, rel), "md5") == want[rel], rel
    assert want["code/fp_jax.py"].startswith("5834e106")           # fix B, as the plan names it


def test_fixture_equals_the_phase_a_candidate_where_present():
    src = os.path.join(os.path.dirname(T.ROOT), "sr_calibration_tester", "reports", "basin", "candidate")
    if not os.path.isdir(src):
        pytest.skip("Phase A candidate tree not present next to this checkout")
    for rel in FILES:
        assert open(os.path.join(src, rel), "rb").read() == open(os.path.join(T.FIXTURE, rel), "rb").read(), rel
