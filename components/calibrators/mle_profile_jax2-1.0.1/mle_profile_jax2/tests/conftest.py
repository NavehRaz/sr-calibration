"""mle_profile_jax2 package tests: one shared fixture (rc2 R2).

The tester's own tests/conftest.py asserts, after every one of ITS tests, that the system under test (SRK) is not in
sys.modules -- the harness's standalone property (SC-014). When both directories run in ONE pytest process
(`pytest methods/mle_profile_jax2/tests tests`, the L0 rung's "both orders"), a package test that imports SRK
in-process (the `sut`-marked real-fit tests of test_jax2_self_consistency.py) leaves SRK behind, and every tester
test that runs after it then errors at teardown. MEASURED (rc2 R2, 1.0.0.dev4): 285 errors in that order (B5's
package tests kept SRK in subprocesses and did not hit it).

So after each package test, every SRK module is removed from sys.modules. A later test that needs SRK imports it
afresh (SRK is pure Python; the package's modules import SRK inside functions, never at their top level). This
changes no test's outcome and no runtime file (VERSION_LOCK covers only the package's top-level *.py).
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _no_srk_left_behind():
    yield
    for k in [k for k in sys.modules if k == "SRK" or k.startswith("SRK.")]:
        del sys.modules[k]
