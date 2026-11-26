"""The seal on v0.3.3's v6 records (spec 003g research R5; tasks T007): the run's name is in no `src/` file; in the
003g tool, only the predecessor phase reads the panel's `sealed_runs`, and it refuses while [seal] is empty. A planted
read elsewhere is caught beside its twin."""
import ast
import glob
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import acceptance_003g as ACC  # noqa: E402

SEALED = "jax033" + "-v6"
ALLOWED = {"phase_predecessor"}


def sealed_reads(source: str) -> set:
    """The functions of `source` that name the sealed run or read the panel's `sealed_runs`."""
    tree = ast.parse(source)
    out = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and \
                        (SEALED in node.value or node.value == "sealed_runs"):
                    out.add(fn.name)
    return out


def test_no_src_file_names_the_sealed_run():
    hits = [p for p in glob.glob(os.path.join(REPO, "src", "**", "*.py"), recursive=True) if SEALED in open(p).read()]
    assert not hits


def test_only_the_predecessor_phase_reads_the_sealed_run():
    assert sealed_reads(open(os.path.join(REPO, "tools", "acceptance_003g.py")).read()) <= ALLOWED


def test_a_planted_read_elsewhere_is_caught_and_its_twin_is_not():
    planted = "def phase_score_w1(args):\n    return panel()['windows']['sealed_runs']\n"
    twin = "def phase_predecessor(args):\n    return panel()['windows']['sealed_runs']\n"
    assert sealed_reads(planted) == {"phase_score_w1"}
    assert sealed_reads(twin) <= ALLOWED


def test_the_predecessor_refuses_while_sealed_and_opens_with_evidence():
    with pytest.raises(ACC.Refused, match="sealed"):
        ACC.require_unsealed({"seal": {"jax033_v6": ""}})
    assert ACC.require_unsealed({"seal": {"jax033_v6": "EXECUTION_V6 l.12: GATE V6 RUN ..."}}).startswith("EXECUTION")
    # T042 (2026-10-27): unsealed by jax2's registered GATE-V6 run, its output's sha256 recorded
    assert ACC.panel()["seal"]["jax033_v6"].startswith("GATE-V6 registered run 2026-10-27")
