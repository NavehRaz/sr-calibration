"""The jax2 1.0.1 oracle loader (spec 003b, T008; research R3; contracts/oracle-snapshots.md).

Each preflight refusal is shown with its twin (constitution V). The real load runs in a subprocess: the 1.0.0
oracle's tests import another `srbench` into this process, and the loader refuses a second one (one oracle per
process). Elsewhere than this Mac the load reports NOT APPLICABLE, never a pass.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import not_applicable

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools")
sys.path.insert(0, TOOLS)
import oracle_jax2_101 as O  # noqa: E402

REPO = O.REPO


def test_preflight_twin_is_clean_on_this_mac():
    problems = O.preflight()
    if any("host" in p or "python" in p for p in problems):
        not_applicable(problems)
    assert problems == []


def test_preflight_refuses_a_non_strict_exception_policy(monkeypatch):
    monkeypatch.setenv("SRB_STRICT", "0")
    assert any("SRB_STRICT" in p for p in O.preflight())


def test_preflight_refuses_another_thread_count(monkeypatch):
    monkeypatch.setenv("PJRT_NPROC", "4")
    assert any("PJRT_NPROC" in p for p in O.preflight())


def test_preflight_refuses_a_dt_ceiling_override(monkeypatch):
    monkeypatch.setenv("SRK_DT_CEILING", "0.001")
    assert any("SRK_DT_CEILING" in p for p in O.preflight())


def test_preflight_refuses_a_changed_method_file(tmp_path, monkeypatch):
    snap = tmp_path / "jax2"
    shutil.copytree(O.SNAP3, snap)
    p = snap / "mle_profile_jax2" / "estimator.py"
    p.write_bytes(p.read_bytes() + b"\n# planted\n")
    monkeypatch.setattr(O, "SNAP3", str(snap))
    assert any("checksum mismatch: mle_profile_jax2/estimator.py" in q for q in O.preflight())


def test_preflight_refuses_a_changed_srbench_file(tmp_path, monkeypatch):
    srb = tmp_path / "srb"
    shutil.copytree(O.SRB, srb)
    p = srb / "srbench" / "contract" / "types.py"
    p.write_bytes(p.read_bytes() + b"\n# planted\n")
    monkeypatch.setattr(O, "SRB", str(srb))
    assert any("srbench snapshot file srbench/contract/types.py" in q for q in O.preflight())


def test_the_real_oracle_loads_from_the_snapshots_in_its_own_process():
    code = ("import sys; sys.path.insert(0, %r); sys.pycache_prefix = %r\n"
            "import oracle_jax2_101 as O, json, os\n"
            "o = O.load()\n"
            "import srbench, SRK\n"
            "print(json.dumps({'srbench': os.path.realpath(srbench.__file__), 'srk': os.path.realpath(SRK.__file__),\n"
            "  'adapter': os.path.realpath(o.adapter.__file__), 'versions': o.adapter.VERSIONS,\n"
            "  'kw': O.fit_kwargs('jax2')}))\n") % (TOOLS, os.path.join(REPO, ".cache", "pycache"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    if r.returncode and "NotApplicable" in r.stderr:
        not_applicable(r.stderr[-500:])
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["srbench"].startswith(os.path.realpath(O.SRB))
    assert d["srk"].startswith(os.path.realpath(os.path.join(REPO, "tests", "oracle", "v033")))
    assert d["adapter"].startswith(os.path.realpath(O.SNAP3))
    assert d["versions"]["mle_profile_jax2"] == "1.0.1"
    assert d["kw"]["dt_rule"] is True and d["kw"]["inner_polish_j"] is True and d["kw"]["box_flag"] is False
    assert "kappa_fixed" not in d["kw"] and "combos" not in d["kw"]
