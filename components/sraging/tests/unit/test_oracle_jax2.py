"""The jax2 oracle loader (spec 003, T003; research R3; contracts/oracle-snapshot.md).

Each preflight refusal is shown (constitution V). Loading the real snapshot needs this Mac and the pinned
environment; elsewhere the load test reports NOT APPLICABLE, never a pass.
"""
import json
import os
import shutil
import sys

import pytest

from conftest import not_applicable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
import oracle_jax2 as O  # noqa: E402


def test_preflight_refuses_a_dt_ceiling_override(monkeypatch):
    monkeypatch.setenv("SRK_DT_CEILING", "0.001")
    assert any("SRK_DT_CEILING" in p for p in O.preflight())


def test_preflight_refuses_a_non_strict_exception_policy(monkeypatch):
    monkeypatch.setenv("SRB_STRICT", "0")
    assert any("SRB_STRICT" in p for p in O.preflight())


def test_preflight_refuses_a_changed_snapshot_file(tmp_path, monkeypatch):
    snap = tmp_path / "jax2"
    shutil.copytree(O.SNAP2, snap)
    p = snap / "mle_profile_jax2" / "fp_jax.py"
    p.write_bytes(p.read_bytes() + b"\n# planted\n")
    monkeypatch.setattr(O, "SNAP2", str(snap))
    assert any("checksum mismatch: mle_profile_jax2/fp_jax.py" in q for q in O.preflight())


def test_preflight_refuses_a_lock_that_no_longer_matches(tmp_path, monkeypatch):
    snap = tmp_path / "jax2"
    shutil.copytree(O.SNAP2, snap)
    lock = json.load(open(snap / "mle_profile_jax2" / "VERSION_LOCK.json"))
    lock["names"]["mle_profile_jax2"]["sha256"]["estimator.py"] = "0" * 64
    json.dump(lock, open(snap / "mle_profile_jax2" / "VERSION_LOCK.json", "w"))
    man = json.load(open(snap / "MANIFEST.json"))                 # keep the md5 chain consistent: only the lock is off
    import hashlib
    for f in man["files"]:
        f["md5"] = hashlib.md5(open(snap / f["path"], "rb").read()).hexdigest()
    json.dump(man, open(snap / "MANIFEST.json", "w"))
    monkeypatch.setattr(O, "SNAP2", str(snap))
    assert any("VERSION_LOCK" in q for q in O.preflight())


def test_the_snapshot_loads_with_both_names_and_its_lock_passes():
    problems = O.preflight()
    if problems:
        not_applicable("; ".join(problems))
    o = O.load()
    assert o.adapter.version_lock_problems() == []
    assert set(o.classes) == {"jax2", "jax2_h"}
    assert o.classes["jax2_h"].HAZARD is True and o.classes["jax2"].HAZARD is False
    assert os.environ.get("SRB_STRICT") == "1"
    assert callable(o.run_fit) and callable(o.final_claim_kwargs)
    srk = os.path.realpath(os.path.dirname(o.SRK.__file__))
    assert srk.startswith(os.path.realpath(O.V.SNAP_SRK_ROOT))
