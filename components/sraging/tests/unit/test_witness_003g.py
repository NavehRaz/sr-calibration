"""Reading jax2's witness files for PR-1 (spec 003g research R3; tasks T008): a file is used only when it is the pinned
file, of the right schema, status, name, cohort and record key; W is W_REF on a human cohort and W4 otherwise. Each
planted bad file is "not scored" beside its accepted twin."""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import witness_003g as W  # noqa: E402

NAME = "mle_profile_jax2"
PURE = "pure-k0.01-p01-c0-s0"
HUMAN = "pure-real-Sweden_F_1910_homo-c1-s0"


def write(tmp_path, cid, **over):
    w = {"schema": W.SCHEMA, "name": NAME, "cohort": cid, "status": "ok", "key_digest": "k" * 64,
         "informative": True, "human": W.is_human(cid), "W4": 0.4, "W_REF": 0.6 if W.is_human(cid) else None,
         "new_fails": False, "theta": [1, 2, 3, 4], "cpu_model": "x"}
    w.update(over)
    p = tmp_path / f"{cid}.json"
    p.write_text(json.dumps(w))
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def read(p, sha, cid, key="k" * 64):
    return W.read_witness(p, name=NAME, cohort_id=cid, pinned_sha256=sha, pinned_key=key)


def test_the_accepted_twin_reads_w4_and_w_ref_on_a_human_cohort(tmp_path):
    p, sha = write(tmp_path, PURE)
    r = read(p, sha, PURE)
    assert r["fresh"] and r["W"] == 0.4 and r["source"] == "jax2"
    p, sha = write(tmp_path, HUMAN)
    assert read(p, sha, HUMAN)["W"] == 0.6


def test_each_planted_bad_file_is_not_scored(tmp_path):
    p, sha = write(tmp_path, PURE)
    assert read(p, "0" * 64, PURE)["reason"].startswith("witness sha256")          # not the pinned bytes
    assert read(p, None, PURE)["reason"] == "witness file not pinned"
    assert read(p, sha, PURE, key="j" * 64)["reason"].startswith("key_digest")     # another record's witness
    for over, why in ((dict(schema="x/0"), "schema"), (dict(status="stale"), "status"),
                      (dict(cohort="other"), "cohort"), (dict(informative=False), "not informative"),
                      (dict(new_fails=True), "new_fails")):
        p, sha = write(tmp_path, PURE, **over)
        r = read(p, sha, PURE)
        assert not r["fresh"] and r["W"] is None and why in r["reason"], (over, r)
    p, sha = write(tmp_path, HUMAN, W_REF=None)
    assert read(p, sha, HUMAN)["reason"] == "W_REF missing on a human cohort"
    assert read(str(tmp_path / "absent.json"), "0" * 64, PURE)["reason"] == "no witness file"


def test_jax2s_export_is_pinned_only_through_its_sha256sums(tmp_path):
    """sesync cd40: export_v6/<W>/witness/ with a SHA256SUMS whose sha256 comes in the WINDOW ENDS message."""
    import window_003g as WIN
    d = tmp_path / "witness" / NAME
    d.mkdir(parents=True)
    (d / f"{PURE}.json").write_text("{}")
    fsha = hashlib.sha256(b"{}").hexdigest()
    sums = tmp_path / "witness" / "SHA256SUMS"
    sums.write_text(f"{fsha}  {NAME}/{PURE}.json\n")
    ssha = hashlib.sha256(sums.read_bytes()).hexdigest()
    got = WIN.export_witness(str(tmp_path / "witness"), ssha, (NAME, "mle_profile_jax2_h"))
    assert got == {NAME: {PURE: fsha}}                                                     # twin
    import pytest
    with pytest.raises(ValueError, match="not the one jax2 sent"):
        WIN.export_witness(str(tmp_path / "witness"), "0" * 64, (NAME,))
    (d / f"{PURE}.json").write_text('{"x": 1}')                                           # bytes changed after
    with pytest.raises(ValueError, match="differ from SHA256SUMS"):
        WIN.export_witness(str(tmp_path / "witness"), ssha, (NAME,))
