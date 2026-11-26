"""jax2's G2 witness outputs, read for spec 003g's PR-1 (research R3; tasks T012). Read-only.

jax2 writes one file per (name, cohort) on the fitting host, schema `gate_v6_witness/1`, mirrored by its `pull` to
GATEV6/cluster_home/witness/<name>/<cid>.json (never merged into the tester store). W = W_REF on a human cohort (ids
containing -real-Sweden_, -real-Denmark_, -real-combined_human_), W4 = l(truth) - l(theta-hat) otherwise; a cohort is
witnessed only when informative (het_var == 0). A file is used only when its sha256 is the pinned one, its schema,
status, name and cohort are right, and its key_digest is the pinned run's key for that cohort; otherwise the cohort is
"not scored", with the reason, and its stratum cannot pass (PR-1: incomplete).
"""
from __future__ import annotations

import hashlib
import json
import os

SCHEMA = "gate_v6_witness/1"
HUMAN = ("-real-Sweden_", "-real-Denmark_", "-real-combined_human_")
W_THR = 1.92


def is_human(cohort_id: str) -> bool:
    return any(h in cohort_id for h in HUMAN)


def witness_path(root: str, name: str, cohort_id: str) -> str:
    return os.path.join(root, name, f"{cohort_id}.json")


def not_scored(reason: str, **extra) -> dict:
    return {"W": None, "fresh": False, "source": None, "reason": reason, **extra}


def read_witness(path: str, *, name: str, cohort_id: str, pinned_sha256: str | None, pinned_key: str | None) -> dict:
    """One cohort's witness row for PR-1: {W, fresh, source, reason, W4, W_REF, human, theta, cpu_model,
    file_sha256}. Never raises on a bad file: the cohort is "not scored" and says why."""
    if not os.path.exists(path):
        return not_scored("no witness file")
    b = open(path, "rb").read()
    sha = hashlib.sha256(b).hexdigest()
    if pinned_sha256 is None:
        return not_scored("witness file not pinned", file_sha256=sha)
    if sha != pinned_sha256:
        return not_scored(f"witness sha256 {sha[:12]} is not the pinned {pinned_sha256[:12]}", file_sha256=sha)
    try:
        w = json.loads(b)
    except ValueError as ex:
        return not_scored(f"unreadable: {ex}", file_sha256=sha)
    for k, want in (("schema", SCHEMA), ("name", name), ("cohort", cohort_id)):
        if w.get(k) != want:
            return not_scored(f"{k} {w.get(k)!r}, expected {want!r}", file_sha256=sha)
    if w.get("status") != "ok":
        return not_scored(f"status {w.get('status')!r}", file_sha256=sha)
    if pinned_key is None or w.get("key_digest") != pinned_key:
        return not_scored("key_digest is not the pinned record's key", file_sha256=sha)
    if not w.get("informative"):
        return not_scored("not informative (het_var != 0): W is undefined", file_sha256=sha, informative=False)
    human = is_human(cohort_id)
    if bool(w.get("human")) != human:
        return not_scored(f"human flag {w.get('human')!r} disagrees with the id", file_sha256=sha)
    W = w.get("W_REF") if human else w.get("W4")
    if W is None:
        return not_scored("W_REF missing on a human cohort" if human else "W4 missing", file_sha256=sha)
    W = float(W)
    if w.get("new_fails") is not None and bool(w["new_fails"]) != (W > W_THR):
        return not_scored(f"new_fails {w['new_fails']!r} disagrees with W {W:.4g}", file_sha256=sha)
    return {"W": W, "fresh": True, "source": "jax2", "reason": None, "W4": w.get("W4"), "W_REF": w.get("W_REF"),
            "human": human, "theta": w.get("theta"), "cpu_model": w.get("cpu_model"), "file_sha256": sha,
            "dt_in_scope": w.get("dt_in_scope"), "cv_c": w.get("cv_c")}
