"""The frozen jax2 1.0.0 oracle, loaded side by side with the port (spec 003, T003; research R3;
contracts/oracle-snapshot.md; constitution VI).

The method package comes from `tests/oracle/jax2_3bc5775/mle_profile_jax2/`. Its SRK and srbench are the
v033 snapshot's (hash-equal, cited in its MANIFEST), put on `sys.path` by `oracle_v033`'s rules. Never edits
anything; bytecode goes to `sys.pycache_prefix`.

    problems = preflight()          # [] or the reasons parity cannot be tested here (NOT APPLICABLE)
    o = load()                      # raises NotApplicable rather than load a wrong oracle
    obs = observation(cohort_id)    # J2's library v2, the library its stored 1.0.0 fits used (f1_fit.py)
    res, claims, sc = fit(obs, "jax2")   # exactly f1_fit.py's call: run_fit(**class constants), final_claim_kwargs
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_v033 as V  # noqa: E402

NotApplicable = V.NotApplicable
REPO = V.REPO
SNAP2 = os.path.join(REPO, "tests", "oracle", "jax2_3bc5775")
KEY = "srbench_methods.mle_profile_jax2"
J2 = os.environ.get(       # J2 checkout, read only: its library v2 is what the stored 1.0.0 fits read
    "SRAGING_J2_ROOT",
    os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset", "sr_calibration_tester_jax2"))   # release 0.1.1
LIBRARY_V2 = os.path.join(J2, "data", "library", "v2")
NAMES = {"jax2": "MleProfileJax2", "jax2_h": "MleProfileJax2H"}


def _sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def preflight() -> list[str]:
    """Every reason the side-by-side comparison would not be valid here (empty = valid): v033's checks
    (pycache, its snapshot, the pins, Python, the platform, numerics overrides) plus this snapshot's."""
    problems = V.preflight()
    if os.environ.get("SRB_STRICT", "1") != "1":
        problems.append(f"SRB_STRICT is {os.environ['SRB_STRICT']!r}: Phase C and the stored fits ran with SRB_STRICT=1")
    man = json.load(open(os.path.join(SNAP2, "MANIFEST.json")))
    for f in man["files"]:
        p = os.path.join(SNAP2, f["path"])
        if not os.path.exists(p):
            problems.append(f"snapshot file missing: {f['path']}")
        elif V._md5(p) != f["md5"]:
            problems.append(f"snapshot checksum mismatch: {f['path']}")
    lock_path = os.path.join(SNAP2, "mle_profile_jax2", "VERSION_LOCK.json")
    if os.path.exists(lock_path):
        lock = json.load(open(lock_path))
        for name, ent in lock["names"].items():
            for fname, h in ent["sha256"].items():
                p = os.path.join(SNAP2, "mle_profile_jax2", fname)
                if not os.path.exists(p) or _sha256(p) != h:
                    problems.append(f"VERSION_LOCK: {name} {fname} does not match the snapshot's file")
    for c in man["cited"]:
        p = os.path.join(V.SNAP, c["v033_path"])
        if not os.path.exists(p) or _sha256(p) != c["sha256"]:
            problems.append(f"cited file {c['v033_path']} differs from the hash the snapshot cites")
    return problems


class Oracle(V.Oracle):
    """Handles on the loaded jax2 oracle: SRK, the package module (`adapter`), its `run_fit` and
    `final_claim_kwargs`, the two method classes by short name, and the estimator / fp_jax siblings."""


_LOADED: Oracle | None = None


def load() -> Oracle:
    """Load the jax2 oracle into this process (once). Raises NotApplicable rather than load a wrong one."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    problems = preflight()
    if problems:
        raise NotApplicable(problems)
    os.environ["SRB_STRICT"] = "1"                   # Phase C's exception policy (phaseC/cluster/env.sh:23)
    for p in (V.SNAP_TESTER, V.SNAP_SRK_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    import SRK
    srk_root = os.path.realpath(os.path.dirname(os.path.dirname(SRK.__file__)))
    if srk_root != os.path.realpath(V.SNAP_SRK_ROOT):
        raise NotApplicable([f"SRK resolved to {srk_root}, not the v033 snapshot"])
    import srbench
    if not os.path.realpath(srbench.__file__).startswith(os.path.realpath(V.SNAP_TESTER)):
        raise NotApplicable([f"srbench resolved to {srbench.__file__}, not the v033 snapshot"])
    pkg = os.path.join(SNAP2, "mle_profile_jax2")
    ad = V._load_path(KEY, os.path.join(pkg, "__init__.py"))
    if os.path.realpath(ad.__file__) != os.path.realpath(os.path.join(pkg, "__init__.py")):
        raise NotApplicable([f"{KEY} resolved to {ad.__file__}, not the jax2 snapshot"])
    lock = ad.version_lock_problems()
    if lock:
        raise NotApplicable([f"version lock: {p}" for p in lock])
    classes = {short: getattr(ad, cls) for short, cls in NAMES.items()}
    _LOADED = Oracle(np=np, SRK=SRK, adapter=ad, run_fit=ad.run_fit, final_claim_kwargs=ad.final_claim_kwargs,
                     classes=classes, estimator=ad._sibling("estimator"), fp_jax=ad._sibling("fp_jax"))
    return _LOADED


def fit_kwargs(name: str) -> dict:
    """`run_fit`'s keywords for a registered name, exactly as `calibrate` and f1_fit.py pass them."""
    cls = load().classes[name]
    return dict(relabel=cls.RELABEL, hazard=cls.HAZARD, window=cls.WINDOW, shadow=cls.SHADOW,
                exact_from_F=cls.EXACT_F, self_consistent=cls.SELF_CONSISTENT, floor_early=cls.FLOOR_EARLY,
                n_rule=cls.N_RULE, dt_rule=cls.DT_RULE, early_all=cls.EARLY_ALL)


def fit(obs, name: str):
    """One oracle fit: (run_fit result, final claims, self-consistency block), as f1_fit.py produces them."""
    o = load()
    res = o.run_fit(obs, **fit_kwargs(name))
    claims, sc_block = o.final_claim_kwargs(res, o.classes[name].SELF_CONSISTENT)
    return res, claims, sc_block


_LIB = None


def library():
    """J2's library v2, through the v033 snapshot's (hash-equal) srbench loader, read-only."""
    global _LIB
    if _LIB is None:
        load()
        from srbench.library.load import Library
        _LIB = Library(root=LIBRARY_V2, version="v2")
    return _LIB


def observation(cohort_id):
    return library().observation(cohort_id)
