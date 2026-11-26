"""The frozen jax2 1.0.1 oracle, loaded side by side with the port (spec 003b, T008; research R3;
contracts/oracle-snapshots.md; constitution VI).

The method package comes from `tests/oracle/jax2_6884fc8/mle_profile_jax2/`; its srbench from
`tests/oracle/tester_f94ec2b/` (the tree 1.0.1 runs against: 7 of the 13 srbench files a jax2 fit loads differ from
the v033 snapshot's, so the 1.0.0 oracle's borrowing does not hold here); its SRK from `tests/oracle/v033` (the 18
files a 1.0.1 fit loads, cited and sha256-equal). Bytecode goes to `sys.pycache_prefix`; nothing is edited.

One oracle per process: this one and `tools/oracle_jax2.py` (1.0.0) both import a module named `srbench` from
different trees, so a phase that loads one never loads the other.

    problems = preflight()          # [] or the reasons parity cannot be tested here (NOT APPLICABLE)
    o = load()                      # raises NotApplicable rather than load a wrong oracle
    obs = observation_v6(cid)       # the tester's library v6 (or observation(cid, "v5")), this snapshot's loader
    res, claims, sc, est = fit(obs, "jax2")   # `calibrate`'s run_fit call, final_claim_kwargs, estimate_block
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
SNAP3 = os.path.join(REPO, "tests", "oracle", "jax2_6884fc8")
SRB = os.path.join(REPO, "tests", "oracle", "tester_f94ec2b")
KEY = "srbench_methods.mle_profile_jax2"
NAMES = {"jax2": "MleProfileJax2", "jax2_h": "MleProfileJax2H"}
LIBRARY_V6 = os.environ.get(
    "SRAGING_TESTER_V6",
    os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset", "sr_calibration_tester", "data", "library", "v6"))   # 0.1.1
#: library v5, where jax2's 1.0.1 S3 records were made (the parity panel's mechanism pairs; manifest 9489fb0d)
LIBRARY_V5 = os.path.join(os.path.dirname(LIBRARY_V6), "v5")
LIBRARIES = {"v5": LIBRARY_V5, "v6": LIBRARY_V6}
SECTION5 = {"PJRT_NPROC": "1", "SRB_STRICT": "1"}          # handoff section 5; SRK_DT_CEILING must be unset


def _sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def preflight() -> list[str]:
    """Every reason the side-by-side comparison would not be valid here (empty = valid): v033's checks (pycache, its
    snapshot, the pins, Python, the platform, numerics overrides) plus both 003b snapshots' and the environment's."""
    problems = V.preflight()
    for var, want in SECTION5.items():
        if os.environ.get(var, want) != want:
            problems.append(f"{var} is {os.environ[var]!r}: the handoff's section-5 environment needs {want!r}")
    man = json.load(open(os.path.join(SNAP3, "MANIFEST.json")))
    for f in man["files"]:
        p = os.path.join(SNAP3, f["path"])
        if not os.path.exists(p):
            problems.append(f"snapshot file missing: {f['path']}")
        elif _sha256(p) != f["sha256"]:
            problems.append(f"snapshot checksum mismatch: {f['path']}")
    lock_path = os.path.join(SNAP3, "mle_profile_jax2", "VERSION_LOCK.json")
    if os.path.exists(lock_path):
        if _sha256(lock_path) != man["version_lock_sha256"]:
            problems.append("VERSION_LOCK.json differs from the sha256 the snapshot records")
        lock = json.load(open(lock_path))
        for name, ent in lock["names"].items():
            for fname, h in ent["sha256"].items():
                p = os.path.join(SNAP3, "mle_profile_jax2", fname)
                if not os.path.exists(p) or _sha256(p) != h:
                    problems.append(f"VERSION_LOCK: {name} {fname} does not match the snapshot's file")
    for c in man["cited"]:
        p = os.path.join(V.SNAP, c["v033_path"])
        if not os.path.exists(p) or _sha256(p) != c["sha256"]:
            problems.append(f"cited file {c['v033_path']} differs from the hash the snapshot cites")
    sman = json.load(open(os.path.join(SRB, "MANIFEST.json")))
    for f in sman["files"]:
        p = os.path.join(SRB, f["path"])
        if not os.path.exists(p) or _sha256(p) != f["sha256"]:
            problems.append(f"srbench snapshot file {f['path']} missing or changed")
    return problems


class Oracle(V.Oracle):
    """Handles on the loaded 1.0.1 oracle: SRK, the package module (`adapter`), its `run_fit`, `final_claim_kwargs`
    and `estimate_block`, the two registered classes by short name, and the estimator / fp_jax siblings."""


_LOADED: Oracle | None = None


def load() -> Oracle:
    """Load the 1.0.1 oracle into this process (once). Raises NotApplicable rather than load a wrong one."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    problems = preflight()
    if problems:
        raise NotApplicable(problems)
    if "srbench" in sys.modules and not os.path.realpath(sys.modules["srbench"].__file__).startswith(
            os.path.realpath(SRB)):
        raise NotApplicable(["another srbench is already imported in this process (one oracle per process)"])
    for var, want in SECTION5.items():
        os.environ[var] = want
    for p in (SRB, V.SNAP_SRK_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    os.environ["SRBENCH_SUT"] = os.path.realpath(V.SNAP_SRK_ROOT)
    import SRK
    srk_root = os.path.realpath(os.path.dirname(os.path.dirname(SRK.__file__)))
    if srk_root != os.path.realpath(V.SNAP_SRK_ROOT):
        raise NotApplicable([f"SRK resolved to {srk_root}, not the v033 snapshot"])
    import srbench
    if not os.path.realpath(srbench.__file__).startswith(os.path.realpath(SRB)):
        raise NotApplicable([f"srbench resolved to {srbench.__file__}, not tests/oracle/tester_f94ec2b"])
    pkg = os.path.join(SNAP3, "mle_profile_jax2")
    ad = V._load_path(KEY, os.path.join(pkg, "__init__.py"))
    if os.path.realpath(ad.__file__) != os.path.realpath(os.path.join(pkg, "__init__.py")):
        raise NotApplicable([f"{KEY} resolved to {ad.__file__}, not the 1.0.1 snapshot"])
    lock = ad.version_lock_problems()
    if lock:
        raise NotApplicable([f"version lock: {p}" for p in lock])
    classes = {short: getattr(ad, cls) for short, cls in NAMES.items()}
    _LOADED = Oracle(np=np, SRK=SRK, adapter=ad, run_fit=ad.run_fit, final_claim_kwargs=ad.final_claim_kwargs,
                     estimate_block=ad.estimate_block, classes=classes, estimator=ad._sibling("estimator"),
                     fp_jax=ad._sibling("fp_jax"))
    return _LOADED


def fit_kwargs(name: str) -> dict:
    """`run_fit`'s keywords for a registered name, exactly as 1.0.1's `calibrate` passes them (the class constants;
    kappa_fixed / combos only when set): FREEZE/snapshot/SWITCHES.tsv's "run_fit (captured)" row."""
    cls = load().classes[name]
    return dict(relabel=cls.RELABEL, hazard=cls.HAZARD, window=cls.WINDOW, shadow=cls.SHADOW, exact_from_F=cls.EXACT_F,
                self_consistent=cls.SELF_CONSISTENT, floor_early=cls.FLOOR_EARLY, n_rule=cls.N_RULE,
                dt_rule=cls.DT_RULE, early_all=cls.EARLY_ALL, inner_polish_j=cls.INNER_POLISH_J,
                bracket_bisect=cls.BRACKET_BISECT, extent_z=cls.EXTENT_Z, box_flag=cls.BOX_FLAG,
                **({"kappa_fixed": cls.KAPPA0} if cls.KAPPA0 is not None else {}),
                **({"combos": cls.COMBOS} if cls.COMBOS else {}))


def fit(obs, name: str, *, stages=None):
    """One oracle fit: (run_fit result, final claims, self-consistency block, estimate block), as `calibrate` and
    `claimset` produce them. `stages=("mle",)` calls the estimator's MLE stage instead (a theta-only pair)."""
    o = load()
    if stages is None:
        res = o.run_fit(obs, **fit_kwargs(name))
    else:
        res = _mle_only(o, obs, name, stages)
    claims, sc_block = o.final_claim_kwargs(res, o.classes[name].SELF_CONSISTENT)
    cls = o.classes[name]
    return res, claims, sc_block, o.estimate_block(res, cls.HAZARD, cls.KAPPA0)


def _mle_only(o, obs, name, stages):
    """`run_fit`'s two steps with `stages` added (as sraging.leverage.refit_mle does on the port)."""
    kw = fit_kwargs(name)
    om = o.adapter._sibling("observation_model")
    prep = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=kw["window"], relabel_on=kw["relabel"])
    extra = {k: True for k in ("shadow", "exact_from_F", "self_consistent", "floor_early", "n_rule", "dt_rule",
                               "early_all", "bracket_bisect", "extent_z", "box_flag") if kw.get(k)}
    if kw.get("inner_polish_j"):
        extra["inner_polish"] = True
    return o.estimator.fit(o.fp_jax, prep.ages, prep.events, delta=obs.delta, levels=(o.adapter.LEVEL,),
                           kappa_max=o.adapter.KAPPA_MAX, hazard=kw["hazard"], window=prep.window_eff,
                           stages=stages, **extra)


_LIBS: dict = {}


def library(version: str = "v6"):
    """The tester's library `version` (v5 or v6), through this snapshot's srbench loader, read-only."""
    if version not in _LIBS:
        load()
        from srbench.library.load import Library
        _LIBS[version] = Library(root=LIBRARIES[version], version=version)
    return _LIBS[version]


def library_v6():
    return library("v6")


def observation(cohort_id, version: str = "v6"):
    return library(version).observation(cohort_id)


def observation_v6(cohort_id):
    return observation(cohort_id, "v6")
