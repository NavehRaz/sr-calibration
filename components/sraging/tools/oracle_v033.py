"""The frozen v0.3.3 oracle, loaded side by side with the port (task T035; research R4; constitution VI).

Used by the parity tests (`tests/conftest.py`) and by `tools/acceptance.py`. Never edits anything: the
snapshot in `tests/oracle/v033` is checksum-verified against its MANIFEST, bytecode is redirected away
from it (`sys.pycache_prefix`), and the live tester is only ever read.

    problems = preflight()        # [] or the reasons parity cannot be tested here ("NOT APPLICABLE")
    o = load()                    # raises NotApplicable(problems) instead of loading a wrong oracle
    obs = observation(cohort_id)  # the tester's Observation from library v1 (read-only)
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(REPO, "tests", "oracle", "v033")
SNAP_SRK_ROOT = os.path.join(SNAP, "SRtools_kappa")
SNAP_TESTER = os.path.join(SNAP, "sr_calibration_tester")
MANIFEST = os.path.join(SNAP, "MANIFEST.json")
#: release 0.1.1: the folder holding the lab's checkouts (sr_calibration_tester, sr_calibration_tester_jax2, its
#: freeze, SRtools_kappa). Unset, the paths below name that it is unset, and whatever needs them is not applicable.
CODE5 = os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset"
TESTER = os.environ.get(
    "SRAGING_TESTER_ROOT",   # a tester checkout, read only. Its data are read here; tools/acceptance_claims.py
                             # also imports its srbench run/score code (adapters.srbench.connect refuses any
                             # revision but f8eaed2). Point it at a checkout pinned there once the live one moves.
    os.path.join(CODE5, "sr_calibration_tester"))
LIBRARY_V1 = os.path.join(TESTER, "data", "library", "v1")
NUMERICS_OVERRIDES = ("SRK_DT_CEILING", "SRK_KAPPA_MAX")
FREEZE_CRITICAL = ("numpy", "scipy", "jax", "jaxlib")


class NotApplicable(RuntimeError):
    """Parity cannot be tested in this process; never a pass."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("NOT APPLICABLE: " + "; ".join(self.problems))


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _freeze():
    out = {}
    for line in open(os.path.join(SNAP, "pip_freeze.txt")):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(.+)$", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def preflight() -> list[str]:
    """Every reason the side-by-side comparison would not be valid in this process (empty = valid)."""
    problems = []
    if not sys.pycache_prefix:
        problems.append("sys.pycache_prefix is not set: importing the oracle would write into tests/oracle")
    man = json.load(open(MANIFEST))
    for f in man["files"]:
        p = os.path.join(SNAP, f["repo"], f["path"])
        if not os.path.exists(p):
            problems.append(f"snapshot file missing: {f['repo']}/{f['path']}")
        elif _md5(p) != f["md5"]:
            problems.append(f"snapshot checksum mismatch: {f['repo']}/{f['path']}")
    for pkg, want in man["environment_pins"].items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = "not installed"
        if got != want:
            problems.append(f"{pkg} {got} != pinned {want}")
    freeze = _freeze()
    for pkg in FREEZE_CRITICAL:
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = "not installed"
        if got != freeze.get(pkg):
            problems.append(f"{pkg} {got} != pip_freeze {freeze.get(pkg)}")
    if sys.version.split()[0] != man["python"].split()[0]:
        problems.append(f"python {sys.version.split()[0]} != snapshot {man['python'].split()[0]}")
    for var in NUMERICS_OVERRIDES:
        if os.environ.get(var):
            problems.append(f"{var} is set ({os.environ[var]!r}): it changes the oracle's numerics")
    # The host must be the snapshot's: same machine type and OS build. The hostname is NOT compared:
    # on macOS it is network-assigned and changes (measured 2026-10-03: the snapshot recorded node
    # "naveh-mac", gethostname() now says "MC06268844" on the same machine -- MANIFEST.json was born on
    # this filesystem 2 s after its own `created` stamp, in the repo whose initial commit was made here).
    # The stable identity, the hardware UUID, is recorded in every acceptance result (host_identity()).
    if platform.machine() != man["host"]["machine"] or platform.platform() != man["host"]["platform"]:
        problems.append(f"host {platform.machine()} {platform.platform()} is not the snapshot host "
                        f"{man['host']['machine']} {man['host']['platform']} (byte identity is claimed only there)")
    return problems


def host_identity() -> dict:
    """Stable host facts for results: hardware UUID (macOS IOPlatformUUID), model, platform, hostname."""
    import subprocess
    uuid = "unknown"
    try:
        out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r'"IOPlatformUUID" = "([^"]+)"', out)
        uuid = m.group(1) if m else uuid
    except (OSError, subprocess.SubprocessError):
        pass
    return {"uuid": uuid, "machine": platform.machine(), "platform": platform.platform(),
            "node": platform.node()}


class Oracle:
    """Handles on the loaded oracle: SRK, the reference adapter + its method object, fp_jax, estimator,
    SRK's read_csv, and the reference entry point's warning functions."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def delta_grid_fraction(self, ages, delta):
        """VERBATIM expression of tools/fit_cohort_jax.py:108-111 (the off-grid delta check)."""
        np = self.np
        off = np.abs(ages / delta - np.round(ages / delta))
        return float(np.mean(off > 1e-6))


_LOADED: Oracle | None = None


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load() -> Oracle:
    """Load the oracle into this process (once). Raises NotApplicable rather than load a wrong one."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    problems = preflight()
    if problems:
        raise NotApplicable(problems)
    for p in (SNAP_TESTER, SNAP_SRK_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    if not hasattr(np, "trapz"):                 # the tester's own shim (srbench/npcompat.py)
        np.trapz = np.trapezoid
    os.environ["SRBENCH_SUT"] = os.path.realpath(SNAP_SRK_ROOT)
    import SRK
    srk_root = os.path.realpath(os.path.dirname(os.path.dirname(SRK.__file__)))
    if srk_root != os.path.realpath(SNAP_SRK_ROOT):
        raise NotApplicable([f"SRK resolved to {srk_root}, not the snapshot {SNAP_SRK_ROOT}"])
    import srbench
    if not os.path.realpath(srbench.__file__).startswith(os.path.realpath(SNAP_TESTER)):
        raise NotApplicable([f"srbench resolved to {srbench.__file__}, not the snapshot"])
    adapter = _load_path("srbench_methods.mle_profile_jax",
                         os.path.join(SNAP_TESTER, "methods", "mle_profile_jax", "__init__.py"))
    from srbench.contract import method as _method
    method = _method._REGISTRY["mle_profile_jax"]        # registered by the import above; no discover()
    fj = adapter._sibling("fp_jax")
    est = adapter._sibling("estimator")
    tool = _load_path("srbench_tools_fit_cohort_jax", os.path.join(SNAP_TESTER, "tools", "fit_cohort_jax.py"))
    from SRK.fit_cohort import read_csv
    from srbench.contract.types import Observation
    _LOADED = Oracle(np=np, SRK=SRK, adapter=adapter, method=method, fp_jax=fj, estimator=est,
                     read_csv=read_csv, convergence_warnings=tool.convergence_warnings,
                     censoring_tie_warning=tool.censoring_tie_warning, Observation=Observation)
    return _LOADED


_LIB = None


def library():
    """The tester's library v1 (the one the reference ran on), via the snapshot's loader, read-only."""
    global _LIB
    if _LIB is None:
        load()
        from srbench.library.load import Library
        _LIB = Library(root=LIBRARY_V1)
        if _LIB.version != "v1":
            raise NotApplicable([f"library at {LIBRARY_V1} is {_LIB.version}, not v1"])
    return _LIB


def observation(cohort_id):
    return library().observation(cohort_id)


def quick_tier() -> list[str]:
    lib = library()
    return [s.cohort_id for s in lib.select(lib.tiers["quick"])]


def stored_reference_record(cohort_id) -> dict:
    """The stored v0.3.3 reference record of a cohort (run mle_profile_jax-full), read-only JSON."""
    man = json.load(open(os.path.join(TESTER, "runs", "mle_profile_jax-full.json")))
    d = man["keys"][cohort_id]
    with open(os.path.join(TESTER, "data", "store", d[:2], d[2:4], f"{d}.json")) as fh:
        return json.load(fh)
