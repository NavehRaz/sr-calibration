"""The tester adapter: srbench scores sraging's calibrators as methods, without writing to the tester (T052; spec 003
T041: a method table -- `sraging_profile_v033` against the shared tester at f8eaed2, `sraging_profile_jax2` and
`sraging_profile_jax2_h` against the jax2 clone at 5578de2, whose srbench is f8eaed2's).

contracts/srbench-adapter.md; research R10; constitution VI. The tester checkout is used READ-ONLY:
  * its HEAD must be the recorded revision;
  * bytecode is redirected (`sys.pycache_prefix`) so importing it writes nothing there;
  * the method is registered under a NEW name (`sraging_profile_v033`), so its store keys can never collide
    with -- or be mistaken for -- the reference's (`store/key.py:112-146`);
  * `discover()` is never called and `get()` never asked for an unregistered name (both would import the
    reference adapter: `contract/method.py:79-128`);
  * port records go to a store owned by sraging (`<repo>/var/srbench_store`); the reference records are read
    from the tester's store through a separate ResultStore, never written;
  * the tester's command line is never used (it cannot split stores and may write explorer output).
`fingerprint()` lets callers prove the tester tree is unchanged before vs after.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

from sraging.adapters.observation import cohort_from_observation
from sraging.compat.v033 import claims as _vclaims
from sraging.errors import UnsupportedOptionError

TESTER_HEAD = "f8eaed2d56d7b237d01e38f140fe1d71a8c83717"
#: The live tester checkout, read only. No machine path is built in (release 0.1.0): it comes from SRAGING_TESTER_ROOT,
#: the variable `tools/oracle_v033.py` reads, or from an explicit `tester_root=`; with neither, `_tester_root` refuses.
TESTER_ROOT_ENV = "SRAGING_TESTER_ROOT"
DEFAULT_TESTER = os.environ.get(TESTER_ROOT_ENV)
METHOD_NAME = "sraging_profile_v033"
REFERENCE_RUN = "mle_profile_jax-full"
REFERENCE_COST = {"median_seconds": 590.0, "min_seconds": 140.0, "max_seconds": 70000.0}  # reference cost()
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PORT_STORE = os.path.join(_REPO, "var", "srbench_store")
PORT_RUNS = os.path.join(_REPO, "var", "srbench_runs")
#: The live tester checkout moves (it is at a later commit and keeps moving), so the jax2 methods are scored against
#: a read-only clone pinned to J2_HEAD, made with `git clone --no-checkout <J2> .cache/j2_5578de2` and
#: `git -C .cache/j2_5578de2 checkout --detach <J2_HEAD>` (SRAGING_J2_PINNED overrides the path). `connect` refuses
#: if the clone's HEAD differs or its tree is dirty. The clone has the code and `library/v2/manifest.json` only: the
#: library's data files are untracked in J2, so they are read from the live checkout (J2_LIVE, from SRAGING_J2_LIVE; no
#: machine path is built in), read-only.
J2_LIVE = os.environ.get("SRAGING_J2_LIVE")
J2_TESTER = os.environ.get("SRAGING_J2_PINNED", os.path.join(_REPO, ".cache", "j2_5578de2"))
J2_HEAD = "5578de259eb2165d5fbb0823b8f1c81b0216238f"

#: Spec 003e: the library is an explicit, recorded input (never the tester's `Library()` default, which moves with the
#: reference). The tester's code comes from a read-only clone of the commit it named (2026-10-17); its data (cohorts,
#: store, runs) are untracked by the tester, so they are read from the live checkout and pinned by content (`Pins`).
LIBRARIES = ("v1", "v2", "v3", "v4", "v5", "v6")
TESTER_PINNED = os.environ.get("SRAGING_TESTER_PINNED", os.path.join(_REPO, ".cache", "tester_42a2789"))
TESTER_PINNED_HEAD = "42a2789a5f0e41e8820cbb354857c450d68da1f8"
H_PORT = "sraging_profile_jax2_h"           # _h 1.0.0: not validated beyond library v2 (refused elsewhere, spec 003e C6)
#: Spec 006 (research R9, R11): the 1.0.1 and W rows are scored on library v6 against 003g's pinned clone of the tester
TESTER_V6_PINNED = os.environ.get("SRAGING_TESTER_V6_PINNED", os.path.join(_REPO, ".cache", "tester_0805d36"))
TESTER_V6_HEAD = "0805d36bb92dd9a1a531683bcc21824ca7dadac8"


def _tester_root(root):
    """`root`, or refuse: with no tester checkout passed and SRAGING_TESTER_ROOT unset there is nothing to read."""
    if not root:
        raise UnsupportedOptionError("tester_root", f"no tester checkout: set {TESTER_ROOT_ENV} to an "
                                     "sr_calibration_tester checkout, or pass tester_root=")
    return root


def library_root(library, tester_root) -> str:
    """The library directory under a tester data root; the version is required and must be one of LIBRARIES."""
    if library is None:
        raise UnsupportedOptionError("library", "library is required (one of v1-v6): the tester's Library() default "
                                                "moves with its reference, so sraging never relies on it (spec 003e)")
    if library not in LIBRARIES:
        raise UnsupportedOptionError("library", f"library {library!r} is not one of {', '.join(LIBRARIES)}")
    return os.path.join(tester_root, "data", "library", library)


def _refuse_h_off_v2(method_name, library):
    if method_name == H_PORT and library is not None and library != "v2":
        raise UnsupportedOptionError("library", f"{H_PORT} (the port of mle_profile_jax2_h 1.0.0) is pinned to library "
                                                f"v2 and refused on {library}: the tester's quarantine does not cover "
                                                f"sraging's name (spec 003e C6)")


def _jax2_bounds():
    from sraging.compat.jax2 import claims as _jc
    return dict(_jc.BOUNDS)


#: Spec 003 (T041): each sraging method, the calibrator behind it ("name@version": spec 006 pins it, so a row's
#: description stays true when a bare name moves) and the tester checkout it is scored against. Spec 006 (FR-013):
#: each row's `capabilities`, in the tester's names (contract/types.py CAPABILITIES), sets only the explorer's "well
#: specified" label; it never changes a score.
METHODS = {
    "sraging_profile_v033": {"calibrator": "profile_v033", "tester_root": DEFAULT_TESTER, "tester_head": TESTER_HEAD,
                             "library": "v1", "version_suffix": "+v033",
                             "description": "sraging's verbatim port of mle_profile_jax v0.3.3 (calibrator profile_v033)",
                             "contact": "sraging / spec 001"},
    "sraging_profile_jax2": {"calibrator": "profile_jax2@1.0.0", "tester_root": J2_TESTER, "tester_head": J2_HEAD,
                             "library": "v2", "version_suffix": "+jax2-1.0.0",
                             "description": "sraging's verbatim port of mle_profile_jax2 1.0.0 (calibrator profile_jax2)",
                             "contact": "sraging / spec 003"},
    "sraging_profile_jax2_h": {"calibrator": "profile_jax2_h@1.0.0", "tester_root": J2_TESTER, "tester_head": J2_HEAD,
                               "library": "v2", "version_suffix": "+jax2_h-1.0.0",
                               "description": "sraging's verbatim port of mle_profile_jax2_h 1.0.0 (profile_jax2_h)",
                               "contact": "sraging / spec 003"},
    "sraging_profile_jax2_101": {"calibrator": "profile_jax2@1.0.1", "tester_root": TESTER_V6_PINNED,
                                 "tester_head": TESTER_V6_HEAD, "library": "v6", "version_suffix": "+jax2-1.0.1",
                                 "description": "sraging's verbatim port of mle_profile_jax2 1.0.1 (profile_jax2)",
                                 "contact": "sraging / spec 006"},
    "sraging_profile_jax2_h_101": {"calibrator": "profile_jax2_h@1.0.1", "tester_root": TESTER_V6_PINNED,
                                   "tester_head": TESTER_V6_HEAD, "library": "v6", "version_suffix": "+jax2_h-1.0.1",
                                   "description": "sraging's verbatim port of mle_profile_jax2_h 1.0.1, windows by the "
                                                  "tester's rule (profile_jax2_h)",
                                   "capabilities": ("fit_window", "external_hazard"), "contact": "sraging / spec 006"},
    "sraging_profile_jax2_w": {"calibrator": "profile_jax2_w@1.0.1+w1", "tester_root": TESTER_V6_PINNED,
                               "tester_head": TESTER_V6_HEAD, "library": "v6", "version_suffix": "+jax2_w-1.0.1+w1",
                               "description": "sraging's composition of jax2 1.0.1+w1: J with the window on and h = 0, "
                                              "never run by jax2 (profile_jax2_w)",
                               "capabilities": ("fit_window",), "contact": "sraging / spec 006"},
}


def methods() -> dict:
    return {k: dict(v) for k, v in METHODS.items()}


def descriptor_fields(method_name: str = METHOD_NAME) -> dict:
    import sraging
    m = METHODS[method_name]
    bounds = dict(_vclaims.BOUNDS) if m["calibrator"] == "profile_v033" else _jax2_bounds()
    return {"name": method_name, "version": f"{sraging.__version__}{m['version_suffix']}", "bounds": bounds,
            "capabilities": tuple(m.get("capabilities", ())), "description": m["description"],
            "contact": m["contact"]}


def fingerprint(root: str):
    """(sorted (relpath, size, mtime_ns) of every file, `git status --porcelain`) -- equal iff unchanged."""
    rows = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f)
            try:
                st = os.lstat(p)
            except FileNotFoundError:
                continue
            rows.append((os.path.relpath(p, root), st.st_size, st.st_mtime_ns))
    git = ""
    if os.path.isdir(os.path.join(root, ".git")):
        git = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True,
                             timeout=120).stdout
    return tuple(rows), git


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def pinned_problems(tester_root: str, tester_head: str) -> list[str]:
    """Reasons `tester_root` is not the pinned tester tree: missing, HEAD differs, or a dirty tree."""
    if not os.path.isdir(tester_root):
        return [f"{tester_root} does not exist (make the pinned clone: see J2_TESTER's comment)"]
    r = subprocess.run(["git", "-C", tester_root, "rev-parse", "HEAD"], capture_output=True, text=True)
    out = []
    head = r.stdout.strip()
    if head != tester_head:
        out.append(f"tester checkout {tester_root} is at {head or '?'}, not the recorded revision {tester_head}")
    st = subprocess.run(["git", "-C", tester_root, "status", "--porcelain"], capture_output=True, text=True,
                        timeout=120)
    if st.returncode != 0 or st.stdout.strip():
        out.append(f"tester checkout {tester_root} is dirty or unreadable: {(st.stdout or st.stderr)[:200]!r}")
    return out


def connect(tester_root: str = DEFAULT_TESTER, tester_head: str = TESTER_HEAD, require_clean: bool = False):
    """Import srbench from the tester checkout, read-only, and return its modules.

    If srbench is already imported from the frozen snapshot (the oracle, in the same process), the
    snapshot's package path is EXTENDED with the tester's -- allowed only because every file the two share
    is byte-identical (checked here), so `srbench.run/store/score` load from the tester and the shared
    modules stay the snapshot's identical copies.
    """
    tester_root = _tester_root(tester_root)
    r = subprocess.run(["git", "-C", tester_root, "rev-parse", "HEAD"], capture_output=True, text=True)
    head = r.stdout.strip()
    if head != tester_head:
        raise UnsupportedOptionError("tester_root", f"tester checkout {tester_root} is at {head or '?'}, not "
                                                    f"the recorded revision {tester_head}")
    if require_clean:
        bad = pinned_problems(tester_root, tester_head)
        if bad:
            raise UnsupportedOptionError("tester_root", "; ".join(bad))
    if not sys.pycache_prefix:
        sys.pycache_prefix = os.path.join(_REPO, ".cache", "pycache")
    live_pkg = os.path.join(tester_root, "srbench")
    if "srbench" in sys.modules:
        pkg = sys.modules["srbench"]
        snap_pkg = os.path.dirname(os.path.realpath(pkg.__file__))
        if os.path.realpath(live_pkg) != snap_pkg:
            for d, _, files in os.walk(snap_pkg):
                for f in files:
                    if f.endswith(".py"):
                        sp = os.path.join(d, f)
                        lp = os.path.join(live_pkg, os.path.relpath(sp, snap_pkg))
                        if not os.path.exists(lp) or _md5(lp) != _md5(sp):
                            raise UnsupportedOptionError(
                                "tester_root", f"snapshot srbench file {os.path.relpath(sp, snap_pkg)} differs "
                                               f"from the tester's; cannot share one srbench package")
            if live_pkg not in pkg.__path__:
                pkg.__path__.append(live_pkg)
    else:
        if tester_root not in sys.path:
            sys.path.insert(0, tester_root)
    import importlib
    mods = {name: importlib.import_module(f"srbench.{name}") for name in (
        "contract.method", "contract.types", "run.execute", "store.store", "library.load",
        "score.verdict", "score.floors", "score.claims", "reference.pointer")}
    return mods


def claimset_from_kwargs(types_mod, cohort_id, method_name, claims_kw, diagnostics, wall_s=None, cpu_s=None,
                         provenance=None):
    """A tester ClaimSet from claim keyword dicts (only the four SR parameters: srbench validates nothing else)."""
    T = types_mod
    claims = tuple(T.Claim(**{k: v for k, v in kw.items() if k in ("parameter", "shape", "lower", "upper", "point",
                                                                      "level", "n_components") and v is not None})
                   for kw in claims_kw if kw["parameter"] in ("eta", "beta", "epsilon", "kappa"))
    return T.ClaimSet(cohort_id=cohort_id, method_name=method_name, claims=claims, status="ok",
                      wall_clock_seconds=wall_s, cpu_seconds=cpu_s, diagnostics=dict(diagnostics or {}),
                      code_provenance=provenance or {})


def make_jax2_method(types_mod, method_name="sraging_profile_jax2", precomputed=None, library=None):
    """A jax2 calibrator as a tester method (spec 003 T041): combinations off (the tester scores the four
    parameters only), the four claims and the fit's diagnostics. `library` (spec 003e) refuses the `_h` port off v2;
    None keeps the method row's own pin."""
    _refuse_h_off_v2(method_name, library)
    T = types_mod
    pre = precomputed if precomputed is not None else {}
    cal, _, version = METHODS[method_name]["calibrator"].partition("@")

    class SragingProfileJax2:
        descriptor = T.MethodDescriptor(**descriptor_fields(method_name))

        def cost(self, spec):
            return T.CostDeclaration(**REFERENCE_COST)

        def calibrate(self, observation):
            import sraging
            try:                                 # spec 006: the window reaches the calibrator, which may refuse it
                cohort = cohort_from_observation(observation)
                r = pre.get(observation.cohort_id) or sraging.fit_cohort(cohort, calibrator=cal,
                                                                         version=version or None, combinations=False)
            except UnsupportedOptionError as ex:
                return T.ClaimSet(cohort_id=observation.cohort_id, method_name=method_name,
                                  status="error:WindowUnsupported", diagnostics={"exception": str(ex)[:500]})
            return claimset_from_kwargs(T, observation.cohort_id, method_name,
                                        [c.reference_dict() for c in r.claims], r.diagnostics,
                                        r.timing.get("wall_s"), r.timing.get("cpu_s"),
                                        {"package": "sraging", **r.provenance.get("package", {})})

    return SragingProfileJax2()


def make_method(types_mod, precomputed=None):
    """The srbench method object. `precomputed` maps cohort_id -> a FitResult already computed by
    `profile_v033.fit` on that cohort (the acceptance claims phase passes its own port fit, so the record
    is the port's real computation and no second fit is run)."""
    T = types_mod
    pre = precomputed if precomputed is not None else {}

    class SragingProfileV033:
        descriptor = T.MethodDescriptor(**descriptor_fields())

        def cost(self, spec):
            return T.CostDeclaration(**REFERENCE_COST)

        def calibrate(self, observation):
            from sraging.calibrators import profile_v033
            t0, c0 = time.perf_counter(), time.process_time()
            try:                                 # spec 006: the window reaches profile_v033, which refuses it
                cohort = cohort_from_observation(observation)
                r = pre.get(observation.cohort_id) or profile_v033.fit(cohort)
            except UnsupportedOptionError as ex:
                return T.ClaimSet(cohort_id=observation.cohort_id, method_name=METHOD_NAME,
                                  status="error:WindowUnsupported", diagnostics={"exception": str(ex)[:500]})
            if r.status != "ok":
                return T.ClaimSet(cohort_id=observation.cohort_id, method_name=METHOD_NAME, status=r.status,
                                  wall_clock_seconds=time.perf_counter() - t0,
                                  cpu_seconds=time.process_time() - c0, diagnostics=dict(r.diagnostics))
            claims = []
            for c in r.claims:
                kw = {k: v for k, v in c.reference_dict().items() if v is not None}
                claims.append(T.Claim(**kw))
            return T.ClaimSet(cohort_id=observation.cohort_id, method_name=METHOD_NAME, claims=tuple(claims),
                              status="ok", wall_clock_seconds=r.timing.get("wall_s"),
                              cpu_seconds=r.timing.get("cpu_s"), diagnostics=dict(r.diagnostics),
                              code_provenance={"package": "sraging", **r.provenance.get("package", {})})

    return SragingProfileV033()


def register_with(method_mod, method):
    """Register without discovery (never `discover()`, never `get()` an unregistered name)."""
    method_mod.register(method)
    return method


def run_records(mods, method, cohort_ids, tester_root=DEFAULT_TESTER, store_root=PORT_STORE, run_id=None,
                library=None):
    """srbench's own `run()` on the given cohorts, into sraging's store; returns (manifest, manifest_path).
    `library=None` keeps spec 001's v1 root (history); spec 003e passes the library explicitly."""
    tester_root = _tester_root(tester_root)
    _refuse_h_off_v2(getattr(method.descriptor, "name", None), library)
    Library = mods["library.load"].Library
    ResultStore = mods["store.store"].ResultStore
    lib = Library(root=os.path.join(tester_root, "data", "library", "v1") if library is None
                  else library_root(library, tester_root))
    os.makedirs(store_root, exist_ok=True)
    manifest = mods["run.execute"].run(method, lib, list(cohort_ids), ResultStore(store_root))
    path = os.path.join(PORT_RUNS, f"{run_id or METHOD_NAME}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mods["run.execute"].save_manifest(manifest, path)
    return manifest, path


def compare(mods, port_manifest, cohort_ids, tester_root=DEFAULT_TESTER, port_store_root=PORT_STORE):
    """The tester's own comparison of the port's records against the stored reference, read-only."""
    tester_root = _tester_root(tester_root)
    import json
    ResultStore = mods["store.store"].ResultStore
    load_results = mods["run.execute"].load_results
    V, F, C = mods["score.verdict"], mods["score.floors"], mods["score.claims"]
    with open(os.path.join(tester_root, "runs", f"{REFERENCE_RUN}.json"), encoding="utf-8") as fh:
        ref_manifest = json.load(fh)
    mods["reference.pointer"].require_comparable(port_manifest, ref_manifest)
    ids = list(cohort_ids)
    a = [cs for cs in load_results(port_manifest, ResultStore(port_store_root)) if cs.cohort_id in ids]
    ref_sub = dict(ref_manifest, cohort_ids=ids, keys={c: ref_manifest["keys"][c] for c in ids})
    b = load_results(ref_sub, ResultStore(os.path.join(tester_root, "data", "store")))
    lib = mods["library.load"].Library(root=os.path.join(tester_root, "data", "library", "v1"))
    truths, clusters = lib.truths(ids), lib.cluster_map(ids)
    specs = [lib.spec(c) for c in ids]
    out = {"N": len(ids), "port_records": len(a), "reference_records": len(b), "parameters": {}}
    for p in ("eta", "beta", "epsilon", "kappa"):
        axes = {
            "calibration": V.calibration_axis(a, b, truths, p, clusters, cohort_set=f"{len(ids)} paired"),
            "calibration_one_sided": V.calibration_one_sided_axis(a, b, truths, p, clusters, f"{len(ids)} paired"),
            "availability": V.availability_axis(a, b, p, f"{len(ids)} paired"),
            "information": V.information_axis(a, b, p, f"{len(ids)} paired"),
            "accuracy": V.accuracy_axis(a, b, truths, p, F.cohort_floors(dict(zip(ids, specs)), p),
                                        f"{len(ids)} paired"),
            "cost": V.cost_axis(a, b, f"{len(ids)} paired"),
        }
        cov = {side: {k: {f: getattr(v, f) for f in ("hits", "claimed", "declined", "failed", "total")}
                      for k, v in C.coverage_all_shapes(cs, truths, p).items()}
               for side, cs in (("port", a), ("reference", b))}
        out["parameters"][p] = {"axes": {k: {"outcome": ax.outcome, "statistic": ax.statistic,
                                             "denominator": ax.denominator} for k, ax in axes.items()},
                                "coverage": cov}
    return out, a, b


# ------------------------------------------------------------------------------------------- spec 003e: v5 and on
def connect_pinned(clone=None, head=TESTER_PINNED_HEAD) -> dict:
    """`connect()` on the read-only clone of the tester commit sraging pins (spec 003e R1), plus the tester's own
    comparison module. Refuses a missing clone, another HEAD, or a dirty tree, before importing anything."""
    clone = clone or TESTER_PINNED
    bad = pinned_problems(clone, head)
    if bad:
        raise UnsupportedOptionError("tester", "the pinned tester clone: " + "; ".join(bad))
    mods = connect(tester_root=clone, tester_head=head, require_clean=True)
    import importlib
    mods["report.comparison"] = importlib.import_module("srbench.report.comparison")
    return mods


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Pins:
    """Content pins (spec 003e R5): the sha256 of every tester file sraging reads, which the tester does not track
    (runs, store records, cohort arrays) or tracks only at a commit (the library manifests). `root` is the tester's
    data root (the live checkout). Every check refuses a mismatch, or a file that was never pinned, by name."""

    def __init__(self, data: dict, root: str, path: str | None = None):
        self.data, self.root, self.path = data, root, path
        self._by_path = {}
        for section in ("libraries", "runs", "arrays"):
            for name, ent in (data.get(section) or {}).items():
                self._by_path[ent["path"]] = (section, name, ent["sha256"])

    def sha256(self) -> str | None:
        return _sha256(self.path) if self.path else None

    def check_file(self, rel_path: str) -> str:
        ent = self._by_path.get(rel_path)
        if ent is None:
            raise UnsupportedOptionError("pins", f"{rel_path} is not pinned (spec 003e reads only pinned files)")
        got = _sha256(os.path.join(self.root, rel_path))
        if got != ent[2]:
            raise UnsupportedOptionError("pins", f"{rel_path}: sha256 {got[:16]} differs from its pin {ent[2][:16]}")
        return got

    def read_file(self, rel_path: str) -> bytes:
        """The file's bytes, read once and checked against its pin: what is checked is what is parsed."""
        ent = self._by_path.get(rel_path)
        if ent is None:
            raise UnsupportedOptionError("pins", f"{rel_path} is not pinned (spec 003e reads only pinned files)")
        with open(os.path.join(self.root, rel_path), "rb") as fh:
            b = fh.read()
        got = hashlib.sha256(b).hexdigest()
        if got != ent[2]:
            raise UnsupportedOptionError("pins", f"{rel_path}: sha256 {got[:16]} differs from its pin {ent[2][:16]}")
        return b

    def read_record(self, run_id: str, cohort_id: str, key: str, store) -> bytes:
        """A pinned run's store record, read once: refused unless `key` (the manifest's) is the pinned key and the
        bytes are the pinned bytes (sr-naveh 003e finding 3: a manifest whose key points at another record would
        otherwise be read unpinned)."""
        ent = ((self.data.get("records") or {}).get(run_id) or {}).get(cohort_id)
        if ent is None:
            raise UnsupportedOptionError("pins", f"{run_id}/{cohort_id}: the record is not pinned")
        if key != ent["key"]:
            raise UnsupportedOptionError("pins", f"{run_id}/{cohort_id}: the manifest's key {key[:12]} is not the "
                                                 f"pinned key {ent['key'][:12]}")
        with open(store._path(key), "rb") as fh:
            b = fh.read()
        got = hashlib.sha256(b).hexdigest()
        if got != ent["sha256"]:
            raise UnsupportedOptionError("pins", f"{run_id}/{cohort_id}: the store record's sha256 {got[:16]} differs "
                                                 f"from its pin {ent['sha256'][:16]}")
        return b

    def check_record(self, run_id: str, cohort_id: str, store) -> str:
        ent = ((self.data.get("records") or {}).get(run_id) or {}).get(cohort_id)
        if ent is None:
            raise UnsupportedOptionError("pins", f"{run_id}/{cohort_id}: the record is not pinned")
        got = _sha256(store._path(ent["key"]))
        if got != ent["sha256"]:
            raise UnsupportedOptionError("pins", f"{run_id}/{cohort_id}: the store record's sha256 {got[:16]} differs "
                                                 f"from its pin {ent['sha256'][:16]}")
        return got


def load_pins(path: str, root: str = DEFAULT_TESTER) -> Pins:
    import json
    root = _tester_root(root)
    with open(path, encoding="utf-8") as fh:
        return Pins(json.load(fh), root, path)


def load_pinned_sets(mods, manifest: dict, store_root: str, *, pins: Pins | None = None, run_id: str | None = None,
                     cohort_ids=None):
    """The ClaimSets a manifest keys, restricted to `cohort_ids`. When `run_id` names a pinned tester run, each record
    is read once, refused unless the manifest's key is the pinned key and the bytes are the pinned bytes, and parsed
    from those bytes as the store's own get() parses them; sraging's own port records are not tester data and carry
    no pin (the tester's load_results reads them)."""
    import json
    S = mods["store.store"]
    store = S.ResultStore(store_root)
    ids = list(cohort_ids) if cohort_ids is not None else list(manifest["keys"])
    if run_id is None:
        sub = dict(manifest, cohort_ids=ids, keys={c: manifest["keys"][c] for c in ids})
        return mods["run.execute"].load_results(sub, store)
    if pins is None:
        raise UnsupportedOptionError("pins", f"run {run_id} is tester data: reading it needs the pins")
    return [S.CalibrationRecord.from_dict(json.loads(pins.read_record(run_id, c, manifest["keys"][c], store))).claim_set
            for c in ids]


def require_library(library, **manifests):
    """Refuse any run manifest (method, reference, null; None skipped) labelled with a library other than the one
    named: the tester's require_comparable checks the two sides against each other, not against the setting, and two
    v5 runs scored on v6's truths would be a fictional pairing (spec 003e FR-001)."""
    for side, m in manifests.items():
        if m is not None and m["library_version"] != library:
            raise UnsupportedOptionError("library", f"the {side} run {m.get('run_id', '?')} is labelled "
                                                    f"{m['library_version']}, not the library named, {library}")


def compare_v5(mods, method: dict, reference: dict, null: dict | None = None, *, library, pins: Pins,
               tester_root=DEFAULT_TESTER, cohort_ids=None) -> dict:
    """The tester's own comparison (`report.comparison.build_comparison`, called, never copied: spec 003e C3) of a
    method's records against a reference, with the data-ignoring null beside every coverage. `method`, `reference` and
    `null` are {run_id, store_root} for a tester run (its manifest is read from the pinned run file; one passed with
    it must equal that file), {manifest, store_root, run_id None} for sraging's own stored records, or
    {manifest, sets, computed_on, run_id None} for sraging's own claims made outside a store. Every tester file read
    is read once and checked against `pins`; the _h port is refused off v2 here too (US2 scenario 4: "run or
    paired"). The returned dict is build_comparison's, unchanged, plus `sraging_provenance`."""
    tester_root = _tester_root(tester_root)
    import json
    root = library_root(library, tester_root)
    lib_rel = os.path.relpath(os.path.join(root, "manifest.json"), tester_root)
    lib_bytes = pins.read_file(lib_rel)
    lib_sha = hashlib.sha256(lib_bytes).hexdigest()
    computed_on = {}

    def resolve(name, side):
        """A tester run is read from its pinned run file here, never taken on trust: a manifest passed with it must
        equal the file's content."""
        if side is None:
            return None
        if side.get("run_id") is None:
            return side
        if "sets" in side:
            raise UnsupportedOptionError("sets", f"run {side['run_id']} is tester data: it is read from the pinned "
                                                 "store, never passed in")
        man = json.loads(pins.read_file(os.path.join("runs", f"{side['run_id']}.json")))
        if "manifest" in side and side["manifest"] != man:
            raise UnsupportedOptionError("pins", f"the {name} manifest passed for run {side['run_id']} is not the "
                                                 f"pinned run file's content")
        return dict(side, manifest=man)
    method, reference, null = resolve("method", method), resolve("reference", reference), resolve("null", null)
    for side in (method, reference, null):
        if side is not None:
            _refuse_h_off_v2(side["manifest"].get("method"), library)
    require_library(library, method=method["manifest"], reference=reference["manifest"],
                    null=null["manifest"] if null else None)
    mods["reference.pointer"].require_comparable(method["manifest"], reference["manifest"])
    Library = mods["library.load"].Library

    def sets(side):
        if "sets" in side:                      # sraging's own claims, made outside any store (spec 003e jcross)
            keep = None if cohort_ids is None else set(cohort_ids)
            out = [cs for cs in side["sets"] if keep is None or cs.cohort_id in keep]
            _require_same_arrays(side, [cs.cohort_id for cs in out])
            return out
        return load_pinned_sets(mods, side["manifest"], side["store_root"], pins=pins, run_id=side.get("run_id"),
                                cohort_ids=cohort_ids)

    def _require_same_arrays(side, ids):
        """Claims computed on another library count as claims on `library` only for cohorts whose arrays are the
        same in both (the libraries' own per-cohort checksums; sr-naveh 003e finding 4)."""
        on = side.get("computed_on")
        if on is None:
            raise UnsupportedOptionError("computed_on", "sraging's own claim sets must name the library they were "
                                                        "computed on (computed_on)")
        if on == library:
            return
        on_path = os.path.join(library_root(on, tester_root), "manifest.json")
        with open(on_path, "rb") as fh:
            on_bytes = fh.read()
        computed_on[on] = hashlib.sha256(on_bytes).hexdigest()
        a, b = json.loads(on_bytes)["checksums"], json.loads(lib_bytes)["checksums"]
        bad = [c for c in ids if c not in a or a.get(c) != b.get(c)]
        if bad:
            raise UnsupportedOptionError("computed_on", f"{len(bad)} cohort(s) computed on {on} have arrays that "
                                                        f"differ in {library} (first {bad[:3]}): not claims on {library}")
    out = mods["report.comparison"].build_comparison(Library(root=root), sets(method), sets(reference),
                                                     method["manifest"], reference["manifest"],
                                                     null_sets=sets(null) if null else None)
    clone = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(mods["report.comparison"].__file__))))
    out["sraging_provenance"] = {
        "library": library, "library_manifest_sha256": lib_sha,
        "tester_clone": clone, "tester_clone_head": subprocess.run(
            ["git", "-C", clone, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "tester_data_root": tester_root, "pins_sha256": pins.sha256(),
        "runs": {k: (v or {}).get("run_id") for k, v in (("method", method), ("reference", reference),
                                                        ("null", null))},
        "cohort_ids": None if cohort_ids is None else len(list(cohort_ids)),
        "computed_on_manifest_sha256": computed_on,                  # read for the checksum rule, not pinned
        "runs_sha256": {k: pins._by_path[os.path.join("runs", f"{v['run_id']}.json")][2]
                        for k, v in (("method", method), ("reference", reference), ("null", null))
                        if v is not None and v.get("run_id") is not None}}
    return out
