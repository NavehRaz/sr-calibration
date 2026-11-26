"""`mle_profile_jax2` and `mle_profile_jax2_h` -- successors to `mle_profile_jax` v0.3.3, which stays FROZEN.

ONE package, TWO registered names, ONE base class (`_MleProfileJax2Base`). Each name fixes its observation
model and likelihood with class constants, never with a runtime flag: srbench's store key takes its
settings from the run CALLER, not from the method, so a runtime switch would put two behaviours under one
key (PHASE_B_DESIGN.txt section 0). Every behaviour change therefore needs a new VERSION, and
VERSION_LOCK.json + tests/test_jax2_version_lock.py fail when the code changes but the version does not.

    name                  HAZARD  WINDOW  RELABEL   capabilities (derived from the constants)
    mle_profile_jax2      False   False   True      ('censoring',)
    mle_profile_jax2_h    True    True    True      ('external_hazard', 'fit_window', 'censoring')

What jax2 is, against v0.3.3 (8b4c2f1) -- the jax2 rc1 candidate (Phase A), ported:
  1. fix B, `fp_jax.py`: a death's interval mass from the absorbed mass F = 1 - S where that is the
     difference with the smaller operands (no S(t) - S(t+delta) round-off on large, sharp cohorts);
  2. v1 + v2, `estimator.py`: re-fit on a witnessed better profile node, library starts after a re-fit;
  3. the floored dt ceiling, `estimator.solver_grid` / `floor_info`;
  4. the censoring-tie relabel, `observation_model.censor_tie_relabel` (rc1's rule), ON for both names.
     Phase A's CENSOR_TIE set decides whether it stays on (then 'censoring' stays declared) or goes off.

and `mle_profile_jax2_h` is jax2 plus (B2, 2026-10-02):
  5. a fitted constant external hazard h (survival S(t) e^{-ht}; `fp_jax.make_loglik(hazard=True)`), a
     NUISANCE parameter: never claimed -- the tester scores eta, beta, epsilon and kappa only -- but every
     record carries `diagnostics.hazard` (h, hL, ExtH = -ln h, interval, verdict, profile, the MLE start
     table, the Makeham status, the ladder) and a full would-be claim `diagnostics.hazard.h_claim` built
     with the same mapping as the four claims (`h_claim_kwargs`), so h can be scored later without a re-fit;
  6. the fit window: the observation model snaps it to the delta grid and censors at hi_eff
     (`observation_model.window_snap`), the likelihood left-truncates at lo_eff.

and both names (rc2 I1, SELF_CONSISTENT, on for dev4):
  7. claim self-consistency: every claim (and the would-be h claim) contains the value of every point the fit
     evaluated within chi2_1(0.95)/2 of the best log-likelihood (the profile there is at least that point's
     value); a widened side that reaches the box edge opens (`final_claim_kwargs`, `sc_widen_claim`;
     the archive is `estimator._sc_archive` / `sc_extents`, read from the likelihood memo at no cost).
     MEASURED (Phase A records, post hoc): rc1's claims excluded such points on 49 of 237 records.
and both names (rc2 I3, FLOOR_EARLY, on for dev4):
  8. the floor early-time fix: where the floored dt ceiling binds, the march starts on a graded mesh (20 BE
     steps of 1e-8, then x1.02 per step up to the floor dt; variable-step BDF2; `fp_jax.early_schedule`,
     `estimator._grid(floor_early=True)`). MEASURED (reports/rc2/I3-floor): the uniform floor grid put a
     death recorded in the first sampling bin 1.9 nats off (het-real-ecoli-c0-s1; Phase A RO1).
and both names (rc2 I2, N_RULE, on for dev4; DT_RULE off):
  9. the N rule for large n: the space grid N = 900 * 2^k whose predicted grid error of the claim-relevant
     deviances is <= 0.2 nats (from n and the Kaplan-Meier CV), capped at 3,600, composed with the floor
     (`estimator._grid(n_rule=True)`, `_n_rule`). MEASURED (reports/rc2/I2): N = 900 misplaced those deviances by
     0.6-3 nats on n = 1e5 human cohorts and up to 4.8 nats on sharp synthetic cohorts. DT_RULE (dt <= CV/80, the
     time error that dominates on CV < 0.05 cohorts) is built but off: it is a separate decision (CPU x2.8).
and (rc2 I4, SHADOW and EXACT_F, on for dev4): _h's shadow chains in the other kappa basin (see below) and the
exact-time density from the absorbed mass.
and both names (rc2 D1, EARLY_ALL, on since dev5; ratified R-8 "extend to all"; growth x1.05 since dev6):
 10. a graded early-time mesh (20 BE steps of 1e-8, then x1.05 per step up to the grid's dt) on EVERY grid with a
     sampling_dt, floor-bound grids included, where it replaces item 8's x1.02 -- one ratio everywhere
     (`estimator._grid(early_all=True)`, `_EARLY_ALL_MESH`, note 12 there). MEASURED (reports/rc2/I3-floor section
     8): the production (non-floor) grid put deaths in the first sampling bins up to 5.7 nats off on two n = 1e5
     Denmark cohorts and 3.1 nats on a synthetic pure cohort; validated at the likelihood level in reports/rc2/dev5/
     (x1.02) and reports/rc2/dev6/ (x1.05). Exact-time data keep the uniform grid (EXACT_F's density assumes it).
Both versions are `1.0.0.devN` until the code passes L0-L5 and Phase C.

IMPORTS. The top level imports the standard library and `srbench.contract` only; numpy, SciPy, JAX and the
system under test (SRK) load lazily inside `calibrate` / `run_fit`, so `srbench methods` and discovery
work without them (tested in a subprocess with `sys.modules['jax'] = None`). The two names are registered
only when this file is loaded under the discovery key `srbench_methods.<this directory's name>` (see the end
of the file: pytest's package collection imports it under another name).

SIBLINGS load under the key `srbench_methods.<this directory's name>.<module>` (`_sibling`), and every
load or cache hit is checked against this directory's file, so a second copy of the package (or the
frozen `mle_profile_jax`, whose key is `srbench_methods.mle_profile_jax.<module>`) can share a process
without either one silently running the other's code.

ENVIRONMENT GUARDS (`env_guard`, `RefusedEnvironment`). `calibrate` refuses -- aborts the run, nothing
stored -- when SRK_DT_CEILING is set (it changes SRK.auto_grid under every grid rule), when the system
under test does not resolve to the requested checkout, or when a sibling key is taken by another file.
Every record carries `diagnostics.srk_sha` (sha256 of each SRK module file loaded),
`diagnostics.observation_model` and `diagnostics.exc_counts`.

VERSION LOCK AT RUN TIME (B5, `version_lock_guard`). `calibrate` and `run_fit` also refuse when the code
that would run is not the code VERSION_LOCK.json records for the running version: a runtime file whose
sha256 differs from the lock while VERSIONS[name] equals the locked version, a stale lock (VERSIONS moved,
lock not rewritten), a missing or unreadable lock, or a sibling module whose file changed on disk after it
was loaded in this process. WHY: the version is part of the store key, so a code change under an unchanged
version would be served from the store SILENTLY; tests/test_jax2_version_lock.py catches it at test time,
but the package tests are not in the tester's bare `pytest` run (D-B1-8), so the run itself must refuse.
Cost: hashing four small files (< 1 ms).
rc2 I4: ... AND ON A FULLY CACHED RUN. srbench's `plan_run` / `run` never call `calibrate` for a stored key, so
B5's guard missed a run whose every key was stored. `descriptor` is now a property that checks the lock on
every access: inside a run (`srbench.run.execute` / `srbench.run.probe` on the call stack) a broken lock REFUSES
before a single key is built; elsewhere (discovery, `srbench methods`, the explorer) it reports the version as
`<version>+UNLOCKED.<digest>`, so discovery never breaks for the other methods and no key can reach a stored
record. Tested end to end through `execute.run` with a temporary store (tests/test_jax2_rc2_i4.py).
rc2 I4 also adds two switches, ON for dev4 (OFF = dev3's call): SHADOW (`estimator.fit(shadow=True)`, the
hazard path only: the jax2-like MLE start, shadow chains in the other kappa basin, the inner-edge polish) and
EXACT_F (`fit(exact_from_F=True)`: exact-time deaths scored by the density from the absorbed mass).

The claim mapping is `methods/mle_profile_jax/__init__.py`'s (= `methods/mle_profile`'s) unchanged; h is
never claimed (the tester scores eta, beta, epsilon and kappa only).

VERSIONS: 1.0.0.dev1 = B1 scaffold (nothing stored); 1.0.0.dev2 = B2 (hazard + window built; nothing stored);
1.0.0.dev3 = B5 (the run-time version-lock guard; with the hazard on, a death beyond the march's early stop
scores the -700 floor as it does without the hazard; nothing stored);
1.0.0.dev4 = rc2 R2 (2026-10-03): I1 + I2 + I3 + I4 integrated, each behind its own switch, the dev4 defaults
being the class constants of `_MleProfileJax2Base`:
    switch           constant          dev4   estimator.fit / make_loglik keyword       what it changes
    claim widening   SELF_CONSISTENT   on     fit(self_consistent=True)                 claims (and h_claim) only
    N rule           N_RULE            on     fit(n_rule=True)                          N (space grid) only
    CV-aware dt      DT_RULE           OFF    fit(dt_rule=True)                         dt only (built, not adopted)
    floor early mesh FLOOR_EARLY       on     fit(floor_early=True)                     the 48 floor-bound cohorts
    shadow chains    SHADOW            on     fit(shadow=True)                          _h (hazard path) only
    exact-time F     EXACT_F           on     fit(exact_from_F=True)                    exact-time data only
    tie relabel      RELABEL           on     observation_model.prepare(relabel_on=)    censoring ties
  With every new switch off (`run_fit`'s defaults) the estimator is called with dev3's exact keyword set and the
  fit is dev3's bit for bit (for jax2 that is also the rc1 candidate's: tests/identity_rc1.py; for _h, the stored
  dev3 L5 fits; reports/rc2/R2/). Nothing is stored under dev4 until PREREG_C (reports/rc2/PREREG_C.txt) is
  ratified.
1.0.0.dev5 = rc2 D1 (2026-10-03): dev4 plus ONE switch, ON for dev5 (ratification R-8, reports/rc2/RATIFICATION.txt):
    early mesh, all  EARLY_ALL         on     fit(early_all=True)                       every grid with a sampling_dt
  With EARLY_ALL off (`run_fit(early_all=False)`, the default) the estimator gets dev4's exact keyword set and the
  fit is dev4's bit for bit (reports/rc2/dev5/identity: 3 quick cohorts, both names). Where the floor binds, the
  grid is the same as dev4's (FLOOR_EARLY already marched this mesh there).
1.0.0.dev6 = rc2 E1 (2026-10-03): dev5 with the EARLY_ALL mesh's growth x1.05 instead of x1.02 (Naveh's addendum to
  the ratification, reports/rc2/RATIFICATION.txt: x1.02 cost x1.72 per value+grad evaluation, x1.05 about x1.31).
  One ratio everywhere: with EARLY_ALL on, the floor-bound grids march the x1.05 mesh too (estimator._EARLY_ALL_MESH
  replaces _FLOOR_EARLY there). No new switch. With EARLY_ALL off (`run_fit`'s default) the fit is dev4's bit for
  bit, the floor-bound grids keeping FLOOR_EARLY's x1.02 (reports/rc2/dev6/identity: 3 quick cohorts, both names).
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
import time

from srbench.contract.method import CohortCharacteristics, register
from srbench.contract.types import Claim, ClaimSet, CostDeclaration, MethodDescriptor, Observation

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.basename(_HERE)
_KEY = f"srbench_methods.{_PKG}"

#: name -> version. Read by tests/version_lock.py with `ast.literal_eval`, so keep it a literal dict.
VERSIONS = {
    "mle_profile_jax2": "1.0.0",
    "mle_profile_jax2_h": "1.0.0",
}

BOUNDS = {
    "eta": (1e-12, 1e3),
    "beta": (1e-12, 1e3),
    "epsilon": (1e-12, 1e3),
    "kappa": (1e-3, 300.0),
}
KAPPA_MAX = 300.0
LEVEL = 0.95

VERDICT_TO_SHAPE = {
    "identified": "closed",
    "one-sided:<": "upper_bound",
    "one-sided:>": "lower_bound",
    "flat": "none",
    "insufficient": "none",
}

#: rc2 I1 (claim self-consistency, `sc_widen_claim`): a side the widening moves to within SC_EDGE_DEX of the box
#: edge becomes OPEN. The walk's own box-edge stop is at 0.98-0.99 x the edge (0.004-0.009 dex), and a nuisance
#: on an active bound sits exactly on it, so 0.01 dex means "at the box edge", not "near it".
SC_EDGE_DEX = 0.01

#: The SRK modules the estimator imports; loaded before the fit so that their sha256 is taken first.
_SRK_MODULES = ("SRK.fp_likelihood", "SRK.fp_gof", "SRK.fp_curvefit")


class RefusedEnvironment(SystemExit):
    """Raised INSTEAD of returning a ClaimSet when the environment would make the numbers not this method's.

    A SystemExit, not an Exception, on purpose: `srbench.run.execute.run` stores every Exception from
    `calibrate` as an `error:` record under this name/version/cohort key, and a stored record is never
    recomputed (the store answers the key). MEASURED: mle_profile_jax 0.3.0's 30 stored quick-tier records
    are all `error:ProvenanceError` -- a launch without SRK on PYTHONPATH, stored as results. A refusal
    aborts the run instead and leaves every key free."""


def env_guard():
    """Refuse when SRK_DT_CEILING is set at all (it changes SRK.auto_grid under every grid rule)."""
    if "SRK_DT_CEILING" in os.environ:
        raise RefusedEnvironment(
            f"REFUSING ({_PKG}): SRK_DT_CEILING is set ({os.environ['SRK_DT_CEILING']!r}). It changes "
            "SRK.fp_likelihood.auto_grid under every grid rule, so the fit would not be this method's. "
            "Unset it and run again.")


def _same_file(a, b):
    return os.path.realpath(a) == os.path.realpath(b)


def _sibling(name):
    """Load a sibling module by path under `srbench_methods.<this directory>.<name>`, cached in sys.modules
    so jitted functions are built once per process. Discovery imports this package without a parent
    package, so relative imports do not resolve.

    The key is DERIVED from this directory's name (the frozen v0.3.3 adapter hard-codes its own), and every
    return -- a fresh load or a cache hit -- is checked to be THIS directory's file: a key taken by another
    file refuses (`RefusedEnvironment`) rather than running someone else's code."""
    key = f"{_KEY}.{name}"
    path = os.path.join(_HERE, f"{name}.py")
    mod = sys.modules.get(key)
    if mod is None:
        sha0 = _sha256(path)                    # B5: what is about to be executed (version_lock_guard)
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(key, None)          # never leave a half-initialised module under the key
            raise
        if _sha256(path) != sha0:
            sys.modules.pop(key, None)
            raise RefusedEnvironment(f"REFUSING ({_PKG}): {path} changed on disk while it was being loaded.")
        _LOADED_SHA[name] = sha0
    f = getattr(mod, "__file__", None)
    if not f or not _same_file(f, path):
        raise RefusedEnvironment(
            f"REFUSING ({_PKG}): sys.modules[{key!r}] is {f!r}, not {path!r}. Another copy of this "
            "package was loaded under the same key in this process.")
    _LOADED_SHA.setdefault(name, _sha256(path))  # loaded under the key by someone else: best effort
    return mod


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: B5: sha256 of every package file executed in this process (this file at import; siblings in `_sibling`).
_LOADED_SHA = {"__init__": _sha256(os.path.abspath(__file__))}


def version_lock_problems(pkg_dir=None, versions=None, loaded=None):
    """B5: every reason the code that would run is not the code VERSION_LOCK.json locks ([] = consistent).

    The rule is tests/version_lock.py's (runtime files = every *.py directly in the package directory; if a
    name's version equals its locked version, every runtime file's sha256 must equal the locked one), applied
    at RUN time, plus: a stale lock (a name's version differs from its locked version: the code under the new
    version is not locked yet) and a missing / unreadable lock are problems too, and so is a package file
    loaded in this process whose file on disk has changed since (`loaded`: name -> sha256 at load).
    Standard library only; never imports the siblings."""
    import json
    pkg_dir = _HERE if pkg_dir is None else pkg_dir
    versions = VERSIONS if versions is None else versions
    loaded = _LOADED_SHA if loaded is None else loaded
    lock_path = os.path.join(pkg_dir, "VERSION_LOCK.json")
    try:
        with open(lock_path) as fh:
            lock = json.load(fh)
    except (OSError, ValueError) as ex:
        return [f"cannot read {lock_path}: {type(ex).__name__}: {ex}"]
    files = sorted(f for f in os.listdir(pkg_dir) if f.endswith(".py") and os.path.isfile(os.path.join(pkg_dir, f)))
    now = {f: _sha256(os.path.join(pkg_dir, f)) for f in files}
    out = []
    names = lock.get("names") or {}
    for name in sorted(versions):
        ent = names.get(name)
        if not isinstance(ent, dict):
            out.append(f"{name} {versions[name]} is not in VERSION_LOCK.json")
            continue
        if ent.get("version") != versions[name]:
            out.append(f"VERSION_LOCK.json is stale: {name} is {versions[name]} but locked at {ent.get('version')!r} "
                       "(rewrite the lock: python tests/version_lock.py --write)")
            continue
        locked = ent.get("sha256") or {}
        diff = sorted((set(locked) ^ set(now)) |
                      {f for f in set(locked) & set(now) if locked[f] != now[f]})
        if diff:
            out.append(f"CODE CHANGED BUT VERSION DID NOT: {name} is still {versions[name]} but {diff} differ from "
                       "VERSION_LOCK.json")
    for name, sha in sorted(loaded.items()):
        p = os.path.join(pkg_dir, f"{name}.py")
        if os.path.isfile(p) and _sha256(p) != sha:
            out.append(f"{name}.py changed on disk after it was loaded in this process")
    return out


def version_lock_guard():
    """B5: refuse (RefusedEnvironment: nothing stored, the key stays free) unless `version_lock_problems()` is
    empty. Called first thing by `calibrate` and `run_fit`."""
    p = version_lock_problems()
    if p:
        raise RefusedEnvironment(
            f"REFUSING ({_PKG}): the code is not the code VERSION_LOCK.json records for this version -- "
            + "; ".join(p) + ". A record stored under this version would not be this code's: bump VERSIONS "
            "and rewrite the lock (python tests/version_lock.py --write).")


def srk_fingerprint():
    """{root, files: {relative path: sha256}, digest, n_files}: every SRK module file loaded in this process.

    It hashes the files on disk NOW, so it identifies the code that ran only if nothing edited them after
    they were imported; `run_fit` takes it before and after the fit and records `changed_during_fit`."""
    import SRK
    root = os.path.realpath(os.path.dirname(SRK.__file__))
    files = {}
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "SRK" or name.startswith("SRK.")):
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        rf = os.path.realpath(f)
        if not rf.startswith(root + os.sep) or not os.path.isfile(rf):
            continue
        files[os.path.relpath(rf, root)] = _sha256(rf)
    digest = hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    return dict(root=root, n_files=len(files), digest=digest, files=dict(sorted(files.items())))


def run_fit(observation: Observation, *, relabel: bool, hazard: bool = False, window: bool = False,
            self_consistent: bool = False, floor_early: bool = False, n_rule: bool = False,
            dt_rule: bool = False,
            shadow: bool = False, exact_from_F: bool = False, early_all: bool = False) -> dict:
    """The estimator's fit on this observation after the observation model. Raises on failure (the caller --
    `calibrate`, or a driver -- decides how a failure is recorded).

    The registered names call it with their class constants; the keyword arguments exist for the identity
    harness (tests/identity_rc1.py) and must never become a runtime setting of a registered name.
    Returns `estimator.fit`'s dict with `diagnostics.censor_tie` (rc1's record), `.observation_model` and
    `.srk_sha` added (`.exc_counts` comes from the estimator), and with the hazard on
    `diagnostics.hazard.h_claim` (the would-be claim on h).
    rc2 I1: `self_consistent=True` also returns `diagnostics.self_consistency` (the estimator's archive extents)
    and widens the would-be h claim (`h_claim`; the profile's own is kept as `h_claim_profile`, the record as
    `h_claim_sc`). The four SR claims are widened by the claim builder (`final_claim_kwargs`). False (the
    default) calls the estimator exactly as dev3 did.
    rc2 I3: `floor_early=True` marches the graded early-time mesh where the floored dt ceiling binds
    (`estimator.fit(floor_early=True)`); False (the default) calls the estimator exactly as dev3 did.
    rc2 I2: `n_rule=True` / `dt_rule=True` solve on the N rule's grid / cap dt at CV/80
    (`estimator.fit(n_rule=, dt_rule=)`); False (the default) calls the estimator exactly as dev3 did.
    rc2 D1 (dev5) / E1 (dev6): `early_all=True` marches the graded early-time mesh (x1.05 since dev6) on every grid
    with a sampling_dt, floor-bound grids included (`estimator.fit(early_all=True)`); False (the default) calls the
    estimator exactly as dev4 did."""
    env_guard()
    version_lock_guard()                                  # B5: before loading or running any package code
    from srbench import npcompat  # noqa: F401  (numpy>=2 shim, before any SRK import)
    om = _sibling("observation_model")
    fj = _sibling("fp_jax")
    est = _sibling("estimator")
    version_lock_guard()                                  # B5: and what was just loaded is that code too
    for m in _SRK_MODULES:
        importlib.import_module(m)
    srk0 = srk_fingerprint()
    prep = om.prepare(observation.ages, observation.events, observation.delta, observation.window,
                      window_on=window, relabel_on=relabel)
    res = est.fit(fj, prep.ages, prep.events, delta=observation.delta, levels=(LEVEL,), kappa_max=KAPPA_MAX,
                  hazard=hazard, window=prep.window_eff,
                  **({"shadow": True} if shadow else {}),                     # rc2 I4: off = dev3 call
                  **({"exact_from_F": True} if exact_from_F else {}),
                  **({"self_consistent": True} if self_consistent else {}),   # rc2 I1: off = dev3's call
                  **({"floor_early": True} if floor_early else {}),           # rc2 I3: off = dev3's call
                  **({"n_rule": True} if n_rule else {}),                     # rc2 I2: off = dev3's call
                  **({"dt_rule": True} if dt_rule else {}),                   # rc2 I2: off = dev3's call
                  **({"early_all": True} if early_all else {}))               # rc2 D1: off = dev4's call
    srk1 = srk_fingerprint()
    files = dict(srk1["files"], **srk0["files"])          # a module's hash from BEFORE the fit wins
    changed = sorted(k for k, v in srk1["files"].items() if k in srk0["files"] and srk0["files"][k] != v)
    srk = dict(root=srk0["root"], n_files=len(files), files=dict(sorted(files.items())),
               digest=hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(files.items())).encode()).hexdigest(),
               changed_during_fit=changed, srk_dt_ceiling_env=os.environ.get("SRK_DT_CEILING"))
    dg = res.setdefault("diagnostics", {})
    dg["censor_tie"] = prep.censor_tie
    dg["observation_model"] = prep.record
    dg["srk_sha"] = srk
    dg.setdefault("exc_counts", {})
    if hazard and isinstance(dg.get("hazard"), dict) and dg["hazard"].get("verdict") is not None:
        hz = dg["hazard"]
        hz["h_claim"] = h_claim_kwargs(res["theta"][4], res["ci"], dg.get("verdicts") or {}, hz["box"])
        if self_consistent:                               # rc2 I1: the would-be h claim gets the same widening
            ext_h = ((dg.get("self_consistency") or {}).get("extents") or {}).get("h")
            hz["h_claim_profile"] = hz["h_claim"]
            hz["h_claim"], hz["h_claim_sc"] = sc_widen_claim(hz["h_claim"], ext_h, hz["box"])
    return res


def h_claim_kwargs(h_point, ci, verdicts, box):
    """The WOULD-BE claim on h, as a keyword dict -- the mapping of `claim_kwargs` applied to h, with the h box
    as its bounds. NOT returned as a Claim (the tester scores the four SR parameters only); stored in
    `diagnostics.hazard.h_claim` so that scoring h later needs no re-fit."""
    lo, hi = ci.get(f"h@{LEVEL}", (float("nan"), float("nan")))
    shape = VERDICT_TO_SHAPE.get(verdicts.get("h"), "none")
    lo_ok = lo is not None and math.isfinite(lo)
    hi_ok = hi is not None and math.isfinite(hi)
    if shape == "closed" and not (lo_ok and hi_ok):
        shape = "upper_bound" if hi_ok else ("lower_bound" if lo_ok else "none")
    if shape == "upper_bound" and not hi_ok:
        shape = "none"
    if shape == "lower_bound" and not lo_ok:
        shape = "none"
    kw = {"parameter": "h", "shape": shape, "level": LEVEL, "verdict": verdicts.get("h"),
          "bounds": [float(box[0]), float(box[1])]}
    if shape in ("closed", "lower_bound"):
        kw["lower"] = float(max(lo, box[0]))
    if shape in ("closed", "upper_bound"):
        kw["upper"] = float(min(hi, box[1]))
    if shape != "none":
        v = float(h_point)
        if math.isfinite(v):
            kw["point"] = float(min(max(v, box[0]), box[1]))
    return kw


def claim_kwargs(theta, ci, verdicts):
    """The claims as keyword dicts. ---- claim mapping: verbatim from methods/mle_profile ----
    Only the four SR parameters are claimed; theta[4] (h, once it exists) never is."""
    point = {"eta": theta[0], "beta": theta[1], "epsilon": theta[2], "kappa": theta[3]}
    claims = []
    for p in ("eta", "beta", "epsilon", "kappa"):
        lo, hi = ci.get(f"{p}@{LEVEL}", (float("nan"), float("nan")))
        shape = VERDICT_TO_SHAPE.get(verdicts.get(p), "none")
        lo_ok = lo is not None and math.isfinite(lo)
        hi_ok = hi is not None and math.isfinite(hi)
        if shape == "closed" and not (lo_ok and hi_ok):
            shape = "upper_bound" if hi_ok else ("lower_bound" if lo_ok else "none")
        if shape == "upper_bound" and not hi_ok:
            shape = "none"
        if shape == "lower_bound" and not lo_ok:
            shape = "none"
        kw = {"parameter": p, "shape": shape, "level": LEVEL}
        if shape in ("closed", "lower_bound"):
            kw["lower"] = float(max(lo, BOUNDS[p][0]))
        if shape in ("closed", "upper_bound"):
            kw["upper"] = float(min(hi, BOUNDS[p][1]))
        if shape != "none":
            v = point[p]
            if math.isfinite(v):
                kw["point"] = float(min(max(v, BOUNDS[p][0]), BOUNDS[p][1]))
        claims.append(kw)
    return claims


def sc_widen_claim(kw, ext, box, edge_dex=SC_EDGE_DEX):
    """rc2 I1: ONE claim (a `claim_kwargs` / `h_claim_kwargs` dict) made consistent with the fit's own points.

    `ext` is the parameter's entry of `diagnostics.self_consistency.extents` (`estimator.sc_extents`): the
    smallest and largest value over every evaluated point within thr/2 of the best log-likelihood. Each such
    point lies inside the parameter's likelihood-ratio set (the profile there is at least the point's value),
    so the claim must contain it. The claim becomes the UNION of the profile interval and [qmin, qmax]:
      * a stated edge that a point lies beyond moves out to that point;
      * a moved edge that lands within `edge_dex` of the box edge OPENS that side: closed -> lower_bound /
        upper_bound (-> none if both open), lower_bound / upper_bound -> none (a decline carries no point);
      * an open side stays open, a stated edge no point lies beyond is untouched, a "none" stays "none",
        and the point estimate never moves.
    Returns (claim dict, record). The claim dict is a copy of `kw`, equal to it when nothing moved."""
    shape0 = kw["shape"]
    lo0, hi0 = kw.get("lower"), kw.get("upper")
    rec = {"parameter": kw["parameter"], "shape_before": shape0, "shape_after": shape0,
           "lower_before": lo0, "upper_before": hi0, "lower_after": lo0, "upper_after": hi0,
           "lower_moved_dex": 0.0, "upper_moved_dex": 0.0, "opened": [], "changed": False}
    if shape0 == "none" or not ext or not ext.get("n"):
        return dict(kw), rec
    blo, bhi = float(box[0]), float(box[1])
    near = 10.0 ** float(edge_dex)
    lo, hi = lo0, hi0
    qmin, qmax = ext.get("qmin"), ext.get("qmax")
    if lo is not None and qmin is not None and qmin < lo:
        if qmin <= blo * near:
            rec["opened"].append("lower")
            rec["lower_moved_dex"] = math.log10(lo0 / blo) if blo > 0 else float("inf")
            lo = None
        else:
            lo = max(float(qmin), blo)
            rec["lower_moved_dex"] = math.log10(lo0 / lo)
    if hi is not None and qmax is not None and qmax > hi:
        if qmax >= bhi / near:
            rec["opened"].append("upper")
            rec["upper_moved_dex"] = math.log10(bhi / hi0)
            hi = None
        else:
            hi = min(float(qmax), bhi)
            rec["upper_moved_dex"] = math.log10(hi / hi0)
    if lo == lo0 and hi == hi0:
        return dict(kw), rec
    shape = ("closed" if lo is not None and hi is not None else "lower_bound" if lo is not None
             else "upper_bound" if hi is not None else "none")
    new = dict(kw)
    new["shape"] = shape
    for key, v in (("lower", lo), ("upper", hi)):
        if v is None:
            new.pop(key, None)
        else:
            new[key] = float(v)
    if shape == "none":
        new.pop("point", None)
    rec.update(shape_after=shape, lower_after=new.get("lower"), upper_after=new.get("upper"), changed=True)
    return new, rec


def sc_widen_claims(claims, sc, bounds=None, edge_dex=SC_EDGE_DEX):
    """rc2 I1: `sc_widen_claim` over a claim list, with `sc` = `diagnostics.self_consistency` (the estimator's
    `sc_extents`) and `bounds` = the claimed parameters' box (BOUNDS). Returns (claims, records)."""
    bounds = BOUNDS if bounds is None else bounds
    ext = (sc or {}).get("extents") or {}
    out, recs = [], []
    for kw in claims:
        p = kw["parameter"]
        new, rec = sc_widen_claim(kw, ext.get(p), bounds[p], edge_dex=edge_dex)
        out.append(new)
        recs.append(rec)
    return out, recs


def final_claim_kwargs(res, self_consistent):
    """The claims a registered name returns for `run_fit`'s result: `claim_kwargs` (v0.3.3's mapping, verbatim),
    then, with `self_consistent`, `sc_widen_claims`. Returns (claims, block): block is None when the widening is
    off, else `diagnostics.self_consistency` plus the widening record (`widening`, `claims_profile` = the claims
    before it, `n_changed`, `n_shape_changes`). Scripts that rebuild a name's claims call this, not claim_kwargs.
    A fit with stated claims but no `self_consistency` block is a programming error (ValueError)."""
    dg = res.get("diagnostics") or {}
    claims = claim_kwargs(list(res["theta"]), res["ci"], dg.get("verdicts") or {})
    if not self_consistent:
        return claims, None
    sc = dg.get("self_consistency")
    if sc is None:
        if any(c["shape"] != "none" for c in claims):
            raise ValueError("mle_profile_jax2: self_consistent claims need diagnostics.self_consistency "
                             "(estimator.fit(..., self_consistent=True))")
        return claims, dict(enabled=False, reason="no self_consistency block (no claim stated)")
    new, recs = sc_widen_claims(claims, sc)
    block = dict(sc, enabled=True, edge_dex=SC_EDGE_DEX, widening=recs, claims_profile=claims,
                 n_changed=sum(1 for r in recs if r["changed"]),
                 n_shape_changes=sum(1 for r in recs if r["shape_after"] != r["shape_before"]))
    return new, block


def capabilities_for(hazard, window, relabel):
    """The declared capabilities, DERIVED from the constants so the two can never disagree."""
    return tuple(c for c, on in (("external_hazard", hazard), ("fit_window", window), ("censoring", relabel))
                 if on)


def _descriptor(cls, description, version=None):
    return MethodDescriptor(
        name=cls.NAME,
        version=VERSIONS[cls.NAME] if version is None else version,
        bounds=dict(BOUNDS),
        capabilities=capabilities_for(cls.HAZARD, cls.WINDOW, cls.RELABEL),
        description=description,
        contact=f"sr_calibration_tester / {cls.NAME}",
    )


#: rc2 I4: the srbench modules whose access to `descriptor` is a RUN -- they build store keys from its version
#: (`plan_run`, `run`, `probe.measure`) and serve the stored records under those keys without calibrating.
_RUN_MODULES = ("srbench.run.execute", "srbench.run.probe")


def _in_run_context(max_depth=200):
    """rc2 I4: True when a frame of `_RUN_MODULES` is on the call stack (a run is planning keys or calibrating)."""
    f = sys._getframe(1)
    depth = 0
    while f is not None and depth < max_depth:
        if f.f_globals.get("__name__") in _RUN_MODULES:
            return True
        f = f.f_back
        depth += 1
    return False


def _unlocked_version(version, pkg_dir=None):
    """rc2 I4: the version a descriptor reports while the lock is broken: `<version>+UNLOCKED.<12 hex>`, the hex a
    digest of the runtime files as they are on disk now. No record is ever stored under it (calibrate refuses),
    and it never equals a real version, so no store key built from it can hit a record of the locked code."""
    pkg_dir = _HERE if pkg_dir is None else pkg_dir
    files = sorted(f for f in os.listdir(pkg_dir) if f.endswith(".py") and os.path.isfile(os.path.join(pkg_dir, f)))
    tag = hashlib.sha256("".join(f"{f} {_sha256(os.path.join(pkg_dir, f))}\n" for f in files).encode()).hexdigest()
    return f"{version}+UNLOCKED.{tag[:12]}"


def _locked_descriptor(cls):
    """rc2 I4: `cls`'s descriptor, with the VERSION LOCK checked on EVERY access (`_CheckedDescriptor`).

    WHY. `version_lock_guard` in `calibrate` / `run_fit` only runs when a cohort is COMPUTED. A run whose every key
    is already stored never calls them: srbench's `plan_run` builds the keys from `descriptor.version` and `run`
    serves the stored records. So a code change under an unchanged version was refused on a partly cached run but
    SILENTLY served on a fully cached one (the B5 re-review's gap). `descriptor` is the one thing of this package
    every such run touches (`plan_run`, `run`, `probe.measure`, each before any store lookup), so the check lives
    here:
      * lock consistent: the registered version, as before;
      * lock broken, and a run is on the call stack (`_in_run_context`: srbench.run.execute / srbench.run.probe):
        REFUSE (RefusedEnvironment, a SystemExit) -- before a single key is built;
      * lock broken elsewhere (discovery's `register`, `srbench methods`, the explorer): a descriptor whose version
        is `<version>+UNLOCKED.<digest>` (`_unlocked_version`), so discovery and listing keep working for every
        other method, the broken state is visible, and no key built from it can reach a record of the locked code
        (and none is stored under it: `calibrate` refuses).
    Cost: hashing the runtime files (< 1 ms) per access."""
    problems = version_lock_problems()
    if not problems:
        return _descriptor(cls, cls.DESCRIPTION)
    if _in_run_context():
        raise RefusedEnvironment(
            f"REFUSING ({cls.NAME}): a run is planning store keys for code that is not the code "
            "VERSION_LOCK.json records for this version -- " + "; ".join(problems) + ". Stored records "
            "under this version are not this code's results: bump VERSIONS and rewrite the lock "
            "(python tests/version_lock.py --write).")
    return _descriptor(cls, cls.DESCRIPTION, version=_unlocked_version(VERSIONS[cls.NAME]))


class _CheckedDescriptor:
    """rc2 I4: `descriptor` on the class (`MleProfileJax2.descriptor`) and on its instances alike, built and
    lock-checked on every access by `_locked_descriptor`."""

    def __get__(self, obj, owner=None):
        return _locked_descriptor(type(obj) if obj is not None else owner)


class _MleProfileJax2Base:
    """Everything but the constants. Subclasses set NAME, HAZARD, WINDOW, RELABEL and DESCRIPTION."""

    NAME = None
    HAZARD = False
    WINDOW = False
    RELABEL = True
    SELF_CONSISTENT = True    # rc2 I1: claims contain every evaluated point within thr/2 of the best (dev4 default)
    N_RULE = True             # rc2 I2: N = 900 * 2^k from n and the KM CV, capped at 3,600 (dev4 default)
    DT_RULE = False           # rc2 I2: dt <= CV/80 for sharp cohorts -- built, OFF pending a decision
    FLOOR_EARLY = True        # rc2 I3: graded early-time mesh where the floored dt ceiling binds (dev4 default)
    EARLY_ALL = True          # rc2 D1/E1: the graded mesh (x1.05 since dev6) on EVERY grid with a sampling_dt (R-8)

    SHADOW = True             # rc2 I4: estimator.fit(shadow=True) -- hazard path only (dev4 default)
    EXACT_F = True            # rc2 I4: estimator.fit(exact_from_F=True) -- exact-time deaths (dev4 default)
    DESCRIPTION = ""
    descriptor = _CheckedDescriptor()   # rc2 I4: the VERSION LOCK on every access (class or instance)

    def cost(self, spec: CohortCharacteristics) -> CostDeclaration:
        """PLACEHOLDER until Phase B/C measure it: mle_profile_jax v0.3.3's MEASURED declaration (full tier,
        2026-09-26: median 588, max 69,410 CPU s, peak <= 718 MB). jax2 is expected cheaper on the 48
        floor-bound cohorts and ~1.1-1.9x where the re-fit fires."""
        return CostDeclaration(median_seconds=590.0, min_seconds=140.0, max_seconds=70000.0)

    def claimset(self, observation: Observation, res: dict, prov, t0: float, c0: float) -> ClaimSet:
        """`calibrate`'s second half: the fit result -> a ClaimSet (v0.3.3's mapping and diagnostics)."""
        from srbench.provenance import format_code_line
        claims_kw, sc_block = final_claim_kwargs(res, self.SELF_CONSISTENT)    # rc2 I1: widened when on
        claims = [Claim(**kw) for kw in claims_kw]
        diag = dict(res.get("diagnostics") or {})
        if sc_block is not None:
            diag["self_consistency"] = sc_block
        diag.pop("profiles", None)
        diag["loglik"] = res.get("loglik")
        diag["dt_ceiling"] = res.get("dt_ceiling")
        diag["code"] = format_code_line(prov)
        diag["switches"] = dict(hazard=bool(self.HAZARD), window=bool(self.WINDOW), relabel=bool(self.RELABEL),
                                shadow=bool(self.SHADOW), exact_from_F=bool(self.EXACT_F),
                                self_consistent=bool(self.SELF_CONSISTENT), floor_early=bool(self.FLOOR_EARLY),
                                n_rule=bool(self.N_RULE), dt_rule=bool(self.DT_RULE),
                                early_all=bool(self.EARLY_ALL))
        return ClaimSet(
            cohort_id=observation.cohort_id,
            method_name=self.descriptor.name,
            claims=tuple(claims),
            status="ok",
            wall_clock_seconds=res.get("wall_s") or (time.perf_counter() - t0),
            cpu_seconds=time.process_time() - c0,
            diagnostics=diag,
            code_provenance=prov.to_dict(),
        )

    def calibrate(self, observation: Observation) -> ClaimSet:
        env_guard()                                   # before anything: a refusal is not a result
        version_lock_guard()                          # B5: the code must be the locked code of this version
        from srbench.provenance import ProvenanceError, resolve_sut
        try:
            prov = resolve_sut()                      # refuses on the stale mount; applies the numpy shim
        except ProvenanceError as ex:
            raise RefusedEnvironment(f"REFUSING ({self.descriptor.name}): {ex}") from ex
        os.environ.setdefault("SRK_CACHE_READONLY", "1")

        t0, c0 = time.perf_counter(), time.process_time()
        try:
            res = run_fit(observation, relabel=self.RELABEL, hazard=self.HAZARD, window=self.WINDOW,
                          shadow=self.SHADOW, exact_from_F=self.EXACT_F,
                          self_consistent=self.SELF_CONSISTENT, floor_early=self.FLOOR_EARLY,
                          n_rule=self.N_RULE, dt_rule=self.DT_RULE, early_all=self.EARLY_ALL)
        except NotImplementedError as ex:             # an unbuilt path is not a result either
            raise RefusedEnvironment(f"REFUSING ({self.descriptor.name}): {ex}") from ex
        except Exception as ex:  # noqa: BLE001
            return ClaimSet(cohort_id=observation.cohort_id, method_name=self.descriptor.name,
                            status=f"error:{type(ex).__name__}",
                            wall_clock_seconds=time.perf_counter() - t0,
                            cpu_seconds=time.process_time() - c0,
                            diagnostics={"exception": str(ex)[:500]},
                            code_provenance=prov.to_dict())
        return self.claimset(observation, res, prov, t0, c0)


class MleProfileJax2(_MleProfileJax2Base):
    NAME = "mle_profile_jax2"
    HAZARD = False
    WINDOW = False
    RELABEL = True        # Phase A's CENSOR_TIE rules confirm or reverse this (then: False, and a new version)
    DESCRIPTION = ("mle_profile_jax's estimator with a cancellation-free interval mass (fix B), a re-fit on a "
                   "witnessed better point, a floored dt ceiling and the censoring-tie relabel.")


class MleProfileJax2H(_MleProfileJax2Base):
    NAME = "mle_profile_jax2_h"
    HAZARD = True         # B2: h = exp(z[4]) fitted as a nuisance parameter (never claimed)
    WINDOW = True         # B2: the window snapped to the delta grid, left truncation at lo_eff
    RELABEL = True
    DESCRIPTION = ("mle_profile_jax2 plus a fitted constant external hazard (a nuisance parameter, reported in "
                   "the diagnostics, never claimed) and the fit window (snapped to the recording grid; left "
                   "truncation).")


# Register ONLY when loaded the way srbench discovery loads a method package (`srbench_methods.<directory>`,
# srbench/contract/method.py `discover`). MEASURED 2026-10-02: pytest imports this file as the plain
# package `mle_profile_jax2` to collect the tests below it; registering there put a second instance under
# each name, and the tester suite's next discover() failed with "two different methods both call
# themselves 'mle_profile_jax2'" (60 errors when `pytest tests methods/mle_profile_jax2/tests` ran together).
if __name__ == _KEY:
    register(MleProfileJax2())
    register(MleProfileJax2H())
