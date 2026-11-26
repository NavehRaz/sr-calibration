"""`mle_profile_jax2` and `mle_profile_jax2_h` -- successors to `mle_profile_jax` v0.3.3, which stays FROZEN --
and (1.0.1.dev2) their kappa-FIXED variants.

ONE package, EIGHT registered names, ONE base class (`_MleProfileJax2Base`), ONE VERSION_LOCK (all names frozen
together). Each name fixes its observation model and likelihood with class constants, never with a runtime flag:
srbench's store key takes its settings from the run CALLER, not from the method, so a runtime switch would put
two behaviours under one key (PHASE_B_DESIGN.txt section 0). Every behaviour change therefore needs a new
VERSION, and VERSION_LOCK.json + tests/test_jax2_version_lock.py fail when the code changes but the version does
not.

    name                              HAZARD  WINDOW  RELABEL  KAPPA0  capabilities (derived from the constants)
    mle_profile_jax2                  False   False   True     -       ('censoring',)
    mle_profile_jax2_h                True    True    True     -       ('external_hazard', 'fit_window', 'censoring')
    mle_profile_jax2_kfix_k0p001      False   False   True     1e-3    ('censoring',)
    mle_profile_jax2_kfix_k0p5        False   False   True     0.5     ('censoring',)
    mle_profile_jax2_kfix_k1          False   False   True     1       ('censoring',)
    mle_profile_jax2_h_kfix_k0p001    True    True    True     1e-3    ('external_hazard', 'fit_window', 'censoring')
    mle_profile_jax2_h_kfix_k0p5      True    True    True     0.5     ('external_hazard', 'fit_window', 'censoring')
    mle_profile_jax2_h_kfix_k1        True    True    True     1       ('external_hazard', 'fit_window', 'censoring')
The kfix names (`kfix_name`, the kfix clone's `kappa_tag`) are J / H with kappa PINNED at KAPPA0 (estimator note 14;
the kfix study, sr_calibration_tester_kfix 0.1.2, ported onto the full-z mask), with every 1.0.1 switch of their
base name, COMBOS = ("sqrt_eta_over_eps",) and KEEP_PROFILES. Their claims (`kfix_claim_kwargs`, Naveh 2026-10-08):
eta and epsilon as J / H (self-consistency widened); BETA = the beta claim AT kappa0, claimed and scored like J's
beta -- NOT like-for-like with a truth beta at another kappa; kappa shape "none" by construction (its availability
loss is not a finding); `diagnostics.combo_claims` = {sqrt_eta_over_eps: the walked combination's claim, widened by
its derived extents; beta_rbar: the final beta claim x rbar(kappa0), rbar(k) = 1 - k ln(1 + 1/k), exact because
kappa is a constant}, in the kfix clone's claim format. beta/(1+kappa) is never claimed (biased toward kappa0).

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
and both names (rc2 I2, N_RULE, on for dev4; DT_RULE off until 1.0.1.dev3, on -- scoped -- since 1.0.1.dev4):
  9. the N rule for large n: the space grid N = 900 * 2^k whose predicted grid error of the claim-relevant
     deviances is <= 0.2 nats (from n and the Kaplan-Meier CV), capped at 3,600, composed with the floor
     (`estimator._grid(n_rule=True)`, `_n_rule`). MEASURED (reports/rc2/I2): N = 900 misplaced those deviances by
     0.6-3 nats on n = 1e5 human cohorts and up to 4.8 nats on sharp synthetic cohorts. DT_RULE (dt <= CV/80, the
     time error that dominates on CV < 0.05 cohorts) was built here but left off as a separate decision (CPU
     x2.8). 1.0.1.dev4: DT_RULE ON, scoped to KM CV < 0.05 (estimator note 15; VERSIONS below); 1.0.1.dev5: the
     scope reads the binning-corrected CV_c < 0.05 (estimator note 16).
and (rc2 I4, SHADOW and EXACT_F, on for dev4): _h's shadow chains in the other kappa basin (see below) and the
exact-time density from the absorbed mass.
and both names (rc2 D1, EARLY_ALL, on since dev5; ratified R-8 "extend to all"; growth x1.05 since dev6):
 10. a graded early-time mesh (20 BE steps of 1e-8, then x1.05 per step up to the grid's dt) on EVERY grid with a
     sampling_dt, floor-bound grids included, where it replaces item 8's x1.02 -- one ratio everywhere
     (`estimator._grid(early_all=True)`, `_EARLY_ALL_MESH`, note 12 there). MEASURED (reports/rc2/I3-floor section
     8): the production (non-floor) grid put deaths in the first sampling bins up to 5.7 nats off on two n = 1e5
     Denmark cohorts and 3.1 nats on a synthetic pure cohort; validated at the likelihood level in reports/rc2/dev5/
     (x1.02) and reports/rc2/dev6/ (x1.05). Exact-time data keep the uniform grid (EXACT_F's density assumes it).
and both names (1.0.1.dev1; the rc2 C1 diagnosis, reports/rc2/C1diag_workflow_result.json; Naveh's decision of
2026-10-05, reports/rc2/RATIFICATION.txt: "1.0.1 is prepared ... tested by a small local A/B before any decision on
its own run"), FOUR switches, each its own constant (estimator note 13):
 11. INNER_POLISH_J: `_polish_inner` (the node just inside each edge re-solved from the MLE, seed, edge-node and
     kappa-corner starts) runs in mle_profile_jax2 too -- in 1.0.0 only _h ran it (under SHADOW);
     BRACKET_BISECT: every crossing bracket profile_ci interpolates that is wider than 0.3 dex gets midpoint nodes;
     EXTENT_Z: diagnostics.self_consistency.extents also stores z_qmin / z_qmax, the exact float64 log-parameters;
     BOX_FLAG: diagnostics.box_limited flags every claim edge set by a point with a coordinate on its box edge
     (`box_limited_block`). EXTENT_Z and BOX_FLAG change no claim.
Versions are `1.0.0` (frozen for Phase C) and `1.0.1.devN` for the successor until it passes its own A/B and checks.

IMPORTS. The top level imports the standard library and `srbench.contract` only; numpy, SciPy, JAX and the
system under test (SRK) load lazily inside `calibrate` / `run_fit`, so `srbench methods` and discovery
work without them (tested in a subprocess with `sys.modules['jax'] = None`). The names (`CLASSES`) are registered
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
1.0.0 = 1.0.0.dev6 frozen with no code change (rc2 F1, commit 3bc5775): the runtime of Phase C.
1.0.1.dev1 (2026-10-05, worktree sr_calibration_tester_jax2_101, branch jax2-101): 1.0.0 plus FOUR switches, ON for
  1.0.1.dev1, each OFF in `run_fit`'s defaults:
    switch           constant          1.0.1  estimator.fit keyword                     what it changes
    inner polish, J  INNER_POLISH_J    on     fit(inner_polish=True)                    J's node solves (no-op on _h
                                                                                         while SHADOW is on)
    bracket bisect   BRACKET_BISECT    on     fit(bracket_bisect=True)                  profile nodes (edges)
    extent z         EXTENT_Z          on     fit(extent_z=True)                        diagnostics only
    box flag         BOX_FLAG          on     fit(box_flag=True) + `box_limited_block`  diagnostics only
  With all four off the estimator gets 1.0.0's exact keyword set and the fit is 1.0.0's bit for bit
  (reports/r101/identity: 3 quick cohorts, both names).
1.0.1.dev2 (2026-10-08, worktree sr_calibration_tester_jax2_rel, branch jax2-rel; plan Part 2, Naveh's release
  decisions): 1.0.1.dev1 with BOX_FLAG OFF (dormant; the switch and `box_limited_block` stay), plus the kfix names
  above. Two new `run_fit` keywords, passed to the estimator only when set (None / () = 1.0.1.dev1's call):
    kappa pinned     KAPPA0            kfix   fit(kappa_fixed=k0)                       the kfix names only
    combinations     COMBOS            kfix   fit(combos=("sqrt_eta_over_eps",))        the kfix names only
  and KEEP_PROFILES (diagnostics.profiles kept; J / H drop it as before). With every new switch off the fit is
  1.0.0's / 1.0.1.dev1's bit for bit (estimator note 14). The validation-only kappa-free combination twin is a
  SEPARATE package (methods/mle_profile_jax2_combotwin/, own version and lock; never handed off).
1.0.1.dev3 (2026-10-08; the implementation review of dev2, reports/rel101/IMPL_REVIEW_LOG.md): no arithmetic change
  with the kfix switches off, and none on a successful kfix fit. `calibrate` passes kappa_fixed / combos to `run_fit`
  only when set (J / H: 1.0.1.dev1's run_fit keyword set); `kfix_claim_kwargs` refuses a result without
  diagnostics.kfix or pinned at another kappa; the estimator returns theta[3] = k0 exactly and diagnostics.kfix on
  the "all starts failed" path too, and documents the `_repair` hook under KFIX and the twin's beta_rbar read-out
  (estimator note 14).
1.0.1.dev4 (2026-10-15; Naveh's DT_RULE decision, reports/rc2/RATIFICATION.txt "D: scoped to CV < 0.05", on
  reports/rc2/DTRULE/DECISION_BRIEF.txt; registered by PREREG_REL ADD3, reports/rel101/ADD3): rc2 I2's CV-aware dt
  cap ON for every name, SCOPED in the estimator (note 15, `_DTR_CV_MAX`):
    switch           constant          1.0.1  estimator.fit keyword                     what it changes
    CV-aware dt      DT_RULE           on     fit(dt_rule=True)                         dt, only where KM CV < 0.05
  dt = min(dt, CV/80) only where the cohort's Kaplan-Meier CV (of the prepared data, `estimator._km_cv`, recorded as
  diagnostics.dt_floor.grid_rule.cv) is < 0.05 (strict); at CV >= 0.05 the grid is 1.0.1.dev3's exactly, below it
  1.0.1.dev3's dt_rule=True grid exactly. Claimed (once PREREG_REL ADD3's checks pass): the truth witness on sharp
  data; interval coverage there is expected but untested by a registered stage; not sharp point estimates; nothing
  at CV >= 0.05. `run_fit`'s default stays dt_rule=False: with every new switch off the fit is 1.0.0's bit for
  bit. The record differs from dev3's on a CV >= 0.05 cohort only in diagnostics.switches.dt_rule (True) and
  dt_floor.grid_rule's rule keys (dt_rule, dt_k, dt_cap, dt_cv_max, dt_in_scope).
1.0.1.dev5 (2026-10-16; Naveh's DT_RULE follow-up decision, reports/rc2/RATIFICATION.txt SCOPE "B: binning-aware
  dev5", on reports/rc2/DTRULE/followup/BRIEF.txt; registered by PREREG_REL ADD3 rev 2, reports/rel101/ADD3): the
  SCOPE's statistic only (estimator note 16, `_cv_c`):
    switch           constant          1.0.1  estimator.fit keyword                     what it changes
    CV-aware dt      DT_RULE           on     fit(dt_rule=True)                         dt, only where CV_c < 0.05
  CV_c = sqrt(CV^2 - (delta/L)^2/12) (Sheppard's binning correction; 0.0 where CV^2 <= (delta/L)^2/12; = CV for
  exact-time data), CV = dev4's statistic (`estimator._km_cv`), delta/L = the estimator grid's own dl (L = the
  Kaplan-Meier restricted mean of the prepared data). dt = min(dt, CV/80) (the cap unchanged) where CV > 0 and
  CV_c < 0.05 (strict). CV_c <= CV, so every dev4 in-scope cohort keeps dev4's grid exactly; dev5 adds the coarsely
  recorded cohorts whose KM CV >= 0.05 but CV_c < 0.05 (their grid becomes 1.0.1.dev3's dt_rule=True grid exactly).
  The record adds dt_floor.grid_rule.cv_c; on a cohort out of scope it differs from dev3's only in
  diagnostics.switches.dt_rule (True) and grid_rule's rule keys (dt_rule, dt_k, dt_cap, dt_cv_max, cv_c,
  dt_in_scope). `run_fit`'s default stays dt_rule=False (1.0.0's call and fit, bit for bit).
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
    "mle_profile_jax2": "1.0.1.dev5",
    "mle_profile_jax2_h": "1.0.1.dev5",
    "mle_profile_jax2_kfix_k0p001": "1.0.1.dev5",
    "mle_profile_jax2_kfix_k0p5": "1.0.1.dev5",
    "mle_profile_jax2_kfix_k1": "1.0.1.dev5",
    "mle_profile_jax2_h_kfix_k0p001": "1.0.1.dev5",
    "mle_profile_jax2_h_kfix_k0p5": "1.0.1.dev5",
    "mle_profile_jax2_h_kfix_k1": "1.0.1.dev5",
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

#: 1.0.1.dev2 KFIX: the pinned kappa values of the registered kfix names
KFIX_KAPPAS = (1e-3, 0.5, 1.0)


def kappa_tag(k: float) -> str:
    """The kfix clone's tag for a kappa value (1e-3 -> '0p001', 0.5 -> '0p5', 1 -> '1')."""
    return f"{k:g}".replace(".", "p").replace("-", "m").replace("+", "")


def kfix_name(base: str, k: float) -> str:
    """The registered name of `base` (mle_profile_jax2 / mle_profile_jax2_h) with kappa fixed at k."""
    return f"{base}_kfix_k{kappa_tag(k)}"


def rbar(k: float) -> float:
    """MEAN removal factor over the damage range, 1 - k ln(1 + 1/k) (srbench.score.combos.removal_mean_factor's
    closed form; the kfix clone's `_rbar`)."""
    k = float(k)
    return 1.0 - k * math.log1p(1.0 / k)


#: 1.0.1.dev2: each combination's feasible range on the claimed box (the walk's own range; the widening's box edge)
COMBO_BOX = {
    "sqrt_eta_over_eps": (math.sqrt(1e-12) / 1e3, math.sqrt(1e3) / 1e-12),
    "beta_rbar": (1e-12 * rbar(300.0), 1e3 * rbar(1e-3)),
}

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
            shadow: bool = False, exact_from_F: bool = False, early_all: bool = False,
            inner_polish_j: bool = False, bracket_bisect: bool = False, extent_z: bool = False,
            box_flag: bool = False, kappa_fixed: float | None = None, combos: tuple = ()) -> dict:
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
    1.0.1.dev4: the dt cap applies only where the KM CV < 0.05 (estimator note 15); the default is still False.
    1.0.1.dev5: only where the binning-corrected CV_c < 0.05 (estimator note 16); the default is still False.
    rc2 D1 (dev5) / E1 (dev6): `early_all=True` marches the graded early-time mesh (x1.05 since dev6) on every grid
    with a sampling_dt, floor-bound grids included (`estimator.fit(early_all=True)`); False (the default) calls the
    estimator exactly as dev4 did.
    1.0.1.dev1: `inner_polish_j`, `bracket_bisect`, `extent_z`, `box_flag` pass `estimator.fit(inner_polish=True)`,
    `(bracket_bisect=True)`, `(extent_z=True)`, `(box_flag=True)` (estimator note 13); all False (the defaults) call
    the estimator exactly as 1.0.0 did. With `box_flag` the fit carries diagnostics.box_points, which
    `box_limited_block` turns into diagnostics.box_limited once the claims are built.
    1.0.1.dev2: `kappa_fixed=k0` / `combos=(...)` pass `estimator.fit(kappa_fixed=k0)` / `(combos=...)` (estimator note
    14) only when set; None / () (the defaults) call the estimator exactly as 1.0.1.dev1 did."""
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
                  **({"early_all": True} if early_all else {}),               # rc2 D1: off = dev4's call
                  **({"inner_polish": True} if inner_polish_j else {}),       # 1.0.1: off = 1.0.0's call
                  **({"bracket_bisect": True} if bracket_bisect else {}),
                  **({"extent_z": True} if extent_z else {}),
                  **({"box_flag": True} if box_flag else {}),
                  **({"kappa_fixed": float(kappa_fixed)} if kappa_fixed is not None else {}),   # 1.0.1.dev2
                  **({"combos": tuple(combos)} if combos else {}))
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


def combo_claim_kwargs(res, name, self_consistent, box=None):
    """1.0.1.dev2: the claim on a COMBINATION from the estimator's `diagnostics.combo_grid[name]` (its profile_ci
    interval, verdict, n_components and centre = the MLE's value), with `claim_kwargs`' mapping on the combination's
    box (`COMBO_BOX`), then, with `self_consistent`, `sc_widen_claim` against
    `diagnostics.self_consistency.derived_extents[name]`. In the kfix clone's claim format (parameter, shape, level,
    lower / upper / point / n_components as the shape has them). Returns (claim dict, widening record or None);
    (None, None) when the fit carries no record of `name`."""
    dg = res.get("diagnostics") or {}
    cg = (dg.get("combo_grid") or {}).get(name)
    if cg is None:
        return None, None
    box = COMBO_BOX[name] if box is None else box
    lo, hi = (cg.get("ci") or {}).get(f"{name}@{LEVEL}", (float("nan"), float("nan")))
    shape = VERDICT_TO_SHAPE.get(cg.get("verdict"), "none")
    lo_ok = lo is not None and math.isfinite(lo)
    hi_ok = hi is not None and math.isfinite(hi)
    if shape == "closed" and not (lo_ok and hi_ok):
        shape = "upper_bound" if hi_ok else ("lower_bound" if lo_ok else "none")
    if shape == "upper_bound" and not hi_ok:
        shape = "none"
    if shape == "lower_bound" and not lo_ok:
        shape = "none"
    kw = {"parameter": name, "shape": shape, "level": LEVEL}
    if shape in ("closed", "lower_bound"):
        kw["lower"] = float(max(lo, box[0]))
    if shape in ("closed", "upper_bound"):
        kw["upper"] = float(min(hi, box[1]))
    if shape != "none":
        v = cg.get("centre")
        if v is not None and math.isfinite(v):
            kw["point"] = float(min(max(v, box[0]), box[1]))
        kw["n_components"] = max(1, int(cg.get("n_components") or 1))
    rec = None
    if self_consistent:
        ext = ((dg.get("self_consistency") or {}).get("derived_extents") or {}).get(name)
        kw, rec = sc_widen_claim(kw, ext, box)
        if kw["shape"] == "none":
            kw.pop("n_components", None)
    return kw, rec


def kfix_claim_kwargs(res, claims, kappa0, self_consistent):
    """1.0.1.dev2 KFIX: the kfix names' claims, post-processing the FINAL claims of `final_claim_kwargs` (whose
    `claim_kwargs` stays byte-identical). Returns (claims, combo_claims, record):
      * eta, epsilon: unchanged (J / H's, self-consistency widened);
      * beta: unchanged -- the beta claim AT kappa0 (Naveh's decision: claimed, scored like J's beta; not
        like-for-like with a truth beta at another kappa);
      * kappa: shape "none" by construction (the fit has no kappa profile);
      * combo_claims["sqrt_eta_over_eps"] = `combo_claim_kwargs` (the walked combination, widened by its derived
        extents with `self_consistent`);
      * combo_claims["beta_rbar"] = the final beta claim with lower / upper / point x rbar(kappa0) (exact: kappa is a
        constant), shape kept, n_components = the beta profile's (diagnostics.kfix.n_components).
    beta/(1+kappa) is NEVER claimed.
    1.0.1.dev3: refuses (ValueError) a result that is not a kappa-fixed fit (no diagnostics.kfix) or whose pinned
    kappa (diagnostics.kfix.kappa_fixed) is not `kappa0`: rbar(kappa0) would scale beta by another kappa's factor."""
    dg = res.get("diagnostics") or {}
    kf = dg.get("kfix")
    if not isinstance(kf, dict) or kf.get("kappa_fixed") is None:
        raise ValueError("mle_profile_jax2 kfix_claim_kwargs: the result has no diagnostics.kfix -- not a "
                         "kappa-fixed fit (estimator.fit(kappa_fixed=k0))")
    if float(kappa0) != float(kf["kappa_fixed"]):
        raise ValueError(f"mle_profile_jax2 kfix_claim_kwargs: kappa0={float(kappa0)!r} but the fit pinned kappa at "
                         f"{float(kf['kappa_fixed'])!r} (diagnostics.kfix.kappa_fixed)")
    out = []
    for kw in claims:
        if kw["parameter"] == "kappa":
            kw = {"parameter": "kappa", "shape": "none", "level": LEVEL}
        out.append(kw)
    combo, rec = {}, {}
    sq, sq_rec = combo_claim_kwargs(res, "sqrt_eta_over_eps", self_consistent)
    if sq is not None:
        combo["sqrt_eta_over_eps"] = sq
        rec["sqrt_eta_over_eps_widening"] = sq_rec
    b = next(kw for kw in out if kw["parameter"] == "beta")
    r = rbar(kappa0)
    br = {"parameter": "beta_rbar", "shape": b["shape"], "level": b["level"]}
    for key in ("lower", "upper", "point"):
        if key in b:
            br[key] = float(b[key] * r)
    if br["shape"] != "none":
        nc = ((dg.get("kfix") or {}).get("n_components") or {}).get("beta")
        br["n_components"] = max(1, int(nc or 1))
    combo["beta_rbar"] = br
    rec.update(kappa0=float(kappa0), rbar_kappa0=r, beta_rbar_from="the final beta claim x rbar(kappa0)")
    return out, combo, rec


def box_limited_block(res, claims, sc_block):
    """1.0.1 BOX_FLAG: `diagnostics.box_limited` -- for every stated edge of the FINAL claims (the four SR claims,
    plus the would-be h claim when the fit has the hazard), which point SET it and whether that point has a
    coordinate ON its box edge (estimator `_box_status`: within _BOX_FLAG_ULPS ulps in z; h at its floor excepted).

    The setter: the self-consistency extent point when the widening moved that side out to it (`sc_widen_claim`:
    `<side>_moved_dex` != 0 and the side not opened), else the profile bracket's two nodes (the pair profile_ci
    interpolated the edge between; either one's value moves the edge). `flag` = any setter point on the box. Such
    an edge is BOX-CONDITIONAL: with the box lifted the profile there could be higher and the edge further out
    (C1diag REVIEW item 1). An open side has no edge (None). No claim is changed.
    `res` is `run_fit`'s result (it carries `diagnostics.box_points`, `estimator.fit(box_flag=True)`), `claims` and
    `sc_block` are `final_claim_kwargs(res, ...)`'s. Returns None when the fit has no box_points."""
    dg = res.get("diagnostics") or {}
    bp = dg.get("box_points")
    if bp is None:
        return None
    todo = [(kw, ((sc_block or {}).get("widening") or [])) for kw in claims]
    hz = dg.get("hazard")
    if isinstance(hz, dict) and isinstance(hz.get("h_claim"), dict):          # the would-be h claim
        todo.append((hz["h_claim"], [hz["h_claim_sc"]] if isinstance(hz.get("h_claim_sc"), dict) else []))
    params, flagged = {}, []
    for kw, recs in todo:
        p = kw["parameter"]
        rec = next((r for r in recs if r.get("parameter") == p), None)
        src = (bp.get("params") or {}).get(p) or {}
        ent = {}
        for side in ("lower", "upper"):
            if side not in kw:
                ent[side] = None                                  # open (or a decline): no edge to flag
                continue
            moved = (rec is not None and float(rec.get(f"{side}_moved_dex") or 0.0) != 0.0
                     and side not in (rec.get("opened") or []))
            s = src.get(side) or {}
            if moved:
                setter, pts = "self_consistency", ([s["sc"]] if s.get("sc") else [])
            else:
                setter = "profile"
                pts = list((s.get("profile") or {}).get("nodes") or [])
            on = sorted({lab for q in pts if q and q.get("status") for lab in q["status"]["on"]})
            ent[side] = dict(edge=float(kw[side]), setter=setter, flag=bool(on), on_box=on,
                             known=bool(pts) and all(q.get("status") is not None for q in pts), points=pts,
                             **({"straddles": (s.get("profile") or {}).get("straddles"),
                                 "width_dex": (s.get("profile") or {}).get("width_dex")} if setter == "profile" else {}))
            if on:
                flagged.append(f"{p}:{side}")
        params[p] = ent
    return dict(ulps=bp.get("ulps"), n_flagged=len(flagged), flagged=flagged, params=params)


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
    """Everything but the constants. Subclasses set NAME, HAZARD, WINDOW, RELABEL and DESCRIPTION (and, for the
    kfix names, KAPPA0, COMBOS and KEEP_PROFILES)."""

    NAME = None
    HAZARD = False
    WINDOW = False
    RELABEL = True
    SELF_CONSISTENT = True    # rc2 I1: claims contain every evaluated point within thr/2 of the best (dev4 default)
    N_RULE = True             # rc2 I2: N = 900 * 2^k from n and the KM CV, capped at 3,600 (dev4 default)
    DT_RULE = True            # 1.0.1.dev4: dt <= CV/80 ONLY where the KM CV < 0.05 (estimator note 15; Naveh
                              #        2026-10-15, DT_RULE "D: scoped"); rc2 I2 built it, off until dev3;
                              #        1.0.1.dev5: where CV_c < 0.05 (note 16; Naveh 2026-10-16, "B")
    FLOOR_EARLY = True        # rc2 I3: graded early-time mesh where the floored dt ceiling binds (dev4 default)
    EARLY_ALL = True          # rc2 D1/E1: the graded mesh (x1.05 since dev6) on EVERY grid with a sampling_dt (R-8)

    SHADOW = True             # rc2 I4: estimator.fit(shadow=True) -- hazard path only (dev4 default)
    EXACT_F = True            # rc2 I4: estimator.fit(exact_from_F=True) -- exact-time deaths (dev4 default)
    INNER_POLISH_J = True     # 1.0.1: estimator.fit(inner_polish=True) -- `_polish_inner` without the hazard too
    BRACKET_BISECT = True     # 1.0.1: estimator.fit(bracket_bisect=True) -- crossing brackets <= 0.3 dex
    EXTENT_Z = True           # 1.0.1: estimator.fit(extent_z=True) -- z_qmin / z_qmax in the extents (no claim effect)
    BOX_FLAG = False          # 1.0.1: estimator.fit(box_flag=True) + `box_limited_block` (no claim change); OFF since
                              #        1.0.1.dev2 (Naveh 2026-10-08: dormant)
    KAPPA0 = None             # 1.0.1.dev2 KFIX: estimator.fit(kappa_fixed=KAPPA0) -- the kfix names only
    COMBOS = ()               # 1.0.1.dev2: estimator.fit(combos=COMBOS) -- the kfix names: ("sqrt_eta_over_eps",)
    KEEP_PROFILES = False     # 1.0.1.dev2: keep diagnostics.profiles (the kfix names); J / H drop it as before
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
        combo = kf_rec = None
        if self.KAPPA0 is not None:                                             # 1.0.1.dev2 KFIX
            claims_kw, combo, kf_rec = kfix_claim_kwargs(res, claims_kw, self.KAPPA0, self.SELF_CONSISTENT)
        claims = [Claim(**kw) for kw in claims_kw]
        diag = dict(res.get("diagnostics") or {})
        if sc_block is not None:
            diag["self_consistency"] = sc_block
        if combo is not None:
            diag["combo_claims"] = combo
            diag["kfix"] = dict(diag.get("kfix") or {}, claims=kf_rec)
        if self.BOX_FLAG:                                                       # 1.0.1: from the final claims
            bl = box_limited_block(res, claims_kw, sc_block)
            if bl is not None:
                diag["box_limited"] = bl
                diag.pop("box_points", None)                                    # its content is in box_limited
        if not self.KEEP_PROFILES:
            diag.pop("profiles", None)
        diag["loglik"] = res.get("loglik")
        diag["dt_ceiling"] = res.get("dt_ceiling")
        diag["code"] = format_code_line(prov)
        diag["switches"] = dict(hazard=bool(self.HAZARD), window=bool(self.WINDOW), relabel=bool(self.RELABEL),
                                shadow=bool(self.SHADOW), exact_from_F=bool(self.EXACT_F),
                                self_consistent=bool(self.SELF_CONSISTENT), floor_early=bool(self.FLOOR_EARLY),
                                n_rule=bool(self.N_RULE), dt_rule=bool(self.DT_RULE),
                                early_all=bool(self.EARLY_ALL),
                                inner_polish_j=bool(self.INNER_POLISH_J), bracket_bisect=bool(self.BRACKET_BISECT),
                                extent_z=bool(self.EXTENT_Z), box_flag=bool(self.BOX_FLAG),
                                **({"kappa_fixed": float(self.KAPPA0)} if self.KAPPA0 is not None else {}),
                                **({"combos": list(self.COMBOS)} if self.COMBOS else {}))
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
                          n_rule=self.N_RULE, dt_rule=self.DT_RULE, early_all=self.EARLY_ALL,
                          inner_polish_j=self.INNER_POLISH_J, bracket_bisect=self.BRACKET_BISECT,
                          extent_z=self.EXTENT_Z, box_flag=self.BOX_FLAG,
                          # 1.0.1.dev3: only when set, so J / H call run_fit with 1.0.1.dev1's keyword set
                          **({"kappa_fixed": self.KAPPA0} if self.KAPPA0 is not None else {}),
                          **({"combos": self.COMBOS} if self.COMBOS else {}))
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


_KFIX_J_DESC = ("mle_profile_jax2 with kappa FIXED at {k:g} (a mask over the full z): eta, epsilon and beta AT "
                "kappa0 claimed, kappa declined by construction; sqrt(eta)/eps and beta*rbar(kappa0) in "
                "diagnostics.combo_claims.")
_KFIX_H_DESC = ("mle_profile_jax2_h (fitted external hazard, fit window) with kappa FIXED at {k:g}: eta, epsilon "
                "and beta AT kappa0 claimed, kappa declined by construction; sqrt(eta)/eps and beta*rbar(kappa0) "
                "in diagnostics.combo_claims.")


class MleProfileJax2KfixK0p001(_MleProfileJax2Base):
    NAME = "mle_profile_jax2_kfix_k0p001"
    HAZARD = False
    WINDOW = False
    RELABEL = True
    KAPPA0 = 1e-3
    COMBOS = ("sqrt_eta_over_eps",)
    KEEP_PROFILES = True
    DESCRIPTION = _KFIX_J_DESC.format(k=1e-3)


class MleProfileJax2KfixK0p5(_MleProfileJax2Base):
    NAME = "mle_profile_jax2_kfix_k0p5"
    HAZARD = False
    WINDOW = False
    RELABEL = True
    KAPPA0 = 0.5
    COMBOS = ("sqrt_eta_over_eps",)
    KEEP_PROFILES = True
    DESCRIPTION = _KFIX_J_DESC.format(k=0.5)


class MleProfileJax2KfixK1(_MleProfileJax2Base):
    NAME = "mle_profile_jax2_kfix_k1"
    HAZARD = False
    WINDOW = False
    RELABEL = True
    KAPPA0 = 1.0
    COMBOS = ("sqrt_eta_over_eps",)
    KEEP_PROFILES = True
    DESCRIPTION = _KFIX_J_DESC.format(k=1.0)


class MleProfileJax2HKfixK0p001(_MleProfileJax2Base):
    NAME = "mle_profile_jax2_h_kfix_k0p001"
    HAZARD = True
    WINDOW = True
    RELABEL = True
    KAPPA0 = 1e-3
    COMBOS = ("sqrt_eta_over_eps",)
    KEEP_PROFILES = True
    DESCRIPTION = _KFIX_H_DESC.format(k=1e-3)


class MleProfileJax2HKfixK0p5(_MleProfileJax2Base):
    NAME = "mle_profile_jax2_h_kfix_k0p5"
    HAZARD = True
    WINDOW = True
    RELABEL = True
    KAPPA0 = 0.5
    COMBOS = ("sqrt_eta_over_eps",)
    KEEP_PROFILES = True
    DESCRIPTION = _KFIX_H_DESC.format(k=0.5)


class MleProfileJax2HKfixK1(_MleProfileJax2Base):
    NAME = "mle_profile_jax2_h_kfix_k1"
    HAZARD = True
    WINDOW = True
    RELABEL = True
    KAPPA0 = 1.0
    COMBOS = ("sqrt_eta_over_eps",)
    KEEP_PROFILES = True
    DESCRIPTION = _KFIX_H_DESC.format(k=1.0)


#: every registered class, in registration order (the kfix ones: J then H, kappa0 ascending)
CLASSES = (MleProfileJax2, MleProfileJax2H, MleProfileJax2KfixK0p001, MleProfileJax2KfixK0p5, MleProfileJax2KfixK1,
           MleProfileJax2HKfixK0p001, MleProfileJax2HKfixK0p5, MleProfileJax2HKfixK1)

# Register ONLY when loaded the way srbench discovery loads a method package (`srbench_methods.<directory>`,
# srbench/contract/method.py `discover`). MEASURED 2026-10-02: pytest imports this file as the plain
# package `mle_profile_jax2` to collect the tests below it; registering there put a second instance under
# each name, and the tester suite's next discover() failed with "two different methods both call
# themselves 'mle_profile_jax2'" (60 errors when `pytest tests methods/mle_profile_jax2/tests` ran together).
if __name__ == _KEY:
    for _cls in CLASSES:
        register(_cls())
