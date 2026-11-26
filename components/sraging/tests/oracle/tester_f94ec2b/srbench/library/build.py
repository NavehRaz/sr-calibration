"""Generate the cohort library and freeze it.

**This is the build path, and it imports the system under test.** It is one of only two places
that does (the other is the shipped reference adapter). Everything else -- selection, scoring,
gates, reports, the whole default test suite -- runs on what this writes.

WHAT IS FROZEN AND WHERE:

  * `manifest.json`  -- the ordered specs, per-cohort checksums, provenance, tier boundaries.
    Small, and COMMITTED. This is the library's identity.
  * `cohorts/*.npz`  -- the generated ages and events. Large, and NOT committed: rebuilt locally
    from the manifest in ~20 minutes, and verified against the committed checksums.

That split is deliberate. Committing tens of millions of floats would make the repository
unusable, while committing only the specs would make "the same library version" unverifiable. The
checksums are what bridge the two: a rebuild that does not reproduce them is a FAILURE, not a new
version.

BUILD-TIME VALIDATIONS, each guarding a specific way a library can be quietly wrong:

  * every truth strictly INSIDE the registered bounds -- a truth on the boundary cannot be covered
    by a bounded interval and scores a guaranteed miss that reads as a method failure (gate G1);
  * arm membership consistent with the features actually present -- a cohort labelled `window`
    whose window does nothing is the upstream `var_trunc` defect, and the build refuses it;
  * a windowed cohort must retain enough individuals to be usable after truncation.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import time

import numpy as np

from ..contract.types import PARAMETERS
from .observation import apply_window, window_is_effective, window_retention
from .spec import CohortSpec, arm_for

#: v2 (2026-09-27) APPENDS the optional sample-size ladder. The tiered body is unchanged and its
#: cohorts are reused byte-for-byte from v1, so every stored calibration stays valid -- which is
#: only true because `library_version` was removed from the store key (schema 2).
#: v3 (2026-10-03) drops the `ecoli` anchor and adds a 10,000-sized condition for the large real
#: anchors. Unlike v2 this CHANGES the tiered body, so `medium` and `full` name a slightly
#: different cohort set -- accepted deliberately. Surviving cohorts are reused byte-for-byte, so
#: every stored calibration for them stays valid.
#:
#: `Library()` still defaults to v2 (see library/load.py): other sessions are mid-benchmark, and
#: the default moves when Naveh says so, not when this is built.
#: v5 (2026-10-07) corrects the FIRST RECORDING BIN -- see `correct_first_bin`. v3 and v4 were
#: committed but never became the default; go straight from v2 to v5. They stay on disk only
#: because a published library version is immutable by rule.
#:
#: v4 (2026-10-04) made the two most expensive blocks opt-in and coarsened celegans.
#: v3 (2026-10-03) dropped ecoli and added n=10,000 for the large real anchors.
LIBRARY_VERSION = "v5"

#: A windowed cohort must keep at least this fraction after left truncation. Below it the cohort
#: is not usable, and shipping it would drop a tiny-n result into an arm average with nothing
#: saying so.
MIN_WINDOW_RETENTION = 0.35

#: The bounds every planted truth must lie strictly inside (gate G1). These are the reference
#: method's; a method declaring narrower ones is checked against its own at registration.
REFERENCE_BOUNDS = {
    "eta": (1e-12, 1e3), "beta": (1e-12, 1e3),
    "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0),
}


class BuildError(RuntimeError):
    """The library cannot be built as specified. Never downgraded to a warning."""


#: The fields that determine a cohort's DATA. Two specs agreeing on all of these generate the
#: same arrays, so one may be reused for the other; anything else (position, stratum, notes, the
#: group label) is metadata about where the cohort sits, not about what it contains.
DATA_FIELDS = ("rho_eta", "rho_beta", "rho_eps", "kappa", "n", "L_phys", "seed", "delta",
               "external_hazard", "het_param", "het_var", "censor_frac")

#: `window` is deliberately NOT here. `_materialize` is never given one -- the array is the
#: untruncated cohort, and truncation is a harness-side transform at observation time. For 202 map
#: cohorts in the `window` arm the window is DERIVED at build time from the generated data, so the
#: plan carries None and the built spec carries the resolved pair; treating that as a data
#: difference would regenerate all 202 for nothing.
#:
#: But the window IS part of the observation digest, so a reused cohort keeps the window it was
#: built with. That is why a reused cohort carries its OLD spec forward rather than the planned
#: one -- see the reuse block in `build`.


def _data_fields(d: dict) -> tuple:
    out = []
    for k in DATA_FIELDS:
        v = d.get(k)
        out.append(tuple(v) if isinstance(v, list) else v)
    return tuple(out)


def _group_counts(specs: list[dict]) -> dict:
    out: dict[str, int] = {}
    for d in specs:
        g = d.get("group", "main")
        out[g] = out.get(g, 0) + 1
    return out


def cohort_digest(ages, events) -> str:
    """Checksum of the generated data. `repr` on floats so the digest is exact, not rounded."""
    h = hashlib.sha256()
    h.update(b"|".join(repr(float(a)).encode() for a in np.asarray(ages)))
    h.update(b"#")
    h.update(b"|".join(str(int(e)).encode() for e in np.asarray(events)))
    return h.hexdigest()


def correct_first_bin(ages, delta):
    """Move a clamped first-bin death back to age 0, so it is scored over the bin it died in.

    THE DEFECT, diagnosed by the mle_profile_jax session on 2026-10-07 and reproduced here.
    `SRK.benchmark.cohorts` floors each age onto the delta grid and then clamps it off zero:

        t = floor(t/delta)*delta ;  t = maximum(t, delta*0.5)

    so everyone who died before the first recording tick is written down at `delta/2`. But every
    likelihood in use -- the frozen reference, SRK's own `fp_likelihood`, and the jax2 methods --
    scores a death recorded at age `a` over `[a, a+delta)`. A first-bin death is therefore scored
    over `[delta/2, 3delta/2)` instead of `[0, delta)`: the mass in `[0, delta/2)` is never
    counted and `[delta, 3delta/2)` is counted twice, so the probabilities do not sum to 1.

    MEASURED CONSEQUENCE, not a tidiness argument. The planted truth is rejected by 41.6 nats on
    `hazard-k0.316-p04-c1-s0`, and by 20-36 nats on the guinea-pig cohorts -- which carry no
    hazard, so this reaches plain SR fitting too. Scoring the first bin correctly brings both to
    about 1 nat. It bites hardest where `delta` is coarse against the early-death timescale and a
    noisy hazard-free fit can imitate an early constant hazard.

    Reproduced independently here on v4: 623 of 2,410 cohorts carry such records, 102,119
    individuals, up to 20.8% of a single cohort. No age lies below `delta/2` and no other age is
    off the delta grid, which is what confirms the mechanism is floor-then-clamp.

    WHY HERE, and not in the method or the observation. In a method it would bake a harness
    artefact into every estimator and need a new version of each. At observation time the frozen
    arrays would stay faithful but the library version would not change -- and that version is
    what `require_comparable` guards on, so a pre-fix run could be paired against a post-fix one
    without a word. Correcting the data makes the library honestly new.

    CONSEQUENCE TO STATE RATHER THAN HIDE: the arrays no longer reproduce raw `materialize` output
    bit for bit. They are that output plus this one correction. Fixing the generator is the SRK
    owner's call; this does not touch it.
    """
    import numpy as _np
    if not delta:
        return ages
    a = _np.asarray(ages, dtype=float)
    return _np.where(_np.abs(a - 0.5 * float(delta)) < 1e-12, 0.0, a)


def _materialize(spec: CohortSpec):
    """Generate one cohort's death times via the system under test.

    `time_range` is deliberately NOT passed: measured 2026-09-22, `materialize` never reads it, so
    passing it would produce a cohort bit-identical to a pure one while looking like it had been
    windowed. Windowing is applied by the harness instead (`observation.py`).
    """
    from SRK.benchmark import cohorts as C  # type: ignore

    kw = dict(
        name=spec.cohort_id,
        rho_eta=spec.rho_eta, rho_beta=spec.rho_beta, rho_eps=spec.rho_eps, kappa=spec.kappa,
        n=spec.n, L_phys=spec.L_phys, seed=spec.seed, delta=spec.delta,
        censor_frac=spec.censor_frac,
    )
    if spec.external_hazard is not None:
        kw["external_hazard"] = spec.external_hazard
    if spec.het_param is not None:
        kw["het_param"], kw["het_var"] = spec.het_param, spec.het_var
    co = C.materialize(C.CohortSpec(**kw))
    return np.asarray(co.death_times, dtype=float), np.asarray(co.events, dtype=int)


#: A window arriving here is ALREADY in the cohort's own age units -- `plan.fit_unit_factor`
#: converts it (and the Makeham level) from the fit's time unit at plan time, using the recorded
#: window against the fitted mean lifetime. Doing it there rather than here avoids a
#: chicken-and-egg: a mis-scaled hazard corrupts the very cohort you would use to detect the
#: mis-scaling. What remains here is a sanity check against the generated data.

#: A window must overlap the data it is supposed to truncate. Below this, something is wrong with
#: the unit conversion rather than with the cohort.
MIN_WINDOW_OVERLAP = 0.05


def window_overlap(spec: CohortSpec, ages) -> float:
    """Fraction of the cohort's deaths falling inside the window."""
    if spec.window is None:
        return 1.0
    a = np.asarray(ages, dtype=float)
    lo, hi = spec.window
    return float(((a >= lo) & (a <= hi)).mean()) if len(a) else 0.0


def derive_binding_window(ages) -> tuple[float, float] | None:
    """A window for a map cohort in the `window` arm, chosen so it actually TRUNCATES.

    Quantiles alone are not enough. On a coarse recording grid a sharp cohort's 5th and 95th
    percentiles can land on the same pair of values, and the resulting window truncates nothing --
    measured: 2 of 1,992 cohorts, both at delta/L = 0.1 with only TWO distinct death times.

    So: try the quantiles, and if they do not bind, fall back to the second-smallest and
    second-largest DISTINCT ages, which binds whenever there are at least three. With fewer than
    four distinct ages the fallback collapses to zero width, so the cohort cannot carry a window, and this returns None rather than
    inventing one.
    """
    a = np.asarray(ages, dtype=float)
    lo, hi = (float(x) for x in np.quantile(a, [0.05, 0.95]))
    if (a < lo).any() or (a > hi).any():
        return (lo, hi)
    u = np.unique(a)
    if len(u) < 4:
        return None
    lo2, hi2 = float(u[1]), float(u[-2])
    # A zero-width window is not a window: it would censor everyone at a single instant. With
    # exactly three distinct ages the fallback collapses to one, so four is the real minimum.
    return (lo2, hi2) if lo2 < hi2 else None


def _resolve_window(spec: CohortSpec, ages):
    """The window in age units, plus the unit factor applied (always 1.0 -- conversion happens at
    plan time).

    A MAP cohort in the `window` arm has no recorded window, so a truncation is derived from its
    own death times: a window is only meaningful relative to when the cohort dies, so it cannot be
    a fixed number in the plan.
    """
    if spec.window is not None:
        # A RECORDED window can be unusable in a SINGLE-FEATURE arm, and cats_vp_F is why. Its
        # fitted parameters imply a ~40-year mean lifetime while its fit window is [2, 25] years,
        # so a pure-SR cohort at those parameters has only 1.1% of its deaths inside the window.
        # That is not a unit error -- the alternative interpretation is far worse -- and it is not
        # wrong in `realfull`, where the accompanying background hazard is what fills the window
        # with deaths. It is wrong only in the `window` arm, which isolates the feature and
        # therefore has no hazard to do that.
        #
        # So: in `realfull` the recorded window stands, because it is the real observation model.
        # In `window` it is replaced by a derived one and the substitution is RECORDED, because
        # that arm exists to measure what truncation costs and cannot do so on a cohort the window
        # empties.
        if spec.arm == "window" and window_overlap(spec, ages) < MIN_WINDOW_OVERLAP:
            derived = derive_binding_window(ages)
            return derived, ("substituted" if derived else None)
        return spec.window, 1.0
    if spec.arm == "window":
        w = derive_binding_window(ages)
        return w, 1.0          # None -> the caller excludes this cohort and records why
    return None, 1.0


def validate_built(spec: CohortSpec, ages, events) -> None:
    """Every check that can only be made once the data exists."""
    truth = spec.physical_truth()
    for p in PARAMETERS:
        lo, hi = REFERENCE_BOUNDS[p]
        v = truth[p]
        if not (lo < v < hi):
            raise BuildError(
                f"{spec.cohort_id}: planted {p}={v:g} is not STRICTLY inside the bounds "
                f"({lo:g}, {hi:g}). A truth on or outside a bound cannot be covered by a bounded "
                "interval, so it scores a guaranteed miss that reads as a method failure (G1)."
            )

    expected = arm_for(spec.external_hazard, spec.window, spec.het_param, spec.censor_frac)
    if spec.arm == "realfull":
        # `realfull` means "this cohort's OWN recorded observation model, whole". For the three
        # anchors that genuinely were pure-SR on a full window, that model is empty -- and forcing
        # a feature onto them would make the arm mean "some features" instead, and would destroy
        # the pure-vs-realfull pairing for exactly the cohorts where it is cleanest.
        if spec.real_name is None:
            raise BuildError(f"{spec.cohort_id}: `realfull` holds only real anchored cohorts")
    elif spec.arm != expected:
        raise BuildError(
            f"{spec.cohort_id}: labelled arm {spec.arm!r} but its features imply {expected!r} "
            f"(features: {spec.features or '(none)'}). A mislabelled arm is the upstream "
            "`var_trunc` defect: a cohort wearing a label its data does not carry."
        )

    if spec.window is not None:
        binds = window_is_effective(spec, ages, events)
        # THE DISTINCTION THAT MATTERS: a window the harness DERIVED and a window the cohort
        # actually HAD are different things when neither binds.
        #
        #   * a derived window that changes nothing is a CONSTRUCTION FAILURE -- the map cohort
        #     was supposed to measure truncation and measures none, which is indistinguishable
        #     from the upstream no-op where the window was never applied;
        #   * a RECORDED window that changes nothing is a FACT ABOUT THAT COHORT. Several real fit
        #     windows simply contain all of their cohort's deaths (drosophila_441's [10,100] days
        #     against a ~50-day mean lifetime). Refusing it would misreport the real situation as
        #     a defect.
        derived = spec.real_name is None
        if derived and not binds:
            raise BuildError(
                f"{spec.cohort_id}: DERIVED window {spec.window} changes nothing -- the "
                "observation is bit-identical to the untruncated cohort, so this cohort measures "
                "no truncation."
            )
        if not binds:
            return          # recorded and non-binding: counted in the manifest, not an error
        ret = window_retention(spec, ages)
        if ret < MIN_WINDOW_RETENTION:
            raise BuildError(
                f"{spec.cohort_id}: window retains only {ret:.1%} of individuals "
                f"(minimum {MIN_WINDOW_RETENTION:.0%}); not a usable cohort.")

    if len(ages) != len(events):
        raise BuildError(f"{spec.cohort_id}: ages and events differ in length")
    if not len(ages):
        raise BuildError(f"{spec.cohort_id}: generated no individuals")


def _build_one(args):
    import dataclasses
    spec_d, = args
    spec = CohortSpec.from_dict(spec_d)
    ages, events = _materialize(spec)
    ages = correct_first_bin(ages, spec.delta)
    win, factor = _resolve_window(spec, ages)
    substituted = (factor == "substituted")
    if substituted:
        factor = 1.0
    if spec.arm == "window" and win is None:
        # A cohort with fewer than three distinct death times cannot carry a window, so it cannot
        # belong to the arm that exists to MEASURE windowing. Excluded and recorded -- not given a
        # window that does nothing, which is the upstream defect in a different costume.
        n_distinct = int(len(np.unique(np.asarray(ages, dtype=float))))
        return ("EXCLUDED", spec.cohort_id,
                f"only {n_distinct} distinct death times at delta/L="
                f"{(spec.delta or 0) / spec.L_phys:.3g}; cannot carry a binding window")
    if win != spec.window:
        note = spec.notes
        if factor != 1.0:
            note = f"{note}; window unit factor {factor:g} (recorded in fit units)"
        if substituted:
            note = (f"{note}; recorded window {spec.window} left <"
                    f"{MIN_WINDOW_OVERLAP:.0%} of deaths inside without the accompanying hazard, "
                    f"so a derived window was SUBSTITUTED for this single-feature arm")
        spec = dataclasses.replace(spec, window=win, notes=note)
    validate_built(spec, ages, events)
    return spec.to_dict(), ages, events, cohort_digest(ages, events)


def build(out_dir: str, jobs: int = 8, limit: int | None = None,
          verbose: bool = True, reuse: str | None = None) -> dict:
    """Generate and freeze. Returns the manifest."""
    from ..provenance import resolve_sut, harness_provenance
    from . import feasible as F
    from . import select as S
    from . import plan as P
    from . import order as O

    prov = resolve_sut()
    if verbose:
        from ..provenance import format_code_line
        print(format_code_line(prov))

    meas = F.load()
    selection = S.select_all(meas["summary"], meas["rows"])
    anchors_path = os.path.join(os.path.dirname(F.default_path()), "real_anchors.json")
    anchors = json.load(open(anchors_path, encoding="utf-8"))["anchors"]
    specs = P.build_plan(selection, anchors)
    if limit:
        specs = specs[:limit]

    os.makedirs(os.path.join(out_dir, "cohorts"), exist_ok=True)
    t0 = time.time()

    # --- REUSE, so growing the library does not regenerate what it already has ----------------
    #
    # WHY THIS EXISTS. The store key contains the OBSERVATION DIGEST, so a cohort regenerated even
    # one bit differently is a different result and every stored calibration for it is orphaned.
    # v1 was generated from an SRK working tree recorded as `dirty: True`, so its exact generator
    # state is not recoverable from the revision alone -- a full rebuild might or might not
    # reproduce it, and "might" is not good enough for 5,776 stored calibrations.
    #
    # So: a cohort whose DATA-DETERMINING fields are unchanged is copied across with its digest
    # intact, and only genuinely new cohorts are generated. A cohort whose spec changed is
    # regenerated, because reusing an array for a changed spec would pair new metadata with old
    # data -- exactly the "different data wearing the same identity" failure the checksums exist
    # to catch.
    reused, reuse_digests, reuse_specs = set(), {}, {}
    if reuse:
        rman_path = os.path.join(reuse, "manifest.json")
        if not os.path.exists(rman_path):
            raise FileNotFoundError(f"--reuse points at {reuse}, which has no manifest.json")
        with open(rman_path, encoding="utf-8") as fh:
            rman = json.load(fh)
        reuse_digests = dict(rman.get("checksums") or {})
        reuse_specs = {d["cohort_id"]: d for d in rman["specs"]}
        for s in specs:
            old = reuse_specs.get(s.cohort_id)
            if old is None or s.cohort_id not in reuse_digests:
                continue
            if _data_fields(old) != _data_fields(s.to_dict()):
                continue                    # the spec changed: regenerate rather than mislabel
            src = os.path.join(reuse, "cohorts", f"{s.cohort_id}.npz")
            if os.path.exists(src):
                reused.add(s.cohort_id)
        if verbose:
            print(f"reusing {len(reused)} cohorts from {reuse}; "
                  f"generating {len(specs) - len(reused)}")

    todo = [s for s in specs if s.cohort_id not in reused]
    results = []
    payload = [(s.to_dict(),) for s in todo]

    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for i, r in enumerate(pool.imap(_build_one, payload, chunksize=8)):
                results.append(r)
                if verbose and (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        for i, a in enumerate(payload):
            results.append(_build_one(a))
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    final_specs, checksums, nonbinding, excluded = [], {}, [], []

    # The reused cohorts, carried across with their digests. Their arrays are copied so the new
    # version is self-contained: a library that silently depended on an older directory would
    # break the moment someone cleaned it up.
    # ⚑ A REUSED ARRAY IS PUT THROUGH THE CURRENT RECIPE, not copied blind.
    #
    # Reuse is decided on the SPEC's data fields, and the first-bin correction changes none of
    # them -- so a blind copy would have carried 623 uncorrected arrays into v5 under a version
    # that claims to be corrected. That is the "different data wearing the same identity" failure
    # the checksums exist to catch, arriving through the one door that bypasses them.
    #
    # Transforming the stored array is also better than regenerating it: the correction is a pure
    # function of the array, so this cannot drift with the generator (v1 was built from an SRK
    # tree recorded `dirty`, so a regeneration might or might not reproduce it), and a cohort the
    # recipe does not touch keeps its digest and every stored result for it.
    import shutil
    n_corrected = 0
    for s_ in specs:
        if s_.cohort_id not in reused:
            continue
        src = os.path.join(reuse, "cohorts", f"{s_.cohort_id}.npz")
        dst = os.path.join(out_dir, "cohorts", f"{s_.cohort_id}.npz")
        old = reuse_specs[s_.cohort_id]
        with np.load(src) as z:
            ages, events = z["ages"], z["events"]
        fixed = correct_first_bin(ages, old.get("delta"))
        if not np.array_equal(fixed, ages):
            np.savez_compressed(dst, ages=fixed.astype(np.float64),
                                events=events.astype(np.int8))
            reuse_digests[s_.cohort_id] = cohort_digest(fixed, events)
            n_corrected += 1
        elif os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        # THE OLD SPEC IS CARRIED FORWARD, not the planned one. The built spec is what these
        # arrays were generated against and what every stored calibration was computed from --
        # including the build-time-derived window, which enters the OBSERVATION DIGEST and hence
        # the store key. Substituting the planned spec here would silently re-describe 202
        # windowed cohorts as unwindowed and orphan their results.
        d = dict(reuse_specs[s_.cohort_id])
        d["position"] = s_.position
        d["group"] = s_.group
        final_specs.append(d)
        checksums[s_.cohort_id] = reuse_digests[s_.cohort_id]
    if reuse and verbose and n_corrected:
        print(f"the current recipe changed {n_corrected} reused cohorts; their digests are new "
              f"and their stored results no longer apply")
    # A window that was non-binding in the reused library is non-binding now: same data.
    if reuse:
        nonbinding.extend([c for c in (rman.get("non_binding_windows") or [])
                           if c in reused])

    for r in results:
        if r[0] == "EXCLUDED":
            excluded.append({"cohort_id": r[1], "reason": r[2]})
            continue
        spec_d, ages, events, digest = r
        final_specs.append(spec_d)
        checksums[spec_d["cohort_id"]] = digest
        sp = CohortSpec.from_dict(spec_d)
        if sp.window is not None and not window_is_effective(sp, ages, events):
            nonbinding.append(sp.cohort_id)
        np.savez_compressed(os.path.join(out_dir, "cohorts", f"{spec_d['cohort_id']}.npz"),
                            ages=ages.astype(np.float64), events=events.astype(np.int8))

    # Positions define every prefix, so they must stay contiguous after an exclusion.
    final_specs = sorted(final_specs, key=lambda d: d["position"])
    for i, d in enumerate(final_specs):
        d["position"] = i + 1

    # The tiered body, which is what every tier is a prefix of. The optional groups follow it.
    n_main = sum(1 for d in final_specs if d.get("group", "main") == "main")

    manifest = {
        "library_version": LIBRARY_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": prov.to_dict(),
        "harness": harness_provenance().to_dict(),
        "hazard_row": "ExtH (the fitted value) -- see library-design.md",
        "n_cohorts": len(final_specs),
        # `full` is the actual size of the TIERED BODY, after exclusions -- not the planned size,
        # and NOT the whole library. A tier that names more cohorts than exist would make
        # `library.tier("full")` refuse, and a tier boundary that disagrees with the data is the
        # kind of small lie that costs an afternoon.
        #
        # Counting the whole library here would be worse than wrong: it would silently pull the
        # optional ladder into `full`, so "the full tier" would mean different cohorts before and
        # after the ladder was added, and two runs of "full" would not be comparable.
        "tiers": {**O.TIERS, "full": n_main},
        "group_counts": _group_counts(final_specs),
        "arm_counts": O.arm_counts([CohortSpec.from_dict(d) for d in final_specs]),
        "kappas": list(F.KAPPAS),
        "n_per_kappa": S.N_PER_KAPPA,
        "specs": final_specs,
        "checksums": checksums,
        "build_seconds": round(time.time() - t0, 1),
        # Real fit windows that contain all of their cohort's deaths. Not a defect -- the cohort's
        # actual observation model -- but it means those cohorts measure no truncation, so the
        # count belongs in the manifest rather than in nobody's head.
        "non_binding_windows": sorted(nonbinding),
        "n_non_binding_windows": len(nonbinding),
        # Cohorts that could not carry the feature their arm exists to test. Recorded rather than
        # silently given a feature that does nothing.
        "excluded": excluded,
        "n_excluded": len(excluded),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    if verbose:
        print(f"\nbuilt {len(final_specs)} cohorts in {manifest['build_seconds']}s -> {out_dir}")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=os.path.join(here, "data", "library", LIBRARY_VERSION))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N (for a smoke test)")
    ap.add_argument("--reuse", default=None, metavar="DIR",
                    help="an existing library directory whose unchanged cohorts are copied across "
                         "with their checksums instead of being regenerated. Use this when GROWING "
                         "the library: regenerating an existing cohort even one bit differently "
                         "changes its observation digest and orphans every stored calibration "
                         "for it.")
    a = ap.parse_args(argv)
    build(a.out, jobs=a.jobs, limit=a.limit, reuse=a.reuse)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
