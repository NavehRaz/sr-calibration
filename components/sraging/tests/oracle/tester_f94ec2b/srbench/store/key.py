"""The content address of a calibration.

THE WHOLE IDEA: the key is derived from everything that could change the result, and from nothing
else. Then reuse is correct BY CONSTRUCTION -- a changed method version or observation produces a
different key and therefore recomputes, and there is no invalidation rule for anyone to forget. A
reuse that should not have happened cannot be expressed.

The converse matters just as much, and is where this got it wrong once: anything that does NOT
change the result must stay OUT, or the store throws away work for no reason.

What goes in, and why each one:

    method name + VERSION   different code, different answer. The version is the method author's
                            responsibility and the one field in the contract whose neglect gives
                            a wrong answer quietly rather than an error loudly.
    cohort id               obvious.
    contract version        a contract change can change what a claim means.
    observation digest      catches everything about what the method was actually SHOWN --
                            including the window transform, which is applied by the harness and
                            so is not implied by the cohort id alone.
    method settings         anything the harness varied that the method reads.

What deliberately does NOT go in: the run id, the tier, the sizing mode, the wall-clock, the
date. Those describe the occasion, not the result. Including them would give every run fresh keys
and defeat the entire point.

⚑ AND NEITHER DOES `library_version`, though it did until 2026-09-24. Removing it was a bug fix.

    Adding cohorts to the library requires a new library version (FR-030), so while the version
    was in the digest, GROWING THE LIBRARY ORPHANED EVERY STORED RESULT -- including cohorts whose
    data had not changed by one bit. Measured at the time: it would have discarded 380
    `mle_profile` calibrations, about 300 CPU-hours, to add one hazard level.

    That is the exact opposite of what the store exists for, and it was redundant as well as
    harmful: the OBSERVATION DIGEST already distinguishes "same cohort id, different data", which
    is the only thing the library version was guarding. A label that describes the occasion was
    sitting in a key that should contain only what changes the answer.

    `library_version` is still RECORDED on the key object, because a key must stay
    self-describing for diagnosis. It simply no longer contributes to the address.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..contract.types import Observation

#: Bumped to 2 on 2026-09-24 when `library_version` was REMOVED from the digest. See
#: `make_key` for why, and `srbench/store/migrate.py` for the migration (which costs no
#: calibrations -- every stored record already carries the components its key is built from).
KEY_SCHEMA_VERSION = "2"


def _canonical(obj) -> bytes:
    """Stable bytes for any JSON-able object. `sort_keys` so dict order cannot change a key."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def observation_digest(obs: Observation) -> str:
    """A digest of exactly what the method was shown.

    Floats are formatted with `repr` so the digest is exact rather than rounded: two observations
    that differ in the last bit ARE different observations, and silently treating them as equal
    is the kind of near-miss reuse this whole scheme exists to prevent.
    """
    payload = {
        "ages": [repr(float(a)) for a in obs.ages],
        "events": [int(e) for e in obs.events],
        "delta": None if obs.delta is None else repr(float(obs.delta)),
        "window": None if obs.window is None else [repr(float(x)) for x in obs.window],
    }
    # ADDED ONLY WHEN PRESENT, and this is load-bearing. `json.dumps` has no skip-null mode, so a
    # `"known": null` key would serialise and change the digest of EVERY observation ever made --
    # orphaning all 5,776 stored calibrations for cohorts nobody was told anything about.
    # Measured precedent for getting this wrong: 380 calibrations, ~300 CPU-hours (see the module
    # docstring on `library_version`).
    #
    # A method that asked to be told something DOES get a different address, and that is correct:
    # it was shown something different, so it is a different calibration.
    known = getattr(obs, "known", None)
    if known:
        from ..contract.types import _canonical_known
        payload["known"] = _canonical_known(known)
    return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class StoreKey:
    """The address, plus the parts it was built from -- kept so a key is self-describing.

    Storing the parts matters for diagnosis: when a run unexpectedly recomputes everything, the
    question is always "which component changed?", and a bare hash cannot answer it.
    """

    digest: str
    method_name: str
    method_version: str
    cohort_id: str
    library_version: str
    contract_version: str
    observation_digest: str
    settings_digest: str

    def __str__(self) -> str:
        return self.digest

    def to_dict(self) -> dict:
        return {
            "digest": self.digest,
            "key_schema_version": KEY_SCHEMA_VERSION,
            "method_name": self.method_name,
            "method_version": self.method_version,
            "cohort_id": self.cohort_id,
            "library_version": self.library_version,
            "contract_version": self.contract_version,
            "observation_digest": self.observation_digest,
            "settings_digest": self.settings_digest,
        }


def make_key(*, method_name: str, method_version: str, cohort_id: str,
             library_version: str, contract_version: str,
             observation: Observation | None = None,
             observation_digest_: str | None = None,
             settings: dict | None = None) -> StoreKey:
    """Build the content address. Either an `observation` or a precomputed digest is required."""
    if observation_digest_ is None:
        if observation is None:
            raise ValueError("make_key needs either `observation` or `observation_digest_`")
        observation_digest_ = observation_digest(observation)

    settings_digest = hashlib.sha256(_canonical(settings or {})).hexdigest()[:16]

    parts = {
        "v": KEY_SCHEMA_VERSION,
        "method": method_name,
        "method_version": method_version,
        "cohort": cohort_id,
        # NO "library": the observation digest already catches a changed cohort, and including the
        # version orphaned the whole store every time the library grew. See the docstring.
        "contract": contract_version,
        "obs": observation_digest_,
        "settings": settings_digest,
    }
    digest = hashlib.sha256(_canonical(parts)).hexdigest()
    return StoreKey(
        digest=digest,
        method_name=method_name,
        method_version=method_version,
        cohort_id=cohort_id,
        library_version=library_version,
        contract_version=contract_version,
        observation_digest=observation_digest_,
        settings_digest=settings_digest,
    )
