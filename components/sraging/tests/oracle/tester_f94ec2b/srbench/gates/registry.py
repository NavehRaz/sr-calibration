"""The gate registry.

A GATE EARNS ITS PLACE BY CATCHING SOMETHING REAL. Every gate here records the specific historical
failure it was written against, and every gate ships with a FIXTURE ON WHICH IT FAILS. A meta-test
iterates the registry and asserts each one fails on its own fixture.

That meta-test is not ceremony. Nineteen regression fingerprints in this project's history once
hashed identically because the fixture produced zero deaths — the suite would have passed whatever
broke, and it passed for months. A gate that has never been shown to fail is not evidence.

WARN IS NOT PASS. The summary counts passes, warnings and failures separately, and a green line
cannot be printed while any check has warned. An earlier version of the predecessor's self-test
counted only failures and printed "1/1 checks passed" for a run whose single check had warned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass(frozen=True, slots=True)
class GateOutcome:
    gate_id: str
    title: str
    outcome: str
    detail: str
    motivating_failure: str
    blocking: bool = True

    def to_dict(self) -> dict:
        return {"gate_id": self.gate_id, "title": self.title, "outcome": self.outcome,
                "detail": self.detail, "motivating_failure": self.motivating_failure,
                "blocking": self.blocking}


@dataclass(frozen=True, slots=True)
class Gate:
    """One check, its history, and the input that makes it fail.

    `failing_fixture` is REQUIRED. A gate without one cannot be shown to work, and this project
    has already shipped a whole suite of tests that could not fail.
    """

    gate_id: str
    title: str
    motivating_failure: str
    check: Callable[..., GateOutcome]
    failing_fixture: Callable[[], dict]
    passing_fixture: Callable[[], dict]
    blocking: bool = True

    def run(self, **kw) -> GateOutcome:
        return self.check(**kw)


_GATES: dict[str, Gate] = {}


def register(gate: Gate) -> Gate:
    if gate.gate_id in _GATES:
        raise ValueError(f"duplicate gate id {gate.gate_id!r}")
    _GATES[gate.gate_id] = gate
    return gate


def all_gates() -> dict[str, Gate]:
    from . import checks  # noqa: F401  - registration by import
    return dict(sorted(_GATES.items()))


def outcome(gate: Gate, ok: bool, detail: str, warn_instead: bool = False) -> GateOutcome:
    o = PASS if ok else (WARN if warn_instead else FAIL)
    return GateOutcome(gate_id=gate.gate_id, title=gate.title, outcome=o, detail=detail,
                       motivating_failure=gate.motivating_failure, blocking=gate.blocking)


def run_against(library, claim_sets, run_manifest, reference_sets=None,
                null_sets=None) -> list[GateOutcome]:
    """Run every gate that can be evaluated against a REAL run, not just against a fixture.

    A gate that only ever sees its own fixture proves it works and nothing else. This is where the
    gates actually guard a result: each one is handed the run's own data, and a blocking failure
    means the result is not publishable.

    Gates whose inputs this call cannot supply are reported as `not applicable` rather than
    silently skipped -- a check that quietly does not run is indistinguishable from one that
    passed, which is the failure mode the whole registry exists to avoid.
    """
    from ..contract.types import PARAMETERS
    from ..library import order as O
    from ..score import claims as CL

    gates = all_gates()
    specs = [library.spec(cs.cohort_id) for cs in claim_sets]
    truths = library.truths([cs.cohort_id for cs in claim_sets])

    # Inputs each gate needs, drawn from the run itself.
    bounds = {p: (1e-12, 1e3) for p in PARAMETERS}
    bounds["kappa"] = (1e-3, 300.0)
    points_per_kappa: dict = {}
    for sp in specs:
        points_per_kappa.setdefault(sp.kappa, set()).add(sp.cluster_key)
    points_per_kappa = {k: len(v) for k, v in points_per_kappa.items()}

    coverages = []
    for p in PARAMETERS:
        c = CL.coverage(claim_sets, truths, p, "closed")
        # G15: a coverage figure is only publishable beside the data-ignoring null. If no null run
        # was supplied, that is a real, actionable failure -- run `null_prior` -- not a nuisance.
        nullc = None
        if null_sets:
            nullc = CL.coverage(null_sets, truths, p, "closed").coverage
        coverages.append({"parameter": p, "shape_class": c.shape_class, "null_coverage": nullc})

    claims = [{"n_components": cl.n_components}
              for cs in claim_sets for cl in cs.claims if cl.is_claim]

    supplied = {
        "G1": {"specs": specs, "bounds": bounds},
        "G2": {"stratum_counts": O.stratum_counts(specs)},
        "G13": {"claims": claims},
        "G14": {"points_per_kappa": points_per_kappa},
        "G15": {"coverages": coverages},
        "G16": {"coverages": coverages},
        "G11": {},          # self-contained
        "G4": {},           # self-contained
    }

    out = []
    for gid, g in gates.items():
        if gid not in supplied:
            out.append(GateOutcome(gid, g.title, WARN,
                                   "not applicable to this call -- no input available from the "
                                   "run; reported rather than skipped",
                                   g.motivating_failure, g.blocking))
            continue
        try:
            out.append(g.run(**supplied[gid]))
        except Exception as ex:  # noqa: BLE001
            out.append(GateOutcome(gid, g.title, FAIL, f"gate raised: {ex}",
                                   g.motivating_failure, g.blocking))
    return out


def summarise(outcomes: list[GateOutcome]) -> dict:
    """Counts, kept SEPARATE. A warn is not a pass."""
    n_pass = sum(1 for o in outcomes if o.outcome == PASS)
    n_warn = sum(1 for o in outcomes if o.outcome == WARN)
    n_fail = sum(1 for o in outcomes if o.outcome == FAIL)
    return {"passed": n_pass, "warned": n_warn, "failed": n_fail, "total": len(outcomes),
            "publishable": n_fail == 0,
            "clean": n_fail == 0 and n_warn == 0}


def format_report(outcomes: list[GateOutcome]) -> str:
    lines = ["=" * 78, "GATES -- each records the failure it was written against", "=" * 78]
    for o in outcomes:
        lines.append(f"\n[{o.outcome}] {o.gate_id}  {o.title}")
        lines.append(f"       {o.detail}")
        lines.append(f"       written against: {o.motivating_failure}")
    s = summarise(outcomes)
    lines += ["", "=" * 78,
              f"{s['passed']} passed, {s['warned']} warned, {s['failed']} failed "
              f"(of {s['total']})"]
    # A green line cannot be printed while anything warned.
    if s["failed"]:
        lines.append("  FAIL -> this result is not publishable")
    if s["warned"]:
        lines.append("  WARN -> a check did not pass; it is NOT counted as a pass")
    if s["clean"]:
        lines.append("  all clear")
    return "\n".join(lines)
