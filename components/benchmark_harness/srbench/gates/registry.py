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


def markdown_table_blocks(text: str) -> list[str]:
    """Split a markdown document into one block per TABLE.

    G10 compares the column count of every row WITHIN a block, so the whole document cannot be
    passed as one block: two legitimate tables of different widths would read as a mismatch. A
    block is therefore a contiguous run of lines starting with `|`, which is exactly one table.
    """
    blocks, cur = [], []
    for line in (text or "").split("\n"):
        if line.strip().startswith("|"):
            cur.append(line)
        elif cur:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def run_against(library, claim_sets, run_manifest, reference_sets=None,
                null_sets=None, report_markdown=None, recomputations=None) -> list[GateOutcome]:
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

    # `reference_sets` was accepted and IGNORED until 2026-10-18, so G5 and G6 -- the two gates
    # whose whole subject is comparing one set of results against another -- reported "not
    # applicable" on every real run, including the reference's own. The input was in the
    # signature the entire time.
    if reference_sets is not None:
        from ..score import floors as F
        a_ids = {cs.cohort_id for cs in claim_sets}
        b_ids = {cs.cohort_id for cs in reference_sets}
        paired_ids = a_ids & b_ids

        # G6. The scorer compares on the INTERSECTION, so the arithmetic is always paired; what
        # is not safe is the intersection being a strict subset of either side, because then the
        # figure covers fewer cohorts than either manifest claims and the dropped ones are
        # invisible. That is the composition difference G6 exists for.
        supplied["G6"] = {"comparisons": [{
            "what": f"{run_manifest.get('method','run')} vs reference",
            "paired": bool(paired_ids) and a_ids == b_ids,
            "n_run": len(a_ids), "n_reference": len(b_ids), "n_paired": len(paired_ids),
        }]}

        # G5. Accuracy must be read at a FIXED floor: the floor goes as 1/sqrt(n), and letting it
        # move once turned a flat ~0.60 dex into an apparent degradation to 0.892.
        kept = [cs for cs in claim_sets if cs.cohort_id in paired_ids]
        spec_by_id = {c: library.spec(c) for c in paired_ids}
        acc = []
        for p in CL.scored_quantities(truths):
            a = CL.accuracy(kept, truths, p, F.cohort_floors(spec_by_id, p),
                            floor_is_fixed=True, cohort_set=f"{len(paired_ids)} paired")
            acc.append(a.to_dict())
        supplied["G5"] = {"accuracy_results": acc}

    if report_markdown is not None:
        supplied["G10"] = {"markdown": list(report_markdown)}
    if recomputations is not None:
        supplied["G8"] = {"recomputations": list(recomputations)}

    # Gates this harness produces NO input for, as opposed to inputs this CALL happened not to be
    # given. Saying "not applicable to this call" for these implies the input might arrive with a
    # richer call, and it never will: they need the estimator's internals -- the profile it walked,
    # the values it evaluated, its likelihood at the planted truth -- and `calibrate` returns a
    # ClaimSet, which carries none of that. Fixing it is a CONTRACT change, not a plumbing one.
    # Wirable, but not from anything this call is handed. Distinguished from both the generic
    # "not given" and the permanent "cannot exist" so nobody has to rediscover which it is.
    NEEDS_A_CHANGE_ELSEWHERE = {
        "G8": "report.verify reports only MISMATCHES, not an enumeration of the figures it "
              "recomputed, so there is nothing to count. Passing [{recomputed: True}] on a clean "
              "verify would make this gate pass vacuously, which is the failure it guards against. "
              "Wiring it means having verify report its own coverage",
    }

    NO_INPUT_EXISTS = {
        "G3": "needs the profile a method walked over a superset and a subset; a ClaimSet does "
              "not carry it",
        "G7": "needs an agreement test; this harness implements none (no se_of_difference_of_means "
              "anywhere), so there is nothing to check the null of",
        "G9": "needs the raw profile evaluations a method rejected or interpolated; a ClaimSet "
              "does not carry them",
        "G12": "needs loglik(truth) - loglik(fit) per cohort, which only the method can compute "
               "on its own objective; a ClaimSet does not carry it",
    }

    out = []
    for gid, g in gates.items():
        if gid not in supplied:
            why = NO_INPUT_EXISTS.get(gid)
            todo = NEEDS_A_CHANGE_ELSEWHERE.get(gid)
            if why:
                detail = f"NOT RUNNABLE against any run: {why}. Reported rather than skipped."
            elif todo:
                detail = f"not wired yet: {todo}. Reported rather than skipped."
            else:
                detail = ("not applicable to this call -- no input available from the run; "
                          "reported rather than skipped")
            out.append(GateOutcome(gid, g.title, WARN, detail, g.motivating_failure, g.blocking))
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
