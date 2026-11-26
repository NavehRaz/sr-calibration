"""Where did the code that produced this number actually come from?

This exists because of a measured failure mode on this machine, not as hygiene. `SRK` is
pip-installed editable into the venv, and the editable install points at
`/Volumes/alon/navehr/SRtools_kappa` -- a cluster mount whose copy is stale. So:

    $ python -c "import SRK, os; print(os.path.dirname(os.path.dirname(SRK.__file__)))"
    /Volumes/alon/navehr/SRtools_kappa          <-- NOT the local checkout

A run against the wrong copy is otherwise indistinguishable from a run against the right one:
same command, same output shape, different numbers. The stale copy predates a correctness fix to
the negligibility floor, so it does not merely lag -- it is wrong in a way that moves results.

Hence: resolve, record, and REFUSE. Not warn.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, asdict

#: Exit code for "refused before doing anything", per the CLI contract.
EXIT_REFUSED = 3

ENV_SUT = "SRBENCH_SUT"


class ProvenanceError(RuntimeError):
    """The loaded code is not the code that was asked for."""


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    """Recorded into every stored calibration, so a result can always be traced to its code."""

    package: str
    resolved_path: str
    revision: str | None = None
    dirty: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def git_revision(path: str) -> tuple[str | None, bool | None]:
    """(short sha, dirty) for the checkout containing `path`, or (None, None) if not a repo."""
    try:
        rev = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return None, None
        status = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
        return rev.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def requested_sut() -> str | None:
    """The checkout of the system under test the user asked for, if any."""
    p = os.environ.get(ENV_SUT)
    return os.path.realpath(os.path.expanduser(p)) if p else None


def resolve_sut(strict: bool = True) -> CodeProvenance:
    """Import the system under test, record where it came from, and refuse a mismatch.

    `strict=False` is for reporting what WOULD be loaded without failing -- used by
    `srbench library info` and diagnostics. Anything that produces a number uses strict=True.
    """
    want = requested_sut()

    # The numpy>=2 shim must be applied before the FP modules are imported. Do it here so no
    # caller can forget; the upstream file is deliberately NOT edited to accommodate us.
    from . import npcompat  # noqa: F401  (import for side effect)

    try:
        import SRK  # type: ignore
    except ImportError as ex:
        raise ProvenanceError(
            f"could not import the system under test (SRK): {ex}. "
            f"Set {ENV_SUT} to the checkout and ensure it is on PYTHONPATH."
        ) from ex

    got = os.path.realpath(os.path.dirname(os.path.dirname(SRK.__file__)))
    rev, dirty = git_revision(got)
    prov = CodeProvenance(package="SRK", resolved_path=got, revision=rev, dirty=dirty)

    if want and got != want and strict:
        raise ProvenanceError(
            "REFUSING: the system under test resolved to a different checkout than requested.\n"
            f"  requested ({ENV_SUT}): {want}\n"
            f"  resolved              : {got}\n"
            "\n"
            "This is almost always the stale cluster mount: SRK is pip-installed editable and\n"
            "points at /Volumes/alon/navehr/SRtools_kappa, which predates a correctness fix.\n"
            "A run against the wrong copy is indistinguishable from a correct one, which is why\n"
            "this refuses rather than warns.\n"
            "\n"
            "  cd <the checkout you want> && export PYTHONPATH=\"$PWD\"\n"
        )
    return prov


def format_code_line(prov: CodeProvenance) -> str:
    """The line every command that loads the system under test prints FIRST."""
    bits = [f"CODE: {prov.package} <- {prov.resolved_path}"]
    if prov.revision:
        bits.append(f"@{prov.revision}{'+dirty' if prov.dirty else ''}")
    return " ".join(bits)


def harness_provenance() -> CodeProvenance:
    """This repository's own revision, stamped into every artifact it produces."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rev, dirty = git_revision(here)
    return CodeProvenance(package="srbench", resolved_path=here, revision=rev, dirty=dirty)
