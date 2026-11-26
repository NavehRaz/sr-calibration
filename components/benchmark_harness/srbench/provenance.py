"""Where did the code that produced this number actually come from?

This exists because of a measured failure mode on this machine, not as hygiene. `SRK` is
pip-installed editable into the venv, and the editable install points at
`<cluster mount>/SRtools_kappa` -- a stale copy on a cluster mount. So:

    $ python -c "import SRK, os; print(os.path.dirname(os.path.dirname(SRK.__file__)))"
    <cluster mount>/SRtools_kappa               <-- NOT the local checkout

A run against the wrong copy is otherwise indistinguishable from a run against the right one:
same command, same output shape, different numbers. The stale copy predates a correctness fix to
the negligibility floor, so it does not merely lag -- it is wrong in a way that moves results.

Hence: resolve, record, and REFUSE. Not warn.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

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
    #: WHY `revision` is None, when it is. "ok" | "not_a_repo" | "timeout" | "error:<Type>".
    #: Without this, a record whose git lookup TIMED OUT is indistinguishable from one legitimately
    #: produced outside a checkout -- and 26 such records exist in the store because a 10 s
    #: `git status` on a loaded cluster filesystem silently returned None (sraging, 2026-10-18).
    revision_status: str = "ok"
    #: Host and library versions. The same claim is not guaranteed bit-identical across CPU
    #: families: jax2 measured box-set caps flipping shape on 9 of 10 human cohorts across hosts,
    #: and sraging's port reproduces our stored claims on 38 of 118 cohorts fitted on WEXAC and
    #: 0 of 22 fitted on a Mac. Without this block that difference cannot be attributed, because
    #: host, library version and cohort all change together.
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _cpu_model() -> str | None:
    """A human-readable CPU model. platform.processor() is usually EMPTY on Linux, which is the
    platform that matters here, so read the OS's own answer first."""
    import platform
    import subprocess
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or None


#: Versions recorded with every calibration. Read from package METADATA, never by importing:
#: importing jax sets float64 and PJRT_NPROC process-wide (sraging's generator note), so a
#: provenance call must not be the thing that configures the process it is describing.
RECORDED_DISTRIBUTIONS = ("numpy", "scipy", "jax", "jaxlib", "ml_dtypes")


def environment() -> dict:
    """Host, CPU and library versions, for attributing a claim that differs across machines."""
    import platform
    from importlib import metadata
    env = {
        "host": platform.node() or None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
    }
    for dist in RECORDED_DISTRIBUTIONS:
        try:
            env[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env[dist] = None
        except Exception:  # noqa: BLE001 - provenance must never break a calibration
            env[dist] = None
    return env


#: `git status --porcelain` walks the whole worktree. On a loaded cluster filesystem, under a
#: 100-way array, 10 s was not enough: 26 records in the store carry no revision for that reason
#: alone. Raised, and a timeout is now RECORDED rather than silently flattened to None.
GIT_TIMEOUT_S = 60


def git_revision(path: str, timeout: float = GIT_TIMEOUT_S) -> tuple[str | None, bool | None, str]:
    """(short sha, dirty, status) for the checkout containing `path`.

    The third element is why there is no revision, and it exists because "not a repo" and "the
    lookup failed" produced the SAME record until 2026-10-18 -- so a result written from an
    unversioned directory and one whose git call timed out under load were indistinguishable
    afterwards, and neither could be attributed to code.
    """
    try:
        rev = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=timeout)
        if rev.returncode != 0:
            return None, None, "not_a_repo"
        status = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                                capture_output=True, text=True, timeout=timeout)
        if status.returncode != 0:
            # The revision is real even if the dirty check failed; say so rather than lose it.
            return rev.stdout.strip(), None, "dirty_unknown"
        return rev.stdout.strip(), bool(status.stdout.strip()), "ok"
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    except (OSError, subprocess.SubprocessError) as ex:
        return None, None, f"error:{type(ex).__name__}"


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
    rev, dirty, status = git_revision(got)
    prov = CodeProvenance(package="SRK", resolved_path=got, revision=rev, dirty=dirty,
                          revision_status=status)

    if want and got != want and strict:
        raise ProvenanceError(
            "REFUSING: the system under test resolved to a different checkout than requested.\n"
            f"  requested ({ENV_SUT}): {want}\n"
            f"  resolved              : {got}\n"
            "\n"
            "This is almost always the stale cluster mount: SRK is pip-installed editable and\n"
            "points at a copy on a cluster mount, which predates a correctness fix.\n"
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
    rev, dirty, status = git_revision(here)
    return CodeProvenance(package="srbench", resolved_path=here, revision=rev, dirty=dirty,
                          revision_status=status, environment=environment())
