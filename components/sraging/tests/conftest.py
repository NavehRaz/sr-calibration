"""Session-wide test configuration.

The first thing this module does is redirect Python's bytecode cache to `<repo>/.cache/pycache`, so that
importing the frozen oracle (`tests/oracle/v033`) or the live tester never writes a `__pycache__` into
either tree (constitution VI; research R4 step 4). It must stay the first statement that runs.

It also provides the `oracle_v033` session fixture and the NOT APPLICABLE guard (task T036): a parity
test that cannot be run validly here is reported as NOT APPLICABLE with its reason, and the session then
ends with a non-zero exit status and a banner -- it can never end green (constitution V).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")

# The released adapter carries no machine path (release 0.1.0): it reads SRAGING_TESTER_ROOT. Tests default it to the
# tester checkout under SRAGING_CODE5 (the folder holding the lab's checkouts; release 0.1.1), before anything imports
# the adapter. With neither set, it names that SRAGING_CODE5 is unset, and every test that needs the tester skips.
os.environ.setdefault("SRAGING_TESTER_ROOT", os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset",
                                                         "sr_calibration_tester"))

TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
PARITY = os.path.join(REPO, "tests", "parity")          # shared case helpers (engine_cases, parity_cases)
if PARITY not in sys.path:
    sys.path.insert(1, PARITY)

import pytest  # noqa: E402

NOT_APPLICABLE = "NOT APPLICABLE"
_not_applicable: list[tuple[str, str]] = []


@pytest.fixture(scope="session")
def oracle_v033():
    """The loaded v0.3.3 oracle, or NOT APPLICABLE (never a silent pass) when it cannot be loaded validly."""
    import oracle_v033 as O
    try:
        return O.load()
    except O.NotApplicable as ex:
        pytest.skip(f"{NOT_APPLICABLE}: {'; '.join(ex.problems)}")


def not_applicable(reason: str):
    """Call from a parity test whose input is not available yet (e.g. pilot-only theta points)."""
    pytest.skip(f"{NOT_APPLICABLE}: {reason}")


def pytest_runtest_logreport(report):
    if report.skipped and NOT_APPLICABLE in str(report.longrepr):
        _not_applicable.append((report.nodeid, str(report.longrepr)))


def pytest_sessionfinish(session, exitstatus):
    if _not_applicable and session.exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    if not _not_applicable:
        return
    tr = terminalreporter
    tr.write_sep("!", f"{len(_not_applicable)} parity test(s) NOT APPLICABLE -- this session is NOT a pass",
                 red=True, bold=True)
    reasons = sorted({r.split(NOT_APPLICABLE, 1)[-1].strip(": ')\"") for _, r in _not_applicable})
    for r in reasons[:10]:
        tr.write_line(f"  reason: {r[:300]}", red=True)
