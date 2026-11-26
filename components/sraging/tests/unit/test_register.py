"""The divergence register is complete and every entry's regression test exists (T061; FR-025)."""
import os
import subprocess
import sys
import tomllib

import pytest

from sraging import divergences as D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_register_loads_and_is_valid():
    entries = D.load()
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))
    for required in ("D-001", "D-002", "D-003", "D-004", "D-005", "D-006", "D-007",
                     "D-101", "D-102", "D-103", "D-104"):
        assert required in ids
    assert all(e.tag in D.TAGS for e in entries)


def test_every_regression_test_is_collected():
    entries = D.load()
    nodes = sorted({e.regression_test for e in entries})
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                        "-m", "", *nodes], cwd=REPO, capture_output=True, text=True, timeout=600)
    collected = {line.strip() for line in r.stdout.splitlines() if "::" in line}
    missing = [n for n in nodes if not any(c == n or c.startswith(n + "[") for c in collected)]
    assert missing == [], r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("breakage,expect", [
    (lambda r: r.pop("regression_test"), "missing field 'regression_test'"),
    (lambda r: r.__setitem__("tag", "BUG"), "tag 'BUG'"),
    (lambda r: r.__setitem__("status", "diverged") or r.__setitem__("impact", "none: not fixed"),
     "needs its measured impact"),
])
def test_a_broken_entry_fails(tmp_path, breakage, expect):
    rows = tomllib.load(open(D.REGISTER, "rb"))["divergence"]
    rows = [dict(r) for r in rows]
    breakage(rows[0])
    assert any(expect in p for p in D.validate(rows))


def test_duplicate_ids_fail():
    rows = tomllib.load(open(D.REGISTER, "rb"))["divergence"]
    assert any("duplicate" in p for p in D.validate(rows + [rows[0]]))


def test_v1_divergences_carry_conditions():
    """Spec 002 (FR-023): every entry that changes numerics "v1" has a machine-checkable condition."""
    by_id = {e.id: e for e in D.load()}
    for eid in ("D-004", "D-106", "D-107"):                     # D-108 retired by spec 006 (both numerics clip)
        cond = by_id[eid].condition
        assert cond and all(k in cond for k in D.CONDITION_KEYS), eid


@pytest.mark.parametrize("breakage,expect", [
    (lambda r: r.pop("condition"), "needs a [divergence.condition] table"),
    (lambda r: r["condition"].pop("terms"), "condition is missing 'terms'"),
    (lambda r: r["condition"].pop("c_source"), "needs 'c_source'"),
    (lambda r: r["condition"].__setitem__("c_stiff", "PENDING"), "c_stiff must be a number"),
    (lambda r: r.__setitem__("downstream", []), "deferred to spec 006 but downstream does not name spec 006"),
])
def test_a_broken_v1_entry_fails(breakage, expect):
    rows = [dict(r) for r in tomllib.load(open(D.REGISTER, "rb"))["divergence"]]
    d004 = next(r for r in rows if r["id"] == "D-004")
    d004["condition"] = dict(d004["condition"])
    breakage(d004)
    assert any(expect in p for p in D.validate(rows))


def test_a_pending_impact_marker_is_accepted_only_until_its_phase_writes_its_result(tmp_path):
    """Spec 004 T021: D-401's impact reads 'measured by phase d401 (T055)' until phase d401 writes its result; from
    then the register refuses it until T055 fills in the measured numbers. Since T055 the live entry is filled, so the
    marker is planted in a copy."""
    live = [dict(r) for r in tomllib.load(open(D.REGISTER, "rb"))["divergence"]]
    marker = "measured by phase d401 (T055)"
    assert not D._pending(next(r for r in live if r["id"] == "D-401")["impact"])             # T055 filled it
    rows = [dict(r, impact={**r["impact"], "measured_here": marker, "dloglik": f"{marker}: per cohort"})
            if r["id"] == "D-401" else r for r in live]
    assert not any("D-401" in p for p in D.validate(rows, repo=str(tmp_path)))                 # the twin: not run yet
    res = tmp_path / D.PENDING_IMPACT[marker]
    res.parent.mkdir(parents=True)
    res.write_text("{}")
    assert any("D-401" in p and "fill in the measured impact" in p for p in D.validate(rows, repo=str(tmp_path)))
    assert not any("D-401" in p for p in D.validate(live, repo=str(tmp_path)))     # the filled entry, its result written


# ------------------------------------------------------------------------- spec 004: sampler rows (tasks T062)
def _sampler(status="diverged", impact=None):
    return {"id": "D-9xx", "title": "t", "tag": "CODE", "kind": "sampler", "status": status, "legacy": "l", "new": "n",
            "source": "s", "downstream": [], "regression_test": "tests/unit/x.py::t",
            "impact": impact if impact is not None else {"sup_abs_dS": 0.012, "n": 25000, "panel": "plants",
                                                         "step": "dt = 0.5", "measured_in": "plants_004.json"}}


def test_a_diverged_sampler_row_needs_its_distribution_shift():
    assert D.validate([_sampler()]) == []                                                        # the twin
    bad = _sampler(impact={"sup_abs_dS": 0.012, "n": 25000})
    assert any("sampler" in p and "panel" in p for p in D.validate([bad]))
    assert any("sampler" in p for p in D.validate([_sampler(impact="measured somewhere")]))


def test_a_ported_as_is_sampler_row_reads_none_not_fixed():
    assert D.validate([_sampler(status="ported-as-is", impact="none: not fixed")]) == []
    assert any("sampler" in p for p in D.validate([_sampler(status="ported-as-is", impact={"dtheta_dex": "x"})]))


@pytest.mark.parametrize("field", ["dtheta_dex", "claim_shape_changes", "dloglik"])
def test_dtheta_and_dloglik_fields_are_refused_on_a_sampler_row(field):
    row = _sampler()
    row["impact"] = {**row["impact"], field: "none"}
    assert any(field in p and "sampler" in p for p in D.validate([row]))


def test_an_unknown_kind_is_refused():
    assert any("kind" in p for p in D.validate([dict(_sampler(), kind="langevin")]))
    assert D.validate([dict(_sampler(), kind="sampler")]) == []                                  # the twin


def test_d410_is_the_ported_as_is_clamp():
    e = {x.id: x for x in D.load()}["D-410"]
    rows = {r["id"]: r for r in tomllib.load(open(D.REGISTER, "rb"))["divergence"]}
    assert (e.tag, e.status, e.impact, rows["D-410"]["kind"]) == ("CODE", "ported-as-is", "none: not fixed", "sampler")
    assert e.regression_test == "tests/unit/langevin/test_kernels.py::test_d410_default_wall_is_clamp"
