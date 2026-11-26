"""The divergence register (`divergences/register.toml`; constitution VII; FR-025)."""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass

TAGS = ("MODEL", "ESTIMATOR/CONVENTION", "CODE")
STATUSES = ("ported-as-is", "diverged", "refused-input")
FIELDS = ("id", "title", "tag", "status", "legacy", "new", "source", "impact", "downstream", "regression_test")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTER = os.path.join(_REPO, "divergences", "register.toml")


@dataclass(frozen=True)
class DivergenceEntry:
    id: str
    title: str
    tag: str
    status: str
    legacy: str
    new: str
    source: str
    impact: object
    downstream: tuple
    regression_test: str
    condition: dict | None = None        # spec 002: machine-checkable condition of a change to the default


CONDITION_KEYS = ("applies_to", "terms", "value_bound", "gradient_bound")
#: an impact still to be measured, by the phase it names: accepted only while that phase's result is not written; once
#: it is, the entry must carry the measured numbers (spec 004 tasks T021, T055)
PENDING_IMPACT = {"measured by phase d401 (T055)": "tests/acceptance/results/d401_004.json"}
#: spec 004 (data-model "Sampler divergence entry"; research R12, Dh): a row's optional kind; a sampler row's impact is
#: the distribution shift at a stated n on a named panel, and constitution VII's dloglik / dtheta fields do not fit it
KINDS = ("likelihood", "sampler")
SAMPLER_IMPACT = ("sup_abs_dS", "n", "panel", "step", "measured_in")
SAMPLER_REFUSED = ("dloglik", "dtheta_dex", "claim_shape_changes")


def _pending(impact) -> list[str]:
    values = impact.values() if isinstance(impact, dict) else [impact]
    return sorted({m for v in values for m in PENDING_IMPACT if str(v).startswith(m)})


def _sampler_problems(rid, r) -> list[str]:
    kind = r.get("kind")
    if kind is None:
        return []
    if kind not in KINDS:
        return [f"{rid}: kind {kind!r} not in {KINDS}"]
    if kind != "sampler":
        return []
    out, imp = [], r.get("impact")
    if isinstance(imp, dict):
        out += [f"{rid}: a sampler row's impact is a distribution shift; {k!r} is refused on it" for k in
                SAMPLER_REFUSED if k in imp]
    if r.get("status") == "diverged":
        missing = [k for k in SAMPLER_IMPACT if not isinstance(imp, dict) or k not in imp]
        if missing:
            out.append(f"{rid}: a diverged sampler row's impact needs {list(SAMPLER_IMPACT)} (Dh); missing {missing}")
    elif r.get("status") == "ported-as-is" and imp != "none: not fixed" and not isinstance(imp, dict):
        out.append(f"{rid}: a ported-as-is sampler row reads 'none: not fixed', got {imp!r}")
    return out


def validate(rows, repo: str = _REPO) -> list[str]:
    """Every problem with the register's rows (empty = valid)."""
    problems, seen = [], set()
    for i, r in enumerate(rows):
        rid = r.get("id", f"#{i}")
        for f in FIELDS:
            if f not in r:
                problems.append(f"{rid}: missing field {f!r}")
        if r.get("tag") not in TAGS:
            problems.append(f"{rid}: tag {r.get('tag')!r} not in {TAGS}")
        if r.get("status") not in STATUSES:
            problems.append(f"{rid}: status {r.get('status')!r} not in {STATUSES}")
        if r.get("status") == "diverged" and r.get("impact") in (None, "", "none: not fixed"):
            problems.append(f"{rid}: a diverged entry needs its measured impact (constitution VII)")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        cond = r.get("condition")
        changes_v1 = r.get("status") == "diverged" and "v1" in str(r.get("new", ""))
        if changes_v1 and not cond:
            problems.append(f"{rid}: a diverged entry that changes numerics 'v1' needs a [divergence.condition] "
                            f"table (FR-023)")
        if cond:
            for k in CONDITION_KEYS:
                if k not in cond:
                    problems.append(f"{rid}: condition is missing {k!r}")
            if cond.get("value_bound") == "mass_stiffness":
                for k in ("c_stiff", "floor_ulps", "c_source"):
                    if k not in cond:
                        problems.append(f"{rid}: a mass_stiffness bound needs {k!r}")
                if "c_stiff" in cond and not isinstance(cond["c_stiff"], (int, float)):
                    problems.append(f"{rid}: c_stiff must be a number, measured and frozen (c_source)")
        imp = r.get("impact")
        problems += _sampler_problems(rid, r)
        for marker in _pending(imp):
            if os.path.exists(os.path.join(repo, PENDING_IMPACT[marker])):
                problems.append(f"{rid}: impact still reads {marker!r} but {PENDING_IMPACT[marker]} is written: fill "
                                f"in the measured impact")
        if isinstance(imp, dict):
            for k in ("dtheta_dex", "claim_shape_changes"):
                named = re.findall(r"spec \d{3}[a-z]?\b", str(imp.get(k, "")))
                spec = named[-1] if named else None          # the deferral target is the last spec the field names
                if spec and not any(spec in d for d in r.get("downstream", [])):
                    problems.append(f"{rid}: impact {k} deferred to {spec} but downstream does not name {spec}")
    return problems


def load(path: str = REGISTER) -> list[DivergenceEntry]:
    rows = tomllib.load(open(path, "rb")).get("divergence", [])
    problems = validate(rows)
    if problems:
        raise ValueError("divergence register invalid:\n  " + "\n  ".join(problems))
    return [DivergenceEntry(**{**{f: r[f] for f in FIELDS}, "downstream": tuple(r["downstream"]),
                               "condition": r.get("condition")}) for r in rows]
