"""Which run currently holds the REFERENCE role.

THE REFERENCE IS A ROLE RECORDED IN DATA, NEVER A METHOD NAMED IN CODE.

The reference method is expected to change: the whole purpose of this repository is that someone
finds a better one. A design that names the incumbent in code makes that a code change plus a
re-run; a design that names it in data makes it a one-line edit that re-scores from the store with
ZERO calibrations.

It also removes an asymmetry that would otherwise creep in. If the reference were special in code
it would stop being scored by the same rules as its challengers -- which is exactly how an
incumbent becomes unfalsifiable.

And every deliverable stamps which reference it used. Once the reference can move, two documents
with the same title can rest on different baselines and be indistinguishable without the stamp.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReferencePointer:
    run_id: str
    method_name: str
    method_version: str
    library_version: str
    assigned_at: str
    assigned_by: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "method_name": self.method_name,
                "method_version": self.method_version, "library_version": self.library_version,
                "assigned_at": self.assigned_at, "assigned_by": self.assigned_by,
                "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "ReferencePointer":
        return cls(run_id=d["run_id"], method_name=d["method_name"],
                   method_version=d["method_version"],
                   library_version=d.get("library_version", ""),
                   assigned_at=d.get("assigned_at", ""), assigned_by=d.get("assigned_by", ""),
                   note=d.get("note", ""))

    def stamp(self) -> str:
        """The line every deliverable carries."""
        return (f"reference: {self.method_name} v{self.method_version} "
                f"(run {self.run_id}, library {self.library_version})")


def default_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "data", "reference", "REFERENCE.json")


def get(path: str | None = None) -> ReferencePointer | None:
    p = path or default_path()
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return ReferencePointer.from_dict(json.load(fh))


def set_reference(run_manifest: dict, note: str = "", by: str = "",
                  path: str | None = None) -> ReferencePointer:
    """Point the role at a run. A DATA EDIT -- no calibration is performed."""
    p = path or default_path()
    ptr = ReferencePointer(
        run_id=run_manifest.get("run_id", run_manifest.get("started_at", "?")),
        method_name=run_manifest["method"],
        method_version=run_manifest["method_version"],
        library_version=run_manifest["library_version"],
        assigned_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        assigned_by=by, note=note,
    )
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(ptr.to_dict(), fh, indent=1)
    return ptr


def require_comparable(a: dict, b: dict) -> None:
    """Refuse to compare runs from different libraries or incompatible contracts.

    The pairing would be FICTIONAL: the same cohort id in a rebuilt library can be different data,
    so a "paired" comparison across versions pairs things that are not the same cohort.
    """
    if a["library_version"] != b["library_version"]:
        raise ValueError(
            f"refusing to compare: library versions differ ({a['library_version']} vs "
            f"{b['library_version']}). The same cohort id in a different library can be different "
            "data, so the pairing would be fictional."
        )
    if a["contract_version"].split(".")[0] != b["contract_version"].split(".")[0]:
        raise ValueError(
            f"refusing to compare: contract major versions differ ({a['contract_version']} vs "
            f"{b['contract_version']}). A contract change can change what a claim MEANS."
        )
