"""The view registry: adding a view edits no existing view.

Each view is a self-contained unit that DECLARES the payload fields it reads. That declaration is
checked at ASSEMBLY time, so a view naming a field the payload lacks fails while the file is being
built -- not silently, in someone's browser, weeks later, on a page that looks fine until you read
the numbers.

Views are additive by construction: `register()` appends, nothing edits a shared list of cases, and
two people adding views on the same day do not conflict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

#: Rendering is JavaScript embedded in the assembled page. A view supplies the function body; the
#: assembler wires it to the payload. Keeping it as source text rather than a Python callback is
#: what lets the finished file open from disk with no network and no server, forever.


@dataclass(frozen=True, slots=True)
class View:
    view_id: str
    title: str
    requires: tuple[str, ...]
    render_js: str
    description: str = ""
    per_arm: bool = True

    def to_dict(self) -> dict:
        return {"view_id": self.view_id, "title": self.title, "requires": list(self.requires),
                "description": self.description, "per_arm": self.per_arm}


_VIEWS: list[View] = []


def register(view: View) -> View:
    if any(v.view_id == view.view_id for v in _VIEWS):
        raise ValueError(f"duplicate view id {view.view_id!r}")
    _VIEWS.append(view)
    return view


def all_views() -> list[View]:
    from . import views as _v  # noqa: F401  - registration by import
    return list(_VIEWS)


def _has_path(payload: dict, path: str) -> bool:
    """`cohort.cv` -> payload['cohort']['cv']; `denominators` -> payload['denominators']."""
    cur = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def check_requirements(payload: dict, views=None) -> None:
    """Fail at ASSEMBLY time if any view names a field the payload does not carry."""
    problems = []
    for v in (views if views is not None else all_views()):
        missing = [r for r in v.requires if not _has_path(payload, r)]
        if missing:
            problems.append(f"  {v.view_id}: requires {missing}")
    if problems:
        raise ValueError(
            "views require payload fields that are not present:\n" + "\n".join(problems) +
            "\nThis fails here, while the file is being built, rather than silently in a browser."
        )
