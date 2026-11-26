"""The acceptance harness survives a worker getting several cohorts (2026-10-03: both full claims runs broke).

A worker's second quick-tier cohort registered a second srbench method object under the same name; srbench
refuses that with a ContractError, whose class does not unpickle in the parent, so the pool reported every
pending cohort as crashed. Fixed at the source (one registration per worker) and in the pool (an exception is
that item's recorded error, never a broken pool).
"""
import types

import acceptance as A
import acceptance_claims as C


class _NeedsTwoArgs(Exception):
    """Like srbench's ContractError in the parent: pickles in the worker, fails to unpickle."""

    def __init__(self, what, why):
        super().__init__(f"{what}: {why}")


def _raise_or_echo(item):          # top level: the spawned worker imports it by name
    if item == "bad":
        raise _NeedsTwoArgs("bad", "unpicklable in the parent")
    return {"cohort_id": item, "status": "ok", "checks": {}}


def test_an_exception_is_that_items_error_not_a_broken_pool():
    rows = {r["cohort_id"]: r for r in A.run_pool(_raise_or_echo, ["a", "bad", "b"], 1)}
    assert rows["a"]["status"] == rows["b"]["status"] == "ok"
    assert rows["bad"]["status"] == "error:_NeedsTwoArgs"
    assert "unpicklable in the parent" in rows["bad"]["error"]
    assert "Traceback" in rows["bad"]["traceback"]


def _fake_srbench(calls):
    registry = {}

    def register(m):                # srbench's rule: one object per method name
        name = m.descriptor.name
        if name in registry and registry[name] is not m:
            raise RuntimeError(f"two different methods both call themselves {name!r}")
        registry[name] = m
        calls.append(name)

    class MD:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    return {"contract.method": types.SimpleNamespace(register=register),
            "contract.types": types.SimpleNamespace(MethodDescriptor=MD, CostDeclaration=MD, Claim=MD,
                                                    ClaimSet=MD)}


def test_a_worker_registers_its_srbench_method_once(monkeypatch):
    from sraging.adapters import srbench as AD
    calls = []
    monkeypatch.setattr(AD, "connect", lambda *a, **k: _fake_srbench(calls))
    monkeypatch.setattr(C, "_SRB", {})
    first = C.srbench_method("port-1", "cohort-1")
    second = C.srbench_method("port-2", "cohort-2")
    assert calls == ["sraging_profile_v033"]
    assert second["method"] is first["method"]
    assert second["pre"] == {"cohort-1": "port-1", "cohort-2": "port-2"}
