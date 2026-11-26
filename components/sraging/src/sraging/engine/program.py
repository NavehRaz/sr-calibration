"""The canonical text of a lowered JAX program (research R1; spec SC-001, FR-021).

"A feature that is off adds no operation" is tested by comparing programs, not timings: with every
feature off, the engine's jitted functions must lower to the same program as spec 001's port of v0.3.3
(`sraging.compat.v033.fp_jax`). Timing cannot test it -- the true ratio is 1.0, so an "upper CI <= 1.0"
test fails by construction (sr-manager seed review, change 8).

`Lowered.as_text()` in JAX 0.11.1 is StableHLO without source locations (`debug_info=False` is the
default; measured: no `loc(` entries). What still differs between two equivalent programs is symbol
names: the module is named after the Python function (`@jit_ll_of_z`) and private functions get
generated names (`@_where_12`). `canonical_text` renames every `@symbol` to `@s0, @s1, ...` in order of
first appearance, so two programs compare equal exactly when they have the same operations in the same
order with the same constants.
"""
from __future__ import annotations

import re

_SYMBOL = re.compile(r"@[A-Za-z_][\w.$-]*")


def canonical_text(lowered) -> str:
    """`lowered` is a `jax.stages.Lowered` (or its text). Returns the text with symbols renamed."""
    text = lowered if isinstance(lowered, str) else lowered.as_text()
    names: dict[str, str] = {}

    def rename(m):
        s = m.group(0)
        if s not in names:
            names[s] = f"@s{len(names)}"
        return names[s]

    return _SYMBOL.sub(rename, text)


def same_program(a, b) -> bool:
    """True when two lowered programs are the same up to symbol names."""
    return canonical_text(a) == canonical_text(b)
