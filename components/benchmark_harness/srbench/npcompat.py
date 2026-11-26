"""numpy>=2 compatibility shim, applied before any FP module is imported.

numpy 2.0 removed `np.trapz` in favour of `np.trapezoid`, and the upstream Fokker-Planck solver
still calls the old name. The shim goes HERE rather than in the upstream file on purpose: that
file belongs to someone else, it is on a shared and concurrently-edited tree, and patching it
would be an edit that a periodic sync can silently revert -- which has happened to other edits on
that tree.

Importing this module is the whole interface. It is idempotent.
"""
from __future__ import annotations

import numpy as np

if not hasattr(np, "trapz"):  # pragma: no branch - trivially true on numpy>=2
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

APPLIED = hasattr(np, "trapz")
