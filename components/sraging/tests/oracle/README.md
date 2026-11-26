# Oracle snapshots — NEVER EDIT

## `v033/` — the srbench reference calibrator, `mle_profile_jax` v0.3.3

Snapshotted 2026-10-02 from the live Dropbox trees, because the reference ran on an **uncommitted**
SRK working tree (`SRtools_kappa` HEAD `cb191f5` + edits to `SRK/fp_gof.py`, `SRK/fp_likelihood.py`,
`SRK/benchmark/space_search.py`; `SRK/fit_cohort.py` untracked). Nothing else preserves it.

- `MANIFEST.json` — every file with md5, tracked/dirty flags, both source repos' HEAD + status, the
  pinned versions (Python 3.14.6, JAX/jaxlib 0.11.1, NumPy 2.4.3, SciPy 1.17.1), and a verification run.
- The file list is the **recorded import closure** of one real calibration (`sys.modules` after
  `calibrate()` on `pure-k0.01-p00-c0-s0`, 189 s), plus the real-cohort CLI path (`SRK/fit_cohort.py`,
  `SRK/benchmark/space_search.py`, `tools/fit_cohort_jax.py`, the method's `__init__.py`).
- **Verification:** that run's claims (shape, bounds, point) are identical to the stored
  `mle_profile_jax-full` reference record for the same cohort.
- `SRK_working_tree_vs_HEAD.patch` — the uncommitted SRK edits, for reference.
- `pip_freeze.txt` — the full environment.

How to use (spec 001): import the oracle from this directory (prepend `v033/SRtools_kappa` and
`v033/sr_calibration_tester` to `sys.path` in the test process only) and compare the port **side by
side in the same process, same host, same environment**. Never compare against bytes recorded on another
host: the hgn26/hgn37 cluster nodes do not reproduce the reference (FINER_GRID.md §3) and CI runners
are Linux. The oracle may only be replaced by a new versioned directory (`v0xx/`), never edited.
