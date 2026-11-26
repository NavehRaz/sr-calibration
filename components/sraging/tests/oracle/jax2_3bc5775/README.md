# Oracle snapshot `jax2_3bc5775/` — NEVER EDIT

The tester's successor candidate `mle_profile_jax2` and `mle_profile_jax2_h` 1.0.0, frozen at
`sr_calibration_tester_jax2` commit `3bc57751a6617e5bbdc76205e58d158c9d434990` (spec 003; permitted by Naveh, 2026-10-05).

- `mle_profile_jax2/` holds byte copies of the four method files and `VERSION_LOCK.json`.
- `MANIFEST.json` gives each file's md5 and sha256, the source commit and HEAD, and the host. It also cites the
  18 SRK and 13 srbench files the fit loads, which are hash-equal to `tests/oracle/v033` and are used
  from there, not copied.

Load it with `tools/oracle_jax2.py` (v033's srbench and SRK on `sys.path`, this package by path under
`srbench_methods.mle_profile_jax2`) and compare the port side by side in the same process, on the same
host, in the pinned environment. It may only be replaced by a new versioned directory, never edited.
