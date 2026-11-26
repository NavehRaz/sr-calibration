# Oracle snapshot `jax2_6884fc8/` — NEVER EDIT

`mle_profile_jax2` and its sibling names at tag `mle_profile_jax2-1.0.1`, commit `6884fc8b4d34932d2e5b6d65720dd0685f5915cd`, read from git objects (spec 003b; the
snapshot permitted by Naveh, 2026-10-19).

- `mle_profile_jax2/` holds the lock and the runtime files, byte for byte as the commit holds them.
- `MANIFEST.json` gives each file's md5 and sha256, the method subtree's listing (sha256 of every file, tests and
  fixtures included) and its digest, the 18 SRK files the fit loads (hash-equal to `tests/oracle/v033`, used from
  there), and the srbench tree, used from `tests/oracle/tester_f94ec2b`.

Load it with `tools/oracle_jax2_101.py` and compare the port side by side, in the same process, on the same host, in
the pinned environment. It may only be replaced by a new versioned directory, never edited.
