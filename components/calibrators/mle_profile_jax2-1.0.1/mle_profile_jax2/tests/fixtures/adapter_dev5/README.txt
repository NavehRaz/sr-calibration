1.0.1.dev5's adapter, VERBATIM -- the oracle of tests/test_jax2_freeze.py (test fixture, NOT runtime: VERSION_LOCK
covers only the package's top-level *.py; srbench discovery loads only methods/<dir>/__init__.py).

  adapter_dev5.py         = git show 258146a:methods/mle_profile_jax2/__init__.py
                            sha256 cd454d06e0def405ced19decb22f3b49868dfc3e9dea6eae688c93d886a1544e
  VERSION_LOCK_dev5.json  = git show 258146a:methods/mle_profile_jax2/VERSION_LOCK.json
                            sha256 60041a75afb12a6a0809c6f9ccdb3f36bf477474c57148071c0eb7f5ecc37cf5

258146a = 1.0.1.dev5, the candidate validated by PREREG_REL ADD3 / S2-DT4 / S3 (branch jax2-rel). 1.0.1 (the freeze)
is dev5 plus ONE ClaimSet.diagnostics key, `estimate`, added by the adapter's claimset; the test loads this file under
a test key (it registers nothing there) and asserts the freeze's ClaimSet == this adapter's ClaimSet + that key.
