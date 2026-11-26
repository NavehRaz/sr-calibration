"""Check a built library against sraging's shipped per-cohort digests.

    python tools/verify_v6_digests.py <library_dir> <determinism_003d.json>

THE TRAP THIS EXISTS TO AVOID. sraging's digests are of the UNTRUNCATED cohorts (time_range=None).
Ours are untruncated too -- the window is a harness transform applied at observation time -- so the
two compare directly, but ONLY through the stored arrays. Digesting `Library.observation()` instead
compares a windowed cohort against an unwindowed digest and reports every one of the 392 windowed
cohorts as a mismatch, which looks exactly like a broken generator.

So this reads the stored `.npz` arrays and never calls `observation()`.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srbench.library.build import cohort_digest  # noqa: E402


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    lib_dir, digest_path = argv

    with open(os.path.join(lib_dir, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    with open(digest_path, encoding="utf-8") as fh:
        theirs = json.load(fh)["digests"]

    specs = man["specs"]
    print(f"library   {lib_dir}")
    print(f"  version {man.get('library_version')}   "
          f"cohort_generator {man.get('cohort_generator')}")
    print(f"  cohorts {len(specs)}   their digests {len(theirs)}")

    ours = {s["cohort_id"] for s in specs}
    missing_here = sorted(set(theirs) - ours)
    extra_here = sorted(ours - set(theirs))
    if missing_here:
        print(f"\n  in their digests but NOT built here: {len(missing_here)}")
        for c in missing_here[:10]:
            print(f"      {c}")
    if extra_here:
        print(f"\n  built here but NOT in their digests: {len(extra_here)}")
        for c in extra_here[:10]:
            print(f"      {c}")

    match, mismatch, absent = [], [], []
    for s in specs:
        cid = s["cohort_id"]
        want = theirs.get(cid)
        if want is None:
            continue
        path = os.path.join(lib_dir, "cohorts", f"{cid}.npz")
        if not os.path.exists(path):
            absent.append(cid)
            continue
        d = np.load(path)
        (match if cohort_digest(d["ages"], d["events"]) == want else mismatch).append(s)

    print(f"\n  MATCH    {len(match)}")
    print(f"  MISMATCH {len(mismatch)}")
    print(f"  no array {len(absent)}")

    if mismatch:
        print("\n  mismatches by arm :", dict(Counter(s.get("arm") for s in mismatch)))
        print("  mismatches by group:", dict(Counter(s.get("group") for s in mismatch)))
        print("  first few:")
        for s in mismatch[:10]:
            print(f"      {s['cohort_id']}  n={s.get('n')} delta={s.get('delta')}")

    ok = not mismatch and not absent and not missing_here and not extra_here
    print("\n  VERDICT:", "all built cohorts reproduce sraging's digests"
          if ok else "DISCREPANCIES ABOVE -- do not adopt this library until they are explained")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
