# sr-calibration

Code, frozen calibrator versions, tests and benchmark data for the methods preprint

> Raz N, Mayo A, Alon U. *Calibrating the saturating-removal model of aging from survival data: exact likelihood,
> identifiability and intervention effects.* bioRxiv (2026). DOI: TBD

Release **0.1.0**. Software record (this repository on Zenodo): DOI TBD. Data record: DOI TBD.
Code: BSD 3-Clause (`LICENSE`). Data record: CC BY 4.0.

## What is here

| Where | What |
|---|---|
| `components/sraging/` | The `sraging` Python package at tag `v0.1.1` (its package code, `src/`, is that of `v0.1.0`): the exact Fokker–Planck likelihood engine, the default calibrator `profile_jax2` (J, version 1.0.1), H (external hazard, fit window) and W (`profile_jax2_w`, windowed fits), the leverage check, the cohort generator and the command line. With its unit, parity and acceptance tests and their stored results (`tests/`). |
| `components/calibrators/` | The calibrator packages exactly as the citing studies ran them: `mle_profile_jax2` 1.0.1 (all names, including H's fit window and the κ-fixed names) and 1.0.0. |
| `components/benchmark_harness/` | Generates the benchmark library, runs a calibrator on it and scores the records. |
| `components/paper/` | The preprint's LaTeX source, the scripts that make every figure, table and number from the stored results, and `verify_claims.py`, which re-reads each printed number from its source. |
| `TESTS.md` | Every test reported in the preprint: what it checks, where its stored result is, how to rerun it. |
| `test_results/` | The rerun of sraging's unit and parity tests at the released tag. |
| data record (Zenodo) | Benchmark library v6 (2,408 cohorts), the per-cohort calibration records, reports and explorer pages, the κ-fixed panel, and the other stored results the preprint reads (`README-DATA.md` there). |

The group-comparison code that each citing study ran is released with that study's paper.

## Fit a survival curve

```bash
cd components/sraging
uv sync --frozen                    # Python 3.14 and the exact pins of uv.lock
.venv/bin/sraging fit-cohort --csv my_cohort.csv --delta 1 --time-unit days --json my_cohort.fit.json
```

`components/sraging/README.md` explains the input format, the choice of calibrator (J, H or W), and how to read the
output; `examples/` has the worked example of the preprint (Section 8).

## Recheck the preprint

1. Download the zips of the data record and unzip them all into a folder `data/` at the top of this repository (or
   anywhere, and set `SR_DATA` to that folder).
2. Build the environment once: `cd components/sraging && uv sync --frozen --extra plots`.
3. Recheck every number: `cd components/paper && ../sraging/.venv/bin/python -B verify_claims.py`.
   It re-extracts each number from its stored source and compares it with the paper. Numbers whose source is not
   public in this version are listed with the reason (TESTS.md).
4. Remake a table or figure: `../sraging/.venv/bin/python -B tables/<script>.py` or `cd figures &&
   ../../sraging/.venv/bin/python -B <script>.py`. Each script names its inputs; `paper_paths.py` finds them in this
   repository or in `data/`.

## Known issues in 0.1.0

- `profile_v033` (the reference calibrator, not the default) never labels a κ interval end as box-limited: its walk
  stops at 1.01× the lower and 0.99× the upper box edge, and the label needs the end within a relative 1e-9 of the edge,
  so a κ end at the box is printed as an ordinary finite end. This is v0.3.3's own behaviour, ported unchanged. The
  default, `profile_jax2`, labels its box-limited ends and is not affected.
- `sraging fit-cohort --help` omits `profile_jax2_w` from the `--calibrator` list (the `--window` help names it);
  `--calibrator profile_jax2_w` works.
- On large human life tables, `profile_v033` can end in a wrong basin. sraging's engine (numerics `v1`) and v0.3.3's
  ported likelihood (numerics `v033`) agree on synthetic cohorts (at most about 6e-8 nats apart on the 38 non-human
  cohorts of sraging's canonical panel) but not on large human life tables. Fitted with the same walker, v0.3.3's
  numerics left the maximum-likelihood estimate 2,268–6,722 nats below a point the fit's own walk had evaluated, on 7 of
  10 human life tables; with `v1`'s numerics it was within 1.03 nats (measured after this release, sraging spec 006b;
  divergence D-004). `profile_v033` keeps v0.3.3's numerics by design. The default, `profile_jax2`, runs the frozen
  calibrator's own numerics, not v0.3.3's.

## Versions

`VERSIONS.toml` pins every component to a commit or tag and lists the data entries; `MANIFEST.lock.json` records the
resolved commit, file count and archive checksum of each component; the data record's `DATA_MANIFEST.json` gives the
sha256 of every file. Each release is a Zenodo version with its own DOI; the concept DOI resolves to the newest. Cite
the version you used. `CHANGELOG.md` lists the changes. `assemble.py` rebuilds a release from the source repositories
(on the authors' machine; it only reads them).

## How to cite

Cite the preprint for the methods and the version DOI of this record (and of the data record) for the code and data
you used; see `CITATION.cff`.
