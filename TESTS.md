# Tests in sr-calibration 0.1.0

Every test reported in the preprint, what it checks, where its stored result is and how to rerun it.
`code:` is a path in this repository; `data:` is a path in the Zenodo data record, unzipped into `data/`.

## 1. sraging's test suite, rerun at the release

The unit tests (`components/sraging/tests/unit`) check each part of the engine, the calibrators, the cohort generator,
the leverage check and the command line. The parity tests (`tests/parity`) compare the ported calibrators with the
frozen reference code (`tests/oracle`) bit for bit.

**Rerun for this release** (2026-11-25; sraging `v0.1.1` = d368157, whose package code is that of `v0.1.0`; Apple M5
Pro laptop, macOS 26.7.3, Python 3.14.6; logs and JUnit XML in `test_results/sraging-v0.1.1/`). The run used a clone of
sraging at the tag with a local `main` branch (some tests compare the tag with it) and with `SRAGING_CODE5` set (below);
the project-inventory test, which is not released, was left out.

| Tests | |
|---|---|
| collected (23 marked `slow` deselected, as by default) | 2,493 |
| passed | 1,958 |
| skipped | 532 |
| failed | 3 |

- **529 skips: not applicable on this host.** Byte identity with the frozen reference code is claimed only on the host
  where its snapshots were taken (macOS 26.5.2). This laptop has since been updated, so the parity comparisons report
  NOT APPLICABLE instead of passing; they were not repeated. The last stored run on the snapshot host passed 949 unit
  and 497 parity tests (29 October 2026; preprint §9; `components/sraging/tests/acceptance/results/unchanged_003g.json`).
  The other 3 skips need a pinned clone of the development tester.
- **3 failures: the same host check**, raised inside the test instead of as a skip.

To rerun them:

```bash
cd components/sraging
uv sync --frozen --group dev --group oracle --extra langevin --extra plots
export SRAGING_CODE5=/path/to/code_5    # optional: the folder holding the authors' other checkouts (below)
.venv/bin/python -m pytest tests/unit tests/parity
```

Some tests read the benchmark's data or the authors' other checkouts (`sr_calibration_tester`,
`sr_calibration_tester_jax2` and its freeze, `SRtools_kappa`). They find them through `SRAGING_CODE5`, the folder that
holds those checkouts (sraging's README lists the variables that override a single checkout). When it is unset, those
tests skip as not applicable.

If pytest stops at collection, add `--continue-on-collection-errors`: one test file reads sraging's internal
specifications when it loads.

**Run from this repository** (same laptop, same command plus `--continue-on-collection-errors`; logs in
`test_results/sraging-v0.1.1/from_release/`): `components/sraging` is the tag without sraging's git history and
without its internal specifications and planning notes. Of 2,491 tests, 1,930 passed and 532 were skipped (as above);
29 did not pass:

| Tests | Why |
|---|---|
| 12 | they compare files with other commits of sraging's git history, which the release does not carry |
| 12 | they read sraging's internal specifications or planning notes, which are not released |
| 2 | they read the cleaning-record files left out of the release (below) |
| 3 | the host check (above) |

The acceptance tests (`tests/acceptance`) are not rerun here. They check the development history (ancestor commits,
staleness against the main checkout, the tester's pinned store) rather than recompute anything; their stored results
(`code: components/sraging/tests/acceptance/results/`) are the measurements that the preprint reports, listed in
section 2. Not in this release: the acceptance records of sraging's data-cleaning specification (005) and the file
that pins the studies' cleaning records (they read the citing studies' unpublished files), and the project-inventory
tool, test and record (they list the authors' project folders).

## 2. The tests reported in the preprint

One row per paragraph of the preprint's validation (§7), worked example (§8) and software (§9) sections: how many
numbers it prints, how many the release can recheck, and the stored files they are read from, with the count of
numbers from each. `verify_claims.py` rechecks them all (README, "Recheck the preprint"); `components/paper/claims.toml`
gives, for each number, its source, the locator that reads it and the sha256 the source had when the number was
recorded.

| § | Test | Numbers | Recheckable from the release | Stored results (code = repository, data = data record) |
|---|---|---|---|---|
| 7a | Numerical accuracy | 14 | 14 | code: components/sraging/tests/acceptance/results/mpref_002.json (5)<br>code: components/sraging/tests/acceptance/results/t1_002.json (3)<br>code: components/sraging/tests/acceptance/results/early_002.json (3)<br>code: components/sraging/tests/acceptance/results/score_002.json (2)<br>code: components/sraging/tests/acceptance/results/anchor_002.json (1) |
| 7a | Independent implementation | 16 | 16 | code: components/sraging/tests/acceptance/results/t1_002.json (16) |
| 7a | High-precision reference and the absorbed-flux form | 17 | 17 | code: components/sraging/tests/acceptance/results/mpref_002.json (15)<br>code: components/sraging/tests/acceptance/results/t1_002.json (1)<br>code: components/sraging/tests/acceptance/results/score_002.json (1) |
| 7a | Analytic anchor | 3 | 3 | code: components/sraging/tests/acceptance/results/anchor_002.json (3) |
| 7a | Score tests against simulated cohorts | 14 | 14 | code: components/sraging/tests/acceptance/results/score_002.json (9)<br>code: components/sraging/tests/acceptance/results/score_window_002.json (3)<br>code: components/sraging/tests/acceptance/results/score_het_002.json (2) |
| 7a | Early deaths | 6 | 6 | code: components/sraging/tests/acceptance/results/early_002.json (5)<br>code: components/sraging/tests/acceptance/results/mpref_002.json (1) |
| 7a | Previous solver: Fokker–Planck against Langevin simulation (historical) | 12 | 12 | data: paper_inputs/code_5/sr_fp_fit_share/validate_01_simulation_shapes.ipynb (5)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/I2/I2_REPORT.txt (4)<br>code: components/paper/tables/convergence_summary.csv (2)<br>code: components/calibrators/mle_profile_jax2-1.0.1/mle_profile_jax2/estimator.py (1) |
| 7a | Convergence with grid size and time step | 68 | 68 | code: components/paper/tables/convergence_summary.csv (48)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/I2/I2_REPORT.txt (17)<br>code: components/paper/tables/convergence_cohorts.csv (2)<br>code: components/calibrators/mle_profile_jax2-1.0.1/mle_profile_jax2/estimator.py (1) |
| 7a | Current engine: Fokker–Planck against Langevin simulation | 39 | 39 | code: components/paper/tables/langevin_ks.csv (14)<br>data: paper_inputs/git/6f359bf/tests/acceptance/results/p1_pooled_004.json (7)<br>data: paper_inputs/git/a2ead86/tests/acceptance/results/s4_004.json (5)<br>data: paper_inputs/git/fe6a416/tests/acceptance/results/multi_004.json (5)<br>data: paper_inputs/git/78f8447/tests/acceptance/results/s5_hom_004.json (5)<br>data: paper_inputs/git/8a9b933/tests/acceptance/results/paths_004.json (2)<br>data: paper_inputs/git/9d9eae1/tests/acceptance/results/multi_004.json (1) |
| 7b | Design | 26 | 26 | data: benchmark/library/v6/manifest.json (18)<br>code: components/paper/tables/library_design.csv (4)<br>data: paper_inputs/code_5/sr_calibration_tester/reports/validity/validity_v6.csv (3)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (1) |
| 7b | Scoring | 4 | 4 | data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (3)<br>data: benchmark/reports/jax2h-v6-full-bign/comparison.json (1) |
| 7b | Coverage on well-specified data | 25 | 25 | code: components/paper/tables/coverage_by_arm.csv (13)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out_reported/REPORT_H_V6.txt (10)<br>code: components/paper/tables/library_design.csv (1)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (1) |
| 7b | Misspecification | 23 | 23 | code: components/paper/tables/coverage_by_arm.csv (14)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out_reported/REPORT_H_V6.txt (9) |
| 7b | Windowed fits (W) | 99 | 99 | data: sraging_evidence/specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json (43)<br>code: components/sraging/tests/acceptance/results/a2_006.json (20)<br>code: components/sraging/tests/acceptance/results/a1_006.json (19)<br>code: components/sraging/tests/acceptance/results/realfull_006.json (11)<br>code: components/sraging/tests/acceptance/rules_006.toml (3)<br>code: components/sraging/src/sraging/calibrators/profile_jax2.py (1)<br>code: components/calibrators/mle_profile_jax2-1.0.1/mle_profile_jax2/estimator.py (1)<br>code: components/sraging/tests/acceptance/results/gate_006/realfull/realfull-real-combined_human_F-c0-s0.json (1) |
| 7b | Sharp cohorts and the n-ladder | 13 | 13 | data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out_reported/REPORT_SHARP_NLADDER_V6.txt (10)<br>data: benchmark/library/v6/manifest.json (3) |
| 7b | One-sided bounds | 12 | 12 | data: benchmark/reports/jax2h-v6-full-bign/comparison.json (10)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (1)<br>data: benchmark/reports/jax2-v6-full-bign/explorer_data.json (1) |
| 7b | Cost | 6 | 6 | data: benchmark/reports/jax2h-v6-full-bign/comparison.json (4)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (2) |
| 7c | The likelihood along the ridge | 18 | 18 | data: paper_inputs/code_5/sr_calibration_tester_kfix/reports/kfix/invariance.md (5)<br>code: components/paper/tables/d2b_ridge_cohort.csv (4)<br>code: components/paper/tables/d2b_ridge.csv (3)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2_rel/reports/rel101/G3_KFIX/BOTH/KAPPA0_TABLE.txt (3)<br>data: paper_inputs/code_5/sr_calibration_tester_kfix/reports/kfix/REPORT.md (2)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (1) |
| 7c | What pinning costs | 21 | 21 | data: paper_inputs/code_5/sr_calibration_tester_jax2_rel/reports/rel101/G3_KFIX/BOTH/KAPPA0_TABLE.txt (14)<br>data: paper_inputs/code_5/sr_calibration_tester_kfix/reports/kfix/REPORT.md (5)<br>code: components/paper/tables/d2b_kfix_panel.csv (1)<br>code: components/paper/tables/d2b_ridge.csv (1) |
| 7c | How often κ is declined | 9 | 9 | data: benchmark/reports/jax2-v6-full-bign/explorer_data.json (4)<br>data: benchmark/reports/jax2h-v6-full-bign/comparison.json (3)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (1)<br>data: benchmark/reports/jax2h-v6-full-bign/explorer_data.json (1) |
| 7d | The effect likelihood | 6 | 3 | not public: a fly study's files; released with that study (3)<br>data: paper_inputs/code_5/itp_analysis/recal/results/gates/V1_j100.json (1)<br>data: paper_inputs/code_5/itp_analysis/recal/results/gates/V2.json (1)<br>data: paper_inputs/code_5/itp_analysis/recal/results/gates/V3.json (1) |
| 7d | Recovery with independent individuals | 19 | 19 | data: paper_inputs/code_5/itp_analysis/recal/results/2026-10-18_J100/pool_pairs/pool_groups_singles.csv (9)<br>data: paper_inputs/code_5/itp_analysis/recal/results/2026-10-18_J100/recovery_verdict/identification.csv (6)<br>data: paper_inputs/code_5/itp_analysis/recal/results/2026-10-18_J100/recovery_verdict/verdict.json (4) |
| 7d | Pooling arms to identify pairs | 12 | 12 | data: paper_inputs/code_5/itp_analysis/recal/results/2026-10-18_J100/pool_pairs/pool_sim_identification.csv (9)<br>data: paper_inputs/code_5/itp_analysis/recal/results/2026-10-18_J100/pool_pairs/pool_groups_pairs.csv (3) |
| 7d | The analytic shortcut | 6 | 5 | data: paper_inputs/interventions_data/calib/ident/ia_validation/summary.json (5)<br>not public: the interventions project's files; not in this version (1) |
| 7d | Clustered individuals | 9 | 0 | not public: a fly study's files; released with that study (9) |
| 7e | Nulls | 17 | 17 | code: components/sraging/tests/acceptance/results/ac11_003b.json (13)<br>code: components/sraging/tests/acceptance/results/panel_003f.json (2)<br>code: components/sraging/tests/acceptance/results/p3_003f.json (2) |
| 7e | Planted individuals | 7 | 7 | code: components/sraging/tests/acceptance/results/ac11_003b.json (7) |
| 7e | Overhead | 2 | 2 | code: components/sraging/tests/acceptance/results/panel_003f.json (2) |
| 7e | Limits | 2 | 2 | code: components/sraging/tests/acceptance/results/lev_small_n_003g.json (1)<br>code: components/sraging/tests/acceptance/results/ac11_003b.json (1) |
| 8 | Worked example | 22 | 22 | code: components/sraging/examples/results/itp_c2016_het3_control_female_jax2.json (17)<br>code: components/paper/tables/example_refit_v010.json (2)<br>code: components/sraging/examples/data/SOURCE.md (1)<br>code: components/paper/tables/example_refit_check.csv (1)<br>code: components/sraging/examples/data/itp_c2016_het3_control_female.csv (1) |
| 8 | Reading the output | 18 | 18 | code: components/paper/tables/example_refit_v010.json (7)<br>code: components/sraging/examples/results/itp_c2016_het3_control_female_jax2.json (7)<br>code: components/paper/tables/example_kappa_ends_summary.csv (4) |
| 8 | Leverage | 6 | 6 | code: components/sraging/examples/results/itp_c2016_het3_control_female_jax2.json (6) |
| 8 | Cost | 5 | 5 | code: components/sraging/examples/results/itp_c2016_het3_control_female_jax2.json (4)<br>code: components/paper/tables/example_refit_check.csv (1) |
| 9 | Package | 4 | 4 | code: components/sraging/pyproject.toml (4) |
| 9 | Frozen calibrator versions | 11 | 11 | code: components/sraging/tests/acceptance/results/t0_002.json (4)<br>code: components/sraging/divergences/register.toml (4)<br>code: components/sraging/tests/acceptance/results/parity_003b.json (2)<br>code: components/sraging/tests/acceptance/results/determinism_003d.json (1) |
| 9 | Tests | 11 | 11 | code: test_results/sraging-v0.1.1/junit.xml (5)<br>code: components/sraging/tests/acceptance/results/unchanged_003g.json (4)<br>code: test_results/sraging-v0.1.1/host.txt (1)<br>code: test_results/sraging-v0.1.1/pytest.txt (1) |
| 9 | Host dependence | 9 | 9 | data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/PLATFORM/PLATFORM_REPORT.txt (9) |
| 9 | Runtime | 5 | 5 | data: benchmark/reports/jax2h-v6-full-bign/comparison.json (4)<br>data: paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt (1) |

## 3. Numbers whose sources are not public in 0.1.0

15 of the 768 numbers the preprint prints are read from files that are not part of this release. They were checked on
the authors' machine; `claims.toml` records each source's path and sha256, and `verify_claims.py` lists them with the
reason:

| Numbers | Source | Where in the preprint |
|---|---|---|
| 12 | the fly study's clustered-design simulation (released with that study) | §6, §7d |
| 2 | the hand-off notes of the calibrator freeze (the recommended pinned κ0, and H's offset on sharp hazard cohorts) | §4, §10, SI S1 |
| 1 | the interventions project's decision notes (the number of replicates of its check of the analytic shortcut) | §7d |

## 4. Rerunning the benchmark

`components/benchmark_harness/` regenerates library v6 from its manifest, runs a calibrator on it and scores the
records against the truths; its README gives the commands. The stored records of the preprint's runs are in
`data: benchmark/records/`, and the explorer pages (`data: benchmark/reports/*/explorer.html`) list every cohort with
its truth, estimate, interval and hit or miss. The cost of a full run is in the preprint, §9 ("Runtime").
