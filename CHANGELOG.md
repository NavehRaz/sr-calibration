# Changelog

## 0.1.0 (2026-11-26; software record doi:10.5281/zenodo.24556080)

First release, accompanying the methods preprint (Raz, Mayo and Alon, bioRxiv 2026).

- **Code:** sraging `v0.1.1` (commit d368157; its package code, `src/`, is identical to `v0.1.0`, 33213ce. v0.1.1 takes
  the citing studies' unpublished counts out of its documentation and divergence register, and the tests and tools
  find the authors' other checkouts through environment variables instead of built-in paths), with its tests, their tools and the stored acceptance results (left
  out: the data-cleaning acceptance records of its specification 005 and the file pinning the studies' cleaning
  records, which read the studies' unpublished files, and the project-inventory tool, test and record);
  calibrators `mle_profile_jax2` 1.0.1 and 1.0.0 (frozen tags); the benchmark harness (tester e137c85); the preprint's
  source, scripts and number check (`components/paper/`, without the two version-bridge scripts, which read the
  studies' own fits).
- **Tests:** `TESTS.md` lists every test of the preprint with its stored result and how to rerun it;
  `test_results/sraging-v0.1.1/` holds a rerun of sraging's unit and parity tests at the tag.
- **Data record:** library v6, the records of three calibrator runs with their cohort lists, the reports and explorer
  pages, the κ-fixed panel, the evidence files of W's validation, and the preprint's other public inputs
  (`paper_inputs/`), as one zip per entry.
- **Data record DOI:** 10.5281/zenodo.24555552 (Zenodo, version 0.1.0).
- **Known issues:** see `README.md`.
- Recorded results keep the absolute paths and host names of the machine that produced them (provenance). The frozen
  calibrator tags are released byte-identical, so one comment line in a test fixture of each
  (`tests/fixtures/candidate_rc1/code/MD5SUMS.txt`) names the author's local folder.
- Licences: code BSD 3-Clause, data CC BY 4.0. The studies' group-comparison code is released with each study's paper.
- Planned for later versions: the group comparisons inside sraging after their own validation, the cage calibration
  of the naming rule, a graphical interface and user documentation.
