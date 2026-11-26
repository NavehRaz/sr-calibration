# SR methods preprint

One methods preprint for the saturating-removal (SR) model fitting framework, written so that other papers can cite it
for their methods. The first two papers to cite it are Alibek's mouse housing-temperature study and Yoonji's mosquito
study.

- **Plan:** `~/.claude/plans/plan-the-migration-vast-scroll.md` (approved 2026-11-05).
- **Wiki:** task `T-METHODS-PREPRINT`, page `wiki/pages/projects/methods-preprint.md`.
- **Owner session:** sesync `methods` in group `sr`.

## Rule for Phase 1

The draft is built from **existing code and saved results only**. No fit, simulation or cluster job is started. A
summary computed from stored records (for example Wilson intervals on stored coverage counts) is allowed. Wherever evidence
is missing, the text gets a visible `\MISSING{...}` box, and the box is listed in `sources/MISSING.md` for Naveh to decide.

## Layout

| Path | What |
|---|---|
| `paper/main.tex`, `paper/si.tex`, `paper/sections/*.tex` | manuscript and supplement |
| `paper/refs.bib` | references; every DOI checked before it is added |
| `figures/*.py` → `figures/out/` | figure scripts; they read stored files only |
| `tables/*.py` → `tables/*.tex` | table scripts; they read stored files only |
| `sources/CODE_MAP.md` | every method component's implementation (`repo@commit:path`) and which project used which version |
| `sources/RESULTS_MAP.md` | every saved result the paper uses (path, sha256, producing commit) |
| `claims.toml`, `verify_claims.py` | each number in the paper, its stored source, and a check that recomputes it |
| `sources/MISSING.md` | evidence gaps, what each blocks, its cost, and a recommendation |

## Build

```
cd paper && latexmk -pdf main.tex      # TinyTeX
python verify_claims.py
```
