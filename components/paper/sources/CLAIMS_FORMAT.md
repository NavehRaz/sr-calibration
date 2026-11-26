# Claim format (claims.toml)

Every number printed in the paper is one `[[claim]]`. `verify_claims.py` re-reads the stored source, extracts the value
with the locator, and compares it with `value`. In LaTeX a claim is printed with `\C{<id>}`. Macros are generated into
`paper/claims.tex`.

```toml
[[claim]]
id      = "v6-pure-eta-cov"          # kebab-case, unique
label   = "J 1.0.1 closed eta coverage, v6 pure arm"
value   = 0.958                      # as printed in the paper (already rounded)
display = "0.958"                    # optional exact LaTeX text; default = value
units   = ""                         # "nats", "CPU-h", "%", ...
rtol    = 0.005                      # relative tolerance allowed by rounding (default 0.005)
source  = "aging:code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out_reported/REPORT_H_V6.txt"
sha256  = "<sha256 of source when recorded>"
commit  = "<repo@commit that produced the source, if known>"
locator = { kind = "regex", pattern = "pure\\s+eta\\s+([0-9.]+)", group = 1 }
# other locator kinds:
#   { kind = "json", path = "a.b[3].c" }
#   { kind = "csv", column = "coverage", where = { arm = "pure", param = "eta" } }
#   { kind = "ratio", num = <locator>, den = <locator> }     # e.g. 617/644
#   { kind = "count", ... }  (count rows matching `where` in a csv/jsonl)
#   { kind = "python", expr = "..." }  only when nothing else works; `src` = parsed source
note    = "free text: how the number is defined"
```

Path prefixes: `aging:` = the author's aging folder (the folder two levels above this repository), `~/` = the home
folder. In the release both resolve through `paper_paths.py`. Absolute paths are also allowed on the author's machine.

Rules:
- A claim's source must be a stored file that existed before Phase 1 (2026-11-05), or a table that a script in
  `tables/` computes from such files. In the second case `source` is the table and `derived_from` lists its inputs.
- Never copy a number from the wiki or from a summary doc when the underlying result file exists. Point at the result file.
- If only a prose report holds the number (for example `REPORT_H_V6.txt`), that report is the source.
