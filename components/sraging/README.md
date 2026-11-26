# sraging

The SR (saturating-removal) ageing model as one library: fit survival data with the Fokker–Planck
likelihood (JAX). `sraging` replaces SRtools and SRK. Spec 001 delivers the **reference fit**: one cohort,
fitted with calibrator `profile_v033`, a verbatim port of `mle_profile_jax` v0.3.3 (the lab benchmark's
reference), whose results are bit-identical to it on the host that ran the parity tests. Spec 003 adds the
benchmark's successor, `profile_jax2` and `profile_jax2_h`, with combination claims; spec 003b ports its version
1.0.1, which these names now mean. Since spec 003g, **`profile_jax2` 1.0.1 is the default**; `profile_v033` is selected
by name ([Choosing a calibrator](#choosing-a-calibrator), with the default's stated limits). Spec 006 adds
`profile_jax2_w`, which fits a time range ([Fit a time range](#fit-a-time-range)).

What it does **not** do yet: external hazard (other than `profile_jax2_h`'s constant one), heterogeneous
fits (spec 004 added heterogeneous evaluation and the Langevin sampler), fitting xc ≠ 1, interventions, MCMC
([spec 001, "Not Covered"](specs/001-reference-fit/spec.md)); life and census tables and census-to-individual
conversion and the QA screen ([spec 005, "Not Covered"](specs/005-data-plots-cli/spec.md));
fitting per-individual entry ages ([spec 006, "Not Covered"](specs/006-calibrator-capabilities/spec.md)) and a
validated leverage check for `profile_v033` (it runs since spec 006 but failed its validation; see the
[worked example](#worked-example)). Known defects of
v0.3.3 are kept on purpose and listed in the [divergence register](divergences/register.toml).

## Install (from a fresh clone)

You need macOS or Linux, [uv](https://docs.astral.sh/uv/) and Python 3.14.

The repository is private: you need read access to `NavehRaz/sraging` on GitHub (SSH key, or use
`https://github.com/NavehRaz/sraging.git` instead).

```bash
git clone git@github.com:NavehRaz/sraging.git && cd sraging
uv venv --python 3.14 .venv
uv sync                      # exact pins from uv.lock (numpy 2.4.3, scipy 1.17.1, jax/jaxlib 0.11.1)
.venv/bin/sraging --help
```

In Cursor or VS Code, choose the interpreter `.venv/bin/python` (*Python: Select Interpreter*).

## Fit a cohort

Your data: a CSV with a header and two columns, the age at death or censoring and an event flag
(1 = died, 0 = right-censored), plus the **recording resolution** δ in the same units as the ages
(1 for daily records of ages in days, 7 for weekly, …).

```bash
.venv/bin/sraging fit-cohort --csv my_cohort.csv --delta 1 --time-unit days --json my_cohort.fit.json
```

δ is required: it changes how every death is scored. With δ, a **death** at age *t* is scored as
happening somewhere in **[t, t + δ)**, so *t* is the last time the individual was known to be alive. A
**censored** age *c* means "known alive at *c*".

- Life-table ages "at last birthday" are already the start of their interval: use them as they are,
  except age 0 (below).
- Census records that give the day an individual was *found dead* (a check every δ) are one interval late
  for deaths: **subtract δ from every death age**. For censored individuals use the last age at which
  they were known to be alive: the final check for those alive at the end of the study (unchanged), the
  previous check for those that went missing.
- If you pass found-dead ages unshifted, the fit is accepted without any warning and every death is
  scored one interval late; no check can detect it.
- **Deaths found at the first check** become age 0: with δ, a death at age 0 is scored over [0, δ),
  somewhere between the start and the first check (register D-105; v0.3.3's own reader drops such rows).
  Age 0 is refused for censored rows and with `--continuous`. Keeping such deaths is your decision, and
  this is what the fit does with them: the SR model has no early-life mortality, so if deaths that early
  are rare compared with the bulk (a mouse dying on its first day, an infant death in a yearly human life
  table), each is scored at the likelihood's floor of −700 nats (register D-004) and can decide the fit.
  `AGE0_DEATHS` flags a few such deaths, and `FLOORED_DEATHS` flags deaths the fit scored by its grid or
  arithmetic. sraging never drops a row unless you ask: to leave out young ages whose deaths are dominated by
  causes the model does not include, fit a time range that starts after them (`--window LO inf` with
  `profile_jax2_w`, [Fit a time range](#fit-a-time-range)). Choose the cut-off age from your biology, state it
  with the result, and fit both with and without the window to see whether those ages change the claims (the
  worked example below shows a case where they do).
- If your ages are exact times, say so with `--continuous` instead.

The fit refuses — and lists every offending row — rather than silently drop data: non-numeric cells, NaN,
negative ages, age 0 other than a death with δ, or event values other than 0/1.

The same from Python:

```python
import sraging

cohort = sraging.Cohort.from_csv("my_cohort.csv", delta=1.0)
result = sraging.fit_cohort(cohort)          # calibrator "profile_jax2" (1.0.1), the default since spec 003g
print(result.theta_hat)                      # SRParams(eta=..., beta=..., epsilon=..., kappa=..., xc=1.0)
for c in result.claims:
    print(c.parameter, c.shape, c.lower, c.upper, c.label)
result.to_json("my_cohort.fit.json")
```

A default fit (`profile_jax2`, with the lead combination claims and the leverage check) takes from under an hour to
many CPU-hours: the worked example's cohort took <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:timing.cpu_s:.0f -->2084<!-- /verify --> CPU-seconds for the fit plus about 2.6 times that for
its combination claims, and a sharp 10,000-individual cohort (spec 003g's fit 1) took
<!-- verify:tests/acceptance/results/fit1_003g.json:cpu_s:,.0f -->59,867<!-- /verify --> CPU-seconds, over its price of 8.6–15 CPU-hours. `--no-combinations` skips the combination claims. `profile_v033` takes from a few to
about 30 CPU-minutes. The cost is in solving the model, not in the number of individuals, and it grows when δ is
small relative to the lifespan (the worked example below, under `profile_v033`: daily records over an 870-day
lifespan, 23 CPU-minutes on the Mac that made it; the time depends on the machine and its load). The command prints nothing until the fit has finished. The measured cost of
every acceptance-panel fit is in `tests/acceptance/results/claims.json`. The JSON file is written only when
the fit finishes; an interrupted fit writes nothing and must be restarted. The header's "reference cost"
is the cost range v0.3.3 declares to the lab benchmark, not a prediction for your cohort.

Pass `--time-unit` (for example `days`) so the result records the unit: η is then per day², β and ε per
day, κ dimensionless (xc = 1). Without it the unit is recorded as `input units`.

## Fit a time range

When your records cover only part of life (animals enrolled at 182 days, a life table that starts at age 30, a study
stopped before the last deaths), give the range as a **window** [LO, HI] and fit it with `profile_jax2_w` (spec 006):

```bash
.venv/bin/sraging fit-cohort --csv my_cohort.csv --delta 1 --time-unit days \
    --window 182 inf --calibrator profile_jax2_w --json my_cohort.w.json
```

```python
cohort = sraging.Cohort.from_csv("my_cohort.csv", delta=1.0, window=(182.0, float("inf")))
result = sraging.fit_cohort(cohort, calibrator="profile_jax2_w")
print(result.diagnostics["window"])          # the window used, the rows dropped and censored
```

`fit-set --window LO HI` fits every cohort of a set on the same window, and its `summary.csv` records the window used
and the deaths censored at its edge. The window is one per cohort: per-individual entry ages are refused (spec 006,
Not Covered).

**What the window does to your rows**, in this order:

- 0 ≤ LO < HI. With δ, LO and a finite HI must be multiples of δ; otherwise the cohort is refused, naming the nearest
  grid ages. HI may be `inf`, which needs δ: HI then becomes the first grid point past your last record,
  (⌊a_max/δ⌋ + 1)·δ, and the mapping is recorded. Without δ (`--continuous`), HI = `inf` is refused.
- The benchmark's own rule (`apply_window`, copied verbatim): a row with age below LO is **dropped**, not censored.
  That individual was never seen; censoring it at LO would claim it was known alive at LO. This is called left
  truncation. A row with age above HI is censored at HI.
- The frozen jax2 code then snaps the window onto the δ grid (HI down, LO up). sraging already refuses a window off the
  grid, so the snap moves neither end. What it adds is the **edge tie**: a death recorded at HI happened in
  [HI, HI + δ), after the window closed, so it is censored at HI. It is common: in spec 006's <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:edge_tie.n_fits:d -->260<!-- /verify --> gate fits
  it fired on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:edge_tie.fired:d -->159<!-- /verify -->, its censored deaths came to more than 1 % of the records on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:edge_tie.above_1pc:d -->67<!-- /verify --> and more than 10 % on
  <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:edge_tie.above_10pc:d -->4<!-- /verify -->, and the median was <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:edge_tie.median_pc:.2f -->0.13<!-- /verify --> %. The largest was <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:edge_tie.max_pc:.1f -->46.7<!-- /verify --> %: <!-- verify:tests/acceptance/results/gate_006/a1/window-k2-p01-c1-s1.json:window.n_deaths_censored:,d -->934<!-- /verify --> deaths among
  <!-- verify:tests/acceptance/results/gate_006/a1/window-k2-p01-c1-s1.json:window.n_rows_given:,d -->2,000<!-- /verify --> records (`window-k2-p01-c1-s1`), where W's η and ε claims were both lower bounds only
  (`<!-- verify:tests/acceptance/results/gate_006/a1/window-k2-p01-c1-s1.json:claims[0].shape: -->lower_bound<!-- /verify -->`, `<!-- verify:tests/acceptance/results/gate_006/a1/window-k2-p01-c1-s1.json:claims[2].shape: -->lower_bound<!-- /verify -->`). If some death ages are off the grid (`DELTA_GRID`), or without δ,
  nothing is snapped and there is no edge tie.
- If fewer deaths are left than the calibrator has free parameters (4; 5 for `profile_jax2_h`), the fit is refused and
  the count is named. The count is taken after the edge tie.
- The fit sees only the rows kept. Rows named by the floor report and the leverage check are mapped back to your CSV's
  rows (`row`; the windowed row is `row_windowed`).

The printout gains a `window:` line. `diagnostics["window"]` records `window_given`, `window_eff` (the window used),
`hi_mapped_from`, `hi_rule` and `hi_used` (for HI = `inf`), `n_rows_given`, `n_dropped_below_lo`,
`n_censored_above_hi`, `n_rows_kept`, `n_deaths_censored` (the edge tie), `n_censored_at_hi_eff`, `n_deaths_fitted`,
`min_deaths` and `S_lo_at_theta_hat`.

**Which calibrator.**

- `profile_jax2_w` (W, version 1.0.1+w1) is `profile_jax2` 1.0.1's likelihood conditioned on survival to LO, with no
  external hazard (h = 0): the terms of the rows kept, minus n·log S(LO) for the n rows at risk. It is sraging's
  composition of jax2 1.0.1 with its window switch on; no jax2 release ran it. It is the calibrator spec 006 gated on
  windowed data. It is not the default.
- `profile_jax2_h` (H) fits a constant external hazard h as well, with the window jax2 shipped. Spec 006 did not gate
  its windowed claims: H was W's reference in R-2 and R-4 below.
- Every other calibrator refuses a window. `profile_jax2` (either version) and `profile_jax2_h@1.0.0` refuse it naming
  W and H; `fit-cohort` does so before it reads the CSV.

**What spec 006's gate showed.** The cells are generated cohorts of the benchmark's library v6, with known truth
(`tests/acceptance/results/`; counts the records hold only as lists are in
`specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json`, built from them by the script beside it).
A1 and A2 split the window arm at n = 25,000. That split is the panel's, set before the run, not a measured change.

- **A1**, gated: <!-- verify:tests/acceptance/results/a1_006.json:n_cohorts:d -->196<!-- /verify --> cohorts with n < 25,000, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.eta.n_cohorts:d -->112<!-- /verify --> at SR parameter grids and <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.n_a1_real:d -->84<!-- /verify --> planted at real life tables
  (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.n_a1_human:d -->22<!-- /verify --> of them human). W's closed 95 % intervals covered the truth for η on <!-- verify:tests/acceptance/results/a1_006.json:R-1.gated.eta.hits:d -->172<!-- /verify --> of <!-- verify:tests/acceptance/results/a1_006.json:R-1.gated.eta.claimed:d -->175<!-- /verify --> cohorts and
  for ε on <!-- verify:tests/acceptance/results/a1_006.json:R-1.gated.epsilon.hits:d -->158<!-- /verify --> of <!-- verify:tests/acceptance/results/a1_006.json:R-1.gated.epsilon.claimed:d -->160<!-- /verify -->, above the pre-registered cut (<!-- verify:tests/acceptance/results/a1_006.json:R-1.gated.eta.cut:.3f -->0.913<!-- /verify --> for η, set to tell a true 95 % from 88 %;
  R-1). The truth witness passed (R-3, passes: <!-- verify:tests/acceptance/results/a1_006.json:R-3.passes: -->True<!-- /verify -->): on no cohort did the truth score more than 1.92
  nats above W's estimate. Against H (R-2), the paired median log10(width_W / width_H) is at most 0.1 dex:
  <!-- verify:tests/acceptance/results/a1_006.json:R-2.eta.vs_H.median_dex:.3f -->-0.006<!-- /verify --> dex for η over the <!-- verify:tests/acceptance/results/a1_006.json:R-2.eta.vs_H.n_paired:d -->66<!-- /verify --> cohorts where both claims are closed and W's upper end is not truncated, and
  <!-- verify:tests/acceptance/results/a1_006.json:R-2.epsilon.vs_H.median_dex:.3f -->-0.051<!-- /verify --> for ε over <!-- verify:tests/acceptance/results/a1_006.json:R-2.epsilon.vs_H.n_paired:d -->85<!-- /verify -->.
- **A2**, gated under the collapse rule: <!-- verify:tests/acceptance/results/a2_006.json:n_cohorts:d -->44<!-- /verify --> cohorts planted at human life tables, with n ≥ 25,000. They give
  too few claims to tell 95 % from 88 %, so the rule asks only that coverage did not collapse to 70 %. Closed
  intervals covered the truth for η on <!-- verify:tests/acceptance/results/a2_006.json:R-1.gated.eta.hits:d -->31<!-- /verify --> of <!-- verify:tests/acceptance/results/a2_006.json:R-1.gated.eta.claimed:d -->32<!-- /verify --> and for ε on <!-- verify:tests/acceptance/results/a2_006.json:R-1.gated.epsilon.hits:d -->28<!-- /verify --> of <!-- verify:tests/acceptance/results/a2_006.json:R-1.gated.epsilon.claimed:d -->29<!-- /verify -->, and the truth witness passed (passes:
  <!-- verify:tests/acceptance/results/a2_006.json:R-3.passes: -->True<!-- /verify -->). More than the pre-registered 10 % of those intervals were truncated above (below), which
  triggers `W_LARGE_N`.
- **realfull**, reported, not gated: <!-- verify:tests/acceptance/results/realfull_006.json:n_cohorts:d -->20<!-- /verify --> cohorts planted at real life tables (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.realfull_human.n:d -->12<!-- /verify --> human; <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull_drosophila.eta.n_cohorts:d -->8<!-- /verify --> Drosophila,
  at n = <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull_drosophila_n_range[0]:,d -->400<!-- /verify --> to <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull_drosophila_n_range[1]:,d -->1,116<!-- /verify -->), observed through a window only. Closed intervals covered the truth for η on <!-- verify:tests/acceptance/results/realfull_006.json:R-1.gated.eta.hits:d -->17<!-- /verify -->
  of <!-- verify:tests/acceptance/results/realfull_006.json:R-1.gated.eta.claimed:d -->18<!-- /verify --> and for ε on <!-- verify:tests/acceptance/results/realfull_006.json:R-1.gated.epsilon.hits:d -->16<!-- /verify --> of <!-- verify:tests/acceptance/results/realfull_006.json:R-1.gated.epsilon.claimed:d -->16<!-- /verify -->. But <!-- verify:tests/acceptance/results/realfull_006.json:truncation.eta.truncated:d -->18<!-- /verify --> of the <!-- verify:tests/acceptance/results/realfull_006.json:truncation.eta.closed:d -->18<!-- /verify --> closed η intervals were truncated above (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.box_limited:d -->16<!-- /verify --> at the box,
  <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.widened:d -->2<!-- /verify --> widened), and <!-- verify:tests/acceptance/results/realfull_006.json:truncation.epsilon.truncated:d -->15<!-- /verify --> of <!-- verify:tests/acceptance/results/realfull_006.json:truncation.epsilon.closed:d -->16<!-- /verify --> ε intervals (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.epsilon.box_limited:d -->13<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.epsilon.widened:d -->2<!-- /verify -->). The data set η's upper end on <!-- verify:tests/acceptance/results/realfull_006.json:shapes.eta.capped_by_data:d -->0<!-- /verify --> of the
  <!-- verify:tests/acceptance/results/realfull_006.json:shapes.eta.n:d -->20<!-- /verify --> cohorts and ε's on <!-- verify:tests/acceptance/results/realfull_006.json:shapes.epsilon.capped_by_data:d -->1<!-- /verify -->.

**Upper ends the data did not set, at every n.** A closed claim's upper end is *truncated above* when it sits at the
box, was widened, or ends below the lead pair's walk or more than 0.3 dex below H's. The two main causes carry
different labels:

- `open above: box-limited` (the label names the box edge, such as `kappa<=300` or `beta<=1000`): on these
  windows η, ε and β/κ slide together along a ridge of the model, and the box ends it. More deaths do not close it:
  A2's larger n did not.
- `edge not found` (widened): J 1.0.1's frozen profile walk stopped short of points the fit had already evaluated, and
  the end was moved out to them. Such an end can sit below the truth: A2's only η miss and only ε miss are truths
  <!-- verify:tests/acceptance/results/a2_006.json:misses.eta.above.window-real-Sweden_F_1910_hetro-c0-s1:.2f -->0.28<!-- /verify --> and <!-- verify:tests/acceptance/results/a2_006.json:misses.epsilon.above.window-real-Sweden_F_1910_hetro-c0-s1:.2f -->0.31<!-- /verify --> dex above an end marked `<!-- verify:tests/acceptance/results/a2_006.json:truncation.eta.named.window-real-Sweden_F_1910_hetro-c0-s1: -->widened<!-- /verify -->` (`window-real-Sweden_F_1910_hetro-c0-s1`).

| Cohorts | n | η: truncated of closed (box, widened) | ε: truncated of closed (box, widened) | upper end set by the data, η / ε |
|---|---|---|---|---|
| A1 at SR parameter grids | < 25,000 | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.eta.truncated:d -->51<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.eta.closed:d -->108<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.eta.box_limited:d -->47<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.eta.widened:d -->3<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.epsilon.truncated:d -->36<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.epsilon.closed:d -->109<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.epsilon.box_limited:d -->30<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.epsilon.widened:d -->3<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.eta.capped_by_data:d -->58<!-- /verify --> / <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.epsilon.capped_by_data:d -->76<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_synthetic.eta.n_cohorts:d -->112<!-- /verify --> |
| A1 planted at real life tables | < 25,000 | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.truncated:d -->56<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.closed:d -->67<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.box_limited:d -->50<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.widened:d -->5<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.epsilon.truncated:d -->37<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.epsilon.closed:d -->51<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.epsilon.box_limited:d -->32<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.epsilon.widened:d -->5<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.capped_by_data:d -->12<!-- /verify --> / <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.epsilon.capped_by_data:d -->14<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.n_cohorts:d -->84<!-- /verify --> |
| A1 at A2's human tables | 10,000 | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.eta.truncated:d -->14<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.eta.closed:d -->15<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.eta.box_limited:d -->8<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.eta.widened:d -->5<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.epsilon.truncated:d -->8<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.epsilon.closed:d -->8<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.epsilon.box_limited:d -->4<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.epsilon.widened:d -->4<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.eta.capped_by_data:d -->2<!-- /verify --> / <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.epsilon.capped_by_data:d -->0<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_same_tables_as_a2_n10000.eta.n_cohorts:d -->22<!-- /verify --> |
| A2, human life tables | ≥ 25,000 | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.truncated:d -->28<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.closed:d -->32<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.box_limited:d -->14<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.widened:d -->13<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.epsilon.truncated:d -->29<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.epsilon.closed:d -->29<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.epsilon.box_limited:d -->13<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.epsilon.widened:d -->16<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.capped_by_data:d -->5<!-- /verify --> / <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.epsilon.capped_by_data:d -->0<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.n_cohorts:d -->44<!-- /verify --> |
| realfull | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull_drosophila_n_range[0]:,d -->400<!-- /verify --> to <!-- verify:tests/acceptance/results/gate_006/realfull/realfull-real-combined_human_F-c0-s0.json:n:,d -->100,000<!-- /verify --> | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.truncated:d -->18<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.closed:d -->18<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.box_limited:d -->16<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.widened:d -->2<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.epsilon.truncated:d -->15<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.epsilon.closed:d -->16<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.epsilon.box_limited:d -->13<!-- /verify -->, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.epsilon.widened:d -->2<!-- /verify -->) | <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.capped_by_data:d -->0<!-- /verify --> / <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.epsilon.capped_by_data:d -->1<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.realfull.eta.n_cohorts:d -->20<!-- /verify --> |

The truncated ends that are neither at the box nor widened are few. Inside A1, η's share does not rise with n:
<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_eta_by_n.0-1999.truncated:d -->43<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_eta_by_n.0-1999.closed:d -->61<!-- /verify --> below 2,000, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_eta_by_n.2000-9999.truncated:d -->36<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_eta_by_n.2000-9999.closed:d -->67<!-- /verify --> at 2,000 to 9,999 and <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_eta_by_n.10000-24999.truncated:d -->28<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_eta_by_n.10000-24999.closed:d -->47<!-- /verify --> at 10,000 to 24,999. Below
the split, the real-table cohorts' η share (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.truncated:d -->56<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a1_real.eta.closed:d -->67<!-- /verify -->) does not differ from A2's (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.truncated:d -->28<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.a2.eta.closed:d -->32<!-- /verify -->; Fisher p = <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:truncation.fisher_p_eta_a1_real_vs_a2:.2f -->0.77<!-- /verify -->).
The variable is the life table, not n.

**The rule, at every n:** read an upper end labelled `open above: box-limited` or `edge not found` as "at least this",
not as the interval's end, and read the lead pair (η/ε, r̄/η; next paragraph) for what the data determine. Use the
lower ends. `W_LARGE_N` fires on every W fit of a window with 25,000 or more rows given (counted before the window
drops any), as pre-registered from A2. Its message gives A2's counts by cause and the real-table rate below 25,000,
says that 25,000 is the panel's split, and states the rule above. Below 25,000 no warning fires: the labels are the
signal.

**Combination claims on windows.** W's default fit gives the lead pair, and spec 006 truth-scored it on windows (`Dg`
in each record). On A1, η/ε's closed intervals covered the truth on <!-- verify:tests/acceptance/results/a1_006.json:Dg.by_quantity.eta_over_eps.hits:d -->52<!-- /verify --> of <!-- verify:tests/acceptance/results/a1_006.json:Dg.by_quantity.eta_over_eps.claimed:d -->54<!-- /verify --> cohorts, and r̄/η's on <!-- verify:tests/acceptance/results/a1_006.json:Dg.by_quantity.rbar_over_eta.hits:d -->20<!-- /verify --> of <!-- verify:tests/acceptance/results/a1_006.json:Dg.by_quantity.rbar_over_eta.claimed:d -->21<!-- /verify --> plus
<!-- verify:tests/acceptance/results/a1_006.json:Dg.by_quantity.rbar_over_eta.one_sided[0]:d -->32<!-- /verify --> of <!-- verify:tests/acceptance/results/a1_006.json:Dg.by_quantity.rbar_over_eta.one_sided[1]:d -->32<!-- /verify --> one-sided. On A2, η/ε covered the truth on <!-- verify:tests/acceptance/results/a2_006.json:Dg.by_quantity.eta_over_eps.hits:d -->11<!-- /verify --> of <!-- verify:tests/acceptance/results/a2_006.json:Dg.by_quantity.eta_over_eps.claimed:d -->11<!-- /verify --> cohorts and r̄/η on <!-- verify:tests/acceptance/results/a2_006.json:Dg.by_quantity.rbar_over_eta.hits:d -->11<!-- /verify --> of <!-- verify:tests/acceptance/results/a2_006.json:Dg.by_quantity.rbar_over_eta.claimed:d -->11<!-- /verify -->, all closed: these are
the human tables whose η and ε upper ends the box or the search set. So where η alone stays open, η/ε and r̄/η close.
That count passes the collapse rule (cut <!-- verify:tests/acceptance/results/a2_006.json:Dg.by_quantity.eta_over_eps.cut:.3f -->0.777<!-- /verify -->) but is too small to show 95 %.

**The other gate results.**

- **β and κ.** On A1, β is an upper bound on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:shapes.a1.beta.upper_bound:d -->107<!-- /verify --> cohorts, all <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:shapes.a1.beta.upper_bound_box_limited:d -->107<!-- /verify --> at the box (none set by the data); it is
  a lower bound on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:shapes.a1.beta.lower_bound:d -->46<!-- /verify --> and closed on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:shapes.a1.beta.closed:d -->17<!-- /verify -->. On A2, β is a lower bound on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:shapes.a2.beta.lower_bound:d -->32<!-- /verify --> cohorts and closed on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:shapes.a2.beta.closed:d -->9<!-- /verify -->, and κ is an upper
  bound on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:shapes.a2.kappa.upper_bound:d -->31<!-- /verify --> (<!-- verify:tests/acceptance/results/a2_006.json:shapes.kappa.one_sided_data_determined:d -->20<!-- /verify --> of them set by the data). One-sided claims set by the data covered the truth: β on <!-- verify:tests/acceptance/results/a1_006.json:shapes.beta.coverage_one_sided_data_determined[0]:d -->43<!-- /verify --> of
  <!-- verify:tests/acceptance/results/a1_006.json:shapes.beta.coverage_one_sided_data_determined[1]:d -->46<!-- /verify --> (A1) and <!-- verify:tests/acceptance/results/a2_006.json:shapes.beta.coverage_one_sided_data_determined[0]:d -->32<!-- /verify --> of <!-- verify:tests/acceptance/results/a2_006.json:shapes.beta.coverage_one_sided_data_determined[1]:d -->32<!-- /verify --> (A2), κ on <!-- verify:tests/acceptance/results/a1_006.json:shapes.kappa.coverage_one_sided_data_determined[0]:d -->7<!-- /verify --> of <!-- verify:tests/acceptance/results/a1_006.json:shapes.kappa.coverage_one_sided_data_determined[1]:d -->7<!-- /verify --> and <!-- verify:tests/acceptance/results/a2_006.json:shapes.kappa.coverage_one_sided_data_determined[0]:d -->20<!-- /verify --> of <!-- verify:tests/acceptance/results/a2_006.json:shapes.kappa.coverage_one_sided_data_determined[1]:d -->20<!-- /verify -->. Read removal from r̄/η.
- **A stale θ̂ (R-4).** The mechanism is J's, and on windows it is more frequent. jax2 1.0.1 keeps θ̂ when a profile
  walk finds a point better by less than 1.92 nats ("stale: minor"), and stores that point in
  `diagnostics["estimate"]["best_point"]`, so θ̂ can sit up to 1.92 nats below the best point the fit saw. R-4 checked
  that W's θ̂ scores within 0.1 nats of H's stored fit at H's smallest h. It failed on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.a1.failures:d -->15<!-- /verify --> of A1's <!-- verify:tests/acceptance/results/a1_006.json:R-4.n:d -->196<!-- /verify --> cohorts
  (largest gap <!-- verify:tests/acceptance/results/a1_006.json:R-4.max_gap:.2f -->1.80<!-- /verify --> nats), <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.a2.failures:d -->3<!-- /verify --> of A2's <!-- verify:tests/acceptance/results/a2_006.json:R-4.n:d -->44<!-- /verify --> (<!-- verify:tests/acceptance/results/a2_006.json:R-4.max_gap:.2f -->0.54<!-- /verify -->) and <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.realfull.failures:d -->2<!-- /verify --> of realfull's <!-- verify:tests/acceptance/results/realfull_006.json:R-4.n:d -->20<!-- /verify --> (<!-- verify:tests/acceptance/results/realfull_006.json:R-4.max_gap:.2f -->0.81<!-- /verify -->); every gap was below 1.92 nats. On A1's
  <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.j_vs_w_synthetic.n:d -->112<!-- /verify --> SR-grid cohorts W's gap exceeded 0.1 nats on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.j_vs_w_synthetic.w_above:d -->11<!-- /verify -->, and J's own gap on its stored, unwindowed fits of their pure
  twins on <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.j_vs_w_synthetic.j_above:d -->3<!-- /verify --> (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.j_vs_w_synthetic.both_above:d -->1<!-- /verify --> cohort both; Fisher p = <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:r4.j_vs_w_synthetic.fisher_p:.3f -->0.050<!-- /verify -->). R-4's pre-registered consequence, stopping W, was overridden: Naveh shipped W with this limit stated (2026-11-12; extended to A2 and realfull on 2026-11-15). J widens each claim
  to contain every point the fit evaluated within 1.92 nats of the best log-likelihood (`sc_widen_claim`), so a claim
  contains the best point. What to do: read `best_point` and its `gap_nats` (the `gap:` line prints it). J's median gap
  on the benchmark's v5 reference is <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.11_minor_stale_points.median_gap_nats:g -->0.0013<!-- /verify --> nats (1.0.1's known limits, below); a gap above
  about 0.1 nats means θ̂ is not the best point the fit saw. Report the claims, not θ̂.
- **Host dependence.** W also inherits 1.0.1's host dependence on human cohorts (issue 1 of 1.0.1's known limits,
  below). The gate ran on WEXAC's AMD EPYC nodes; A2's worst R-4 cohort (`window-real-Sweden_M_1910_hetro-c1-s1`) ran
  on an <!-- verify:tests/acceptance/results/gate_006/a2/window-real-Sweden_M_1910_hetro-c1-s1.json:cpu_model: -->AMD EPYC 9754 128-Core Processor<!-- /verify -->. A Mac refit of it found a θ̂ <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:host.mac_minus_wexac_loglik:.3f -->0.047<!-- /verify --> nats better, with η <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:host.eta_pc_apart:.0f -->5<!-- /verify --> % and κ
  <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:host.kappa_ratio:.0f -->5<!-- /verify -->× apart, both inside WEXAC's claims, and κ ≪ 1 on both (<!-- verify:specs/006-calibrator-capabilities/review/evidence/r4_refits_006.json:refits.window-real-Sweden_M_1910_hetro-c1-s1.wexac_theta.kappa:.2g -->0.0028<!-- /verify --> on WEXAC, <!-- verify:specs/006-calibrator-capabilities/review/evidence/r4_refits_006.json:refits.window-real-Sweden_M_1910_hetro-c1-s1.mac_theta.kappa:.2g -->0.014<!-- /verify --> on the Mac: saturated removal).
  The Mac refits behind the R-4 decisions, one per cell, all found a θ̂ inside WEXAC's claims (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:host.refits_inside_wexac_claims:d -->3<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:host.n_refits:d -->3<!-- /verify -->; their
  own claims were not recorded; `specs/006-calibrator-capabilities/review/evidence/r4_refits_006.json`).
  On a human life table, compare claims across machines, not θ̂.
- **Leverage.** The check raised no flag on the <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.assessed_total:d -->209<!-- /verify --> fits it assessed (flags: <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.flags_total:d -->0<!-- /verify -->; <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.slides_total:d -->52<!-- /verify --> of them were flat-valley
  slides), so it raised no false flag. It was not tested for detection on windows: no case was planted.
  On cohorts planted at human life tables it could not assess <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.not_assessed_total:d -->47<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.n_total:d -->78<!-- /verify --> fits
  (<!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.a1_human_n10000.not_assessed:d -->13<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.a1_human_n10000.n:d -->22<!-- /verify --> at n = 10,000 in A1, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.a2.not_assessed:d -->26<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.a2.n:d -->44<!-- /verify --> in A2, <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.realfull_human.not_assessed:d -->8<!-- /verify --> of <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:leverage.human.realfull_human.n:d -->12<!-- /verify --> in realfull): there θ̂ was
  not stationary, its free gradient above the check's threshold, so no refit could be compared with it (`not_assessed`,
  see "The leverage check").
- **Identity** (`identity_006.json`). Without a window, W equals `profile_jax2` 1.0.1 on
  <!-- verify:tests/acceptance/results/identity_006.json:equal_by_rule.R-5:d -->3<!-- /verify --> cohorts, every float equal by its hex (θ̂, log-likelihood, warnings and claims; only the two
  diagnostics keys that record the window switch differ). With a window, W's and H's window path gives the same fit as
  the frozen jax2 1.0.1 code run with the same switches, on <!-- verify:tests/acceptance/results/identity_006.json:equal_by_rule.R-6:d -->4<!-- /verify --> pairs (W and H on two window cohorts).
- **Cost.** A W fit cost <!-- verify:tests/acceptance/results/a1_006.json:r.fit_over_j:.2f -->1.28<!-- /verify --> (A1), <!-- verify:tests/acceptance/results/a2_006.json:r.fit_over_j:.2f -->0.52<!-- /verify --> (A2) and <!-- verify:tests/acceptance/results/realfull_006.json:r.fit_over_j:.2f -->0.70<!-- /verify --> (realfull) times the CPU of J's stored fits of the
  same cohorts (J is `profile_jax2` 1.0.1). A default fit also computes the lead pair: with it, a pilot cohort cost
  <!-- verify:tests/acceptance/results/pilot_006.json:r.element_over_j:.3f -->3.627<!-- /verify --> times J's fit, the gate's own extra evaluations included (`pilot_006.json`; the gate's price `r_lead`).
- **Not tested**: the <!-- verify:specs/003b-jax2-101-port/review/evidence/census_scope.json:windowed:d -->52<!-- /verify --> sharp window cohorts (CV_c < 0.05), which have no H record (spec 006, Not Covered).

**What a window cannot tell you.** Every row is conditioned on being alive at LO, so the data say nothing about
survival before LO except through S(LO): two parameter values with the same S(LO) and the same survival after LO fit
equally well. A narrow window leaves fewer deaths and constrains fewer combinations of the parameters. The
minimum-deaths rule refuses only a window with fewer deaths than parameters, not one that is merely short. Where
S(LO) at θ̂ is below the frozen code's clip of 1e-12, the conditioning carries no information and the fit warns
`WINDOW_S_LO_CLIPPED`.

## Fit many cohorts at once: `fit-set` (spec 005)

`fit-set` fits every cohort of a long CSV (or of a directory with one CSV per cohort) in its own process, `--workers`
of them at a time, and writes one `fit-cohort --json` result per cohort plus `summary.csv`:

```bash
.venv/bin/sraging fit-set --csv all_cohorts.csv --cohort-col cohort --delta 1 --time-unit days \
    --workers 4 --out fits/
```

A cohort whose fit fails is recorded in `summary.csv` with its error and the others complete; the exit code is then 4.
`summary.csv` lists θ̂ and the log-likelihood but not the claims: θ̂ for κ and β is one point on a flat ridge, so read
each cohort's claims in its JSON before comparing parameters across cohorts. `--calibrator NAME[@VERSION]`
selects the calibrator as in `fit-cohort` (default `profile_jax2`). `--delta` (or `--continuous`) is required as for
`fit-cohort`, and so is `--time-unit`, which `fit-cohort` leaves optional. From Python: `sraging.fitset.fit_set(cohort_set, workers=1, calibrator=None, out_dir=...)`.
Phase `workers` checks, on one WEXAC node, that the fits of three cohorts give the same θ̂ bytes, log-likelihood and
claims in-process and with one and two workers. In the acceptance run (WEXAC job <!-- verify:tests/acceptance/results/workers_005.json:job_id: -->469166<!-- /verify -->, one
<!-- verify:tests/acceptance/results/workers_005.json:cpu: -->AMD EPYC 9634 84-Core Processor<!-- /verify --> node) all three agreed, at
<!-- verify:tests/acceptance/results/workers_005.json:cap.cpu_spent_h:.1f -->7.0<!-- /verify --> CPU-hours for the nine fits. Warnings and diagnostics are not
compared. Within one run every worker is a local process on the same machine. Across machines, compare claims at a
tolerance, not bit for bit: the same in-process fits were bit-identical on two WEXAC CPU models (EPYC 9754 and 9634,
three cohorts) and differed on a third (EPYC 7552, pilot P1): θ̂ by <!-- verify:specs/005-data-plots-cli/review/evidence/crossnode_005.json:pairs.epyc7552_vs_epyc9634.theta_max_rel:.1e -->7.2e-10<!-- /verify -->
relative, the log-likelihood by <!-- verify:specs/005-data-plots-cli/review/evidence/crossnode_005.json:pairs.epyc7552_vs_epyc9634.loglik_abs:.1e -->2.2e-11<!-- /verify -->, and a claim end by
<!-- verify:specs/005-data-plots-cli/review/evidence/crossnode_005.json:pairs.epyc7552_vs_epyc9634.claim_end_max_rel:.1e -->2.9e-05<!-- /verify --> relative (`review/evidence/crossnode_005.json` in
`specs/005-data-plots-cli/`). On a Mac, worker invariance is tested only with the unit tests' stand-in calibrator. The
worker count is recorded as provenance only.

## Reading the result

- **θ̂** (`theta_hat`): a value for every parameter η, β, ε, κ (xc = 1 is fixed). It is one representative
  point; for κ and raw β it is one point on a flat ridge — do not use it as a regression target.
- **Claims**, one per parameter, each with a *shape*:
  `closed` (a two-sided 95% profile-likelihood interval: the values whose best fit, with the other
  parameters re-optimised, is within 1.92 log-likelihood units — "nats" — of the maximum),
  `upper_bound` or `lower_bound` (only one side is supported by the data), or `none` (the data do not
  determine it inside the prior box). Every value is shown with a label saying what the data do and do
  not support.
- **The prior box** is the range the fit may search: η, β and ε in [1e-12, 1000] and κ in [0.001, 300], in
  the units of your ages. A θ̂ on an edge of the box means the likelihood kept rising out of it, along a
  *ridge* (a direction in which the likelihood barely changes); then the intervals of the other parameters
  end where the ridge leaves the box, not where the data put them, even when they are labelled
  `closed` (register D-018). The `BOX_EDGE` check below flags it; do not use the η, β and ε intervals of
  such a fit.
- **Checks**: seven truth-free checks run on every fit and are listed on the `checks:` line:
  - `BASIN_GAIN` / `BASIN_OUTSIDE` — the signature of v0.3.3's wrong fits on large, sharp cohorts
    (human life tables): the reported point is not the maximum. **Do not use those numbers.**
  - `CENSOR_TIE` — many deaths share the end-of-study censoring bin, which biases η upward.
  - `DELTA_GRID` — some ages are not multiples of δ; check the recording resolution.
  - `BOX_EDGE` — θ̂ sits on an upper edge of the prior box (within 0.1 %): the fit ran up a ridge (D-020)
    and the η, β and ε intervals are not usable. On the lab benchmark's
    <!-- verify:specs/001-reference-fit/review/measurements.json:box_edge_rule_rtol_1e-3.N:d -->1990<!-- /verify --> stored v0.3.3 fits it fires on at least <!-- verify:specs/001-reference-fit/review/measurements.json:box_edge_rule_rtol_1e-3.fires:d -->215<!-- /verify -->
    (fits whose `none` claims hide θ̂ cannot be scored), and in those a closed η interval contained the
    true η in <!-- verify:specs/001-reference-fit/review/measurements.json:box_edge_rule_rtol_1e-3.eta_closed_covered.fires[0]:d -->8<!-- /verify --> of
    <!-- verify:specs/001-reference-fit/review/measurements.json:box_edge_rule_rtol_1e-3.eta_closed_covered.fires[1]:d -->214<!-- /verify --> cases. A quiet `BOX_EDGE` does not make an
    interval right. (sraging's own check; v0.3.3 has none.)
  - `FLOORED_DEATHS` — some deaths were scored by the arithmetic or the time grid rather than by the model:
    v0.3.3's arithmetic put them on the −700 floor, or a corrected or refined calculation moves them by more
    than 1 nat. Each is listed with its age, its term in this fit, corrected and on a refined grid, and why:
    beyond where the solver stopped, essentially impossible under the fitted model, or limited by the grid.
    Check those rows; if they are real early deaths, refit on a time range that starts at a cut-off age you
    choose and state (`--window` with `profile_jax2_w`, [Fit a time range](#fit-a-time-range)), and compare the
    fits with and without the window. (sraging's own check; register D-004, D-105.)
  - `AGE0_DEATHS` — a few deaths (at most 5, or 1 % of the deaths) are recorded in the first interval [0, δ).
    Checked from the data alone, before anything else. The model gives deaths that early essentially no
    probability, so a handful can move the whole fit: one age-0 death added to the ITP example below moved
    η̂ about 5-fold and ε̂ about 10-fold and made the basin warnings fire. Check those rows, and if they are
    real, refit without them (a cut-off at δ; state it) and compare. (sraging's own check.)
- The last line, `RESULT ok — no warnings fired`, is printed only if all seven checks ran and none fired.
  It does **not** mean the fit is right or that any parameter is determined: read each claim's shape.
  No check detects a cohort the model does not describe (for example a mixture of two populations, such
  as both sexes pooled), or the p15 grid barrier (D-005).
- `RESULT FAILED` means the fit did not produce estimates: an error, or every optimiser start failed
  (D-015; the status in the JSON stays `ok`, as in v0.3.3, but nothing in it is an estimate). The
  command then exits with code 5.
- **Provenance**: the package version and git SHA, the numerics (`v033`), the calibrator, the grid, the
  library versions and the host are recorded in every result.
- **Both ends of the κ ridge** (spec 006). κ is rarely determined, so θ̂ is one point on a flat ridge. A fit records θ
  at both ends of the κ range it leaves open, in `diagnostics["kappa_ends"]`: `kappa_lo` and `kappa_hi`, the full θ
  at each end (`theta_lo`, `theta_hi`: η, β, ε, κ, then h for `profile_jax2_h`), `box_limited_lo` and `box_limited_hi`
  (that end sits at the calibrator's κ box), and its `definition`: "archive points within 1.92 nats of ll_best" for
  the jax2 calibrators, "walk nodes within 1.92 nats of the walk's best" for `profile_v033` (absent when its walk kept
  no nodes). `sraging.plots.overlay(result, cohort)` draws the data against the fit at both ends ([Plots](#plots)).
  Both are tested by unit tests only: no gate fit in spec 006 recorded them.

## Worked example

The repository ships one public cohort: <!-- verify:examples/results/itp_c2016_het3_control_female.json:cohort.n:d -->304<!-- /verify --> genetically heterogeneous UM-HET3
**control female** mice of the NIA Interventions Testing Program (cohort C2016, three sites pooled), of
which <!-- verify:examples/results/itp_c2016_het3_control_female.json:cohort.n_deaths:d -->303<!-- /verify --> died and <!-- verify:examples/results/itp_c2016_het3_control_female.json:cohort.n_censored:d -->1<!-- /verify --> was removed from the
study alive. The data are from the Mouse Phenome Database (project ITP1); [`examples/data/SOURCE.md`](examples/data/SOURCE.md)
says how the group was chosen (a rule set before the fit) and transformed, and
[`examples/data/CITATION`](examples/data/CITATION) how to cite it. Deaths are recorded to the day, so
δ = 1 day, and each death age has already been moved to the start of its day, as described above.

```bash
.venv/bin/sraging fit-cohort --csv examples/data/itp_c2016_het3_control_female.csv --delta 1 \
    --time-unit days --calibrator profile_v033 --json my_fit.json
```

This example names `profile_v033`, the reference it was written for; the same cohort under the default `profile_jax2`
is shown under [Choosing a calibrator](#choosing-a-calibrator).

The fit took <!-- verify:examples/results/itp_c2016_het3_control_female.json:timing.cpu_s:.0f -->1732<!-- /verify --> CPU-seconds, about 29 minutes: daily records on a
mean lifespan of <!-- verify:examples/results/itp_c2016_het3_control_female.json:provenance.grid.L_km:.0f -->870<!-- /verify --> days force v0.3.3's fine time step (`dt_ceiling` in the last line; register
D-011). This is the printout of the stored run, `examples/results/itp_c2016_het3_control_female.json` (the command also prints the `-> my_fit.json`
line; on your machine the provenance line shows the commit you run):

<!-- verify-output:examples/results/itp_c2016_het3_control_female.json -->
```text
sraging 0.1.0 · calibrator profile_v033@0.3.3 · n=304 (303 deaths, 1 censored) · delta=1 · reference cost median 590 CPU-s, range 140–70000 CPU-s
parameter  claim           theta-hat        lower        upper  label
eta        closed         3.5809e-05   1.5421e-05   0.00021411  two-sided 95% profile-likelihood interval
beta       closed           0.031686     0.010647       132.29  two-sided 95% profile-likelihood interval; theta-hat for kappa and raw beta is one point on a flat ridge; it depends on the start and the box and is not a regression target
epsilon    closed          0.0023706   0.00081361     0.015532  two-sided 95% profile-likelihood interval
kappa      none                0.001                            not determined by the data inside the prior box (the profile does not close on either side); the value shown is the fit's representative point, not an estimate; theta-hat for kappa and raw beta is one point on a flat ridge; it depends on the start and the box and is not a regression target
loglik -1956.3352137306206  (wall 1760.7 s, cpu 1732.3 s)
leverage: NOT RUN (a profile_v033 result written before spec 006)
note: combination claims are given only by the jax2 calibrators, whose combination claims were truth-scored at 1.0.0 (profile_jax2 1.0.0, profile_jax2_h 1.0.0; spec 003) and are not on 1.0.1 (a stated limit, spec 003g); profile_v033 gives none (spec 003 FR-018)
checks: BASIN_GAIN=ran/quiet BASIN_OUTSIDE=ran/quiet CENSOR_TIE=ran/quiet DELTA_GRID=ran/quiet BOX_EDGE=ran/quiet FLOORED_DEATHS=ran/quiet AGE0_DEATHS=ran/quiet
RESULT ok — no warnings fired
provenance: sraging 0.1.0 @ 20dbefce201a · numerics v033 · engine fp · grid N=900 dt=0.0003831620814843195 tau_max=2.0 dt_ceiling=0.0003831620814843195
```

How to read it:

- **All seven checks ran and none fired.** θ̂ for η, β and ε lies inside the prior box. The diagnostics
  say where each profile stopped (`diagnostics.grid`): `dev_stop` is the data, `box_edge` the box.
- **η** (per day²) lies between <!-- verify:examples/results/itp_c2016_het3_control_female.json:claims.eta.lower:.2g -->1.5e-05<!-- /verify --> and
  <!-- verify:examples/results/itp_c2016_het3_control_female.json:claims.eta.upper:.2g -->0.00021<!-- /verify -->, and **ε** (per day) between
  <!-- verify:examples/results/itp_c2016_het3_control_female.json:claims.epsilon.lower:.2g -->0.00081<!-- /verify --> and <!-- verify:examples/results/itp_c2016_het3_control_female.json:claims.epsilon.upper:.2g -->0.016<!-- /verify -->: both
  ends are set by the data, each fixed to about one order of magnitude.
- **β** (per day) is at least <!-- verify:examples/results/itp_c2016_het3_control_female.json:claims.beta.lower:.2g -->0.011<!-- /verify -->. That is a bound on raw β with κ flat
  over its whole box, so on its own it does not show that removal is detectable, and the early deaths below drive
  it. Its
  upper end, <!-- verify:examples/results/itp_c2016_het3_control_female.json:claims.beta.upper:.3g -->132<!-- /verify -->, is **not** a data bound: the profile walk reached the
  box there (κ at its upper limit, 300), as with every raw-β upper end set by κ's box (register D-009).
  Read β as a lower bound, although the label says two-sided.
- **κ** is not determined: its profile is flat across the whole box (both ends `box_edge`). Its θ̂ sits
  at the box's lower edge (0.001), a value the data allow, so `BOX_EDGE` stays quiet.
- **Three early deaths drive these numbers.** The cohort has 3 deaths before day 400 (days 54, 197 and
  200), 1 % of the mice. Refitted without them, η lies between <!-- verify:specs/001-reference-fit/review/evidence/itp_early_deaths_leverage.json:without.eta[1]:.2g -->2.1e-06<!-- /verify -->
  and <!-- verify:specs/001-reference-fit/review/evidence/itp_early_deaths_leverage.json:without.eta[2]:.2g -->1.9e-05<!-- /verify -->, ε between <!-- verify:specs/001-reference-fit/review/evidence/itp_early_deaths_leverage.json:without.epsilon[1]:.2g -->4.7e-05<!-- /verify -->
  and <!-- verify:specs/001-reference-fit/review/evidence/itp_early_deaths_leverage.json:without.epsilon[2]:.2g -->0.00071<!-- /verify --> (no overlap with the interval above), and β has only
  an upper bound, <!-- verify:specs/001-reference-fit/review/evidence/itp_early_deaths_leverage.json:without.beta[2]:.2g -->6.5<!-- /verify --> (`review/evidence/itp_early_deaths_leverage.json`).
  No check in this library detects such leverage. Both leverage checks drop one individual at a time, so neither
  can see three deaths that act together: `profile_jax2`'s (see "The leverage check" below), which does not flag on
  this cohort, and `profile_v033`'s, which runs since spec 006 (the stored result above predates it, hence `NOT RUN`).
  `profile_v033`'s check is not validated on any cohort (spec 006, R-9;
  `tests/acceptance/results/v033_leverage_006.json`). Its null failed: on a panel of <!-- verify:tests/acceptance/results/v033_leverage_006.json:R-9.null.n:d -->26<!-- /verify --> cohorts it flagged <!-- verify:tests/acceptance/results/v033_leverage_006.json:R-9.null.labels.flag:d -->6<!-- /verify --> of
  the <!-- verify:tests/acceptance/results/v033_leverage_006.json:R-9.null.assessed:d -->18<!-- /verify --> fits it could assess, all on cohorts planted at human life tables, with at most <!-- verify:tests/acceptance/results/v033_leverage_006.json:R-9.null.max_flags:d -->1<!-- /verify --> allowed, and fewer
  than the required <!-- verify:tests/acceptance/results/v033_leverage_006.json:R-9.null.min_assessed:d -->20<!-- /verify --> fits were assessed (conclusive: <!-- verify:tests/acceptance/results/v033_leverage_006.json:R-9.null.conclusive: -->False<!-- /verify -->). <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:v033.flags_hazard_arm:d -->5<!-- /verify --> of the <!-- verify:specs/006-calibrator-capabilities/review/evidence/windows_counts_006.json:v033.flags:d -->6<!-- /verify --> flags fell on
  cohorts with external deaths, which v033 does not model. Its positive control never ran: the run reached its cap of
  <!-- verify:tests/acceptance/results/v033_leverage_006.json:cap.cap_cpu_h:.0f -->8<!-- /verify --> CPU-hours first. So a flag from `profile_v033` ends "(profile_v033's check failed its validation, spec 006
  R-9)".
  The usual remedy is to fit a time range that leaves out young ages,
  whose deaths are dominated by causes the model does not
  include (`--window` with `profile_jax2_w`, [Fit a time range](#fit-a-time-range)), and to compare that fit with
  the same calibrator's fit of all ages to see whether the earliest deaths decide your result.
- UM-HET3 mice are genetically heterogeneous by design and the model has no heterogeneity term, so ε
  also absorbs differences between mice.
- Report the claims with their labels, not θ̂.

## Choosing a calibrator

Four calibrators fit the same model. Choose one with `--calibrator NAME` (Python: `calibrator="NAME"`):

| Calibrator | What it is | For |
|---|---|---|
| `profile_v033` | `mle_profile_jax` v0.3.3, the lab benchmark's reference until 2026-10-13, ported bit for bit; by name | results that must equal v0.3.3's |
| `profile_jax2` (the default) | `mle_profile_jax2` 1.0.1, the benchmark's successor (frozen at tag `mle_profile_jax2-1.0.1`, commit 6884fc8), ported bit for bit. It fixes registered defects of v0.3.3: wrong-basin optima on large, sharp cohorts (a smaller stale gap remains, below), early deaths scored by round-off, the censor-tie bias where its relabel fires, and claims that exclude points the fit itself evaluated. 1.0.1 adds a finer time step on sharp cohorts (below) and records jax2's `diagnostics["estimate"]`. It also gives combination claims (below). `profile_jax2@1.0.0` selects the previous version, unchanged | a cohort without external mortality |
| `profile_jax2_h` | the same, with a constant external hazard h fitted beside the four SR parameters: the survival is the SR survival times e^(−h t). h is reported as a nuisance, not as a claim. Version 1.0.1 is usable, opt-in, and every fit carries a `note:` of its known limit on sharp hazard data (below). `profile_jax2_h@1.0.0` is not usable: the jax2 session reports it not validated on library v5, and sraging warns (`H_NOT_USABLE`) | a cohort with deaths not caused by ageing (accidents, infection) at a roughly constant rate |
| `profile_jax2_w` | `profile_jax2` 1.0.1 conditioned on survival to the window's start, with h = 0 (version 1.0.1+w1; sraging's composition of jax2 1.0.1 with its window switch on, never run by jax2). Without a window it equals `profile_jax2` 1.0.1 float for float. Gated on windowed data in spec 006. At any n, read an upper end labelled box-limited or 'edge not found' as 'at least this'; it warns `W_LARGE_N` at 25,000 rows or more, the panel's split ([Fit a time range](#fit-a-time-range)) | a cohort observed over part of life (a time range), without external mortality |

**The default is `profile_jax2` 1.0.1** (spec 003g; Naveh's yes on 2026-10-28, after its gate). The benchmark moved its
reference to jax2 1.0.0 on library v5 on 2026-10-13, and to 1.0.1 on library v6 on 2026-10-27 (tester f5dcc742). Spec
003g's gate (`tests/acceptance/results/gate_003g.json`) scored 1.0.1 on a frozen panel of the strata where a registered
defect of 1.0.1 or of v0.3.3 fires. The panel and its rules were committed before the first window of records (W1) was
read; the five deviations decided later (DV-1 to DV-5, the panel's `[[deviations]]`) are recorded with their dates. The
gate reads `<!-- verify:tests/acceptance/results/gate_003g.json:status:s -->blocked; decided (11 of 11)<!-- /verify -->`: each blocking entry carries Naveh's decision (the panel's
`[[decisions]]`), every one a stated limit below.

- **Inside the benchmark's reference set.** No cohort's truth beats 1.0.1's estimate by more than 1.92 nats (the
  largest W = ll(truth) − ll(θ̂) is <!-- verify:tests/acceptance/results/score_w1_003g.json:verdicts[12].statistic.w_max:.3f -->0.287<!-- /verify -->). Coverage paired against v0.3.3 lost on no stratum
  where a loss could be measured; strata with no measured axis, or too few paired cells for a loss to show (DV-5), are
  listed as not covered. The intervals are narrower than the data-ignoring null's everywhere, but 1.0.1 often declines
  to cap η or ε from above: on the censored LB-2 cohorts η's claim is a lower bound only on <!-- verify:tests/acceptance/results/score_w1_003g.json:verdicts[13].statistic.shapes_j.eta.open_above:d -->30<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w1_003g.json:verdicts[13].statistic.shapes_j.eta.n:d -->66<!-- /verify --> (`gate_003g.json`, `pr4_declines`).
- **v0.3.3 shown failing first, on one of its four registered defects: the wrong basin.** It lands in a wrong basin on
  <!-- verify:tests/acceptance/results/predecessor_003g.json:verdicts[0].statistic.k_shown:d -->29<!-- /verify --> of <!-- verify:tests/acceptance/results/predecessor_003g.json:verdicts[0].statistic.n:d -->32<!-- /verify --> basin cohorts. On the round-off stratum it is shown on <!-- verify:tests/acceptance/results/predecessor_003g.json:verdicts[1].statistic.k_shown:d -->19<!-- /verify --> of <!-- verify:tests/acceptance/results/predecessor_003g.json:verdicts[1].statistic.n:d -->178<!-- /verify --> cohorts, all human and all
  with the basin's signature (15 are basin cohorts), and on none of the 112 non-human ones: its round-off defect itself
  is not shown on v6 (DV-4). Its time-step ceiling has no v6 cohort in its regime (DV-1, a library gap like E. coli),
  and its censor-tie bias has no v6 cohort.
- **Outside the reference set** (the sharp and nladder cohorts): no truth beats the estimate by more than 1.92 nats on
  the <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[12].statistic.n_witnessed:d -->56<!-- /verify --> censored and <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[18].statistic.n_witnessed:d -->171<!-- /verify --> pure sharp cohorts with a witness (largest W <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[18].statistic.w_max:.2f -->-0.01<!-- /verify -->).

**The default's stated limits** (Naveh, 2026-10-28 and 2026-10-29; spec 003g, Not Covered). A default fit in the sharp
regime prints the first one on its `DT_RULE` line.

- **Sharp cohorts** (CV_c < 0.05, where `DT_RULE` fires), measured on <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[12].statistic.n_witnessed:d -->56<!-- /verify --> censored and <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[18].statistic.n_witnessed:d -->171<!-- /verify --> pure benchmark
  cohorts:
  - Closed η intervals covered the truth on <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[20].statistic.k_out:d -->132<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[20].statistic.n_out:d -->133<!-- /verify --> pure and <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[14].statistic.k_out:d -->43<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[14].statistic.n_out:d -->45<!-- /verify --> censored cohorts; ε on
    <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[22].statistic.k_out:d -->156<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[22].statistic.n_out:d -->157<!-- /verify --> and <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[16].statistic.k_out:d -->45<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[16].statistic.n_out:d -->47<!-- /verify -->. The censored η count missed the gate's margin against <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[14].statistic.k_in:d -->194<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[14].statistic.n_in:d -->199<!-- /verify --> inside
    by one claim, but a count this low has probability <!-- verify:tests/acceptance/results/gate_003g.json:pr3_notes[5].p_at_most_k_out_if_equal:.2f -->0.31<!-- /verify --> under exactly equal coverage: it is consistent
    with no difference.
  - η has no upper end on <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[19].statistic.shapes_j.eta.open_above:d -->38<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[19].statistic.shapes_j.eta.n:d -->171<!-- /verify --> pure and <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[13].statistic.shapes_j.eta.open_above:d -->11<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[13].statistic.shapes_j.eta.n:d -->56<!-- /verify --> censored cohorts, about 1 in 5. Every
    one of those lower bounds covered the truth (<!-- verify:tests/acceptance/results/score_w3_003g.json:limits.SRK-1/outside/pure.one_sided.eta.hits:d -->38<!-- /verify --> and <!-- verify:tests/acceptance/results/score_w3_003g.json:limits.SRK-1/outside/censor.one_sided.eta.hits:d -->11<!-- /verify -->). On each of them β is undetermined
    too: no claim, or only the κ box's end (`limits` … `eta_beta` in `score_w3_003g.json`).
  - The intervals are wide: the median closed width is <!-- verify:tests/acceptance/results/score_w3_003g.json:limits.SRK-1/outside/pure.closed_widths.eta.median_dex:.2f -->1.50<!-- /verify --> dex for η on pure cohorts and <!-- verify:tests/acceptance/results/score_w3_003g.json:limits.SRK-1/outside/censor.closed_widths.eta.median_dex:.2f -->1.67<!-- /verify --> on censored
    ones, and <!-- verify:tests/acceptance/results/score_w3_003g.json:limits.SRK-1/outside/pure.closed_widths.epsilon.median_dex:.2f -->2.42<!-- /verify --> and <!-- verify:tests/acceptance/results/score_w3_003g.json:limits.SRK-1/outside/censor.closed_widths.epsilon.median_dex:.2f -->2.47<!-- /verify --> dex for ε. That is narrower than the data-ignoring null (the box's 15 decades), not
    a value to two digits.
  - **Raw β is not determined.** It was closed on <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[19].statistic.shapes_j.beta.closed:d -->4<!-- /verify --> pure and <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[13].statistic.shapes_j.beta.closed:d -->2<!-- /verify --> censored cohorts, and every raw-β upper end
    is the κ box's, not the data's (<!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[19].statistic.shapes_j.beta.upper_end_at_kappa_box:d -->131<!-- /verify --> pure, <!-- verify:tests/acceptance/results/score_w3_003g.json:verdicts[13].statistic.shapes_j.beta.upper_end_at_kappa_box:d -->41<!-- /verify --> censored; the fit's own label says so). Read removal
    from r̄/η, the lead pair, which is not truth-scored on `profile_jax2` 1.0.1 (it is for `profile_jax2_w` on windows:
    [Fit a time range](#fit-a-time-range)).
- **The sharpest cohorts are few.** Below CV_c 0.01 two cohorts were witnessed; a third, pure-k0.0316-p15-c0-s1, has
  no record (it was stopped after repeated preemption). No censored cohort is sharper than CV_c 0.018 (`census_003b.json`).
  On the 3 p15 basin cohorts, where removal and production balance, and on the one-cohort LB-1x and SRK-2 strata, every
  closed claim covered the truth (p15: η 3/3, ε 3/3, β 1/1; its 2 β lower bounds covered too), but coverage cannot be
  tested at N ≤ 3.
- **Fine time resolution** (δ/L below about 6e-4) was never scored: v6's finest cohorts are the 16 C. elegans ones at
  about 6e-4 (DV-1), and E. coli (δ/L ≈ 1e-4), where v0.3.3 failed on its registered basin cases, has no v5 or v6
  cohort.
- **Human life tables.** A default fit prints `LEVERAGE … could not assess this fit` (`not_assessed`, below): on all 8
  human fits measured (7 pilot fits and fit 2). `BOX_EDGE` fired on <!-- verify:tests/acceptance/results/score_w1_003g.json:human_box_edge.box_edge:d -->85<!-- /verify --> of the <!-- verify:tests/acceptance/results/score_w1_003g.json:human_box_edge.n:d -->396<!-- /verify --> human benchmark
  records (β <!-- verify:tests/acceptance/results/score_w1_003g.json:human_box_edge.by_parameter.beta:d -->45<!-- /verify -->, κ <!-- verify:tests/acceptance/results/score_w1_003g.json:human_box_edge.by_parameter.kappa:d -->40<!-- /verify -->); where it fires, do not use the raw η, β and ε intervals (above). Where it does
  not, an upper end can still be the box's (217 η and 199 ε of the 396 records by jax2's count, CONTEXT-SYNC f59a, not
  re-measured here): read such an
  end as no upper limit. Human closed η widths are bimodal (quartiles <!-- verify:tests/acceptance/results/score_w1_003g.json:human_box_edge.closed_widths.eta.q25_dex:.2f -->0.11<!-- /verify -->, <!-- verify:tests/acceptance/results/score_w1_003g.json:human_box_edge.closed_widths.eta.median_dex:.2f -->0.62<!-- /verify --> and <!-- verify:tests/acceptance/results/score_w1_003g.json:human_box_edge.closed_widths.eta.q75_dex:.2f -->2.28<!-- /verify --> dex):
  the wide ones end where the box does [I]. Read the lead pair (η/ε, r̄/η) for what the data determine; it is not
  truth-scored on `profile_jax2` 1.0.1. For `profile_jax2_w` on windows, both covered the truth on all <!-- verify:tests/acceptance/results/a2_006.json:Dg.n:d -->11<!-- /verify --> of A2's human
  cohorts on which they were scored ([Fit a time range](#fit-a-time-range)). Give human ages in days.
- **External mortality and fit windows**: 1.0.1 models neither, so on such data its claims are not calibrated. On the
  <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.n:d -->66<!-- /verify --> realfull benchmark cohorts (planted at real life tables, with external deaths) the truth sat more than 10 nats
  below the estimate on <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.W.below_minus_10:d -->39<!-- /verify --> (W from <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.W.max:.2f -->-0.09<!-- /verify --> to <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.W.min:,.0f -->-14,259<!-- /verify --> nats; per individual of the library cohort, which can be ~6 % larger than the n fitted, a median of
  <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.W.per_individual.median:.3f -->-0.010<!-- /verify -->). J's closed intervals there covered the truth on <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.closed.eta.hits:d -->26<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.closed.eta.claimed:d -->61<!-- /verify --> (η) and <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.closed.epsilon.hits:d -->20<!-- /verify --> of <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.closed.epsilon.claimed:d -->57<!-- /verify --> (ε),
  at a median width of <!-- verify:tests/acceptance/results/score_w1_003g.json:realfull_all.closed_widths.eta.median_dex:.2f -->0.59<!-- /verify --> dex for η. Use `profile_jax2_h` for external deaths, and `profile_jax2_w`
  for a time range without them ([Fit a time range](#fit-a-time-range)).
- **Heterogeneous cohorts** are not witnessed: 1.0.1's likelihood cannot score their truth.
- **Combination claims** (η/ε, r̄/η and the others) were scored against the truth at 1.0.0 (spec 003), not at 1.0.1.
  The lead pair of `profile_jax2_w` was scored on windows (spec 006, [Fit a time range](#fit-a-time-range)).
- **Host agreement** was checked on one human cohort (spec 003g's fit 2): sraging's port on this Mac and jax2's record
  on its cluster gave the same claim shapes and coverage hits and the same physical answer (β at the box, κ ≪ xc).
  <!-- verify:tests/acceptance/results/fit2_compare_003g.json:compare.n_ends_beyond:d -->2<!-- /verify --> of its 6 interval ends differ by more than 0.03 dex (<!-- verify:tests/acceptance/results/fit2_compare_003g.json:compare.rows[0].end_dex.lower:.2f -->0.04<!-- /verify --> for β's lower end and 0.11 for κ's upper end),
  and the log-likelihoods by <!-- verify:tests/acceptance/results/fit2_compare_003g.json:loglik.mac_minus_record:.2f -->-0.41<!-- /verify --> nats (Mac minus cluster; 1.7e-5 per individual at the n = 23,448 fitted), presumably
  along the ridge [I]. The basin re-fit fired on the cluster and has not run on this Mac.

The κ-fixed names jax2 1.0.1 also ships (`mle_profile_jax2_kfix_k0p001` and five more) are refused: a later spec
registers them. `profile_jax2` refuses a window; `profile_jax2_w` and `profile_jax2_h` fit one (spec 006,
[Fit a time range](#fit-a-time-range)).

**1.0.1's time step on sharp cohorts.** On tightly bunched lifespans (CV below about <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.13_shared_time_step_defect.cv_below:g -->0.025<!-- /verify -->), 1.0.0 and v0.3.3 can let
the survival fall to 0 past the mean, so each later death scores the −700 floor, about <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.13_shared_time_step_defect.nats_per_floored_death:d -->690<!-- /verify --> nats (handoff §6
issue 13). 1.0.1 shrinks the time step wherever the binning-corrected coefficient of variation CV_c is below 0.05
(`DT_RULE`). On the benchmark's library v6, <!-- verify:tests/acceptance/results/census_003b.json:in_scope.sraging.J:d -->290<!-- /verify --> of <!-- verify:tests/acceptance/results/census_003b.json:views.J:, -->2,408<!-- /verify --> cohorts are in that scope
(`tests/acceptance/results/census_003b.json`); <!-- verify:specs/003b-jax2-101-port/review/evidence/census_scope.json:windowed:d -->52<!-- /verify --> of them are windowed
(`specs/003b-jax2-101-port/review/evidence/census_scope.json`); spec 006 did not test `profile_jax2_w` on them.
Such fits cost much more: a full fit of a
10,000-individual cohort in scope took <!-- verify:tests/acceptance/results/parity_003b.json:rows[10].port_cpu_s:,.0f -->19,654<!-- /verify --> CPU-seconds on this Mac, against <!-- verify:tests/acceptance/results/parity_003b.json:rows[8].port_cpu_s:,.0f -->1,850<!-- /verify --> for a
full fit of a 10,000-individual cohort outside it (`parity_003b.json`), both without combination claims; with the
lead pair and sraging's own checks, a default fit of that in-scope cohort took
<!-- verify:tests/acceptance/results/fit1_003g.json:cpu_s:,.0f -->59,867<!-- /verify --> CPU-seconds (spec 003g's fit 1,
`tests/acceptance/results/fit1_003g.json`). Coverage inside the scope was measured by spec 003g on 227 benchmark
cohorts, all in scope: see the sharp-cohort limit above.

**The leverage check** (`profile_jax2` and `profile_jax2_h`, spec 003f; on 1.0.1, spec 003b; `profile_jax2_w`, spec 006).
Every jax2 fit asks whether its estimate
rests on one individual. It ranks the individuals by their one-step influence on η/ε, r̄/η, η and ε, and drops the most
influential. It then refits without that individual twice: the fit's own maximum-likelihood stage, and an optimizer
started at the reported θ̂. The better refit is compared with the reported θ̂:

- **Flag** (warning `LEVERAGE`): any of the four quantities moves by more than 1.5×, and the refit gains more than 1.92
  nats. That is the 95 % threshold the claims themselves use, so the reported θ̂ lies outside what the data without
  that individual allow.
- **Flat-valley slide**, reported but not flagged: a larger move at a smaller gain.
- **Failed**, which also warns: a refit that ends more than 0.1 nats below θ̂ on its own objective. Its optimizer
  stopped short, so its move is not a measurement.

The result is the `leverage:` line and `diagnostics["leverage"]`; it never changes θ̂ or the claims. What it has been
tested on:

- **One positive case.** On the ITP cohort with one added age-0 death, the check (on 1.0.1) drops that death and
  flags. The four quantities move up to <!-- verify:tests/acceptance/results/itp_003b.json:R.all4:.3g -->38.2<!-- /verify -->×
  (r̄/η <!-- verify:tests/acceptance/results/itp_003b.json:R.lead_pair:.3g -->1.81<!-- /verify -->× and η/ε <!-- verify:tests/acceptance/results/itp_003b.json:fits.ITP+age0.leverage.ratios.eta_over_eps.ratio:.3g -->1.59<!-- /verify -->× among them), the
  refit gains <!-- verify:tests/acceptance/results/itp_003b.json:delta_total:.2f -->5.42<!-- /verify --> nats, and the
  check cost <!-- verify:tests/acceptance/results/itp_003b.json:fits.ITP+age0.leverage.cost.share_of_fit:.0% -->3%<!-- /verify --> of that fit. Its refit equals 1.0.1's own fit of the cohort without that death
  (`tests/acceptance/results/itp_003b.json`).
- **Nulls and planted outliers, pre-registered** (spec 003b, on 1.0.1, `tests/acceptance/results/ac11_003b.json`, read in
  `specs/003b-jax2-101-port/review/evidence/ac11_reading.json`). No flag on <!-- verify:tests/acceptance/results/ac11_003b.json:nulls.held_out.per_set.all4.n:d -->12<!-- /verify --> held-out library
  cohorts of 1,000 individuals (95 % upper bound <!-- verify:tests/acceptance/results/ac11_003b.json:nulls.held_out.per_set.all4.wilson[1]:.2f -->0.24<!-- /verify -->), on
  <!-- verify:tests/acceptance/results/ac11_003b.json:nulls.small_n.per_set.all4.n:d -->47<!-- /verify --> cohorts of 150 (<!-- verify:tests/acceptance/results/ac11_003b.json:nulls.small_n.per_set.all4.wilson[1]:.3f -->0.076<!-- /verify -->) or on
  <!-- verify:tests/acceptance/results/ac11_003b.json:nulls.twins.per_set.all4.n:d -->18<!-- /verify --> generated cohorts (<!-- verify:tests/acceptance/results/ac11_003b.json:nulls.twins.per_set.all4.wilson[1]:.2f -->0.18<!-- /verify -->). These cohorts sit where removal
  is negligible (ρβ/ρη below 0.1 on <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:nulls.rho_beta_over_rho_eta_below_0p1:d -->56<!-- /verify --> of the <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:nulls.n:d -->59<!-- /verify --> library ones, and
  <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:planted_points.rho_beta_over_rho_eta_range[0]:.3f -->0.055<!-- /verify -->–<!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:planted_points.rho_beta_over_rho_eta_range[1]:.3f -->0.060<!-- /verify --> at every planted point). There r̄/η is not
  determined and its ratio slides freely: the quantities moved more than 1.5× on <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:nulls.R_all4_above_1p5:d -->46<!-- /verify --> of those
  <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:nulls.n:d -->59<!-- /verify --> without a flag, and <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:nulls.small_n_flat_valley_slide:d -->40<!-- /verify --> of the <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:nulls.small_n:d -->47<!-- /verify --> cohorts of 150 read `flat_valley_slide`, the
  normal state of a clean small cohort. So these nulls test the 1.92-nat gain alone (the largest gain was
  <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:nulls.max_delta.small_n:.2f -->0.63<!-- /verify --> nats). One planted age-0 death was found (flagged, and the death dropped) in
  <!-- verify:tests/acceptance/results/ac11_003b.json:detection.age0.100.detected:d -->5<!-- /verify -->, <!-- verify:tests/acceptance/results/ac11_003b.json:detection.age0.300.detected:d -->6<!-- /verify --> and <!-- verify:tests/acceptance/results/ac11_003b.json:detection.age0.1000.detected:d -->6<!-- /verify --> of 6 cohorts at n = 100, 300 and 1,000, with η and ε moving
  <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:age0_detected.R_raw_min:.0f -->23<!-- /verify -->–<!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:age0_detected.R_raw_max:.0f -->253<!-- /verify -->×. One death at 1.5× the largest age was found in
  <!-- verify:tests/acceptance/results/ac11_003b.json:detection.late.100.detected:d -->5<!-- /verify -->, <!-- verify:tests/acceptance/results/ac11_003b.json:detection.late.300.detected:d -->6<!-- /verify --> and <!-- verify:tests/acceptance/results/ac11_003b.json:detection.late.1000.detected:d -->4<!-- /verify --> of 6, but <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:late_detected.R_raw_below_1p5:d -->9<!-- /verify --> of those
  <!-- verify:specs/003b-jax2-101-port/review/evidence/ac11_reading.json:late_detected.n:d -->15<!-- /verify --> moved only r̄/η, so they may ride the same slide. Not tested: cohorts where removal matters (as in
  ITP or human life tables), fits inside DT_RULE's scope, and `profile_jax2_h`'s check.
- **A null panel on 1.0.0** (spec 003f). On <!-- verify:tests/acceptance/results/panel_003f.json:median_drops.n:d -->26<!-- /verify --> library cohorts of <!-- verify:tests/acceptance/results/panel_003f.json:median_drops.n_range[0]:, -->1,000<!-- /verify --> to
  <!-- verify:tests/acceptance/results/panel_003f.json:median_drops.n_range[1]:, -->100,000<!-- /verify --> individuals, dropping a median-influence individual flagged
  <!-- verify:tests/acceptance/results/panel_003f.json:median_drops.flags:d -->0<!-- /verify --> (95 % upper bound <!-- verify:tests/acceptance/results/panel_003f.json:median_drops.wilson95[1]:.2f -->0.13<!-- /verify -->). Only the
  <!-- verify:tests/acceptance/results/panel_003f.json:median_drops.effective_null.n_le_10000:d -->12<!-- /verify --> cohorts with at most 10,000 individuals are an informative null
  (<!-- verify:tests/acceptance/results/panel_003f.json:median_drops.effective_null.flags:d -->0<!-- /verify --> flags, 95 % upper bound
  <!-- verify:tests/acceptance/results/panel_003f.json:median_drops.effective_null.wilson95[1]:.2f -->0.24<!-- /verify -->). In the cohorts of 25,000 or more no drop flagged; the largest move, on pure-real-Denmark_M_1900_hetro-c1-s1, was η and ε
  <!-- verify:tests/acceptance/results/panel_003f.json:top_drops.pure-real-Denmark_M_1900_hetro-c1-s1.R:.3g -->1.37<!-- /verify -->× at a gain of
  <!-- verify:tests/acceptance/results/panel_003f.json:top_drops.pure-real-Denmark_M_1900_hetro-c1-s1.delta_total:.2f -->0.03<!-- /verify --> nats.
- **Overhead.** The check added a median <!-- verify:tests/acceptance/results/panel_003f.json:overhead.median:.0% -->9%<!-- /verify --> and at most
  <!-- verify:tests/acceptance/results/panel_003f.json:overhead.max:.0% -->27%<!-- /verify --> to a fit.
- **g0**, recorded beside each drop, is the gain the same optimizer makes started at θ̂ on the full data. It was 0 on
  all <!-- verify:specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json:n:d -->8<!-- /verify --> of 003b's pilot fits, because the estimator's own L-BFGS-B stopped at its first line search without a step
  (<!-- verify:specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json:abnormal_at_nit0:d -->8<!-- /verify --> of <!-- verify:specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json:n:d -->8<!-- /verify -->, `specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json`): on ITP at a gradient norm of
  <!-- verify:specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json:itp_grad_norm:.2g -->0.0018<!-- /verify --> (the optimiser stopped at θ̂), on the human cohorts at <!-- verify:specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json:human_grad_norm_range[0]:.0f -->10<!-- /verify -->–<!-- verify:specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json:human_grad_norm_range[1]:.0f -->642<!-- /verify --> (the optimiser could not
  move from θ̂, β at or near its box). So g0 cannot fail where it matters. On such human cohorts the label is `not_assessed` (next),
  not `no_flag`: the check could not see. There its two refits disagreed by up to
  <!-- verify:specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json:human_refit_starts_max_disagreement_nats:.1f -->12.5<!-- /verify --> nats. No label reads g0 (OPEN_TASKS [T-LEV-G0-LNSRCH]).
- **`not_assessed`** (spec 003g; warning `LEVERAGE`, "could not assess this fit"): the optimiser could not move from θ̂
  (its warm start stopped where it began) at a free-gradient norm above <!-- verify:tests/acceptance/results/lev_pilot_003g.json:g_crit_per_ind:g -->5e-05<!-- /verify --> per individual, so no refit
  without one individual can be compared with θ̂. Measured: on the 7 human hazard pilot fits (<!-- verify:tests/acceptance/results/lev_pilot_003g.json:rows[4].free_gradient_per_individual:.2g -->0.0004<!-- /verify --> to
  <!-- verify:tests/acceptance/results/lev_pilot_003g.json:rows[2].free_gradient_per_individual:.2g -->0.026<!-- /verify --> per individual) and on fit 2; not on ITP (<!-- verify:tests/acceptance/results/lev_pilot_003g.json:rows[6].free_gradient_per_individual:.2g -->5.8e-06<!-- /verify -->) or fit 1. The threshold is per individual
  because the norm is a sum over individuals (Naveh, 2026-10-28). It holds with a wide margin at n = 150: on the
  <!-- verify:tests/acceptance/results/lev_small_n_003g.json:n_measured:d -->22<!-- /verify --> of <!-- verify:tests/acceptance/results/lev_small_n_003g.json:n_rows:d -->47<!-- /verify --> small benchmark fits whose θ̂ could be recovered exactly, |g|/n ran from <!-- verify:tests/acceptance/results/lev_small_n_003g.json:free_gradient_per_individual.min:.2g -->2e-09<!-- /verify --> to
  <!-- verify:tests/acceptance/results/lev_small_n_003g.json:free_gradient_per_individual.max:.2g -->1.6e-06<!-- /verify --> (median <!-- verify:tests/acceptance/results/lev_small_n_003g.json:free_gradient_per_individual.median:.2g -->1.7e-07<!-- /verify -->), <!-- verify:tests/acceptance/results/lev_small_n_003g.json:n_above:d -->0<!-- /verify --> above the threshold (spec 003g, `lev_small_n_003g.json`). No cohort near
  n = 36, the size of many worm and fly cohorts, was measured.

The check cannot see a cluster: dropping one of several similar deaths moves nothing while the others remain. On the
worked example below, it drops the death at day <!-- verify:tests/acceptance/results/readme_example_003b.json:leverage.dropped.age:g -->53<!-- /verify -->
in the CSV (recorded day 54; the CSV stores age − 1) and reports a slide, not a flag:
- η and ε move about 2× (largest <!-- verify:tests/acceptance/results/readme_example_003b.json:leverage.R.all4:.3g -->2.18<!-- /verify -->×).
- The refit gains only <!-- verify:tests/acceptance/results/readme_example_003b.json:leverage.delta_total:.2f -->0.51<!-- /verify --> nats.
- Without that death, η is <!-- verify:tests/acceptance/results/readme_example_003b.json:leverage.refit.theta[0]:.2g -->2e-05<!-- /verify -->
  and ε <!-- verify:tests/acceptance/results/readme_example_003b.json:leverage.refit.theta[2]:.2g -->0.0011<!-- /verify -->, both inside
  the intervals shown below.

The leverage in this cohort is its three early deaths together (above), which this check does not test.

The worked example's cohort under `profile_jax2`:

```bash
.venv/bin/sraging fit-cohort --csv examples/data/itp_c2016_het3_control_female.csv --delta 1 \
    --time-unit days --calibrator profile_jax2 --json my_fit_jax2.json
```

<!-- verify-output:examples/results/itp_c2016_het3_control_female_jax2.json -->
```text
sraging 0.1.0 · calibrator profile_jax2@1.0.1 · n=304 (303 deaths, 1 censored) · delta=1 · fit cost 2084 CPU-s (measured)
parameter  claim           theta-hat        lower        upper  label
eta        closed         3.5791e-05   1.5422e-05   0.00021462  two-sided 95% profile-likelihood interval · open above: box-limited (kappa<=300): the upper end is set by the box, not by the data
beta       closed           0.031667     0.010649       132.17  two-sided 95% profile-likelihood interval; theta-hat for kappa and raw beta is one point on a flat ridge; it depends on the start and the box and is not a regression target; its upper end is set by the kappa box (kappa_max = 300), not by the data (spec 003's kappa_max = 3000 study): read it as a lower bound; the data determine combinations such as r-bar and beta/(1+kappa) · open above: box-limited (kappa<=300): the upper end is set by the box, not by the data
epsilon    closed          0.0023692   0.00081315     0.015788  two-sided 95% profile-likelihood interval
kappa      none                0.001                            not determined by the data inside the prior box (the profile does not close on either side); the value shown is the fit's representative point, not an estimate; theta-hat for kappa and raw beta is one point on a flat ridge; it depends on the start and the box and is not a regression target
eta_over_eps closed           0.015107     0.011308     0.021478  open above: box-limited (kappa<=300): the upper end is set by the box, not by the data
rbar_over_eta closed             878.65       629.76       1207.7  
loglik -1956.3349841006857  (wall 8086 s, cpu 2084 s)
leverage: flat_valley_slide -- dropped row 74 (age 53, death); eta/eps 1.2104x, rbar/eta 0.83967x, eta 0.55502x, eps 0.45854x; refit gains 0.51419 nats (by the MLE stage); cost 133.61 s CPU (6% of the fit)
note: profile_jax2's claims assume no external mortality: on the tester's hazard arm v0.3.3's closed eta claims covered the truth 8 times in 261 (PREREG_C:104); use profile_jax2_h for a cohort with external deaths
mechanisms: N=900 · basin re-fit not fired · censor-tie relabel 0 rows · floored dt no · widened none · Nelder-Mead fallbacks 0
combination claims (the lead group: eta_over_eps, rbar_over_eta; --combinations all adds the other 4): 5427.8 CPU-s (2.6x the fit); order measured (tests/acceptance/results/order_003.json (run at 12264a1, committed 8d89aac))
L_SR = 872.21 input units (the SR mean lifetime at theta-hat); box in L_SR units: eta*L^2 in [7.6075e-07, 7.6075e+08], beta*L in [8.7221e-10, 8.7221e+05], eps*L in [8.7221e-10, 8.7221e+05], kappa in [0.001, 300]
gap: the best point the fit evaluated beats theta-hat by 6.21e-11 nats (below the 1.92-nat threshold)
checks: BASIN_GAIN=ran/quiet BASIN_OUTSIDE=ran/quiet CENSOR_TIE=ran/quiet DELTA_GRID=ran/quiet BOX_EDGE=ran/quiet FLOORED_DEATHS=ran/quiet AGE0_DEATHS=ran/quiet
RESULT ok — no warnings fired
provenance: sraging 0.1.0 @ c309fff60a37 · numerics jax2 · engine fp · grid N=900 dt=0.0003831620814843195 tau_max=2.0 dt_ceiling=0.0003831620814843195
```

How to read it, beside the v0.3.3 fit above:

- **The per-parameter claims agree with v0.3.3's** on this cohort: η between <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.eta.lower:.2g -->1.5e-05<!-- /verify --> and
  <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.eta.upper:.2g -->0.00021<!-- /verify -->, ε between <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.epsilon.lower:.2g -->0.00081<!-- /verify --> and <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.epsilon.upper:.2g -->0.016<!-- /verify -->.
  Under 1.0.1, θ̂ and the log-likelihood are exactly 1.0.0's, and so is every lower end; three upper ends moved a little,
  as 1.0.1's profile stage refines them: η from <!-- verify:tests/acceptance/results/readme_example_003b.json:claims.eta.old.upper:.3g -->0.000218<!-- /verify --> to <!-- verify:tests/acceptance/results/readme_example_003b.json:claims.eta.new.upper:.3g -->0.000215<!-- /verify -->, ε from
  <!-- verify:tests/acceptance/results/readme_example_003b.json:claims.epsilon.old.upper:.3g -->0.0155<!-- /verify --> to <!-- verify:tests/acceptance/results/readme_example_003b.json:claims.epsilon.new.upper:.3g -->0.0158<!-- /verify -->, and raw β's (set by κ's box) from <!-- verify:tests/acceptance/results/readme_example_003b.json:claims.beta.old.upper:.3g -->122<!-- /verify --> to
  <!-- verify:tests/acceptance/results/readme_example_003b.json:claims.beta.new.upper:.3g -->132<!-- /verify --> (`tests/acceptance/results/readme_example_003b.json`). Two labels are new. η's upper end is marked **open above: box-limited**: it sits where κ reaches its
  box edge (300), so the box sets it, not the data. Every raw-β upper end now says that it is set by κ's
  box: spec 003 refitted with κ up to 3000, and the caps moved with the box (`tests/acceptance/results/kmax_003.json`).
- **Combination claims** are intervals for quantities the data determine better than the raw parameters,
  each from its own profile likelihood over the same box. By default a fit gives the **lead pair**:
  - **η/ε**, between <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.eta_over_eps.lower:.3g -->0.0113<!-- /verify --> and <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.eta_over_eps.upper:.3g -->0.0215<!-- /verify --> per day here;
  - **r̄/η**, between <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.rbar_over_eta.lower:.4g -->629.8<!-- /verify --> and <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:claims.rbar_over_eta.upper:.4g -->1208<!-- /verify --> days, where r̄ = β(1 − κ ln(1 + 1/κ)) is the
    mean removal rate.

  These two are the quantities that stay put both along κ's flat valley and along the scaling ridge (η, β, ε)
  → λ(η, β, ε); spec 003 measured the order (`src/sraging/claims/order.json`). `--combinations all` adds
  √η/ε, r̄, β/(η(κ+1)) and β/(1+κ); `--no-combinations` gives none.
- **They cost more than the fit.** Here the lead pair took <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:combinations.cost_s:.0f -->5428<!-- /verify --> CPU-seconds, <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:combinations.ratio_to_fit:.1f -->2.6<!-- /verify -->× the fit's <!-- verify:examples/results/itp_c2016_het3_control_female_jax2.json:timing.cpu_s:.0f -->2084<!-- /verify -->;
  all six take several times that. Use `--no-combinations` when you need only the per-parameter claims.
- **How well they cover, on 1.0.0 (spec 003; not re-measured on 1.0.1)** (simulated cohorts with known truth,
  `tests/acceptance/results/coverage_003.json`). On cohorts without censoring, η/ε's closed intervals covered the
  truth <!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2|pure.quantities.eta_over_eps.closed_coverage:.3f -->0.935<!-- /verify --> of the time, and no worse than the same fits' η and ε claims. On censored cohorts they covered
  only <!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2|censor.quantities.eta_over_eps.closed_coverage:.3f -->0.818<!-- /verify --> (<!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2|censor.quantities.eta_over_eps.closed_n:d -->44<!-- /verify --> intervals), worse than η and ε, with every miss below the truth: the
  test's one failure. We attribute it to the test data (the measurements support this but do not prove it): the
  benchmark's generator spreads lifespans wider than the model does at the true parameters, which pulls η/ε down.
  On the censored cohorts the benchmark flags, η/ε's point sat below the truth in <!-- verify:specs/003-calibrator/review/evidence/coverage_cv_bias_003.json:arms.profile_jax2|censor.valid_vs_flagged.flagged.point_below_truth:.2f -->0.79<!-- /verify --> of fits,
  against <!-- verify:specs/003-calibrator/review/evidence/coverage_cv_bias_003.json:arms.profile_jax2|censor.valid_vs_flagged.valid.point_below_truth:.2f -->0.47<!-- /verify --> on those it marks valid. On the valid ones, η/ε
  covered <!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2|censor.quantities.eta_over_eps.valid.closed_coverage:.3f -->0.933<!-- /verify --> (<!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2|censor.quantities.eta_over_eps.valid.closed_n:d -->15<!-- /verify --> intervals; Wilson 95 % <!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2|censor.quantities.eta_over_eps.valid.wilson95[0]:.2f -->0.70<!-- /verify -->–<!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2|censor.quantities.eta_over_eps.valid.wilson95[1]:.2f -->0.99<!-- /verify -->, too few to tell 0.82 from 0.95); where the bias is largest,
  <!-- verify:specs/003-calibrator/review/evidence/coverage_cv_bias_003.json:arms.profile_jax2|censor.by_cv_stat.10_and_above.hits:d -->5<!-- /verify --> of <!-- verify:specs/003-calibrator/review/evidence/coverage_cv_bias_003.json:arms.profile_jax2|censor.by_cv_stat.10_and_above.n:d -->9<!-- /verify -->. Real data have no generator; where the model misses the data's lifespan spread, we
  expect the same to show first in η/ε, whose intervals are the narrowest (an inference, not measured). On hazard
  cohorts (`profile_jax2_h`) η/ε nearly lost too: against η, z = <!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2_h|hazard.quantities.eta_over_eps.gate.vs_eta.z:.2f -->-2.13<!-- /verify --> on <!-- verify:tests/acceptance/results/coverage_003.json:scores.profile_jax2_h|hazard.quantities.eta_over_eps.gate.vs_eta.n_cells:d -->29<!-- /verify --> paired cohorts, one short of the
  30 the test needs to call a loss. r̄, β/(1+κ), β/(η(κ+1)), and r̄/η on censored and hazard cohorts, were too often
  one-sided to be compared, so the test does not show them as good as η and ε; r̄/η, in the default lead pair, was
  compared with η and ε only on cohorts without censoring.
- Other labels a combination claim can carry: **negligible** (an upper bound on r̄/η below the measured
  floor at which removal can be told from none, `tests/acceptance/results/neg_floor_003.json`), **saturated removal** (a κ upper bound below 0.364/√n_eff: small κ means
  saturated removal, not no removal) and **edge not found** (the self-consistency rule widened a side).
- The new lines report the mechanisms that ran (the basin re-fit, the censor-tie relabel, the grid), the
  SR mean lifetime L_SR, the prior box in L_SR units, and the gap between θ̂ and the best point the fit
  evaluated.

**What the jax2 calibrators' claims are, and are not.** jax2 re-measured 1.0.1's coverage only
on the pure cohorts of its J-MED set, where closed coverage did not change (η <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:section4_j_med_coverage.eta_covered:d -->44<!-- /verify --> and ε <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:section4_j_med_coverage.eps_covered:d -->45<!-- /verify -->
of <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:section4_j_med_coverage.n:d -->48<!-- /verify -->; handoff §4). Censored, hazard and human coverage were not re-measured for 1.0.1, whose profile stage
can move an end (the example above), so read what follows as 1.0.0's. The benchmark's validation of 1.0.0
(`specs/003-calibrator/review/evidence/jax2_reference_notes.json`, from its reports) found:

- **95 % intervals cover a little less than 95 %**, even when the model is right: on valid pure cohorts the
  closed η intervals covered the truth <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:coverage_valid_pure_closed.eta:.3f -->0.892<!-- /verify --> of the time and ε <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:coverage_valid_pure_closed.epsilon:.3f -->0.909<!-- /verify --> (v0.3.3:
  <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:coverage_valid_pure_closed.v033_eta:.3f -->0.884<!-- /verify --> and <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:coverage_valid_pure_closed.v033_epsilon:.3f -->0.890<!-- /verify -->).
- **Their ends run narrow**: on a selected set of <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:narrow_edges.n_cohorts:d -->17<!-- /verify --> cohorts, every end that a later version moved was too
  narrow, by up to <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:narrow_edges.max_dex:.2f -->0.11<!-- /verify --> dex.
- **κ's bounds are not calibrated**: one-sided κ bounds covered the truth <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:kappa_one_sided_coverage.pooled:.3f -->0.402<!-- /verify --> of the time, against a
  nominal 0.95. Do not use κ claims.
- **An η or ε upper end set at β's box edge** (<!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:box_conditional_human_uppers.beta_box_upper:.0f -->1000<!-- /verify -->), as on many human life tables, is the box's, not the
  data's, and can open or close on another machine.
- **The censor-tie relabel changes how the data are read**: when every censored record shares the last age,
  a death recorded in that same δ-interval is re-read as censored at its own age ("alive then"). That is
  what restores η's coverage on censored cohorts (from <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:censor_tie_relabel.eta_coverage_before:.3f -->0.634<!-- /verify --> to
  <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:censor_tie_relabel.eta_coverage_after:.3f -->0.912<!-- /verify -->); the `mechanisms:` line says how many rows it relabelled.
- **`profile_jax2` is the wrong model for external deaths or a fit window**: use `profile_jax2_h` for external
  deaths and `profile_jax2_w` for a window (spec 006). On cohorts observed through a window, `profile_jax2`'s closed η
  intervals covered the truth
  <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:misspecified_on_hazard_and_window.window_eta_coverage_j:.3f -->0.533<!-- /verify --> of the time, no better than v0.3.3. `profile_jax2` refuses a
  window.
- **Record first-bin deaths at age 0**, not δ/2, as described under *Fit a cohort*: the benchmark's own
  cohorts wrote them at δ/2 until its library v5.
- **Results depend on the machine**, as for v0.3.3: the benchmark's 1.0.0 claims differ between this Mac and
  its Linux cluster on <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:hosts.cohorts_differ:d -->17<!-- /verify --> of <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:hosts.cohorts:d -->17<!-- /verify --> cohorts compared, and <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:hosts.human_shape_changes:d -->3<!-- /verify --> of <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:hosts.human_records:d -->4<!-- /verify --> human records change shape.

**1.0.1's known limits**, as jax2's handoff lists them (§6; the numbers are copied, with their lines, into
`specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json`):

- **Human claims depend on the CPU** (issue 1): on human cohorts whose ends sit at the box, the same code and data on
  another CPU model flipped η's shape on <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.1_host_dependent_human_claims.eta_flips:d -->4<!-- /verify -->, β's on <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.1_host_dependent_human_claims.beta_flips:d -->3<!-- /verify --> and ε's on
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.1_host_dependent_human_claims.epsilon_flips:d -->5<!-- /verify --> of <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.1_host_dependent_human_claims.human_cohorts:d -->20<!-- /verify --> cohorts, and none of <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.1_host_dependent_human_claims.non_human_cohorts:d -->13<!-- /verify --> others.
- **`profile_jax2_h` on sharp hazard data** (issue 8, [T-H-HAZARD-SHARP]): on the <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.measured_cohorts:d -->3<!-- /verify --> of the
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.registered_sharp_hazard_cohorts:d -->56<!-- /verify --> registered sharp hazard cohorts where jax2 measured it, all on library v5 data (<!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.measured_v6:d -->0<!-- /verify --> on v6), its
  grid was <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.nats_off_at_truth_low:.1f -->1.2<!-- /verify --> to <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.nats_off_at_truth_high:.1f -->13.6<!-- /verify --> nats off at the truth (<!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.prod_best_max:.1f -->14.9<!-- /verify --> against a second reference).
  The register inferred <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.inferred_more_v6:d -->69<!-- /verify --> more v6 hazard cohorts (the two of them measured, on v5, were
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.beyond_cut_measured_low:.2f -->0.11<!-- /verify -->–<!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.8_h_on_sharp_hazard_data.beyond_cut_measured_high:.2f -->0.60<!-- /verify --> nats off); of those <!-- verify:tests/acceptance/results/score_w2_003g.json:strata.HZ-4.n_witnessed:d -->69<!-- /verify -->, <!-- verify:tests/acceptance/results/score_w2_003g.json:strata.HZ-4.verdict.statistic.n:d -->36<!-- /verify --> are valid on v6. On the 34 human ones jax2's witness scored the truth on a
  reference grid, and none is more than 1.92 nats below it (the largest W is <!-- verify:tests/acceptance/results/score_w2_003g.json:strata.HZ-4.verdict.statistic.w_max:.2f -->-0.37<!-- /verify -->); the other 35 have only the
  fit's own grid, on which this grid error cancels (`tests/acceptance/results/score_w2_003g.json`). The 56 sharp ones have no v6 `_h` record. This is the `note:` every `_h`
  1.0.1 fit prints; since spec 003g each fit also says whether its own SR CV at θ̂ puts it in that stratum.
- **Stale points** (issue 11): the reported θ̂ can lie below the best point the fit evaluated; nothing bounds the gap
  (on the benchmark's v5 reference, <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.11_minor_stale_points.records_gap_above_1e-9:, -->1,300<!-- /verify --> of <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.11_minor_stale_points.records:, -->1,776<!-- /verify --> records had a gap above 1e-9 nats; median
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.11_minor_stale_points.median_gap_nats:g -->0.0013<!-- /verify --> nats, the largest <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.11_minor_stale_points.major_gap_nats:.2f -->2.26<!-- /verify -->). The stored log-likelihood can sit below the likelihood at θ̂ too, by
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.11_minor_stale_points.stored_value_below_likelihood_nats:.2f -->1.77<!-- /verify --> nats on a 1.0.0 record (library v2); the largest found under 1.0.1 is
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.11_minor_stale_points.largest_detected_1_0_1:.3f -->0.011<!-- /verify --> nats. The claims cover the best point; the `gap:` line reports it.
- **`profile_jax2` without h is misspecified** on hazard and window data (issue 12): use `profile_jax2_h` (for a window
  without external deaths, sraging's `profile_jax2_w`, spec 006).
- **The time-step defect stays in 1.0.0 and v0.3.3** (issue 13): only 1.0.1 fixes it, and only where CV_c <
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.13_shared_time_step_defect.dt_rule_cv_c_cut:g -->0.05<!-- /verify --> (above).
- **`diagnostics["estimate"]`** is jax2's own block (schema 1). Its `best_point.theta` is empty where the best point is
  at most <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.14_best_point_theta_not_recorded.gap_max_nats:g -->0.01<!-- /verify --> nats above θ̂ (issue 14: on <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.14_best_point_theta_not_recorded.not_recorded_s3:d -->415<!-- /verify --> of jax2's
  <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.14_best_point_theta_not_recorded.s3_records:, -->1,068<!-- /verify --> S3 records); its `theta_hat_loglik` is the value as stored, not a fresh evaluation (issue
  11); and θ̂ carries no exact log-parameters, so re-evaluating at log θ̂ on box-corner human cohorts can differ by
  about <!-- verify:specs/003b-jax2-101-port/review/evidence/jax2_101_limits.json:issues.17_dt_rule_claim_limits.ulp_discontinuity_nats:g -->1<!-- /verify --> nat (issue 17c).
- **`DT_RULE`'s own limits** (issue 17): the remaining grid error at the profile edges on sharp cohorts of ~10,000 is
  not measured, the scope's edge cases are in scope but untested, and nothing is claimed for CV_c ≥ 0.05.

## The likelihood engine

`sraging.engine` evaluates the SR model's log-likelihood and its gradient for features v0.3.3 does not have:
- an external hazard, constant or by age segment;
- a fit window, and staggered entry per individual;
- interval or exact death times;
- age segments that change a parameter, the DPU accumulation rate A (it speeds up production only) or the threshold
  on an age window;
- heterogeneity in one parameter (the truncated normal on a checked Gauss rule by default since spec 004; spec 002's
  Gauss–Hermite mixture by name, `definition="normal_frac_sd"`);
- the model's survival and hazard curves, with no data.

It does not fit; the calibrators that use it arrive in later specs.
`examples/engine_example.py` runs one example of each on the ITP cohort, at the example fit's θ̂:

```python
import json
from sraging import Cohort, SRParams
from sraging.engine import AgeSegments, Heterogeneity, Segment, curves, evaluator

fit = json.load(open("examples/results/itp_c2016_het3_control_female.json"))
p = SRParams(**{k: fit["theta_hat"][k] for k in ("eta", "beta", "epsilon", "kappa")}, time_unit="days")
c = Cohort.from_csv("examples/data/itp_c2016_het3_control_female.csv", delta=1.0, time_unit="days")
ll, g = evaluator(c).value_and_grad(p)          # g: d loglik / d log(parameter), by name
seg = AgeSegments(Segment(id="rapa", parameter="beta", value=1.5, window=(600.0, 900.0)))
ev = evaluator(c, segments=seg, free=("eta", "beta", "epsilon", "kappa", "seg:rapa"))
mix = evaluator(c, heterogeneity=Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7))
S = curves(p, [600.0, 800.0, 1000.0, 1200.0]).S
```

The example's results:
- **The plain log-likelihood** is <!-- verify:examples/results/engine_example.json:plain.loglik:.4f -->-1956.3352<!-- /verify -->, the fit's own value.
  The gradient there is near zero. For η it is <!-- verify:examples/results/engine_example.json:plain.grad.eta:.3g -->-0.00189<!-- /verify -->.
- **With a window** of [400, 1400) days the log-likelihood is <!-- verify:examples/results/engine_example.json:window.loglik:.4f -->-1907.1700<!-- /verify -->, on
  <!-- verify:examples/results/engine_example.json:window.n_at_risk:d -->300<!-- /verify --> mice at risk.
- **With an external hazard** of 1e-4 per day it is <!-- verify:examples/results/engine_example.json:hazard.loglik:.4f -->-1961.4584<!-- /verify -->.
- **With β × 1.5 on [600, 900) days** it is <!-- verify:examples/results/engine_example.json:segment.loglik:.4f -->-2473.0968<!-- /verify -->. Its gradient in the
  segment's log-value is <!-- verify:examples/results/engine_example.json:segment.grad_seg_rapa:.4g -->-2358<!-- /verify -->.
- **With x_c heterogeneous** (v = 0.2) it is <!-- verify:examples/results/engine_example.json:heterogeneity.loglik:.4f -->-1964.6218<!-- /verify -->.
  With `normal_frac_sd` the node count M is yours to choose and is not checked (the default definition checks it:
  see spec 004's section below). On this
  cohort, at θ̂, M = 7 is 0.04 nats from M = 41 and M = 81, which agree to 4e-4. On lifetimes as tight as the
  score test's (cv ≈ 4 %), M = 11 was 3–7 nats off per 2,000 deaths, while M = 41 and M = 81 agreed
  (`specs/002-fp-engine/review/evidence/mixture_M.json`).
- **The model's survival** at 1,000 days is <!-- verify:examples/results/engine_example.json:curves.S[2]:.3f -->0.211<!-- /verify -->.

Every result records its grid and its numerics:
- **"v033"** is v0.3.3's arithmetic, bit for bit. With every feature off it is the same compiled program
  (`tests/acceptance/results/t0_002.json`).
- **"v1"**, the default, differs from "v033" only where the divergence register says so (D-004, D-106,
  D-107, D-108), and a CI guard checks that term by term.

The main difference is D-004: v0.3.3's death mass for an early death, S(t) − S(t+δ), is round-off when S ≈ 1,
and v1 takes it from the absorbed mass instead. That removes the round-off, not the time grid's own error
on such a death (see `FLOORED_DEATHS` above).

`fit-cohort` still fits with "v033". It uses the engine only for the `FLOORED_DEATHS` check.

## Generate a cohort

`sraging.generator` draws a cohort from the SR model as the lab benchmark's library specifies one. It replaces
SRK's generator (spec 003d, tag `sraging-gen-1`):

```python
from sraging.generator import CohortSpec, materialize

co = materialize(CohortSpec(name="k0.1-p00", rho_eta=2.2434680350647502, rho_beta=0.22189823414589713,
                            rho_eps=0.001995262314968879, kappa=0.1, n=1000, L_phys=100.0, seed=7,
                            delta=1.0, censor_frac=0.2))
co.death_times, co.events          # recorded ages (multiples of delta); 1 death, 0 censored
co.censor_time, co.realised_censored_fraction, co.caveats, co.provenance["generator_version"]
```

Importing it sets JAX to float64 and `PJRT_NPROC=1` for the whole process, so generate in its own process.

What it guarantees:
- **The stated model.** Each cohort is an inverse-CDF draw from the Fokker–Planck solution on one frozen grid per
  version: N <!-- verify:tests/acceptance/results/sweep_003d.json:choice.N:d -->6000<!-- /verify -->, dt
  <!-- verify:tests/acceptance/results/sweep_003d.json:choice.dt:.0e -->2e-04<!-- /verify -->, and
  <!-- verify:tests/acceptance/results/sweep_003d.json:choice.K:d -->161<!-- /verify --> heterogeneity nodes. The grid
  passed refinement on every library configuration (`tests/acceptance/results/sweep_003d.json`).
- **The same bytes.** A given (spec, seed, version) gives the same arrays:
  <!-- verify:tests/acceptance/results/determinism_003d.json:n_match_run1:,d -->2,410<!-- /verify --> of
  <!-- verify:tests/acceptance/results/determinism_003d.json:n:,d -->2,410<!-- /verify --> library cohorts across two
  Mac runs, and <!-- verify:tests/acceptance/results/determinism_epyc_003d.json:n_match:,d -->2,410<!-- /verify --> of
  <!-- verify:tests/acceptance/results/determinism_epyc_003d.json:n_epyc:,d -->2,410<!-- /verify --> against WEXAC
  AMD EPYC.
- **The benchmark's observation model.**
  - Every age is a multiple of δ, and a death in the first bin is recorded at 0.
  - Censoring happens at one time C on the δ grid, and no death is recorded at C.
  - `time_range` equals the benchmark's `apply_window` byte for byte:
    <!-- verify:tests/acceptance/results/window_003d.json:n_bytes_equal:d -->345<!-- /verify --> of
    <!-- verify:tests/acceptance/results/window_003d.json:n_compared:d -->345<!-- /verify --> windows. A window off
    the δ grid is refused by name.
- **Its limits are stated.** Anything outside the version's validated envelope says so in `co.caveats`.

Its limits:
- **One model, independent code.** The generator and the calibrators solve the same Fokker–Planck model, so an error
  in the model's equations would be common to both. Their discretisations are independent. At the truth of
  <!-- verify:tests/acceptance/results/calibrator_dt_003d.json:summary.n:d -->40<!-- /verify --> expected-count
  cohorts, `profile_jax2`'s likelihood on a refined grid matches the generator's law to
  <!-- verify:tests/acceptance/results/calibrator_dt_003d.json:summary.max_abs_refined_deficit:.1e -->1.1e-06<!-- /verify -->
  nats per individual (`tests/acceptance/results/calibrator_dt_003d.json`).
- **Fitting a sharp cohort with `profile_jax2@1.0.0` meets that calibrator's time step** (1.0.1 uses the finer grid
  where CV_c < 0.05; see *Choosing a calibrator*). On its production grid,
  its likelihood at the truth is up to
  <!-- verify:tests/acceptance/results/calibrator_dt_003d.json:summary.worst.calibrator_deficit_at_truth_per_individual:.3f -->0.214<!-- /verify -->
  nats per individual too low. On the grid its own `DT_RULE` would choose, the gap is at most
  <!-- verify:tests/acceptance/results/calibrator_dt_003d.json:summary.max_abs_dt_rule_deficit:.1e -->1.8e-05<!-- /verify -->.
- **The benchmark's planned CV labels** came from SRK's coarse grid. Judge a generated cohort's CV against
  `tests/acceptance/results/point_table_full_003d.json`.

The full list of limits, with numbers, is in the hand-off message,
[`docs/planning/handoff_2026-10-14_generator_to_tester.md`](docs/planning/handoff_2026-10-14_generator_to_tester.md).

## Heterogeneity, the Langevin engine and `engine=` (spec 004)

Spec 004 gives sraging one heterogeneity definition at a known cost, its own Langevin sampler, and one switch between
the three ways to get a population. The algorithms, with code citations, are in `docs/algorithms/main.pdf`
(sections "Heterogeneity" and "The Langevin engine"); every result below is in `tests/acceptance/results/*_004.json`.

### Heterogeneity: the truncated normal, with a checked node count

`Heterogeneity(parameter=, sd=)` gives each individual the parameter P·s, with s = 1 + sd·Z and Z a standard normal
truncated to s > 0. This is the generator's law, so FP, the generator and the Langevin share one model. It is the
default (register D-401). Spec 002's `normal_frac_sd`, P(1 + vZ) on Gauss–Hermite nodes with the non-positive ones
dropped, stays selectable by name; a spread given as `v` without naming that definition is refused.

```python
from sraging.engine import Heterogeneity, evaluator

ev = evaluator(c, heterogeneity=Heterogeneity(parameter="xc", sd=0.2))     # M="auto", the default
r = ev.mixture_value(p, grad=True)
r.value, r.node_record["check"]["M"], r.node_record["check"]["verdict"], r.clean
```

The likelihood is a mixture over s on the Gauss rule of the truncated measure, which is smooth in sd: no node drops
out as sd moves. The node count M is chosen per cohort, not fixed:
- **The ladder** 5, 8, 11, 17, 25, 38, 57, 85, 128, 192 climbs from a guided start until two consecutive rungs agree
  within 0.05 nats, and returns the lower one.
- **At the cap** it returns M = 192 with `HET_UNCONVERGED`, and the result is not clean.
- **A fixed M** is checked against the next rung up, unless you pass `check=False`; then the result is never clean.

On the frozen panel, at the true θ, the check passed no node count that was 0.1 nats or more off:
<!-- verify:tests/acceptance/results/hr1_004.json:verdicts[0].statistic.n_false_pass:d -->0<!-- /verify --> false
passes among <!-- verify:tests/acceptance/results/hr1_004.json:verdicts[0].statistic.n_checked_valid:d -->159<!-- /verify -->
checked cells (HR-1). The worst checked error was
<!-- verify:tests/acceptance/results/hr1_004.json:verdicts[0].statistic.max_abs_err_checked:.3f -->0.086<!-- /verify --> nats, the
Clopper–Pearson upper bound on the false-pass rate is
<!-- verify:tests/acceptance/results/hr1_004.json:verdicts[0].statistic.cp_upper_95:.1% -->1.9%<!-- /verify -->, and
<!-- verify:tests/acceptance/results/hr1_004.json:verdicts[0].statistic.n_plant_informative:d -->106<!-- /verify --> plants were
informative. The gradient in sd agrees with central differences at all
<!-- verify:tests/acceptance/results/hr2_004.json:verdicts[0].n_valid:d -->156<!-- /verify --> points (HR-2). The
gradient is correct, so a calibrator can use it; whether sd can be told apart from ε is spec 006's to measure.

The adaptive-M series, a low M while a maximiser explores and the checked M at its end, is not accepted (Naveh
2026-10-31): adaptive and full M reached maxima up to
<!-- verify:tests/acceptance/results/hr3_004.json:verdicts[0].statistic.max_abs_dll:.3g -->7.35e+04<!-- /verify --> nats
apart (HR-3), attributed to the maximiser stopping at different points, not to the quadrature. Spec 006 uses the
automatic M per evaluation.

### What a heterogeneous evaluation costs

Each rung states its CPU before it runs, M × t_hom × f: t_hom is the CPU of one homogeneous value, measured in the
same process, and f is
<!-- verify:tests/acceptance/results/p0_het_cost_004.json:f_value:.2f -->0.87<!-- /verify --> for a value and
<!-- verify:tests/acceptance/results/p0_het_cost_004.json:f_grad:.2f -->3.13<!-- /verify --> for a value with its
gradient, in homogeneous values per node (pilot P0). The statement held within a factor of 2 on all but
<!-- verify:tests/acceptance/results/hr5_004.json:verdicts[0].statistic.n_outside:d -->1<!-- /verify --> of
<!-- verify:tests/acceptance/results/hr5_004.json:verdicts[0].n_valid:d -->160<!-- /verify --> cells, with a median
measured/stated ratio of <!-- verify:tests/acceptance/results/hr5_004.json:verdicts[0].statistic.median_ratio:.2f -->1.10<!-- /verify -->
(HR-5). The cost grows linearly in M, so the ladder's start saves rungs; the result records every statement beside
the measured CPU.

The nodes can run over worker processes (`sraging.engine.pool.NodePool`). The mixture is summed in node order on the
parent, so the value and gradient bytes are those of one worker on all
<!-- verify:tests/acceptance/results/hr4_004.json:verdicts[0].n_valid:d -->90<!-- /verify --> rows tested (W 1-15, HR-4).
The wall-time speedup at W = <!-- verify:tests/acceptance/results/hr4_004.json:verdicts[0].statistic.speedup.W:d -->5<!-- /verify -->
is claimed: at least 1.9×, the upper 95 % bound on the wall-time ratio being
<!-- verify:tests/acceptance/results/hr4_004.json:verdicts[0].statistic.speedup.upper_95_ratio:.3f -->0.518<!-- /verify -->. HR-4 itself reads fail on its plant rule, which a later spec resizes (Naveh 2026-10-29).

### The Langevin engine

`sraging.langevin.simulate` draws a seeded sample of first-passage times from the SR model's stochastic equation. It
replaces SRK's samplers (`getSrHetro`, `getISR`), with numba as an optional extra (`uv sync --extra langevin`) that
the FP path never imports:

```python
from sraging.langevin import simulate

s = simulate(p, n=25_000, key=7, dt=0.03125)                  # wall="clamp", the bridge kernel, error="ladder"
s.times, s.ages, s.events, s.error.verdict, s.error.step_bias["bias_at_dt"], s.clean
r = simulate(p, n=25_000, key=7, dt=0.125, wall="reflect")    # the symmetrized wall
X = simulate(p, n=2_000, key=7, dt=0.125, wall="reflect", record_ages=[40.0, 80.0]).paths.X   # NaN iff dead
```

On the example's `p` (the ITP fit, in days) both samples read clean at n = 25,000
(`specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json`):
- **The clamp at dt 0.03125:** KS D <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[4].ks_D:.4f -->0.0062<!-- /verify --> against crit <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[4].crit_95:.4f -->0.0086<!-- /verify -->,
  step bias b̂ <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[4].bias_at_dt:.4f -->0.0076<!-- /verify -->, in <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[4].cpu_s:.0f -->21<!-- /verify --> CPU-s.
- **Reflect at dt 0.125:** D <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[3].ks_D:.4f -->0.0036<!-- /verify -->, b̂ <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[3].bias_at_dt:.4f -->0.0046<!-- /verify -->, in
  <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[3].cpu_s:.0f -->6<!-- /verify --> CPU-s.
- **The clamp at dt 0.5,** this example's step until spec 004's review, reads not clean: D
  <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].ks_D:.4f -->0.0242<!-- /verify --> and b̂ <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].bias_at_dt:.4f -->0.0132<!-- /verify --> both exceed the crit, so it carries
  <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].warnings[0]: -->LANGEVIN_INCONSISTENT_AT_N<!-- /verify --> and <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].warnings[1]: -->LANGEVIN_STEP_BIAS_ABOVE_CRIT<!-- /verify -->.

- **One step, keyed streams.** Each individual steps at dt, its age k·dt from k, on its own random stream from
  (key, i). A sample is the same bytes at any worker count and chunk size: W = 1, 7 and 15 agree at
  n = <!-- verify:tests/acceptance/results/hr8_004.json:config.n:, -->25,013<!-- /verify --> (HR-8). SRK's options
  (`nsteps`, `time_step_multiplier`, `parallel`, `seed`, ...) are refused by name.
- **The walls.** The default `wall="clamp"` is SRK's max(0, ·), kept for parity with SRK (register D-410, ported
  as is; Naveh 2026-11-05). It leaves a lump of exact zeros in young damage. `wall="reflect"` reflects at 0. The wall
  is in every sample's provenance. Choose reflect for a population near the wall (see the limits). Which wall needs
  the finer step depends on the configuration (P1's ladders, below the frozen steps): halve dt until the sample reads
  clean.
- **The error statement, at your n.** Every sample carries a KS test of its raw times against the FP survival at
  1.358/√n, and a step bias from a coupled half-step rung reported as the bias of your own sample,
  b̂(dt) = D/(1 − 2^−p̂), with p̂ = ½ for every configuration (3.41·D; Naveh 2026-11-05). b̂ is an estimate from one
  halving, not a bound:
  for the clamp at dt 0.5 above it reads <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].bias_at_dt:.4f -->0.0132<!-- /verify -->, while the sample's own KS distance
  from FP is <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].ks_D:.4f -->0.0242<!-- /verify -->. `error="ks"` skips the rung, and then the sample is not clean.
- **What `clean` certifies.** A clean sample passed the KS at its n, has an FP reference, has no survivor at t_max,
  and states a b̂(dt) no larger than its KS crit: above it, the sample carries `LANGEVIN_STEP_BIAS_ABOVE_CRIT` and is
  not clean (Naveh 2026-11-05). `clean` does not certify that the step is converged: the crit is wide at small n, and
  b̂ is not a bound. At n = 2,000 the clamp at dt 4.0 on the example's `p` passes the KS (D
  <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clean_vs_bias.json:rows[2].ks_D:.3f -->0.029<!-- /verify --> against crit <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clean_vs_bias.json:rows[2].crit_95:.3f -->0.030<!-- /verify -->) with b̂
  <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clean_vs_bias.json:rows[2].bias_at_dt:.3f -->0.109<!-- /verify -->, so its clean now reads <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clean_vs_bias.json:rows[2].clean: -->False<!-- /verify --> (`specs/004-samplers/review/evidence/sr_naveh_004/clean_vs_bias.json`). For a measured step,
  use the table below.
- **The observation.** Ages are recorded by the generator's own transform: on the δ grid, with its censoring and its
  window.
- **Paths.** `record_ages` keeps each individual's damage at those ages, read without a draw (the times are the same
  with and without). Each age is tested against the FP damage density among the living; under the clamp, above the
  lump.
- **SRK's defects are registered,** each measured against SRK's own code beside its twin (D-402 to D-411;
  `plants_004.json` reads <!-- verify:tests/acceptance/results/plants_004.json:status: -->pass<!-- /verify -->). Each
  is fixed except D-410, the clamp, which stays the default.

**The frozen steps.** The acceptance judged each preset at a step dt = Δt/m, with Δt SRK's own step for that preset
(mice_F <!-- verify:tests/acceptance/results/p1_pooled_004.json:cells.mice_F|hom|clamp.base_dt:.4f -->0.0428<!-- /verify -->, Jack Russell
<!-- verify:tests/acceptance/results/p1_pooled_004.json:cells.Jack_Russell|hom|clamp.base_dt:.4f -->0.0024<!-- /verify -->, E. coli <!-- verify:tests/acceptance/results/p1_pooled_004.json:cells.ecoli|hom|clamp.base_dt:.4f -->0.0300<!-- /verify -->,
human <!-- verify:tests/acceptance/results/p1_pooled_004.json:cells.human|hom|clamp.base_dt:.5f -->0.00314<!-- /verify -->) and m measured once at pooled n (`p1_pooled_004.json`,
`p1_pooled_b_004.json`; amendment A1; `langevin_004.toml [judged.steps]`). The CPU is one core's for one draw of
25,000 with its dt/2 rung, what the default `error="ladder"` runs (the FP reference aside), the median over its keys:

<table>
<tr><th>preset (time unit)</th><th>heterogeneity</th><th>wall</th><th>m</th><th>dt</th><th>CPU-s per draw</th></tr>
<tr><td>mice_F (weeks)</td><td>none</td><td>clamp</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|hom|clamp.m:d -->170<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|hom|clamp.dt:.2e -->2.52e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|hom|clamp.draw_cpu_s:,.0f -->421<!-- /verify --></td></tr>
<tr><td>mice_F (weeks)</td><td>none</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|hom|reflect.m:d -->81<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|hom|reflect.dt:.2e -->5.29e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|hom|reflect.draw_cpu_s:,.0f -->192<!-- /verify --></td></tr>
<tr><td>mice_F (weeks)</td><td>xc, sd 0.2</td><td>clamp</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|het|clamp.m:d -->170<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|het|clamp.dt:.2e -->2.52e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|het|clamp.draw_cpu_s:,.0f -->391<!-- /verify --></td></tr>
<tr><td>mice_F (weeks)</td><td>xc, sd 0.2</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|het|reflect.m:d -->75<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|het|reflect.dt:.2e -->5.71e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.mice_F|het|reflect.draw_cpu_s:,.0f -->173<!-- /verify --></td></tr>
<tr><td>Jack Russell (years)</td><td>none</td><td>clamp</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|hom|clamp.m:d -->261<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|hom|clamp.dt:.2e -->9.20e-06<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|hom|clamp.draw_cpu_s:,.0f -->1,674<!-- /verify --></td></tr>
<tr><td>Jack Russell (years)</td><td>none</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|hom|reflect.m:d -->335<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|hom|reflect.dt:.2e -->7.17e-06<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|hom|reflect.draw_cpu_s:,.0f -->2,147<!-- /verify --></td></tr>
<tr><td>Jack Russell (years)</td><td>xc, sd 0.2</td><td>clamp</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|het|clamp.m:d -->138<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|het|clamp.dt:.2e -->1.74e-05<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|het|clamp.draw_cpu_s:,.0f -->855<!-- /verify --></td></tr>
<tr><td>Jack Russell (years)</td><td>xc, sd 0.2</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|het|reflect.m:d -->180<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|het|reflect.dt:.2e -->1.33e-05<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.Jack_Russell|het|reflect.draw_cpu_s:,.0f -->1,063<!-- /verify --></td></tr>
<tr><td>E. coli (hours)</td><td>none</td><td>clamp</td><td colspan="3"><!-- verify:tests/acceptance/results/p1_pooled_004.json:cells.ecoli|hom|clamp.status: -->not met<!-- /verify --></td></tr>
<tr><td>E. coli (hours)</td><td>none</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|hom|reflect.m:d -->133<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|hom|reflect.dt:.2e -->2.26e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|hom|reflect.draw_cpu_s:,.0f -->250<!-- /verify --></td></tr>
<tr><td>E. coli (hours)</td><td>xc, sd 0.2</td><td>clamp</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|het|clamp.m:d -->166<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|het|clamp.dt:.2e -->1.81e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|het|clamp.draw_cpu_s:,.0f -->314<!-- /verify --></td></tr>
<tr><td>E. coli (hours)</td><td>xc, sd 0.2</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|het|reflect.m:d -->51<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|het|reflect.dt:.2e -->5.88e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|het|reflect.draw_cpu_s:,.0f -->92<!-- /verify --></td></tr>
<tr><td>human (years)</td><td>none</td><td>clamp</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|hom|clamp.m:d -->106<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|hom|clamp.dt:.2e -->2.97e-05<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|hom|clamp.draw_cpu_s:,.0f -->2,160<!-- /verify --></td></tr>
<tr><td>human (years)</td><td>none</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|hom|reflect.m:d -->16<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|hom|reflect.dt:.2e -->1.96e-04<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|hom|reflect.draw_cpu_s:,.0f -->315<!-- /verify --></td></tr>
<tr><td>human (years)</td><td>xc, sd 0.2</td><td>clamp</td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|het|clamp.m:d -->97<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|het|clamp.dt:.2e -->3.24e-05<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.human|het|clamp.draw_cpu_s:,.0f -->2,111<!-- /verify --></td></tr>
<tr><td>human (years)</td><td>xc, sd 0.2</td><td>reflect</td><td><!-- verify:tests/acceptance/results/p1_pooled_b_004.json:frozen.human|het|reflect.m:d -->36<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_b_004.json:frozen.human|het|reflect.dt:.2e -->8.73e-05<!-- /verify --></td><td><!-- verify:tests/acceptance/results/p1_pooled_b_004.json:frozen.human|het|reflect.draw_cpu_s:,.0f -->738<!-- /verify --></td></tr>
</table>

S4, HR-10 and HR-11 below hold at these steps; S5 ran at its own configuration's (`p1_pooled_b_004.json`). A coarser
step is not covered: its sample states its own b̂, and is not clean where b̂ exceeds the crit.

**Coarser than the frozen steps, neither wall is uniformly better.** P1 drew one key per rung at n = 25,000, at
dt = Δt/m (`tests/acceptance/results/p1_langevin_004.json`). Against the KS crit <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[8].ladder[2].crit_95:.4f -->0.0086<!-- /verify -->, the sample's KS distance from FP was:
- **E. coli, homogeneous, m 4:** the clamp <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[8].ladder[2].ks_D:.4f -->0.0065<!-- /verify -->, reflect <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[9].ladder[2].ks_D:.4f -->0.0345<!-- /verify -->;
- **the human, homogeneous, m 2:** the clamp <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[12].ladder[1].ks_D:.4f -->0.0073<!-- /verify -->, reflect
  <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[13].ladder[1].ks_D:.4f -->0.0221<!-- /verify -->;
- **Jack Russell, homogeneous, m 16:** the clamp <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[4].ladder[4].ks_D:.4f -->0.0399<!-- /verify -->, reflect
  <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[5].ladder[4].ks_D:.4f -->0.0388<!-- /verify -->, about equal and both far off;
- **mice_F, homogeneous, m 2:** the clamp <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[0].ladder[1].ks_D:.4f -->0.0109<!-- /verify -->, reflect <!-- verify:tests/acceptance/results/p1_langevin_004.json:cells[1].ladder[1].ks_D:.4f -->0.0045<!-- /verify -->; on
  the example's `p` reflect is also the closer (above).

So at a coarse step, halve dt until the sample reads clean, or use a frozen step. Reflect's advantage is near the wall
at a converged step: E. coli's homogeneous reflect met its bar at m <!-- verify:tests/acceptance/results/p1_pooled_004.json:frozen.ecoli|hom|reflect.m:d -->133<!-- /verify -->, its
clamp at no step tried.

How well it agrees with FP:
- **S4 (HR-6):** KS against FP passes on all
  <!-- verify:tests/acceptance/results/s4_004.json:verdicts[0].n_valid:d -->46<!-- /verify --> clamp rows and all
  <!-- verify:tests/acceptance/results/s4_004.json:verdicts[1].n_valid:d -->8<!-- /verify --> reflect rows, each at its
  frozen step (`specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json`).
  - **Coverage.** <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.clamp.borrowed:d -->42<!-- /verify --> of the clamp rows (η, β, ε, κ, and xc at cv 0.1 and 0.3) ran
    at their preset's xc sd 0.2 step, borrowed. The homogeneous clamp is one row, <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.clamp.homogeneous[0]: -->Jack_Russell/v0/clamp<!-- /verify -->.
    The <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.reflect.rows:d -->8<!-- /verify --> reflect rows are each preset's homogeneous row and its xc sd 0.2 row: no
    η, β, ε or κ row, and no HR-11 cell, ran on reflect.
  - **Resolution.** At n = 25,000 the crit is <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.clamp.crit:.4f -->0.0123<!-- /verify --> on the clamp (Bonferroni over
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.clamp.G:d -->46<!-- /verify --> rows) and <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.reflect.crit:.4f -->0.0107<!-- /verify --> on reflect, so a pass rules out
    a sup|ΔS| above about that. The plants, shifted by δ <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:plants[0].delta:.3f -->0.012<!-- /verify --> and
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:plants[1].delta:.3f -->0.022<!-- /verify -->, read D <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:plants[0].ks_D:.3f -->0.036<!-- /verify --> and
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:plants[1].ks_D:.3f -->0.036<!-- /verify -->.
  - **The aggregate.** No offset hides under the Bonferroni: the clamp rows' mean √n·D is
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.clamp.mean_sqrt_n_D:.3f -->0.843<!-- /verify --> against <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:null_mean_sqrt_n_D:.3f -->0.869<!-- /verify --> under the null (its asymptotic mean), and
    Fisher's combined p is <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.clamp.fisher_p:.2f -->0.77<!-- /verify --> (clamp) and
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s4_aggregate.json:walls.reflect.fisher_p:.2f -->0.69<!-- /verify --> (reflect).
- **Paths (HR-10):** the damage at the recorded ages passes on
  <!-- verify:tests/acceptance/results/paths_004.json:verdicts[0].n_valid:d -->11<!-- /verify --> clamp tests and
  <!-- verify:tests/acceptance/results/paths_004.json:verdicts[1].n_valid:d -->16<!-- /verify --> reflect tests.
- **Against the generator's heterogeneous law (S5, HR-7):** the score test at
  R = <!-- verify:tests/acceptance/results/s5_het_004.json:s5.R:d -->274<!-- /verify --> cohorts reads
  <!-- verify:tests/acceptance/results/s5_het_004.json:status: -->pass<!-- /verify -->: the planted arm crosses zcrit
  <!-- verify:tests/acceptance/results/s5_het_004.json:verdicts[0].statistic.zcrit:.3f -->2.498<!-- /verify --> (in ρ_η, z <!-- verify:tests/acceptance/results/s5_het_004.json:s5.plant.z.rho_eta:.2f -->2.76<!-- /verify -->) and the twin does
  not (its largest, in ε, z <!-- verify:tests/acceptance/results/s5_het_004.json:s5.twin.z.rho_eps:.2f -->2.09<!-- /verify -->).
  - **Its margins are gen-1's, not the Langevin's.** At S5's configuration gen-1's 161-node law is
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json:kl_per_individual:.2e -->4.70e-06<!-- /verify --> nats per individual from the converged FP mixture,
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json:kl_over_plant_pct:.0f -->25<!-- /verify --> % of the plant's KL. At the pooled n that allows a score offset of up to
    about <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json:max_offset_se:.1f -->2.3<!-- /verify --> SE, the size of the twin's (`specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json`). Inferred, not measured [I]:
    through the score's local quadratic approximation (a shift grows like √KL), S5 separates the Langevin from gen-1
    only to about half the planted shift (√ of that KL ratio,
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json:sqrt_kl_over_plant:.2f -->0.50<!-- /verify -->).
  - Its pooled KS is reported, not gating: D
  <!-- verify:tests/acceptance/results/s5_het_004.json:s5.ks.D:.2g -->0.0019<!-- /verify -->, p
  <!-- verify:tests/acceptance/results/s5_het_004.json:s5.ks.p_value:.2g -->0.033<!-- /verify -->, n
  <!-- verify:tests/acceptance/results/s5_het_004.json:s5.ks.n:, -->548,000<!-- /verify -->.
- **The homogeneous half (S5, spec 002's score test on this sampler)** passes under spec 002's gate condition (d)
  (Naveh 2026-11-05): pure, hazard, window and entry pass, and the heterogeneity plant, a wrong conditioning in the FP mixture's score,
  is "<!-- verify:tests/acceptance/results/s5_hom_004.json:features.heterogeneity.verdict: -->not informative at this R<!-- /verify -->"
  at R <!-- verify:tests/acceptance/results/s5_hom_004.json:features.heterogeneity.R:d -->200<!-- /verify -->, as in
  spec 002. That check stays Not Covered and goes to spec 006.

### The Langevin's limits

- **The clamp near the wall.** The clamp's error at the wall can reach the survival. Two measured cases show it, with
  different parameters, so saturated removal is not the whole story [I].
  - E. coli's homogeneous clamp met no step in P1 (its status in `p1_pooled_004.json` reads
    <!-- verify:tests/acceptance/results/p1_pooled_004.json:cells.ecoli|hom|clamp.status: -->not met<!-- /verify -->): the clamp is not
    shown to agree with FP for E. coli at any affordable step (Naveh 2026-11-03), and E. coli is judged on
    `wall="reflect"` only. E. coli's removal is not saturated: its κ is above ε/β.
  - The example's `p`, whose removal is saturated (κ far below ε/β), so the living's damage sits near the wall [I].
    At n = 25,000 the clamp fails the KS at dt 0.5 and 0.125 (D
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].ks_D:.4f -->0.0242<!-- /verify --> and <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[2].ks_D:.4f -->0.0157<!-- /verify --> against crit
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].crit_95:.4f -->0.0086<!-- /verify -->) and passes at 0.03125, while reflect passes at both 0.5 and 0.125 (D
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[1].ks_D:.4f -->0.0072<!-- /verify --> and <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[3].ks_D:.4f -->0.0036<!-- /verify -->).
  - sr-naveh reads the regime as β√(2dt/ε): here <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[0].beta_sqrt_2dt_over_eps:.2f -->0.65<!-- /verify --> and
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[2].beta_sqrt_2dt_over_eps:.2f -->0.33<!-- /verify --> where the clamp failed,
    <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/clamp_vs_reflect.json:rows[4].beta_sqrt_2dt_over_eps:.2f -->0.16<!-- /verify --> where it passed. That is a reading of the physics, not a
    measured law.
  - For such a population choose `wall="reflect"`, or a step from the table above.
- **Damage under the clamp:** below the clamp's censoring point (its lump of zeros) the damage is not validated, and
  near the wall it is off above the lump too.
  - Jack Russell's heterogeneous damage at its first age,
    <!-- verify:tests/acceptance/results/paths_004.json:configurations[9].tests[0].age:.1f -->11.8<!-- /verify --> years, is judged on reflect
    only (amendment A2).
  - A development check (one synthetic configuration, n 20,000) found the clamp's damage CDF above the lump off by
    an amount that shrinks like √dt, while reflect converged (spec.md Clarifications, 2026-10-30).
  - The sample reports the lump and warns. For the paths of a population near the wall use `wall="reflect"`, which has
    no lump.
- **No age segments:** the Langevin takes constant parameters, neither a parameter changed on an age window nor the
  DPU accumulation rate A; `segments` is refused by name.

### Several heterogeneous parameters

A tuple of `Heterogeneity` objects, one per parameter, each an independent truncated normal, is the Langevin's only:
FP's likelihood and curves refuse it.
- **Each parameter draws from its own stream,** so a member at sd = 0 (`Heterogeneity.zero(parameter)`) is s = 1
  exactly, and a tuple whose other spreads are 0 is the single-parameter draw, byte for byte.
- **The error statement's reference** is a tensor product of the per-dimension Gauss rules, for two or three spreads,
  within 20,000 nodes. With four or more spreads, or above the budget, the statement reads "no FP reference" and the
  sample is not clean.
- **Multi-spread paths are recorded, not checked** (register D-413): spec 004 builds no tensor damage density.
- **Checked on six two- and three-parameter cells** (HR-11, mice_F, sd 0.2 each, n 25,000): every cell is below
  its critical value, the largest D <!-- verify:tests/acceptance/results/multi_004.json:cells[4].D:.3g -->0.00709<!-- /verify --> (xc+η+ε) against crit <!-- verify:tests/acceptance/results/multi_004.json:cells[4].crit:.4g -->0.01047<!-- /verify -->;
  the reference at its own 1e-4 bar agrees with an independent composite rule to <!-- verify:tests/acceptance/results/multi_004.json:composite.sup_dS:.2g -->2.6e-05<!-- /verify -->.
  The cells were judged against references built at crit/10; xc+η's (M 8×5) is
  <!-- verify:specs/004-samplers/review/evidence/multi_004_first_252ad3a.json:composite.sup_dS:.2g -->0.00063<!-- /verify -->
  from the converged one (amendment A9 (a)), well inside that cell's margin below its crit.
  This is the pre-declared re-run at fresh keys (amendment A9). The first run failed one cell (xc+η, D
  <!-- verify:specs/004-samplers/review/evidence/multi_004_first_252ad3a.json:cells[0].D:.4g -->0.01112<!-- /verify -->), diagnosed as chance, about a 2 % event for a correct sampler
  (`specs/004-samplers/review/evidence/multi_diag_2026-11-05/`).

### `engine=`: FP, the generator or the Langevin

`sraging.population.population(params, ages, heterogeneity=, engine=, n=, key=, delta=, dt=, wall=)` returns the
survival at `ages` and says which engine made it:

| engine | n = None | n = an integer |
|---|---|---|
| `"fp"` | the FP curves | the generator inside its domain; outside it, refused naming why |
| `"langevin"` | refused: "pass n" | `simulate` |
| `"auto"` | the FP curves (a tuple: "pass n") | the generator inside its domain; outside it the Langevin, with `ENGINE_AUTO_LANGEVIN` naming the reason |

The generator's domain is xc or no heterogeneity, n ≤ 20,000 when heterogeneous, and a valid cohort spec. The
boundary is that domain, not the cost. Pilot P3 timed both engines on the human preset with xc sd 0.2, the Langevin
at the preset's frozen heterogeneous clamp step (dt <!-- verify:tests/acceptance/results/p3_cost_004.json:dt:.2e -->3.24e-05<!-- /verify -->, with its rung), in one process, one timing
each:
- **n = 2,000:** the generator's heterogeneous draw took
  <!-- verify:tests/acceptance/results/p3_cost_004.json:boundary.rows[3].cpu_gen1_s:.0f -->690<!-- /verify --> CPU-s,
  the Langevin's
  <!-- verify:tests/acceptance/results/p3_cost_004.json:boundary.rows[3].cpu_langevin_s:.0f -->127<!-- /verify -->.
- **n = 20,000:** the generator <!-- verify:tests/acceptance/results/p3_cost_004.json:boundary.rows[4].cpu_gen1_s:,.0f -->691<!-- /verify -->, the Langevin
  <!-- verify:tests/acceptance/results/p3_cost_004.json:boundary.rows[4].cpu_langevin_s:,.0f -->1,254<!-- /verify -->.

The generator is exact up to its stated grid error, so `auto` keeps it wherever it can draw.

### The generator's heterogeneous cohorts above n = 20,000

FR-009 compared the generator's 161-node heterogeneous law with the converged mixture on the panel's
<!-- verify:tests/acceptance/results/hr1_004.json:verdicts[1].n:d -->40<!-- /verify --> cohorts. It is
within 0.1 nats on every cohort at n ≤ 20,000; the misses, up to
<!-- verify:tests/acceptance/results/hr1_004.json:verdicts[1].max_abs_diff:.2f -->0.54<!-- /verify --> nats, are all
above n = 20,000 (register D-412; Naveh 2026-10-29). v6 heterogeneous cohorts above n = 20,000 carry this limit, beside
the generator's own caveat there.

The limit also covers an analysis that pools replicate heterogeneous cohorts past n = 20,000 (Naveh 2026-11-05). At
S5's configuration gen-1's law is <!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json:kl_per_individual:.2e -->4.70e-06<!-- /verify --> nats per individual from the converged mixture:
<!-- verify:specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json:kl_at_20000_nats:.3f -->0.094<!-- /verify --> nats per 20,000 pooled individuals, already near FR-009's 0.1-nat bar
(`specs/004-samplers/review/evidence/sr_naveh_004/s5_binned_kl.json`).

## Data: cohort sets, units, survival and cleaning (spec 005)

These modules are imported as `sraging.<module>`; none adds a name to `import sraging`. Plots need
`uv sync --extra plots`. Every function takes its options by keyword and refuses an unknown one by name.

### Cohort sets

`read_cohorts` reads one long CSV, or a directory of CSVs (one cohort per file, named by its stem), into a `CohortSet`: named
cohorts plus a design table: `cohort`, `cage` (with `cage_col`) and every other column that is constant within each
member. It also returns the rows it could not
use; nothing is skipped silently (on the command line a refused row exits 3).

```python
from sraging.cohorts import read_cohorts
cages, refused = read_cohorts("cages.csv", delta=1, time_unit="days", cohort_col="condition", cage_col="cage")
assert not refused
sub = cages.select(cohort="LVP_M")          # exact, typed match on a design column; an unknown column is refused
groups = cages.by("cohort")                 # {value: CohortSet}
one = cages["LVP_M/cage_3"]                 # a Cohort; keys are "cohort" or "cohort/cage"
```

The age and event columns are found by header name (`age`, `death times`; `event`, `events`). A censoring at age 0 is refused
by `Cohort.from_arrays` (it carries no information; unchanged since spec 001). Weighted support rows, where used, start
their censorings at day 1 ([T-ENGINE-XC-PROGRAM-CACHE]).

### Units

`sraging.units` converts parameters and cohorts between hours, days, weeks, years (365 days) and generations (1/8 day,
SRK's fixed 3-hour convention for its yeast and E. coli presets, not a property of your organism). `convert` refuses
months and any other unit; a `time_unit` given when reading is a label, checked only when you convert. Nothing converts
unless you call `convert`; an object read without a unit says `input units` and must be named first with `with_unit`.

```python
from sraging import presets, units
p, info = presets.get("mice_F")                 # in days
p_weeks = units.convert(p, to="weeks")          # eta x 49, beta, epsilon, h_ext x 7; kappa and xc unchanged
weekly = units.convert(cages["LVP_M/cage_3"], to="weeks")
```

The factor table is checked against SRK's `getTheta` on <!-- verify:tests/acceptance/results/units_005.json:factors.cells:d -->175<!-- /verify --> cells (largest relative difference <!-- verify:tests/acceptance/results/units_005.json:factors.max_rel:.1e -->2.2e-16<!-- /verify -->), round trips return the
parameters to <!-- verify:tests/acceptance/results/units_005.json:round_trips.params_rel:.1e -->1.4e-16<!-- /verify --> and the cohort to <!-- verify:tests/acceptance/results/units_005.json:round_trips.cohort_rel:.1e -->2.7e-16<!-- /verify -->, and converting a δ cohort (deaths recorded on a grid) and its parameters together changes the log-likelihood by at most
<!-- verify:tests/acceptance/results/units_005.json:worst_abs_dloglik:.2e -->9.55e-12<!-- /verify --> (<!-- verify:tests/acceptance/results/units_005.json:loglik_tolerance:.2e -->9.55e-11<!-- /verify --> tolerance, a regression lock
set from pilot P3, which ran the same computation). A continuous cohort's log-likelihood is a density and shifts by
n_deaths · ln s; θ̂ and likelihood ratios are unaffected, but never compare log-likelihoods or AIC across units. Algorithm: [`units`](docs/algorithms/main.pdf), section "Time units".

### Presets

`presets.get(name, source="legacy", type="mode_overall", ExtH=False)` returns `(SRParams in days, PresetInfo)` for one of SRK's
legacy presets (`presets.names()` lists them); `type` is `"mode_overall"`, `"mode"` or `"max_likelihood"`. Any other `source` is
refused. SRK's `getTheta` returned humans, dogs and cats in years, mice in weeks, E. coli in hours and yeast in
generations by default; `presets.get` always returns days, so a human preset's η is 365² times SRK's default number.
κ = 0.5/xc restates SRK's parametrisation (`info.kappa_rule`) and is not a fitted κ. With `ExtH=True`, `h_ext` is the
preset's external hazard per day: the table stores −ln h in the preset's own unit (years for every preset that has
one), and `info.ext_h_unit` names that unit. The parameters equal `getTheta` on all <!-- verify:tests/acceptance/results/presets_005.json:cells:d -->1050<!-- /verify --> cells (every preset, type, ExtH and unit; largest relative difference <!-- verify:tests/acceptance/results/presets_005.json:max_rel:.1e -->2.2e-16<!-- /verify -->), and `h_ext` converted back to the preset's own unit equals e^(−ExtH) on all <!-- verify:tests/acceptance/results/presets_005.json:h_ext_cells:d -->105<!-- /verify --> cells (<!-- verify:tests/acceptance/results/presets_005.json:h_ext_finite:d -->48<!-- /verify --> with an ExtH; largest relative difference <!-- verify:tests/acceptance/results/presets_005.json:h_ext_max_rel:.1e -->1.5e-16<!-- /verify -->).
`sraging.terms.per_individual(cohort, params)` returns each row's log-likelihood term, summing to the engine's value.

### The Kaplan–Meier estimator and the statistics

```python
from sraging.survival import km, summary
k = km(cages["LVP_M/cage_3"])                   # deaths first at a tie; 95 % log(-log) Greenwood band
s = summary(k)                                  # quartiles="first-crossing" by default; "nearest" is SRK Dataset's
s.median.value, s.median.definition, s.median.ci
s.steepness.value, s.steepness.definition
```

The median is the first age where Ŝ ≤ 0.5. Steepness is median / (t(0.25) − t(0.75)), with the quartile ages first-crossing
or `nearest` (SRK Dataset's `getSteepness`), and every statistic carries its definition as text. The mean and CV are restricted to the
last observed age unless Ŝ reaches 0. `condition_on=t0` gives the KM given survival to `t0`. Against lifelines 0.30.3, side by
side on the cages of the mosquito and fly records and on synthetic ties, the largest |ΔŜ| is <!-- verify:tests/acceptance/results/km_005.json:cells[0].max_dS:g -->0<!-- /verify -->, <!-- verify:tests/acceptance/results/km_005.json:cells[1].max_dS:g -->0<!-- /verify --> and <!-- verify:tests/acceptance/results/km_005.json:cells[2].max_dS:g -->0<!-- /verify --> (synthetic ties, mosquito, fly), the largest band
difference <!-- verify:tests/acceptance/results/km_005.json:cells[0].max_dband:g -->0<!-- /verify -->, <!-- verify:tests/acceptance/results/km_005.json:cells[1].max_dband:g -->0<!-- /verify --> and <!-- verify:tests/acceptance/results/km_005.json:cells[2].max_dband:g -->0<!-- /verify -->, and the medians mismatch on <!-- verify:tests/acceptance/results/km_005.json:cells[0].median_mismatches:d -->0<!-- /verify -->, <!-- verify:tests/acceptance/results/km_005.json:cells[1].median_mismatches:d -->0<!-- /verify --> and <!-- verify:tests/acceptance/results/km_005.json:cells[2].median_mismatches:d -->0<!-- /verify --> cohorts; a planted censoring-before-death tie rule is caught (Ŝ(2) = <!-- verify:tests/acceptance/results/km_005.json:planted.planted:.3f -->0.533<!-- /verify --> against <!-- verify:tests/acceptance/results/km_005.json:planted.ours:.3f -->0.600<!-- /verify -->). HR-3 compares Ŝ, the band and
the median only, and Ŝ equals lifelines' exactly because `km` uses lifelines' arithmetic, so the comparison locks that
arithmetic. In a development check (not an acceptance rule) the restricted mean matched lifelines'
`restricted_mean_survival_time` to <!-- verify:specs/005-data-plots-cli/review/evidence/devchecks_005.json:n5.rmst_rel:.1e -->6.4e-16<!-- /verify --> relative and the median's CI matched on every cage of the two records
(<!-- verify:specs/005-data-plots-cli/review/evidence/devchecks_005.json:n5.ci_mismatch:d -->0<!-- /verify --> mismatches); the quartile ages, CV and the conditional statistics have no reference. Algorithm: section
"The Kaplan–Meier estimator".

### Cleaning cages

The lab's MAD rule: each cage is scored by its KM median and steepness (`nearest` quartiles, as the records use), and judged against
the other cages of its cohort (leave-one-out, with more than two cages) by an axis-aligned distance in units of
the radius max(k·1.4826·MAD, k·φ·|centre|); a cage is **flagged** when that distance exceeds 1. The floors φ are pooled over cohorts
with at least 10 cages, or given. `k` and `seed` have no default, and **nothing is removed unless you call `drop`**.

```python
from sraging.cleaning import clean, drop
report = clean(cages, by="cohort", k=2.5, seed=1)       # flags only; axes="median" judges on the median alone
                                                        # spread_by="diet": pool the own spread per diet (below)
report.flagged                                          # the keys of the flagged cages
kept = drop(cages, report)                              # removes exactly those, naming the report's sha256
```

```bash
.venv/bin/sraging clean --csv cages.csv --cohort-col condition --cage-col cage --delta 1 --time-unit days \
    --k 2.5 --seed 1 --out cleaning/ [--spread-by diet] [--drop]
```

This writes `cleaning.csv` (one row per cage) and `cleaning.json` (per-cohort rows, floors, definitions, provenance), and
`cleaned/<cohort>.csv` only with `--drop`. A cage with a non-finite score on an axis the rule uses is refused, naming it (the old code kept
it at distance 0).

**How to read the noise baseline.** Even a cohort with no real outlier flags some cages, so each cohort row carries the flags
the same rule gives exchangeable cages whose scores scatter (Gaussian around 1, independently on the two axes) under two
named nulls: **floor**, scatter the size of the pooled floor φ, and **own spread**, scatter the size of the leave-one-out
corrected spread pooled over the cohorts that share `spread_by`'s value (every cohort of the set by default). It is pooled
because a cohort of three cages cannot measure its own spread: two tied cages give it zero. Neither null depends on the
number of animals per cage. Compare `n_flagged` with the two `expected_flags`: each is the mean number of flags of that
scatter in a cohort of that size, not a significance test. How many flags read as noise depends on the null, so both are always shown, never one. `clean`'s `noise_reps` (default 4000) sets the Monte Carlo size; the CLI uses the default.

The code reproduces the lab's stored cleaning records, the mosquito record (at k = 2.5) and the fly record (at k = 2.5
and 3): every verdict identical and the largest distance difference <!-- verify:tests/acceptance/results/exit_005.json:records[0].per_k[0].max_abs_distance:g -->0<!-- /verify --> (mosquito), <!-- verify:tests/acceptance/results/exit_005.json:records[1].per_k[0].max_abs_distance:g -->0<!-- /verify --> and <!-- verify:tests/acceptance/results/exit_005.json:records[1].per_k[1].max_abs_distance:g -->0<!-- /verify -->
(fly, k = 2.5 and 3). Planted defects fail
first: without leave-one-out, or without floors, the mosquito record's flags differ from the record's. The noise
baseline matches the fly redo's null tables within 4 binomial SE on all its floor cells and all its groups, its own-spread null pooled per group
as here (phase status <!-- verify:tests/acceptance/results/noise_005.json:status: -->pass<!-- /verify -->; for m = 3, k = 2.5: <!-- verify:tests/acceptance/results/noise_005.json:floor_cells[0].ours:.3f -->0.091<!-- /verify --> against <!-- verify:tests/acceptance/results/noise_005.json:floor_cells[0].theirs:.3f -->0.090<!-- /verify -->). That is a replay of pilot P2 with the
same seed; with another seed the floor cells stay within <!-- verify:specs/005-data-plots-cli/review/evidence/devchecks_005.json:n2_floor_seed1.max_z:.2f -->2.28<!-- /verify --> SE (development check). A rule without floors
flags <!-- verify:tests/acceptance/results/noise_005.json:planted[1].ours:.3f -->0.514<!-- /verify --> of such cages against <!-- verify:tests/acceptance/results/noise_005.json:planted[1].theirs:.3f -->0.090<!-- /verify -->. **The limits:**
- The cleaning is tested against stored records, not side by side against the old code, so a defect shared by a record
  and the code that wrote it would not show.
- The records hold small cohorts of cages judged on both axes with pooled floors. Two or fewer cages, `axes="median"`,
  given floors and `min_cages` other than 10 are tested on synthetic data only.
- The fly record's `kept` column predates the fly redo's final choices (its D19: the median alone for the 2-8 LONG
  cohorts, flag-only for SWITCH), so those choices were not reproduced.
- The mosquito record's scores come from cages already shifted (its D2) and spike-censored (D8): raw census data
  cleaned with `clean` will not give its verdicts.
- Verdicts near the radius are fragile: removing one animal from a cage flips its own verdict for a few cages of each
  record (gate C11, `docs/planning/spec005_c11_sr-naveh.md`; not re-measured here).
Algorithm: section "MAD cage cleaning and its noise baseline".

### Plots

`sraging.plots` draws in the lab's plot style and takes result objects: `survival(km_or_cohort)`, `hazard(cohort, bins=...)`,
`death_times(cohort, bins=...)`, `set_survival(cohort_set)`, `curves(params, ages, n=None, seed=None, delta=None)`,
`cleaning(report, cohort=...)`, `k_sweep(cage_set, by=..., ks=..., floors=...)` and `overlay(result, cohort)`. Each returns `(fig, data)`: the figure and the arrays
drawn with their provenance, so a figure can be read back. `curves` with `n=<int>` draws a sample and then needs `seed` and `delta`.
Without matplotlib every call is refused, naming the `plots` extra. `overlay` (spec 006) draws the cohort's KM with its band
against the FP survival curve at both ends of the fit's κ range (`diagnostics["kappa_ends"]`), each end labelled with its κ
and whether it sits at the box. It never refits, so a result written before spec 006, which has no `kappa_ends`, is refused.

```python
from sraging import plots
fig, data = plots.survival(km(cages["LVP_M/cage_3"]))
fig, data = plots.cleaning(report, cohort="LVP_M")
fig, data = plots.overlay(result, cohort)             # result = sraging.fit_cohort(cohort, ...)
```

## Is it really the reference?

Bit-identity with the frozen v0.3.3 oracle (`tests/oracle/v033`) is tested side by side — port and oracle in
the same process, on the same host, in the pinned environment — and only there:

```bash
uv sync --group dev --group oracle
.venv/bin/pytest                 # fast suite; parity checks report NOT APPLICABLE (never "pass") elsewhere
```

Tests that read the lab benchmark's data (the tester's library and records) or the other lab checkouts find them
through environment variables, and skip as not applicable when those are unset (release 0.1.1). `SRAGING_CODE5` names
the folder holding the checkouts (`sr_calibration_tester`, `sr_calibration_tester_jax2` and its freeze,
`SRtools_kappa`); `SRAGING_TESTER_ROOT`, `SRAGING_TESTER_MAIN`, `SRAGING_TESTER_V6`, `SRAGING_J2_ROOT`,
`SRAGING_J2F_ROOT` and `SRAGING_SRK_ROOT` override single checkouts.

The slow acceptance run (side by side, on that host) passed on
<!-- verify:tests/acceptance/results/claims.json:checks.claims_identity.passed:d -->40<!-- /verify --> of
<!-- verify:tests/acceptance/results/claims.json:checks.claims_identity.total:d -->40<!-- /verify --> panel
cohorts: identical claims, identical likelihood values and gradients at six θ points each, identical cohort
arrays and warnings. Through the lab benchmark's own scoring, the port's records equal the stored v0.3.3
records on <!-- verify:tests/acceptance/results/adapter.json:checks.adapter_claims_vs_stored.passed:d -->30<!-- /verify -->
of <!-- verify:tests/acceptance/results/adapter.json:checks.adapter_claims_vs_stored.total:d -->30<!-- /verify -->
quick-tier cohorts. How to rerun it: [quickstart](specs/001-reference-fit/quickstart.md); the results:
`tests/acceptance/results/`.

**The 1.0.1 port against the frozen 1.0.1 code** (spec 003b, side by side on this Mac,
`tests/acceptance/results/*_003b.json`). 1.0.1 with its new switches off equals the 1.0.0 port in result, claims
and self-consistency block on <!-- verify:tests/acceptance/results/identity_off_003b.json:equal:d -->11<!-- /verify --> of <!-- verify:tests/acceptance/results/identity_off_003b.json:lines:d -->11<!-- /verify --> of jax2's own identity lines (its G1; on this Mac the basin re-fit fired
on its `_h` line, not on its J line). At its own settings the port equals the frozen 1.0.1 code on <!-- verify:tests/acceptance/results/parity_003b.json:done:d -->7<!-- /verify --> of
<!-- verify:tests/acceptance/results/parity_003b.json:pairs:d -->7<!-- /verify --> pairs chosen by mechanism: 5 compared as the fit and as the benchmark's record, 2 at the maximum-likelihood
stage; 3 inside the new time step's scope, one of them a full fit. 1.0.1's grid equals jax2's census on
<!-- verify:tests/acceptance/results/census_003b.json:equal:, -->2,798<!-- /verify --> of <!-- verify:tests/acceptance/results/census_003b.json:rows:, -->2,798<!-- /verify --> library rows. Whether 1.0.1 should replace v0.3.3 as the default is spec 003g's
comparison with the benchmark's v6 reference.

**Against the benchmark's current reference.** The lab benchmark's reference is now `mle_profile_jax2` 1.0.0 on
its library v5. sraging scores against it with the benchmark's own comparison code, read from a clone pinned at the
commit the benchmark named. `sraging.adapters.srbench.compare_v5(..., library="v5", pins=...)` reads each reference
run file and record once and checks it against a recorded sha256 (a run's keys must be the pinned keys). It refuses a
missing library, a run labelled with another library, the `_h` port off v2, claims computed on another library for
cohorts whose arrays differ, and a clone at another commit (`tests/acceptance/results/refusals_003e.json`; the pinned
clone, the pins and how to rerun it: [quickstart](specs/003e-library-move/quickstart.md)). Through it
the benchmark's own v5 score file is reproduced line for line
(<!-- verify:tests/acceptance/results/known_answer_003e.json:lines_compared:d -->47<!-- /verify --> lines,
<!-- verify:tests/acceptance/results/known_answer_003e.json:paired:,d -->1,776<!-- /verify --> paired cohorts). The
v0.3.3 port's stored records (the ones spec 001 matched; v5 reuses them, since these arrays are unchanged) equal the
benchmark's v0.3.3 v5 records on
<!-- verify:tests/acceptance/results/v033_003e.json:claims_equal:d -->22<!-- /verify --> of
<!-- verify:tests/acceptance/results/v033_003e.json:cohorts:d -->22<!-- /verify --> quick cohorts, with
<!-- verify:tests/acceptance/results/v033_003e.json:run_manifest.n_computed:d -->0<!-- /verify --> calibrations.

**The 1.0.0 port (`profile_jax2@1.0.0`) against that reference, without a fit** (`tests/acceptance/results/jcross_003e.json`), on
<!-- verify:tests/acceptance/results/jcross_003e.json:mac_22.cohorts:d -->22<!-- /verify --> cohorts fitted on this
Mac and <!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.cohorts:d -->118<!-- /verify --> on the
cluster: pure, censored and non-human real cohorts only (no hazard cohort, no human life table, no age-0 cohort).
Claim shapes agree on all of them, and for every η, β and ε claim the truth is inside on both sides or outside on
both (<!-- verify:tests/acceptance/results/jcross_003e.json:coverage_hits_all.flips:d -->0<!-- /verify --> of
<!-- verify:tests/acceptance/results/jcross_003e.json:coverage_hits_all.claims:d -->419<!-- /verify --> change).
Digits agree on <!-- verify:tests/acceptance/results/jcross_003e.json:mac_22.by_arm.all.bit_equal:d -->0<!-- /verify -->
of the Mac's and <!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.by_arm.all.bit_equal:d -->38<!-- /verify -->
of the cluster's. "Computed on the cluster" does not mean "the same machine": the cluster mixes CPU generations and
the reference records name no host, so which rows match cannot be attributed -- the day a reference record was
written, the port's node type and the cohort change together (on the same EPYC 7702 nodes, <!-- verify:tests/acceptance/results/jcross_003e.json:bit_equal_by_batch_and_cpu.table[0].bit_equal:d -->5<!-- /verify -->
of <!-- verify:tests/acceptance/results/jcross_003e.json:bit_equal_by_batch_and_cpu.table[0].of:d -->12<!-- /verify -->
against one day's records,
<!-- verify:tests/acceptance/results/jcross_003e.json:bit_equal_by_batch_and_cpu.table[2].bit_equal:d -->3<!-- /verify -->
of <!-- verify:tests/acceptance/results/jcross_003e.json:bit_equal_by_batch_and_cpu.table[2].of:d -->60<!-- /verify -->
against the next day's, but the two days hold different cohorts). Interval ends move by a median of
<!-- verify:tests/acceptance/results/jcross_003e.json:mac_22.by_arm.all.ends_dex.median:.0e -->7e-08<!-- /verify -->
dex (Mac) and <!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.by_arm.all.ends_dex.median:.0e -->4e-09<!-- /verify -->
dex (cluster);
<!-- verify:tests/acceptance/results/jcross_003e.json:mac_22.n_ends_above_0_03:d -->1<!-- /verify --> of
<!-- verify:tests/acceptance/results/jcross_003e.json:mac_22.n_ends:d -->111<!-- /verify --> and
<!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.n_ends_above_0_03:d -->5<!-- /verify --> of
<!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.n_ends:d -->634<!-- /verify --> ends move by more
than 0.03 dex: β's one-sided caps, and the ends of two wide ε intervals. Closed-claim points move by more than
0.03 dex on <!-- verify:tests/acceptance/results/jcross_003e.json:mac_22.by_arm.all.closed_points_dex.n_above_0_03:d -->0<!-- /verify -->
of <!-- verify:tests/acceptance/results/jcross_003e.json:mac_22.by_arm.all.closed_points_dex.n:d -->45<!-- /verify -->
(Mac) and <!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.by_arm.all.closed_points_dex.n_above_0_03:d -->6<!-- /verify -->
of <!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.by_arm.all.closed_points_dex.n:d -->281<!-- /verify -->
(cluster), by up to <!-- verify:tests/acceptance/results/jcross_003e.json:wexac_118.by_arm.all.closed_points_dex.max:.2f -->0.21<!-- /verify -->
dex; on pure-k0.01-p17-c0-s0 the two fits stop at opposite ends of the same β–κ ridge
([review](specs/003e-library-move/review/sr-naveh_003e.md)): **compare shapes and intervals, not points.** The reference itself has only a lower bound for β on
<!-- verify:tests/acceptance/results/known_answer_003e.json:reference_composition.shapes.beta.lower_bound:d -->544<!-- /verify -->
of its 1,776 cohorts (ε
<!-- verify:tests/acceptance/results/known_answer_003e.json:reference_composition.shapes.epsilon.lower_bound:d -->141<!-- /verify -->,
η <!-- verify:tests/acceptance/results/known_answer_003e.json:reference_composition.shapes.eta.lower_bound:d -->79<!-- /verify -->),
so there is no upper end to compare there. On human life tables, jax2's own report found claim shapes change between
machines on <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:hosts.human_shape_changes:d -->3<!-- /verify -->
of <!-- verify:specs/003-calibrator/review/evidence/jax2_reference_notes.json:hosts.human_records:d -->4<!-- /verify --> records (see "Choosing a calibrator").

**Results depend on the machine.** v0.3.3's results are a function of the cohort *and* the host. On the
acceptance panel, its stored results from the lab's Linux cluster differ from this Mac's on
<!-- verify:specs/001-reference-fit/review/measurements.json:host.n_differ:d -->10<!-- /verify --> of <!-- verify:specs/001-reference-fit/review/measurements.json:host.n:d -->10<!-- /verify --> named cohorts: on the
<!-- verify:specs/001-reference-fit/review/measurements.json:host.n_basin:d -->4<!-- /verify --> where it lands in a wrong basin (the `BASIN_*` warnings fire) a reported
value moves by up to <!-- verify:specs/001-reference-fit/review/measurements.json:host.max_abs_dex_basin:.1f -->3.2<!-- /verify --> orders of magnitude and claim shapes
change on <!-- verify:specs/001-reference-fit/review/measurements.json:host.n_shape_flips:d -->2<!-- /verify -->; on the other <!-- verify:specs/001-reference-fit/review/measurements.json:host.n_other:d -->6<!-- /verify --> by at most
<!-- verify:specs/001-reference-fit/review/measurements.json:host.max_abs_dex_other:.2f -->0.15<!-- /verify --> (register D-019). Expect identical digits only on the same
host in the pinned environment.

## Documentation

- Algorithms: [`docs/algorithms/main.pdf`](docs/algorithms/main.pdf) (the `profile_v033`, FP engine, `profile_jax2`,
  its 1.0.1 port, cohort generator, heterogeneity, Langevin, Kaplan–Meier and statistics, units, cleaning, and fit
  windows sections).
- The design: [`specs/001-reference-fit/`](specs/001-reference-fit/), [`specs/003-calibrator/`](specs/003-calibrator/)
  [`specs/003d-cohort-generator/`](specs/003d-cohort-generator/) and [`specs/003b-jax2-101-port/`](specs/003b-jax2-101-port/)
  (spec, plan, contracts). Spec 005 (data, units, cleaning, plots, `fit-set`, `clean`):
  [`specs/005-data-plots-cli/`](specs/005-data-plots-cli/). Spec 006 (fit windows, `profile_jax2_w`):
  [`specs/006-calibrator-capabilities/`](specs/006-calibrator-capabilities/).
- Principles: [`.specify/memory/constitution.md`](.specify/memory/constitution.md).
