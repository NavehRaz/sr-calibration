# Source of the example cohort

`itp_c2016_het3_control_female.csv`: 304 genetically heterogeneous UM-HET3 **control** female mice from
the NIA Interventions Testing Program (ITP), cohort C2016, three sites pooled (TJL 96, UM 92, UT 116).
303 deaths and 1 mouse removed from the study alive (censored at day 557).

## Where it comes from

- Mouse Phenome Database (MPD), project ITP1: <https://phenome.jax.org/projects/ITP1>, file
  `ITP_C2016_Lifespan.xlsx` (MPD file id 754; ITP1 last updated 2026-08-17).
- Terms: MPD's terms of use, quoted in [`LICENSE`](LICENSE) (a waiver of copyright with a citation
  requirement, not a named licence). Citation: [`CITATION`](CITATION).
- SHA-256 of the source file, as downloaded on 2026-10-03:
  `8c23eb208a1c4dd46b6ce9cee788796a25a2844cbee62e72a7693198cda330df`. MPD may revise files; the tool
  refuses a file with another digest.
- SHA-256 of the derived file: `d2c1066a8907c6475bbff153bd557e21171ce7fc3eb23077e7debaff9b365aa3`.

MPD serves the file to scripts (space requests by at least 2 s, per its terms):

```bash
curl -sSLJ -o .cache/itp/ITP_C2016_Lifespan.xlsx "https://phenome.jax.org/getcurationfile?id=754&mode=download"
.venv/bin/python tools/make_example_cohort.py
```

The tool reads the spreadsheet with the Python standard library, checks the digest, writes the CSV,
and stores the site check in `specs/001-reference-fit/review/itp_selection.json`.

## Selection (set before the fit)

Columns: population, cohort, site, sex, id, group, dose, age_initiation, status, dead, age (days at
death or removal). Rows kept: cohort `C2016`, group `Control`, sex `f`.

No single site × sex × year control group reaches 200 mice (the largest has 127), so the three sites of
one cohort year and one sex are pooled, under a rule set before the fit was started: pool only if the sites'
survival does not differ (log-rank p ≥ 0.05) and the pooled hazard rises without a dip. It did:
log-rank p = 0.50, and the death rate rises in every 100-day window from day 500 to day 1,300
([`review/itp_selection.md`](../../specs/001-reference-fit/review/itp_selection.md)). Females, because
male controls have early deaths "probably related to mouse urinary syndrome" (Miller et al. 2014, Aging Cell,
doi:10.1111/acel.12170).

## Recording resolution and transformation

- δ = **1 day**: mice were checked daily (UM and UT; at TJL, daily once marked ill), and `age` is an
  integer number of days.
- A death's `age` is the day the mouse was found dead or euthanized when moribund, so it died in
  (age − 1, age]. sraging scores a death at *t* with δ over [t, t + δ), so every death is moved to the
  previous day, **age − 1**.
- The one censored mouse was removed from the study alive at day 557: it was known alive then, so its
  age is kept.
- Kept as recorded (spec 001 does no data cleaning): three early deaths at days 54, 197 and 200.
- Events are `dead` unchanged; row order is the file's.
