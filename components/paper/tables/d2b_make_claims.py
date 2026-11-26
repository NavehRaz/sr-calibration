"""Write sources/claims_new_d2b.toml (claims of drafter d2b) by extracting each value from its stored source with the
same locators that verify_claims.py uses. Reads stored files only.
Run (after tables/d2b_ident.py): python -B tables/d2b_make_claims.py
"""
import json, math, os, sys, tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import verify_claims as V  # noqa: E402

T = "~/code/sraging/tests/acceptance/results/"
G = "aging:code_5/sr_calibration_tester_jax2/reports/rc2/GATEV6/out/GATE_V6.txt"
I2 = "aging:code_5/sr_calibration_tester_jax2/reports/rc2/I2/I2_REPORT.txt"
KT = "aging:code_5/sr_calibration_tester_jax2_rel/reports/rel101/G3_KFIX/BOTH/KAPPA0_TABLE.txt"
KR = "aging:code_5/sr_calibration_tester_kfix/reports/kfix/REPORT.md"
KI = "aging:code_5/sr_calibration_tester_kfix/reports/kfix/invariance.md"
EXJ = "aging:code_5/sr_calibration_tester/reports/jax2-v6-full-bign/explorer_data.json"
EXH = "aging:code_5/sr_calibration_tester/reports/jax2h-v6-full-bign/explorer_data.json"
CMP = "aging:code_5/sr_calibration_tester/reports/jax2h-v6-full-bign/comparison.json"
NB = "aging:code_5/sr_fp_fit_share/validate_01_simulation_shapes.ipynb"

existing = {c["id"]: c for c in tomllib.load(open(os.path.join(ROOT, "claims.toml"), "rb"))["claim"]}
claims = []


def sci(x, sig=2):
    s = f"{x:.{sig - 1}e}"
    m, e = s.split("e")
    return float(s), m, int(e)


def add(cid, label, source, locator, fmt="int", units="", note="", derived_from=None, scale=1.0, post=None):
    path = V.resolve(source)
    got = V.extract(locator, path)
    c = {"id": cid, "label": label, "units": units, "source": source, "sha256": V.sha(path),
         "commit": "stored result file", "locator": locator, "note": note}
    if derived_from:
        c["derived_from"] = derived_from
    if fmt == "str":
        c["value"] = str(got).strip()
        c["display"] = c["value"]
        c["rtol"] = 0
    else:
        x = float(str(got).replace(",", "")) * scale
        if fmt == "int":
            c["value"] = int(round(x)); c["rtol"] = 0 if abs(x - round(x)) < 1e-9 else 0.5 / abs(x) + 0.003; c["display"] = str(c["value"])
        elif fmt == "comma":
            c["value"] = int(round(x)); c["rtol"] = 0; c["display"] = f"{c['value']:,}".replace(",", "{,}")
        elif fmt.startswith("f"):
            nd = int(fmt[1:])
            c["value"] = round(x, nd); c["display"] = f"{c['value']:.{nd}f}"
            c["rtol"] = 0.5 * 10 ** (-nd) / max(abs(x), 1e-12) + 0.003
        elif fmt in ("sci", "pow"):
            v, m, e = sci(x)
            c["value"] = v; c["rtol"] = 0.06
            if fmt == "pow" or m in ("1.0",) and fmt == "pow":
                c["display"] = f"$10^{{{e}}}$"
            else:
                sign = "-" if v < 0 else ""
                m = m.lstrip("-")
                c["display"] = f"${sign}{m}\\times10^{{{e}}}$"
        else:
            raise ValueError(fmt)
    claims.append(c)


def clone(newid, oldid, fmt="sci", label=None):
    o = existing[oldid]
    path = V.resolve(o["source"])
    got = V.extract(o["locator"], path)
    c = {"id": newid, "label": label or (o["label"] + " (display variant)"), "units": o.get("units", ""),
         "source": o["source"], "sha256": V.sha(path), "commit": o.get("commit", ""), "locator": o["locator"],
         "note": f"same value as {oldid}, LaTeX display"}
    if o["locator"].get("negate"):
        got = -float(got)
    v, m, e = sci(float(got))
    c["value"] = v; c["rtol"] = 0.06
    sign = "-" if v < 0 else ""
    c["display"] = f"${sign}{m.lstrip('-')}\\times10^{{{e}}}$"
    claims.append(c)


def js(path):
    return {"kind": "json", "path": path}


def py(expr):
    return {"kind": "python", "expr": expr}


def rx(pattern, group=1):
    return {"kind": "regex", "pattern": pattern, "group": group}


# ---------------------------------------------------------------- 7a numerics (t1, mpmath, anchor, early, score)
clone("acc-t1-dS-s", "acc-t1-dS")
clone("acc-t1-grad-max-s", "acc-t1-grad-max")
clone("acc-t1-hetwindow-total-s", "acc-t1-hetwindow-total")
clone("acc-t1-library-nonSweden-total-s", "acc-t1-library-nonSweden-total")
clone("acc-t1-resolved-term-s", "acc-t1-resolved-term")
clone("acc-t1-sweden-total-s", "acc-t1-sweden-total")
clone("acc-mp-v1-lowmass-s", "acc-mp-v1-lowmass")
for k, lab in (("total", "total"), ("term", "resolved term"), ("dS", "max |dS|"), ("grad_rel", "relative gradient")):
    add(f"acc-t1-thr-{ {'grad_rel':'grad'}.get(k,k) }-s".replace(" ", ""), f"T1 acceptance limit, {lab}", T + "t1_002.json",
        js(f"thresholds.{k}"), fmt="pow")
add("acc-t1-fail", "T1 evaluations failed", T + "t1_002.json", py("src['summary']['evaluations']-src['summary']['passed']"))
add("acc-t1-fail-sweden", "T1 failures on the Swedish realfull cohort", T + "t1_002.json",
    py("sum(1 for r in src['results'] if not r['ok'] and 'Sweden' in r['case'])"))
add("acc-t1-fail-cd", "T1 failures of the central-difference gradient cross-check only", T + "t1_002.json",
    py("sum(1 for r in src['results'] if not r['ok'] and any('central differences' in f for f in r['fails']))"))
add("acc-t1-sweden-n", "T1 largest cohort n", T + "t1_002.json", js("summary.largest_n"), fmt="comma")
add("acc-mp-dps", "mpmath digits", T + "mpref_002.json", js("dps"))
add("acc-mp-sweden-n", "n of the Swedish pure cohort in the round-off panel", T + "mpref_002.json",
    py("src['roundoff'][0]['n']"), fmt="comma")
add("acc-mp-sweden-v1-s", "Sweden stored theta: v1 minus mpmath total (display variant)", T + "mpref_002.json",
    py("[x for x in src['roundoff'] if x['theta']=='stored'][0]['dl_v1']"), fmt="sci", units="nats")
add("acc-mp-v033-seed", "round-off panel: |v0.3.3 - mpmath| at the seed point, Denmark F 1900", T + "mpref_002.json",
    py("abs([x for x in src['roundoff'] if x['theta']=='seed'][0]['dl_v033'])"), fmt="int", units="nats")
add("acc-mp-roundoff-v033-max-s", "round-off panel: max |v0.3.3 - mpmath| total", T + "mpref_002.json",
    py("max(abs(x['dl_v033']) for x in src['roundoff'])"), fmt="sci", units="nats")
add("acc-mp-truth-fail-max-s", "failing stiff human truths: max |v1 - mpmath|", T + "mpref_002.json",
    py("max(abs(x['dl_v1']) for x in src['roundoff'] if x['theta']=='truth' and not x['ok'])"), fmt="sci", units="nats")
add("acc-mp-truth-fail-min-s", "failing stiff human truths: min |v1 - mpmath|", T + "mpref_002.json",
    py("min(abs(x['dl_v1']) for x in src['roundoff'] if x['theta']=='truth' and not x['ok'])"), fmt="sci", units="nats")
add("acc-mp-sweden-eps15-dl-s", "Sweden realfull, epsilon x1.5: |v1 - mpmath|", T + "mpref_002.json",
    py("abs([x for x in src['roundoff'] if x['theta']=='epsilon x1.5'][0]['dl_v1'])"), fmt="sci", units="nats")
add("acc-mp-sweden-eta15-spread-s", "Sweden realfull, eta x1.5: v1 +-3-ulp spread", T + "mpref_002.json",
    py("[x for x in src['roundoff'] if x['theta']=='eta x1.5'][0]['spread_v1']"), fmt="sci", units="nats")
add("acc-anchor-dS-coarse-s", "anchor: max |dS| at the coarsest grid, rho_eps 0.05", T + "anchor_002.json",
    py("src['diffusion'][0]['max_dS'][0]"), fmt="sci")
add("acc-anchor-dS-fine-s", "anchor: max |dS| at the finest grid, rho_eps 0.05", T + "anchor_002.json",
    py("src['diffusion'][0]['max_dS'][-1]"), fmt="sci")
add("acc-early-unconv", "early-death references not converged", T + "early_002.json",
    py("sum(r['n_early']-r['n_converged'] for r in src['results'])"))
add("acc-early-sweden-max", "largest single early-death residual, v1, Sweden F 1910 pure", T + "early_002.json",
    py("[r for r in src['results'] if r['cohort']=='pure-real-Sweden_F_1910_homo-c0-s0'][0]['residual_v1']['max_abs']"),
    fmt="int", units="nats")
add("acc-score-zcrit", "score test multiplicity-corrected |z| threshold", T + "score_002.json", js("zcrit"), fmt="f2")
add("acc-score-n", "score test individuals per simulated cohort", T + "score_002.json", js("features.pure.n"), fmt="comma")
add("acc-score-R-pure", "score test replicates, pure", T + "score_002.json", js("features.pure.R"))
add("acc-score-R-hazard", "score test replicates, hazard", T + "score_002.json", js("features.hazard.R"))
add("acc-score-R-entry", "score test replicates, entry", T + "score_002.json", js("features.entry.R"))
add("acc-score-R-window-big", "score test replicates, window, larger run (not informative)", T + "score_002.json",
    js("features.window.R"))
add("acc-score-R-window", "score test replicates, window, informative run", T + "score_window_002.json",
    js("features.window.R"))
add("acc-score-R-het", "score test replicates, heterogeneity", T + "score_het_002.json", js("features.heterogeneity.R"))
add("acc-score-entry", "score test entry max |z|", T + "score_002.json",
    py("max(abs(z) for z in src['features']['entry']['z'])"), fmt="f2")
add("acc-score-window-plant", "score test window: planted wrong closing rule, max |z|", T + "score_window_002.json",
    py("max(abs(z) for z in src['features']['window']['plant']['z'])"), fmt="f1")

# N rule and grid error (I2 report)
add("nrule-N0", "production grid space cells", I2, rx(r"\(N = (\d+) space cells"))
add("nrule-cap", "N rule cap", I2, rx(r"capped at N = (\d+)"), fmt="comma")
add("nrule-target", "N rule target error", I2, rx(r"predicted error is\s+<= ([0-9.]+) nats"), fmt="f1", units="nats")
add("nrule-resid-sd", "N-rule regression residual sd", I2, rx(r"residual sd ([0-9.]+) dex"), fmt="f2", units="dex")
add("nrule-share-before", "within 0.2 nats at N=900", I2, rx(r"goes (\d+)/134 -> (\d+)/134", 1))
add("nrule-share-after", "within 0.2 nats with the rule", I2, rx(r"goes (\d+)/134 -> (\d+)/134", 2))
add("nrule-med-before", "median error at N=900", I2, rx(r"median error goes (\d+\.\d+) -> (\d+\.\d+)", 1), fmt="f2", units="nats")
add("nrule-med-after", "median error with the rule", I2, rx(r"median error goes (\d+\.\d+) -> (\d+\.\d+)", 2), fmt="f2", units="nats")
add("nrule-cpu-x", "full-tier CPU factor of the rule", I2, rx(r"Full-tier CPU\s+x([0-9.]+)"), fmt="f2")
add("nrule-human-med", "N=900 error, human n>=2.5e4, median", I2, rx(r"human n>=2\.5e4\s+\d+\s+([0-9.]+)\s+([0-9.]+)", 1), fmt="f2", units="nats")
add("nrule-human-max", "N=900 error, human n>=2.5e4, max", I2, rx(r"human n>=2\.5e4\s+\d+\s+([0-9.]+)\s+([0-9.]+)", 2), fmt="f1", units="nats")
add("nrule-sharp-med", "N=900 error, non-human CV<0.05, median", I2, rx(r"non-human CV<0\.05\s+\d+\s+([0-9.]+)\s+([0-9.]+)", 1), fmt="f2", units="nats")
add("nrule-sharp-max", "N=900 error, non-human CV<0.05, max", I2, rx(r"non-human CV<0\.05\s+\d+\s+([0-9.]+)\s+([0-9.]+)", 2), fmt="f1", units="nats")
add("nrule-human-needN", "N needed for the 0.2-nat target on n=1e5 human cohorts", I2, rx(r"need N >= ([0-9,]+)"), fmt="comma")

# previous solver
add("oldsrk-n-presets", "preset organisms in the FP-vs-Langevin comparison", NB, rx(r"Organisms with KS < 0\.05: \d+ / (\d+)"))
add("oldsrk-npeople", "Langevin individuals per preset", NB, rx(r"NPEOPLE   = (\d+)"), fmt="comma")
add("oldsrk-ks-worst", "worst preset survival KS (Jack Russell, vetCompass)", NB, rx(r"Jack_Russell_vetCompass\s+KS=([0-9.]+)"), fmt="f2")

# ---------------------------------------------------------------- 5 / 7c identifiability
add("ident-rbar-closed-panel", "beta*rbar closed on valid panel cohorts", KR, rx(r"closed on (\d+)/(\d+) valid panel cohorts and (\d+)/(\d+) valid\s+medium", 1))
add("ident-panel-valid", "valid panel cohorts (kfix report)", KR, rx(r"closed on (\d+)/(\d+) valid panel cohorts and (\d+)/(\d+) valid\s+medium", 2))
add("ident-rbar-closed-medium", "beta*rbar closed on valid medium cohorts", KR, rx(r"closed on (\d+)/(\d+) valid panel cohorts and (\d+)/(\d+) valid\s+medium", 3))
add("ident-medium-valid", "valid medium cohorts (kfix report)", KR, rx(r"closed on (\d+)/(\d+) valid panel cohorts and (\d+)/(\d+) valid\s+medium", 4))
add("kfix-medium-n", "valid medium cohorts, kfix kappa0 = 1", KR, rx(r"closed on (\d+)/(\d+) valid panel cohorts and (\d+)/(\d+) valid\s+medium", 4))
add("ident-rbar-width-lo", "closed beta*rbar width, low end", KR, rx(r"about ([0-9.]+)–([0-9.]+)\s+dex wide", 1), fmt="f2", units="dex")
add("ident-rbar-width-hi", "closed beta*rbar width, high end", KR, rx(r"about ([0-9.]+)–([0-9.]+)\s+dex wide", 2), fmt="f2", units="dex")
add("ident-log2", "log10 of 2: the limit of the drift of beta/(1+kappa) along the ridge", KI, py("math.log10(2)"), fmt="f2", units="dex",
    note="arithmetic from rbar(k)(1+k) -> 1/2 for k -> infinity; the measured median drift is ridge-drift-kp1")
add("ident-rbar-k0", "rbar(kappa0) at kappa0 = 0.001", KR, py("1-0.001*math.log(1+1/0.001)"), fmt="f3",
    note="computed from the definition; the report states 0.992")
for q, key, lab in (("eta", "eta", "eta, eps, sqrt(eta)/eps"), ("rbar", "beta_rbar", "beta*rbar"),
                    ("kp1", "beta_over_kp1", "beta/(1+kappa)"), ("beta", "beta", "raw beta")):
    add(f"ridge-drift-{q}", f"median drift across the 1.92-nat kappa valley, valid panel, {lab}", KI,
        rx(r"panel valid: cohorts 19.*?`" + key + r"` \| ([0-9.]+)"), fmt="f3" if q != "beta" else "f2", units="dex")
RC = "tables/d2b_ridge_cohort.csv"
RD = "tables/d2b_ridge.csv"
KP = "tables/d2b_kfix_panel.csv"
add("ridge-valley-n", "valid panel cohorts with all kappa0 slices within 1.92 nats", RC,
    py("sum(1 for r in csv.DictReader(io.StringIO(src)) if r['valid']=='1' and float(r['dll_span'])<=1.92)"),
    derived_from=["aging:code_5/sr_calibration_tester_kfix/runs/s2-kfix_k*-panel.json", "data/store (kfix)"])
add("ridge-valley-of", "valid panel cohorts", RC, py("sum(1 for r in csv.DictReader(io.StringIO(src)) if r['valid']=='1')"))
add("ridge-span-max", "largest log-likelihood span over kappa0 slices among valid panel cohorts", RC,
    py("max(float(r['dll_span']) for r in csv.DictReader(io.StringIO(src)) if r['valid']=='1')"), fmt="f1", units="nats")
add("ridge-n-kappa0", "pinned kappa0 values per panel cohort", RC, py("max(int(r['n_slices']) for r in csv.DictReader(io.StringIO(src)))"))
add("ridge-c-n", "valid panel cohorts whose median beta*rbar over slices is >= 1e-3 (Figure C)", RD,
    py("(lambda rows: sum(1 for c in set(r['cohort_id'] for r in rows if r['valid']=='1') if statistics.median(float(r['beta_rbar']) for r in rows if r['cohort_id']==c) >= 1e-3))(list(csv.DictReader(io.StringIO(src))))"))
add("kfix-panel-n", "kfix panel cohorts", RD, {"kind": "count", "where": {"kappa0": "0.001"}})
add("kfix-panel-valid", "kfix panel valid cohorts", RD, {"kind": "count", "where": {"kappa0": "0.001", "valid": "1"}})
add("kfix-panel-ktrue-lo", "kfix panel true kappa, low", KR, rx(r"Panel \(19 valid, true κ ([0-9.]+)–([0-9.]+)\)", 1), fmt="f1")
add("kfix-panel-ktrue-hi", "kfix panel true kappa, high", KR, rx(r"Panel \(19 valid, true κ ([0-9.]+)–([0-9.]+)\)", 2), fmt="int")
add("ident-kp1-lost-k0001", "valid panel cohorts losing coverage of beta/(1+kappa) at kappa0 = 0.001", KP,
    py("[int(r['hits']) for r in csv.DictReader(io.StringIO(src)) if r['kappa0']=='free' and r['quantity']=='beta_over_kp1'][0]-[int(r['hits']) for r in csv.DictReader(io.StringIO(src)) if r['kappa0']=='0.001' and r['quantity']=='beta_over_kp1'][0]"),
    derived_from=[KR])

def kt(var, k, q, what="net"):
    key = {"eta": r"eta", "eps": r"eps", "sqrt": r"sqrt\(eta\)/eps", "rbar": r"beta\*rbar"}[q]
    kk = {"0.001": r"0\.001", "0.5": r"0\.5", "1": r"1"}[k]
    return rx(rf"{var} kfix kappa0 = {kk}  .*?\n    {key} +K1.*?  n +(\d+)  L/G +\d+/\d+  net ([+-]\d+)", 2 if what == "net" else 1)

add("kfix-J0001-net-sqrt", "J kfix 0.001 net loss sqrt(eta)/eps", KT, kt("J", "0.001", "sqrt"))
add("kfix-J0001-net-max", "J kfix 0.001 largest net loss over eta, eps, sqrt(eta)/eps, beta*rbar", KT,
    py("max(int(x) for x in re.findall(r'\\n    (?:eta|eps|sqrt\\(eta\\)/eps|beta\\*rbar) +K1.*?net ([+-]\\d+)', src.split('--- J kfix kappa0 = 0.001')[1].split('\\n--- ')[0]))"))
add("kfix-J-n", "J kfix paired cohorts (eta)", KT, kt("J", "0.001", "eta", "n"))
add("kfix-H-n", "H kfix paired cohorts (eta, eps)", KT, kt("H", "0.001", "eta", "n"))
add("kfix-H-n-comb", "H kfix paired cohorts (combinations)", KT, kt("H", "0.001", "sqrt", "n"))
add("kfix-J1-net-eta", "J kfix 1 net loss eta", KT, kt("J", "1", "eta"))
add("kfix-J1-net-eps", "J kfix 1 net loss eps", KT, kt("J", "1", "eps"))
add("kfix-H0001-net-eta", "H kfix 0.001 net loss eta", KT, kt("H", "0.001", "eta"))
add("kfix-H0001-net-eps", "H kfix 0.001 net loss eps", KT, kt("H", "0.001", "eps"))
add("kfix-H05-net-eps", "H kfix 0.5 net loss eps", KT, kt("H", "0.5", "eps"))
add("kfix-H1-net-eta", "H kfix 1 net loss eta", KT, kt("H", "1", "eta"))
add("kfix-H1-net-eps", "H kfix 1 net loss eps", KT, kt("H", "1", "eps"))
add("kfix-medium-kp1-lost", "kfix medium kappa0 = 1: beta/(1+kappa) lost", KR, rx(r"β/\(1\+κ\) on medium:\*\* (\d+) lost"))
add("kfix-medium-rbar-lost", "kfix medium kappa0 = 1: beta*rbar lost", KR, rx(r"\| β·r̄ \| (\d+) \| \d+ \|"))
add("kfix-medium-eps-lost", "kfix medium kappa0 = 1: eps lost", KR, rx(r"\| ε \| (\d+) \| \d+ \|"))
add("kfix-medium-eta-lost", "kfix medium kappa0 = 1: eta lost", KR, rx(r"\| η \| (\d+) \| \d+ \|"))

add("ident-kappa-declined-pct", "J v6 reference: kappa declined, percent of cohorts", EXJ,
    py("100*sum(1 for x in src['result']['shape'][3] if x==0)/src['n_cohorts']"))
add("v6-J-pure-kappa-declined-pct", "J v6 reference pure arm: kappa declined, percent", EXJ,
    py("100*sum(1 for a,x in zip(src['cohort']['arm'],src['result']['shape'][3]) if a=='pure' and x==0)/sum(1 for a in src['cohort']['arm'] if a=='pure')"))
add("v6-J-hazard-kappa-declined-pct", "J v6 reference hazard arm: kappa declined, percent", EXJ,
    py("100*sum(1 for a,x in zip(src['cohort']['arm'],src['result']['shape'][3]) if a=='hazard' and x==0)/sum(1 for a in src['cohort']['arm'] if a=='hazard')"))
add("v6-H-kappa-declined-pct", "H v6 reference: kappa declined, percent", EXH,
    py("100*sum(1 for x in src['result']['shape'][3] if x==0)/src['n_cohorts']"))
add("ident-onesided-kappa-pct", "J pooled one-sided kappa coverage, percent", CMP,
    py("100*src['by_parameter']['kappa']['coverage']['reference_one_sided']['coverage']"))
add("ident-onesided-beta-pct", "J pooled one-sided beta coverage, percent", CMP,
    py("100*src['by_parameter']['beta']['coverage']['reference_one_sided']['coverage']"))
add("ident-H-onesided-kappa-pct", "H pooled one-sided kappa coverage, percent", CMP,
    py("100*src['by_parameter']['kappa']['coverage']['method_one_sided']['coverage']"))

# ---------------------------------------------------------------- 7e leverage
AC = T + "ac11_003b.json"
for nm, key, lab in (("heldout", "held_out", "held-out"), ("smalln", "small_n", "small-n"), ("twins", "twins", "twin")):
    add(f"lev-null-{nm}-ub", f"exact one-sided 95% upper bound on the false-flag rate, {lab} null (0 flags), percent", AC,
        py(f"100*(1-0.05**(1/src['nulls']['{key}']['per_set']['all4']['n']))"), fmt="f1" if nm == "smalln" else "int",
        note="Clopper-Pearson one-sided limit for zero events: 1 - 0.05**(1/n)")
add("lev-null-slide-heldout", "held-out null cohorts labelled flat_valley_slide (ratio > 1.5, gain <= 1.92 nats)", AC,
    py("sum(1 for r in src['rows'] if r['mode']=='held_out' and r['label']=='flat_valley_slide')"))
add("lev-null-slide-smalln", "small-n null cohorts labelled flat_valley_slide", AC,
    py("sum(1 for r in src['rows'] if r['mode']=='small_n' and r['label']=='flat_valley_slide')"))
add("lev-heldout-size", "held-out null cohort n", AC, py("max(r['n'] for r in src['rows'] if r['mode']=='held_out')"), fmt="comma")
add("lev-smalln-size", "small-n null cohort n", AC, py("max(r['n'] for r in src['rows'] if r['mode']=='small_n')"), fmt="int")
for n in ("100", "300", "1000"):
    add(f"lev-late-detected-n{n}", f"planted late death detected at n={n} (of 6)", AC, js(f"detection.late['{n}'].detected"))
add("lev-p3-nmin", "P3 pilot panel smallest n", T + "p3_003f.json", py("min(r['n'] for r in src['rows'])"), fmt="comma")
add("lev-p3-nmax", "P3 pilot panel largest n", T + "p3_003f.json", py("max(r['n'] for r in src['rows'])"), fmt="comma")
add("lev-smalln-measured", "small-n null cohorts with a stored estimate (convergence measured)", T + "lev_small_n_003g.json", js("n_measured"))

# ---------------------------------------------------------------- SI S3 gates
for g, lab in (("1", "G1"), ("2", "G2"), ("3", "G3"), ("4", "G4"), ("5", "G5")):
    add(f"gate-v6-g{g}-verdict", f"GATE-V6 {lab} verdict", G, rx(rf"\nG{g} (PASS|FAIL|WARN)"), fmt="str")
add("gate-v6-overall", "GATE-V6 overall verdict", G, rx(r"GATE-V6 mle_profile_jax2 1\.0\.1: (\w+)"), fmt="str")
add("gate-v6-g2-informative", "G2 informative records", G, rx(r"witness (\d+) informative"))
add("gate-v6-g2-new", "G2 failures of the released calibrator, summed over arms", G,
    py("sum(int(x) for x in re.findall(r'informative\\s+\\d+: new (\\d+) vs', src))"))
add("gate-v6-registry-total", "gate registry size", G, rx(r"'total': (\d+)"))
add("gate-v6-g14-n", "G14: kappa values in the library", G, rx(r"G14 (\d+) kappa values; (\d+) below", 1))
add("gate-v6-g14-low", "G14: kappa values below 20 distinct manifold points", G, rx(r"G14 (\d+) kappa values; (\d+) below", 2))
add("gate-v6-g2cells-n", "registry G2: strata", G, rx(r"G2 (\d+) strata; (\d+) below 10 cells", 1))
add("gate-v6-g2cells-low", "registry G2: strata below 10 cells", G, rx(r"G2 (\d+) strata; (\d+) below 10 cells", 2))
add("gate-v6-g5-integrity", "G5 integrity failures", G, rx(r"integrity failures (\d+)"))
add("gate-v6-g5-sc3", "G5 re-evaluation failures", G, rx(r"SC3 FAIL (\d+)"))
GG = T + "gate_003g.json"
add("gate-003g-n-verdicts", "003g gate verdicts", GG, js("n_verdicts"))
add("gate-003g-required", "003g required stratum-rule items", GG, py("len(src['required'])"))
add("gate-003g-notcovered", "003g not-covered cells", GG, py("len(src['not_covered'])"))
add("gate-003g-blocking", "003g blocking entries", GG, py("len(src['blocking'])"))
add("gate-003g-blocking-pr3", "003g blocking entries under PR-3", GG, py("sum(1 for b in src['blocking'] if b['rule']=='PR-3')"))
add("gate-003g-blocking-pr1", "003g blocking entries under PR-1", GG, py("sum(1 for b in src['blocking'] if b['rule']=='PR-1')"))
add("gate-003g-decided", "003g blocking entries decided", GG, js("decisions.decided"))
add("gate-003g-deviations", "003g recorded deviations", GG, py("len(src['deviations'])"))
CE = ("re.search(r'(\\d+)/(\\d+) against (\\d+)/(\\d+) inside \\(lower bound (-[0-9.]+)\\).*?p = ([0-9.]+)', "
      "[b for b in src['blocking'] if b['stratum']=='SRK-1/outside' and b['arm']=='censor' and b['quantity']=='eta'][0]['decision']['reason'])")
for i, (nm, fmt) in enumerate((("censor-hits", "int"), ("censor-n", "int"), ("censor-in-hits", "int"), ("censor-in-n", "int"),
                               ("censor-lb", "f3"), ("censor-p", "f2")), start=1):
    add(f"gate-003g-{nm}", f"003g SRK-1/outside censor eta: {nm}", GG, py(CE + f".group({i})"), fmt=fmt)

for c in claims:
    if c["id"] == "gate-003g-censor-lb":
        c["display"] = "$" + c["display"] + "$"

# verdict-string claims have no tolerance; write toml
def q(v):
    return json.dumps(v, ensure_ascii=False)


def val(v):
    return q(v) if isinstance(v, str) else repr(v)


out = ["# Claims of drafter d2b (generated by tables/d2b_make_claims.py from stored sources; merge into claims.toml)", ""]
for c in claims:
    out.append("[[claim]]")
    for k in ("id", "label", "value", "display", "units", "rtol", "source", "sha256", "commit"):
        if k in c:
            out.append(f"{k:<8}= {val(c[k])}")
    def tv(v):
        if isinstance(v, str):
            return q(v)
        if isinstance(v, dict):
            return "{ " + ", ".join(f"{kk} = {q(vv)}" for kk, vv in v.items()) + " }"
        return repr(v).lower() if isinstance(v, bool) else repr(v)
    loc = ", ".join(f"{k} = {tv(v)}" for k, v in c["locator"].items())
    out.append(f"locator = {{ {loc} }}")
    if c.get("derived_from"):
        out.append(f"derived_from = {json.dumps(c['derived_from'], ensure_ascii=False)}")
    out.append(f"note    = {q(c['note'])}")
    out.append("")
with open(os.path.join(ROOT, "sources", "claims_new_d2b.toml"), "w") as f:
    f.write("\n".join(out))
print(len(claims), "claims written")
