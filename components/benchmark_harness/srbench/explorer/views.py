"""The shipped views.

Each is a self-contained unit declaring what it reads. Adding another one means appending a
`register(View(...))` here (or in your own module) and touching nothing else.

Every view obeys two rules the payload makes enforceable:
  * coverage is never drawn without its denominator and decline rate;
  * arms are never pooled -- `per_arm=True` means the assembler renders one series per arm.
"""
from __future__ import annotations

from .registry import View, register

# Shared helpers available to every view's render body, injected once by the assembler.
_COMMON_JS = ""


register(View(
    view_id="coverage_by_arm",
    title="Coverage by arm, with decline rate",
    description=("The fraction of cohorts whose interval contained the truth, per arm, against the "
                 "nominal level (dashed). The decline rate is drawn beside every bar, because a "
                 "coverage computed on a self-selected denominator is not a measurement."),
    requires=("cohort.arm", "result.hit", "result.shape", "denominators", "quantities"),
    render_js="""
      const q = ui.quantity();
      const qi = D.quantities.indexOf(q);
      // A combination has a point estimate but no interval, so it has no coverage to show --
      // saying so beats drawing six empty bars.
      if ((D.accuracy_only_quantities||[]).indexOf(q) >= 0)
        return denomLine(q + ' is a combination of the fitted parameters: it carries a point '
          + 'estimate but no interval, so there is no coverage to report. See the error panel.');
      const IDX = ui.applies(qi);
      const arms = [...new Set(IDX.map(i => D.cohort.arm[i]))];
      const rows = arms.map(a => {
        let hit=0, claimed=0, declined=0, total=0;
        for (const i of IDX) {
          if (D.cohort.arm[i] !== a) continue;
          total++;
          const h = D.result.hit[qi][i], s = D.result.shape[qi][i];
          if (s === 0) { declined++; continue; }
          if (s !== 3) continue;              // closed only; shapes are never pooled
          claimed++; if (h === 1) hit++;
        }
        return {arm:a, cov: claimed? hit/claimed : null, claimed, declined, total};
      });
      return barsWithDenominator(rows, q, D.provenance.level);
    """,
))

register(View(
    view_id="error_on_the_manifold",
    title="Error against the manifold coordinates",
    description=("Absolute log-10 error against removal timescale and curve steepness -- the two "
                 "coordinates a survival curve actually determines. The error is FLOORED at each "
                 "cohort's negligibility floor, exactly as the comparison report floors it, so a "
                 "value nobody could have measured scores 0 rather than a large error."),
    requires=("cohort.log_ratio", "cohort.kappa", "result.err", "result.clamped", "result.spread",
              "result.err_theta", "result.clamped_theta", "quantities"),
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      const IDX = ui.idx();
      const pts = [];
      for (const i of IDX) {
        const e = errAt(qi, i);
        if (e === null) continue;
        pts.push({x: D.cohort.log_ratio[i], y: e, arm: D.cohort.arm[i], k: D.cohort.kappa[i]});
      }
      const nclamp = IDX.reduce((t,i)=>t+(errAt(qi,i)!==null&&clampedAt(qi,i)?1:0),0);
      const box = scatter(pts, 'log10 removal ratio', '|log10(est/truth)| (dex)');
      if (comboNote(qi, IDX) && box.appendChild) box.appendChild(denomLine(comboNote(qi, IDX)));
      if (nclamp && box.appendChild) box.appendChild(denomLine(
        nclamp + ' of these sit at the negligibility floor, where the truth is below what n '
        + 'could resolve -- the floor, not the estimator, sets the value.'));
      return box;
    """,
))

register(View(
    view_id="interval_width",
    title="Interval width of closed claims",
    description=("Width in dex, CLOSED claims only. A one-sided claim has no width, and giving it "
                 "one is how widths get pooled across shapes."),
    requires=("result.width", "cohort.arm", "quantities"),
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      if ((D.accuracy_only_quantities||[]).indexOf(q) >= 0)
        return denomLine(q + ' is a combination of the fitted parameters: the method claims '
          + 'intervals on the four parameters, not on functions of them, so there is no width to '
          + 'show. Deriving one would invent a bound the profile never produced.');
      const vals = ui.applies(qi).map(i => D.result.width[qi][i]).filter(v => v !== null);
      return histogram(vals, 'interval width (dex)', q);
    """,
))

register(View(
    view_id="claim_shape_mix",
    title="Claim shapes",
    description=("How often each shape was returned. A method that converts closed intervals into "
                 "one-sided bounds has answered less of the question, not more."),
    requires=("result.shape", "cohort.arm", "quantities"),
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      const IDX = ui.applies(qi);
      const arms = [...new Set(IDX.map(i => D.cohort.arm[i]))];
      const rows = arms.map(a => {
        const c = [0,0,0,0];
        for (const i of IDX) if (D.cohort.arm[i]===a) c[D.result.shape[qi][i]]++;
        return {arm:a, none:c[0], upper:c[1], lower:c[2], closed:c[3]};
      });
      return stackedShapes(rows, q);
    """,
))

register(View(
    view_id="run_time",
    title="Run time per calibration",
    description=("Each cohort against what the reference spent on that same cohort. Above the "
                 "dashed line is slower. Cost is a reported result, not a footnote -- and never "
                 "decides on its own."),
    requires=("timing.wall_clock_seconds", "timing.ref_wall_clock_seconds", "cohort.n"),
    per_arm=False,
    render_js="""
      // Against the REFERENCE, not against cohort size. This cost is flat in sample size -- the
      // Fokker-Planck grid comes from the lifespan scale, not the number of individuals -- so
      // seconds-vs-n plotted a dependence known not to exist, and the old view also computed the
      // reference timing and then never drew it.
      //
      // A cohort contributes only when BOTH timings exist: comparing a method's cost against a
      // missing reference cost would turn a gap in the reference run into an apparent speed-up.
      const pts = [];
      for (const i of ui.idx()) {
        const w = D.timing.wall_clock_seconds[i];
        const r = D.timing.ref_wall_clock_seconds[i];
        if (w === null || r === null) continue;
        pts.push({x: r, y: w, arm: D.cohort.arm[i]});
      }
      if (!pts.length) return denomLine('no cohort has both a method and a reference timing');
      return refCompare(pts, 'reference wall-clock seconds', 'this method, wall-clock seconds');
    """,
))

register(View(
    view_id="decline_rate_by_kappa",
    title="Decline rate against removal half-saturation",
    description=("Where a method refuses to answer. For the flattest parameter the reference "
                 "declines on most cohorts, and that is the correct answer, not a failure."),
    requires=("cohort.kappa", "result.shape", "quantities"),
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      const IDX = ui.applies(qi);
      const ks = [...new Set(IDX.map(i => D.cohort.kappa[i]))].sort((a,b)=>a-b);
      const rows = ks.map(k => {
        let dec=0, tot=0;
        for (const i of IDX) if (D.cohort.kappa[i]===k) {
          tot++; if (D.result.shape[qi][i]===0) dec++;
        }
        return {k, rate: tot? dec/tot : null, n: tot};
      });
      return lineWithN(rows, 'kappa', 'decline rate', q);
    """,
))


# ---------------------------------------------------------------------------------------------
# Ported from the predecessor explorer (`sr_fp_fit_share/benchmark_reports/mle_report/
# explorer.html`), rebuilt on this payload. Two of its rules are kept deliberately because each
# was written against a real mistake: one marker per PARAMETER POINT rather than per cohort, so
# seed replicates cannot make a thin design look dense; and grey is not red, because "declined"
# and "wrong" are different statements.
# ---------------------------------------------------------------------------------------------

register(View(
    view_id="quantity_table",
    title="Every quantity, with its denominator",
    description=("All four fitted parameters and the ten combinations of them, in one table. "
                 "Coverage and claims must be read together: declining more often raises coverage "
                 "for free. The SE is cluster-robust, clustered on the parameter point."),
    requires=("result.hit", "result.err", "result.width", "result.shape", "result.clamped",
              "result.spread", "result.err_theta", "result.clamped_theta", "cohort.cluster_key",
              "quantities"),
    per_arm=False,
    render_js="""
      return quantityTable(ui.idx());
    """,
))

register(View(
    view_id="accuracy_cube",
    title="Accuracy across the parameter space",
    description=("Median |log10(est/truth)| at each parameter point, in the dimensionless "
                 "coordinates. Green <0.2 dex, amber <0.5, red above. For a COMBINATION the "
                 "'combinations' control decides what is drawn: by default only where the "
                 "fit's own near-optimal points pin it, a ring elsewhere. Drag to rotate, scroll to "
                 "zoom -- both cubes share one camera, so a marker in the same place is the same "
                 "parameter point in both."),
    requires=("cohort.rho_beta", "cohort.rho_eta", "cohort.rho_eps", "cohort.cluster_key",
              "result.err", "result.spread", "result.err_theta", "result.theta_fill",
              "cohort.fit_stale", "cohort.theta_at_edge", "quantities"),
    per_arm=False,
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      return cube(ui.idx(), qi, 'acc');
    """,
))

register(View(
    view_id="coverage_cube",
    title="Coverage across the parameter space",
    description=("The same points, coloured by coverage against the nominal level: green >=0.90, "
                 "amber >=0.80, red below, and GREY where the method never claimed. Declining and "
                 "being wrong are different statements and are never coloured the same."),
    requires=("cohort.rho_beta", "cohort.rho_eta", "cohort.rho_eps", "cohort.cluster_key",
              "result.hit", "quantities"),
    per_arm=False,
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      if ((D.accuracy_only_quantities||[]).indexOf(q) >= 0)
        return denomLine(q + ' is a combination: it has a point estimate but no interval, so '
          + 'there is no coverage to colour by. The accuracy cube above still applies.');
      return cube(ui.idx(), qi, 'cov');
    """,
))

register(View(
    view_id="n_dependence",
    title="Dependence on sample size",
    description=("Coverage and accuracy across n, over the parameter points present at every "
                 "sample size SHOWN -- paired, so a change is a change in the estimator and not "
                 "in which cohorts exist at each n. The sizes are chosen as the largest set "
                 "sharing 8+ points, and named beneath the figure. Accuracy uses the unclamped "
                 "subset, because the negligibility floor shrinks with n and a clamped median "
                 "moves even when the estimator does not."),
    requires=("cohort.n", "cohort.cluster_key", "result.hit", "result.err", "result.clamped",
              "result.spread", "result.err_theta", "result.clamped_theta", "quantities"),
    per_arm=False,
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      // The ladder: n values carrying enough cohorts to mean anything.
      const IDX = ui.idx();
      const cnt = {}; for (const i of IDX) cnt[D.cohort.n[i]] = (cnt[D.cohort.n[i]]||0)+1;
      const ladder = Object.keys(cnt).map(Number).filter(n => cnt[n] >= 100).sort((a,b)=>a-b);
      if (ladder.length < 2)
        return denomLine('fewer than two sample sizes carry enough cohorts to compare.');
      // PAIRED BY CONSTRUCTION. The predecessor's own note: 62% of one apparent coverage decline
      // across n turned out to be composition, not behaviour.
      const at = new Map();
      for (const i of IDX) {
        const k = D.cohort.cluster_key[i];
        if (!at.has(k)) at.set(k, new Set());
        at.get(k).add(D.cohort.n[i]);
      }
      // CHOOSE the rungs rather than demanding all of them. Requiring a point at every rung gave
      // ZERO on this library: only n=1000 and n=10000 share parameter points (84 of them), and
      // three rungs share just 3. So take the largest set of rungs that still has 8+ points in
      // common -- more rungs preferred, and the chosen set is stated beneath the figure, because
      // "which n are being compared" is part of the claim.
      const MIN = 8;
      let best = null;
      for (let mask=1; mask < (1<<ladder.length); mask++) {
        const sub = ladder.filter((_,b) => mask & (1<<b));
        if (sub.length < 2) continue;
        let common = 0;
        at.forEach(ns => { if (sub.every(n => ns.has(n))) common++; });
        if (common < MIN) continue;
        if (!best || sub.length > best.sub.length ||
            (sub.length === best.sub.length && common > best.common))
          best = {sub, common};
      }
      if (!best)
        return denomLine('No two sample sizes share ' + MIN + '+ parameter points, so an n-trend '
          + 'here would compare different cohort sets rather than different n. It is not drawn.');
      const rungs = best.sub;
      const paired = new Set();
      at.forEach((ns,k) => { if (rungs.every(n => ns.has(n))) paired.add(k); });
      const rows = rungs.map(n => {
        let hits=[], errs=[], nerr=0;
        for (const i of IDX) {
          if (D.cohort.n[i] !== n || !paired.has(D.cohort.cluster_key[i])) continue;
          const h = D.result.hit[qi][i], e = errAt(qi, i);
          if (h !== null) hits.push(h);
          if (e !== null && !clampedAt(qi, i)) { errs.push(e); nerr++; }
        }
        return {x: Math.log10(n), label: 'n=' + n,
                cov: hits.length ? hits.reduce((a,b)=>a+b,0)/hits.length : null,
                claims: hits.length, err: median(errs), nerr: nerr};
      });
      return dualAxis(rows, 'n', t => fmtNum(Math.pow(10,t)),
        'n = ' + rungs.join(', ') + ' · ' + paired.size + ' parameter points present at EVERY one '
        + 'of them, so this compares n and not which cohorts exist · accuracy on the unclamped '
        + 'subset, because the floor shrinks with n and a clamped median moves even when the '
        + 'estimator does not' + (comboNote(qi, IDX) ? '. ' + comboNote(qi, IDX) : ''));
    """,
))

register(View(
    view_id="regime_dependence",
    title="Ballistic against quasi-steady",
    description=("Against beta/((kappa+1)*eta*L) -- removal at the damage threshold over "
                 "production at the lifespan scale. Below 1 is ballistic, above 1 quasi-steady. "
                 "The L is what makes the 1 meaningful: the ratio without it is a time, so its "
                 "sign against 1 would depend on the recording unit."),
    requires=("cohort.qs", "cohort.cluster_key", "result.hit", "result.err", "result.clamped",
              "result.spread", "result.err_theta", "result.clamped_theta", "quantities"),
    per_arm=False,
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      const have = ui.idx().filter(i => D.cohort.qs[i] !== null);
      const MINB = 8, MINPT = 4, NB = 10;
      if (have.length < 3*MINB)
        return denomLine('only ' + have.length + ' cohorts carry the coordinates this needs.');
      const xs = have.map(i => D.cohort.qs[i]);
      const lo = Math.min(...xs), hi = Math.max(...xs);
      if (hi - lo < 0.3)
        return denomLine('this selection spans only ' + (hi-lo).toFixed(2) + ' dex of the regime '
          + 'axis, too narrow to read a trend from.');
      const rows = []; let thin = 0, thinN = 0;
      for (let b=0;b<NB;b++) {
        const a = lo + (hi-lo)*b/NB, z = lo + (hi-lo)*(b+1)/NB;
        const inb = have.filter(i => D.cohort.qs[i] >= a && (b===NB-1 ? D.cohort.qs[i] <= z
                                                                     : D.cohort.qs[i] < z));
        const npt = new Set(inb.map(i => D.cohort.cluster_key[i])).size;
        // A bin holding seed replicates of one or two truths would draw a trend segment out of a
        // single parameter point.
        if (inb.length < MINB || npt < MINPT) { thin++; thinN += inb.length; continue; }
        let hits=[], errs=[];
        inb.forEach(i => {
          const h = D.result.hit[qi][i], e = errAt(qi, i);
          if (h !== null) hits.push(h);
          if (e !== null && !clampedAt(qi, i)) errs.push(e);
        });
        rows.push({x:(a+z)/2, label:'ratio 10^'+((a+z)/2).toFixed(2),
                   cov: hits.length ? hits.reduce((x,y)=>x+y,0)/hits.length : null,
                   claims: hits.length, points: npt, err: median(errs), nerr: errs.length});
      }
      let note = rows.length + ' of ' + NB + ' bins drawn; a bin needs ' + MINB + '+ cohorts at '
               + MINPT + '+ distinct parameter points';
      if (thin) note += ', so ' + thin + ' thin bin(s) holding ' + thinN + ' cohort(s) are omitted';
      if (comboNote(qi, have)) note += '. ' + comboNote(qi, have);
      return dualAxis(rows, 'beta/((kappa+1)*eta*L)', t => fmtNum(Math.pow(10,t)), note);
    """,
))

register(View(
    view_id="manifold_coverage",
    title="Coverage of the manifold",
    description=("One marker per distinct parameter point -- not per cohort, so replicate seeds "
                 "do not make a thin design look dense. Colour is that point's median accuracy "
                 "for the selected quantity."),
    requires=("cohort.log_ratio", "cohort.cv", "cohort.cluster_key", "result.err",
              "result.spread", "result.err_theta", "result.theta_fill", "quantities"),
    per_arm=False,
    render_js="""
      const q = ui.quantity(); const qi = D.quantities.indexOf(q);
      return manifoldScatter(ui.idx(), qi);
    """,
))
