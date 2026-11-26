"""Spec 003g's per-window machinery (research R3, R4; contracts/acceptance.md): pin a window's runs, records and witness
files; load them back only through the pins; score a stratum under PR-1, PR-3 and PR-4 with the tester's own coverage
and information statistics. Read-only on the tester checkout, its store and jax2's witness mirror; the pinned tester
clone is made outside any merge window. No v0.3.3 v6 record is read here (the predecessor phase does that, sealed).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess

from sraging import default_gate as G

QUANTITIES_PR4 = G.PR4_QUANTITIES


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_path(store_root: str, key: str) -> str:
    """srbench.store.ResultStore._path."""
    return os.path.join(store_root, key[:2], key[2:4], f"{key}.json")


def ensure_clone(tester_root: str, clone: str, head: str) -> str:
    """A read-only clone of the tester at `head` (003e's connect_pinned then refuses another HEAD or a dirty tree)."""
    if not os.path.isdir(clone):
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", tester_root, clone], check=True)
        subprocess.run(["git", "-C", clone, "checkout", "--quiet", "--detach", head], check=True)
    return clone


def pin_window(data: dict, *, window: str, runs: list, witness_name: str | None, tester_root: str, mirror: str,
               library: str = "v6", head: str, clone_rel: str, exported: dict | None = None) -> dict:
    """Append one window's pins to `data` (in place) and return the counts. Append-only: a run, record or witness file
    already pinned with other bytes is refused (ValueError). A missing null run is recorded as absent."""
    store = os.path.join(tester_root, "data", "store")
    counts = {}
    for run in runs:
        rel = os.path.join("runs", f"{run}.json")
        p = os.path.join(tester_root, rel)
        if not os.path.exists(p):
            if run.startswith("null_prior"):
                counts[run] = "absent"
                continue
            raise ValueError(f"{rel} does not exist: the window has not merged it")
        b = open(p, "rb").read()
        sha = hashlib.sha256(b).hexdigest()
        man = json.loads(b)
        if man.get("library_version") != library:
            raise ValueError(f"{run} is labelled {man.get('library_version')}, not {library}")
        old = data["runs"].get(run)
        if old and old["sha256"] != sha:
            raise ValueError(f"{run} was pinned as {old['sha256'][:12]}; it is now {sha[:12]} (append-only)")
        data["runs"][run] = {"path": rel, "sha256": sha, "library_version": library, "n_keys": len(man["keys"]),
                             "window": window, "method": man.get("method"), "method_version": man.get("method_version")}
        recs = data["records"].setdefault(run, {})
        missing = []
        for cid, key in man["keys"].items():
            rp = store_path(store, key)
            if not os.path.exists(rp):
                missing.append(cid)
                continue
            s = file_sha(rp)
            if cid in recs and (recs[cid]["sha256"] != s or recs[cid]["key"] != key):
                raise ValueError(f"{run}/{cid}: the record differs from its pin (append-only)")
            recs[cid] = {"key": key, "sha256": s}
        n_w = 0
        if witness_name and not run.startswith("null_prior"):
            wit = data["witness"].setdefault(witness_name, {})
            src = (exported or {}).get(witness_name)
            for cid in man["keys"]:
                if src is not None:                  # jax2's export: already verified against its SHA256SUMS
                    s = src.get(cid)
                else:
                    wp = os.path.join(mirror, witness_name, f"{cid}.json")
                    s = file_sha(wp) if os.path.exists(wp) else None
                if s is None:
                    continue
                if cid in wit and wit[cid] != s:
                    raise ValueError(f"witness {witness_name}/{cid} differs from its pin (append-only)")
                wit[cid] = s
                n_w += 1
        counts[run] = {"keys": len(man["keys"]), "records": len(man["keys"]) - len(missing),
                       "missing_records": missing[:20], "n_missing_records": len(missing), "witness_files": n_w}
    data["tester"][window] = {"head": head, "clone": clone_rel}
    return counts


def witness_rows(rows: list, data: dict, name: str, keys: dict, mirror: str) -> list:
    """PR-1's input: each panel row with its witness read only through the pins (tools/witness_003g)."""
    import witness_003g as W
    out = []
    for r in rows:
        cid = r["cohort_id"]
        w = W.read_witness(os.path.join(mirror, name, f"{cid}.json"), name=name, cohort_id=cid,
                           pinned_sha256=(data["witness"].get(name) or {}).get(cid), pinned_key=keys.get(cid))
        out.append({"cohort_id": cid, "valid": r["valid"], "informative": r["informative"],
                    "witness": {"W": w["W"], "fresh": w["fresh"]} if w.get("W") is not None else
                    {"W": None, "fresh": False}, "witness_detail": w})
    return out


def info_widths(V, j_sets: list, null_sets: list, label: str) -> dict:
    """PR-4's input, per quantity: the tester's information axis with the null as the reference side, read for its
    WIDTH comparison only (Naveh, 2026-10-22): "wins" iff J's median closed-interval width is below the null's by more
    than the tester's WIDTH_TOLERANCE_DEX. The axis's closed-share rule is not applied -- the null gives a closed claim
    (the whole box) on every cohort, so any decline would make J "lose" -- and J's closed share is reported beside."""
    out = {}
    tol = float(V.WIDTH_TOLERANCE_DEX)
    for q in QUANTITIES_PR4:
        ax = V.information_axis(j_sets, null_sets, q, label).to_dict()
        det = ax.get("detail") or {}
        ma, mb = det.get("method_median_dex"), det.get("reference_median_dex")
        width = ("not measured" if ma is None or mb is None else
                 "wins" if ma - mb < -tol else "loses" if ma - mb > tol else "no difference")
        out[q] = {"j": ma, "null": mb, "outcome": width, "axis_outcome": ax.get("outcome"),
                  "closed_share_j": det.get("method_closed_share"), "closed_share_null": det.get("reference_closed_share"),
                  "shape_mix_j": det.get("method_shape_mix"), "n_closed_j": ax.get("denominator"),
                  "width_tolerance_dex": tol}
    return out


def coverage_counts(CL, sets: list, truths: dict, quantity: str) -> tuple[int, int]:
    """(hits, claimed) of closed claims, the tester's own coverage (score.claims.coverage)."""
    c = CL.coverage(sets, truths, quantity, "closed").to_dict()
    return int(c["hits"]), int(c["claimed"])


def mechanisms(sets: list) -> dict:
    """The Db bound's counts: fits where 1.0.1 left 1.0.0's path (DT_RULE in scope, the censor-tie relabel applied, the
    basin re-fit fired), from the records' own diagnostics."""
    n = {"n": 0, "dt_in_scope": 0, "censor_tie_applied": 0, "basin_fix_fired": 0}
    for cs in sets:
        d = getattr(cs, "diagnostics", None) or {}
        n["n"] += 1
        n["dt_in_scope"] += bool(((d.get("dt_floor") or {}).get("grid_rule") or {}).get("dt_in_scope"))
        n["censor_tie_applied"] += bool((d.get("censor_tie") or {}).get("applied"))
        n["basin_fix_fired"] += bool((d.get("basin_fix") or {}).get("fired"))
    return n


def srk_diff(files: dict, oracle_dir: str) -> dict:
    """FR-010: each SRK file a record says its fit loaded ({path: sha256}) against sraging's v0.3.3 oracle copy, by
    basename. {n_files, different: [paths], missing_in_oracle: [paths]}."""
    diff, missing = [], []
    for f, s in sorted(files.items()):
        o = os.path.join(oracle_dir, os.path.basename(f))
        if not os.path.exists(o):
            missing.append(f)
        elif file_sha(o) != s:
            diff.append(f)
    return {"n_files": len(files), "different": diff, "missing_in_oracle": missing}


END_DEX_REPORT = 0.03          # 003e's jcross threshold for an interval end's |dex| difference (reported)
RAW_PARAMETERS = ("eta", "beta", "epsilon", "kappa")


def claim_compare(mine, theirs, truth: dict) -> dict:
    """SC-007, fit 1 against its W3 record: per parameter, the claim shape, whether it covers the truth, and each end's
    |dex| difference. Gate: shapes equal and coverage hits equal, and no raw parameter claimed on one side only (T067);
    ends beyond END_DEX_REPORT are reported. Claims are a list, or a sraging result's {parameter: claim} (T067, B1)."""
    import math

    def by(cl):
        return {c["parameter"]: c for c in (cl.values() if isinstance(cl, dict) else cl)}

    def hit(c, t):
        if t is None or c.get("shape") == "none":
            return None
        lo, hi = c.get("lower"), c.get("upper")
        return (lo is None or t >= lo) and (hi is None or t <= hi)
    a, b = by(mine), by(theirs)
    rows = []
    for p in sorted(set(a) & set(b)):           # the parameters both sides claim; the rest are listed below
        x, y = a.get(p) or {}, b.get(p) or {}
        ends = {}
        for k in ("lower", "upper"):
            if x.get(k) and y.get(k):
                ends[k] = abs(math.log10(x[k]) - math.log10(y[k]))
        rows.append({"parameter": p, "shape_mine": x.get("shape"), "shape_record": y.get("shape"),
                     "shape_equal": x.get("shape") == y.get("shape"),
                     "hit_mine": hit(x, truth.get(p)), "hit_record": hit(y, truth.get(p)),
                     "hit_equal": hit(x, truth.get(p)) == hit(y, truth.get(p)), "end_dex": ends,
                     "ends_beyond": [k for k, v in ends.items() if v > END_DEX_REPORT]})
    raw_one_side = sorted((set(a) ^ set(b)) & set(RAW_PARAMETERS))
    return {"rows": rows, "only_mine": sorted(set(a) - set(b)), "only_record": sorted(set(b) - set(a)),
            "raw_one_side": raw_one_side, "shapes_equal": all(r["shape_equal"] for r in rows),
            "hits_equal": all(r["hit_equal"] for r in rows),
            "n_ends_beyond": sum(len(r["ends_beyond"]) for r in rows),
            "pass": not raw_one_side and all(r["shape_equal"] and r["hit_equal"] for r in rows)}


def p15_rows(p15: dict, meta: dict) -> list:
    """PR-5's rows for BASIN-P15 (T067, B2): p15_v033's theta-only v0.3.3 port fits, W33 = W4 in 1.0.1's likelihood,
    on the valid informative cohorts only (as every other PR-5 stratum). `meta`: {cohort_id: {valid, informative}}."""
    out = []
    for r in p15.get("rows") or []:
        m = meta.get(r["cohort_id"]) or {}
        if not (m.get("valid") and m.get("informative")):
            continue
        w = r.get("witness") or {}
        out.append({"cohort_id": r["cohort_id"], "W33": w.get("W4"), "shown": r.get("shown"), "source": "p15_v033"})
    return out


#: the tester's scoring code; inside (W1) and outside (W3) verdicts must come from the same (the interim review)
SCORER_FILES = ("srbench/score/claims.py", "srbench/score/verdict.py", "srbench/score/pairing.py")


def scorer_diff(repo: str, head_a: str, head_b: str, files=SCORER_FILES) -> dict:
    """Each scorer file's blob at two tester heads (read through git objects, no checkout)."""
    def blob(head, f):
        r = subprocess.run(["git", "--no-optional-locks", "-C", repo, "rev-parse", f"{head}:{f}"], capture_output=True,
                           text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    out = {f: {"a": blob(head_a, f), "b": blob(head_b, f)} for f in files}
    return {"head_a": head_a, "head_b": head_b, "files": out,
            "differs": sorted(f for f, x in out.items() if x["a"] != x["b"] or x["a"] is None)}


def scorer_blocking(block) -> list:
    """The gate's entries for the scorer check: blocking when score-w3 did not record it or a file differs."""
    if not block:
        return [{"stratum": "ALL", "rule": "scorer", "verdict": "missing",
                 "reasons": ["score-w3 did not record the scorer comparison with W1's tester head"]}]
    if block["differs"]:
        return [{"stratum": "ALL", "rule": "scorer", "verdict": "differs",
                 "reasons": [f"the tester's scorer differs between W1 ({block['head_a'][:8]}) and W3 "
                             f"({block['head_b'][:8]}): {', '.join(block['differs'])}"]}]
    return []


def export_witness(export_dir: str, sums_sha256: str, names: tuple) -> dict:
    """jax2's per-window witness export (sesync cd40, 2026-10-22): export_v6/<W>/witness/ with a SHA256SUMS manifest
    whose sha256 jax2 sends in the window's WINDOW ENDS message. Refused unless the manifest is that file and every
    listed file has its listed bytes. {name: {cohort_id: sha256}}; a file's name is its parent directory when that is
    one of `names`, else `names[0]` (a flat export)."""
    sums = os.path.join(export_dir, "SHA256SUMS")
    if file_sha(sums) != sums_sha256:
        raise ValueError(f"{sums}: sha256 is not the one jax2 sent ({sums_sha256[:12]}...)")
    out = {}
    for line in open(sums):
        if not line.strip():
            continue
        sha, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        p = os.path.join(export_dir, rel)
        if file_sha(p) != sha:
            raise ValueError(f"{rel}: bytes differ from SHA256SUMS")
        parent = os.path.basename(os.path.dirname(rel))
        name = parent if parent in names else names[0]
        if rel.endswith(".json"):
            out.setdefault(name, {})[os.path.basename(rel)[:-5]] = sha
    return out



# ------------------------------------------------- the interim review (T068-T071): reported beside the verdicts
def with_invalid(rows: list) -> list:
    """FR-005's 'same numbers with the INVALID cohorts included' (T068): the rows with every cohort counted as valid.
    Reported under `with_invalid`, never gating; the input is not changed."""
    return [dict(r, valid=True) for r in rows]


def shape_shares(sets, quantities=("eta", "epsilon", "beta")) -> dict:
    """PR-4's declines (T070): per quantity, J's claims by shape on the sets that ran (`failed` apart). `open_above` =
    lower_bound only (no cap from above: the information PR-4's widths cannot see). On beta, `upper_end_at_kappa_box` =
    upper_bound: by labels.py's rule every raw-beta upper end is the kappa box's, not the data's (spec 003's kappa_max
    study; N1, Naveh 2026-10-29), so it is not a cap."""
    out = {}
    for q in quantities:
        n = {"n": 0, "failed": 0, "closed": 0, "lower_bound": 0, "upper_bound": 0, "none": 0}
        for cs in sets:
            if not cs.ok:
                n["failed"] += 1
                continue
            c = cs.claim_for(q)
            shape = "none" if c is None or not c.is_claim else c.shape
            n["n"] += 1
            n[shape] = n.get(shape, 0) + 1
        n["open_above"] = n["lower_bound"]
        if q == "beta":
            n["upper_end_at_kappa_box"] = n["upper_bound"]
        out[q] = n
    return out


def pr3_power_row(*, k_in: int, n_in: int, n_valid_out: int, n_expected: int, margin: float = G.PR3_MARGIN,
                  z: float = G.Z_ONE_SIDED_95) -> dict:
    """B3's power table row (T069), from W1's inside counts and the panel's N only: for every closed-claim count n_out
    from 1 to the stratum's valid N, the smallest k_out PR-3 passes (None: cannot pass), and whether equal coverage
    (k_out = round(p_in n_out)) passes at the n_out expected from the inside closed share."""
    min_k = [None]
    for n_out in range(1, n_valid_out + 1):            # no closed claim inside: PR-3 reads "incomplete", nothing passes
        k = next((k for k in range(n_out + 1) if G.newcombe_diff(k, n_out, k_in, n_in, z)[0] > -margin), None) \
            if n_in > 0 else None
        min_k.append(k)
    p_in = k_in / n_in if n_in else None
    k_eq = None if p_in is None or not n_expected else round(p_in * n_expected)
    ok = (k_eq is not None and 0 < n_expected <= n_valid_out and min_k[n_expected] is not None
          and k_eq >= min_k[n_expected])
    return {"k_in": k_in, "n_in": n_in, "p_in": p_in, "n_valid_out": n_valid_out, "n_expected": n_expected,
            "k_equal_at_expected": k_eq, "equal_coverage_passes_at_expected": ok, "min_k_out": min_k,
            "margin": margin}


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(K <= k) for K ~ Binomial(n, p), exactly (1.0 at k >= n, not a rounded sum)."""
    if k >= n:
        return 1.0
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, min(k, n) + 1)))


def pr3_note(row: dict, *, n_out: int, k_out: int | None = None, alpha: float = 0.05) -> dict:
    """What the gate prints beside a PR-3 FAIL (T069): 'cannot pass at this N', 'underpowered' (equal coverage fails
    at this N), or, where equal coverage would pass, the post-W3 reported probability of a count this low under exactly
    equal coverage, P(K <= k_out | n_out, p_in) (N3, Naveh 2026-10-29): 'consistent with equal coverage (p = ...)' at
    p >= alpha, else 'worse (p = ...)'. The pre-W3 power row is not changed."""
    mk = row["min_k_out"][n_out] if 0 < n_out < len(row["min_k_out"]) else None
    k_eq = None if row["p_in"] is None else round(row["p_in"] * n_out)
    p_eq = None if row["p_in"] is None or k_out is None or n_out <= 0 else binom_cdf(k_out, n_out, row["p_in"])
    reading = ("no comparator inside the reference set" if row["p_in"] is None else
               "cannot pass at this N" if mk is None else
               "underpowered: equal coverage fails at this N" if k_eq is None or k_eq < mk else
               f"{'consistent with equal coverage' if p_eq is not None and p_eq >= alpha else 'worse'} "
               f"(p = {p_eq:.2f})" if p_eq is not None else "worse")
    return {"n_out": n_out, "k_out": k_out, "min_k_out": mk, "k_equal": k_eq,
            "p_at_most_k_out_if_equal": p_eq, "reading": reading}


def apply_deviations(verdicts: list, deviations: list, *, strata) -> tuple[list, list]:
    """The panel's [deviations] (T071; Naveh 2026-10-28). kind 'verdict': that stratum's rule reads `as`, with the
    deviation's id, date and reason appended; refused if the stratum is not a panel stratum or the deviation matches no
    verdict. kind 'rule-code': copied to the output, nothing applied. Returns (verdicts, applied); inputs unchanged."""
    out, applied = [dict(v) for v in verdicts], []
    for d in deviations:
        a = dict(d, n_applied=0)
        if d["kind"] == "verdict":
            if d["stratum"] not in strata:
                raise ValueError(f"{d['id']}: unknown stratum {d['stratum']!r}")
            for i, v in enumerate(out):
                if v["stratum"] == d["stratum"] and v["rule"] == d["rule"]:
                    out[i] = dict(v, verdict=d["as"], deviation=d["id"],
                                  reasons=list(v.get("reasons") or []) + [f"{d['id']} ({d['decided']}): {d['reason']}"])
                    a["n_applied"] += 1
            if not a["n_applied"]:
                raise ValueError(f"{d['id']}: {d['stratum']}/{d['rule']} matches no verdict")
        elif d["kind"] != "rule-code":
            raise ValueError(f"{d['id']}: unknown kind {d['kind']!r}")
        applied.append(a)
    return out, applied


def apply_decisions(blocking: list, decisions: list) -> tuple[list, dict]:
    """N9 (Naveh 2026-10-29): the panel's [[decisions]] onto the gate's blocking entries. A decision matches an entry by
    stratum, rule, arm and quantity (absent = None); each decision must match at least one entry. Returns (entries, each
    with `decision` or None, and {"decided": k, "of": n})."""
    out, used = [], set()
    for b in blocking:
        key = (b.get("stratum"), b.get("rule"), b.get("arm"), b.get("quantity"))
        hit = [i for i, d in enumerate(decisions)
               if (d["stratum"], d["rule"], d.get("arm"), d.get("quantity")) == key]
        used.update(hit)
        out.append(dict(b, decision={k: decisions[hit[0]][k] for k in ("decision", "decided", "reason")}
                        if hit else None))
    unused = [d for i, d in enumerate(decisions) if i not in used]
    if unused:
        raise ValueError(f"decisions matching no blocking entry: {[(d['stratum'], d['rule']) for d in unused]}")
    return out, {"decided": sum(1 for b in out if b["decision"]), "of": len(out)}


def closed_widths(sets, quantity: str) -> dict:
    """N4: the median and quartiles (dex; statistics.quantiles, 'exclusive') of J's closed claims' widths on
    `quantity`, and how many. Human widths are bimodal (narrow, and box-set ends), so the median alone misleads."""
    import statistics
    w = sorted(math.log10(c.upper / c.lower) for cs in sets if cs.ok for c in [cs.claim_for(quantity)]
               if c is not None and c.is_claim and c.shape == "closed" and c.lower and c.lower > 0)
    med = None if not w else statistics.median(w)
    q = statistics.quantiles(w, n=4) if len(w) >= 2 else [None, med, None]
    return {"n_closed": len(w), "median_dex": med, "q25_dex": q[0], "q75_dex": q[2]}


def eta_beta_crosstab(sets) -> dict:
    """N4: per cohort, eta's shape against beta's ('none' when no claim), on the sets that ran: is eta's missing upper
    end the production-removal ridge, where beta is undetermined?"""
    def shape(cs, q):
        c = cs.claim_for(q)
        return "none" if c is None or not c.is_claim else c.shape
    tab = {}
    for cs in sets:
        if cs.ok:
            k = f"eta {shape(cs, 'eta')} / beta {shape(cs, 'beta')}"
            tab[k] = tab.get(k, 0) + 1
    return dict(sorted(tab.items()))


def one_sided_coverage(CL, sets, truths: dict, quantities=("eta", "epsilon", "beta")) -> dict:
    """N8 (reported, never gating): the tester's own coverage of J's one-sided claims (score.claims.coverage,
    shape_class 'one_sided') per quantity."""
    out = {}
    for q in quantities:
        c = CL.coverage(sets, truths, q, "one_sided").to_dict()
        out[q] = {"hits": int(c["hits"]), "claimed": int(c["claimed"]), "total": int(c["total"]),
                  "coverage": c["coverage"]}
    return out
