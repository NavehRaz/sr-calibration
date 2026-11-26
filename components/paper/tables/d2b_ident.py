"""Derived tables for Sections 5 and 7c (identifiability), from stored files only (no fits, no simulations).
Run: python -B d2b_ident.py
Outputs (beside this file):
  d2b_ridge.csv           kfix panel: one row per (cohort, kappa0): loglik, MLE, beta*rbar(kappa0)
  d2b_ridge_cohort.csv    one row per panel cohort: valid flag, spread of the loglik over the kappa0 slices (nats)
  d2b_kfix_panel.csv      kfix REPORT.md panel table: covered counts of the 19 valid cohorts per kappa0 and quantity
  d2b_kfix_net.csv        KAPPA0_TABLE.txt: net coverage change (kappa-fixed minus kappa-free) per variant, kappa0, quantity
  d2b_kappa_declined.csv  v6 reference runs J and H: kappa claim shape counts per arm
"""
import csv, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)

AG = "aging:code_5/"
KFIX = AG + "sr_calibration_tester_kfix/"
TAGS = [("0p001", 0.001), ("0p1", 0.1), ("0p5", 0.5), ("1", 1.0), ("1p5", 1.5), ("2", 2.0), ("300", 300.0)]


def rbar(k):
    return 1.0 - k * math.log(1.0 + 1.0 / k)


def ridge():
    invalid = {r["cohort_id"] for r in csv.DictReader(open(PP.path(KFIX + "reports/validity/validity.csv")))
               if r["flag_invalid"] not in ("False", "0", "false", "")}
    rows = {}
    for tag, k in TAGS:
        for cid, rec in PP.records(KFIX.rstrip("/"), f"s2-kfix_k{tag}-panel"):
            d = rec.get("diagnostics") or (rec.get("claim_set") or {}).get("diagnostics")
            if not d or not d.get("theta_mle"):
                continue
            th = d["theta_mle"]
            rows.setdefault(cid, {})[k] = (d["loglik"], th)
    out, coh = [], []
    for cid, sl in rows.items():
        lmax = max(v[0] for v in sl.values())
        valid = int(cid not in invalid)
        for k, (ll, th) in sorted(sl.items()):
            out.append(dict(cohort_id=cid, valid=valid, kappa_true=re.search(r"-k([0-9.]+)-", cid).group(1),
                            kappa0=k, loglik=ll, dll=ll - lmax, eta=th["eta"], beta=th["beta"],
                            epsilon=th["epsilon"], rbar=rbar(k), beta_rbar=th["beta"] * rbar(k),
                            beta_over_kp1=th["beta"] / (1 + k)))
        coh.append(dict(cohort_id=cid, valid=valid, n_slices=len(sl),
                        dll_span=lmax - min(v[0] for v in sl.values())))
    for name, data in (("d2b_ridge.csv", out), ("d2b_ridge_cohort.csv", coh)):
        with open(os.path.join(HERE, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


def kfix_panel():
    text = open(PP.path(KFIX + "reports/kfix/REPORT.md")).read()
    sec = text.split("**Valid cohorts (19).**")[1].split("- **Coverage**")[0]
    out = []
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 7 or not re.match(r"\d|κ-free", cells[0]):
            continue
        k = cells[0].replace("κ-free", "free").replace("10⁻³", "0.001")
        for q, c in zip(("eta", "epsilon", "sqrt_eta_over_eps", "beta_rbar", "beta_over_kp1"), cells[1:6]):
            out.append(dict(kappa0=k, quantity=q, hits=int(c.split("/")[0]), n=int(c.split("/")[1])))
    with open(os.path.join(HERE, "d2b_kfix_panel.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kappa0", "quantity", "hits", "n"])
        w.writeheader()
        w.writerows(out)


def kfix_net():
    p = AG + "sr_calibration_tester_jax2_rel/reports/rel101/G3_KFIX/BOTH/KAPPA0_TABLE.txt"
    text = open(PP.path(p)).read()
    out = []
    for blk in re.split(r"\n--- ", text)[1:]:
        m = re.match(r"([JH]) kfix kappa0 = ([0-9.]+)", blk)
        if not m:
            continue
        cpu = re.search(r"K5 CPU vs [a-z0-9_]+: median ratio ([0-9.]+), total ([0-9.]+)", blk)
        for qname, key in (("eta", "eta"), ("epsilon", "eps"), ("sqrt_eta_over_eps", r"sqrt\(eta\)/eps"),
                           ("beta_rbar", r"beta\*rbar")):
            r = re.search(rf"\n    {key} +K1 .*?n +(\d+)  L/G +(\d+)/(\d+)  net ([+-]\d+)", blk)
            out.append(dict(variant=m.group(1), kappa0=float(m.group(2)), quantity=qname, n=int(r.group(1)),
                            lost=int(r.group(2)), gained=int(r.group(3)), net=int(r.group(4)),
                            cpu_total=float(cpu.group(2))))
    with open(os.path.join(HERE, "d2b_kfix_net.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def kappa_declined():
    R = AG + "sr_calibration_tester/reports/"
    out = []
    for var, fn in (("J", "jax2-v6-full-bign"), ("H", "jax2h-v6-full-bign")):
        d = json.load(open(PP.path(R + fn + "/explorer_data.json")))
        iq = d["quantities"].index("kappa")
        arms = {}
        for arm, sh in zip(d["cohort"]["arm"], d["result"]["shape"][iq]):
            a = arms.setdefault(arm, [0, 0])
            a[0] += 1
            a[1] += int(sh == 0)
        tot = [sum(v[0] for v in arms.values()), sum(v[1] for v in arms.values())]
        arms["all"] = tot
        for arm, (n, nd) in arms.items():
            out.append(dict(variant=var, arm=arm, n=n, declined=nd, frac=nd / n))
    with open(os.path.join(HERE, "d2b_kappa_declined.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    ridge()
    kfix_panel()
    kfix_net()
    kappa_declined()
    print("wrote d2b_*.csv")
