"""Is the example refit with sraging tag v0.1.0 (tables/example_refit_v010.json) the same fit as the one stored with
sraging's examples (read at the released tag v0.1.1; the file and src/ are identical at v0.1.0)? Compares, for exact equality, theta-hat, the log-likelihood, every claim's shape and ends, the two
combination claims and the leverage verdict. The stored fit is read from the tag with `git show` (read-only).

Run: python -B tables/example_refit_check.py   ->  tables/example_refit_check.csv (key,value)
"""
import hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)

HERE = os.path.dirname(os.path.abspath(__file__))
SR = "~/code/sraging"
TAG, PATH = "v0.1.1", "examples/results/itp_c2016_het3_control_female_jax2.json"

raw = PP.git_show(SR, TAG, PATH)
old = json.loads(raw)
new = json.load(open(PP.path("tables/example_refit_v010.json")))

pairs = [(f"theta_hat.{k}", old["theta_hat"][k], new["theta_hat"][k]) for k in ("eta", "beta", "epsilon", "kappa")]
pairs.append(("loglik", old["loglik"], new["loglik"]))
for q, c in old["claims"].items():
    pairs.append((f"claims.{q}.shape", c["shape"], new["claims"][q]["shape"]))
    for e in ("lower", "upper", "point"):
        pairs.append((f"claims.{q}.{e}", c.get(e), new["claims"][q].get(e)))
for q, w in old["combinations"]["walks"].items():
    for e in ("ci_lo", "ci_hi", "argmax_value"):
        pairs.append((f"combinations.{q}.{e}", w["ci"][e], new["combinations"]["walks"][q]["ci"][e]))
lo, ln = old["diagnostics"]["leverage"], new["diagnostics"]["leverage"]
pairs.append(("leverage.label", lo["label"], ln["label"]))
pairs.append(("leverage.share.row", lo["share"]["row"], ln["share"]["row"]))

differ = [k for k, a, b in pairs if a != b]
out = os.path.join(HERE, "example_refit_check.csv")
with open(out, "w") as f:
    f.write("key,value\n")
    f.write(f"n_compared,{len(pairs)}\n")
    f.write(f"n_differ,{len(differ)}\n")
    f.write(f"dloglik,{new['loglik'] - old['loglik']:.6g}\n")
    f.write(f"stored_sha256,{hashlib.sha256(raw).hexdigest()}\n")
    f.write(f"stored_commit,{old['provenance']['package']['git_sha'][:7]}\n")
    f.write(f"refit_cpu_s,{new['timing']['cpu_s']:.6g}\n")
print("wrote", out, "compared", len(pairs), "differ", differ or "none")
