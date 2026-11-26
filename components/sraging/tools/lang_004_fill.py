"""Writes freeze-L, tests/acceptance/langevin_004.toml (tasks T115, T119), from the committed results, Naveh's
2026-11-03 decisions (ff794f5) and the code the file names; every number is read or computed here, never typed.

    .venv/bin/python tools/lang_004_fill.py [--out PATH] [--s5-m 32|64 [--a6-text TEXT]] [--a7-text TEXT]
                                            [--a8-text TEXT] [--a9-text TEXT] [--naveh-ok-l TEXT]

Freeze-L as Naveh decided it at T119 (2026-11-04, spec.md Clarifications 2026-11-04) is the one command
`tools/lang_004_fill.py --s5-m 64 --a6-text A6 --a7-text A7 --naveh-ok-l OK`, with the three texts as written in LANG,
and with A8 (Naveh 2026-11-04 at T121, during the run) `--a8-text A8` and A9 (Naveh 2026-11-05 at T126, after
multi's first run) `--a9-text A9` too; `tools/lang_004_fill.py --from-lang` re-reads those inputs from the LANG file
itself and regenerates it.

P2's values are read from tests/acceptance/results/p2_s5_power_004.json (refused unless it is written with no error
rows). The few values computed from code (S4's and HR-10's plant delta, HR-11's plant sizes and the above-budget node
count) are computed here by the functions LANG names, and cite the commit they ran at: refused when src/ or those
tools differ from HEAD. Each computed value is cached in .cache/lang_004_t119.json, keyed by the git trees of that code
at HEAD, so a re-run at the same code reuses it (first run about 40 CPU-min: HR-11's tensor references ~17, the
above-budget climb ~22, the plants' deltas ~0.5). P3's result is not read: its `auto` boundary is encoded in
src/sraging/population.py (T130). If the above-budget configuration's tensor reference is not above the budget, its
node_count stays a {t119 = ...} reference with the measured statement beside it: an open item, never a typed value.

--s5-m applies amendment A6 (prepared for Naveh's T119 answer, not applied by default): S5 runs at m 32 or 64, dt the
frozen s5 step's x frozen m / m, with s5-het's re-check at that step ([s5].recheck_at_step, P2's per-replicate SD) and
its CPU scaled from P2's. Above gate C2's m_max (2 x the frozen m) it is refused unless the amendment's text (--a6-text,
else drafted here) says p2_s5_power_004.A6_C2_CLAUSE.

--a7-text applies amendment A7: HR-11 (iii) runs the 4-parameter cell A7_CELL, which is above the tensor reference's
d <= 3 limit (research R28), so it reads "no FP reference" without any climb; [multi].above_budget records the
sampler's own reason (sample.prepare, computed here), and no node count. The node-budget path rests on T104's unit test.

--a8-text applies amendment A8: the plants' [plants.recording] (D-408) runs at m = A8_M, SRK at (Delta t, m) and sraging
at Delta t / m recorded on delta; at m 1 sraging steps at Delta t and every death falls on the Delta t grid, so the plant
cannot be informative. Written here, not in tools/presets_004.py, so the computed values' cache keeps its code key.

--a9-text applies amendment A9: HR-11 re-runs once at the fresh key base A9_KEY (its first run, at 40600, is kept as
evidence), and [multi].composite names its bar, A9_BAR: the composite cross-checks the tensor reference built at 1e-4,
not at the cells' crit/10.

--naveh-ok-l writes naveh_ok_l, Naveh's OK of freeze-L (T119), verbatim."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "tools")]
os.chdir(REPO)

import acceptance_004 as ACC  # noqa: E402
import multi_004 as MU  # noqa: E402
import p2_s5_power_004 as P2  # noqa: E402
import paths_004 as PA  # noqa: E402
import presets_004 as PR  # noqa: E402
import s4_fp_langevin_004 as S4  # noqa: E402
from sraging import sampler_rules as R  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", default="tests/acceptance/langevin_004.toml")
ap.add_argument("--s5-m", type=int, choices=(32, 64), help="amendment A6: S5's m (Naveh's answer at T119)")
ap.add_argument("--a6-text", help="A6's text (default: drafted from P2's result); above C2's m_max it must say "
                                  "\"C2's m_max does not bind S5's step\"")
ap.add_argument("--a7-text", help="amendment A7's text: HR-11 (iii) on the 4-parameter cell (Naveh's answer at T119)")
ap.add_argument("--a8-text", help="amendment A8's text: the plants' recording at m = A8_M (Naveh's answer at T121)")
ap.add_argument("--a9-text", help="amendment A9's text: multi's re-run at A9_KEY, its composite at A9_BAR (Naveh's "
                                  "answer at T126)")
ap.add_argument("--naveh-ok-l", default="", help="naveh_ok_l: Naveh's OK of freeze-L, '<date>: <his words>' (T119)")
ap.add_argument("--from-lang", action="store_true", help="take --s5-m, --a6-text to --a9-text and --naveh-ok-l "
                                                         "from the LANG file at --out (its decisions, regenerated)")
args = ap.parse_args()
if args.from_lang:
    if args.s5_m or args.a6_text or args.a7_text or args.a8_text or args.a9_text or args.naveh_ok_l:
        sys.exit("refused: --from-lang takes the decisions from LANG; give them there or on the command line, not both")
    import tomllib
    with open(os.path.join(REPO, args.out), "rb") as fh:
        _prev = tomllib.load(fh)
    _am = _prev.get("amendments") or {}
    args.s5_m = (_prev.get("s5") or {}).get("m")
    args.a6_text, args.a7_text, args.a8_text, args.a9_text = (_am.get("A6"), _am.get("A7"), _am.get("A8"),
                                                             _am.get("A9"))
    args.naveh_ok_l = _prev.get("naveh_ok_l", "")
if args.a6_text and not args.s5_m:
    sys.exit("refused: --a6-text without --s5-m")
A7_CELL = {"members": {"xc": 0.2, "eta": 0.2, "beta": 0.2, "epsilon": 0.2}, "n": 25000}   # spec.md Clarif. 2026-11-04
A8_M = 16                                  # the plants' recording m (A8, spec.md Clarifications 2026-11-04, T121)
A9_KEY = 60600                             # multi's re-run key base (A9, spec.md Clarifications 2026-11-05, T126)
A9_BAR = 0.0001                            # the composite's tensor reference bar (A9): HR-11's composite, 1e-4

RES = "tests/acceptance/results"
T119_CODE = ("src", "tools/s4_fp_langevin_004.py", "tools/paths_004.py", "tools/multi_004.py", "tools/presets_004.py")
T119_EXCLUDE = ("src/sraging/sampler_rules.py",)   # the rules: no computed value's function imports them (T119 check)
PATHSPEC = list(T119_CODE) + [f":(exclude){x}" for x in T119_EXCLUDE]
CACHE = os.path.join(REPO, ".cache", "lang_004_t119.json")


def git(*a) -> str:
    return subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True, text=True).stdout.strip()


if subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", *PATHSPEC]).returncode != 0:
    sys.exit(f"refused: {', '.join(T119_CODE)} (but {', '.join(T119_EXCLUDE)}) differ from HEAD; commit them so each "
             f"computed value cites a commit")
AT = git("rev-parse", "--short", "HEAD")


def last_commit(path) -> str:
    return git("log", "-1", "--format=%h", "--", path)


def _tree_key(path) -> str:
    """The code's identity at HEAD: the blob listing of `path` without T119_EXCLUDE, hashed (a file's own blob)."""
    rows = [r for r in git("ls-tree", "-r", "HEAD", "--", path).splitlines() if r.split("\t")[-1] not in T119_EXCLUDE]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


CODE_KEY = {path: _tree_key(path) for path in T119_CODE}
_OLD_KEY = {path: git("rev-parse", f"HEAD:{path}") for path in T119_CODE}     # the cache's first key: whole trees
try:
    with open(CACHE) as fh:
        _file = json.load(fh)
    _cache = {k: (v if isinstance(v, dict) and set(v) == {"value", "at"} else {"value": v, "at": _file["at"]})
              for k, v in _file["values"].items()} if _file.get("code") in (CODE_KEY, _OLD_KEY) else {}
    if _cache and _file.get("code") != CODE_KEY:                     # re-keyed once, to survive a rules-only commit
        with open(CACHE, "w") as fh:
            json.dump({"code": CODE_KEY, "at": _file["at"], "values": _cache}, fh, indent=1)
except (OSError, ValueError, KeyError):
    _cache = {}
RAN_AT = []                                        # the commits the computed values ran at (the code's trees equal)


def computed(label, key, f):
    """f() at this code, from the cache when the code's trees are the cached ones (cited at the commit it ran at, the
    same code); else computed now, at HEAD, and cached. `computed.at` is the last value's commit."""
    k = json.dumps(key)
    if k not in _cache:
        c0 = time.process_time()
        v = f()
        print(f"[lang] {label}: {time.process_time() - c0:.1f} CPU-s", flush=True)
        _cache[k] = {"value": v, "at": AT}
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as fh:
            json.dump({"code": CODE_KEY, "at": AT, "values": _cache}, fh, indent=1)
    else:
        print(f"[lang] {label}: cached, ran at {_cache[k]['at']} (the same code)", flush=True)
    computed.at = _cache[k]["at"]
    RAN_AT.append(computed.at)
    return _cache[k]["value"]


def above_budget_count(members: dict, n: int, t_max: float) -> dict:
    """The above-budget configuration's tensor reference as the sampler builds it (sample.prepare: R26 per dimension
    at crit_95 / sqrt(n) / 10, budget tensor_reference.BUDGET), and the first node product of its climb above the
    budget: its own count when the climb stopped there, else the first trial rung (parameter order) past it."""
    from sraging.engine import tensor_reference as TR
    from sraging.langevin import sample as LS
    cfg = LS.LangevinConfig(params=PR.preset("mice_F"), n=n, key=0, dt=1.0, heterogeneity=MU.members(members),
                            error="ks", t_max=t_max)
    tr = LS.prepare(cfg)["tensor"]
    Ms, verdict = dict(tr.check["M"]), tr.check["verdict"]
    first = None
    if verdict == "over budget":
        first = tr.node_count if tr.node_count > tr.budget else None
        for p in tr.parameters:
            if first is None and tr.sd[p] > 0 and TR._next_rung(Ms[p]) is not None:
                trial = TR.TensorReference._count({**Ms, p: TR._next_rung(Ms[p])})
                first = trial if trial > tr.budget else None
    return {"above": verdict == "over budget", "node_count": first, "kind": tr.kind, "verdict": verdict, "M": Ms,
            "nodes": tr.node_count, "budget": tr.budget, "bar": tr.check["bar"]}


def above_budget_reason(members: dict, n: int, t_max: float) -> dict:
    """A7's cell as the sampler builds its reference (sample.prepare): its kind and the reason it has no FP reference."""
    from sraging.langevin import sample as LS
    cfg = LS.LangevinConfig(params=PR.preset("mice_F"), n=n, key=0, dt=1.0, heterogeneity=MU.members(members),
                            error="ks", t_max=t_max)
    st = LS.prepare(cfg)
    return {"kind": st["ref_kind"], "reason": st["no_reference"]}


p1 = json.load(open(f"{RES}/p1_langevin_004.json"))
pa = json.load(open(f"{RES}/p1_pooled_004.json"))
pb = json.load(open(f"{RES}/p1_pooled_b_004.json"))
P2_PATH = f"{RES}/p2_s5_power_004.json"
p2 = json.load(open(P2_PATH))
if p2["status"] != "written" or p2["errors"]:
    sys.exit(f"refused: {P2_PATH} reads {p2['status']} with {len(p2['errors'])} error rows")
P2_AT = last_commit(P2_PATH)
view = ACC.p1_steps()
fs = view["steps"]
A6 = None
if args.s5_m:
    _w6 = p2["branch"]["wall"]
    _st6 = dict.__getitem__(fs, f"s5|het|{_w6}")
    _b6 = p2["halving"][_w6]["bias_in_se"]
    _text6 = args.a6_text or (
        f"HR-7 / S5: S5 runs at m {args.s5_m}, dt = [judged.steps.\"s5|het|{_w6}\"].dt x {_st6['m']} / {args.s5_m}, not "
        f"at its frozen m {_st6['m']}: P2's step bias there, as A5's upper bound, is {min(_b6):.2f}-{max(_b6):.2f} SE in "
        f"every component; s5-het re-checks the bias at its own step by P2's coupled halving before its replicates "
        f"([s5].recheck_at_step) and cites that bias")
    try:
        A6 = {**P2.a6_step(_st6["m"], _st6["dt"], args.s5_m, _text6), "text": _text6, "frozen_m": int(_st6["m"])}
    except ValueError as ex:
        sys.exit(f"refused: {ex}")
DEC = "spec.md Clarifications 2026-11-03 (ff794f5)"
REV = "specs/004-samplers/review/sr-naveh_rules_freezeL.md (b3ae194)"
EXCLUDED = ["ecoli"]
LIMIT = "the default clamp is not shown to agree with FP for E. coli at any affordable step"


def q(x):
    return json.dumps(x)


def inline(d):
    return "{ " + ", ".join(f"{k} = {inline(v) if isinstance(v, dict) else q(v)}" for k, v in d.items()) + " }"


AMENDS = "A0-A5" + (", A6" if A6 else "") + (", A7" if args.a7_text else "") + (", A8" if args.a8_text else "") + \
    (", A9" if args.a9_text else "")
AMENDS = {"A0-A5, A6, A7, A8, A9": "A0-A9", "A0-A5, A6, A7, A8": "A0-A8", "A0-A5, A6, A7": "A0-A7",
          "A0-A5, A6": "A0-A6"}.get(AMENDS, AMENDS)
OK_LINE = ("naveh_ok_l records Naveh's OK of freeze-L (T119)." if args.naveh_ok_l else
           "naveh_ok_l is empty: no frozen Langevin phase runs from this file until T119.")
L = []
w = L.append
w(f"""# Spec 004's freeze-L (tasks T115; data-model "freeze-L, langevin_004.toml"; contracts/register-and-panel.md §2-§3;
# contracts/acceptance.md; research R14-R30), with Naveh's decisions of 2026-11-03 on sr-naveh's findings and the
# draft's open items ({DEC}; {REV}): the amendments {AMENDS} below are named in contracts/register-and-panel.md §3.
# {OK_LINE} Every value from a committed result
# carries its source (file#key), P2's from {P2_PATH} ({P2_AT}). The few computed from
# code (the plants' delta, HR-11's plant sizes and the above-budget {"cell's reason" if args.a7_text else "node count"}) name the function and the commit it ran
# at (@@RAN_AT@@). P3's result is not in this file: its `auto` boundary is encoded in src/sraging/population.py (T130).
# Written by tools/lang_004_fill.py from the results and that code, so no number is typed.

naveh_ok_l = {q(args.naveh_ok_l)}
""")

# ------------------------------------------------------------------------------------------------- [limits]
w(f"""[limits]   # stated in the README (T136) and in every affected result ({DEC})
ecoli_clamp = {q(LIMIT)}
""")

# ------------------------------------------------------------------------------------------------- [amendments]
w(f"""[amendments]   # of contracts/register-and-panel.md §3 and of the 2026-10-30 decision, before the run ({DEC})
A0 = {q("E. coli leaves the clamp family: every E. coli clamp row leaves S4's clamp family and HR-10's clamp tests; it stays in the reflect family ([limits].ecoli_clamp)")}
A1 = {q("HR-6, HR-10, HR-11 at frozen steps (research R18's branch: P1's floor exceeds crit/30): each row runs ONCE at its frozen step, with no per-row ladder; its assessability is its frozen cell's pooled bias, so m <= m_max (2 x the frozen m) is vacuous; HR-11 checks its own bar (1.655 / sqrt(n) / 10); a borrowed row (a step measured on another configuration) reads unassessable when its own coupled D exceeds 3 x its cell's mean per-key D")}
A2 = {q("HR-10: E. coli has no clamp configuration; the clamp path tests that exceed crit at P1's m 64 are judged on reflect only (see [paths].reflect_only); the other clamp tests run at their frozen steps; assessability is the death-time frozen step's pooled bias, and the path bias is reported, not gating")}
A3 = {q("HR-6: G is each family's actual row count, from this file's grid ([s4].G), and crit = kstwobign.isf(0.05 / G) / sqrt(n)")}
A4 = {q("HR-9: the fix arm runs sraging at mice_F hom clamp's frozen step, recorded through the same Delta t grid")}
A5 = {q("HR-7 / gate C5: S5's step bias is the halving difference x 1/(1 - 2^-p), p from S5's p1-pooled cell, reported as an upper bound ((|mean| + 2 SE) x the factor)")}""")
if A6:
    w(f'A6 = {q(A6["text"])}   # tools/lang_004_fill.py --s5-m {A6["m"]}')
if args.a7_text:
    w(f'A7 = {q(args.a7_text)}   # tools/lang_004_fill.py --a7-text')
if args.a8_text:
    w(f'A8 = {q(args.a8_text)}   # tools/lang_004_fill.py --a8-text')
if args.a9_text:
    w(f'A9 = {q(args.a9_text)}   # tools/lang_004_fill.py --a9-text')
w("")

# ------------------------------------------------------------------------------------------------- [rules]
w("""[rules]   # src/sraging/sampler_rules.py (HR-6 to HR-11 freeze here; a later change is named in RULE_CHANGES)""")
w(f'"HR-6" = {{ G = "per family, [s4].G (A3)", bar = {R.HR6_BAR}, cap = {R.HR6_CAP}, guard_x = {R.GUARD_X} }}'
  f'   # assessable iff the frozen pooled b <= crit / bar (A1)')
w(f'"HR-7" = {{ alpha = {R.HR7_ALPHA}, components = 4 }}   # two-sided Bonferroni over the score components')
w('"HR-8" = { W = [1, 7, 15], n = 25013 }   # research R17')
w(f'"HR-9" = {{ alpha = {inline(R.HR9_ALPHA)} }}')
w(f'"HR-10" = {{ crit = {R.HR10_CRIT}, G = {R.HR10_G}, bar = {R.HR10_BAR}, cap = {R.HR10_CAP}, u_min = {R.HR10_U_MIN}, '
  f'u_lump = {R.HR10_U_LUMP}, x_steps = {R.HR10_X_STEPS} }}   # G stays 16 per wall: conservative where fewer are judged')
w(f'"HR-11" = {{ crit = {R.HR11_CRIT}, G = {R.HR11_G}, bar = {R.HR11_BAR}, detect = {R.HR11_DETECT}, '
  f'informative = {R.HR11_INFORMATIVE}, composite = {R.HR11_COMPOSITE}, budget = 20000, cap = {R.HR11_CAP}, '
  f'guard_x = {R.GUARD_X} }}')
w("")

# ------------------------------------------------------------------------------------------------- [judged]
nf = p1["noise_floor"]
w("""[judged]   # gate C1 (research R18, "the noise floor"); Naveh 2026-11-01 (8f0f269): steps measured at pooled n; A1""")
w('rung = "dt/2"   # the judged sample: the dt/2 rung of the row\'s one coupled draw')
w('branch = "frozen from P1 at pooled n"   # P1\'s floor exceeds crit/30: each step frozen, not re-measured (A1)')
w(f'noise_floor_p1 = {{ "mice_F|hom|clamp" = {nf[0]["floor"]}, "human|het|clamp" = {nf[1]["floor"]} }}   '
  f'# {RES}/p1_langevin_004.json#noise_floor (2 cells x 3 keys, m 16, n 25,000)')
w(f'crit_over_30 = {nf[0]["crit_over_30"]}   # p1_langevin_004.json#noise_floor[0].crit_over_30 (clamp)')
w(f'pooled = {{ K = {pa["K"]}, n_per_key = 25000, n_pooled = {25000 * pa["K"]} }}   # p1_pooled_004.json#K')
w('m_max_rule = "2 x the frozen m (gate C2); vacuous at a frozen step (A1)"')
w("")
w("[judged.p_hat]   # each cell's weak order in [1/2, 1]: P1's fit (p1_langevin_004.json#p_hat); S5's from its own ladder "
  "(p1_pooled_b_004.json#frozen)")
cells = {}
for src, d in (("p1_pooled_004.json", pa), ("p1_pooled_b_004.json", pb)):
    for k, v in d["frozen"].items():
        cells[k] = (src, v, d["cells"][k])
for k, (src, v, c) in cells.items():
    w(f'{q(k)} = {v["p_hat"]}')
w("")
for k, (src, v, c) in cells.items():
    last = c["tried"][-1]
    Dk = last["D_per_key"]
    w(f"[judged.steps.{q(k)}]   # {RES}/{src}#frozen.{k}, #cells.{k}.tried[-1]")
    if v["met"]:
        w(f'm = {v["m"]}')
        w(f'dt = {v["dt"]}')
        w(f'm_max = {2 * v["m"]}')
        w(f'steps_per_individual = {v["steps_per_individual"]}')
        w(f'draw_cpu_s = {v["draw_cpu_s"]}   # median CPU of one key\'s coupled pair (dt and dt/2 rungs), under W = 5')
    w(f'met = {str(v["met"]).lower()}' + ("   # no step: E. coli leaves the clamp family ([limits])" if not v["met"]
                                          else ""))
    w(f'pooled_b = {last["b"]}')
    w(f'bar = {last["bar"]}')
    w(f'pooled_noise_floor = {last["noise_floor"]}')
    w(f'D_key_mean = {sum(Dk) / len(Dk)}   # the guard\'s reference (A1): mean of #cells.{k}.tried[-1].D_per_key')
    w(f'tried_m = {[t["m"] for t in c["tried"]]}')
    w("")

# ------------------------------------------------------------------------------------------------- [s4]
s4 = {"het_presets": list(PR.NAMES), "parameters": ["eta", "beta", "epsilon", "kappa", "xc"], "cvs": [0.1, 0.2, 0.3],
      "v0_presets": ["Jack_Russell", "ecoli"], "n": 25000, "clamp_excluded": EXCLUDED}
grid = S4.build_grid(s4, view)
G = {wl: len(grid[wl]) for wl in ("clamp", "reflect")}
w(f"""[s4]   # research R18; tasks T077, T081; A1, A3; {DEC}""")
w(f'parameters = {q(s4["parameters"])}')
w(f'het_presets = {q(s4["het_presets"])}')
w(f'cvs = {q(s4["cvs"])}   # item 5 ({DEC})')
w(f'v0_presets = {q(s4["v0_presets"])}   # C13')
w(f'clamp_excluded = {q(EXCLUDED)}   # [limits].ecoli_clamp: E. coli\'s clamp rows leave the family; its reflect rows stay')
w('n = 25000')
w(f'G = {inline(G)}   # A3: the actual rows per family, from S4.build_grid over this table and the frozen steps')
w(f'crit_x = {inline({wl: R.family_crit(g) for wl, g in G.items()})}   # kstwobign.isf(0.05 / G)')
w('borrowed = "every heterogeneous row other than xc cv 0.2 borrows its preset\'s xc-0.2 step (item 6), guarded (A1)"')
w('bar = "crit / 10"')
w('key = 40400')
PLANT_CELLS = ["Jack_Russell/v0/clamp", "mice_F/xc/cv0.2/clamp"]
w(f'plant_cells = {q(PLANT_CELLS)}   # item 7')
w(f'plant_target_sup_dS = {S4.PLANT_TARGET}   # 2 x 0.0165 (gate C2)')
by_id = {r["id"]: r for r in grid["clamp"]}
s4_plants = {}
for pid in PLANT_CELLS:
    row = by_id[pid]
    s4_plants[pid] = computed(f"s4 plant_delta {pid}", ["s4_fp_langevin_004.plant_delta", pid],
                              lambda: S4.plant_delta(PR.preset(row["preset"]), S4._het(row["parameter"], row["cv"])))
w('plant_delta = { ' + ", ".join(f'{q(pid)} = {d["delta"]}' for pid, d in s4_plants.items()) + ' }   '
  f'# s4_fp_langevin_004.plant_delta(params, het)["delta"] per plant cell, at {computed.at}: eta (1 + delta) whose predicted '
  f'sup|dS| is the target')
w('plant_predicted_sup_dS = { ' + ", ".join(f'{q(pid)} = {d["predicted_sup_dS"]}' for pid, d in s4_plants.items())
  + " }   # the same calls' predicted_sup_dS, at " + computed.at)
w("")

# ------------------------------------------------------------------------------------------------- [s5]
s5 = PR.role("s5")
w(f"""[s5]   # research R20, R21; gate C5; spec.md Clarifications 2026-11-01 (8f0f269); A5{"; A6" if A6 else ""}; P2's values from {P2_PATH} ({P2_AT})""")
w(f'spec = {inline(s5["spec"])}   # tools/presets_004.py ROLES["s5"]')
w(f'key = {s5["key"]}')
w(f'p_hat = {inline({wl: dict.__getitem__(fs, f"s5|het|{wl}")["p_hat"] for wl in ("clamp", "reflect")})}   '
  f'# p1_pooled_b_004.json#frozen.s5|het|*.p_hat')
wall5 = p2["branch"]["wall"]
h5 = p2["halving"][wall5]
dt5 = dict.__getitem__(fs, f"s5|het|{wall5}")["dt"]
if dt5 != h5["dt"]:
    sys.exit(f"refused: P2's halving ran at dt {h5['dt']}, not s5|het|{wall5}'s frozen step {dt5}")
w(f'R = {p2["R"]["R_score"]}   # p2_s5_power_004.json#R.R_score: any component, Bonferroni over 4, 90 % power')
if A6 is not None:
    S5_AT_A6 = ACC.s5_het_projected_cpu_h({"R": p2["R"]["R_score"], "recheck_at_step": True,
                                           "cpu_per_cohort_s": h5["cpu_per_cohort_s"] * A6["factor"]})
    if p2["branch"]["branch"] != 1 or S5_AT_A6 > P2.CAP_S5_HET:
        sys.exit(f"refused: at A6's m {A6['m']} s5-het projects {S5_AT_A6:.3f} CPU-h against {P2.CAP_S5_HET}: gate "
                 f"C5's branch must be re-decided")
w(f'branch = {inline(p2["branch"])}   # p2_s5_power_004.json#branch: gate C5, the wall and its reason'
  + ("" if A6 is None else
     f'; at A6\'s m {A6["m"]} s5-het projects {S5_AT_A6:.3f} CPU-h with its re-check (<= {P2.CAP_S5_HET:g}), so '
     f'branch 1 holds'))
if A6 is None:
    w(f'dt = {dt5}   # [judged.steps."s5|het|{wall5}"].dt, the frozen step of the wall '
      f'p2_s5_power_004.json#branch.wall picks (= #halving.{wall5}.dt)')
    w(f'cpu_per_cohort_s = {h5["cpu_per_cohort_s"]}   # p2_s5_power_004.json#halving.{wall5}.cpu_per_cohort_s')
    w(f'step_bias_in_se = {h5["bias_in_se"]}   # p2_s5_power_004.json#halving.{wall5}.bias_in_se, '
      f'(rho_eta, rho_beta, rho_eps, kappa); A5: '
      f'an upper bound, (|mean| + 2 SE) x 1/(1 - 2^-p)')
else:
    w(f'm = {A6["m"]}   # A6: S5\'s step, not its frozen m {A6["frozen_m"]} ([judged.steps."s5|het|{wall5}"].m)'
      + ("; above gate C2's m_max " + str(A6["m_max"]) + ", which A6 says does not bind S5's step"
         if A6["above_m_max"] else ""))
    w(f'dt = {A6["dt"]}   # A6: [judged.steps."s5|het|{wall5}"].dt x {A6["frozen_m"]} / {A6["m"]}')
    w(f'cpu_per_cohort_s = {h5["cpu_per_cohort_s"] * A6["factor"]}   # [I] p2_s5_power_004.json#halving.{wall5}.'
      f'cpu_per_cohort_s at m {A6["frozen_m"]} x {A6["factor"]} (the CPU scales with m); the re-check measures it')
    w(f'step_bias_in_se_at_frozen_m = {h5["bias_in_se"]}   # p2_s5_power_004.json#halving.{wall5}.bias_in_se at m '
      f'{A6["frozen_m"]}; the bias cited at m {A6["m"]} is s5-het\'s re-check\'s (A6)')
    w(f'sd_per_replicate = {p2["R"]["sd_per_replicate"]}   # p2_s5_power_004.json#R.sd_per_replicate: the re-check\'s '
      f'SE units at R (A5)')
    w('recheck_at_step = true   # A6: P2\'s coupled halving at this step runs first in s5-het, reusing its law and '
      'gradient')
w(f'plant_sd = {p2["plant"]["plant_sd"]}   # p2_s5_power_004.json#plant.plant_sd')
w(f'smallest_detectable_nats_per_1e5 = {p2["smallest_detectable_nats_per_1e5"]}   '
  f'# p2_s5_power_004.json#smallest_detectable_nats_per_1e5')
w('recheck = "P2\'s coupled halving at S5\'s R is the re-check of S5\'s step (item 3)"' if A6 is None else
  'recheck = "s5-het runs P2\'s coupled halving at its own step before its replicates and cites that bias (item 3, A6)"')
w('od9_fail_response = ["record it in OPEN_TASKS and the register", "send it to the tester\'s owner by sesync", '
  '"state a limit on the v6 heterogeneous cohorts", "no gen-2 in spec 004"]   # OD-9 (a), Naveh 2026-10-29')
w("")

# ------------------------------------------------------------------------------------------------- [paths]
role = PR.role("paths")
reflect_only = []
for c in p1["cells"]:
    if c["wall"] != "clamp":
        continue
    f = max(c["ladder"] + c["extension"], key=lambda r: r["m"])
    idx = [j for j, x in enumerate(f["paths"]) if x["ks_D_censored"] > R.HR10_CRIT / math.sqrt(x["n_alive"])]
    if idx:
        reflect_only.append({"cell": c["id"], "ages": idx,
                             "Dc_m64": [f["paths"][j]["ks_D_censored"] for j in idx],
                             "crit": [R.HR10_CRIT / math.sqrt(f["paths"][j]["n_alive"]) for j in idx]})
ages = {}
for k, a in p1["cell_ages"].items():
    pr_, het, _ = k.split("|")
    ages[f"{pr_}|{het}"] = a
mice2 = p1["cell_ages"]["mice_F|hom|clamp"][1]
w(f"""[paths]   # HR-10; tasks T098; research R24, R25; contracts/acceptance.md §4; A1, A2; {DEC}""")
w(f'presets = {q(role["presets"])}')
w(f'n = {role["n"]}')
w(f'G = {role["G"]}')
w('W = [1, 7, 15]')
w('key = 40500')
w(f'clamp_excluded = {q(EXCLUDED)}   # [limits].ecoli_clamp: no E. coli clamp configuration (A2)')
w('censoring = "u_c = max(0.01, 3 p0); x_c = max(FP quantile at u_c, 5 sqrt(2 eps_max dt))"')
w('walls = ["clamp", "reflect"]   # each configuration once at its frozen step ([judged.steps]; A1)')
w('srk_n = 2000   # item 9')
pd = computed("paths plant_delta", ["paths_004.plant_delta", "mice_F", mice2, int(role["n"])],
              lambda: PA.plant_delta(PR.preset("mice_F"), mice2, int(role["n"]), het=None))
w(f'plant = {{ preset = "mice_F", het = "hom", age = {mice2}, delta = {pd["delta"]} }}   '
  f'# item 9: mice_F hom at its second HR-10 age, {RES}/p1_langevin_004.json#cell_ages.mice_F|hom|clamp[1]; delta: '
  f'paths_004.plant_delta(params, age, n)["delta"] at {computed.at}, predicted sup|dF| {pd["predicted_sup_dF"]} against the '
  f'target {pd["target"]}')
w("")
w(f"[paths.ages]   # item 8: per preset and class, P1's own ages (S near 0.75 and 0.4 on the base grid): "
  f"{RES}/p1_langevin_004.json#cell_ages")
for k, a in ages.items():
    w(f'{q(k)} = {a}')
w("")
w(f"# A2: the clamp path tests whose censored D at P1's m 64 exceeds crit (1.797 / sqrt(n_alive)) are judged on reflect "
  f"only; computed here from {RES}/p1_langevin_004.json#cells[clamp].ladder|extension[m 64].paths")
for r in reflect_only:
    w("[[paths.reflect_only]]")
    w(f'cell = {q(r["cell"])}')
    w(f'ages = {r["ages"]}')
    w(f'Dc_m64 = {r["Dc_m64"]}')
    w(f'crit = {r["crit"]}')
    w("")
w(f"[paths.lump_p0_p1]   # the clamp's lump p0 at the HR-10 ages, P1's finest rung (m 64): "
  f"{RES}/p1_langevin_004.json#cells[].ladder|extension[-1].paths[].p0")
for c in p1["cells"]:
    if c["wall"] != "clamp":
        continue
    f = max(c["ladder"] + c["extension"], key=lambda r: r["m"])
    w(f'{q(c["id"])} = {[x["p0"] for x in f["paths"]]}')
w("")

# ------------------------------------------------------------------------------------------------- [multi]
MEM = (("xc+eta", ("xc", "eta")), ("xc+epsilon", ("xc", "epsilon")), ("eta+beta", ("eta", "beta")),
       ("epsilon+kappa", ("epsilon", "kappa")), ("xc+eta+epsilon", ("xc", "eta", "epsilon")),
       ("eta+beta+kappa", ("eta", "beta", "kappa")))
w(f"""[multi]   # HR-11; tasks T109; research R27-R29; A1{"; A7" if args.a7_text else ""}{"; A9" if args.a9_text else ""}; item 10 ({DEC})""")
w('preset = "mice_F"')
w('cells = [')
for cid, mem in MEM:
    w(f'  {{ id = {q(cid)}, members = {{ {", ".join(f"{m} = 0.2" for m in mem)} }} }},')
w(']')
w('n = 25000')
w('G = 6')
w('budget = 20000')
w(f'key = {A9_KEY}   # A9: the re-run\'s fresh keys (the first run\'s base, 40600, kept as evidence)' if args.a9_text else
  'key = 40600')
w('step = "mice_F|het|clamp"   # borrowed by every cell (A1, guarded); HR-11\'s own bar checked by the rule')
w('composite = { cell = "xc+eta", panels = [16, 32, 64], order = 8, agree = 0.0001'
  + (f', bar = {A9_BAR} }}   # A9: the tensor reference cross-checked at 1e-4, not the cells\' crit/10' if args.a9_text
     else ' }'))
w('plants = [')
for cid, mem in MEM:
    ps = computed(f"multi plant_size {cid}", ["multi_004.plant_size", "mice_F", cid, mem[0]],
                  lambda: MU.plant_size(PR.preset("mice_F"), {m: 0.2 for m in mem}, mem[0]))
    w(f'  {{ cell = {q(cid)}, drop = {q(mem[0])}, predicted_sup_dS = {ps["predicted_sup_dS"]} }},   '
      f'# multi_004.plant_size(params, members, drop) at {computed.at} (bar {ps["bar"]}, M {inline(ps["M"])})')
w(']')
w('reductions = [ { single = { xc = 0.2 }, zero = ["eta"] }, { single = { eta = 0.2 }, zero = ["xc"] } ]')
w('reduction_n = 2000')
w(f'reduction_t_max = {float(PR.PRESETS["mice_F"]["srk_step"]["t_end"])}   # mice_F\'s own horizon (wk)')
T_END_MICE = float(PR.PRESETS["mice_F"]["srk_step"]["t_end"])
if args.a7_text:
    ar = computed("multi above_budget (A7) reason", ["above_budget_reason", {**A7_CELL, "t_max": T_END_MICE}],
                  lambda: above_budget_reason(A7_CELL["members"], A7_CELL["n"], T_END_MICE))
    if ar["kind"] != "none" or not ar["reason"]:
        sys.exit(f"refused: A7's cell reads a {ar['kind']} reference at this code, not \"no FP reference\"")
    w(f'above_budget = {{ members = {inline(A7_CELL["members"])}, n = {A7_CELL["n"]}, t_max = {T_END_MICE}, '
      f'reason = {q(ar["reason"])} }}   # A7: sample.prepare at {computed.at} reads kind "none" (no climb); the '
      f'node-budget path rests on T104\'s unit test')
AB = {"members": {"xc": 0.5, "eta": 0.5, "epsilon": 0.5}, "n": 2000, "t_max": T_END_MICE}
ab = None if args.a7_text else computed("multi above_budget node count", ["above_budget_count", AB],
              lambda: above_budget_count(AB["members"], AB["n"], AB["t_max"]))
ab_where = None if ab is None else (f'sample.prepare\'s TensorReference at {computed.at} (bar crit_95 / sqrt(n) / 10 = {ab["bar"]}): kind {q(ab["kind"])}, '
            f'{ab["verdict"]}, M {inline(ab["M"])}, {ab["nodes"]} nodes')
if ab is None:
    pass
elif ab["above"]:
    w(f'above_budget = {{ members = {inline(AB["members"])}, n = {AB["n"]}, t_max = {AB["t_max"]}, '
      f'node_count = {ab["node_count"]} }}   # the first node product above the budget {ab["budget"]}; {ab_where}')
else:
    w(f'above_budget = {{ members = {inline(AB["members"])}, n = {AB["n"]}, t_max = {AB["t_max"]}, '
      f'node_count = {{ t119 = "tensor_reference: above 20,000 first" }} }}   # OPEN: not above the budget '
      f'{ab["budget"]}; {ab_where}')
w("")

# ------------------------------------------------------------------------------------------------- [plants]
pl = PR.resolve(PR.role("plants"), p1, steps=fs)
w(f"""[plants]   # research R14; register D-402 to D-411; tools/presets_004.py ROLES["plants"] (Naveh 2026-10-30); A4""")
w('key = 40700')
notes = {"hr9": f"   # dt: {RES}/p1_langevin_004.json#b1.mice_F.hom.dt_for_target; predicted_lag: #b1.mice_F.hom.target; "
                f"fix_dt (A4): [judged.steps.\"mice_F|hom|clamp\"].dt",
         "d411": f"   # age: {RES}/p1_langevin_004.json#cell_ages.mice_F|hom|clamp[0]",
         "b2": "   # item 11: mice_F's base step", "d410": "   # item 11: 10 base steps, at the base step"}
if args.a8_text:
    pl = {**pl, "recording": {**pl["recording"], "m": A8_M}}
    notes["recording"] = f"   # m: A8 (tools/lang_004_fill.py A8_M): SRK at (Delta t, m), sraging at Delta t / m on delta"
for name in ("hr9", "b2", "b4", "b11", "seeding", "hazard", "recording", "d409", "d410", "d411"):
    w(f"[plants.{name}]" + notes.get(name, ""))
    for k, v in pl[name].items():
        w(f'{k} = {inline(v) if isinstance(v, dict) else q(v)}')
w("")

# ------------------------------------------------------------------------------------------------- [caps]
w("""[caps]   # CPU-h; contracts/acceptance.md "The caps"; the pilots' as run, with the OKs that changed them""")
for k, v in (("s4", 35.0), ("s5-hom", 1.5), ("s5-het", 86.0), ("paths", 6.0), ("multi", 4.0), ("plants", 3.0),
             ("hr8", 0.5)):
    w(f'{q(k)} = {v}')
w('"p1-langevin" = 6.0')
w(f'"p1-pooled" = 45.0   # run under Naveh\'s --naveh-ok 65 (2a7a12e); capped at {pa["stop_rule"]["cpu_spent_h"]:.1f} '
  f'(p1_pooled_004.json#stop_rule)')
w(f'"p1-pooled-b" = 15.0   # the ceiling; its computed cap {pb["stop_rule"]["cap_cpu_h"]:.2f} '
  f'(p1_pooled_b_004.json#stop_rule.cap_cpu_h)')
w('"p2-s5-power" = 6.0')
w('"p3-cost" = 3.0   # raised to 7 by Naveh (spec.md Clarifications 2026-11-02, 18223de), given as --naveh-ok')
w('"p4-jax" = 1.0')
out = ("\n".join(L) + "\n").replace("@@RAN_AT@@", ", ".join(dict.fromkeys(RAN_AT)))
while "\n\n\n" in out:
    out = out.replace("\n\n\n", "\n\n")
open(args.out, "w").write(out)
print("written", len(out.splitlines()), "lines; G", G, "reflect_only", [(r["cell"], r["ages"]) for r in reflect_only])
