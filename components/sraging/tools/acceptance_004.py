"""Spec 004's acceptance phases (specs/004-samplers/contracts/acceptance.md). Mac only; nothing runs on WEXAC; nothing
is written to the tester checkout or any tester store; SRK and the tester are read only through tests/oracle.

Pilots H:       p5-inventory, p0-het-cost, hr2-noise           -> freeze-H (panel_004.toml, naveh_ok_h)
Heterogeneity:  panel -> reference (pinned) -> hr1 -> hr5;  hr2;  hr3-pilot -> hr3;  hr4;  d401
Pilots L:       p1-langevin, p2-s5-power, p3-cost, p4-jax      -> freeze-L (langevin_004.toml, naveh_ok_l)
Langevin:       plants, hr8, s4, paths, multi, s5-hom, s5-het
Close:          docs, unchanged

Every phase runs through spec 003's `acceptance_003.Run` (suffix "004"): a clean committed tree, its price printed
first, its result in tests/acceptance/results/<phase>_004.json. A phase refuses to run before its prerequisites
(PREREQUISITES). Every phase with a cap follows SR, the stop rule (gate C4, Naveh 2026-10-29; `stop_rule`):
  1. it prints its projected CPU, from the named pilot, against its cap and the naveh_ok in force;
  2. it refuses to start when the projection exceeds the cap, naming the cap;
  3. it stops at the cap (the parent's CPU plus its workers'), keeps what finished and writes `capped`, never PASS;
  4. only a new naveh_ok for that phase lifts the cap: for a pilot, `--naveh-ok "<date>: <his words>"` with
     `--naveh-ok-cap`; for a frozen phase, a `[[naveh_ok_cap]]` committed in its freeze file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")

PANEL_PATH = "tests/acceptance/panel_004.toml"              # freeze-H
IDS_PATH = "tests/acceptance/panel_004_ids.tsv"
LANG_PATH = "tests/acceptance/langevin_004.toml"            # freeze-L
RESULTS_REL = "tests/acceptance/results"

PILOTS_H = ("p5-inventory", "p0-het-cost", "hr2-noise")
HETEROGENEITY = ("panel", "reference", "hr1", "hr5", "hr2", "hr3-pilot", "hr3", "hr4", "d401")
PILOTS_L = ("p1-langevin", "p1-pooled", "p1-pooled-b", "p2-s5-power", "p3-cost", "p4-jax")
LANGEVIN = ("plants", "hr8", "s4", "paths", "multi", "s5-hom", "s5-het")
CLOSE = ("docs", "unchanged")
PHASE_NAMES = PILOTS_H + HETEROGENEITY + PILOTS_L + LANGEVIN + CLOSE

#: SR for the pilots, which run before their freeze exists (tasks.md, the hard rules); the freeze files copy them
PILOT_CAPS = {"p5-inventory": 0.1, "p0-het-cost": 1.0, "hr2-noise": 0.2, "p1-langevin": 6.0, "p2-s5-power": 6.0,
              "p3-cost": 3.0, "p4-jax": 1.0,
              "p1-pooled": 45.0,           # Naveh 2026-11-01 (spec.md Clarifications, 8f0f269): ~30 CPU-h [I], cap 45
              "p1-pooled-b": 15.0}         # Naveh 2026-11-01 (c30e1ca): the follow-up's ceiling; its cap is computed
#: the freeze each frozen phase runs under: (file, its OK's key, its name), and the files it commits
FREEZES = {"H": (PANEL_PATH, "naveh_ok_h", "freeze-H", (PANEL_PATH, IDS_PATH)),
           "L": (LANG_PATH, "naveh_ok_l", "freeze-L", (LANG_PATH,))}
FREEZE_OF = {**{p: "H" for p in HETEROGENEITY}, **{p: "L" for p in LANGEVIN}}


class Refused(SystemExit):
    """A phase that must not run: a prerequisite missing, or a projection above its cap (SR 2)."""


# ------------------------------------------------------------------------------------------------ the tree's state
def _git(*a):
    return subprocess.run(["git", "--no-optional-locks", "-C", REPO, *a], capture_output=True, text=True)


def committed(rel: str) -> bool:
    """`rel` is tracked and unchanged since HEAD."""
    return _git("ls-files", "--error-unmatch", rel).returncode == 0 and not _git("status", "--porcelain", "--",
                                                                                  rel).stdout.strip()


def read_toml(rel: str) -> dict:
    return tomllib.load(open(os.path.join(REPO, rel), "rb"))


def result_rel(phase: str) -> str:
    return f"{RESULTS_REL}/{phase.replace('-', '_')}_004.json"


def read_result(phase: str) -> dict:
    return json.load(open(os.path.join(REPO, result_rel(phase))))


def file_sha256(rel: str) -> str:
    return hashlib.sha256(open(os.path.join(REPO, rel), "rb").read()).hexdigest()


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


# --------------------------------------------------------------------------------------------------- prerequisites
#: contracts/acceptance.md's prerequisite column, with §6's ordering: freeze-H precedes every §3 phase and freeze-L
#: every §4 phase, including d401 and hr8, whose rows read "—". A result prerequisite names the statuses that let the
#: phase run (None: any status, from an acceptance run); §5's docs and unchanged name none.
PREREQUISITES = {
    "p5-inventory": (),
    "p0-het-cost": (("code", "sraging.engine.mixture"), ("code", "sraging.engine.pool")),
    "hr2-noise": (("code", "sraging.engine.mixture"),),
    "panel": (("freeze", "H"),),
    "reference": (("freeze", "H"), ("result", "panel", ("pass",))),
    "hr1": (("freeze", "H"), ("pinned", "reference")),
    "hr5": (("freeze", "H"), ("result", "hr1", None)),
    "hr2": (("freeze", "H"),),
    "hr3-pilot": (("freeze", "H"),),
    "hr3": (("freeze", "H"), ("pilot", "hr3-pilot")),
    "hr4": (("freeze", "H"),),
    "d401": (("freeze", "H"),),
    **{p: (("code", "sraging.langevin"), ("code", "numba")) for p in PILOTS_L},
    "p1-pooled": (("code", "sraging.langevin"), ("code", "numba"), ("result", "p1-langevin", ("written",))),
    "p1-pooled-b": (("code", "sraging.langevin"), ("code", "numba"), ("result", "p1-langevin", ("written",)),
                    ("result", "p1-pooled", ("written", "capped", "incomplete"))),
    "plants": (("freeze", "L"), ("result", "p1-langevin", ("written",))),
    "hr8": (("freeze", "L"),),
    "s4": (("freeze", "L"), ("result", "p1-langevin", ("written",))),
    "paths": (("freeze", "L"), ("result", "p1-langevin", ("written",))),
    "multi": (("freeze", "L"),),
    "s5-hom": (("freeze", "L"),),
    "s5-het": (("freeze", "L"), ("result", "p2-s5-power", ("written",))),
    "docs": (),
    "unchanged": (),
}


def _missing_freeze(which) -> list:
    path, key, name, files = FREEZES[which]
    out = [f"{name}: {f} is not committed" for f in files if not committed(f)]
    if not out and not str(read_toml(path).get(key, "")).strip():
        out.append(f"{name}: {path} {key} is empty (Naveh's OK, tasks T044/T119)")
    return out


def _missing_result(phase, statuses) -> list:
    rel = result_rel(phase)
    if not committed(rel):
        return [f"{rel} is not committed (run `{phase}` first)"]
    d = read_result(phase)
    if d.get("status_for_acceptance") != "acceptance":
        return [f"{rel} is from a {d.get('status_for_acceptance')} run, not an acceptance run"]
    if statuses is not None and d.get("status") not in statuses:
        return [f"{rel} reads {d.get('status')!r}, not {' or '.join(statuses)}"]
    return []


def _missing_pilot(pilot, phase) -> list:
    """The pilot that projects `phase`: written, or capped when a `[[naveh_ok_cap]]` naming `phase` is committed in its
    freeze file (his OK takes the capped pilot's projection; Naveh 2026-10-29 on hr3, tasks T052), with the projection
    either way."""
    capped_ok = bool(naveh_oks(phase, None))
    out = _missing_result(pilot, ("written", "capped") if capped_ok else ("written",))
    if out:
        if not capped_ok and committed(result_rel(pilot)) and read_result(pilot).get("status") == "capped":
            out.append(f"a capped {pilot} projects {phase} only with a [[naveh_ok_cap]] naming {phase} (SR 4)")
        return out
    key = f"{phase.replace('-', '_')}_projection_cpu_h"
    return [] if key in read_result(pilot) else [f"{result_rel(pilot)} has no {key}"]


def _missing_pin(phase) -> list:
    """The reference is pinned when PANEL's [reference] sha256 is its committed file's, written by an acceptance run
    (contracts/acceptance.md §3; a reference that failed its composite check is never pinned)."""
    out = _missing_result(phase, ("written",))
    if out or not committed(PANEL_PATH):
        return out
    pinned = (read_toml(PANEL_PATH).get("reference") or {}).get("sha256")
    if pinned != file_sha256(result_rel(phase)):
        out.append(f"{result_rel(phase)} is not pinned: {PANEL_PATH} [reference] sha256 is {pinned!r}")
    return out


def missing_prerequisites(phase: str) -> list:
    """What `phase` still waits for, each named; empty when it may run."""
    out = []
    for req in PREREQUISITES[phase]:
        kind = req[0]
        if kind == "code":
            out += [] if has_module(req[1]) else [f"{req[1]} is not built yet"]
        elif kind == "freeze":
            out += _missing_freeze(req[1])
        elif kind == "result":
            out += _missing_result(req[1], req[2])
        elif kind == "pilot":
            out += _missing_pilot(req[1], phase)
        elif kind == "pinned":
            out += _missing_pin(req[1])
        else:
            raise ValueError(f"unknown prerequisite {req!r}")
    return out


def require_prerequisites(phase: str) -> None:
    missing = missing_prerequisites(phase)
    if missing:
        raise Refused(f"[{phase}] REFUSED: prerequisites missing: " + "; ".join(missing))


# ------------------------------------------------------------------------------------------------- SR, the stop rule
@dataclass(frozen=True)
class NavehOk:
    """Naveh's OK lifting one phase's cap (SR 4): the phase it names, the cap it sets, and "<date>: <his words>"."""
    phase: str
    cap_cpu_h: float
    said: str


_SAID = re.compile(r"^\d{4}-\d{2}-\d{2}: \S")


class Budget:
    """SR 3: the CPU one capped phase has spent, the parent's process time since `stop_rule` plus each worker task's
    reported CPU (a spawned worker's CPU is not in the parent's process time), against the cap in force."""

    def __init__(self, phase, projected_cpu_h, cap_cpu_h, cap_frozen_cpu_h, naveh_ok, source, clock):
        self.phase, self.projected_cpu_h, self.cap_cpu_h = phase, projected_cpu_h, cap_cpu_h
        self.cap_frozen_cpu_h, self.naveh_ok, self.source, self.clock = cap_frozen_cpu_h, naveh_ok, source, clock
        self.c0, self.worker_cpu_s, self.capped = clock(), 0.0, False

    def add_worker_cpu(self, cpu_s: float) -> None:
        self.worker_cpu_s += float(cpu_s)

    @property
    def parent_cpu_s(self) -> float:
        return self.clock() - self.c0

    @property
    def spent_cpu_h(self) -> float:
        return (self.parent_cpu_s + self.worker_cpu_s) / 3600.0

    def reached(self) -> bool:
        return self.spent_cpu_h >= self.cap_cpu_h

    def record(self) -> dict:
        """What the result keeps (contracts/acceptance.md §6): the projection, the cap, the CPU spent and the OK in
        force."""
        return {"projected_cpu_h": self.projected_cpu_h, "projected_from": self.source, "cap_cpu_h": self.cap_cpu_h,
                "cap_frozen_cpu_h": self.cap_frozen_cpu_h,
                "naveh_ok": asdict(self.naveh_ok) if self.naveh_ok else None, "cpu_spent_h": self.spent_cpu_h,
                "parent_cpu_s": self.parent_cpu_s, "worker_cpu_s": self.worker_cpu_s, "capped": self.capped}


def stop_rule(phase, projected_cpu_h, cap_cpu_h, naveh_ok=None, *, source="", clock=time.process_time) -> Budget:
    """SR 1, 2 and 4: print the projection against the cap in force, refuse above it, and return the Budget that SR 3
    runs under. `cap_cpu_h` is the frozen cap (PILOT_CAPS, or the freeze file's); `naveh_ok` is None, a NavehOk or a
    sequence of them. Only one naming `phase` counts; the last such one sets the cap."""
    p = float(projected_cpu_h)
    if not (math.isfinite(p) and p >= 0):
        raise Refused(f"[{phase}] REFUSED: the projection {projected_cpu_h!r} CPU-h is not a finite number >= 0 (SR 1)")
    oks = (naveh_ok,) if isinstance(naveh_ok, NavehOk) else tuple(naveh_ok or ())
    mine = [ok for ok in oks if ok.phase == phase]
    ok = mine[-1] if mine else None
    cap = float(ok.cap_cpu_h) if ok else float(cap_cpu_h)
    print(f"[{phase}] SR: projected {p:.3g} CPU-h{f' ({source})' if source else ''} against the cap {cap:g} CPU-h; "
          + (f"naveh_ok {ok.said!r} (frozen cap {float(cap_cpu_h):g})" if ok else "no new naveh_ok"), flush=True)
    if p > cap:
        raise Refused(f"[{phase}] REFUSED: projected {p:.3g} CPU-h exceeds the cap {cap:g} CPU-h (SR 2); only a new "
                      f"naveh_ok naming {phase} lifts it (SR 4)")
    return Budget(phase, p, cap, float(cap_cpu_h), ok, source, clock)


def run_capped(budget: Budget, cells, work, *, on_error=None, on_cell=None) -> list:
    """SR 3: `work(cell) -> (row, worker_cpu_s)` for each cell in order, the worker CPU added to the budget. At the cap
    the run stops before the next cell, keeps the rows that finished and marks the budget capped.

    Opt-in (hr3 only, Naveh 2026-10-30): `on_error(cell, exc, traceback_text, cell_cpu_s) -> row` turns a cell whose
    work raises into that row and the run goes on; `on_cell(index, n, cell, row, exc_or_None, cell_cpu_s, budget)` is
    called after each cell. Without them a raising cell ends the run, as before."""
    rows = []
    cells = list(cells)
    for i, cell in enumerate(cells, 1):
        if budget.reached():
            budget.capped = True
            break
        c0, err = budget.clock(), None
        if on_error is None:
            row, worker_cpu_s = work(cell)
        else:
            try:
                row, worker_cpu_s = work(cell)
            except Exception as ex:                            # noqa: BLE001 -- recorded as the cell's error row
                import traceback
                err, worker_cpu_s = ex, 0.0
                row = on_error(cell, ex, traceback.format_exc(), budget.clock() - c0)
        budget.add_worker_cpu(worker_cpu_s)
        rows.append(row)
        if on_cell is not None:
            on_cell(i, len(cells), cell, row, err, budget.clock() - c0 + worker_cpu_s, budget)
    return rows


# ------------------------------------------------------------- a raising cell is an error row (Naveh 2026-10-30)
TRACEBACK_LINES = 12


def progress_path(phase: str) -> str:
    """`.cache/<phase>_004_progress.jsonl` (gitignored; for diagnosis, never read back)."""
    return os.path.join(REPO, ".cache", f"{phase.replace('-', '_')}_004_progress.jsonl")


def error_record(ex, tb, cpu_s) -> dict:
    """What a raising cell keeps: the exception's type and message, the traceback's last lines, the CPU it spent."""
    return {"type": type(ex).__name__, "message": str(ex), "traceback": tb.rstrip().splitlines()[-TRACEBACK_LINES:],
            "cpu_s": float(cpu_s)}


def progress(phase, path, label):
    """After each cell: one flushed line (index/n, the cell's label, ok or error, its CPU, the CPU spent against the
    cap) and the cell's row appended to `path` as a JSON line."""
    def on_cell(i, n, cell, row, err, cpu_s, budget):
        import json
        what = "ok" if err is None else f"error ({type(err).__name__}: {err})"
        print(f"[{phase}] cell {i}/{n} {label(cell)}: {what}; cell CPU {cpu_s:.0f} s; spent "
              f"{budget.spent_cpu_h:.3g} of {budget.cap_cpu_h:g} CPU-h", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps({"phase": phase, "index": i, "n": n, "cell": label(cell),
                                 "status": "ok" if err is None else "error", "cpu_s": cpu_s,
                                 "spent_cpu_h": budget.spent_cpu_h, "row": row}, default=_json_default) + "\n")
            fh.flush()
    return on_cell


def robust_run(phase, budget, cells, work, *, label=str, error_row=None, path=None) -> list:
    """run_capped with hr3's treatment (370e536) for any phase: a cell whose work raises becomes
    `error_row(cell, record)` (default {"id": label(cell), "error": record}), the run goes on, and each cell prints a
    progress line and appends a JSON line to `.cache/<phase>_004_progress.jsonl`. The phase's rule reads an error row
    as incomplete (sampler_rules.raised)."""
    def on_error(cell, ex, tb, cpu_s):
        rec = error_record(ex, tb, cpu_s)
        return error_row(cell, rec) if error_row is not None else {"id": label(cell), "error": rec}
    return run_capped(budget, cells, work, on_error=on_error,
                      on_cell=progress(phase, path or progress_path(phase), label))


# ------------------------------------------------- rows across worker processes: wall time only (Naveh 2026-11-04 ask)
WORKER_PHASES = ("s4", "paths", "p1-pooled")    # the phases that take --workers; any other refuses it by name


def s4_row_task(row, key, frozen) -> dict:
    """One S4 row, as phase_s4 runs it: a pure function of (row, key, frozen step), so any process gives its bytes."""
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    return S4.run_frozen(row, PR.preset(row["preset"]), key, frozen, het=S4._het(row["parameter"], row["cv"]))


def paths_row_task(row, key, frozen, skip_ages) -> dict:
    """One HR-10 configuration, as phase_paths runs it (pure in its arguments, like s4_row_task)."""
    import paths_004 as PA
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    return PA.run_frozen(row, PR.preset(row["preset"]), key, frozen, het=S4._het(row["parameter"], row["sd"]),
                         skip_ages=skip_ages)


def _pool_task(fn_name, args) -> dict:
    """In a spawned worker: `fn_name(*args)` from this module, its exception kept as data, with the process's
    cumulative CPU (imports and JIT compile included) for the parent's budget."""
    import traceback
    try:
        out = {"ok": True, "row": globals()[fn_name](*args)}
    except Exception as ex:                                    # noqa: BLE001 -- returned as the cell's error row
        out = {"ok": False, "type": type(ex).__name__, "message": str(ex), "traceback": traceback.format_exc()}
    return {**out, "pid": os.getpid(), "cpu_total": time.process_time()}


def robust_run_pool(phase, budget, cells, fn_name, args_of, workers, *, label=str, error_row=None, path=None) -> list:
    """robust_run across `workers` spawned processes, for wall time only. Each cell is `fn_name(*args_of(cell))`, a pure
    function of its arguments (the key comes from the cell's place in the frozen grid), so the rows are workers = 1's,
    returned in cell order. As each row returns: its worker's CPU since that worker's last row goes to
    budget.add_worker_cpu, its progress line is written, and the cap is checked before the next cell starts. At the cap
    no new cell starts, the ones in flight (at most workers - 1 more) finish and are kept, and the budget reads capped.
    A raising cell is its error row, as in robust_run. workers = 1 is robust_run itself, in this process."""
    W = int(workers)
    if W <= 1:
        fn = globals()[fn_name]
        return robust_run(phase, budget, cells, lambda c: (fn(*args_of(c)), 0.0), label=label, error_row=error_row,
                          path=path)
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    cells = list(cells)
    on_cell = progress(phase, path or progress_path(phase), label)
    rows, last_cpu, n_done, nxt, inflight = [None] * len(cells), {}, 0, 0, {}
    with ProcessPoolExecutor(max_workers=W, mp_context=mp.get_context("spawn")) as ex:
        while True:
            while len(inflight) < W and nxt < len(cells) and not budget.reached():
                inflight[ex.submit(_pool_task, fn_name, args_of(cells[nxt]))] = nxt
                nxt += 1
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done:
                i = inflight.pop(f)
                r = f.result()
                cpu = r["cpu_total"] - last_cpu.get(r["pid"], 0.0)
                last_cpu[r["pid"]] = r["cpu_total"]
                budget.add_worker_cpu(cpu)
                err = None
                if r["ok"]:
                    rows[i] = r["row"]
                else:
                    err = type(r["type"], (Exception,), {})(r["message"])
                    rec = {"type": r["type"], "message": r["message"],
                           "traceback": r["traceback"].rstrip().splitlines()[-TRACEBACK_LINES:], "cpu_s": float(cpu)}
                    rows[i] = error_row(cells[i], rec) if error_row is not None else {"id": label(cells[i]),
                                                                                        "error": rec}
                n_done += 1
                on_cell(n_done, len(cells), cells[i], rows[i], err, cpu, budget)
    if nxt < len(cells):
        budget.capped = True
    return [r for r in rows if r is not None]


def errors_of(rows) -> list:
    """The error rows among `rows`: {"id", "error"} each."""
    return [{"id": r.get("id"), **r["error"]} for r in rows if isinstance(r, dict) and r.get("error")]


def phase_status(status: str, budget: Budget | None) -> str:
    """A capped run's status is `capped`, whatever its finished cells read (never PASS)."""
    return "capped" if budget is not None and budget.capped else status


def counts_as_pass(status) -> bool:
    """The close's reading (constitution V): only pass is PASS; capped and incomplete never are."""
    return str(status).lower() == "pass"


def close_tally(statuses: dict) -> dict:
    """{phase: status} -> the close's counts. All pass only when there is something and every one is a pass."""
    low = {k: str(v).lower() for k, v in statuses.items()}
    return {"pass": sum(map(counts_as_pass, low.values())), "fail": sorted(k for k, v in low.items() if v == "fail"),
            "incomplete": sorted(k for k, v in low.items() if v == "incomplete"),
            "capped": sorted(k for k, v in low.items() if v == "capped"),
            "all_pass": bool(low) and all(map(counts_as_pass, low.values()))}


def naveh_oks(phase: str, args) -> tuple:
    """SR 4's new OKs for `phase`: a pilot's from the command line (`--naveh-ok` with `--naveh-ok-cap`, both or
    neither); a frozen phase's only from `[[naveh_ok_cap]]` entries ({phase, cap_cpu_h, said}) committed in its freeze
    file, never from the command line."""
    said, cap = getattr(args, "naveh_ok", None), getattr(args, "naveh_ok_cap", None)
    if said is not None or cap is not None:
        if phase not in PILOT_CAPS:
            where = FREEZES[FREEZE_OF[phase]][0] if phase in FREEZE_OF else "no file: only a pilot's cap"
            raise Refused(f"[{phase}] REFUSED: --naveh-ok lifts a pilot's cap only; {phase}'s is lifted by a new "
                          f"naveh_ok committed in its freeze file ({where}; SR 4)")
        if said is None:
            raise Refused(f"[{phase}] REFUSED: --naveh-ok-cap without --naveh-ok (only Naveh lifts a cap)")
        if cap is None:
            raise Refused(f"[{phase}] REFUSED: --naveh-ok without --naveh-ok-cap (the cap his OK sets)")
        if not _SAID.match(said):
            raise Refused(f"[{phase}] REFUSED: --naveh-ok must read '<date>: <his words>', as 2026-11-02: ..., "
                          f"got {said!r}")
        return (NavehOk(phase, float(cap), said),)
    if phase in FREEZE_OF and committed(FREEZES[FREEZE_OF[phase]][0]):
        rows = read_toml(FREEZES[FREEZE_OF[phase]][0]).get("naveh_ok_cap", [])
        return tuple(NavehOk(r["phase"], float(r["cap_cpu_h"]), r["said"]) for r in rows if r["phase"] == phase)
    return ()


# --------------------------------------------------------------------------------------------------- phase frame
def manifests(phase: str) -> list:
    """The committed inputs a phase's Run checks: its freeze's files (none for a pilot or the close)."""
    return list(FREEZES[FREEZE_OF[phase]][3]) if phase in FREEZE_OF else []


def cores() -> dict:
    """This Mac's cores (contracts/acceptance.md: every result records them)."""
    def sysctl(k):
        r = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True)
        return int(r.stdout) if r.returncode == 0 and r.stdout.strip().isdigit() else None
    return {"logical": os.cpu_count(), "performance": sysctl("hw.perflevel0.physicalcpu"),
            "efficiency": sysctl("hw.perflevel1.physicalcpu")}


def _run(phase, args, cost):
    import acceptance_003 as A
    return A.Run(phase, manifests=manifests(phase), dev=args.dev, suffix="004", cost=cost)


def write(run, args, payload: dict, budget: Budget | None = None):
    """The result file: the phase's payload with the provenance contracts/acceptance.md names beyond Run's (cores, the
    manifests' sha256) and SR's record, the `--naveh-ok` given included; a capped run's status is `capped`."""
    out = dict(payload)
    out["cores"] = cores()
    out["manifests_sha256"] = {m: file_sha256(m) if os.path.exists(os.path.join(REPO, m)) else None
                               for m in manifests(run.phase)}
    out["naveh_ok_arg"] = {"said": getattr(args, "naveh_ok", None), "cap_cpu_h": getattr(args, "naveh_ok_cap", None)}
    if budget is not None:
        out["stop_rule"] = budget.record()
        out["status"] = phase_status(out.get("status"), budget)
    return run.write(out)


def _todo(name):
    def phase(args):
        sys.exit(f"{name}: not implemented yet (specs/004-samplers/tasks.md)")
    return phase


# ------------------------------------------------------------------------------------------- frozen caps (SR 1-4)
#: a cap shared by several phases: its [caps] key, and the phases whose spent CPU counts against it (tasks T047-T049)
CAP_SHARED = {"reference": "reference", "hr1": "reference", "hr5": "reference"}
#: a phase whose cap is also spent by another's run (tasks T051: hr3-pilot's CPU counts against hr3's 10)
CAP_COUNTED = {"hr3": ("hr3-pilot",)}


def frozen_cap(phase: str) -> tuple:
    """(the cap left for `phase`, its record): its freeze file's [caps] entry (keyed by phase, or by the shared key of
    CAP_SHARED), less the CPU the other phases of a shared cap (and CAP_COUNTED's) spent in their committed acceptance
    results."""
    path = FREEZES[FREEZE_OF[phase]][0]
    caps = read_toml(path).get("caps") or {}
    key = CAP_SHARED.get(phase, phase)
    if key not in caps:
        raise Refused(f"[{phase}] REFUSED: {path} [caps] has no {key!r} (the freeze sets every cap; SR)")
    spent = {}
    others = [p for p, k in CAP_SHARED.items() if k == key and p != phase] + list(CAP_COUNTED.get(phase, ()))
    for other in others:
        if committed(result_rel(other)):
            d = read_result(other)
            if d.get("status_for_acceptance") == "acceptance":
                spent[other] = float((d.get("stop_rule") or {}).get("cpu_spent_h") or 0.0)
    left = float(caps[key]) - sum(spent.values())
    return left, {"key": key, "frozen_cpu_h": float(caps[key]), "spent_by": spent, "left_cpu_h": left}


# ------------------------------------------------------------------------------------- the inputs the phases share
SDS = (0.1, 0.2, 0.3, 0.5)
HR2_NOISE_COHORTS = ("het-real-Jack_Russell_vetCompass-c0-s0", "het-real-Sweden_F_1910_homo-c0-s0")   # held out
HR2_NOISE_H = (1e-3, 1e-4, 1e-5)
HR2_NOISE_SD = (0.30, 0.35, 0.40)
HR2_GRID = tuple(round(0.30 + 0.005 * i, 3) for i in range(21)) + (0.19, 0.195, 0.2, 0.205, 0.21)
HR2_M_AT = 0.35                     # each cohort's M: the ladder's at sd 0.35, held fixed across the grid (R11)
NOISE_TOL_FACTOR = 10.0             # HR-2's tolerance is 10 x the pilot's floor at its chosen h (R11)
COMPOSITE = {"order": 8, "panels": (16, 32, 64, 128), "agree": 1e-4, "tol": 1e-3}    # research R9
REF_M, REF_CHECK_M, REF_VALID = 192, 128, 0.01                                        # OD-6 (a)
FR009_K, FR009_SD, FR009_TOL = 161, 0.2, 0.1                                         # FR-009, R20
P5_S_PER_FILE = 0.05                # [I] the inventory's CPU per scanned file (c1_grep: 656 files in 1.1 s)
PANEL_S_PER_COHORT = 0.8            # [M] panel_004's CV_c march and digest check, 0.3-1.0 CPU-s per cohort
FREE = ("eta", "beta", "epsilon", "kappa")


def p0_per_node() -> dict:
    """P0's per-node CPU (tasks T039: `per_node_cpu_s` {value, value_and_grad} and `compile_cpu_s`), the projections'
    source after T039; refused before it is committed from an acceptance run."""
    rel = result_rel("p0-het-cost")
    if not committed(rel):
        raise Refused(f"no projection: {rel} is not committed (run p0-het-cost first, tasks T039)")
    d = read_result("p0-het-cost")
    if d.get("status_for_acceptance") != "acceptance":
        raise Refused(f"no projection: {rel} is from a {d.get('status_for_acceptance')} run")
    try:
        return {"value": float(d["per_node_cpu_s"]["value"]), "value_and_grad": float(d["per_node_cpu_s"]
                                                                                         ["value_and_grad"]),
                "compile": float(d.get("compile_cpu_s") or 0.0), "source": f"{rel} per_node_cpu_s"}
    except (KeyError, TypeError, ValueError):
        raise Refused(f"no projection: {rel} has no per_node_cpu_s {{value, value_and_grad}}") from None


def v6_specs() -> dict:
    import panel_004 as P
    man = json.loads(P.read_source(P.MANIFEST_V6))
    return man, {s["cohort_id"]: s for s in man["specs"]}


def v6_case(cid, man, specs):
    """(Cohort, SRParams at the truth, h_ext) for a v6 cohort: the arrays digest-checked against the pinned manifest
    (panel_004), the window applied as the tester's library applies it (srbench/library/observation.py apply_window:
    rows before the window dropped, after it censored at its close) and kept on the Cohort, so the engine conditions on
    it (research R9); the external hazard as the tester's rate exp(-ExtH) (srbench spec.physical_truth)."""
    import panel_004 as P
    from sraging import Cohort, SRParams
    spec = specs[cid]
    a, e = P.cohort_arrays(cid, man)
    win = spec.get("window")
    if win is not None:
        lo, hi = float(win[0]), float(win[1])
        keep = a >= lo
        a, e = a[keep], e[keep]
        past = a > hi
        a, e = np.where(past, hi, a), np.where(past, 0, e)
        win = (lo, hi)
    d = spec.get("delta") or None
    coh = Cohort.from_arrays(a, e, delta=d, continuous=d is None, window=win, name=cid)
    h = math.exp(-float(spec["external_hazard"])) if spec.get("external_hazard") is not None else 0.0
    return coh, SRParams(**P.physical_truth(spec), h_ext=h), h


def evaluator_for(coh, h, parameter="xc"):
    """One evaluator per cohort: its per-node programs serve every sd and M (mixture_value's `het` override)."""
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.inputs import Heterogeneity
    return Evaluator(coh, numerics="v1", fixed={"h_ext": h},
                     heterogeneity=Heterogeneity(parameter=parameter, sd=0.2, M=REF_M, check=False))


def het(sd, M="auto", check=True, parameter="xc"):
    from sraging.engine.inputs import Heterogeneity
    return Heterogeneity(parameter=parameter, sd=sd, M=M, check=check)


def panel_rows() -> list:
    import panel_004 as P
    return P.read_ids(os.path.join(REPO, IDS_PATH))


def require_frozen_ladder(pan: dict) -> None:
    """The ladder, guide and HR constants the freeze froze are the module's (a frozen rule never moves silently)."""
    from sraging.engine import mixture as MIX
    lad = pan.get("ladder") or {}
    want = {"rungs": list(MIX.RUNGS), "threshold": MIX.THRESHOLD, "cap": MIX.CAP}
    got = {k: lad.get(k) for k in want}
    if got != want:
        raise Refused(f"panel_004.toml [ladder] {got} is not sraging.engine.mixture's {want}")


def phase_header(phase, args, cost, projected, cap, source):
    """The run envelope, then SR 1-2: the price printed, the tree checked, the projection against the cap in force."""
    run = _run(phase, args, cost)
    return run, stop_rule(phase, projected, cap, naveh_oks(phase, args), source=source)


def finish(run, args, payload, budget, cap_record=None):
    if cap_record is not None:
        payload = {**payload, "cap": cap_record}
    return write(run, args, payload, budget)


# ------------------------------------------------------------------------------------------------- p5-inventory
def phase_p5_inventory(args):
    """C1 (tasks T038): every downstream call into SRK's Langevin, reconciled with PHASE_PLAN's 1xx row (read only)."""
    import p5_inventory_004 as P5
    c1 = json.load(open(os.path.join(REPO, "specs", "004-samplers", "evidence", "c1_grep.json")))
    projected = c1["files_scanned"] * P5_S_PER_FILE / 3600.0
    run, budget = phase_header("p5-inventory", args, "0 fits; a read-only scan of ~660 files", projected,
                               PILOT_CAPS["p5-inventory"], f"c1_grep.json: {c1['files_scanned']} files x "
                               f"{P5_S_PER_FILE} s")
    rows = run_capped(budget, ["scan"], lambda _: (P5.inventory(), 0.0))
    payload = rows[0] if rows else {}
    finish(run, args, {"status": "written" if rows else "capped", "summary": P5.summary(payload) if rows else None,
                       **payload}, budget)


# --------------------------------------------------------------------------------------------------- hr2-noise
def plant_crossing(M: int) -> float:
    """The spread v* where normal_frac_sd's lowest Gauss-Hermite node z_min crosses s = 0: v* = -1 / z_min (M = 5:
    0.350; M = 3: 0.577)."""
    from numpy.polynomial.hermite_e import hermegauss
    return -1.0 / float(hermegauss(M)[0][0])


def plant_row(coh, theta, h, M, step) -> dict:
    """HR-2's plant on one cohort (research R11): normal_frac_sd at M across its crossing v*: the jump
    |l(v* + 1e-9) - l(v* - 1e-9)|, the central difference with `step` across v*, and the v-gradient below v*, where no
    node is dropped (register D-116 refuses it above)."""
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.inputs import Heterogeneity
    ev = Evaluator(coh, numerics="v1", fixed={"h_ext": h}, free=FREE + ("het_v",),
                   heterogeneity=Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.35, M=M))
    vs = plant_crossing(M)
    eps = 1e-9 * vs
    jump = abs(ev.value(theta, het_v=vs + eps) - ev.value(theta, het_v=vs - eps))
    cd = (ev.value(theta, het_v=vs + step) - ev.value(theta, het_v=vs - step)) / (2.0 * step)
    grad = ev.value_and_grad(theta, het_v=vs - step)[1]["het_v"]
    return {"M": M, "v_cross": vs, "jump": jump, "cd": cd, "grad": grad, "abs_diff": abs(cd - grad), "step": step}


def cd_vs_grad(ev, theta, sd, M, step) -> dict:
    """At one sd: the engine's dl/dsd (het_sd_grad) and the central difference (l(sd + step) - l(sd - step)) / 2step,
    at a fixed M, unchecked (research R11: M held fixed across the grid)."""
    from sraging.engine import mixture as MIX
    c0 = time.process_time()
    g = MIX.mixture_value(ev, theta, het_sd_grad=True, het=het(sd, M, check=False)).grad["het_sd"]
    up = MIX.mixture_value(ev, theta, het=het(sd + step, M, check=False)).value
    dn = MIX.mixture_value(ev, theta, het=het(sd - step, M, check=False)).value
    cd = (up - dn) / (2.0 * step)
    return {"sd": sd, "M": M, "h": step, "cd": cd, "grad": g, "abs_diff": abs(cd - g), "cpu_s": time.process_time() - c0}


def noise_summary(rows, h_values=HR2_NOISE_H, factor=NOISE_TOL_FACTOR) -> dict:
    """The floor per h (the largest |cd - grad| over the pilot's points), the h with the smallest floor, and the
    tolerance factor x that floor (research R11)."""
    floor = {h: max(r["abs_diff"] for r in rows if r["h"] == h) for h in h_values if any(r["h"] == h for r in rows)}
    if not floor:
        return {"floor": {}, "h": None, "tol": None}
    h = min(floor, key=lambda x: (floor[x], x))
    return {"floor": {f"{k:g}": v for k, v in floor.items()}, "h": h, "tol": factor * floor[h]}


def phase_hr2_noise(args):
    """The central-difference floor (tasks T040; research R11): h in HR2_NOISE_H on 2 panel-external cohorts at sd 0.30,
    0.35, 0.40, each at its checked M at sd 0.35; HR-2's h and tolerance (10 x the floor); the plants' jumps."""
    from sraging.engine import mixture as MIX
    p0 = p0_per_node()
    M_assumed = 85
    per = (M_assumed + 128) * p0["value"] + len(HR2_NOISE_SD) * len(HR2_NOISE_H) * M_assumed * (
        p0["value_and_grad"] + 2 * p0["value"])
    projected = (len(HR2_NOISE_COHORTS) * (per + p0["compile"])) / 3600.0
    run, budget = phase_header("hr2-noise", args, f"2 cohorts x 3 sd x 3 h at M ~ {M_assumed}; the plants",
                               projected, PILOT_CAPS["hr2-noise"], f"{p0['source']} at M ~ {M_assumed}")
    man, specs = v6_specs()

    def work(cid):
        coh, theta, h = v6_case(cid, man, specs)
        ev = evaluator_for(coh, h)
        lad = MIX.mixture_value(ev, theta, het=het(HR2_M_AT)).node_record["check"]
        pts = [dict(cd_vs_grad(ev, theta, sd, lad["M"] if lad["verdict"] == "converged" else MIX.CAP, step),
                    cohort_id=cid) for sd in HR2_NOISE_SD for step in HR2_NOISE_H]
        return {"cohort_id": cid, "ladder_at_0.35": {k: lad[k] for k in ("M", "M_check", "abs_dll", "verdict")},
                "points": pts}, 0.0
    cohorts = run_capped(budget, HR2_NOISE_COHORTS, work)
    rows = [p for c in cohorts for p in c["points"]]
    summ = noise_summary(rows)
    plants = []
    if cohorts and summ["h"] is not None:
        coh, theta, h = v6_case(HR2_NOISE_COHORTS[0], man, specs)
        plants = [dict(plant_row(coh, theta, h, M, summ["h"]), cohort_id=HR2_NOISE_COHORTS[0]) for M in (5, 3)]
    for p in plants:
        p["informative"] = summ["tol"] is not None and p["jump"] > NOISE_TOL_FACTOR * summ["tol"]
    finish(run, args, {"status": "written" if len(cohorts) == len(HR2_NOISE_COHORTS) else "capped",
                       "cohorts": cohorts, "noise": summ, "plants": plants, "tol_factor": NOISE_TOL_FACTOR,
                       "cpu_point_s": float(np.median([r["cpu_s"] for r in rows])) if rows else None}, budget)


# ------------------------------------------------------------------------------------------------------- panel
def phase_panel(args):
    """The committed panel is the rebuild from the pinned manifest, with the strata counts (tasks T046)."""
    import panel_004 as P
    left, cap_rec = frozen_cap("panel")
    man, specs = v6_specs()
    n_cand = sum(P.candidate(s) is not None for s in specs.values())
    run, budget = phase_header("panel", args, f"0 fits; {n_cand} CV_c marches", n_cand * PANEL_S_PER_COHORT /
                               3600.0, left, f"{n_cand} candidates x {PANEL_S_PER_COHORT} CPU-s")
    pan = read_toml(PANEL_PATH)
    out = run_capped(budget, ["check"], lambda _: (P.check(pan), 0.0))
    rows = P.read_ids(os.path.join(REPO, IDS_PATH))
    counts = P.counts(rows)
    ok = bool(out) and out[0] == [] and counts == P.STRATA and not ({r["cohort_id"] for r in rows} & set(P.HELD_OUT))
    finish(run, args, {"status": "pass" if ok else "fail", "differences": out[0] if out else None, "counts": counts,
                       "rows": len(rows), "ids_sha256": pan["sources"]["ids"]["sha256"]}, budget, cap_rec)


# --------------------------------------------------------------------------------------------------- reference
def composite_value(ev, theta, sd) -> dict:
    """The independent composite rule refined until two refinements agree (research R9)."""
    from sraging.engine import mixture as MIX
    from sraging.engine import quadrature as Q
    prev, ladder = None, []
    for panels in COMPOSITE["panels"]:
        ll = MIX.value_at_rule(ev, theta, Q.composite_rule(sd, panels, COMPOSITE["order"]))
        ladder.append({"panels": panels, "nodes": panels * COMPOSITE["order"], "ll": ll})
        if prev is not None and abs(ll - prev) < COMPOSITE["agree"]:
            return {"ll_composite": ll, "converged": True, "ladder": ladder}
        prev = ll
    return {"ll_composite": prev, "converged": False, "ladder": ladder}


def reference_cells(rows) -> list:
    """(cohort_id, stratum, sd, composite?) for every panel cohort x SDS, the composite subset's cells marked."""
    return [(r["cohort_id"], r["stratum"], sd, f"subset-composite@{sd:g}" in r["roles"]) for r in rows for sd in SDS]


class Evaluators:
    """One evaluator per cohort, built on first use (a cohort's cells are consecutive)."""

    def __init__(self):
        self.man, self.specs = v6_specs()
        self.cid, self.ev, self.theta = None, None, None

    def __call__(self, cid):
        if cid != self.cid:
            coh, self.theta, h = v6_case(cid, self.man, self.specs)
            self.cid, self.ev = cid, evaluator_for(coh, h)
        return self.ev, self.theta


def phase_reference(args):
    """Per cell: l(192), l(128), validity |l(128) - l(192)| < 0.01; the composite cross-check on the 8 subset-composite
    cells to 1e-3 nats (tasks T047; OD-6 (a))."""
    from sraging.engine import mixture as MIX
    pan = read_toml(PANEL_PATH)
    require_frozen_ladder(pan)
    left, cap_rec = frozen_cap("reference")
    cells = reference_cells(panel_rows())
    p0 = p0_per_node()
    comp_nodes = sum(1 for c in cells if c[3]) * COMPOSITE["order"] * sum(COMPOSITE["panels"])
    projected = ((REF_M + REF_CHECK_M) * len(cells) + comp_nodes) * p0["value"] / 3600.0 + 40 * p0["compile"] / 3600.0
    run, budget = phase_header("reference", args, f"{len(cells)} cells x (192 + 128) nodes + the composite subset",
                               projected, left, f"{p0['source']} x {(REF_M + REF_CHECK_M) * len(cells) + comp_nodes}"
                               f" nodes")
    evs = Evaluators()

    def work(cell):
        cid, stratum, sd, comp = cell
        ev, theta = evs(cid)
        c0 = time.process_time()
        l192 = MIX.mixture_value(ev, theta, het=het(sd, REF_M, check=False)).value
        l128 = MIX.mixture_value(ev, theta, het=het(sd, REF_CHECK_M, check=False)).value
        row = {"cohort_id": cid, "stratum": stratum, "sd": sd, "ll_192": l192, "ll_128": l128,
               "abs_dll_128_192": abs(l128 - l192), "valid": abs(l128 - l192) < REF_VALID}
        if comp:
            cmp_ = composite_value(ev, theta, sd)
            row.update(cmp_, abs_diff=abs(cmp_["ll_composite"] - l192),
                       composite_ok=cmp_["converged"] and abs(cmp_["ll_composite"] - l192) <= COMPOSITE["tol"])
        row["cpu_s"] = time.process_time() - c0
        return row, 0.0
    rows = run_capped(budget, cells, work)
    comp = [r for r in rows if "composite_ok" in r]
    bad = [r for r in comp if not r["composite_ok"]]
    status = "fail" if bad else ("written" if len(rows) == len(cells) else "capped")
    finish(run, args, {"status": status, "rows": rows, "composite_failures": [(r["cohort_id"], r["sd"]) for r in bad],
                       "n_valid": sum(r["valid"] for r in rows), "n_cells": len(cells), "composite": COMPOSITE,
                       "M_ref": REF_M, "valid_below": REF_VALID}, budget, cap_rec)
    if bad:
        sys.exit(f"[reference] FAIL: the composite cross-check missed {COMPOSITE['tol']:g} nats on {len(bad)} cell(s): "
                 f"stop and report (tasks T047)")


# --------------------------------------------------------------------------------------------------------- hr1
def gen1_rule(sd=FR009_SD, K=FR009_K):
    """FR-009: gen-1's K-node hat law at sd, as a quadrature.Rule (read-only import of sraging-gen-1)."""
    from sraging import generator as GEN
    from sraging.engine import quadrature as Q
    d = GEN._het_nodes(sd, K)
    z, w, s = (Q._ro(d[k]) for k in ("z", "W", "s"))
    return Q.Rule("gen1_hat", float(sd), int(K), z, w, s, float(s[0]), Q._sha(z, w))


def hr1_cell(ev, theta, cid, stratum, sd, ref) -> dict:
    """One HR-1 cell: automatic M (with its cost record, HR-5's input) and the M = 11 plant with its check value."""
    from sraging.engine import mixture as MIX
    r = MIX.mixture_value(ev, theta, het=het(sd))
    c = r.node_record["check"]
    plant = MIX.mixture_value(ev, theta, het=het(sd, 11)).node_record["check"]       # l(11), l(17), |l(11) - l(17)|
    return {"id": f"{cid}/sd{sd:g}", "cohort_id": cid, "stratum": stratum, "sd": sd, "ll": r.value,
            "verdict": c["verdict"], "M": c["M"], "M_used": r.node_record["M_used"], "abs_dll": c["abs_dll"],
            "start": c["start"], "ladder": c["ladder"], "ll_ref": ref["ll_192"], "ref_valid": ref["valid"],
            "ll_11": plant["ll_M"], "ll_17": plant["ll_check"], "check_11": plant["abs_dll"],
            "warnings": [w.code for w in r.warnings], "cost": r.cost_record}


def fr009_verdict(rows, tol=FR009_TOL) -> dict:
    """FR-009: gen-1's 161-node law within tol nats of the reference on every panel cohort (sd 0.2). Not a pass with
    nothing measured, nor with an invalid reference (gate C2)."""
    bad = [r["cohort_id"] for r in rows if not r["abs_diff"] < tol]
    invalid = [r["cohort_id"] for r in rows if not r["ref_valid"]]
    status = "fail" if bad else ("incomplete" if (invalid or not rows) else "pass")
    return {"rule": "FR-009", "status": status, "n": len(rows), "over_tol": bad, "reference_invalid": invalid,
            "tol": tol, "max_abs_diff": max((r["abs_diff"] for r in rows), default=None)}


def combined_status(statuses) -> str:
    s = list(statuses)
    return "fail" if "fail" in s else "capped" if "capped" in s else "incomplete" if "incomplete" in s else "pass"


def phase_hr1(args):
    """HR-1 (tasks T048): automatic M on the 160 cells against the pinned reference, the M = 11 unchecked plant, FR-009
    (gen-1's 161-node law within 0.1 nats on every panel cohort) and each cell's stated and measured cost (HR-5)."""
    from sraging import sampler_rules as R
    from sraging.engine import mixture as MIX
    pan = read_toml(PANEL_PATH)
    require_frozen_ladder(pan)
    rules = (pan.get("rules") or {}).get("HR-1") or {}
    left, cap_rec = frozen_cap("hr1")
    ref = read_result("reference")
    refs = {(r["cohort_id"], r["sd"]): r for r in ref["rows"]}
    cells = [(cid, st, sd) for cid, st, sd, _ in reference_cells(panel_rows())]
    p0 = p0_per_node()
    ref_h = float(ref["stop_rule"]["cpu_spent_h"])
    projected = 0.5 * ref_h + 40 * FR009_K * p0["value"] / 3600.0
    run, budget = phase_header("hr1", args, f"{len(cells)} cells: the ladder, the M = 11 plant; FR-009 on 40 cohorts",
                               projected, left, f"reference_004.json's CPU {ref_h:.3g} CPU-h x 0.5 + FR-009")
    evs = Evaluators()
    g1 = gen1_rule()
    fr009 = []

    def work(cell):
        cid, stratum, sd = cell
        ev, theta = evs(cid)
        row = hr1_cell(ev, theta, cid, stratum, sd, refs[(cid, sd)])
        if sd == FR009_SD:
            l = MIX.value_at_rule(ev, theta, g1)
            fr009.append({"cohort_id": cid, "ll_gen1": l, "ll_ref": refs[(cid, sd)]["ll_192"],
                          "ref_valid": refs[(cid, sd)]["valid"], "abs_diff": abs(l - refs[(cid, sd)]["ll_192"])})
        return row, 0.0
    rows = run_capped(budget, cells, work)
    v = R.hr1_verdict(rows, **{k: rules[k] for k in ("target", "threshold", "sd_strata") if k in rules})
    f = fr009_verdict(fr009)
    status = "capped" if budget.capped else combined_status([v.status, f["status"]])
    finish(run, args, {"status": status, "verdicts": [v.as_dict(), f], "rows": rows, "fr009": fr009,
                       "gen1_rule_sha256": g1.sha256, "reference_sha256": file_sha256(result_rel("reference")),
                       "printed": {"n_valid": v.n_valid, "n_unconverged": len(v.statistic["unconverged"]),
                                   "n_reference_unconverged": v.statistic["n_reference_unconverged"]}},
           budget, cap_rec)


# --------------------------------------------------------------------------------------------------------- hr2
def phase_hr2(args):
    """HR-2 (tasks T050; research R11): on the 6 subset-hr2 cohorts, sd over [0.30, 0.40] by 0.005 and 0.19-0.21, at
    each cohort's checked M at sd 0.35 held fixed: central difference against dl/dsd within the frozen tolerance; the
    normal_frac_sd plant across its crossing (M = 5, else M = 3)."""
    from sraging import sampler_rules as R
    from sraging.engine import mixture as MIX
    pan = read_toml(PANEL_PATH)
    require_frozen_ladder(pan)
    rule = (pan.get("rules") or {}).get("HR-2") or {}
    if "tol" not in rule or "h" not in rule:
        raise Refused(f"[hr2] REFUSED: {PANEL_PATH} [rules.HR-2] needs tol and h (from hr2-noise, tasks T041)")
    left, cap_rec = frozen_cap("hr2")
    noise = read_result("hr2-noise")
    ids = [r["cohort_id"] for r in panel_rows() if "subset-hr2" in r["roles"]]
    projected = len(ids) * (len(HR2_GRID) + 2) * float(noise["cpu_point_s"]) / 3600.0     # + the ladder at 0.35
    run, budget = phase_header("hr2", args, f"{len(ids)} cohorts x {len(HR2_GRID)} sd; the plant", projected, left,
                               f"hr2_noise_004.json cpu_point_s {noise['cpu_point_s']:.3g}")
    evs = Evaluators()
    step, tol = float(rule["h"]), float(rule["tol"])

    def work(cid):
        ev, theta = evs(cid)
        lad = MIX.mixture_value(ev, theta, het=het(HR2_M_AT)).node_record["check"]
        M = lad["M"] if lad["verdict"] == "converged" else MIX.CAP
        pts = [dict(cd_vs_grad(ev, theta, sd, M, step), id=f"{cid}/sd{sd:g}", cohort_id=cid) for sd in HR2_GRID]
        return {"cohort_id": cid, "M": M, "ladder_at_0.35": {k: lad[k] for k in ("M", "abs_dll", "verdict")},
                "points": pts}, 0.0
    cohorts = run_capped(budget, ids, work)
    cells = [p for c in cohorts for p in c["points"]]
    plants = []
    if ids:
        coh, theta, h = v6_case(ids[0], evs.man, evs.specs)
        plants = [dict(plant_row(coh, theta, h, M, step), cohort_id=ids[0]) for M in (5, 3)]
    v = R.hr2_verdict(cells, plants, tol=tol)
    finish(run, args, {"status": "capped" if budget.capped else v.status, "verdicts": [v.as_dict()],
                       "cohorts": cohorts, "plants": plants, "h": step, "tol": tol}, budget, cap_rec)


# -------------------------------------------------------------------------------------------------------- d401
def phase_d401(args):
    """D-401's impact (tasks T054): delta-loglik per cohort at fixed theta on spec 002's 4 heterogeneous T1 cohorts x 5
    theta (tests/acceptance/panel_002_t1.toml), 'normal_frac_sd' (v 0.2, M 7) against 'truncated_normal' (sd 0.2,
    checked M), numerics v1 and each cohort's v033 grid on both sides."""
    import acceptance_002_t1 as T1
    from sraging import Cohort, SRParams
    from sraging.engine import mixture as MIX
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.inputs import Heterogeneity
    t1 = read_toml("tests/acceptance/panel_002_t1.toml")
    ids, labels = list(t1["arms"]["het"]), list(t1["theta_labels"])
    left, cap_rec = frozen_cap("d401")
    p0 = p0_per_node()
    projected = len(ids) * (2 * p0["compile"] + len(labels) * (7 + 3 * 64) * p0["value"]) / 3600.0
    run, budget = phase_header("d401", args, f"{len(ids)} cohorts x {len(labels)} theta", projected, left,
                               f"{p0['source']} at M ~ 64")

    def thetas(truth):
        mult = {"truth": {}, "eta x1.5": {"eta": 1.5}, "beta x1.5": {"beta": 1.5}, "epsilon x1.5": {"epsilon": 1.5},
                "eta x10": {"eta": 10.0}}
        return [(lab, {k: v * mult[lab].get(k, 1.0) for k, v in truth.items()}) for lab in labels]

    def work(cid):
        c = T1.library_case(cid, t1)
        h = 0.0 if c["ext"] is None else math.exp(-c["ext"])
        coh = Cohort.from_arrays(c["t"], c["e"], delta=c["d"], window=c["window"], name=cid)
        par, v = c["het"]
        old = Evaluator(coh, numerics="v1", fixed={"h_ext": h},
                        heterogeneity=Heterogeneity(parameter=par, definition="normal_frac_sd", v=v, M=int(t1["het_M"])))
        new = Evaluator(coh, numerics="v1", fixed={"h_ext": h}, heterogeneity=Heterogeneity(parameter=par, sd=v))
        out = []
        for lab, th in thetas(c["truth"]):
            p = SRParams(**th, h_ext=h)
            r = MIX.mixture_value(new, p)
            lo = old.value(p)
            out.append({"theta": lab, "params": th, "ll_normal_frac_sd": lo, "ll_truncated_normal": r.value,
                        "dloglik": r.value - lo, "M_used": r.node_record["M_used"],
                        "verdict": r.node_record["check"]["verdict"], "clean": r.clean})
        return {"cohort_id": cid, "n": coh.n, "het": [par, v], "rows": out}, 0.0
    cohorts = run_capped(budget, ids, work)
    d = [abs(r["dloglik"]) for c in cohorts for r in c["rows"]]
    finish(run, args, {"status": "written" if len(cohorts) == len(ids) else "capped", "cohorts": cohorts,
                       "max_abs_dloglik": max(d, default=None), "definition_old": {"name": "normal_frac_sd", "v": 0.2,
                                                                                 "M": int(t1["het_M"])},
                       "definition_new": {"name": "truncated_normal", "sd": 0.2, "M": "auto"}}, budget, cap_rec)


# ------------------------------------------------------------------------------------------------- p0-het-cost
def phase_p0_het_cost(args):
    """P0 (tasks T039; research R8): the cost factors, the compilation, the timing noise, the gradients against central
    differences, a W in {1, 2, 3} identity smoke and node finiteness at M = 192, on 6 panel-external cohorts
    (tools/p0_het_cost_004.py)."""
    import p0_het_cost_004 as P0
    rel = "specs/004-samplers/evidence/cost_probe.json"
    projected = P0.projection_cpu_h(json.load(open(os.path.join(REPO, rel))))
    run, budget = phase_header("p0-het-cost", args, f"{len(P0.COHORTS)} cohorts at sd {P0.SD}, M = {P0.M}; a W = 1..3 "
                               f"identity smoke", projected, PILOT_CAPS["p0-het-cost"],
                               f"{rel} per-node costs (research R4)")
    man, specs = v6_specs()
    rows = run_capped(budget, P0.COHORTS, lambda cid: (P0.measure(cid, man, specs), 0.0))
    smoke = []
    if not budget.capped and rows:
        small = min(rows, key=lambda r: r["n"])["cohort_id"]
        smoke = P0.identity_smoke(small, man, specs)
        for x in smoke:
            budget.add_worker_cpu(x["worker_cpu_s"] + sum(x["compile_cpu_s_workers"].values()))
    summ = P0.summarise(rows, smoke) if rows else {}
    ok = len(rows) == len(P0.COHORTS) and summ.get("identity_smoke_all_equal") and summ.get("all_finite_at_192")
    finish(run, args, {"status": "written" if ok else ("capped" if budget.capped else "incomplete"), **summ,
                       "rows": rows, "identity_smoke": smoke, "cohorts": list(P0.COHORTS), "sd": P0.SD, "M": P0.M,
                       "repeats": P0.REPEATS}, budget)


# ------------------------------------------------------------------------------------------------- hr3, the pairs
def hr3_rules(pan) -> dict:
    """[rules.HR-3]: what the documents size but do not fix, frozen at freeze-H (tasks T041): `start` (multipliers on
    the truth, e.g. {eta = 1.5}), `profile` ({parameter, factors}: each profile point holds the parameter at the full-M
    arm's theta-hat times a factor), `sharp_strata` (the cells where the never-raise plant must exceed the threshold),
    `sds` (default [0.2, 0.3]); threshold 0.05."""
    r = dict((pan.get("rules") or {}).get("HR-3") or {})
    missing = [k for k in ("start", "profile", "sharp_strata") if k not in r]
    if missing:
        raise Refused(f"panel_004.toml [rules.HR-3] lacks {missing} (frozen at freeze-H, tasks T041)")
    r.setdefault("sds", [0.2, 0.3])
    r.setdefault("threshold", 0.05)
    return r


def hr3_pair(e, truth, sd, rules, pid, sharp) -> dict:
    """One HR-3 cell on `e`, an evaluator at this sd with automatic M: the adaptive and full-M maximisations from the
    frozen start, the never-raise plant, and the profile points (each arm, the parameter held at the full arm's
    theta-hat times each factor); every comparison at a common M (the larger final M), unchecked there."""
    from sraging import SRParams
    from sraging.engine import mixture as MIX
    from sraging.engine.series import AdaptivePolicy, maximise_fixed_sd
    d = truth.as_dict()
    start = SRParams(**{**d, **{k: d[k] * float(v) for k, v in rules["start"].items()}})

    def common(ra, rf):
        Mc = max(ra.M_final, rf.M_final)
        la = MIX.mixture_value(e, ra.theta_hat, het=het(sd, Mc, check=False)).value
        lf = MIX.mixture_value(e, rf.theta_hat, het=het(sd, Mc, check=False)).value
        return Mc, la, lf

    rows = []
    ra, rf = maximise_fixed_sd(e, start), maximise_fixed_sd(e, start, full_M=True)
    Mc, la, lf = common(ra, rf)
    rows.append(_pair_row(pid, "max", sharp, ra, rf, Mc, la, lf))
    plant = maximise_fixed_sd(e, start, policy=AdaptivePolicy(raise_M=False))
    lp = MIX.mixture_value(e, plant.theta_hat, het=het(sd, Mc, check=False)).value
    plant_row = {"id": pid, "sharp": sharp, "ll_plant": lp, "ll_full": lf, "M_common": Mc, "M_plant": plant.M_final,
                 "cpu_plant": plant.cost_record["total_cpu_s"]}
    par = rules["profile"]["parameter"]
    for f in rules["profile"]["factors"]:
        fx = {par: getattr(rf.theta_hat, par) * float(f)}
        pa, pf = maximise_fixed_sd(e, rf.theta_hat, fixed=fx), maximise_fixed_sd(e, rf.theta_hat, fixed=fx, full_M=True)
        Mp, lpa, lpf = common(pa, pf)
        rows.append(_pair_row(f"{pid}/profile{float(f):g}", "profile", sharp, pa, pf, Mp, lpa, lpf))
    return {"pairs": rows, "plant": plant_row}


def _pair_row(pid, kind, sharp, ra, rf, Mc, la, lf) -> dict:
    return {"id": pid, "kind": kind, "sharp": sharp, "M_common": Mc, "ll_adaptive": la, "ll_full": lf,
            "ll_final_adaptive": ra.ll_final, "ll_final_full": rf.ll_final, "M_final": [ra.M_final, rf.M_final],
            "M_low": ra.M_low, "cpu_adaptive": ra.cost_record["total_cpu_s"], "cpu_full": rf.cost_record["total_cpu_s"],
            "checked": ra.check["verdict"] == "converged" and rf.check["verdict"] == "converged",
            "verdicts": [ra.check["verdict"], rf.check["verdict"]],
            "phases": {"adaptive": list(ra.phases), "full": list(rf.phases)},
            "theta_hat": {"adaptive": ra.theta_hat.as_dict(), "full": rf.theta_hat.as_dict()}}


def hr3_cells(rows, rules, pilot=False) -> list:
    """(cohort_id, stratum, sd, sharp) for subset-hr3 x sds; the pilot's 2: the first sharp and the first human."""
    sub = [r for r in rows if "subset-hr3" in r["roles"]]
    if pilot:
        sharp = next(r for r in sub if r["stratum"] == "syn-sharp")
        human = next(r for r in sub if r["stratum"].startswith("human-"))
        return [(sharp["cohort_id"], sharp["stratum"], 0.2, True), (human["cohort_id"], human["stratum"], 0.2,
                                                                      human["stratum"] in rules["sharp_strata"])]
    return [(r["cohort_id"], r["stratum"], float(sd), r["stratum"] in rules["sharp_strata"]) for r in sub
            for sd in rules["sds"]]


def _hr3_work(rules):
    from sraging.engine.evaluator import Evaluator
    man, specs = v6_specs()

    def work(cell):
        cid, stratum, sd, sharp = cell
        coh, theta, h = v6_case(cid, man, specs)
        ev = Evaluator(coh, numerics="v1", fixed={"h_ext": h}, heterogeneity=het(sd))     # automatic M at this sd
        c0 = time.process_time()
        out = hr3_pair(ev, theta, sd, rules, f"{cid}/sd{sd:g}", sharp)
        out.update(cohort_id=cid, stratum=stratum, sd=sd, cpu_s=time.process_time() - c0)
        return out, 0.0
    return work


HR3_PROGRESS = os.path.join(REPO, ".cache", "hr3_004_progress.jsonl")      # gitignored; for diagnosis, never read back
HR3_TRACEBACK_LINES = 12


def hr3_error_row(cell, ex, tb, cpu_s) -> dict:
    """A cell whose work raised (Naveh 2026-10-30): its id, the exception, the traceback's last lines and the CPU it
    spent, with one pair row the frozen rule (hr3_verdict, 2f09ab9) reads as "no checked final value", hence
    incomplete. That rule subtracts ll_adaptive - ll_full on every pair, so the row carries equal 0.0 placeholders
    (named as such), never a measured value; its CPU fields are 0, so it adds nothing to the saving statistic."""
    cid, stratum, sd, sharp = cell
    pid = f"{cid}/sd{sd:g}"
    msg = f"{type(ex).__name__}: {ex}"
    pair = {"id": pid, "kind": "error", "sharp": sharp, "checked": False, "error": msg,
            "ll_adaptive": 0.0, "ll_full": 0.0, "cpu_adaptive": 0.0, "cpu_full": 0.0, "M_common": None,
            "placeholder": "ll_adaptive, ll_full, cpu_adaptive and cpu_full are 0.0 placeholders: the cell raised and "
                           "has no values (the frozen rule needs numbers; checked False makes it incomplete)"}
    return {"cohort_id": cid, "stratum": stratum, "sd": sd, "cpu_s": float(cpu_s), "pairs": [pair], "plant": None,
            "error": {"type": type(ex).__name__, "message": str(ex),
                      "traceback": tb.rstrip().splitlines()[-HR3_TRACEBACK_LINES:]}}


def hr3_progress(path):
    """After each cell: one flushed line (index/n, cell id, sd, ok or error, the cell's CPU, the CPU spent against
    the cap) and the cell's row appended to `path` as a JSON line."""
    def on_cell(i, n, cell, row, err, cpu_s, budget):
        import json
        cid, _, sd, _ = cell
        what = "ok" if err is None else f"error ({type(err).__name__}: {err})"
        print(f"[hr3] cell {i}/{n} {cid}/sd{sd:g} sd {sd:g}: {what}; cell CPU {cpu_s:.0f} s; spent "
              f"{budget.spent_cpu_h:.3g} of {budget.cap_cpu_h:g} CPU-h", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps({"index": i, "n": n, "cell": list(cell), "status": "ok" if err is None else "error",
                                 "cpu_s": cpu_s, "spent_cpu_h": budget.spent_cpu_h, "row": row},
                                default=_json_default) + "\n")
            fh.flush()
    return on_cell


def _json_default(o):
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    return repr(o)


def hr3_run(budget, cells, work, *, progress_path=HR3_PROGRESS) -> list:
    """hr3's cells under SR 3, a raising cell kept as an error row, with a progress line and a JSON line per cell."""
    return run_capped(budget, cells, work, on_error=hr3_error_row, on_cell=hr3_progress(progress_path))


def hr3_rows(out) -> tuple:
    """(pair rows, plant rows) for hr3_verdict; an error row has a pair and no plant."""
    return [p for o in out for p in o["pairs"]], [o["plant"] for o in out if o.get("plant") is not None]


def phase_hr3_pilot(args):
    """HR-3's pilot (tasks T051; gate C4): 2 pairs (one sharp, one human) with their profile points and plant; it
    projects hr3's CPU against its cap."""
    pan = read_toml(PANEL_PATH)
    require_frozen_ladder(pan)
    rules = hr3_rules(pan)
    left, cap_rec = frozen_cap("hr3-pilot")
    cells = hr3_cells(panel_rows(), rules, pilot=True)
    p0 = p0_per_node()
    n_max = 2 * (2 + 2 * len(rules["profile"]["factors"])) + 2          # maximisations, plants
    projected = n_max * 100 * 64 * p0["value_and_grad"] / 3600.0         # research Costs: ~100 evaluations x M ~ 64
    run, budget = phase_header("hr3-pilot", args, f"2 pairs: {n_max} maximisations", projected, left,
                               f"{p0['source']} x 100 evaluations x M ~ 64 per maximisation")
    out = run_capped(budget, cells, _hr3_work(rules))
    n_full = len(hr3_cells(panel_rows(), rules))
    per_cell_h = (float(np.mean([o["cpu_s"] for o in out])) / 3600.0) if out else None
    proj = None if per_cell_h is None else per_cell_h * n_full
    finish(run, args, {"status": "written" if len(out) == len(cells) else "capped", "cells": out,
                       "hr3_projection_cpu_h": proj, "hr3_cells": n_full, "per_cell_cpu_h": per_cell_h,
                       "rules": rules}, budget, cap_rec)


def phase_hr3(args):
    """HR-3 (tasks T052): 24 paired maximisations (adaptive / full M) plus the profile points, and the never-raise
    plant; projected from hr3-pilot, refused above the cap (10 CPU-h less the pilot's)."""
    from sraging import sampler_rules as R
    pan = read_toml(PANEL_PATH)
    require_frozen_ladder(pan)
    rules = hr3_rules(pan)
    left, cap_rec = frozen_cap("hr3")
    pilot = read_result("hr3-pilot")
    cells = hr3_cells(panel_rows(), rules)
    run, budget = phase_header("hr3", args, f"{len(cells)} pairs with their profile points and plant",
                               float(pilot["hr3_projection_cpu_h"]), left, "hr3_pilot_004.json hr3_projection_cpu_h")
    out = hr3_run(budget, cells, _hr3_work(rules))
    pairs, plants = hr3_rows(out)
    v = R.hr3_verdict(pairs, plants, cpu_spent=budget.spent_cpu_h, cap=left, threshold=rules["threshold"])
    finish(run, args, {"status": "capped" if budget.capped else v.status, "verdicts": [v.as_dict()], "cells": out,
                       "errors": [{"id": o["pairs"][0]["id"], **o["error"]} for o in out if o.get("error")],
                       "rules": rules}, budget, cap_rec)


# --------------------------------------------------------------------------------------------------------- hr4
HR4_W = tuple(range(1, 16))
HR4_M = (21, 31, 64)
HR4_WALL_REPEATS = 3


def hr4_rules(pan) -> dict:
    """[rules.HR-4]: `cohort` (the panel cohort the bytes are compared on) and `stated_W` (the W a speedup is claimed
    at), frozen at freeze-H (tasks T041)."""
    r = dict((pan.get("rules") or {}).get("HR-4") or {})
    missing = [k for k in ("cohort", "stated_W") if k not in r]
    if missing:
        raise Refused(f"panel_004.toml [rules.HR-4] lacks {missing} (frozen at freeze-H, tasks T041)")
    return r


def hr4_cell(ev, theta, W, M, pool, stated_W) -> dict:
    """One (W, M): pooled value and gradient bytes against in process, the remainder plant, and the walls."""
    from sraging.engine import mixture as MIX
    from sraging.engine import pool as POOL
    from sraging.engine import quadrature as Q
    h = het(ev.het.sd, M, check=False)
    w0 = time.perf_counter()
    r0 = MIX.mixture_value(ev, theta, grad=True, het=h)
    wall_serial = time.perf_counter() - w0
    w0 = time.perf_counter()
    r = MIX.mixture_value(ev, theta, grad=True, het=h, pool=pool)
    wall_pool = time.perf_counter() - w0
    plant = POOL.drop_remainder_value(ev, theta, Q.truncated_normal_rule(ev.het.sd, M), W)
    rows = [{"W": W, "M": M, "kind": "value", "equal": r.value.hex() == r0.value.hex()},
            {"W": W, "M": M, "kind": "grad", "equal": all(r.grad[k].hex() == r0.grad[k].hex() for k in r0.grad)}]
    walls = [{"W": W, "M": M, "rep": 0, "wall_pool": wall_pool, "wall_serial": wall_serial,
              "cpu_pool": r.cost_record["total_cpu_s"], "cpu_serial": r0.cost_record["total_cpu_s"]}]
    for i in range(1, HR4_WALL_REPEATS if W == stated_W else 1):
        w0 = time.perf_counter()
        MIX.mixture_value(ev, theta, grad=True, het=h)
        ws = time.perf_counter() - w0
        w0 = time.perf_counter()
        MIX.mixture_value(ev, theta, grad=True, het=h, pool=pool)
        walls.append({"W": W, "M": M, "rep": i, "wall_pool": time.perf_counter() - w0, "wall_serial": ws})
    return {"rows": rows, "plant": {"W": W, "M": M, "dropped": M % W, "differs": plant.hex() != r0.value.hex(),
                                    "abs_diff": abs(plant - r0.value)},
            "walls": walls, "worker_cpu_s": r.cost_record["worker_cpu_s"]}


def phase_hr4(args):
    """HR-4 (tasks T053; FR-007): bytes for W 1..15 x M in {21, 31, 64}, value and gradient; the wall and CPU curves;
    the remainder plant."""
    from sraging import sampler_rules as R
    from sraging.engine.pool import NodePool
    pan = read_toml(PANEL_PATH)
    require_frozen_ladder(pan)
    rules = hr4_rules(pan)
    left, cap_rec = frozen_cap("hr4")
    p0 = p0_per_node()
    nodes = sum(HR4_M)
    projected = (sum(HR4_W) * p0["compile"] + len(HR4_W) * 2 * nodes * p0["value_and_grad"]) / 3600.0
    run, budget = phase_header("hr4", args, f"{len(HR4_W)} pools x M {HR4_M}", projected, left,
                               f"{p0['source']}: compile x {sum(HR4_W)} worker starts + 2 x {nodes} nodes per W")
    evs = Evaluators()
    ev, theta = evs(rules["cohort"])

    def work(W):
        with NodePool(ev, workers=W) as pool:
            cells = [hr4_cell(ev, theta, W, M, pool, int(rules["stated_W"])) for M in HR4_M]
            comp = sum(pool.compile_cpu_s.values())
        return {"W": W, "cells": cells, "compile_cpu_s_workers": comp}, sum(c["worker_cpu_s"] for c in cells) + comp
    out = run_capped(budget, HR4_W, work)
    rows = [x for o in out for c in o["cells"] for x in c["rows"]]
    plants = [c["plant"] for o in out for c in o["cells"]]
    walls = [w for o in out for c in o["cells"] for w in c["walls"]]
    v = R.hr4_verdict(rows, plants, walls=walls, stated_W=int(rules["stated_W"]))
    finish(run, args, {"status": "capped" if budget.capped else v.status, "verdicts": [v.as_dict()], "pools": out,
                       "cohort": rules["cohort"], "sd": ev.het.sd}, budget, cap_rec)


# --------------------------------------------------------------------------------------------------------- hr5
def hr5_cells(hr1_rows) -> list:
    """HR-5's input from hr1's cells: per cell the stated and measured CPU summed over its statements (compilation
    apart) and the M-blind predictor's statement (cost.blind)."""
    from sraging.engine import cost as COST
    out = []
    for r in hr1_rows:
        st = r["cost"]["statements"]
        out.append({"id": r["id"], "M_used": r["M_used"], "stated_cpu_s": sum(s["stated_cpu_s"] for s in st),
                    "measured_cpu_s": sum(s["measured_cpu_s"] for s in st),
                    "blind_stated_cpu_s": sum(COST.blind(s)["stated_cpu_s"] for s in st)})
    return out


def phase_hr5(args):
    """HR-5 (tasks T049; OD-5 (a)): stated against measured CPU over hr1's cells, and the M-blind plant; it reads hr1's
    records, so it is projected at 0."""
    from sraging import sampler_rules as R
    pan = read_toml(PANEL_PATH)
    rules = (pan.get("rules") or {}).get("HR-5") or {}
    left, cap_rec = frozen_cap("hr5")
    hr1 = read_result("hr1")
    run, budget = phase_header("hr5", args, "0 evaluations: hr1's cost records", 0.0, left, "hr1_004.json's records")
    cells = hr5_cells(hr1["rows"])
    kw = {k: tuple(rules[k]) if isinstance(rules[k], list) else rules[k] for k in ("band", "share", "median_band")
          if k in rules}
    v = R.hr5_verdict(cells, **kw)
    finish(run, args, {"status": v.status, "verdicts": [v.as_dict()], "cells": cells,
                       "hr1_sha256": file_sha256(result_rel("hr1"))}, budget, cap_rec)


# ----------------------------------------------------------------------------------------------- P1's result
#: what the Langevin phases read from P1's result (tasks T072 writes them): "steps" {preset|hom-or-het|wall: {m, dt}},
#: "cells" [{preset, het, wall, ...}], "draw_cpu_s" {step key: CPU of one coupled draw at n = 25,000 at that step},
#: "ns_per_individual_step" and "steps_per_individual" {step key: mean steps per individual at that step}
def p1_result() -> dict:
    rel = result_rel("p1-langevin")
    if not committed(rel):
        raise Refused(f"no P1 result: {rel} is not committed (run p1-langevin first, tasks T111)")
    d = read_result("p1-langevin")
    if d.get("status_for_acceptance") != "acceptance":
        raise Refused(f"{rel} is from a {d.get('status_for_acceptance')} run")
    return d


# ------------------------------------------------------------------------------ the frozen steps: one accessor
class FrozenSteps(dict):
    """{cell id ("preset|hom|clamp", "s5|het|reflect"): {m, dt, steps_per_individual, draw_cpu_s, p_hat, met}} and
    its `source`. Reading a cell that is absent, or whose step is not met, is refused by name."""

    def __init__(self, entries, source):
        super().__init__(entries)
        self.source = source

    def __getitem__(self, key):
        if not dict.__contains__(self, key):
            raise Refused(f"step {key!r}: no such cell in {self.source}'s frozen steps")
        e = dict.__getitem__(self, key)
        if not e.get("met"):
            raise Refused(f"step {key!r}: not met in {self.source} (its judged bias did not reach crit/10): refused, "
                          f"never a silent step")
        return e

    def get(self, key, default=None):
        return self[key] if dict.__contains__(self, key) else default

    def met(self) -> dict:
        return {k: v for k, v in dict.items(self) if v.get("met")}


def frozen_steps() -> FrozenSteps:
    """THE accessor for the Langevin steps (P2, P3, S4, paths, multi, the plants and HR-8 read theirs here; Naveh
    2026-11-01, 8f0f269): p1-pooled's frozen steps where its result is committed from an acceptance run, merged with
    its follow-up p1-pooled-b's (c30e1ca; a cell in both is refused), else P1's (only the cells P1 met). Never P1's
    top-level "steps" once p1-pooled is written."""
    pooled = [ph for ph in ("p1-pooled", "p1-pooled-b") if committed(result_rel(ph))
              and read_result(ph).get("status_for_acceptance") == "acceptance"]
    if pooled:
        merged = {}
        for ph in pooled:
            d = read_result(ph)
            for k, v in d["frozen"].items():
                if k in merged:
                    raise Refused(f"step {k!r} is in both p1-pooled and p1-pooled-b: one cell, one measurement "
                                  f"(the follow-up runs only the cells p1-pooled did not finish)")
                last = ((d.get("cells") or {}).get(k) or {}).get("tried", [{}])[-1]
                Dk = last.get("D_per_key") or []
                merged[k] = {**v, "from": ph, "pooled_b": last.get("b"), "pooled_D": last.get("D_pooled"),
                             "bar": last.get("bar"), "D_key_mean": (sum(Dk) / len(Dk)) if Dk else None}
        return FrozenSteps(merged, "+".join(pooled))
    p1 = p1_result()
    spi, draw, p_hat = (p1.get(k) or {} for k in ("steps_per_individual", "draw_cpu_s", "p_hat"))
    return FrozenSteps({k: {"m": v.get("m"), "dt": v.get("dt"), "steps_per_individual": spi.get(k),
                            "draw_cpu_s": draw.get(k), "p_hat": p_hat.get(k), "met": bool(v.get("met"))}
                        for k, v in p1["steps"].items()}, "p1-langevin")


def p1_steps() -> dict:
    """P1's result with its step fields from the accessor: "steps" the FrozenSteps, "draw_cpu_s" and
    "steps_per_individual" of the met cells only; the rest (cells, b1, cell_ages, ns per individual-step) P1's."""
    p1 = p1_result()
    fs = frozen_steps()
    met = fs.met()
    return {**p1, "steps": fs, "steps_source": fs.source,
            "draw_cpu_s": {k: v["draw_cpu_s"] for k, v in met.items()},
            "steps_per_individual": {k: v["steps_per_individual"] for k, v in met.items()}}


# --------------------------------------------------------------------------------------------------------- s4
# ---------------------------------------------------------------------------------- the plants frozen in LANG
def frozen_plants(phase: str, frozen: dict, compute: dict) -> dict:
    """The plant values frozen in LANG beside the code's value now: {name: {"frozen", "now"}}. The phase uses the
    frozen one. Refused when one is not filled in LANG, or when the code computes another value now (the code drifted
    since freeze-L): exact equality, the same function on the same host and pinned environment."""
    out = {}
    for name, f in compute.items():
        fz = frozen.get(name)
        if fz is None or isinstance(fz, (dict, list, str, bool)):
            raise Refused(f"[{phase}] REFUSED: the plant value {name!r} is not filled in {LANG_PATH} (freeze-L; "
                          f"tools/lang_004_fill.py writes it)")
        now = float(f())
        if now != float(fz):
            raise Refused(f"[{phase}] REFUSED: the plant value {name!r} is {float(fz)!r} in {LANG_PATH} (freeze-L), but "
                          f"the code computes {now!r} now: it drifted since freeze-L (restore the code or re-freeze)")
        out[name] = {"frozen": float(fz), "now": now}
    return out


def s4_plant_checks(spec: dict, grid: dict) -> tuple:
    """(LANG's [s4].plant_delta, {plant cell: the code's delta now}) for frozen_plants: S4.plant_delta(params, het)
    on each plant cell's grid row."""
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    by_id = {r["id"]: r for r in grid["clamp"]}

    def one(row):
        return lambda: S4.plant_delta(PR.preset(row["preset"]), S4._het(row["parameter"], row["cv"]))["delta"]
    frozen = spec.get("plant_delta")
    return (frozen if isinstance(frozen, dict) and "t119" not in frozen else {},
            {pid: one(by_id[pid]) for pid in spec["plant_cells"]})


def paths_plant_check(spec: dict) -> tuple:
    """(LANG's [paths].plant.delta, the code's delta now: paths_004.plant_delta(params, age, n)) for frozen_plants."""
    import paths_004 as PA
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    pl = spec["plant"]
    het = S4._het("xc", 0.2) if pl["het"] == "het" else None
    return ({"delta": pl.get("delta")},
            {"delta": lambda: PA.plant_delta(PR.preset(pl["preset"]), float(pl["age"]), int(spec["n"]), het=het)["delta"]})


def multi_plant_checks(spec: dict, rows: list) -> tuple:
    """(LANG's [multi].plants[].predicted_sup_dS, the code's now: multi_004.plant_size(params, members, drop)) for
    frozen_plants, keyed "<cell>/drop-<drop>"."""
    import multi_004 as MU
    import presets_004 as PR
    by_id = {r["id"]: r for r in rows}
    params = PR.preset(spec["preset"])

    def one(pl):
        return lambda: MU.plant_size(params, by_id[pl["cell"]]["members"], pl["drop"])["predicted_sup_dS"]
    return ({f"{pl['cell']}/drop-{pl['drop']}": pl.get("predicted_sup_dS") for pl in spec["plants"]},
            {f"{pl['cell']}/drop-{pl['drop']}": one(pl) for pl in spec["plants"]})


def multi_plant_check_cpu_h(spec: dict) -> float:
    """The plant sizes' drift check: one tensor reference per plant at multi's frozen per-reference estimate
    (tensor_cpu_s_per_cell, 200 CPU-s; the six measured 1,259 CPU-s at 685a72e)."""
    return len(spec.get("plants", [])) * float(spec.get("tensor_cpu_s_per_cell", 200.0)) / 3600.0


def phase_s4(args):
    """S4 (tasks T081, T124; research R18; HR-6): the frozen grid, the default clamp on every row but E. coli's
    (clamp_excluded) and reflect on P1's 8 cells, each row ONCE at its frozen step and judged on its dt/2 rung
    (amendment A1, R18's branch: assessability from the frozen cell's pooled bias, a borrowed row guarded); HR-6's G is
    each family's actual row count (amendment A3); the eta(1 + delta) plant on its 2 cells; a verdict per wall. Cap 35
    CPU-h, projected from each row's frozen draw CPU."""
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    from sraging import sampler_rules as R
    lang = read_toml(LANG_PATH)
    spec = lang["s4"]
    left, cap_rec = frozen_cap("s4")
    p1 = p1_steps()
    fs = p1["steps"]
    grid = S4.build_grid(spec, p1)
    rows = grid["clamp"] + grid["reflect"]
    G = {w: len(grid[w]) for w in ("clamp", "reflect")}
    projected = s4_projected_cpu_h(grid, p1["draw_cpu_s"])
    run, budget = phase_header("s4", args, f"{len(rows)} cells (clamp {G['clamp']}, reflect {G['reflect']}) once at "
                               f"their frozen steps, and 2 plants", projected, left,
                               "each row's frozen draw CPU (p1-pooled's per-key coupled pair), once")
    keys = {r["id"]: int(spec["key"]) + i for i, r in enumerate(rows)}
    plant_check = frozen_plants("s4", *s4_plant_checks(spec, grid))      # LANG's deltas; refused on code drift
    W = int(getattr(args, "workers", None) or 1)
    cells = robust_run_pool("s4", budget, rows, "s4_row_task",
                            lambda r: (r, keys[r["id"]], dict(fs[r["p1_step"]])), W,
                            label=lambda r: r["id"], error_row=s4_error_row)
    plants = []
    if not budget.capped:
        by_id = {r["id"]: r for r in grid["clamp"]}

        def plant(pid):
            row = by_id[pid]
            params, het = PR.preset(row["preset"]), S4._het(row["parameter"], row["cv"])
            return S4.run_plant(row, params, keys[pid] + 10_000, plant_check[pid]["frozen"], het=het), 0.0
        plants = robust_run("s4", budget, list(spec["plant_cells"]), plant, label=lambda pid: f"plant {pid}",
                            error_row=lambda pid, e: {"id": f"{pid}/plant", "error": e})
    verdicts = [R.hr6_verdict([c for c in cells if c["wall"] == w], plants if w == "clamp" else [], wall=w,
                              cpu_spent=budget.spent_cpu_h, cap=left, G=G[w]) for w in ("clamp", "reflect")]
    ok = [c for c in cells if not c.get("error")]
    crit = {w: R.family_crit(G[w]) / math.sqrt(int(spec["n"])) for w in G}
    finish(run, args, {"status": "capped" if budget.capped else combined_status(v.status for v in verdicts),
                       "verdicts": [v.as_dict() for v in verdicts], "cells": cells, "plants": plants, "G": G,
                       "plant_check": plant_check, "workers": W, "errors": errors_of(cells + plants),
                       "listed": [{"id": c["id"], "wall": c["wall"], "n": c["n"], "ks_D": c["ks_D"],
                                   "crit": crit[c["wall"]], "frozen_b": c["frozen_b"], "D_own": c["D_own"]}
                                  for c in ok if c["ks_D"] > crit[c["wall"]]]},
           budget, cap_rec)


def s4_projected_cpu_h(grid, draw_cpu_s) -> float:
    """Each row once at its frozen step: its step's draw CPU (p1-pooled's per-key coupled pair, dt and dt/2 rungs)."""
    return sum(float(draw_cpu_s[r["p1_step"]]) for w in ("clamp", "reflect") for r in grid[w]) / 3600.0


def s4_error_row(row, err) -> dict:
    """An S4 cell that raised: its id, wall and n, and the error; HR-6 reads it as unassessable."""
    return {"id": row["id"], "wall": row["wall"], "n": row["n"], "error": err}


# ------------------------------------------------------------------------------------------------ p1-langevin
def phase_p1_langevin(args):
    """P1 (tasks T072, T111; gate C1, C4): the 16 cells (4 presets x hom and xc sd 0.2 x both walls) at m in {1, 2,
    4, 8, 16}, n = 25,000, with paths at the HR-10 ages; each cell's p and step, the clamp's extension to 32 and 64
    where 16 misses the bar; D's noise floor, Dd's overhead, B1's predicted lag, SRK's 7.2 s / 0.45 s re-measured.
    Cap 6 CPU-h, projected from a probe draw per preset."""
    import oracle_fp_piecewise as OP
    import p1_langevin_004 as P1
    import presets_004 as PR
    from sraging.engine.inputs import Heterogeneity
    cells = P1.cells()
    probes = {p: P1.probe(PR.preset(p), PR.base_dt(p)) for p in PR.NAMES}
    ns = float(np.median([x["ns_per_individual_step"] for x in probes.values()]))
    projected = P1.projected_cpu_h(ns, {p: x["steps_per_individual"] for p, x in probes.items()})
    run, budget = phase_header("p1-langevin", args, f"{len(cells)} cells x m {list(P1.M_RUNGS)} at n {P1.N:,}, the "
                               f"noise floor, Dd, B1's lag, SRK's re-measure", projected, PILOT_CAPS["p1-langevin"],
                               f"a probe draw per preset (n 1,000, m 1): {ns:.3g} ns per individual-step")
    keys = {c["id"]: 70_000 + 100 * i for i, c in enumerate(cells)}
    rows = robust_run("p1-langevin", budget, cells, lambda c: (P1.run_cell(c, PR.preset(c["preset"]), keys[c["id"]]),
                                                               0.0),
                      label=lambda c: c["id"], error_row=lambda c, e: {**c, "error": e})
    extras, noise, ext = {}, [], []
    if not budget.capped:
        by_id = {c["id"]: c for c in cells}
        noise = robust_run("p1-langevin", budget, list(P1.NOISE_CELLS), lambda k: (P1.noise_floor(
            by_id[_p1_key(*k)], PR.preset(k[0]), [keys[_p1_key(*k)] + j for j in range(P1.NOISE_KEYS)]), 0.0),
            label=lambda k: f"noise {_p1_key(*k)}", error_row=lambda k, e: {"id": f"noise {_p1_key(*k)}", "error": e})
        extras["noise"] = noise
    if not budget.capped:
        def extra(item):
            what, p = item
            if what == "dd":
                return {"id": "dd", **P1.dd_overhead(PR.preset(p), PR.base_dt(p))}, 0.0
            if what == "b1":
                return {"id": f"b1 {p}", "preset": p,
                        **{het: P1.b1_lag(PR.preset(p), None if het == "hom" else Heterogeneity(parameter="xc", sd=0.2),
                                          PR.base_dt(p)) for het in ("hom", "het")}}, 0.0
            with OP.session() as srk:                                  # its worker's CPU is the ops' cpu_s
                r = P1.srk_remeasure(srk, p)
            return {"id": f"srk {p}", **r}, r["langevin_cpu_s"] + r["fp_mixture_cpu_s"]
        items = [("dd", "mice_F")] + [("b1", p) for p in PR.NAMES] + [("srk", p) for p in PR.NAMES]
        ext = robust_run("p1-langevin", budget, items, extra, label=lambda it: f"{it[0]} {it[1]}")
        ok = [r for r in ext if not r.get("error")]
        extras["dd"] = next((r for r in ok if r["id"] == "dd"), None)
        extras["b1"] = {r["preset"]: {h: r[h] for h in ("hom", "het")} for r in ok if r["id"].startswith("b1 ")}
        extras["srk"] = {"host": P1.host(), "rows": [r for r in ok if r["id"].startswith("srk ")]}
    out = P1.result(rows, noise=extras.get("noise"), dd=extras.get("dd"), b1=extras.get("b1"), srk=extras.get("srk"))
    errs = errors_of(rows + noise + ext)
    done = len(rows) == len(cells) and bool(ext) and not errs
    status = "capped" if budget.capped else "written" if done else "incomplete"
    finish(run, args, {"status": status, "probes": probes, "errors": errs, **out}, budget)


# ----------------------------------------------------------------------------------------------- plants, hr8
def phase_plants(args):
    """R14's plants and twins against SRK's frozen code (tasks T074, T121; gate C3): D-402 (HR-9 like for like) to
    D-411, each at its frozen configuration (langevin_004.toml [plants], one table per plant with its preset), each
    with its status (pass / fail / incomplete) and sup|dS|. Cap 3 CPU-h, projected from P1's draw CPU and SRK's
    re-measure per individual; the oracle workers' CPU is counted."""
    import oracle_fp_piecewise as OP
    import plants_004 as PL
    import presets_004 as PR
    lang = read_toml(LANG_PATH)
    spec = lang.get("plants") or {}
    missing = [k for k in PL.ORDER if k not in spec]
    if missing:
        raise Refused(f"[plants] REFUSED: langevin_004.toml [plants] lacks {missing} (frozen at freeze-L)")
    left, cap_rec = frozen_cap("plants")
    p1 = p1_steps()
    spec = PR.resolve(spec, p1, steps=p1["steps"])              # HR-9's fix_dt: the frozen step, by name
    projected = PL.projected_cpu_h(spec, plants_rates(p1))
    run, budget = phase_header("plants", args, f"{len(PL.ORDER)} plants with their twins (D-402 to D-411)",
                               projected, left, "each plant at its own steps: n x lifetime / step x the CPU per "
                               "individual-step (SRK's from P1's re-measure, sraging's P1's)")
    key = int(spec.get("key", 4400))
    with OP.session() as raw:
        srk = PL.CountingSession(raw)

        def work(name):
            before = srk.cpu_s
            cfg = spec[name]
            row = PL.run_plant(name, srk, PR.preset(cfg["preset"]), cfg, key + 100 * PL.ORDER.index(name))
            return {**row, "preset": cfg["preset"]}, srk.cpu_s - before
        rows = robust_run("plants", budget, list(PL.ORDER), work, error_row=plant_error_row)
    status = "capped" if budget.capped else combined_status(r["status"] for r in rows)
    finish(run, args, {"status": status, "plants": rows, "errors": errors_of(rows),
                       "sup_abs_dS": {r["id"]: r["sup_abs_dS"] for r in rows},
                       "hr9": next((r["verdict"] for r in rows if r["id"] == "D-402"), None)}, budget, cap_rec)


def plants_rates(p1) -> dict:
    """The plants' price inputs from P1 (Naveh 2026-11-03: each plant at its own step): per preset the mean lifetime
    (P1's m = 1 rung's steps per individual x the base step), sraging's CPU per individual-step (P1's ns), and SRK's
    (P1's re-measure of mice_F: its CPU / (n x the same steps per individual at SRK's own step, the base step))."""
    life, m1 = {}, {}
    for c in p1["cells"]:
        if c["het"] == "hom" and c["wall"] == "clamp":
            r = min(c["ladder"], key=lambda r: r["m"])
            life[c["preset"]] = float(r["steps_per_individual"]) * float(r["dt"])
            m1[c["preset"]] = float(r["steps_per_individual"])
    srk = next(r for r in p1["srk_remeasure"]["rows"] if r["preset"] == "mice_F")
    return {"lifetime": life, "ns_sraging": float(p1["ns_per_individual_step"]) * 1e-9,
            "ns_srk": float(srk["langevin_cpu_s"]) / (float(srk["n"]) * m1["mice_F"])}


def plant_error_row(name, err) -> dict:
    """A plant that raised: incomplete (it measured nothing), never pass."""
    return {"id": f"plant {name}", "name": name, "status": "incomplete", "sup_abs_dS": None, "error": err}


def phase_hr8(args):
    """HR-8 (tasks T074, T123; research R17): the bytes at W in {1, 7, 15} against W = 1 at n = 25,013, and the B4
    plant per W; the configuration is ROLES["hr8"] (Naveh 2026-10-30: mice_F homogeneous at the clamp step), its step
    from the frozen-step accessor. Cap 0.5 CPU-h."""
    import plants_004 as PL
    import presets_004 as PR
    from sraging import sampler_rules as R
    cfg = PR.role("hr8")
    left, cap_rec = frozen_cap("hr8")
    p1 = p1_steps()
    cfg = PR.resolve(cfg, p1, steps=p1["steps"])                # its step: the frozen clamp step, by name
    k = f"{cfg['preset']}|{cfg.get('het', 'hom')}|clamp"
    dt = float(cfg["dt"])
    n, Ws = int(cfg.get("n", 25_013)), [int(w) for w in cfg.get("W", (1, 7, 15))]
    spi = float((p1.get("steps_per_individual") or {}).get(k) or 0.0)
    projected = (2 * len(Ws) - 1) * n * spi * float(p1["ns_per_individual_step"]) * 1e-9 / 3600.0
    run, budget = phase_header("hr8", args, f"{cfg['preset']} at dt {dt:g}, n {n:,}, W {Ws}", projected, left,
                               f"P1's {k} steps per individual x ns per individual-step x {2 * len(Ws) - 1} draws")
    out = run_capped(budget, ["hr8"], lambda _: (PL.hr8_rows(PR.preset(cfg["preset"]), {**cfg, "dt": dt, "n": n,
                                                                                         "W": Ws},
                                                             int(cfg.get("key", 4800))), 0.0))
    rows, plants = out[0] if out else ([], [])
    v = R.hr8_verdict(rows, plants)
    finish(run, args, {"status": "capped" if budget.capped else v.status, "verdicts": [v.as_dict()], "rows": rows,
                       "plants": plants, "config": {**cfg, "dt": dt}}, budget, cap_rec)


# ------------------------------------------------------------------------------------------------------ p4-jax
P4_PRICE_CPU_H = 0.2                # research R19's [I]; re-stated from P1's ns per individual-step when committed


def p4_steps_per_individual(p1, k):
    """Steps per individual at P1's m = 1 rung of cell `k`: P4 draws at the preset's base step (m = 1). P1's top-level
    `steps_per_individual` holds the cell's finest rung (m = 64 on a clamp cell), 64 times too many for P4."""
    cell = next((c for c in p1.get("cells") or [] if c.get("id") == k), None)
    rung = next((x for x in (cell or {}).get("ladder") or [] if x.get("m") == 1), None)
    return (rung or {}).get("steps_per_individual")


def phase_p4_jax(args):
    """P4 (tasks T073; OD-7 (a)): the JAX prototype against the numba kernel on one preset (ROLES["p4"]: the human
    preset, n = 25,000; Naveh 2026-10-30), 20 paired repeats, compilation apart, in a subprocess with single-threaded
    XLA; the ratio's upper 95 % bound and the decision at 1.2. Cap 1 CPU-h."""
    import resource
    import subprocess
    import tempfile
    import p4_jax_langevin_004 as P4
    import presets_004 as PR
    cfg = PR.role("p4")
    preset, n = cfg["preset"], int(cfg.get("n", 25_000))
    projected, source = P4_PRICE_CPU_H, "research R19's [I] 0.2 CPU-h"
    if committed(result_rel("p1-langevin")):
        p1 = p1_result()
        k = f"{preset}|hom|clamp"
        spi = p4_steps_per_individual(p1, k)
        if spi:
            # the numba draws, and the JAX draws over every step (t_max / dt per individual), 20 repeats each
            projected = 2 * P4.REPEATS * n * spi * float(p1["ns_per_individual_step"]) * 1e-9 * 3 / 3600.0
            source = (f"P1's {k} m = 1 rung: steps per individual x ns per individual-step x 20 pairs (x 3 for the "
                      f"scan)")
    run, budget = phase_header("p4-jax", args, f"{preset}, n {n:,}, {P4.REPEATS} paired repeats", projected,
                               PILOT_CAPS["p4-jax"], source)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory(dir=os.path.join(REPO, ".cache")) as d:
        out_json = os.path.join(d, "p4.json")
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "p4_jax_langevin_004.py"), "run", "--preset",
                            preset, "--n", str(n), "--repeats", str(P4.REPEATS), "--out", out_json],
                           capture_output=True, text=True)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        budget.add_worker_cpu((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime))
        if r.returncode != 0:
            raise Refused(f"[p4-jax] the prototype failed: {r.stderr.strip()[-2000:]}")
        import json
        out = json.load(open(out_json))
    finish(run, args, {"status": "written", "config": cfg, **out}, budget)


# ------------------------------------------------------------------------------------------------- p1-pooled
P1_POOLED_W = 5                     # bias is not timed: the cells' keys run across W processes


def phase_p1_pooled(args):
    """p1-pooled (tasks T111b; spec.md Clarifications 2026-11-01, 8f0f269; research gate C1, "the noise floor"): each
    Langevin cell's step frozen from a measurement at pooled n. P1's 16 cells start at the step extrapolated from P1's
    own ladder; S5's two cells (ROLES["s5"]) run a short ladder first. At a step, K = 18 keys of a coupled dt / dt-2
    pair at n = 25,000 are pooled (n = 450,000) and b = D / (2^p - 1) judged against the n = 25,000 bar; over it, m
    doubles (at most twice), then "not met". W = 5 processes; the keys come from (cell, k), so nothing depends on W.
    Cap 45 CPU-h (Naveh 2026-11-01), projected from P1's rung CPU at each start step."""
    import p1_pooled_004 as PP
    import presets_004 as PR
    p1 = p1_result()
    W = int(getattr(args, "workers", None) or P1_POOLED_W)
    s5 = PR.role("s5")
    cells = [PP.with_t_max(c) for c in PP.p1_cells(p1)]
    proj = PP.projected_cpu_h(cells)
    s5_probe = {w: PP.s5_probe(s5, w) for w in ("clamp", "reflect")}
    s5_proj = {w: PP.s5_projected_cpu_h(pr) for w, pr in s5_probe.items()}
    total = proj["total_cpu_h"] + sum(x["cpu_h"] for x in s5_proj.values())
    for k_, r in proj["cells"].items():
        print(f"[p1-pooled] {k_}: start m {r['m_start']}, {r['per_key_s']:.0f} CPU-s per key, {r['cpu_h']:.2f} CPU-h "
              f"at K = {proj['K']}", flush=True)
    for w, x in s5_proj.items():
        print(f"[p1-pooled] s5|het|{w}: ladder {x['ladder_cpu_h']:.2f} CPU-h + K keys at the nominal m "
              f"{x['m_nominal']} {x['pooled_cpu_h']:.2f} CPU-h (its start step is extrapolated after its ladder)",
              flush=True)
    run, budget = phase_header("p1-pooled", args, f"{len(cells) + 2} cells x K = {PP.K} keys at n {PP.N:,} at each "
                               f"start step (doublings not counted); W = {W}", total, PILOT_CAPS["p1-pooled"],
                               "P1's rung CPU (dt and dt/2 rungs) x m_start / m_f x K; S5's from a probe pair")
    with PP.Pool(W) as pool:
        def s5_work(w):
            c = PP.s5_ladder(s5, w, pool)
            return c, c["worker_cpu_s"]
        s5_cells = robust_run("p1-pooled", budget, ["clamp", "reflect"], s5_work, label=lambda w: f"s5 ladder {w}",
                              error_row=lambda w, e: {"id": f"s5|het|{w}", "error": e})
        cells += [c for c in s5_cells if not c.get("error")]
        recs = [c for c in s5_cells if c.get("error")]
        if not budget.capped:
            recs += robust_run("p1-pooled", budget, cells, lambda c: PP.run_cell(c, pool), label=lambda c: c["id"],
                               error_row=lambda c, e: {"id": c["id"], "error": e})
    out = PP.result(recs)
    errs = errors_of(recs)
    status = "capped" if budget.capped else "incomplete" if errs or len(out["cells"]) < 18 else "written"
    finish(run, args, {"status": status, "errors": errs, "projection": {**proj, "s5": s5_proj, "total_cpu_h": total},
                       "workers": W, "K": PP.K, **out}, budget)


# ----------------------------------------------------------------------------------------------- p1-pooled-b
#: Naveh's pre-authorization (spec.md Clarifications 2026-11-01, c30e1ca): the follow-up's cap is the missing cells'
#: projection + 25 %, at most 15 CPU-h; SR 4's OK is recorded as given there unless --naveh-ok gives another
P1_POOLED_B = {"factor": 1.25, "ceiling": 15.0, "said": "2026-11-01: OK a follow-up if it caps (Recommended)",
               "source": "spec.md Clarifications 2026-11-01 (c30e1ca)"}
#: p1-pooled's CPU over its per-cell projection, measured during its run ([M] 2026-11-01 18:45, 5 of 18 cells: mice_F
#: hom clamp 7,431 s against 6,010, Jack Russell hom clamp 29,860 against 24,660): the larger ratio. The per-key work
#: is the kernels only (no FP reference or KS is built per key; the pooled ECDFs are milliseconds), so this is the
#: kernel's CPU under W = 5 concurrent processes against P1's single-process rung CPU [I].
P1_POOLED_OVERHEAD = 1.24


def p1_pooled_b_cap(projected_cpu_h) -> float:
    """The follow-up's cap: its projection x 1.25, at most 15 CPU-h (c30e1ca)."""
    return min(P1_POOLED_B["factor"] * float(projected_cpu_h), P1_POOLED_B["ceiling"])


def p1_pooled_missing(prev: dict, p1: dict) -> list:
    """The cells p1-pooled did not finish (unreached at its cap, or an error row): P1's 16 and S5's 2, in p1-pooled's
    own order."""
    import p1_pooled_004 as PP
    done = set(prev.get("cells") or {})
    expected = [c["id"] for c in p1["cells"]] + list(PP.S5_KEYS)
    return [k for k in expected if k not in done]


def run_p1_pooled_b(budget, p1, missing, s5cfg, W) -> list:
    """The missing cells under p1-pooled's own code and keys (p1_pooled_004: start steps from P1's ladders, S5's
    ladders, K pooled keys, two doublings), as p1-pooled would have run them."""
    import p1_pooled_004 as PP
    cells = [PP.with_t_max(c) for c in PP.p1_cells(p1) if c["id"] in missing]
    walls = [w for w in ("clamp", "reflect") if f"s5|het|{w}" in missing]
    with PP.Pool(W) as pool:
        def s5_work(w):
            c = PP.s5_ladder(s5cfg, w, pool)
            return c, c["worker_cpu_s"]
        s5_cells = robust_run("p1-pooled-b", budget, walls, s5_work, label=lambda w: f"s5 ladder {w}",
                              error_row=lambda w, e: {"id": f"s5|het|{w}", "error": e})
        cells += [c for c in s5_cells if not c.get("error")]
        recs = [c for c in s5_cells if c.get("error")]
        if not budget.capped:
            recs += robust_run("p1-pooled-b", budget, cells, lambda c: PP.run_cell(c, pool), label=lambda c: c["id"],
                               error_row=lambda c, e: {"id": c["id"], "error": e})
    return recs


def phase_p1_pooled_b(args):
    """p1-pooled's follow-up (tasks T111c; spec.md Clarifications 2026-11-01, c30e1ca): only the cells p1-pooled did
    not finish (a capped run's unreached cells, any error row), with p1-pooled's code, keys, K, start steps and
    doubling rule, so its result is what p1-pooled would have produced; the accessor merges the two. Refused when no
    cell is missing. Cap: the missing cells' projection (x the measured overhead) + 25 %, at most 15 CPU-h."""
    import p1_pooled_004 as PP
    import presets_004 as PR
    p1, prev = p1_result(), read_result("p1-pooled")
    missing = p1_pooled_missing(prev, p1)
    if not missing:
        raise Refused("[p1-pooled-b] REFUSED: nothing missing: p1-pooled finished all 18 cells")
    W = int(getattr(args, "workers", None) or P1_POOLED_W)
    s5 = PR.role("s5")
    proj = PP.projected_cpu_h([c for c in PP.p1_cells(p1) if c["id"] in missing], overhead=P1_POOLED_OVERHEAD)
    s5_proj = {w: PP.s5_projected_cpu_h(PP.s5_probe(s5, w), overhead=P1_POOLED_OVERHEAD)
               for w in ("clamp", "reflect") if f"s5|het|{w}" in missing}
    total = proj["total_cpu_h"] + sum(x["cpu_h"] for x in s5_proj.values())
    cap = p1_pooled_b_cap(total)
    oks = naveh_oks("p1-pooled-b", args) or (NavehOk("p1-pooled-b", cap, P1_POOLED_B["said"]),)
    print(f"[p1-pooled-b] missing {missing}; projected {total:.2f} CPU-h (x {P1_POOLED_OVERHEAD} measured overhead); "
          f"cap {cap:.2f} = min(1.25 x projected, 15) ({P1_POOLED_B['source']})", flush=True)
    run = _run("p1-pooled-b", args, f"{len(missing)} missing cells x K = {PP.K} keys at n {PP.N:,}; W = {W}")
    budget = stop_rule("p1-pooled-b", total, PILOT_CAPS["p1-pooled-b"], oks,
                       source="P1's rung CPU at each start step x the measured overhead; S5's from a probe pair")
    recs = run_p1_pooled_b(budget, p1, missing, s5, W)
    out = PP.result(recs)
    errs = errors_of(recs)
    status = "capped" if budget.capped else "incomplete" if errs or len(out["cells"]) < len(missing) else "written"
    finish(run, args, {"status": status, "errors": errs, "missing": missing,
                       "previous": {"phase": "p1-pooled", "repo_head": prev.get("repo_head"),
                                    "status": prev.get("status")},
                       "projection": {**proj, "s5": s5_proj, "total_cpu_h": total, "cap_cpu_h": cap,
                                      "overhead": P1_POOLED_OVERHEAD},
                       "workers": W, "K": PP.K, **out}, budget)


def _p1_key(preset, het, wall) -> str:
    import s4_fp_langevin_004 as S4
    return S4.step_key(preset, het, wall)


# ------------------------------------------------------------------------------------------------------ paths
def paths_extra(item, rows, spec, plant_check, params, het) -> tuple:
    """One of paths' extras after its configurations, `item` = (what, i, wall): the plant, twin or checks of a wall, or
    SRK's recorder ("srk", no wall), each on the plant configuration's row of its wall (SRK's: the clamp row, its step
    and ages, as projected_cpu_h prices it). -> (row, worker CPU 0.0). At 252ad3a the srk item looked up a row for
    wall None (StopIteration), and behind that indexed [paths.ages] by preset alone (they are keyed preset|class)."""
    import oracle_fp_piecewise as OP
    import paths_004 as PA
    what, i, w = item
    plant_spec = spec["plant"]
    row = next(r for r in rows if r["wall"] == ("clamp" if what == "srk" else w)
               and r["preset"] == plant_spec["preset"] and (r["sd"] > 0) == (plant_spec["het"] == "het"))
    if what == "srk":
        with OP.session() as srk:
            return PA.srk_recorder(srk, params, row["ages"], n=int(spec["srk_n"]), dt=row["dt"],
                                   seed=int(spec["key"]) + 20_000), 0.0
    k0 = int(spec["key"]) + 10_000 + 10 * i
    if what == "plant":
        return PA.run_plant(row, params, float(plant_spec["age"]), plant_check["delta"]["frozen"], k0, het=het,
                            kind="plant"), 0.0
    if what == "twin":
        return PA.run_plant(row, params, float(plant_spec["age"]), 0.0, k0 + 1, het=het, kind="twin"), 0.0
    return {"id": f"checks/{w}", "wall": w, **PA.deterministic_checks(row, params, k0 + 2, het=het,
                                                                        W_list=spec["W"])}, 0.0


def phase_paths(args):
    """HR-10 (tasks T098, T125; research R24, R25): the frozen configurations (each preset homogeneous and at xc sd
    0.2, two ages each, per preset and class) under both walls, each ONCE at its frozen step, judged on the dt/2 rung
    against damage_density (the clamp censored at its lump, reflect uncensored); amendment A2 (Naveh 2026-11-03): E.
    coli has no clamp configuration, the named clamp tests are judged on reflect only, and assessability is the
    death-time frozen step's pooled bias (the path bias reported, not gating). The eta(1 + delta) plant and its twin
    per wall; the deterministic checks per wall; SRK's zero-for-dead recorder (OFP simulate_isr_paths) through the
    NaN check. A verdict per wall. Cap 6 CPU-h, projected from each configuration's frozen draw CPU."""
    import paths_004 as PA
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    from sraging import sampler_rules as R
    lang = read_toml(LANG_PATH)
    spec, judged = lang["paths"], lang.get("judged") or {}
    left, cap_rec = frozen_cap("paths")
    p1 = p1_steps()
    rows = PA.build_configs(spec, p1)
    cost, p_hat = p1.get("draw_cpu_s") or {}, judged.get("p_hat") or {}
    missing = sorted({r["p1_step"] for r in rows if r["p1_step"] not in cost or r["p1_step"] not in p_hat})
    if missing:
        raise Refused(f"[paths] REFUSED: no P1 draw CPU or frozen p_hat ([judged]) for {missing}")
    projected = PA.projected_cpu_h(rows, cost, spec)
    run, budget = phase_header("paths", args, f"{len(rows)} configurations x 2 ages once at their frozen steps (G = "
                               f"{PA.G} per wall), the plant and twin and the checks per wall, SRK's recorder", projected,
                               left, "each configuration's frozen draw CPU, plus one draw per plant, twin, check and "
                               "SRK's recorder at the plant configuration's")
    keys = {r["id"]: int(spec["key"]) + i for i, r in enumerate(rows)}
    skip = PA.reflect_only(spec)
    fs = p1["steps"]
    W = int(getattr(args, "workers", None) or 1)
    results = robust_run_pool("paths", budget, rows, "paths_row_task", lambda r: (
        r, keys[r["id"]], dict(fs[r["p1_step"]]), [j for c, j in skip if c == r["p1_step"]]), W,
        label=lambda r: r["id"], error_row=paths_error_row)
    plants, checks, extras, plant_spec = [], {}, [], spec["plant"]
    plant_check = frozen_plants("paths", *paths_plant_check(spec))       # LANG's delta; refused on code drift
    if not budget.capped:
        params = PR.preset(plant_spec["preset"])
        het = S4._het("xc", 0.2) if plant_spec["het"] == "het" else None

        items = [(what, i, w) for i, w in enumerate(PA.WALLS) for what in ("plant", "twin", "checks")]
        items.append(("srk", 0, None))
        extras = robust_run("paths", budget, items, lambda it: paths_extra(it, rows, spec, plant_check, params, het),
                            label=lambda it: f"{it[0]}/{it[2]}",
                            error_row=lambda it, e: {"id": f"{it[0]}/{it[2]}", "kind": it[0], "wall": it[2],
                                                     "error": e})
        for r in extras:
            if r.get("kind") == "checks" or r.get("id", "").startswith("checks/"):
                if not r.get("error"):
                    checks[r["wall"]] = {k: v for k, v in r.items() if k not in ("id", "wall")}
            else:
                plants.append(r)
    verdicts = []
    for w in PA.WALLS:
        mine = [r for r in results if r["wall"] == w]
        wall_checks = dict(checks.get(w, {}))
        done = [r for r in mine if not r.get("error")]
        if done:
            wall_checks["nan_iff_dead"] = {"holds": all(r["nan_check"]["holds"] for r in done),
                                           "per_configuration": {r["id"]: r["nan_check"] for r in done}}
        verdicts.append(R.hr10_verdict([t for r in mine for t in r["tests"]], wall_checks,
                                       [p for p in plants if p.get("wall") in (w, None)], wall=w,
                                       cpu_spent=budget.spent_cpu_h, cap=left))
    finish(run, args, {"status": "capped" if budget.capped else combined_status(v.status for v in verdicts),
                       "verdicts": [v.as_dict() for v in verdicts], "configurations": results, "plants": plants,
                       "checks": checks, "plant_spec": plant_spec, "plant_check": plant_check, "workers": W,
                       "errors": errors_of(results + extras),
                       "listed": [t for r in results if not r.get("error") for t in r["tests"]
                                  if (t["D_c"] if r["wall"] == "clamp" else t["D"]) is None
                                  or (t["D_c"] if r["wall"] == "clamp" else t["D"]) > PA.crit(t["n_alive"])]},
           budget, cap_rec)


def paths_error_row(row, err) -> dict:
    """A configuration that raised: one test row carrying the error, which HR-10 reads as unassessable."""
    return {"id": row["id"], "wall": row["wall"], "error": err, "nan_check": None,
            "tests": [{"id": f"{row['id']}/raised", "wall": row["wall"], "error": err}]}


# ------------------------------------------------------------------------------------------------------ multi
def multi_composite_bar(spec: dict) -> float:
    """[multi].composite.bar: the bar the cross-checked tensor reference is built at (amendment A9, Naveh 2026-11-05:
    1e-4, not the cells' crit/10); refused when the freeze does not name it."""
    c = spec["composite"]
    if "bar" not in c:
        raise Refused("[multi] REFUSED: [multi].composite names no bar (amendment A9: the tensor reference it "
                      "cross-checks is built at 1e-4, never at the cells' crit/10 by default)")
    return float(c["bar"])


def multi_composite(spec: dict, by_id: dict, params) -> dict:
    """HR-11's composite cross-check: the composite rule against the tensor reference at [multi].composite.bar (A9)."""
    import multi_004 as MU
    c = spec["composite"]
    return MU.composite(params, by_id[c["cell"]]["members"], panels=tuple(c["panels"]), order=int(c["order"]),
                        agree=float(c["agree"]), bar=multi_composite_bar(spec))


def phase_multi(args):
    """HR-11 (tasks T109, T126; research R27-R29): (i) the frozen reduction cases under both walls and the
    shared-stream plant; (ii) the 6 cells on the default clamp, each ONCE at the preset's frozen heterogeneous clamp
    step (amendment A1: a borrowed step, guarded; HR-11's own bar checked by the rule), judged on the dt/2 rung against
    the tensor reference, the dropped-spread plants (their predicted sizes frozen), the composite cell; (iii) the
    above-budget statement. One verdict (the default clamp's). Cap 4 CPU-h, projected from the frozen draw CPU."""
    import multi_004 as MU
    import presets_004 as PR
    from sraging import sampler_rules as R
    lang = read_toml(LANG_PATH)
    spec, judged = lang["multi"], lang.get("judged") or {}
    multi_composite_bar(spec)                                                   # refused before any CPU (A9)
    left, cap_rec = frozen_cap("multi")
    p1 = p1_steps()
    rows = MU.build_cells(spec, p1)
    cost, p_hat = p1.get("draw_cpu_s") or {}, judged.get("p_hat") or {}
    missing = sorted({r["p1_step"] for r in rows if r["p1_step"] not in cost or r["p1_step"] not in p_hat})
    if missing:
        raise Refused(f"[multi] REFUSED: no P1 draw CPU or frozen p_hat ([judged]) for {missing}")
    projected = MU.projected_cpu_h(rows, cost, spec) + multi_plant_check_cpu_h(spec)
    run, budget = phase_header("multi", args, f"{len(rows)} cells (G = {MU.G}), {len(spec['plants'])} plants, "
                               f"{len(spec['reductions'])} reduction cases x 2 walls, the composite, one above-budget "
                               f"statement", projected, left,
                               "P1's draw CPU per cell x 3, one draw per plant, the reductions at their n, the "
                               "composite's and the tensor references' frozen estimates, the plant sizes' "
                               "re-check (one tensor reference each), and one frozen draw for (iii)'s 4-parameter cell "
                               "(A7: past d <= 3, no tensor climb)")
    params = PR.preset(spec["preset"])
    key = int(spec["key"])
    k1 = rows[0]["p1_step"]
    plant_check = frozen_plants("multi", *multi_plant_checks(spec, rows))   # LANG's sizes; refused on code drift
    red = robust_run("multi", budget, ["reductions"], lambda _: (
        MU.reductions(params, spec["reductions"], n=int(spec["reduction_n"]), key=key,
                      dt=float(p1["steps"][k1]["dt"]), t_max=float(spec["reduction_t_max"])), 0.0))
    reductions = red[0] if red and isinstance(red[0], list) else red          # an error row stays a row: incomplete
    fs = p1["steps"]
    cells = robust_run("multi", budget, rows, lambda r: (MU.run_frozen(r, params, key + 100 + rows.index(r),
                                                                       fs[r["p1_step"]]), 0.0),
                       label=lambda r: r["id"], error_row=lambda r, e: {"id": r["id"], "n": r["n"], "error": e})
    plants, comp, above, ext = [], None, None, []
    if not budget.capped:
        by_id = {r["id"]: r for r in rows}

        def extra(item):
            what, i = item
            if what == "plant":
                pl = spec["plants"][i]
                return MU.run_plant(by_id[pl["cell"]], params, pl["drop"], key + 1000 + i,
                                    plant_check[f"{pl['cell']}/drop-{pl['drop']}"]["frozen"]), 0.0
            if what == "composite":
                return multi_composite(spec, by_id, params), 0.0
            a = spec["above_budget"]
            return MU.above_budget(params, a["members"], n=int(a["n"]), key=key + 2000,
                                   dt=float(p1["steps"][k1]["dt"]), t_max=float(a["t_max"])), 0.0
        items = [("plant", i) for i in range(len(spec["plants"]))] + [("composite", 0), ("above_budget", 0)]
        ext = robust_run("multi", budget, items, extra, label=lambda it: f"{it[0]} {it[1]}")
        plants = [r for it, r in zip(items, ext) if it[0] == "plant"]
        comp = next((r for it, r in zip(items, ext) if it[0] == "composite"), None)
        above = next((r for it, r in zip(items, ext) if it[0] == "above_budget"), None)
    v = R.hr11_verdict(reductions, cells, plants, comp, above, cpu_spent=budget.spent_cpu_h, cap=left)
    finish(run, args, {"status": "capped" if budget.capped else v.status, "verdicts": [v.as_dict()],
                       "reductions": reductions, "cells": cells, "plants": plants, "composite": comp,
                       "plant_check": plant_check, "above_budget": above, "errors": errors_of(list(reductions) + cells + ext),
                       "listed": [c for c in cells if not c.get("error") and (c["D"] is None or c["D"] > c["crit"])]},
           budget, cap_rec)


# ---------------------------------------------------------------------------------------------------- p3-cost
def phase_p3_cost(args):
    """P3 (tasks T080, T113): gen-1 against Langevin CPU at n in {2k, 20k, 100k}, eta (out of gen-1's table) and xc
    (in it) heterogeneity at sd 0.2, on P3's preset; its rows are `auto`'s boundary (T130). Cap 3 CPU-h, projected from
    P1's ns per individual-step."""
    import p3_cost_004 as P3
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    cfg = PR.role("p3")
    params = PR.preset(cfg["preset"])
    p1 = p1_steps()
    k = S4.step_key(cfg["preset"], "het", "clamp")
    st = p1["steps"][k]                                         # the frozen step; refused by name if not met
    dt, steps = float(st["dt"]), float(st["steps_per_individual"])
    projected = (sum(P3.NS) * len(P3.CONFIGS) * steps * float(p1["ns_per_individual_step"]) * 3e-9
                 + 2 * 120.0) / 3600.0                                  # 3: the coupled rung; gen-1 ~120 CPU-s each [I]
    run, budget = phase_header("p3-cost", args, f"{len(P3.CONFIGS)} configurations x n {P3.NS}", projected,
                               PILOT_CAPS["p3-cost"], f"P1's {k} steps x ns per individual-step")
    cells = [(p, n) for _, p in P3.CONFIGS for n in P3.NS]
    rows = robust_run("p3-cost", budget, cells, lambda c: (P3.measure(params, c[0], c[1], int(cfg["key"]),
                                                                      float(cfg["delta"]), dt), 0.0),
                      label=lambda c: f"{c[0]} n {c[1]}",
                      error_row=lambda c, e: {"id": f"{c[0]} n {c[1]}", "parameter": c[0], "n": c[1], "error": e})
    errs = errors_of(rows)
    status = "capped" if budget.capped else "incomplete" if errs else "written"
    finish(run, args, {"status": status, "rows": rows, "errors": errs,
                       "boundary": P3.boundary([r for r in rows if not r.get("error")]), "config": cfg, "dt": dt,
                       "step_key": k}, budget)


# ----------------------------------------------------------------------------------------------- S5 (US5)
REFLECT_CONVERGED_SE = 0.5          # P2: reflect "converged at S5's resolution" when its halving moves < 0.5 SE at R


def s5_spec(cfg: dict):
    """gen-1's CohortSpec of S5's configuration (ROLES["s5"] for P2; [s5] spec for s5-het)."""
    from sraging import generator as GEN
    return GEN.CohortSpec(**cfg["spec"])


def phase_p2_s5_power(args):
    """P2 (tasks T087, T112; gate C4, C5): S5's plant, R (any component) and KS's R, the instrument's step bias at R in
    SE units by a coupled halving (clamp; reflect when the clamp does not fit), the per-cohort CPU, the projection
    against 86 and the branch. Cap 6 CPU-h, projected from P1's ns per individual-step."""
    import p2_s5_power_004 as P2
    import presets_004 as PR
    from sraging import sampler_rules as R
    p1 = p1_steps()
    # the clamp's step now (refused by name if not met); reflect's only if gate C5's second branch needs it
    cfg = PR.resolve(PR.role("s5"), p1, steps=p1["steps"], only=("dt_clamp", "steps_per_individual"))
    spec = s5_spec(cfg)
    steps = float(cfg["steps_per_individual"])
    projected = 2 * 2 * P2.HALVING_COHORTS * spec.n * steps * float(p1["ns_per_individual_step"]) * 1e-9 / 3600.0
    run, budget = phase_header("p2-s5-power", args, f"the plant, R, and {P2.HALVING_COHORTS} coupled halvings per wall",
                               projected, PILOT_CAPS["p2-s5-power"], "P1's ns per individual-step x 2 rungs x 2 walls")
    plant = P2.plant_sd(spec)
    Rn = P2.r_needed(spec, plant)
    out = {"plant": plant, "R": Rn, "config": cfg}
    halv = robust_run("p2-s5-power", budget, ["clamp"], lambda w: (P2.coupled_halving(spec, cfg["dt_clamp"], w,
                                                                                    cfg["key"]), 0.0),
                      label=lambda w: f"halving {w}")
    errs = errors_of(halv)
    halv = [h for h in halv if not h.get("error") and h.get("mean") is not None]
    if halv:
        h = halv[0]
        p_c = float(p1["steps"]["s5|het|clamp"]["p_hat"])
        out["halving"] = {"clamp": {**h, "p_hat": p_c, "bias_factor": P2.bias_factor(p_c),
                                    "bias_in_se": P2.step_bias_in_se(h, Rn["R_score"], Rn["sd_per_replicate"], p_c),
                                    "bias_in_se_is": "an upper bound: (|mean| + 2 SE) x 1/(1 - 2^-p)"}}
        proj_clamp = Rn["R_score"] * h["cpu_per_cohort_s"] / 3600.0
        proj_reflect, conv = math.inf, False
        if proj_clamp > P2.CAP_S5_HET:
            dt_reflect = PR.resolve({"d": cfg["dt_reflect"]}, steps=p1["steps"])["d"]
            hr = robust_run("p2-s5-power", budget, ["reflect"],
                            lambda w: (P2.coupled_halving(spec, dt_reflect, w, cfg["key"]), 0.0),
                            label=lambda w: f"halving {w}")
            errs += errors_of(hr)
            hr = [h for h in hr if not h.get("error") and h.get("mean") is not None]
            if hr:
                p_r = float(p1["steps"]["s5|het|reflect"]["p_hat"])
                se = P2.step_bias_in_se(hr[0], Rn["R_score"], Rn["sd_per_replicate"], p_r)
                out["halving"]["reflect"] = {**hr[0], "p_hat": p_r, "bias_factor": P2.bias_factor(p_r),
                                             "bias_in_se": se}
                proj_reflect = Rn["R_score"] * hr[0]["cpu_per_cohort_s"] / 3600.0
                conv = max(se) < REFLECT_CONVERGED_SE
        R_cap = int(P2.CAP_S5_HET * 3600.0 / h["cpu_per_cohort_s"])
        smallest = P2.smallest_detectable(Rn["R_score"], R_cap)
        out["projection_cpu_h"] = {"clamp": proj_clamp, "reflect": proj_reflect}
        out["branch"] = R.s5_wall_branch(proj_clamp, proj_reflect, conv, cap=P2.CAP_S5_HET, smallest_detectable=smallest,
                                         plant_size=P2.PLANT_NATS_PER_1E5)
        out["smallest_detectable_nats_per_1e5"] = smallest
    errs += [{"id": f"halving {w} cohort {e['cohort']}", **e} for w, h in (out.get("halving") or {}).items()
             for e in h.get("errors", [])]
    status = "capped" if budget.capped else "incomplete" if errs or "branch" not in out else "written"
    finish(run, args, {"status": status, "errors": errs, **out}, budget)


RECHECK_RUNG_FACTOR = 3.0      # a coupled halving cohort's CPU over its dt rung's: rung 2 costs at most 2x rung 1
                               # (P2's run measured 1.84x, 685a72e), so 3 bounds it


def s5_het_projected_cpu_h(s5: dict) -> float:
    """s5-het's projection: R x [s5].cpu_per_cohort_s, plus, when [s5].recheck_at_step (A6), P2's coupled halving at
    S5's own step, HALVING_COHORTS x RECHECK_RUNG_FACTOR x the per-cohort CPU. gen-1's laws and gradients are not
    counted (about 0.4 CPU-h [I], from P2's run)."""
    import p2_s5_power_004 as P2
    cpu = float(s5["cpu_per_cohort_s"])
    recheck = P2.HALVING_COHORTS * RECHECK_RUNG_FACTOR * cpu if s5.get("recheck_at_step") else 0.0
    return (int(s5["R"]) * cpu + recheck) / 3600.0


def s5_recheck_ready(s5: dict) -> None:
    """The re-check needs P2's per-replicate SD in LANG (the fill's --s5-m writes it): refused by name otherwise."""
    if "sd_per_replicate" not in s5:
        raise Refused("[s5-het] REFUSED: [s5].recheck_at_step needs [s5].sd_per_replicate (P2's #R.sd_per_replicate): "
                      "fill LANG with tools/lang_004_fill.py --s5-m")


def s5_recheck(spec, s5: dict, dt, wall, base, d, run=None, cohorts=None) -> dict:
    """A6's re-check (prepared; LANG's [s5].recheck_at_step turns it on): P2's coupled halving at S5's own step (dt
    against dt/2, P2's cohorts from [s5].key), reusing S5.run's law and gradient; the step bias at dt as A5's upper
    bound in SE units at S5's R, from [s5].sd_per_replicate (P2's) and S5's p_hat; its CPU. `run(cells, work)` runs
    the halving as one robust cell (default: in process); an error row is returned as {"error": ...}."""
    import p2_s5_power_004 as P2
    s5_recheck_ready(s5)
    c0 = time.process_time()
    run = run or (lambda cells, work: [work(c)[0] for c in cells])
    n_c = P2.HALVING_COHORTS if cohorts is None else int(cohorts)
    rows = run(["recheck"], lambda _: (P2.coupled_halving(spec, dt, wall, int(s5["key"]), n_c, base=base, d=d), 0.0))
    h = rows[0]
    if h.get("error") or h.get("mean") is None:
        return {"error": h.get("error") or {"type": "NoHalving", "message": f"{h.get('cohorts')} cohorts finished"},
                "halving": h, "cpu_s": time.process_time() - c0}
    p = float(s5["p_hat"][wall])
    return {"halving": h, "dt": float(dt), "p_hat": p, "bias_factor": P2.bias_factor(p),
            "bias_in_se": P2.step_bias_in_se(h, int(s5["R"]), list(s5["sd_per_replicate"]), p),
            "bias_in_se_is": "an upper bound: (|mean| + 2 SE) x 1/(1 - 2^-p) (A5)", "cpu_s": time.process_time() - c0}


def phase_s5_het(args):
    """S5, heterogeneous half (tasks T088, T128; HR-7): Langevin cohorts at gen-1's configuration through gen-1's
    transform, the score (any component) and KS against gen-1's law at P2's R, replicates at error="ks" citing P2's
    step bias, the wall by gate C5's branch, the plant and its twin; OD-9's response record on a fail. Cap 86. With
    [s5].recheck_at_step (amendment A6, prepared for T119), P2's coupled halving at S5's own step runs first, reusing
    S5.run's law and gradient, and its bias is the one cited."""
    import s5_gen1_004 as S5
    from sraging import sampler_rules as R
    lang = read_toml(LANG_PATH)
    s5 = lang["s5"]
    branch = s5["branch"]
    if branch["action"] == "ask_naveh":
        raise Refused(f"[s5-het] REFUSED: gate C5's third branch, {branch['reason']}: ask Naveh before the run")
    left, cap_rec = frozen_cap("s5-het")
    spec = s5_spec(s5)
    Rr, dt, wall = int(s5["R"]), float(s5["dt"]), branch["wall"]
    recheck = bool(s5.get("recheck_at_step", False))
    if recheck:
        s5_recheck_ready(s5)                                                 # refused by name, before the run
    projected = s5_het_projected_cpu_h(s5)
    run, budget = phase_header("s5-het", args, f"R = {Rr} cohorts of {spec.n} on {wall}" +
                               (", after P2's coupled halving at this step (A6)" if recheck else ""), projected, left,
                               "P2's per-cohort CPU x R" + (" + the re-check's halving" if recheck else ""))

    def reps(R_):
        return robust_run("s5-het", budget, list(range(R_)), lambda i: (S5.replicate(spec, int(s5["key"]) + i, dt,
                                                                                     wall), 0.0),
                          label=lambda i: f"replicate {i}")
    base = d_true = rc = None
    if recheck:
        base = S5.law(spec)
        S5.horizon(spec)                                     # while gen-1's cache still holds the spec's curves
        d_true = S5.spec_gradient(spec, base)
        rc = s5_recheck(spec, s5, dt, wall, base, d_true,
                        run=lambda cells, work: robust_run("s5-het", budget, cells, work,
                                                           label=lambda c: f"re-check halving {wall} at dt {dt:g}"))
    res = S5.run(spec, Rr, dt, wall, int(s5["key"]), float(s5["plant_sd"]), work=reps, base=base, d_true=d_true)
    rc_error = None
    if rc is not None and rc.get("error"):
        rc_error = f"{rc['error'].get('type')}: {rc['error'].get('message')}"
    v = R.hr7_verdict({"plant": {"z": res["plant"]["z"]}, "twin": {"z": res["twin"]["z"]}}, branch=branch["branch"],
                      smallest_detectable=s5.get("smallest_detectable_nats_per_1e5"), n_errors=len(res["errors"]),
                      recheck_error=rc_error)
    status = "capped" if budget.capped else v.status
    cited = rc["bias_in_se"] if rc is not None and not rc.get("error") else s5.get("step_bias_in_se")
    finish(run, args, {"status": status, "verdicts": [v.as_dict()], "s5": res, "branch": branch,
                       "step_bias_cited": cited, "recheck": rc,
                       "od9_response": S5.od9_response(spec, status) if status == "fail" else None}, budget, cap_rec)


# spec 002's score runs, in order: a later run's feature replaces an earlier one's (002 re-ran its amended window, and ran
# heterogeneity on its own)
S5_HOM_002_RUNS = ("score_002.json", "score_window_002.json", "score_het_002.json")


def s5_hom_projected_cpu_h(*runs: dict) -> float:
    """s5-hom's projection from spec 002's runs (S5_HOM_002_RUNS, in order): each run's parent CPU plus each feature's
    CPU from the last run that has it. A run's top-level cpu_s is its parent's alone (~1 s); its features' workers spent
    the rest: score_002.json's four ~2,200 CPU-s, score_het_002.json's heterogeneity 15,911. From score_002.json alone
    the projection read 0.61 CPU-h, and the run at 252ad3a went on to ~5 (Naveh 2026-11-04: let it finish)."""
    feats = {}
    for r in runs:
        feats.update({k: float(f.get("cpu_s") or 0.0) for k, f in (r.get("features") or {}).items()
                      if isinstance(f, dict)})
    return (sum(float(r.get("cpu_s") or 0.0) for r in runs) + sum(feats.values())) / 3600.0


def phase_s5_hom(args):
    """S5, homogeneous half (tasks T089, T127): spec 002's score harness (tools/score_test.py, untouched) with its
    simulator replaced by sraging's bridge kernel at the default wall; 002's verdict rules. Cap 1.5, projected from
    002's runs (4.8 CPU-h: above the cap, so a re-run needs --naveh-ok)."""
    import score_004 as SC
    import score_test as ST
    left, cap_rec = frozen_cap("s5-hom")
    runs = [json.load(open(os.path.join(REPO, RESULTS_REL, f))) for f in S5_HOM_002_RUNS]
    projected = s5_hom_projected_cpu_h(*runs)
    run, budget = phase_header("s5-hom", args, f"002's {len(ST.FEATURES)} features on sraging's kernel", projected, left,
                               "the CPU of 002's runs, " + ", ".join(S5_HOM_002_RUNS) + ", each feature's latest")
    zc, n_pairs = SC.zcrit()
    feats = run_capped(budget, list(ST.FEATURES), lambda f: ((f, SC.run_feature(f, zc)), 0.0))
    features = dict(feats)
    missing = [f for f in ST.FEATURES if f not in features]
    status = "capped" if budget.capped else SC.summarise(features)
    finish(run, args, {"status": status, "features": features, "features_not_run": missing, "zcrit": zc,
                       "n_pairs": n_pairs, "theta0_cohort": ST.THETA0_COHORT}, budget, cap_rec)


# ------------------------------------------------------------------------------------- the close (contracts §5)
#: the close phases' caps, CPU-h (tasks.md T135, T139): they run under no freeze file
CLOSE_CAPS = {"docs": 0.1, "unchanged": 2.0}


def _with_children_cpu(work):
    """(work(), the CPU its child processes spent): RUSAGE_CHILDREN's user + system time across the call."""
    import resource
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    out = work()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return out, (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)


# ------------------------------------------------------------------------------------------------------- docs
DOCS_DIR = "docs/algorithms"
DOCS_SECTIONS = ("docs/algorithms/sections/heterogeneity.tex", "docs/algorithms/sections/langevin.tex")
LATEX_CMD = ("pdflatex", "-interaction=nonstopmode", "-halt-on-error")      # the latex-pdf skill's compile
DOCS_PROJECTED_CPU_S = 5.0          # [M] the two passes took 1.4 CPU-s on this Mac (2026-11-05), x 3.5 for headroom
VERIFY_TEX = re.compile(r"\\verify\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}")


def docs_markers(paths, base=REPO) -> list:
    """Every \verify{json}{path}{spec}{text} marker (docs/algorithms/macros.tex) in `paths`, in order: {file, json,
    path, spec, text, recomputed, match}, recomputed as format(value, spec) (str(value) for an empty spec) with
    tools/verify_readme.py's getter. A marker whose value cannot be read has recomputed None and an error; a
    \verify{ that is not four brace groups is a row of its own with match False."""
    from verify_readme import get
    rows = []
    for f in paths:
        text = open(os.path.join(base, f)).read()
        found = list(VERIFY_TEX.finditer(text))
        for m in found:
            jp, key, spec, shown = m.groups()
            row = {"file": f, "json": jp, "path": key, "spec": spec, "text": shown.strip()}
            try:
                value = get(json.load(open(os.path.join(base, jp))), key)
                row["recomputed"] = format(value, spec) if spec else str(value)
            except (OSError, KeyError, IndexError, ValueError, TypeError) as ex:
                row["recomputed"], row["error"] = None, f"{type(ex).__name__}: {ex}"
            row["match"] = row["recomputed"] == row["text"]
            rows.append(row)
        rows += [{"file": f, "text": None, "recomputed": None, "match": False,
                  "error": "a \\verify{ that is not four brace groups"}
                 for _ in range(text.count("\\verify{") - len(found))]
    return rows


def docs_build(out_dir, *, cwd=DOCS_DIR, cmd=LATEX_CMD) -> dict:
    """The latex-pdf skill's compile of main.tex, twice (references and contents resolve), into `out_dir`, so the
    committed main.pdf is not rewritten: {ok, returncodes, pages, pdf_sha256, log_tail}."""
    where = os.path.join(REPO, cwd)
    rcs, last = [], None
    for _ in range(2):
        last = subprocess.run([*cmd, f"-output-directory={out_dir}", "main.tex"], cwd=where, capture_output=True,
                              text=True, errors="replace")
        rcs.append(last.returncode)
        if last.returncode:
            break
    pdf, log = os.path.join(out_dir, "main.pdf"), os.path.join(out_dir, "main.log")
    ok = rcs == [0, 0] and os.path.exists(pdf)
    m = re.search(r"Output written on .*?\((\d+) pages?", open(log, errors="replace").read(), re.S) \
        if os.path.exists(log) else None
    return {"ok": ok, "returncodes": rcs, "pages": int(m.group(1)) if m else None,
            "pdf_sha256": hashlib.sha256(open(pdf, "rb").read()).hexdigest() if ok else None,
            "log_tail": "" if ok else (last.stdout + last.stderr)[-2000:]}


def docs_verdict(build, markers) -> dict:
    """pass iff the PDF built and every marker recomputes to its text; no marker at all is not a pass."""
    bad = [m for m in markers if not m["match"]]
    reasons = ([] if build.get("ok") else ["the PDF did not build"]) + \
        ([] if markers else ["no verify marker found"]) + \
        ([f"{len(bad)} marker(s) do not recompute to their text"] if bad else [])
    return {"status": "fail" if reasons else "pass", "reasons": reasons, "n_markers": len(markers),
            "n_mismatch": len(bad)}


def phase_docs(args):
    """T135 (contracts/acceptance.md §5): docs/algorithms built to PDF with the latex-pdf skill's compile (into a
    scratch directory: the committed main.pdf is not rewritten), and every \verify marker of the heterogeneity and
    Langevin sections recomputed from its stored result. pass iff the build succeeds and every marker matches. Cap 0.1
    CPU-h, projected as the LaTeX build's CPU."""
    import tempfile
    run, budget = phase_header("docs", args, "two pdflatex passes and the sections' verify markers; seconds",
                               DOCS_PROJECTED_CPU_S / 3600.0, CLOSE_CAPS["docs"],
                               "the LaTeX build's CPU, two passes [M 2026-11-05] with headroom")
    with tempfile.TemporaryDirectory(prefix="sraging004_docs_") as tmp:
        rows = run_capped(budget, ["build"], lambda _: _with_children_cpu(lambda: docs_build(tmp)))
    build = rows[0] if rows else {"ok": False, "returncodes": [], "pages": None, "pdf_sha256": None,
                                  "log_tail": "not built: the cap was reached"}
    markers = docs_markers(DOCS_SECTIONS)
    v = docs_verdict(build, markers)
    finish(run, args, {"status": "capped" if budget.capped else v["status"], "verdict": v, "build": build,
                       "sections": list(DOCS_SECTIONS), "markers": markers}, budget)


# -------------------------------------------------------------------------------------------------- unchanged
#: every earlier spec's results test (SC-010), 003g's under its CLOSED_AT (4e17529), as 003g's own T057 listed them
UNCHANGED_RESULTS_004 = ["tests/acceptance/test_results.py", "tests/acceptance/test_results_002.py",
                         "tests/acceptance/test_results_003.py", "tests/acceptance/test_results_003b.py",
                         "tests/acceptance/test_results_003e.py", "tests/acceptance/test_results_003f.py",
                         "tests/acceptance/test_results_003g.py"]
UNCHANGED_GROUPS_004 = {"unit": ["tests/unit"], "parity": ["tests/parity"],
                        "results_001_to_003g": UNCHANGED_RESULTS_004}
#: §5's named checks, each a test module inside a group: it must have run (at least one test) with no failure, error
#: or skip
UNCHANGED_NAMED = {"parity_t1_normal_frac_sd_v033": ("parity", "tests.parity.test_engine_t1_het"),
                   "generator_checksums": ("unit", "tests.unit.test_generator_checksums"),
                   "register": ("unit", "tests.unit.test_register")}
UNCHANGED_PROJECTED_FROM = "tests/acceptance/results/unchanged_003g.json"


def unchanged_group(name, paths, tmp) -> dict:
    """One group by pytest in its own process at the committed results, read from its junit XML (003g's T057): the
    counts, the exit, the summary line, the skipped and failed tests, and every test case's kind."""
    import xml.etree.ElementTree as ET
    xml = os.path.join(tmp, f"{name}.xml")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *paths],
                       capture_output=True, text=True, cwd=REPO)
    root = ET.parse(xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    c = {k: sum(int(x.get(k, 0)) for x in suites) for k in ("tests", "failures", "errors", "skipped")}

    def listed(tag):
        return [(f"{tc.get('classname')}::{tc.get('name')}", (el.get("message") or "")[:300])
                for tc in root.iter("testcase") for el in tc.findall(tag)]
    cases = [{"classname": tc.get("classname"), "name": tc.get("name"),
              "kind": next((k for k in ("failure", "error", "skipped") if tc.find(k) is not None), "ok")}
             for tc in root.iter("testcase")]
    return {**c, "passed": c["tests"] - c["failures"] - c["errors"] - c["skipped"], "returncode": p.returncode,
            "summary": (p.stdout.strip().split("\n") or [""])[-1], "wall_s": round(time.time() - t0, 1),
            "paths": list(paths), "skipped_list": listed("skipped"), "failed_list": listed("failure") + listed("error"),
            "cases": cases}


def unchanged_verdict_004(groups) -> dict:
    """PASS iff every group passes 003e's rule as 003g used it (0 failures and 0 errors; every skip named on its
    SUPERSEDED or RECORDED_SKIPS lists) and every named check ran with no failure, error or skip."""
    import acceptance_003e as E
    groups_ok, unlisted = E.unchanged_verdict(groups)
    named = {}
    for check, (g, module) in UNCHANGED_NAMED.items():
        cases = [c for c in (groups.get(g) or {}).get("cases", [])
                 if c["classname"] == module or str(c["classname"]).startswith(module + ".")]
        kinds = sorted({c["kind"] for c in cases})
        named[check] = {"group": g, "module": module, "n": len(cases), "kinds": kinds,
                        "ok": bool(cases) and kinds == ["ok"]}
    ok = groups_ok and all(v["ok"] for v in named.values())
    return {"status": "PASS" if ok else "FAIL", "groups_ok": groups_ok, "skips_unlisted": unlisted, "named": named}


def phase_unchanged(args):
    """T139 (contracts/acceptance.md §5; SC-010), as 003e's and 003g's: at the final code, the unit suite, the parity
    suite (spec 002's T1 for normal_frac_sd, v033, named) and every earlier spec's results test (003g's under
    CLOSED_AT = 4e17529), each by pytest in its own process; the generator checksums and the register named within
    the unit group. PASS iff all hold. Cap 2 CPU-h, projected from 003g's unchanged record; the pytest processes' CPU
    is counted (RUSAGE_CHILDREN)."""
    import tempfile
    import acceptance_003e as E
    prior = json.load(open(os.path.join(REPO, UNCHANGED_PROJECTED_FROM)))
    projected = sum(float(g["wall_s"]) for g in prior["groups"].values()) / 3600.0
    run, budget = phase_header("unchanged", args, "pytest in 3 processes: the unit suite (~17 min), the parity suite "
                               "and the 001-003g results tests; 0 calibrations", projected, CLOSE_CAPS["unchanged"],
                               "003g's unchanged per-group wall (its record's cpu_s counts the parent only) [I]")
    with tempfile.TemporaryDirectory(prefix="sraging004_unchanged_") as tmp:
        rows = run_capped(budget, list(UNCHANGED_GROUPS_004), lambda g: _with_children_cpu(
            lambda: {"group": g, **unchanged_group(g, UNCHANGED_GROUPS_004[g], tmp)}))
    groups = {r.pop("group"): r for r in rows}
    v = unchanged_verdict_004(groups) if not budget.capped else {"status": "capped", "groups_ok": None,
                                                                 "skips_unlisted": [], "named": {}}
    for g in groups.values():
        g.pop("cases", None)                                    # the result keeps the counts and the lists, as 003g's
    finish(run, args, {"status": v["status"], "verdict": v, "groups": groups, "superseded": E.SUPERSEDED,
                       "recorded_skips": E.RECORDED_SKIPS}, budget)


PHASES = {name: _todo(name) for name in PHASE_NAMES}
PHASES.update({"p5-inventory": phase_p5_inventory, "hr2-noise": phase_hr2_noise, "panel": phase_panel,
               "reference": phase_reference, "hr1": phase_hr1, "hr2": phase_hr2, "d401": phase_d401,
               "p0-het-cost": phase_p0_het_cost, "hr3-pilot": phase_hr3_pilot, "hr3": phase_hr3, "hr4": phase_hr4,
               "hr5": phase_hr5, "s4": phase_s4, "p3-cost": phase_p3_cost, "p2-s5-power": phase_p2_s5_power,
               "s5-het": phase_s5_het, "s5-hom": phase_s5_hom, "paths": phase_paths,
               "multi": phase_multi, "p1-langevin": phase_p1_langevin, "p4-jax": phase_p4_jax,
               "plants": phase_plants, "hr8": phase_hr8, "p1-pooled": phase_p1_pooled,
               "p1-pooled-b": phase_p1_pooled_b, "docs": phase_docs, "unchanged": phase_unchanged})


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=PHASE_NAMES)
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--naveh-ok", help="a pilot only (SR 4): Naveh's OK lifting its cap, as '<date>: <his words>'")
    ap.add_argument("--naveh-ok-cap", type=float, help="a pilot only: the cap in CPU-h that --naveh-ok sets")
    ap.add_argument("--workers", type=int, help=f"worker processes, for wall time only ({', '.join(WORKER_PHASES)}): "
                                                "the rows are the same at any W")
    return ap


def check_workers(args) -> None:
    """--workers only where a phase uses it (WORKER_PHASES), as an integer >= 1: refused by name otherwise."""
    W = getattr(args, "workers", None)
    if W is None:
        return
    if args.phase not in WORKER_PHASES:
        raise Refused(f"[{args.phase}] REFUSED: --workers is taken by {', '.join(WORKER_PHASES)} only")
    if int(W) < 1:
        raise Refused(f"[{args.phase}] REFUSED: --workers {W}: an integer >= 1")


def main(argv=None):
    args = parser().parse_args(argv)
    check_workers(args)
    naveh_oks(args.phase, args)                 # refuses an OK the command line cannot carry, before anything runs
    require_prerequisites(args.phase)
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
