"""Check every number in the paper against its stored source; write paper/claims.tex. Exits 1 on any failure.

    python -B verify_claims.py [--write-only]

1. CLAIMS. Each [[claim]] in claims.toml is re-extracted from its `source` with its `locator` and compared with
   `value` (numbers within `rtol`, strings exactly). A changed source sha256 is a WARNING (the value check decides).
2. MACROS. paper/claims.tex defines \\C{<id>} for every claim (display text, else the value).
3. USAGE. Every \\C{id} in paper/**/*.tex names a known claim; claims that no .tex file uses are listed (info).
4. MISSING. Every \\MISSING{...} box in the .tex files has exactly one entry in sources/MISSING.md (matched by the
   first argument's leading words, or by an explicit `% MISSING-ID: <id>` comment on the line before the box).

Formats are in sources/CLAIMS_FORMAT.md.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_paths as PP  # noqa: E402  (where the sources are: on the author's machine, or in the release)

PAPER = os.path.join(HERE, "paper")
FAIL: list[str] = []
WARN: list[str] = []
NOT_PUBLIC: list[str] = []   # release only: claims whose source is not part of the release


def resolve(p: str) -> str:
    return PP.path(p)


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_path(obj, path: str):
    """a.b[3]['key/with.dots'].c"""
    for tok in re.findall(r"\[\'[^\']*\'\]|\[\"[^\"]*\"\]|\[\d+\]|[^.\[\]]+", path):
        if tok.startswith("['") or tok.startswith('["'):
            obj = obj[tok[2:-2]]
        elif tok.startswith("["):
            obj = obj[int(tok[1:-1])]
        else:
            obj = obj[tok]
    return obj


def read_rows(path: str):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def match(row, where: dict) -> bool:
    return all(str(row.get(k)) == str(v) for k, v in where.items())


def extract(loc: dict, path: str):
    got = _extract(loc, path)
    return -float(got) if loc.get("negate") else got


def _regex(pattern: str, text: str, group=1):
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        raise ValueError(f"pattern not found: {pattern!r}")
    return m.group(group)


def _extract(loc: dict, path: str):
    kind = loc["kind"]
    if kind == "git":
        # { kind = "git", repo = "~/code/x", rev = "<commit>", path = "<file in repo>", inner = <locator> }
        blob = PP.git_show(loc["repo"], loc["rev"], loc["path"])
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(loc["path"])[1]) as tmp:
            tmp.write(blob)
            tmp.flush()
            return _extract(loc["inner"], tmp.name)
    if kind in ("regex", "regex-ratio"):
        with open(path, errors="replace") as f:
            text = f.read()
        if kind == "regex":
            return _regex(loc["pattern"], text, loc.get("group", 1))
        return float(_regex(loc["num"], text)) / float(_regex(loc["den"], text))
    if kind == "json":
        with open(path) as f:
            return json_path(json.load(f), loc["path"])
    if kind == "csv":
        rows = [r for r in read_rows(path) if match(r, loc.get("where", {}))]
        if len(rows) != 1:
            raise ValueError(f"csv where={loc.get('where')} matched {len(rows)} rows (need 1)")
        return rows[0][loc["column"]]
    if kind == "count":
        return sum(1 for r in read_rows(path) if match(r, loc.get("where", {})))
    if kind == "ratio":
        num = float(extract(loc["num"], resolve(loc["num"].get("source", path))))
        den = float(extract(loc["den"], resolve(loc["den"].get("source", path))))
        return num / den
    if kind == "python":
        with open(path, errors="replace") as f:
            raw = f.read()
        try:
            src = json.loads(raw)
        except ValueError:
            src = raw
        env = {"math": math, "re": re, "json": json, "csv": csv, "io": io, "statistics": statistics}
        return eval(loc["expr"], env, {"src": src})  # noqa: S307
    raise ValueError(f"unknown locator kind {kind!r}")


def same(expected, got, rtol: float) -> bool:
    if isinstance(expected, str):
        return str(got).strip() == expected.strip()
    try:
        g = float(str(got).replace(",", "").rstrip("%"))
    except ValueError:
        return False
    e = float(expected)
    return g == e or abs(g - e) <= rtol * max(abs(e), abs(g))


def tex_files():
    for root, _, files in os.walk(PAPER):
        for fn in files:
            if fn.endswith(".tex") and fn != "claims.tex":
                yield os.path.join(root, fn)


def check_claims(claims) -> None:
    seen = set()
    for c in claims:
        cid = c.get("id", "?")
        if cid in seen:
            FAIL.append(f"{cid}: duplicate id")
        seen.add(cid)
        try:
            if c["locator"]["kind"] == "git":
                path = c["locator"]["repo"]
                PP.git_show(path, c["locator"]["rev"], c["locator"]["path"])
            else:
                path = resolve(c["source"])
                for sub in ("num", "den"):
                    if sub in c["locator"] and c["locator"][sub].get("source"):
                        resolve(c["locator"][sub]["source"])
        except PP.NotPublic as exc:
            NOT_PUBLIC.append(f"{cid}: {exc}")
            continue
        if c["locator"]["kind"] != "git" and not os.path.isfile(path):
            FAIL.append(f"{cid}: source missing: {c['source']}")
            continue
        if c.get("sha256") and c["locator"]["kind"] != "git" and os.path.isfile(path) and sha(path) != c["sha256"]:
            WARN.append(f"{cid}: source changed since recorded (sha256): {c['source']}")
        try:
            got = extract(c["locator"], path)
        except Exception as exc:  # noqa: BLE001 - report every locator failure, keep going
            FAIL.append(f"{cid}: locator failed: {exc}")
            continue
        if not same(c["value"], got, float(c.get("rtol", 0.005))):
            FAIL.append(f"{cid}: paper says {c['value']!r}, source gives {got!r}")


def write_macros(claims) -> None:
    lines = ["% GENERATED by verify_claims.py from claims.toml - do not edit."]
    for c in claims:
        text = c.get("display", str(c["value"]))
        lines.append(f"\\expandafter\\def\\csname claim@{c['id']}\\endcsname{{{text}}}")
    with open(os.path.join(PAPER, "claims.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def check_usage(claims) -> list[str]:
    ids = {c["id"] for c in claims}
    used = set()
    for p in tex_files():
        with open(p) as f:
            text = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in f.read().splitlines()
                             if "\\newcommand" not in ln)
            for m in re.finditer(r"\\C\{([^}]+)\}", text):
                used.add(m.group(1))
                if m.group(1) not in ids:
                    FAIL.append(f"{os.path.relpath(p, HERE)}: \\C{{{m.group(1)}}} is not in claims.toml")
    return sorted(ids - used)


def check_missing() -> int:
    if PP.RELEASE:          # sources/MISSING.md is the authors' working list; it is not released
        return 0
    boxes = []
    for p in tex_files():
        with open(p) as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if "\\MISSING{" in line and not line.lstrip().startswith("%") and "\\newcommand" not in line:
                m = re.match(r"\s*%\s*MISSING-ID:\s*(\S+)", lines[i - 1]) if i else None
                boxes.append((m.group(1) if m else None, os.path.relpath(p, HERE), i + 1))
    md_path = os.path.join(HERE, "sources", "MISSING.md")
    md = open(md_path).read() if os.path.isfile(md_path) else ""
    md_ids = re.findall(r"^##+\s*\[?([A-Za-z0-9_.-]+)\]?", md, re.MULTILINE)
    for bid, rel, ln in boxes:
        if bid is None:
            FAIL.append(f"{rel}:{ln}: \\MISSING box without a '% MISSING-ID: <id>' line above it")
        elif md_ids.count(bid) != 1:
            FAIL.append(f"{rel}:{ln}: MISSING-ID {bid} appears {md_ids.count(bid)} times in sources/MISSING.md")
    box_ids = {b[0] for b in boxes if b[0]}
    for mid in md_ids:
        if mid not in box_ids:
            WARN.append(f"sources/MISSING.md entry {mid} has no \\MISSING box in the paper")
    return len(boxes)


def main() -> int:
    if "--check" in sys.argv:  # check a proposal file only: no macros, usage or MISSING checks
        with open(sys.argv[sys.argv.index("--check") + 1], "rb") as f:
            claims = tomllib.load(f).get("claim", [])
        check_claims(claims)
        for x in WARN + FAIL:
            print("WARN " if x in WARN else "FAIL ", x)
        print(f"checked {len(claims)} claims: {len(FAIL)} failure(s), {len(WARN)} warning(s)")
        return 1 if FAIL else 0
    with open(os.path.join(HERE, "claims.toml"), "rb") as f:
        claims = tomllib.load(f).get("claim", [])
    write_macros(claims)
    if "--write-only" in sys.argv:
        print(f"wrote paper/claims.tex ({len(claims)} claims)")
        return 0
    check_claims(claims)
    unused = check_usage(claims)
    n_boxes = check_missing()
    for w in WARN:
        print("WARN ", w)
    for x in FAIL:
        print("FAIL ", x)
    if PP.RELEASE:
        used = len(claims) - len(unused)
        np_used = [x for x in NOT_PUBLIC if x.split(":", 1)[0] not in set(unused)]
        print(f"release: {len(NOT_PUBLIC)} claim(s) ({len(np_used)} used in the paper) have a source that is not part of "
              f"this release; they were checked on the author's machine (recorded sha256 in claims.toml):")
        for x in sorted(np_used):
            print("  not public ", x)
    print(f"claims: {len(claims)} ({len(unused)} not used in the paper); MISSING boxes: {n_boxes}; "
          f"{len(FAIL)} failure(s), {len(WARN)} warning(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
