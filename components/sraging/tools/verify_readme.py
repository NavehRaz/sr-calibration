"""Recompute every marked number in README.md from stored results; exit non-zero on any mismatch (T058).

Constitution II: prose numbers are regenerated from stored output and regression-tested. Two markers:

  inline  <!-- verify:examples/results/x.json:loglik:.6g -->-1234.57<!-- /verify -->
          the text between the markers must equal format(value at the dotted JSON path, spec)
          (path elements: dict keys, or [i] list indices, e.g. claims.eta.upper, curves.S[2])

  block   <!-- verify-output:examples/results/x.json -->
          ```text
          ...the `sraging fit-cohort` printout...
          ```
          the fenced block must equal the CLI's rendering of the stored FitResult, line for line

    .venv/bin/python tools/verify_readme.py [README.md]
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INLINE = re.compile(r"<!--\s*verify:([^:\s]+):([^:\s]+):([^\s]*)\s*-->(.*?)<!--\s*/verify\s*-->", re.S)
BLOCK = re.compile(r"<!--\s*verify-output:(\S+)\s*-->\s*\n```[a-z]*\n(.*?)\n```", re.S)


def get(obj, path):
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        obj = obj[int(part[1:-1])] if part.startswith("[") else obj[part]
    return obj


def render(result_path):
    sys.path.insert(0, os.path.join(REPO, "src"))
    from sraging import cli
    from sraging.result import FitResult
    r = FitResult.from_json(os.path.join(REPO, result_path))
    buf = io.StringIO()
    cli._print_result(r, None, buf)
    return buf.getvalue().rstrip("\n")


def verify(readme_text, base=REPO) -> list[str]:
    problems, n = [], 0
    for m in INLINE.finditer(readme_text):
        n += 1
        path, key, spec, text = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        try:
            value = get(json.load(open(os.path.join(base, path))), key)
            want = format(value, spec) if spec else str(value)
        except (OSError, KeyError, IndexError, ValueError, TypeError) as ex:
            problems.append(f"{path}:{key}: cannot recompute ({type(ex).__name__}: {ex})")
            continue
        if text != want:
            problems.append(f"{path}:{key}: README says {text!r}, stored output gives {want!r}")
    for m in BLOCK.finditer(readme_text):
        n += 1
        path, block = m.group(1), m.group(2).rstrip("\n")
        try:
            want = render(path)
        except (OSError, ValueError, KeyError) as ex:
            problems.append(f"{path}: cannot render ({type(ex).__name__}: {ex})")
            continue
        if block != want:
            a, b = block.splitlines(), want.splitlines()
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            problems.append(f"{path}: example output differs from the stored result at line {first + 1}")
    if n == 0:
        problems.append("no verify markers found: every number in the README must be marked")
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    readme = argv[0] if argv else os.path.join(REPO, "README.md")
    problems = verify(open(readme).read())
    for p in problems:
        print(f"MISMATCH {p}")
    print(f"verify_readme: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
