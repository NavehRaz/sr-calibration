#!/bin/sh
# Build main.pdf and si.pdf in a scratch copy (keeps LaTeX by-products out of Dropbox), then copy the PDFs back.
# Usage: sh build.sh [outdir]   (default: paper/)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/SRvenv/bin/python}"; [ -x "$PY" ] || PY=python3      # set PY to the environment to use
"$PY" -B "$HERE/verify_claims.py" --write-only
B="$(mktemp -d)"
cp -R "$HERE/paper" "$HERE/figures" "$HERE/tables" "$B/"
cd "$B/paper"
run() { pdflatex -interaction=nonstopmode "$1.tex" >/dev/null 2>&1 || true; }
run main; run si; bibtex main >/dev/null 2>&1 || true; bibtex si >/dev/null 2>&1 || true
run main; run si; run main; run si
for t in main si; do
  echo "$t: $(grep -c '^!' $t.log) error(s); unresolved: $(grep 'undefined' $t.log | grep -o "[\`'][^']*'" | sort -u | tr '\n' ' ')"
done
OUT="${1:-$HERE/paper}"
cp main.pdf si.pdf "$OUT/"
echo "PDFs in $OUT (build dir $B)"
