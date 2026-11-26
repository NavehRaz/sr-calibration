#!/usr/bin/env bash
# Spec 003's coverage fits on WEXAC (Naveh 2026-10-08, tests/acceptance/decisions_003.toml [coverage_host]).
# Run from the sraging checkout on the Mac; everything remote lives under ~/sraging_cov003 on the cluster (no other
# directory there is touched). The fits run at the Mac's HEAD, moved as a git bundle (nothing is pushed anywhere);
# the Mac then imports the rows and scores them at the same commit:
#
#   tools/wexac/coverage_003.sh push           bundle HEAD (clean tree), copy it, the library v2 data and the
#                                              requirements; check out HEAD there; build the venv once
#   tools/wexac/coverage_003.sh check          on the login node, seconds: imports, pins, the library loads
#   tools/wexac/coverage_003.sh jobscript      write and show the array element's script (submits nothing)
#   tools/wexac/coverage_003.sh smoke --yes    element 1 alone, a smoke whose row is discarded (Naveh's yes)
#   tools/wexac/coverage_003.sh submit --yes   the LSF array, one fit per element (only with Naveh's yes)
#   tools/wexac/coverage_003.sh status         the array's job states and the rows written so far
#   tools/wexac/coverage_003.sh fetch          copy the rows to .cache/out/wexac_rows/<sha>/
#
# then on the Mac, at the same HEAD:
#   python tools/acceptance_003_coverage.py import-rows .cache/out/wexac_rows/<sha>/*.jsonl
#   python tools/acceptance_003.py coverage --score-only
#   python tools/acceptance_003.py coverage-xhost --workers 8     (the Mac subset, reported, not gated)
set -euo pipefail

HOSTALIAS=wexac
RDIR=sraging_cov003                                      # relative to the cluster home
QUEUE=long                                               # RUNLIMIT 10080 min; short's 1440 is too near the slowest fits
WLIMIT=10080
SELECT='select[model==AMD_EPYC]'                         # one CPU family for every fit (Naveh 2026-10-09)
MEM_MB=4096                                              # jax2 fits peaked at 1038 MB (MEM_MB_C2); walks add
PY=3.14.6
NJOBS=215                                                # panel_003.toml [coverage]: 160 + 55
REPO=$(git rev-parse --show-toplevel)
VPY="$REPO/.venv/bin/python"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 "$HOSTALIAS")

sha=$(git -C "$REPO" rev-parse HEAD)
short=${sha:0:7}

remote() { "${SSH[@]}" "bash -lc $(printf '%q' "$1")"; }

case "${1:-}" in
push)
    if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
        echo "REFUSED: the tracked tree is dirty; commit first" >&2; exit 2
    fi
    n=$("$VPY" -c "import tomllib;d=tomllib.load(open('$REPO/tests/acceptance/panel_003.toml','rb'))['coverage'];print(len(d['profile_jax2'])+len(d['profile_jax2_h']))")
    [ "$n" = "$NJOBS" ] || { echo "REFUSED: the panel has $n jobs, this script submits $NJOBS" >&2; exit 2; }
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/cov003.XXXXXX")
    git -C "$REPO" bundle create "$tmp/$short.bundle" HEAD
    j2=$(cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 "$VPY" -c "import sys;sys.path.insert(0,'tools');import oracle_jax2 as OJ;print(OJ.LIBRARY_V2)")
    remote "mkdir -p ~/$RDIR/bundles ~/$RDIR/j2/data/library ~/$RDIR/rows/$sha ~/$RDIR/logs/$sha"
    scp -q -o BatchMode=yes "$tmp/$short.bundle" "$REPO/tools/wexac/requirements_003.txt" "$HOSTALIAS:$RDIR/bundles/"
    rsync -a --delete -e "ssh -o BatchMode=yes" "$j2/" "$HOSTALIAS:$RDIR/j2/data/library/v2/"
    rm -rf "${tmp:?}"
    remote "set -e; cd ~/$RDIR
        if [ ! -d repo/.git ]; then git clone -q --no-checkout bundles/$short.bundle repo; fi
        git -C repo fetch -q ../bundles/$short.bundle HEAD
        git -C repo checkout -q --detach $sha
        test \"\$(git -C repo rev-parse HEAD)\" = $sha
        if [ ! -x venv/bin/python ]; then uv venv -q -p $PY venv; fi
        uv pip install -q --python venv/bin/python -r bundles/requirements_003.txt
        echo pushed $short; venv/bin/python --version"
    ;;
check)
    remote "set -e; cd ~/$RDIR/repo; test \"\$(git rev-parse HEAD)\" = $sha
        SRAGING_J2_ROOT=~/$RDIR/j2 PYTHONDONTWRITEBYTECODE=1 ~/$RDIR/venv/bin/python -c '
import sys; sys.path[:0] = [\"tools\", \"src\"]
import acceptance_003_coverage as CV
CV._init()
import sraging, jax, numpy, scipy
lib = CV.library()
jobs = CV.panel_jobs()
print(\"sraging from\", sraging.__file__); print(\"jax\", jax.__version__, \"numpy\", numpy.__version__, \"scipy\", scipy.__version__)
print(\"jobs\", len(jobs), \"first\", jobs[0], lib.spec(jobs[0][0]).arm); print(\"host\", CV.host())'"
    ;;
jobscript|submit|smoke)
    if [ "$1" != jobscript ] && [ "${2:-}" != "--yes" ]; then
        echo "REFUSED: submitting needs --yes (Naveh's go; the full array only after walker step 2 passes)" >&2; exit 2
    fi
    remote "set -e; cd ~/$RDIR; test \"\$(git -C repo rev-parse HEAD)\" = $sha
        cat > logs/$sha/job.sh <<EOF
#!/bin/bash
export SRAGING_J2_ROOT=\$HOME/$RDIR/j2 PYTHONDONTWRITEBYTECODE=1 PJRT_NPROC=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd \$HOME/$RDIR/repo
exec \$HOME/$RDIR/venv/bin/python tools/acceptance_003_coverage.py shard --index \\\$LSB_JOBINDEX --of $NJOBS \
    --rows-out \$HOME/$RDIR/rows/$sha/shard_\\\$LSB_JOBINDEX.jsonl
EOF
        chmod +x logs/$sha/job.sh; cat logs/$sha/job.sh"
    if [ "$1" != jobscript ]; then
        if [ "$1" = smoke ]; then range=1; else range=1-$NJOBS; fi
        remote "cd ~/$RDIR && bsub -q $QUEUE -W $WLIMIT -n 1 -R '$SELECT rusage[mem=$MEM_MB] span[hosts=1]' \
            -J 'cov003_$short[$range]' -o logs/$sha/%I.out -e logs/$sha/%I.err logs/$sha/job.sh"
    fi
    ;;
status)
    remote "bjobs -a -J 'cov003_$short' 2>/dev/null | awk 'NR>1{print \$3}' | sort | uniq -c
        echo rows: \$(cat ~/$RDIR/rows/$sha/*.jsonl 2>/dev/null | wc -l) of $NJOBS"
    ;;
fetch)
    mkdir -p "$REPO/.cache/out/wexac_rows/$sha"
    rsync -a -e "ssh -o BatchMode=yes" "$HOSTALIAS:$RDIR/rows/$sha/" "$REPO/.cache/out/wexac_rows/$sha/"
    rsync -a -e "ssh -o BatchMode=yes" "$HOSTALIAS:$RDIR/logs/$sha/" "$REPO/.cache/out/wexac_rows/$sha/logs/"
    echo "rows: $(cat "$REPO/.cache/out/wexac_rows/$sha"/*.jsonl 2>/dev/null | wc -l) of $NJOBS"
    ;;
*)
    awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"; exit 2 ;;
esac
