#!/usr/bin/env bash
# Spec 003d T046-T047 (FR-018, R13, SC-011): the re-test and truth-witness fits on WEXAC AMD EPYC. Run from the sraging
# checkout on the Mac. Everything remote lives under ~/sraging_cov003 (spec 003's directory, the same venv); nothing
# else there is touched. The code moves as a git bundle of HEAD (clean tree; it must carry the "first3" mode); the
# arrays move as generated on the Mac (tools/retest_003d.py generate: the tagged generator, sha-checked on arrival
# and again by every fit).
#
#   tools/wexac/retest_003d.sh dry-run        print what push and submit would run; touches nothing
#   tools/wexac/retest_003d.sh push           bundle HEAD, copy it and the 67 arrays + manifest, check out HEAD there
#   tools/wexac/retest_003d.sh check          on the login node, seconds: imports, pins, the arrays against the manifest
#   tools/wexac/retest_003d.sh smoke --yes    element 1 alone (only with Naveh's yes)
#   tools/wexac/retest_003d.sh submit --yes   the LSF array, one fit per element (only with Naveh's yes)
#   tools/wexac/retest_003d.sh status         job states and rows written
#   tools/wexac/retest_003d.sh fetch          copy the rows to .cache/out/retest_rows/<sha>/
#   (status and fetch use HEAD; after later commits, name the run's: RT_SHA=<sha> tools/wexac/retest_003d.sh fetch)
#
# then on the Mac: python tools/acceptance_003d.py retest-score / witness (T048).
set -euo pipefail

HOSTALIAS=wexac
RDIR=sraging_cov003                                      # relative to the cluster home
QUEUE=long                                               # the witness's pure-k2-p05 (delta/L 0.001) may take hours
WLIMIT=10080
SELECT='select[model==AMD_EPYC]'                         # one CPU family, as spec 003's coverage fits
MEM_MB=4096                                              # jax2 fits peaked at 1038 MB; walks add
NJOBS=67                                                 # 55 censored re-test + 12 witness (amended 2026-10-14)
REPO=$(git rev-parse --show-toplevel)
VPY="$REPO/.venv/bin/python"
ARR="$REPO/.cache/out/retest_003d"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 "$HOSTALIAS")
sha=${RT_SHA:-$(git -C "$REPO" rev-parse HEAD)}       # RT_SHA=<the run's commit> for status/fetch after later commits
short=${sha:0:7}

remote() { "${SSH[@]}" "bash -lc $(printf '%q' "$1")"; }

jobscript() {
    cat <<EOF
#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1 PJRT_NPROC=1 SRAGING_J2_ROOT=\$HOME/$RDIR/j2
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd \$HOME/$RDIR/repo
exec \$HOME/$RDIR/venv/bin/python tools/retest_003d.py fit --index \$LSB_JOBINDEX --of $NJOBS \
    --out \$HOME/$RDIR/retest_rows/$sha/shard_\$LSB_JOBINDEX.jsonl
EOF
}

bsub_line() {   # $1: the index range
    echo "bsub -q $QUEUE -W $WLIMIT -n 1 -R '$SELECT rusage[mem=$MEM_MB] span[hosts=1]' -J 'rt003d_$short[$1]' -o logs/retest_$sha/%I.out -e logs/retest_$sha/%I.err logs/retest_$sha/job.sh"
}

preflight() {   # on the Mac: a clean tree carrying first3, and the arrays' manifest complete and not a dev run
    if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
        echo "REFUSED: the tracked tree is dirty; commit first" >&2; exit 2
    fi
    (cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 "$VPY" - "$NJOBS" <<'PY'
import json, sys
sys.path[:0] = ["tools", "src"]
from sraging.calibrators import profile_jax2 as PJ
import retest_003d as RT
assert "first3" in PJ.COMBINATION_MODES, "HEAD has no first3 mode"
m = json.load(open(RT.MANIFEST))
n = int(sys.argv[1])
assert len(m["cohorts"]) == n, f"the manifest has {len(m['cohorts'])} cohorts, this script submits {n}"
assert not m["dev"] and not m["generator_differs_from_tag"], "the arrays are a dev run or not the tagged generator's"
assert [r["cohort_id"] for r in m["cohorts"]] == [p["cohort_id"] for p in RT.panel_ids()], "the manifest is not the panel"
print("preflight ok: %d cohorts generated at %s (%s)" % (n, m["repo_head"][:7], m["tag"]))
PY
    ) || { echo "REFUSED: preflight failed" >&2; exit 2; }
    (cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 "$VPY" tools/retest_003d.py check) || exit 2
}

case "${1:-}" in
dry-run)
    echo "# push: git bundle create <tmp>/$short.bundle HEAD; scp it to $HOSTALIAS:$RDIR/bundles/"
    echo "#       rsync $ARR/ -> $HOSTALIAS:$RDIR/repo/.cache/out/retest_003d/"
    echo "#       on the cluster: git -C repo fetch bundles/$short.bundle; checkout --detach $sha"
    echo "# job script (logs/retest_$sha/job.sh):"; jobscript
    echo "# smoke (only with --yes):"; bsub_line 1
    echo "# submit (only with --yes):"; bsub_line "1-$NJOBS"
    ;;
push)
    preflight
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/rt003d.XXXXXX")
    git -C "$REPO" bundle create "$tmp/$short.bundle" HEAD
    remote "mkdir -p ~/$RDIR/bundles ~/$RDIR/retest_rows/$sha ~/$RDIR/logs/retest_$sha"
    scp -q -o BatchMode=yes "$tmp/$short.bundle" "$HOSTALIAS:$RDIR/bundles/"
    rm -rf "${tmp:?}"
    remote "set -e; cd ~/$RDIR
        if [ ! -d repo/.git ]; then git clone -q --no-checkout bundles/$short.bundle repo; fi
        git -C repo fetch -q ../bundles/$short.bundle HEAD
        git -C repo checkout -q --detach $sha
        test \"\$(git -C repo rev-parse HEAD)\" = $sha
        test -x venv/bin/python
        mkdir -p repo/.cache/out/retest_003d"
    rsync -a --delete -e "ssh -o BatchMode=yes" "$ARR/" "$HOSTALIAS:$RDIR/repo/.cache/out/retest_003d/"
    echo "pushed $short and $(ls "$ARR"/*.npz | wc -l | tr -d ' ') arrays"
    ;;
check)
    remote "set -e; cd ~/$RDIR/repo; test \"\$(git rev-parse HEAD)\" = $sha
        SRAGING_J2_ROOT=~/$RDIR/j2 PYTHONDONTWRITEBYTECODE=1 ~/$RDIR/venv/bin/python tools/retest_003d.py check
        SRAGING_J2_ROOT=~/$RDIR/j2 PYTHONDONTWRITEBYTECODE=1 ~/$RDIR/venv/bin/python -c '
import sys; sys.path[:0] = [\"tools\", \"src\"]
import acceptance_003_coverage as CV
CV._init()
import sraging, jax, numpy, scipy
from sraging.calibrators import profile_jax2 as PJ
print(\"sraging from\", sraging.__file__); print(\"jax\", jax.__version__, \"numpy\", numpy.__version__, \"scipy\", scipy.__version__)
print(\"modes\", PJ.COMBINATION_MODES); print(\"host\", CV.host())'"
    ;;
smoke|submit)
    if [ "${2:-}" != "--yes" ]; then
        echo "REFUSED: submitting needs --yes (Naveh's go)" >&2; exit 2
    fi
    if [ "$1" = smoke ]; then range=1; else range=1-$NJOBS; fi
    remote "set -e; cd ~/$RDIR; test \"\$(git -C repo rev-parse HEAD)\" = $sha
        cat > logs/retest_$sha/job.sh <<'JOB'
$(jobscript)
JOB
        chmod +x logs/retest_$sha/job.sh; $(bsub_line "$range")"
    ;;
status)
    remote "bjobs -a -J 'rt003d_$short' 2>/dev/null | awk 'NR>1{print \$3}' | sort | uniq -c
        echo rows: \$(cat ~/$RDIR/retest_rows/$sha/*.jsonl 2>/dev/null | wc -l) of $NJOBS"
    ;;
fetch)
    mkdir -p "$REPO/.cache/out/retest_rows/$sha"
    rsync -a -e "ssh -o BatchMode=yes" "$HOSTALIAS:$RDIR/retest_rows/$sha/" "$REPO/.cache/out/retest_rows/$sha/"
    rsync -a -e "ssh -o BatchMode=yes" "$HOSTALIAS:$RDIR/logs/retest_$sha/" "$REPO/.cache/out/retest_rows/$sha/logs/"
    echo "rows: $(cat "$REPO/.cache/out/retest_rows/$sha"/*.jsonl 2>/dev/null | wc -l) of $NJOBS"
    ;;
*)
    awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"; exit 2 ;;
esac
