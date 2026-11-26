#!/usr/bin/env bash
# Spec 003d SC-006 (T031-T032): generate the whole library on WEXAC AMD EPYC nodes and compare every cohort's checksum
# with the Mac's (the G-test's run 1). Run from the sraging checkout on the Mac. Everything remote lives under
# ~/sraging_cov003 (spec 003's directory; the same venv); nothing else there is touched. The code moves as a git bundle
# of HEAD (clean tree); only library v5's spec manifest is copied (the generator needs the specs, not the arrays).
#
#   tools/wexac/determinism_003d.sh dry-run        print what push and submit would run; touches nothing
#   tools/wexac/determinism_003d.sh push           bundle HEAD, copy it and v5's manifest, check out HEAD there
#   tools/wexac/determinism_003d.sh check          on the login node: imports, pins, the manifest loads
#   tools/wexac/determinism_003d.sh submit --yes   the LSF array, NSHARDS elements (only with Naveh's yes)
#   tools/wexac/determinism_003d.sh status         job states and shards written
#   tools/wexac/determinism_003d.sh fetch          copy the shards to .cache/out/epyc_det/<sha>/
#
# then on the Mac, at the same HEAD:
#   python tools/acceptance_003d.py determinism --compare-epyc .cache/out/epyc_det/<sha>
set -euo pipefail

HOSTALIAS=wexac
RDIR=sraging_cov003
QUEUE=short
WLIMIT=1440
SELECT='select[model==AMD_EPYC]'          # the CPU family is recorded per element (the reference used Zen 2 and Zen 3)
MEM_MB=4096
NSHARDS=40
REPO=$(git rev-parse --show-toplevel)
VPY="$REPO/.venv/bin/python"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 "$HOSTALIAS")
sha=$(git -C "$REPO" rev-parse HEAD)
short=${sha:0:7}
V5MAN=$(cd "$REPO" && "$VPY" -c "import sys;sys.path.insert(0,'tools');import v5_reproduce_003d as R,os;print(os.path.join(R.V5,'manifest.json'))")

remote() { "${SSH[@]}" "bash -lc $(printf '%q' "$1")"; }

jobscript() {
    cat <<EOF
#!/bin/bash
export SRAGING_TESTER_MAIN=\$HOME/$RDIR/tester_v5 PYTHONDONTWRITEBYTECODE=1 PJRT_NPROC=1
export PYTHONPATH=\$HOME/$RDIR/repo/src     # sraging is not installed in the venv (job 824277 failed on the import)
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd \$HOME/$RDIR/repo
exec \$HOME/$RDIR/venv/bin/python tools/acceptance_003d.py determinism-shard --index \$LSB_JOBINDEX --of $NSHARDS \
    --out \$HOME/$RDIR/epyc_det/$sha/shard_\$LSB_JOBINDEX.json
EOF
}

bsub_line() {
    echo "bsub -q $QUEUE -W $WLIMIT -n 1 -R '$SELECT rusage[mem=$MEM_MB] span[hosts=1]' -J 'det003d_$short[1-$NSHARDS]' -o logs/det_$sha/%I.out -e logs/det_$sha/%I.err logs/det_$sha/job.sh"
}

case "${1:-}" in
dry-run)
    echo "# push: git bundle create <tmp>/$short.bundle HEAD; scp it to $HOSTALIAS:$RDIR/bundles/"
    echo "#       rsync $V5MAN -> $HOSTALIAS:$RDIR/tester_v5/data/library/v5/manifest.json"
    echo "#       on the cluster: git -C repo fetch bundles/$short.bundle; checkout --detach $sha"
    echo "# job script (logs/det_$sha/job.sh):"; jobscript
    echo "# submit (only with --yes):"; bsub_line
    ;;
push)
    if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
        echo "REFUSED: the tracked tree is dirty; commit first" >&2; exit 2
    fi
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/det003d.XXXXXX")
    git -C "$REPO" bundle create "$tmp/$short.bundle" HEAD
    remote "mkdir -p ~/$RDIR/bundles ~/$RDIR/tester_v5/data/library/v5 ~/$RDIR/epyc_det/$sha ~/$RDIR/logs/det_$sha"
    scp -q -o BatchMode=yes "$tmp/$short.bundle" "$HOSTALIAS:$RDIR/bundles/"
    rsync -a -e "ssh -o BatchMode=yes" "$V5MAN" "$HOSTALIAS:$RDIR/tester_v5/data/library/v5/manifest.json"
    rm -rf "${tmp:?}"
    remote "set -e; cd ~/$RDIR
        if [ ! -d repo/.git ]; then git clone -q --no-checkout bundles/$short.bundle repo; fi
        git -C repo fetch -q ../bundles/$short.bundle HEAD
        git -C repo checkout -q --detach $sha
        test \"\$(git -C repo rev-parse HEAD)\" = $sha
        test -x venv/bin/python
        echo pushed $short; venv/bin/python --version"
    ;;
check)
    # the job's own environment and import path (no sys.path edits), so check cannot pass where a job fails
    remote "set -e; cd ~/$RDIR/repo; test \"\$(git rev-parse HEAD)\" = $sha
        SRAGING_TESTER_MAIN=~/$RDIR/tester_v5 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/$RDIR/repo/src:~/$RDIR/repo/tools ~/$RDIR/venv/bin/python -c '
import acceptance_003d as ACC
import sraging, jax, numpy, scipy
from sraging import generator as G
_, rows = ACC.library_specs()
print(\"sraging from\", sraging.__file__); print(\"jax\", jax.__version__, \"numpy\", numpy.__version__, \"scipy\", scipy.__version__)
print(\"cohorts\", len(rows), \"grid\", G.GRID, \"version\", G.GENERATOR_VERSION); print(\"cpu\", ACC.platform_cpu())'"
    ;;
submit)
    if [ "${2:-}" != "--yes" ]; then
        echo "REFUSED: submitting needs --yes (Naveh's go)" >&2; exit 2
    fi
    remote "set -e; cd ~/$RDIR; test \"\$(git -C repo rev-parse HEAD)\" = $sha
        cat > logs/det_$sha/job.sh <<'JOB'
$(jobscript)
JOB
        chmod +x logs/det_$sha/job.sh; $(bsub_line)"
    ;;
status)
    remote "bjobs -a -J 'det003d_$short' 2>/dev/null | awk 'NR>1{print \$3}' | sort | uniq -c
        echo shards: \$(ls ~/$RDIR/epyc_det/$sha/shard_*.json 2>/dev/null | wc -l) of $NSHARDS"
    ;;
fetch)
    mkdir -p "$REPO/.cache/out/epyc_det/$sha"
    rsync -a -e "ssh -o BatchMode=yes" "$HOSTALIAS:$RDIR/epyc_det/$sha/" "$REPO/.cache/out/epyc_det/$sha/"
    rsync -a -e "ssh -o BatchMode=yes" "$HOSTALIAS:$RDIR/logs/det_$sha/" "$REPO/.cache/out/epyc_det/$sha/logs/"
    echo "shards: $(ls "$REPO/.cache/out/epyc_det/$sha"/shard_*.json 2>/dev/null | wc -l) of $NSHARDS"
    ;;
*)
    awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"; exit 2 ;;
esac
