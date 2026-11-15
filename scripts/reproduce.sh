#!/usr/bin/env bash
# Reproduce the complete study into a separate directory; the repository's data/ and results/ are never written.
#
# Usage: scripts/reproduce.sh OUT_DIR [--reuse-raw DIR] [--stages LIST] [--compare RESULTS_DIR] [--config FILE]
#
#   OUT_DIR               new run directory (receives config.yaml, data/, results/, logs/)
#   --reuse-raw DIR       start from an existing data/raw instead of re-downloading. Files are cloned
#                         copy-on-write (APFS) or hard-linked, so DIR itself is never modified.
#   --stages LIST         comma-separated subset of stages (default: all)
#   --compare RESULTS_DIR compare the new results with an earlier results directory when the run finishes
#   --config FILE         study configuration to copy (default: config/nsclc.yaml)
#
# Environment: PYTHON (default: python) selects the interpreter with scrna2ml's dependencies installed.
# Rerunning with the same OUT_DIR resumes from the last completed checkpoint.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="$REPO/config/nsclc.yaml"
STAGES="download,prepare,seurat,features,ml,analysis,report"
REUSE_RAW=""
COMPARE=""

die() { echo "error: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: scripts/reproduce.sh OUT_DIR [--reuse-raw DIR] [--stages LIST] [--compare RESULTS_DIR] [--config FILE]"
OUT="$1"; shift
while [ $# -gt 0 ]; do
  case "$1" in
    --reuse-raw) REUSE_RAW="$(cd "$2" && pwd)"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --compare) COMPARE="$(cd "$2" && pwd)"; shift 2 ;;
    --config) CONFIG="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
case "$OUT/" in
  "$REPO/data/"*|"$REPO/results/"*|"$REPO/") die "OUT_DIR must be outside $REPO/data and $REPO/results" ;;
esac
mkdir -p "$OUT/data/raw" "$OUT/results" "$OUT/logs"

"$PYTHON" -c "import scrna2ml" 2>/dev/null || (cd "$REPO" && "$PYTHON" -c "import scrna2ml") \
  || die "$PYTHON cannot import scrna2ml; set PYTHON to the environment's interpreter"
command -v Rscript >/dev/null || die "Rscript not found"

if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  echo "warning: uncommitted changes in $REPO; run_log.jsonl will record a -dirty commit" >&2
fi

# Approximate space each stage adds (GB): raw downloads 14, prepared matrices 8, Seurat output 14, the rest < 1.
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
need_gb=2
if has_stage download && [ -z "$REUSE_RAW" ] && ! ls "$OUT"/data/raw/*/COMPLETE >/dev/null 2>&1; then need_gb=$((need_gb + 15)); fi
if has_stage prepare && [ ! -e "$OUT/data/prepared/libraries.tsv" ]; then need_gb=$((need_gb + 9)); fi
if has_stage seurat && [ ! -e "$OUT/results/seurat/qc_audit.tsv" ]; then need_gb=$((need_gb + 15)); fi
free_gb=$(df -Pk "$OUT" | awk 'NR==2 {print int($4 / 1048576)}')
[ "$free_gb" -ge "$need_gb" ] || die "only ${free_gb} GB free at $OUT; about ${need_gb} GB needed"

sed -e "s#^\([[:space:]]*data_dir:\).*#\1 $OUT/data#" \
    -e "s#^\([[:space:]]*results_dir:\).*#\1 $OUT/results#" "$CONFIG" > "$OUT/config.yaml"
grep -qF "data_dir: $OUT/data" "$OUT/config.yaml" && grep -qF "results_dir: $OUT/results" "$OUT/config.yaml" \
  || die "could not set data_dir/results_dir in $OUT/config.yaml"

if [ -n "$REUSE_RAW" ]; then
  for src in "$REUSE_RAW"/*/; do
    src="${src%/}"
    name="$(basename "$src")"
    dest="$OUT/data/raw/$name"
    [ -e "$dest" ] && continue
    echo "cloning raw/$name"
    if ! cp -cR "$src" "$dest" 2>/dev/null; then
      rm -rf "$dest"
      mkdir -p "$dest"
      ln "$src"/* "$dest/" || die "could not clone or hard-link $src (different volume?); omit --reuse-raw to download"
      # Hard links share content with DIR: give the small files the download stage rewrites their own copy.
      for f in "$dest"/*.json "$dest"/COMPLETE; do
        if [ -e "$f" ]; then cp "$f" "$f.tmp" && mv "$f.tmp" "$f"; fi
      done
    fi
  done
fi

echo "repository : $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "output     : $OUT"
echo "stages     : $STAGES"
echo "log        : $OUT/logs/run.log"

cd "$REPO"
"$PYTHON" -m scrna2ml run --config "$OUT/config.yaml" --stages "$STAGES" 2>&1 | tee -a "$OUT/logs/run.log"

if has_stage download; then
  "$PYTHON" scripts/compare_runs.py --verify-raw "$OUT/data/raw" | tee -a "$OUT/logs/run.log"
fi
if [ -n "$COMPARE" ]; then
  "$PYTHON" scripts/compare_runs.py "$COMPARE" "$OUT/results" | tee "$OUT/logs/compare.log"
fi
echo "done: results in $OUT/results"
