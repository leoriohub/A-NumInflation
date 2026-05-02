#!/usr/bin/env bash
# sync_from_lab.sh — Pull sweep results from the lab machine to local.
#
# Usage:
#   ./scripts/sync_from_lab.sh                    # dry run
#   ./scripts/sync_from_lab.sh --run              # actual sync
#   ./scripts/sync_from_lab.sh --run --watch       # sync + open analysis notebook
#
# Configuration — edit these for your lab machine:
LAB_HOST="lab"
LAB_PROJECT_DIR="~/A-NumInflation"
LOCAL_OUTPUTS_DIR="outputs"

set -euo pipefail

DRY_RUN=true
WATCH=false

for arg in "$@"; do
  case "$arg" in
    --run) DRY_RUN=false ;;
    --watch) WATCH=true ;;
    --help)
      echo "Usage: $0 [--run] [--watch]"
      echo "  --run     actually rsync (default: dry-run)"
      echo "  --watch   open the analysis notebook after syncing"
      exit 0
      ;;
  esac
done

RSYNC_OPTS="-avz --progress"
if $DRY_RUN; then
  RSYNC_OPTS="$RSYNC_OPTS --dry-run"
fi

echo "========================================"
if $DRY_RUN; then
  echo " DRY RUN — add --run to actually sync"
  echo "========================================"
else
  echo " SYNCING from $LAB_HOST:$LAB_PROJECT_DIR"
  echo "========================================"
fi

rsync $RSYNC_OPTS "$LAB_HOST:$LAB_PROJECT_DIR/outputs/" "$LOCAL_OUTPUTS_DIR/"

echo ""
echo "Done. Results are in $LOCAL_OUTPUTS_DIR/"

if $WATCH && ! $DRY_RUN; then
  echo "Opening analysis notebook..."
  jupyter lab notebooks/Mapping_ns_r_Sensitivity.ipynb 2>/dev/null || \
    echo "Run 'jupyter lab notebooks/Mapping_ns_r_Sensitivity.ipynb' manually"
fi
