#!/usr/bin/env bash
# sync-to-huggingface.sh
#
# Syncs all application source files and HF deployment files from this repo
# (employee-attrition-prediction) into the sibling HF space repo, then commits and pushes
# to both HuggingFace Spaces and the GitHub mirror.
#
# Usage:  ./scripts/sync-to-huggingface.sh
#         ./scripts/sync-to-huggingface.sh --dry-run   (shows what would change)
#
# Never edit files directly in the HF space repo — always run this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
HF="$(cd "$SRC/../employee-attrition-prediction-space/employee-attrition-prediction-space" && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "[dry-run] No files will be written or pushed."
fi

# Remove a directory in HF space then copy fresh from source.
# This avoids nested-directory bugs from cp -r when the dest already exists.
sync_dir() {
  local name="$1"
  if $DRY_RUN; then
    echo "  [dry-run] sync dir: $name/"
  else
    rm -rf "$HF/$name"
    cp -r "$SRC/$name" "$HF/$name"
  fi
}

sync_file() {
  local src_path="$1"
  local dst_path="$2"
  if $DRY_RUN; then
    echo "  [dry-run] sync file: $dst_path"
  else
    mkdir -p "$(dirname "$HF/$dst_path")"
    cp "$SRC/$src_path" "$HF/$dst_path"
  fi
}

echo "==> Source : $SRC"
echo "==> HF repo: $HF"
echo ""

# ---------------------------------------------------------------------------
# 1. Application source directories (fully mirrored from this repo)
# ---------------------------------------------------------------------------
echo "--- Syncing source directories ---"

sync_dir api
sync_dir core
sync_dir database
sync_dir ui
sync_dir data

# ---------------------------------------------------------------------------
# 2. Model artifacts
# Only the 5 files the app actually loads — not snapshots or report exports.
# ---------------------------------------------------------------------------
echo "--- Syncing model artifacts ---"

mkdir -p "$HF/models"
for f in employee_attrition_pipeline.pkl X_train.parquet X_test.parquet y_train.parquet y_test.parquet; do
  sync_file "models/$f" "models/$f"
done

# ---------------------------------------------------------------------------
# 3. Individual files
# ---------------------------------------------------------------------------
echo "--- Syncing individual files ---"

sync_file "scripts/utils.py"        "scripts/utils.py"
sync_file ".streamlit/config.toml"  ".streamlit/config.toml"

# ---------------------------------------------------------------------------
# 4. HF deployment files (live in docker/ of this repo)
# ---------------------------------------------------------------------------
echo "--- Syncing deployment files ---"

sync_file "docker/Dockerfile.huggingface"       "Dockerfile"
sync_file "docker/requirements.huggingface.txt"  "requirements.txt"
sync_file "docker/start.huggingface.sh"          "start.sh"

if $DRY_RUN; then
  echo ""
  echo "Dry-run complete. No changes written."
  exit 0
fi

# ---------------------------------------------------------------------------
# 5. Commit and push
# ---------------------------------------------------------------------------
echo ""
echo "--- Committing in HF repo ---"

cd "$HF"
git add -A

if git diff --cached --quiet; then
  echo "Nothing changed — HF space is already up to date."
  exit 0
fi

DATE=$(date +%Y-%m-%d)
# SYNC_FROM_MAIN=1 signals the pre-commit hook that this is an authorised sync
SYNC_FROM_MAIN=1 git commit -m "Sync from employee-attrition-prediction $DATE"

echo ""
echo "--- Pushing to HuggingFace Spaces ---"
git push origin main

echo "--- Pushing to GitHub mirror ---"
git push github main

echo ""
echo "Done. HuggingFace will rebuild the Space shortly."
echo "Live demo: https://huggingface.co/spaces/shah-data-scientist/employee-attrition-prediction"
