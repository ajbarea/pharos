#!/usr/bin/env bash
# Push this checkout to the cluster.
#
# The exclusions matter more than the command. `--delete` keeps the remote from
# accumulating files a local delete removed, but anything generated ONLY on the
# cluster looks locally-absent and would be deleted by it. cluster/logs/ is exactly
# that: job output written by Slurm, never written here. Excluding it once, in a
# script, is more reliable than remembering the flag every time.
set -euo pipefail

REMOTE="${1:-sporc:~/ajsoftworks/pharos/}"
cd "$(dirname "$0")/.."

rsync -az --delete \
  --exclude='.venv/' \
  --exclude='site/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.coverage' \
  --exclude='export/' \
  --exclude='.cache/' \
  --exclude='adapter-out/' \
  --exclude='cluster/logs/' \
  --exclude='.git/' \
  -e "ssh -o BatchMode=yes" \
  ./ "$REMOTE"

echo "synced to $REMOTE (cluster/logs/ preserved)"
