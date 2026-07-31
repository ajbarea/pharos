#!/usr/bin/env bash
# Put the cluster on a specific, named commit.
#
# This used to rsync the working tree, which had a subtle and expensive flaw: rsync
# copies files but never touches git, so the cluster's HEAD stayed wherever it was
# last cloned while the files moved underneath it. Every artifact produced there
# recorded `git_dirty: true` against a stale commit, which means the exact code
# behind a measurement could not be reconstructed from its own provenance stamp. For
# a headline number that is the difference between a citable result and a screenshot.
#
# The repository is public, so the cluster can just fetch it. `--ff-only` refuses to
# invent a merge: if the remote moved in a way that cannot fast-forward, that is
# worth stopping for rather than papering over.
#
#   scripts/sync_cluster.sh              # sync to origin/main
#   scripts/sync_cluster.sh <ref>        # sync to a specific commit or tag
set -euo pipefail

REF="${1:-origin/main}"
REMOTE_HOST="${PHAROS_CLUSTER_HOST:-sporc}"
REMOTE_PATH="${PHAROS_CLUSTER_PATH:-\$HOME/ajsoftworks/pharos}"

cd "$(dirname "$0")/.."

# Refuse to sync from a dirty tree. Pushing first is the caller's job; quietly
# measuring code that matches no commit is exactly the failure this replaces.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "local tree is dirty. Commit and push first, or the cluster will run code" >&2
  echo "that matches no commit and its artifacts will say so." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

git fetch --quiet origin
if ! git merge-base --is-ancestor "$(git rev-parse HEAD)" origin/main 2>/dev/null; then
  echo "HEAD is not an ancestor of origin/main. Push first, or the cluster cannot" >&2
  echo "fetch the commit you are about to measure with." >&2
  exit 1
fi

echo ">>> syncing $REMOTE_HOST to $REF"
ssh -o BatchMode=yes "$REMOTE_HOST" "
  set -euo pipefail
  cd $REMOTE_PATH
  git fetch --quiet origin
  git checkout --quiet main 2>/dev/null || git checkout --quiet -B main origin/main
  git merge --ff-only --quiet $REF
  echo \"    HEAD: \$(git rev-parse --short=12 HEAD)\"
  dirty=\$(git status --porcelain --untracked-files=no | wc -l)
  echo \"    dirty files: \$dirty\"
"
echo ">>> done. Artifacts produced there will name this commit."
