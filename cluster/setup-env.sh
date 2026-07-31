#!/usr/bin/env bash
# One-time setup for Pharos on an RIT Research Computing node.
#
# NOT YET EXECUTED ON THE CLUSTER. Read before running; it writes to $HOME.
#
# Everything here is ARM64 on purpose. The GPU nodes are aarch64 (Grace), and an
# x86_64 binary fails in ways that read like a missing library rather than a wrong
# architecture.
set -euo pipefail

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.15.0}"
BIN="$HOME/bin"

log() { printf '\n>>> %s\n' "$*"; }

log "Checking architecture"
ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "This node reports '$ARCH', not aarch64." >&2
  echo "Run this on a grace-partition node, not the submit host." >&2
  exit 1
fi
echo "ok: $ARCH"

# ---------------------------------------------------------------------------
# uv, which brings its own Python.
#
# The cluster's spack Python is 3.11.7 and Pharos requires >=3.12. Rather than
# fight the module system, let uv fetch a standalone aarch64 interpreter. This also
# makes the environment identical to a local checkout.
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  log "uv already present: $(uv --version)"
else
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# Ollama. No installer path here: the ARM64 tarball is unpacked by hand.
# ---------------------------------------------------------------------------
if [[ -x "$BIN/ollama" ]]; then
  log "Ollama already present: $("$BIN/ollama" --version 2>&1 | head -1)"
else
  log "Installing Ollama $OLLAMA_VERSION (arm64)"
  mkdir -p "$BIN"
  TARBALL="ollama-linux-arm64.tgz"
  URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/${TARBALL}"
  cd "$HOME"
  if ! curl -fL "$URL" -o "$TARBALL"; then
    cat >&2 <<'HINT'
Download failed. Release asset names change between Ollama versions; the notes
that produced this script used ollama-linux-arm64.tar.zst, which needs `zstd -d`
before `tar -xf`. Check the release page for the current arm64 asset name and set
OLLAMA_VERSION, or fetch it locally and scp it to ~.
HINT
    exit 1
  fi
  tar -xzf "$TARBALL" -C "$HOME"
  rm -f "$TARBALL"
fi

export PATH="$BIN:$HOME/.local/bin:$PATH"

log "Syncing the project"
cd "$(dirname "$0")/.."
uv sync --all-groups

log "Done. Add to ~/.bashrc if it is not there already:"
cat <<'EOF'

  export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
  export OLLAMA_CONTEXT_LENGTH=8192

EOF
log "Verify with:  nvidia-smi && uv run python -m pharos.cli models"
