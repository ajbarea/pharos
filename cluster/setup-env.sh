#!/usr/bin/env bash
# One-time setup for Pharos on RIT Research Computing (SPORC).
#
# Verified against the live cluster on 2026-07-31. An earlier version of this file
# was written from older notes and was wrong in three ways that all mattered: it
# targeted a `grace` partition that no longer exists, assumed aarch64 nodes that are
# gone, and hand-installed an ARM64 Ollama. The GPU nodes are x86_64 Skylake and
# Sapphire Rapids, which makes all of that unnecessary.
set -euo pipefail

log() { printf '\n>>> %s\n' "$*"; }

log "Architecture"
ARCH="$(uname -m)"
echo "$ARCH"
if [[ "$ARCH" != "x86_64" ]]; then
  echo "Expected x86_64. If this is an aarch64 node you are on TIGRIS, not SPORC," >&2
  echo "and the Ollama download below is the wrong build." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# uv, which brings its own Python.
#
# The cluster's spack Python is 3.11.x and Pharos requires >=3.12, so rather than
# fight the module system we let uv fetch a standalone interpreter. That also makes
# the cluster environment identical to a local checkout, which is the point of
# pinning the version at all.
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  log "uv present: $(uv --version)"
else
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

# ---------------------------------------------------------------------------
# Ollama, unpacked into $HOME. No root here, so the official install script is not
# usable; the release tarball is.
# ---------------------------------------------------------------------------
if [[ -x "$HOME/bin/ollama" ]]; then
  log "Ollama present: $("$HOME/bin/ollama" --version 2>&1 | tail -1)"
else
  log "Installing Ollama (linux-amd64) into ~/bin"
  mkdir -p "$HOME/bin"
  cd "$HOME"
  curl -fL https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
  tar -xzf ollama-linux-amd64.tgz -C "$HOME"
  rm -f ollama-linux-amd64.tgz
fi

log "Syncing the project"
cd "$(dirname "$0")/.."
uv sync --all-groups

log "Add to ~/.bashrc if absent:"
cat <<'EOF'

  export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
  export OLLAMA_MODELS="$HOME/.ollama/models"
  export OLLAMA_CONTEXT_LENGTH=8192

EOF
log "Verify:  uv run python -m pharos.cli models"
