#!/usr/bin/env bash
# One-time setup for Pharos on RIT Research Computing (SPORC).
#
# Verified against the live cluster on 2026-07-31. An earlier version of this file
# was written from older notes and was wrong in three ways that all mattered: it
# targeted a `grace` partition that no longer exists, assumed aarch64 nodes that are
# gone, and hand-installed an ARM64 Ollama. The GPU nodes are x86_64 Skylake and
# Sapphire Rapids, which makes all of that unnecessary.
set -euo pipefail

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.32.5}"

# Resolved once, before anything cds. The install steps below cd to $HOME, which
# breaks a relative $0 lookup later; that bug cost a run.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
# Presence is not the test; execution is. A previous install of this account's
# tooling was aarch64, and after the cluster moved to x86_64 those binaries were
# still on PATH and still passed `command -v`. They fail with "Exec format error"
# only when actually run, so that is what gets checked.
works() { "$1" --version >/dev/null 2>&1; }

if works "$HOME/.local/bin/uv" || works uv; then
  log "uv runs: $(uv --version 2>/dev/null || "$HOME/.local/bin/uv" --version)"
else
  log "Installing uv (existing binary missing or wrong architecture)"
  rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

# ---------------------------------------------------------------------------
# Ollama, unpacked into $HOME. No root here, so the official install script is not
# usable; the release tarball is.
# ---------------------------------------------------------------------------
if works "$HOME/bin/ollama"; then
  log "Ollama runs: $("$HOME/bin/ollama" --version 2>&1 | tail -1)"
else
  log "Installing Ollama $OLLAMA_VERSION (linux-amd64) into ~/bin"
  # Clear an aarch64 leftover, including its lib/ollama payload, or the new tarball
  # unpacks alongside stale ARM shared objects.
  rm -rf "$HOME/bin/ollama" "$HOME/lib/ollama"
  mkdir -p "$HOME/bin"
  cd "$HOME"
  # Release assets are zstd-compressed tarballs on GitHub, not .tgz from ollama.com
  # (that URL 404s). Asset names have changed across versions, so this is pinned.
  TARBALL="ollama-linux-amd64.tar.zst"
  curl -fL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/${TARBALL}" \
       -o "$TARBALL"
  zstd -d --rm "$TARBALL"
  tar -xf "ollama-linux-amd64.tar" -C "$HOME"
  rm -f "ollama-linux-amd64.tar"
  works "$HOME/bin/ollama" || { echo "ollama still will not run" >&2; exit 1; }
fi

log "Syncing the project"
cd "$REPO"
uv sync --all-groups

log "Add to ~/.bashrc if absent:"
cat <<'EOF'

  export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
  export OLLAMA_MODELS="$HOME/.ollama/models"
  export OLLAMA_CONTEXT_LENGTH=8192

EOF
log "Verify:  uv run python -m pharos.cli models"
