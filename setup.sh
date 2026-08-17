#!/usr/bin/env bash
# ARIA Ultra — Ubuntu setup script
# Installs Python deps + Ollama models. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

echo "── ARIA Ultra setup ──────────────────────────────"

# 1. Python virtualenv
if [ ! -d .venv ]; then
    echo "▶ Creating virtualenv"
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Dependencies + install the package
echo "▶ Installing Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet -e .

# 3. Ollama (if missing)
if ! command -v ollama >/dev/null 2>&1; then
    echo "▶ Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 4. Pull models (skipped if already present)
pull_if_missing() {
    if ! ollama list 2>/dev/null | grep -q "$1"; then
        echo "▶ Pulling $1 (~1-2 GB download)"
        ollama pull "$1"
    else
        echo "  $1 already present"
    fi
}
pull_if_missing granite4.1:3b   # main model (chat + code)
pull_if_missing hermes3:3b      # fallback provider (auto-failover)
pull_if_missing nomic-embed-text  # embeddings for semantic memory

# 5. .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "▶ Created .env (defaults are fine)"
fi

echo
echo "✓ ARIA Ultra ready. Start it with:"
echo "    source .venv/bin/activate && aria-ultra"
echo "  (start Ollama first if it isn't running: ollama serve)"
