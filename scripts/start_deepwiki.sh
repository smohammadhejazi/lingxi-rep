#!/usr/bin/env bash
# The DeepWiki-Open API server behind v1.5's `ask_repository_agent` tool, on port
# 8008 as in v1.5. Answers with Claude (ANTHROPIC_API_KEY), embeds with OpenAI
# text-embedding-3-small (OPENAI_API_KEY, DeepWiki's default embedder). Its
# indexes go to ~/.adalflow.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set (DeepWiki embeddings)}"
export PORT="${LINGXI_DEEPWIKI_PORT:-8008}"
export NODE_ENV=production  # no auto-reload
cd "$ROOT/external/deepwiki-open"
exec "$ROOT/external/deepwiki-venv/bin/python" -m api.main
