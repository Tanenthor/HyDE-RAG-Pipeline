#!/usr/bin/env bash
# scripts/init_ollama_models.sh
#
# Pull the required Ollama models after `docker-compose up -d`.
# Run once during initial setup:
#   bash scripts/init_ollama_models.sh
#
# Override models via environment variables, e.g.:
#   GENERATION_MODEL=mistral bash scripts/init_ollama_models.sh

set -euo pipefail

GENERATION_MODEL="${GENERATION_MODEL:-llama3}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
CONTAINER="${OLLAMA_CONTAINER:-ollama}"

echo "==> Waiting for Ollama container to be ready…"
until docker exec "${CONTAINER}" ollama list &>/dev/null; do
  sleep 2
done
echo "    Ollama is ready."

echo "==> Pulling generation model: ${GENERATION_MODEL}"
docker exec "${CONTAINER}" ollama pull "${GENERATION_MODEL}"

echo "==> Pulling embedding model: ${EMBEDDING_MODEL}"
docker exec "${CONTAINER}" ollama pull "${EMBEDDING_MODEL}"

echo ""
echo "✅ Models ready. You can now ingest documents and query the pipeline."
