#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Load .env file
if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ .env file not found at $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

# Validate required vars
if [[ -z "${QDRANT_URL:-}" || -z "${QDRANT_API_KEY:-}" ]]; then
  echo "❌ QDRANT_URL and QDRANT_API_KEY must be set in your .env file"
  exit 1
fi

source .venv/bin/activate


COLLECTION_NAME="${COLLECTION_NAME:-benchmark}"

PARQUET_FILE="$HOME/src/python/local-embed-gen/data/gte_fineweb_embeddings.parquet"
EMBEDDING_COLUMN="query_text_embedding.dense_vecs"

echo "🚀 Starting Qdrant benchmark..."
echo "   URL:        ${QDRANT_URL:0:30}..."
echo "   Collection: $COLLECTION_NAME"
echo "   Parquet:    $PARQUET_FILE"
echo "   Column:     $EMBEDDING_COLUMN"
echo ""

python main.py \
  --qdrant-url "$QDRANT_URL" \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --collection-name "$COLLECTION_NAME" \
  --parquet-file "$PARQUET_FILE" \
  --embedding-column "$EMBEDDING_COLUMN" \
  --vector-name "dense" \
  --limit 1000 \
  --prefer-grpc \
  -n 100000 \
  -b 500

deactivate
