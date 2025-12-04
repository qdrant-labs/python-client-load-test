# python-client-load-test
For testing QPS with a single Python client

# Example usage

`python main.py --qdrant-url $QDRANT_URL \
  --qdrant-api-key $QDRANT_API_KEY \
  --collection-name benchmark \
  --vector-name \
  --vector-dimension 1024 \
  --n 20000
  --b 10
  --limit 100
  --prefer-grpc
`
