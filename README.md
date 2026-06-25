# python-client-load-test

Qdrant load-testing and recall-sweep tool. Measures QPS and recall@k for a single Python client against a live Qdrant instance.

## Single-run mode

```bash
uv run main.py \
  --qdrant-url $QDRANT_URL \
  --qdrant-api-key $QDRANT_API_KEY \
  --collection-name benchmark \
  --vector-dimension 1024 \
  --limit 100 \
  --prefer-grpc \
  -n 20000 -b 10
```

Runs an ANN phase followed by a brute-force exact phase, computes recall, and writes an HTML report.

## Sweep mode

Sweeps through a YAML-configured list of runs, running brute-force once per unique `(collection, vector, limit, filter, mode)` combination.

```bash
uv run main.py \
  --qdrant-url $QDRANT_URL \
  --qdrant-api-key $QDRANT_API_KEY \
  --collection-name my_collection \
  --parquet-file queries.parquet \
  --num-queries 1000 \
  --concurrency 100 \
  --limit 10 \
  --output sweep_report.html \
  --experiment-file experiment_example.yaml
```

### Sweep YAML structure

Runs are expressed as a three-level hierarchy. The tool auto-generates one `ExperimentRun` per `(group × index_config × search_param)` combination:

```yaml
stopping_metric: p10
stopping_threshold: 0.99

groups:
  - name: scalar                        # prefix for all auto-generated run names
    collection_config:                  # quantization applied at collection level
      quantization: scalar
      always_ram: true
    index_configs:                      # HNSW graph build params; one rebuild each
      - m: 16
        ef_construct: 128
      - m: 32
        ef_construct: 256
    search_params:                      # swept within each index build
      - hnsw_ef: 64
      - hnsw_ef: 128
      - hnsw_ef: 256
        rescore: true
        oversampling: 2.0
```

The above generates 6 runs named e.g. `scalar/m16-efc128/ef64`, `scalar/m32-efc256/ef256-rescore-os2x`.  
The **first** `search_param` under each `index_config` triggers a collection rebuild (`apply_config=True`); the rest reuse the built index.

Add `apply_config: false` on a group or index_config to skip rebuilds (collection already built).

For one-off or custom runs (cross-collection tests, hybrid/sparse, filters), use the flat `runs:` list — appended after groups, backward-compatible with the old format.

See `experiment_example.yaml` for the full schema.

## Checkpoint / Resume

A sweep interrupted by `Ctrl-C` or a transient error can be resumed without re-running completed experiments or re-computing brute-force baselines:

```bash
# Start a long sweep
uv run main.py ... --output sweep_report.html --experiment-file sweep.yaml

# After interruption — resume from where it left off
uv run main.py ... --output sweep_report.html --experiment-file sweep.yaml --resume
```

Checkpoint files are derived from `--output` automatically:
- `sweep_report.checkpoint.jsonl` — one JSON line per completed run
- `sweep_report.exact_cache.json` — brute-force results cache

To use a custom checkpoint location (e.g. to write a fresh report from old progress):

```bash
uv run main.py ... --output new_report.html --resume --checkpoint-file sweep_report
```

## Filtered recall benchmarking

Add a `query_filter` to any run to benchmark filtered ANN recall. The identical filter is applied to both the ANN search and the brute-force baseline, so recall numbers are correct:

```yaml
# experiment_filtered.yaml
stopping_metric: p10

runs:
  - name: "scalar-filtered"
    description: "INT8 scalar, category=A filter"
    apply_config: true
    collection_config:
      quantization: scalar
      always_ram: true
    hnsw_ef: 128
    query_filter:
      must:
        - key: category
          match:
            value: "A"
```

Runs with different filters get separate brute-force baselines — no cache collisions.

## Hybrid / sparse benchmarking

To benchmark sparse or hybrid (dense+sparse RRF fusion) retrieval, provide sparse query embeddings via `--sparse-parquet-file`. The parquet column must contain structs with `indices` (list[int]) and `values` (list[float]) fields:

```bash
uv run main.py \
  --qdrant-url $QDRANT_URL \
  --qdrant-api-key $QDRANT_API_KEY \
  --collection-name hybrid_collection \
  --parquet-file dense_queries.parquet \
  --sparse-parquet-file sparse_queries.parquet \
  --num-queries 1000 \
  --output hybrid_report.html \
  --experiment-file hybrid_sweep.yaml
```

Example YAML for hybrid/sparse runs:

```yaml
# hybrid_sweep.yaml
stopping_metric: p10

runs:
  # Pure sparse retrieval
  - name: "sparse-only"
    description: "SPLADE sparse retrieval"
    query_mode: sparse
    sparse_vector_name: sparse_vectors
    apply_config: false

  # Hybrid: RRF fusion of dense + sparse prefetches
  - name: "hybrid-rrf"
    description: "Dense + sparse RRF fusion"
    query_mode: hybrid
    sparse_vector_name: sparse_vectors
    apply_config: true
    collection_config:
      quantization: scalar
      always_ram: true
    hnsw_ef: 128

  # Dense baseline for comparison
  - name: "dense-scalar"
    description: "Dense-only INT8 scalar (control)"
    query_mode: dense
    apply_config: false
    hnsw_ef: 128
```

For hybrid runs, the brute-force baseline also uses RRF fusion with `exact=True` on each prefetch, so recall reflects true hybrid retrieval quality.

## All flags

```
Client Connection:
  --qdrant-url          Qdrant server URL (or QDRANT_URL env var)
  --qdrant-api-key      API key (or QDRANT_API_KEY env var)
  --timeout             Request timeout in seconds (default: 10000)
  --prefer-grpc / --no-prefer-grpc

Test Configuration:
  --collection-name     Collection to query (or COLLECTION_NAME env var)
  -n / --num-queries    Queries per phase (default: 10000)
  -b / --num-batches    Number of batches (default: 1)
  -c / --concurrency    Max concurrent requests (default: 250)
  --output              HTML report path (default: load_test_report.html)

Vector / Query Configuration:
  --vector-name         Named vector to query
  --vector-dimension    Dimension for random vectors
  --parquet-file        Parquet file with dense query embeddings
  --embedding-column    Column name for dense embeddings (default: embedding)
  --sparse-parquet-file Parquet file with sparse embeddings (required for sparse/hybrid mode)
  --sparse-embedding-column  Column name for sparse embeddings (default: sparse_embedding)

Search Parameters (single-run mode):
  --limit               Top-k per query (default: 10)
  --rescore / --no-rescore

Reliability:
  --max-retries         Retry attempts on transient errors (default: 3)
  --allow-wrap / --no-allow-wrap   Whether to wrap query index when num-queries > embeddings
  --warmup-queries      Warmup queries before each timed run (default: 200, 0 disables)

Sweep Mode:
  --experiment-file     Path to sweep YAML config

Checkpoint / Resume:
  --resume / --no-resume  Resume from checkpoint (sweep only)
  --checkpoint-file       Custom checkpoint base path (default: derived from --output)
```
