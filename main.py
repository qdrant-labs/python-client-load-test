import argparse
import asyncio
import os

from dotenv import find_dotenv, load_dotenv

from load_test.config import load_sweep_config
from load_test.runner import run_load_test, run_sweep
from load_test.vectors import load_embeddings_from_parquet, load_sparse_embeddings_from_parquet


class _AddVectorAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        try:
            name, dim_str = values.split(":")
            getattr(namespace, self.dest)[name] = int(dim_str)
        except ValueError:
            raise argparse.ArgumentError(self, f"Expected 'name:dimension', got '{values}'")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant load testing and recall sweep tool.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    client_group = parser.add_argument_group("Client Connection")
    client_group.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL"), help="Qdrant server URL.")
    client_group.add_argument("--qdrant-api-key", type=str, default=os.getenv("QDRANT_API_KEY"), help="Qdrant API Key.")
    client_group.add_argument("--timeout", type=float, default=10000.0, help="Request timeout in seconds.")
    client_group.add_argument("--prefer-grpc", action=argparse.BooleanOptionalAction, default=True, help="Use gRPC.")

    test_group = parser.add_argument_group("Test Configuration")
    test_group.add_argument("--collection-name", type=str, default=os.getenv("COLLECTION_NAME"), help="Collection name.")
    test_group.add_argument("-n", "--num-queries", type=int, default=10000, help="Total queries per phase.")
    test_group.add_argument("-b", "--num-batches", type=int, default=1, help="Number of batches.")
    test_group.add_argument("-c", "--concurrency", type=int, default=250, help="Max concurrent requests.")
    test_group.add_argument("--output", type=str, default="load_test_report.html", help="Path for HTML report output.")

    vector_group = parser.add_argument_group("Vector / Query Configuration")
    vector_group.add_argument("--vector-name", type=str, default=None, help="Named vector to search against.")
    vector_group.add_argument("--vector-dimension", type=int, default=None, help="Dimension for random vectors.")
    vector_group.add_argument("--parquet-file", type=str, default=None, help="Path to parquet file with dense query embeddings.")
    vector_group.add_argument("--embedding-column", type=str, default="embedding", help="Column in parquet file (default: 'embedding').")
    vector_group.add_argument("--sparse-parquet-file", type=str, default=None,
                              help="Path to parquet file with sparse query embeddings (required for query_mode: sparse/hybrid).")
    vector_group.add_argument("--sparse-embedding-column", type=str, default="sparse_embedding",
                              help="Column name for sparse embeddings (default: 'sparse_embedding').")
    vector_group.add_argument("--dense-vector", action=_AddVectorAction, dest="dense_vectors")
    vector_group.add_argument("--sparse-vector", action=_AddVectorAction, dest="sparse_vectors")

    search_group = parser.add_argument_group("Search Parameters (single-run mode)")
    search_group.add_argument("--limit", type=int, default=10, help="Top-k results per query.")
    search_group.add_argument("--rescore", action=argparse.BooleanOptionalAction, default=False, help="Enable quantization rescoring.")

    reliability_group = parser.add_argument_group("Reliability")
    reliability_group.add_argument("--max-retries", type=int, default=3,
                                   help="Max retry attempts for transient query errors (default: 3).")
    reliability_group.add_argument("--allow-wrap", action=argparse.BooleanOptionalAction, default=True,
                                   help="Allow query index wrap when num-queries > available embeddings (default: True).\n"
                                        "Use --no-allow-wrap to raise instead.")
    reliability_group.add_argument("--warmup-queries", type=int, default=200,
                                   help="Warmup queries per run before timing (default: 200, 0 disables).")

    sweep_group = parser.add_argument_group("Sweep Mode")
    sweep_group.add_argument(
        "--experiment-file", type=str, default=None,
        help=(
            "Path to a YAML file defining an ordered experiment sweep.\n"
            "When provided, all --rescore and single-run search flags are ignored;\n"
            "per-run params come from the YAML. Brute-force runs once and is reused.\n"
            "See experiment_example.yaml for the config schema."
        ),
    )

    resume_group = parser.add_argument_group("Checkpoint / Resume (sweep mode only)")
    resume_group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                              help="Resume a previously interrupted sweep from checkpoint.")
    resume_group.add_argument("--checkpoint-file", type=str, default=None,
                              help=(
                                  "Base path for checkpoint files.\n"
                                  "Default: derived from --output (e.g. sweep.html → sweep.checkpoint.jsonl / sweep.exact_cache.json).\n"
                                  "Decouples resume state from the output report filename."
                              ))

    return parser


def main():
    load_dotenv(find_dotenv())
    args = _build_parser().parse_args()
    args.dense_vectors = args.dense_vectors or {}
    args.sparse_vectors = args.sparse_vectors or {}

    if not args.qdrant_url or not args.qdrant_api_key or not args.collection_name:
        raise ValueError("QDRANT_URL, QDRANT_API_KEY, and COLLECTION_NAME must be provided (env or CLI).")

    query_vectors = None
    query_source = f"Random vectors (dim={args.vector_dimension})"

    if args.parquet_file:
        query_vectors = load_embeddings_from_parquet(args.parquet_file, args.embedding_column)
        query_source = f"{args.parquet_file} (column: {args.embedding_column})"
        if args.vector_dimension is None:
            args.vector_dimension = query_vectors.shape[1]
    elif args.vector_dimension is None and args.experiment_file is None:
        raise ValueError("Either --parquet-file or --vector-dimension must be specified.")

    sparse_query_vectors = None
    if args.sparse_parquet_file:
        sparse_query_vectors = load_sparse_embeddings_from_parquet(
            args.sparse_parquet_file, args.sparse_embedding_column
        )

    common = dict(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection_name,
        vector_name=args.vector_name,
        num_queries=args.num_queries,
        num_batches=args.num_batches,
        concurrency=args.concurrency,
        limit=args.limit,
        prefer_grpc=args.prefer_grpc,
        timeout=args.timeout,
        output=args.output,
        vector_dimension=args.vector_dimension,
        query_vectors=query_vectors,
        allow_wrap=args.allow_wrap,
        warmup_queries=args.warmup_queries,
        max_retries=args.max_retries,
        sparse_query_vectors=sparse_query_vectors,
    )

    if args.experiment_file:
        sweep = load_sweep_config(args.experiment_file)
        asyncio.run(run_sweep(
            sweep=sweep,
            resume=args.resume,
            checkpoint_file=args.checkpoint_file,
            **common,
        ))
    else:
        asyncio.run(run_load_test(rescore=args.rescore, query_source=query_source, **common))


if __name__ == "__main__":
    main()
