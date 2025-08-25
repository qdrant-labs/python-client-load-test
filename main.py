import argparse
import asyncio
import os
from time import time
from typing import Optional, Dict

import numpy as np
from dotenv import load_dotenv, find_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


CLIENT: Optional[AsyncQdrantClient] = None


def random_dense_vector(dim: int) -> list[float]:
    """Generates a random dense vector of a given size."""
    return np.random.rand(dim).astype(np.float32).tolist()


def random_sparse_vector(dim: int) -> models.SparseVector:
    """Generates a random sparse vector."""
    num_non_zero = np.random.randint(1, min(dim, 100)) # Limit non-zero elements
    indices = np.random.choice(dim, size=num_non_zero, replace=False).tolist()
    values = np.random.rand(num_non_zero).astype(np.float32).tolist()
    indices.sort()
    return models.SparseVector(indices=indices, values=values)


async def search(
    collection_name: str,
    vector_name: str,
    limit: int,
    rescore: bool,
    dense_vectors: Dict[str, int],
    sparse_vectors: Dict[str, int],
):
    """Performs a single search query against the collection."""
    search_params = {
        "quantization": {"rescore": rescore}
    } if rescore is not None else None

    query_vector = None
    if vector_name in dense_vectors:
        query_vector = random_dense_vector(dense_vectors[vector_name])
    elif vector_name in sparse_vectors:
        query_vector = random_sparse_vector(sparse_vectors[vector_name])
    else:
        query_vector = random_dense_vector(dense_vectors.get(vector_name, 384))


    await CLIENT.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
        limit=limit,
        search_params=search_params,
    )

async def execute_batch(
    num_queries_in_batch: int,
    args: argparse.Namespace
):
    """Creates and executes a batch of search coroutines."""
    futures = [
        search(
            args.collection_name,
            args.vector_name,
            args.limit,
            args.rescore,
            args.dense_vectors,
            args.sparse_vectors
        )
        for _ in range(num_queries_in_batch)
    ]
    await asyncio.gather(*futures)

async def run_load_test(args):
    """Main function to set up the client and run the load test."""
    global CLIENT

    print("--- Configuration ---")
    print(f"Target URL:        {args.qdrant_url[:20]}...")
    print(f"Collection Name:   {args.collection_name}")
    print(f"Vector Name (for search): {args.vector_name}")
    print(f"Dense Vectors:     {args.dense_vectors}")
    print(f"Sparse Vectors:    {args.sparse_vectors}")
    print(f"Total Queries:     {args.num_queries}")
    print(f"Number of Batches: {args.num_batches}")
    print("-----------------------")

    CLIENT = AsyncQdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        prefer_grpc=args.prefer_grpc,
        timeout=args.timeout,
        check_compatibility=False,
    )

    queries_per_batch = args.num_queries // args.num_batches
    all_qps = []

    try:
        print("----- PINGING SERVER ------")
        await CLIENT.get_collections()
        print("✅ Connection successful.")

        overall_start_time = time()

        for i in range(args.num_batches):
            print(f"\n----- Starting Batch {i+1}/{args.num_batches} ({queries_per_batch} queries) ------")
            batch_start_time = time()
            
            await execute_batch(queries_per_batch, args)
            
            batch_time = time() - batch_start_time
            batch_qps = queries_per_batch / batch_time if batch_time > 0 else 0
            all_qps.append(batch_qps)
            print(f"Batch Time: {batch_time:.2f}s, QPS: {batch_qps:.2f}")

    finally:
        print("\n----- CLOSING CLIENT CONNECTION ------")
        await CLIENT.close()

    overall_time = time() - overall_start_time
    average_qps = np.mean(all_qps)
    total_qps = args.num_queries / overall_time if overall_time > 0 else 0

    print("\n----- FINAL RESULTS ------")
    print(f"Total Time:      {overall_time:.2f} seconds")
    print(f"Average QPS:     {average_qps:.2f} (from batch averages)")
    print(f"Overall QPS:     {total_qps:.2f} (from total time)")


class AddVectorAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        
        try:
            name, dim_str = values.split(':')
            dim = int(dim_str)
            getattr(namespace, self.dest)[name] = dim
        except ValueError:
            raise argparse.ArgumentError(self, f"Invalid format for {option_string}. Expected 'name:dimension', got '{values}'")


def main():
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Qdrant load testing script.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    # Client Connection Arguments
    client_group = parser.add_argument_group('Client Connection')
    client_group.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL"), help="Qdrant server URL.")
    client_group.add_argument("--qdrant-api-key", type=str, default=os.getenv("QDRANT_API_KEY"), help="Qdrant API Key.")
    client_group.add_argument("--timeout", type=float, default=10000.0, help="Request timeout in seconds.")
    client_group.add_argument("--prefer-grpc", action=argparse.BooleanOptionalAction, default=True, help="Use gRPC for communication.")

    # Test Configuration Arguments
    test_group = parser.add_argument_group('Test Configuration')
    test_group.add_argument("--collection-name", type=str, default=os.getenv("COLLECTION_NAME"), help="Name of the collection.")
    test_group.add_argument("-n", "--num-queries", type=int, default=10000, help="Total number of queries to run.")
    test_group.add_argument("-b", "--num-batches", type=int, default=1, help="Number of batches to split the queries into.")
    test_group.add_argument("-c", "--concurrency", type=int, default=250, help="Number of concurrent requests.")

    # Vector Arguments
    vector_group = parser.add_argument_group('Vector Configuration')
    vector_group.add_argument("--vector-name", type=str, default="all-MiniLM-L6-v2", help="Name of the vector to use for searching.")
    vector_group.add_argument("--vector-dimension", type=int, help="(Optional) Dimension for the default unnamed vector.")
    vector_group.add_argument(
        '--dense-vector',
        action=AddVectorAction,
        dest='dense_vectors',
        help="Define a named dense vector. Format: --dense-vector <name>:<dimension>\nCan be specified multiple times."
    )
    vector_group.add_argument(
        '--sparse-vector',
        action=AddVectorAction,
        dest='sparse_vectors',
        help="Define a named sparse vector. Format: --sparse-vector <name>:<max_dimension>\nCan be specified multiple times."
    )
    
    # Search Parameter Arguments
    search_group = parser.add_argument_group('Search Parameters')
    search_group.add_argument("--limit", type=int, default=10, help="Number of results to return per search.")
    search_group.add_argument("--rescore", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable rescoring with original vectors.")

    args = parser.parse_args()

    args.dense_vectors = args.dense_vectors or {}
    args.sparse_vectors = args.sparse_vectors or {}
    
    if args.vector_dimension:
        if args.vector_name not in args.dense_vectors:
             args.dense_vectors[args.vector_name] = args.vector_dimension

    if not args.qdrant_url or not args.qdrant_api_key or not args.collection_name:
        raise ValueError("QDRANT_URL, QDRANT_API_KEY, and COLLECTION_NAME must be provided")

    asyncio.run(run_load_test(args))

if __name__ == "__main__":
    main()
