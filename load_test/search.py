import asyncio
from time import time
from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from .vectors import get_query_vector


async def search_single(
    client: AsyncQdrantClient,
    query_vector: list[float],
    collection_name: str,
    limit: int,
    vector_name: Optional[str],
    search_params: Optional[models.SearchParams],
    semaphore: asyncio.Semaphore,
) -> Tuple[float, List]:
    async with semaphore:
        start = time()
        results = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            search_params=search_params,
        )
        latency = time() - start
    return latency, [r.id for r in results.points]


async def run_phase(
    phase_name: str,
    client: AsyncQdrantClient,
    collection_name: str,
    limit: int,
    vector_name: Optional[str],
    num_batches: int,
    vector_dimension: Optional[int],
    query_indices: List[int],
    search_params: Optional[models.SearchParams],
    semaphore: asyncio.Semaphore,
    query_vectors: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[List], float, List[float]]:
    """Run a search phase. Returns (batch_qps, per_query_results, overall_time, latencies)."""
    queries_per_batch = len(query_indices) // num_batches
    remainder = len(query_indices) % num_batches

    all_qps: List[float] = []
    all_results: List[List] = []
    all_latencies: List[float] = []

    print(f"\n{'='*55}")
    print(f"  PHASE: {phase_name}")
    print(f"{'='*55}")

    overall_start = time()
    offset = 0

    for i in range(num_batches):
        batch_size = queries_per_batch + (1 if i < remainder else 0)
        batch_indices = query_indices[offset: offset + batch_size]
        offset += batch_size

        print(f"\n----- Batch {i+1}/{num_batches} ({batch_size} queries) ------")
        batch_start = time()

        tasks = [
            search_single(
                client,
                get_query_vector(idx, vector_dimension, query_vectors),
                collection_name,
                limit,
                vector_name,
                search_params,
                semaphore,
            )
            for idx in batch_indices
        ]
        batch_outputs = await asyncio.gather(*tasks)

        batch_time = time() - batch_start
        batch_qps = batch_size / batch_time if batch_time > 0 else 0
        all_qps.append(batch_qps)
        print(f"Batch Time: {batch_time:.2f}s, QPS: {batch_qps:.2f}")

        for latency, result_ids in batch_outputs:
            all_latencies.append(latency)
            all_results.append(result_ids)

    overall_time = time() - overall_start
    return all_qps, all_results, overall_time, all_latencies


def build_search_params(
    hnsw_ef: Optional[int] = None,
    oversampling: Optional[float] = None,
    rescore: bool = False,
    exact: bool = False,
) -> Optional[models.SearchParams]:
    if exact:
        return models.SearchParams(exact=True)
    quant_params = None
    if rescore or oversampling is not None:
        quant_params = models.QuantizationSearchParams(rescore=rescore, oversampling=oversampling)
    if hnsw_ef is None and quant_params is None:
        return None
    return models.SearchParams(hnsw_ef=hnsw_ef, quantization=quant_params)
