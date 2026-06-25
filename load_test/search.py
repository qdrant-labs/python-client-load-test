import asyncio
from time import time
from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from .vectors import get_query_vector, get_sparse_query_vector


def _is_transient_error(exc: Exception) -> bool:
    """Return True for errors worth retrying (network/timeout), False for config/logic errors."""
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
        if isinstance(exc, UnexpectedResponse):
            return exc.status_code >= 500
    except ImportError:
        pass
    try:
        import grpc
        if isinstance(exc, grpc.RpcError):
            non_transient = {
                grpc.StatusCode.NOT_FOUND,
                grpc.StatusCode.INVALID_ARGUMENT,
                grpc.StatusCode.PERMISSION_DENIED,
                grpc.StatusCode.UNAUTHENTICATED,
            }
            return exc.code() not in non_transient
    except ImportError:
        pass
    return True


def build_filter(query_filter: Optional[dict]) -> Optional[models.Filter]:
    """Convert a raw dict to models.Filter (supports pydantic v1 and v2)."""
    if not query_filter:
        return None
    try:
        return models.Filter.model_validate(query_filter)
    except AttributeError:
        return models.Filter.parse_obj(query_filter)


async def search_single(
    client: AsyncQdrantClient,
    query_vector: list[float],
    collection_name: str,
    limit: int,
    vector_name: Optional[str],
    search_params: Optional[models.SearchParams],
    semaphore: asyncio.Semaphore,
    *,
    max_retries: int = 3,
    query_filter: Optional[models.Filter] = None,
    query_mode: str = "dense",
    sparse_query_vector: Optional[dict] = None,
    sparse_vector_name: Optional[str] = None,
) -> Tuple[float, List]:
    """Execute a single query with bounded retry for transient errors.

    query_mode:
      dense   — single dense query_points call (existing behaviour)
      sparse  — single sparse query_points call (requires sparse_query_vector)
      hybrid  — prefetch dense + sparse, fuse with RRF (requires sparse_query_vector)
    """
    if query_mode in ("sparse", "hybrid") and sparse_query_vector is None:
        raise ValueError(f"sparse_query_vector is required for query_mode='{query_mode}'")

    is_exact = search_params is not None and getattr(search_params, "exact", False)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                start = time()
                if query_mode == "dense":
                    results = await client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        using=vector_name,
                        limit=limit,
                        search_params=search_params,
                        query_filter=query_filter,
                    )
                elif query_mode == "sparse":
                    sv = models.SparseVector(
                        indices=sparse_query_vector["indices"],
                        values=sparse_query_vector["values"],
                    )
                    results = await client.query_points(
                        collection_name=collection_name,
                        query=sv,
                        using=sparse_vector_name,
                        limit=limit,
                        search_params=search_params,
                        query_filter=query_filter,
                    )
                elif query_mode == "hybrid":
                    sv = models.SparseVector(
                        indices=sparse_query_vector["indices"],
                        values=sparse_query_vector["values"],
                    )
                    prefetch_params = models.SearchParams(exact=True) if is_exact else None
                    prefetch_limit = limit * 4
                    results = await client.query_points(
                        collection_name=collection_name,
                        prefetch=[
                            models.Prefetch(
                                query=query_vector,
                                using=vector_name,
                                limit=prefetch_limit,
                                params=prefetch_params,
                            ),
                            models.Prefetch(
                                query=sv,
                                using=sparse_vector_name,
                                limit=prefetch_limit,
                                params=prefetch_params,
                            ),
                        ],
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        limit=limit,
                        query_filter=query_filter,
                    )
                else:
                    raise ValueError(f"Unknown query_mode: {query_mode!r}. Use: dense sparse hybrid")
                latency = time() - start
            return latency, [r.id for r in results.points]
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            last_exc = exc
            if attempt < max_retries - 1:
                delay = 0.1 * (2 ** attempt)
                print(f"    ⚠️  Transient error (attempt {attempt + 1}/{max_retries}): {exc}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

    raise RuntimeError(f"All {max_retries} retries exhausted") from last_exc


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
    query_mode: str = "dense",
    sparse_query_vectors: Optional[list] = None,
    sparse_vector_name: Optional[str] = None,
    query_filter: Optional[models.Filter] = None,
    max_retries: int = 3,
    quiet: bool = False,
) -> Tuple[List[float], List[List], float, List[float]]:
    """Run a search phase. Returns (batch_qps, per_query_results, overall_time, latencies)."""
    queries_per_batch = len(query_indices) // num_batches
    remainder = len(query_indices) % num_batches

    all_qps: List[float] = []
    all_results: List[List] = []
    all_latencies: List[float] = []

    if not quiet:
        print(f"\n{'='*55}")
        print(f"  PHASE: {phase_name}")
        print(f"{'='*55}")

    overall_start = time()
    offset = 0

    for i in range(num_batches):
        batch_size = queries_per_batch + (1 if i < remainder else 0)
        batch_indices = query_indices[offset: offset + batch_size]
        offset += batch_size

        if not quiet:
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
                max_retries=max_retries,
                query_filter=query_filter,
                query_mode=query_mode,
                sparse_query_vector=(
                    get_sparse_query_vector(idx, sparse_query_vectors)
                    if sparse_query_vectors is not None else None
                ),
                sparse_vector_name=sparse_vector_name,
            )
            for idx in batch_indices
        ]
        batch_outputs = await asyncio.gather(*tasks)

        batch_time = time() - batch_start
        batch_qps = batch_size / batch_time if batch_time > 0 else 0
        all_qps.append(batch_qps)
        if not quiet:
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
