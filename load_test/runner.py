import asyncio
from typing import Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import AsyncQdrantClient

from .collection import apply_collection_config
from .metrics import compute_metric, compute_recall
from .models import RunResult, SweepConfig
from .reports import _search_label, generate_html_report, generate_sweep_html_report
from .search import build_search_params, run_phase


async def run_load_test(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str,
    vector_name: Optional[str],
    num_queries: int,
    num_batches: int,
    concurrency: int,
    limit: int,
    rescore: bool,
    prefer_grpc: bool,
    timeout: float,
    output: str,
    vector_dimension: Optional[int] = None,
    query_vectors: Optional[np.ndarray] = None,
    query_source: str = "Random vectors",
):
    print("\n--- Configuration ---")
    print(f"Target URL:        {qdrant_url[:60]}{'...' if len(qdrant_url) > 60 else ''}")
    print(f"Collection:        {collection_name}")
    print(f"Vector Name:       {vector_name or '<default>'}")
    print(f"Total Queries:     {num_queries}")
    print(f"Batches:           {num_batches}")
    print(f"Concurrency:       {concurrency}")
    print(f"Limit (top-k):     {limit}")
    print(f"Query Source:      {query_source}")
    print("-" * 23)

    client = AsyncQdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=prefer_grpc,
        timeout=timeout,
        check_compatibility=False,
    )

    semaphore = asyncio.Semaphore(concurrency)
    query_indices = list(range(num_queries))
    ann_params = build_search_params(rescore=rescore)
    exact_params = build_search_params(exact=True)

    try:
        print("\n----- PINGING SERVER ------")
        await client.get_collections()
        print("✅ Connection successful.")

        ann_qps, ann_results, ann_time, ann_latencies = await run_phase(
            phase_name="Phase 1: ANN Search",
            client=client,
            collection_name=collection_name,
            limit=limit,
            vector_name=vector_name,
            num_batches=num_batches,
            vector_dimension=vector_dimension,
            query_indices=query_indices,
            search_params=ann_params,
            semaphore=semaphore,
            query_vectors=query_vectors,
        )

        exact_qps, exact_results, exact_time, exact_latencies = await run_phase(
            phase_name="Phase 2: Exact / Brute-Force Search (exact=True)",
            client=client,
            collection_name=collection_name,
            limit=limit,
            vector_name=vector_name,
            num_batches=num_batches,
            vector_dimension=vector_dimension,
            query_indices=query_indices,
            search_params=exact_params,
            semaphore=semaphore,
            query_vectors=query_vectors,
        )

    finally:
        print("\n----- CLOSING CLIENT CONNECTION ------")
        await client.close()

    recalls = compute_recall(ann_results, exact_results)
    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    print("\n" + "=" * 55)
    print("  FINAL RESULTS")
    print("=" * 55)
    print(f"Phase 1 (ANN)   — Avg QPS: {np.mean(ann_qps):.2f}  |  Total time: {ann_time:.2f}s")
    print(f"Phase 2 (Exact) — Avg QPS: {np.mean(exact_qps):.2f}  |  Total time: {exact_time:.2f}s")
    p10 = compute_metric(recalls, "p10")
    p99 = compute_metric(recalls, "p99")
    print(f"Recall@{limit}         — Mean: {mean_recall*100:.2f}%  |  p10: {p10*100:.2f}%  |  p99: {p99*100:.2f}%")

    num_embeddings = len(query_vectors) if query_vectors is not None else "N/A"
    generate_html_report(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        vector_name=vector_name,
        num_queries=num_queries,
        num_batches=num_batches,
        concurrency=concurrency,
        limit=limit,
        rescore=rescore,
        prefer_grpc=prefer_grpc,
        output_path=output,
        ann_qps=ann_qps,
        exact_qps=exact_qps,
        ann_time=ann_time,
        exact_time=exact_time,
        ann_latencies=ann_latencies,
        exact_latencies=exact_latencies,
        recalls=recalls,
        query_source=query_source,
        num_embeddings=num_embeddings,
    )


async def run_sweep(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str,
    vector_name: Optional[str],
    num_queries: int,
    num_batches: int,
    concurrency: int,
    limit: int,
    prefer_grpc: bool,
    timeout: float,
    output: str,
    sweep: SweepConfig,
    vector_dimension: Optional[int] = None,
    query_vectors: Optional[np.ndarray] = None,
):
    print("\n--- Sweep Configuration ---")
    print(f"Target URL:     {qdrant_url[:60]}{'...' if len(qdrant_url) > 60 else ''}")
    print(f"Collection:     {collection_name}")
    print(f"Queries/run:    {num_queries}  |  Limit: {limit}  |  Concurrency: {concurrency}")
    print(f"Runs planned:   {len(sweep.runs)}")
    print(f"Stop when:      {sweep.stopping_metric} >= {(sweep.stopping_threshold or 0)*100:.1f}%")
    print("-" * 27)

    client = AsyncQdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=prefer_grpc,
        timeout=timeout,
        check_compatibility=False,
    )

    semaphore = asyncio.Semaphore(concurrency)
    query_indices = list(range(num_queries))
    exact_params = build_search_params(exact=True)

    # Brute-force cache: collection_name -> (results, latencies)
    exact_cache: Dict[str, Tuple[List[List], List[float]]] = {}
    results: List[RunResult] = []
    stopping_run_index: Optional[int] = None

    try:
        print("\n----- PINGING SERVER ------")
        await client.get_collections()
        print("✅ Connection successful.")

        all_collections = {collection_name}
        for run in sweep.runs:
            if run.collection_name:
                all_collections.add(run.collection_name)

        for coll_name in sorted(all_collections):
            _, exact_results, _, exact_latencies = await run_phase(
                phase_name=f"Brute Force (exact=True) — {coll_name}",
                client=client,
                collection_name=coll_name,
                limit=limit,
                vector_name=vector_name,
                num_batches=num_batches,
                vector_dimension=vector_dimension,
                query_indices=query_indices,
                search_params=exact_params,
                semaphore=semaphore,
                query_vectors=query_vectors,
            )
            exact_cache[coll_name] = (exact_results, exact_latencies)

        for run_idx, exp_run in enumerate(sweep.runs):
            coll_name = exp_run.collection_name or collection_name

            print(f"\n{'#'*60}")
            print(f"  RUN {run_idx+1}/{len(sweep.runs)}: {exp_run.name}")
            if exp_run.description:
                print(f"  {exp_run.description}")
            print(f"  Collection: {coll_name}  |  {_search_label(exp_run)}")
            print(f"{'#'*60}")

            if exp_run.apply_config and exp_run.collection_config:
                await apply_collection_config(client, coll_name, exp_run.collection_config)

            ann_params = build_search_params(
                hnsw_ef=exp_run.hnsw_ef,
                oversampling=exp_run.oversampling,
                rescore=exp_run.rescore,
            )

            ann_qps, ann_results, _, ann_latencies = await run_phase(
                phase_name=f"ANN — {exp_run.name}",
                client=client,
                collection_name=coll_name,
                limit=limit,
                vector_name=vector_name,
                num_batches=num_batches,
                vector_dimension=vector_dimension,
                query_indices=query_indices,
                search_params=ann_params,
                semaphore=semaphore,
                query_vectors=query_vectors,
            )

            exact_results, exact_latencies = exact_cache[coll_name]
            recalls = compute_recall(ann_results, exact_results)

            p01 = compute_metric(recalls, "p01")
            p05 = compute_metric(recalls, "p05")
            p10 = compute_metric(recalls, "p10")
            p50 = compute_metric(recalls, "p50")
            mean_r = float(np.mean(recalls)) if recalls else 0.0
            mean_qps = float(np.mean(ann_qps)) if ann_qps else 0.0

            print(f"\n  Recall@{limit}: p1={p01*100:.1f}%  p5={p05*100:.1f}%  p10={p10*100:.1f}%  p50={p50*100:.1f}%  mean={mean_r*100:.1f}%")
            print(f"  QPS: {mean_qps:.1f}")

            results.append(RunResult(
                run=exp_run,
                qps=mean_qps,
                p01=p01, p05=p05, p10=p10, p50=p50,
                mean_recall=mean_r,
                recalls=recalls,
                ann_qps_batches=ann_qps,
                ann_latencies=ann_latencies,
                exact_latencies=exact_latencies,
            ))

            if sweep.stopping_threshold is not None:
                stop_val = compute_metric(recalls, sweep.stopping_metric)
                print(f"  Stop check: {sweep.stopping_metric} = {stop_val*100:.2f}% (need >= {sweep.stopping_threshold*100:.1f}%)")
                if stop_val >= sweep.stopping_threshold:
                    print(f"\n✅ Stopping condition met at run '{exp_run.name}'")
                    stopping_run_index = run_idx
                    break

        if stopping_run_index is None and sweep.stopping_threshold is not None:
            final = compute_metric(results[-1].recalls, sweep.stopping_metric) if results else 0.0
            print(
                f"\n⚠️  All {len(results)} run(s) completed — stop condition not met "
                f"({sweep.stopping_metric}={final*100:.2f}% < {sweep.stopping_threshold*100:.1f}%)"
            )

    finally:
        print("\n----- CLOSING CLIENT CONNECTION ------")
        await client.close()

    generate_sweep_html_report(
        results=results,
        sweep=sweep,
        stopping_run_index=stopping_run_index,
        collection_name=collection_name,
        num_queries=num_queries,
        limit=limit,
        concurrency=concurrency,
        output_path=output,
    )
