import asyncio
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from .checkpoint import (
    append_run_result,
    checkpoint_paths,
    load_exact_cache,
    load_run_results,
    save_exact_cache,
)
from .collection import apply_collection_config
from .metrics import compute_metric, compute_recall
from .models import ExperimentRun, RunResult, SweepConfig
from .reports import _search_label, generate_html_report, generate_sweep_html_report
from .search import build_filter, build_search_params, run_phase


def _make_cache_key(
    coll_name: str,
    vector_name: Optional[str],
    limit: int,
    query_filter: Optional[dict],
    query_mode: str,
    sparse_vector_name: Optional[str],
) -> tuple:
    """Build a stable cache key covering all dimensions that affect brute-force results."""
    filter_hash = ""
    if query_filter:
        filter_hash = hashlib.md5(
            json.dumps(query_filter, sort_keys=True).encode()
        ).hexdigest()[:8]
    sparse_name = (sparse_vector_name or "") if query_mode in ("sparse", "hybrid") else ""
    return (coll_name, vector_name or "", limit, filter_hash, query_mode, sparse_name)


def _check_wrap(
    num_queries: int,
    query_vectors: Optional[np.ndarray],
    sparse_query_vectors: Optional[list],
    allow_wrap: bool,
) -> None:
    """Print warning (or raise) when queries will repeat due to wrap-around."""
    sources = []
    if query_vectors is not None and num_queries > len(query_vectors):
        sources.append(f"dense embeddings: {num_queries} queries requested, {len(query_vectors)} available")
    if sparse_query_vectors is not None and num_queries > len(sparse_query_vectors):
        sources.append(f"sparse embeddings: {num_queries} queries requested, {len(sparse_query_vectors)} available")
    if not sources:
        return
    msg = "⚠️  Wrap warning — queries will repeat (modulo wrap):\n" + "\n".join(f"     {s}" for s in sources)
    print(msg)
    if not allow_wrap:
        raise ValueError(
            "--no-allow-wrap is set but query count exceeds available embeddings.\n"
            + "\n".join(sources)
        )


async def _run_single_experiment(
    client: AsyncQdrantClient,
    exp_run: ExperimentRun,
    *,
    collection_name: str,
    vector_name: Optional[str],
    limit: int,
    num_batches: int,
    vector_dimension: Optional[int],
    query_indices: List[int],
    semaphore: asyncio.Semaphore,
    query_vectors: Optional[np.ndarray],
    sparse_query_vectors: Optional[list],
    exact_cache: Dict,
    warmup_queries: int,
    max_retries: int,
) -> RunResult:
    """Run a single sweep experiment: optional config apply, optional warmup, ANN, recall."""
    coll_name = exp_run.collection_name or collection_name
    effective_limit = exp_run.limit if exp_run.limit is not None else limit

    if exp_run.apply_config and exp_run.collection_config:
        await apply_collection_config(client, coll_name, exp_run.collection_config)

    ann_params = build_search_params(
        hnsw_ef=exp_run.hnsw_ef,
        oversampling=exp_run.oversampling,
        rescore=exp_run.rescore,
    )

    effective_warmup = exp_run.warmup_queries if exp_run.warmup_queries is not None else warmup_queries
    qfilter = build_filter(exp_run.query_filter)
    qmode = exp_run.query_mode
    sparse_vname = exp_run.sparse_vector_name

    if effective_warmup > 0:
        warmup_indices = list(range(effective_warmup))
        print(f"  🔥 Warmup: {effective_warmup} queries (results discarded)...")
        await run_phase(
            phase_name=f"Warmup — {exp_run.name}",
            client=client,
            collection_name=coll_name,
            limit=effective_limit,
            vector_name=vector_name,
            num_batches=1,
            vector_dimension=vector_dimension,
            query_indices=warmup_indices,
            search_params=ann_params,
            semaphore=semaphore,
            query_vectors=query_vectors,
            query_mode=qmode,
            sparse_query_vectors=sparse_query_vectors,
            sparse_vector_name=sparse_vname,
            query_filter=qfilter,
            max_retries=max_retries,
            quiet=True,
        )

    ann_qps, ann_results, _, ann_latencies = await run_phase(
        phase_name=f"ANN — {exp_run.name}",
        client=client,
        collection_name=coll_name,
        limit=effective_limit,
        vector_name=vector_name,
        num_batches=num_batches,
        vector_dimension=vector_dimension,
        query_indices=query_indices,
        search_params=ann_params,
        semaphore=semaphore,
        query_vectors=query_vectors,
        query_mode=qmode,
        sparse_query_vectors=sparse_query_vectors,
        sparse_vector_name=sparse_vname,
        query_filter=qfilter,
        max_retries=max_retries,
    )

    cache_key = _make_cache_key(coll_name, vector_name, effective_limit, exp_run.query_filter, qmode, sparse_vname)
    exact_results, exact_latencies = exact_cache[cache_key]
    recalls = compute_recall(ann_results, exact_results)

    p01 = compute_metric(recalls, "p01")
    p05 = compute_metric(recalls, "p05")
    p10 = compute_metric(recalls, "p10")
    p50 = compute_metric(recalls, "p50")
    mean_r = float(np.mean(recalls)) if recalls else 0.0
    mean_qps = float(np.mean(ann_qps)) if ann_qps else 0.0

    print(f"\n  Recall@{effective_limit}: p1={p01*100:.1f}%  p5={p05*100:.1f}%  p10={p10*100:.1f}%  p50={p50*100:.1f}%  mean={mean_r*100:.1f}%")
    print(f"  QPS: {mean_qps:.1f}")

    return RunResult(
        run=exp_run,
        qps=mean_qps,
        p01=p01, p05=p05, p10=p10, p50=p50,
        mean_recall=mean_r,
        recalls=recalls,
        ann_qps_batches=ann_qps,
        ann_latencies=ann_latencies,
        exact_latencies=exact_latencies,
    )


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
    allow_wrap: bool = True,
    warmup_queries: int = 0,
    max_retries: int = 3,
    sparse_query_vectors: Optional[list] = None,
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

    _check_wrap(num_queries, query_vectors, sparse_query_vectors, allow_wrap)

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

        if warmup_queries > 0:
            print(f"\n🔥 Warmup: {warmup_queries} queries (results discarded)...")
            await run_phase(
                phase_name="Warmup",
                client=client,
                collection_name=collection_name,
                limit=limit,
                vector_name=vector_name,
                num_batches=1,
                vector_dimension=vector_dimension,
                query_indices=list(range(warmup_queries)),
                search_params=ann_params,
                semaphore=semaphore,
                query_vectors=query_vectors,
                max_retries=max_retries,
                quiet=True,
            )

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
            max_retries=max_retries,
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
            max_retries=max_retries,
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
    resume: bool = False,
    checkpoint_file: Optional[str] = None,
    allow_wrap: bool = True,
    warmup_queries: int = 200,
    max_retries: int = 3,
    sparse_query_vectors: Optional[list] = None,
):
    # YAML limit (sweep.limit) overrides the CLI --limit when set.
    # Per-group and per-search-param limits override this in turn.
    global_limit = sweep.limit if sweep.limit is not None else limit

    print("\n--- Sweep Configuration ---")
    print(f"Target URL:     {qdrant_url[:60]}{'...' if len(qdrant_url) > 60 else ''}")
    print(f"Collection:     {collection_name}")
    print(f"Queries/run:    {num_queries}  |  Limit: {global_limit}  |  Concurrency: {concurrency}")
    print(f"Runs planned:   {len(sweep.runs)}")
    print(f"Stop when:      {sweep.stopping_metric} >= {(sweep.stopping_threshold or 0)*100:.1f}%")
    print(f"Warmup/run:     {warmup_queries}")
    print("-" * 27)

    _check_wrap(num_queries, query_vectors, sparse_query_vectors, allow_wrap)

    # ── Checkpoint setup ────────────────────────────────────────────────────────
    ckpt_jsonl, ckpt_cache_json = checkpoint_paths(output, checkpoint_file)

    completed_names: set = set()
    if resume and os.path.exists(ckpt_jsonl):
        prior_results = load_run_results(ckpt_jsonl)
        results: List[RunResult] = list(prior_results)
        completed_names = {r.run.name for r in results}
        print(f"⏩ Resuming: loaded {len(results)} completed run(s) from '{ckpt_jsonl}'.")
    else:
        if not resume and os.path.exists(ckpt_jsonl):
            print(f"⚠️  Checkpoint exists at '{ckpt_jsonl}' but --resume was not set. Starting fresh (checkpoint will be overwritten).")
        results = []

    exact_cache: Dict[tuple, Tuple[List[List], List[float]]] = {}
    if resume and os.path.exists(ckpt_cache_json):
        exact_cache = load_exact_cache(ckpt_cache_json)
        print(f"   Loaded {len(exact_cache)} exact-cache entries from '{ckpt_cache_json}'.")

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
    stopping_run_index: Optional[int] = None

    try:
        print("\n----- PINGING SERVER ------")
        await client.get_collections()
        print("✅ Connection successful.")

        # ── Task 10: Pre-flight collection validation ───────────────────────────
        all_collection_names = {collection_name}
        for run in sweep.runs:
            if run.collection_name:
                all_collection_names.add(run.collection_name)

        bad_collections = []
        not_green = []
        for coll_name in sorted(all_collection_names):
            try:
                info = await client.get_collection(coll_name)
                if info.status != qdrant_models.CollectionStatus.GREEN:
                    not_green.append(f"{coll_name} (status: {info.status.value})")
            except Exception:
                bad_collections.append(coll_name)

        if bad_collections:
            raise ValueError(
                f"Pre-flight check failed — collections not found or unreachable: {bad_collections}\n"
                "Fix the collection names in your sweep config before retrying."
            )
        if not_green:
            for entry in not_green:
                print(f"⚠️  Collection not GREEN: {entry} — the sweep will proceed but index may be incomplete.")

        # ── Build unique brute-force cache keys ─────────────────────────────────
        # key -> (coll_name, query_filter_model, query_mode, sparse_vector_name)
        # Each key includes the effective limit so runs with different limits get
        # independent brute-force ground-truth sets.
        brute_force_configs: Dict[tuple, dict] = {}
        for run in sweep.runs:
            coll = run.collection_name or collection_name
            run_limit = run.limit if run.limit is not None else global_limit
            key = _make_cache_key(coll, vector_name, run_limit, run.query_filter, run.query_mode, run.sparse_vector_name)
            if key not in brute_force_configs:
                brute_force_configs[key] = {
                    "coll_name": coll,
                    "limit": run_limit,
                    "query_filter": build_filter(run.query_filter),
                    "query_mode": run.query_mode,
                    "sparse_vector_name": run.sparse_vector_name,
                }

        for cache_key, cfg in sorted(brute_force_configs.items(), key=lambda x: str(x[0])):
            if cache_key in exact_cache:
                print(f"⏭️  Skipping brute-force for {cache_key[0]!r} ({cfg['query_mode']}) — loaded from checkpoint.")
                continue

            if cfg["query_mode"] in ("sparse", "hybrid") and sparse_query_vectors is None:
                raise ValueError(
                    f"Run with query_mode='{cfg['query_mode']}' requires --sparse-parquet-file. "
                    "Provide sparse embeddings via --sparse-parquet-file."
                )

            _, exact_results, _, exact_latencies = await run_phase(
                phase_name=f"Brute Force — {cfg['coll_name']} ({cfg['query_mode']}) limit={cfg['limit']}",
                client=client,
                collection_name=cfg["coll_name"],
                limit=cfg["limit"],
                vector_name=vector_name,
                num_batches=num_batches,
                vector_dimension=vector_dimension,
                query_indices=query_indices,
                search_params=exact_params,
                semaphore=semaphore,
                query_vectors=query_vectors,
                query_mode=cfg["query_mode"],
                sparse_query_vectors=sparse_query_vectors,
                sparse_vector_name=cfg["sparse_vector_name"],
                query_filter=cfg["query_filter"],
                max_retries=max_retries,
            )
            exact_cache[cache_key] = (exact_results, exact_latencies)
            save_exact_cache(ckpt_cache_json, exact_cache)

        # ── Per-run loop ─────────────────────────────────────────────────────────
        for run_idx, exp_run in enumerate(sweep.runs):
            if exp_run.name in completed_names:
                print(f"\n⏭️  Skipping '{exp_run.name}' (already completed, loaded from checkpoint)")
                continue

            coll_name = exp_run.collection_name or collection_name
            print(f"\n{'#'*60}")
            print(f"  RUN {run_idx+1}/{len(sweep.runs)}: {exp_run.name}")
            if exp_run.description:
                print(f"  {exp_run.description}")
            print(f"  Collection: {coll_name}  |  {_search_label(exp_run)}")
            print(f"{'#'*60}")

            result = await _run_single_experiment(
                client=client,
                exp_run=exp_run,
                collection_name=collection_name,
                vector_name=vector_name,
                limit=global_limit,
                num_batches=num_batches,
                vector_dimension=vector_dimension,
                query_indices=query_indices,
                semaphore=semaphore,
                query_vectors=query_vectors,
                sparse_query_vectors=sparse_query_vectors,
                exact_cache=exact_cache,
                warmup_queries=warmup_queries,
                max_retries=max_retries,
            )

            results.append(result)
            append_run_result(ckpt_jsonl, result)

            if sweep.stopping_threshold is not None:
                stop_val = compute_metric(result.recalls, sweep.stopping_metric)
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
        limit=global_limit,
        concurrency=concurrency,
        output_path=output,
    )
