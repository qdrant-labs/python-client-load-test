from typing import List

from .models import (
    CollectionConfig,
    ExperimentGroup,
    ExperimentRun,
    IndexConfig,
    SearchConfig,
    SweepConfig,
)


# ── Name auto-generation ──────────────────────────────────────────────────────

def _index_name(ic: IndexConfig) -> str:
    """Auto-generate a name from HNSW graph params when none is given."""
    parts = []
    if ic.m is not None:
        parts.append(f"m{ic.m}")
    if ic.ef_construct is not None:
        parts.append(f"efc{ic.ef_construct}")
    return "-".join(parts) or "default"


def _search_name(sc: SearchConfig) -> str:
    """Auto-generate a name from search-time params when none is given."""
    parts = []
    if sc.limit is not None:
        parts.append(f"lim{sc.limit}")
    if sc.hnsw_ef is not None:
        parts.append(f"ef{sc.hnsw_ef}")
    if sc.rescore:
        parts.append("rescore")
    if sc.oversampling is not None:
        # drop trailing .0 for whole numbers
        os_str = str(int(sc.oversampling)) if sc.oversampling == int(sc.oversampling) else str(sc.oversampling)
        parts.append(f"os{os_str}x")
    if sc.query_mode != "dense":
        parts.append(sc.query_mode)
    return "-".join(parts) or "default"


# ── Group expansion ───────────────────────────────────────────────────────────

def _expand_group(group: ExperimentGroup) -> List[ExperimentRun]:
    """Expand one ExperimentGroup into a flat list of ExperimentRuns.

    For each (index_config × search_config) pair:
    - The first search_config under each index_config sets apply_config=True
      (triggering a collection rebuild with merged quantization + HNSW graph params).
    - All subsequent search_configs in that index_config set apply_config=False
      (reusing the rebuilt index, only search-time params vary).
    - If group.apply_config=False, no run in the group ever triggers a rebuild.
    """
    runs = []
    # An empty index_configs list means "no HNSW rebuild needed, one implicit index".
    index_configs = group.index_configs if group.index_configs else [IndexConfig()]

    for ic in index_configs:
        ic_label = ic.name or _index_name(ic)

        for sp_idx, sc in enumerate(group.search_params):
            sc_label = sc.name or _search_name(sc)
            run_name = f"{group.name}/{ic_label}/{sc_label}"

            # Merge: group quantization settings + this index's HNSW graph params.
            # The merged CollectionConfig is stored on every run so reports can
            # always show the full active config (even for apply_config=False runs).
            cc = CollectionConfig(
                quantization=group.collection_config.quantization,
                bits=group.collection_config.bits,
                encoding=group.collection_config.encoding,
                query_encoding=group.collection_config.query_encoding,
                always_ram=group.collection_config.always_ram,
                ef_construct=ic.ef_construct,
                m=ic.m,
            )

            # Only the first search_config under each index_config rebuilds,
            # and only if both group.apply_config and ic.apply_config allow it.
            apply = group.apply_config and ic.apply_config and (sp_idx == 0)

            # Resolve limit: search_param override > group default > None (use global --limit)
            effective_limit = sc.limit if sc.limit is not None else group.limit

            runs.append(ExperimentRun(
                name=run_name,
                hnsw_ef=sc.hnsw_ef,
                oversampling=sc.oversampling,
                rescore=sc.rescore,
                collection_name=group.collection_name,
                apply_config=apply,
                collection_config=cc,
                query_mode=sc.query_mode,
                sparse_vector_name=sc.sparse_vector_name,
                query_filter=sc.query_filter,
                warmup_queries=sc.warmup_queries,
                limit=effective_limit,
            ))

    return runs


# ── YAML parsing ──────────────────────────────────────────────────────────────

def _parse_index_config(raw: dict) -> IndexConfig:
    return IndexConfig(
        name=raw.get("name"),
        m=raw.get("m"),
        ef_construct=raw.get("ef_construct"),
        apply_config=raw.get("apply_config", True),
    )


def _parse_search_config(raw: dict) -> SearchConfig:
    return SearchConfig(
        name=raw.get("name"),
        hnsw_ef=raw.get("hnsw_ef"),
        oversampling=raw.get("oversampling"),
        rescore=raw.get("rescore", False),
        query_mode=raw.get("query_mode", "dense"),
        sparse_vector_name=raw.get("sparse_vector_name"),
        query_filter=raw.get("query_filter"),
        warmup_queries=raw.get("warmup_queries"),
        limit=raw.get("limit"),
    )


def _parse_group(raw: dict) -> ExperimentGroup:
    cc_raw = raw.get("collection_config", {})
    cc_fields = set(CollectionConfig.__dataclass_fields__)
    cc = CollectionConfig(**{k: v for k, v in cc_raw.items() if k in cc_fields})

    return ExperimentGroup(
        name=raw["name"],
        collection_config=cc,
        index_configs=[_parse_index_config(ic) for ic in raw.get("index_configs", [])],
        search_params=[_parse_search_config(sp) for sp in raw.get("search_params", [{}])],
        collection_name=raw.get("collection_name"),
        apply_config=raw.get("apply_config", True),
        limit=raw.get("limit"),
    )


def _parse_flat_run(raw: dict) -> ExperimentRun:
    raw = dict(raw)
    cc_raw = raw.pop("collection_config", None)
    cc = CollectionConfig(**cc_raw) if cc_raw else None
    return ExperimentRun(
        name=raw["name"],
        description=raw.get("description", ""),
        hnsw_ef=raw.get("hnsw_ef"),
        oversampling=raw.get("oversampling"),
        rescore=raw.get("rescore", False),
        collection_name=raw.get("collection_name"),
        apply_config=raw.get("apply_config", False),
        collection_config=cc,
        query_mode=raw.get("query_mode", "dense"),
        sparse_vector_name=raw.get("sparse_vector_name"),
        query_filter=raw.get("query_filter"),
        warmup_queries=raw.get("warmup_queries"),
        limit=raw.get("limit"),
    )


def load_sweep_config(path: str) -> SweepConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)

    all_runs: List[ExperimentRun] = []

    # New format: groups expand into runs automatically.
    for grp_raw in raw.get("groups", []):
        all_runs.extend(_expand_group(_parse_group(grp_raw)))

    # Legacy / custom format: flat run list (backward-compatible).
    # Appended after group-generated runs so they appear at the end of the sweep.
    for r in raw.get("runs", []):
        all_runs.append(_parse_flat_run(r))

    return SweepConfig(
        runs=all_runs,
        stopping_metric=raw.get("stopping_metric", "p10"),
        stopping_threshold=raw.get("stopping_threshold"),
        limit=raw.get("limit"),
    )
