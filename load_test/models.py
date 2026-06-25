import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CollectionConfig:
    """Mutable collection settings applied before a run."""
    quantization: Optional[str] = None      # none | scalar | binary | turbo
    bits: Optional[str] = None             # turbo: bits1 | bits1_5 | bits2 | bits4
    encoding: Optional[str] = None         # binary: two_bits | one_and_half_bits (default = 1-bit)
    query_encoding: Optional[str] = None   # binary: default | binary | scalar8bits | scalar4bits
    always_ram: Optional[bool] = None
    ef_construct: Optional[int] = None
    m: Optional[int] = None                # WARNING: changing m requires full index rebuild


@dataclass
class IndexConfig:
    """Index-time HNSW graph build parameters.

    Merged with the parent group's CollectionConfig at expansion time.
    The first SearchConfig in each IndexConfig triggers a collection rebuild
    (unless apply_config=False here or at the group level).
    """
    name: Optional[str] = None        # auto-generated from m/ef_construct if omitted
    m: Optional[int] = None
    ef_construct: Optional[int] = None
    apply_config: bool = True         # whether this index build triggers apply_config


@dataclass
class SearchConfig:
    """Search-time parameters for one run within an IndexConfig."""
    name: Optional[str] = None        # auto-generated from params if omitted
    hnsw_ef: Optional[int] = None
    oversampling: Optional[float] = None
    rescore: bool = False
    query_mode: str = "dense"         # dense | sparse | hybrid
    sparse_vector_name: Optional[str] = None
    query_filter: Optional[dict] = None
    warmup_queries: Optional[int] = None
    limit: Optional[int] = None        # per-search-param override; overrides group.limit


@dataclass
class ExperimentGroup:
    """A quantization config paired with index builds and a search param sweep.

    Expands to one ExperimentRun per (index_config × search_config) combination.
    The first SearchConfig under each IndexConfig sets apply_config=True (triggering a
    collection rebuild). Subsequent SearchConfigs reuse the rebuilt index.

    Set apply_config=False on a group to skip all rebuilds (e.g. collection already built).
    Set limit to override the global --limit for all runs in this group.
    """
    name: str
    collection_config: CollectionConfig
    index_configs: List[IndexConfig] = field(default_factory=list)
    search_params: List[SearchConfig] = field(default_factory=list)
    collection_name: Optional[str] = None
    apply_config: bool = True          # group-level override: False = never rebuild
    limit: Optional[int] = None        # top-k override; None means use global --limit


@dataclass
class ExperimentRun:
    """A single benchmarking run — the canonical runtime unit."""
    name: str
    description: str = ""
    hnsw_ef: Optional[int] = None
    oversampling: Optional[float] = None
    rescore: bool = False
    collection_name: Optional[str] = None
    apply_config: bool = False
    collection_config: Optional[CollectionConfig] = None
    query_mode: str = "dense"
    sparse_vector_name: Optional[str] = None
    query_filter: Optional[dict] = None
    warmup_queries: Optional[int] = None
    limit: Optional[int] = None        # top-k override; None means use global --limit


@dataclass
class SweepConfig:
    runs: List[ExperimentRun] = field(default_factory=list)
    stopping_metric: str = "p10"           # p01 | p05 | p10 | p50 | mean
    stopping_threshold: Optional[float] = None
    limit: Optional[int] = None            # top-k override for all runs; None means use CLI --limit


@dataclass
class RunResult:
    run: ExperimentRun
    qps: float
    p01: float
    p05: float
    p10: float
    p50: float
    mean_recall: float
    recalls: List[float]
    ann_qps_batches: List[float]
    ann_latencies: List[float]
    exact_latencies: List[float]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        d = dict(d)
        run_d = dict(d.pop("run"))
        cc_d = run_d.pop("collection_config", None)
        cc = CollectionConfig(**cc_d) if cc_d else None
        exp_run = ExperimentRun(collection_config=cc, **run_d)
        return cls(run=exp_run, **d)
