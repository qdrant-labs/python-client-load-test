from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CollectionConfig:
    """Mutable collection settings that can be applied before a run."""
    quantization: Optional[str] = None      # none | scalar | binary | turbo
    bits: Optional[str] = None             # turbo only: bits1 | bits1_5 | bits2 | bits4
    encoding: Optional[str] = None         # binary only: two_bits | one_and_half_bits (default = 1-bit)
    query_encoding: Optional[str] = None   # binary only: default | binary | scalar8bits | scalar4bits
    always_ram: Optional[bool] = None
    ef_construct: Optional[int] = None
    m: Optional[int] = None                # WARNING: changing m requires full index rebuild


@dataclass
class ExperimentRun:
    name: str
    description: str = ""
    hnsw_ef: Optional[int] = None
    oversampling: Optional[float] = None
    rescore: bool = False
    collection_name: Optional[str] = None  # overrides global collection_name for this run
    apply_config: bool = False             # if True, push collection_config to Qdrant before searching
    collection_config: Optional[CollectionConfig] = None


@dataclass
class SweepConfig:
    runs: List[ExperimentRun] = field(default_factory=list)
    stopping_metric: str = "p10"           # p01 | p05 | p10 | p50 | mean
    stopping_threshold: Optional[float] = None


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
