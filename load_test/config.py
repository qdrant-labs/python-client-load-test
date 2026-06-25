from typing import List

from .models import CollectionConfig, ExperimentRun, SweepConfig


def load_sweep_config(path: str) -> SweepConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)

    runs: List[ExperimentRun] = []
    for r in raw.get("runs", []):
        cc_raw = r.pop("collection_config", None)
        cc = CollectionConfig(**cc_raw) if cc_raw else None
        runs.append(ExperimentRun(
            name=r["name"],
            description=r.get("description", ""),
            hnsw_ef=r.get("hnsw_ef"),
            oversampling=r.get("oversampling"),
            rescore=r.get("rescore", False),
            collection_name=r.get("collection_name"),
            apply_config=r.get("apply_config", False),
            collection_config=cc,
        ))

    return SweepConfig(
        runs=runs,
        stopping_metric=raw.get("stopping_metric", "p10"),
        stopping_threshold=raw.get("stopping_threshold"),
    )
