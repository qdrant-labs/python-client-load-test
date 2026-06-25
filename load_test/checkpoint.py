from __future__ import annotations

import json
import os
from typing import Optional

from .models import RunResult


def checkpoint_paths(output_path: str, checkpoint_file: Optional[str]) -> tuple[str, str]:
    """Return (run_results_path, exact_cache_path). Derive from output_path if checkpoint_file is None."""
    if checkpoint_file:
        base = checkpoint_file
    else:
        dot = output_path.rfind(".")
        base = output_path[:dot] if dot > 0 else output_path
    return f"{base}.checkpoint.jsonl", f"{base}.exact_cache.json"


def append_run_result(path: str, result: RunResult) -> None:
    """Append one JSON object per line (JSONL). Flush + fsync so a crash can't lose the line."""
    with open(path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_run_results(path: str) -> list[RunResult]:
    """Read JSONL; return [] if file missing. Tolerate a truncated final line."""
    if not os.path.exists(path):
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(RunResult.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # skip malformed / truncated last line
    return results


def _encode_cache_key(key: tuple) -> str:
    return json.dumps(list(key))


def _decode_cache_key(s: str) -> tuple:
    return tuple(json.loads(s))


def save_exact_cache(path: str, cache: dict) -> None:
    """Persist the brute-force cache. Keys are tuples encoded as JSON arrays."""
    serializable = {}
    for k, v in cache.items():
        exact_results, exact_latencies = v
        serializable[_encode_cache_key(k)] = {
            "results": exact_results,
            "latencies": exact_latencies,
        }
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_exact_cache(path: str) -> dict:
    """Restore the brute-force cache; return {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    cache = {}
    for k_str, v in raw.items():
        key = _decode_cache_key(k_str)
        cache[key] = (v["results"], v["latencies"])
    return cache
