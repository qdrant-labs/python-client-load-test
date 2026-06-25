from typing import Optional

import numpy as np


def random_dense_vector(dim: int) -> list[float]:
    return np.random.uniform(-1, 1, dim).astype(np.float32).tolist()


def load_embeddings_from_parquet(parquet_file: str, column_name: str) -> np.ndarray:
    import pandas as pd
    parts = column_name.split(".")
    df = pd.read_parquet(parquet_file, columns=[parts[0]])
    series = df[parts[0]]
    for part in parts[1:]:
        series = series.apply(lambda x: x[part])
    embeddings = np.array(series.tolist(), dtype=np.float32)
    print(f"Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]}) from '{parquet_file}' column '{column_name}'")
    return embeddings


def load_sparse_embeddings_from_parquet(parquet_file: str, column_name: str = "sparse_embedding") -> list[dict]:
    """Load sparse embeddings from parquet.

    Expects a struct/dict column where each row is {"indices": [...], "values": [...]}.
    Returns a list of dicts with "indices" (list[int]) and "values" (list[float]).
    """
    import pandas as pd
    df = pd.read_parquet(parquet_file, columns=[column_name])
    series = df[column_name]

    result = []
    for item in series:
        if isinstance(item, dict):
            result.append({"indices": list(item["indices"]), "values": list(item["values"])})
        else:
            # object with .indices / .values attributes (e.g. a named tuple or struct)
            result.append({"indices": list(item.indices), "values": list(item.values)})

    print(f"Loaded {len(result)} sparse embeddings from '{parquet_file}' column '{column_name}'")
    return result


def get_query_vector(
    index: int,
    vector_dimension: Optional[int],
    query_vectors: Optional[np.ndarray] = None,
) -> list[float]:
    if query_vectors is not None:
        return query_vectors[index % len(query_vectors)].tolist()
    if vector_dimension is None:
        raise ValueError("--vector-dimension must be specified when not using --parquet-file")
    return random_dense_vector(vector_dimension)


def get_sparse_query_vector(index: int, sparse_query_vectors: list[dict]) -> dict:
    """Return sparse vector dict {"indices": [...], "values": [...]} for the given query index."""
    return sparse_query_vectors[index % len(sparse_query_vectors)]
