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
