import asyncio
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from .models import CollectionConfig

_TURBO_BITS = {"bits1", "bits1_5", "bits2", "bits4"}


async def wait_for_green(
    client: AsyncQdrantClient,
    collection_name: str,
    poll_secs: float = 5.0,
):
    print(f"  Waiting for '{collection_name}' to reach GREEN status...")
    while True:
        info = await client.get_collection(collection_name)
        if info.status == models.CollectionStatus.GREEN:
            print("  ✅ Collection is GREEN.")
            return
        print(f"    Status: {info.status.value} — retrying in {poll_secs:.0f}s")
        await asyncio.sleep(poll_secs)


async def apply_collection_config(
    client: AsyncQdrantClient,
    collection_name: str,
    config: CollectionConfig,
):
    print(f"  Applying collection config to '{collection_name}'...")
    if config.m is not None:
        print("  ⚠️  WARNING: Changing 'm' triggers a full HNSW index rebuild.")

    quant_model = None
    q = config.quantization
    if q and q != "none":
        ram = config.always_ram if config.always_ram is not None else True
        if q == "scalar":
            quant_model = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=ram)
            )
        elif q == "binary":
            _valid_encoding = {None, "two_bits", "one_and_half_bits"}
            _valid_query_enc = {None, "default", "binary", "scalar8bits", "scalar4bits"}
            if config.encoding not in _valid_encoding:
                raise ValueError(
                    f"Unknown binary encoding {config.encoding!r}. "
                    "Use: two_bits one_and_half_bits (or omit for 1-bit)"
                )
            if config.query_encoding not in _valid_query_enc:
                raise ValueError(
                    f"Unknown binary query_encoding {config.query_encoding!r}. "
                    "Use: default binary scalar8bits scalar4bits"
                )
            quant_model = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=ram,
                    encoding=config.encoding,
                    query_encoding=config.query_encoding,
                )
            )
        elif q == "turbo":
            bits = config.bits or "bits4"
            if bits not in _TURBO_BITS:
                raise ValueError(f"Unknown turbo bits {bits!r}. Use: bits1 bits1_5 bits2 bits4")
            quant_model = models.TurboQuantization(
                turbo=models.TurboQuantizationConfig(bits=bits, always_ram=ram)
            )
        else:
            raise ValueError(f"Unknown quantization type {q!r}. Use: none scalar binary turbo")

    hnsw_diff: Optional[models.HnswConfigDiff] = None
    if config.ef_construct is not None or config.m is not None:
        hnsw_diff = models.HnswConfigDiff(ef_construct=config.ef_construct, m=config.m)

    if quant_model is None and hnsw_diff is None:
        print("  (no collection changes to apply)")
        return

    await client.update_collection(
        collection_name=collection_name,
        quantization_config=quant_model,
        hnsw_config=hnsw_diff,
    )
    await wait_for_green(client, collection_name)
