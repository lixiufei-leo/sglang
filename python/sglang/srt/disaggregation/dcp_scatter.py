"""Transport-free transfer planning for DeepSeek-V4 physical DCP.

DeepSeek-V4 keeps three different KV layouts behind one page-index protocol:

* C4/C128 unified KV is physically sharded by compressed-row ownership.
* The C4 indexer cache remains page-addressed and replicated on every rank.
* The SWA ring is transferred separately as request state.

The generic DCP planner operates in token-position space and therefore cannot
describe this mixed layout.  Keep the row mapping here so every transport can
exercise the exact same numpy-only contract.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence, Tuple

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class DSV4KVDescriptorSpec:
    kind: Literal["main", "indexer"]
    compression_ratio: int


@dataclasses.dataclass(frozen=True)
class DSV4PhysicalRowTransferPlan:
    src_row_indices: npt.NDArray[np.int64]
    dst_row_indices: npt.NDArray[np.int64]


@dataclasses.dataclass(frozen=True)
class DSV4ReplicatedPageTransferPlan:
    src_page_indices: npt.NDArray[np.int64]
    dst_page_indices: npt.NDArray[np.int64]


def dsv4_kv_descriptor_specs(
    compression_ratios: Sequence[int],
    *,
    start_layer: int = 0,
    end_layer: int | None = None,
) -> Tuple[DSV4KVDescriptorSpec, ...]:
    """Return the flat descriptor order emitted by the DSV4 KV pool.

    ``get_contiguous_buf_infos`` emits ``[c4 main, c4 indexer, c128 main]``;
    each section preserves compressed-layer order within the selected PP stage.
    """

    if end_layer is None:
        end_layer = len(compression_ratios)
    if not 0 <= start_layer <= end_layer <= len(compression_ratios):
        raise ValueError(
            "Invalid DSV4 layer slice: "
            f"start={start_layer}, end={end_layer}, total={len(compression_ratios)}"
        )

    local_ratios = tuple(
        int(ratio) for ratio in compression_ratios[start_layer:end_layer]
    )
    unsupported = sorted({ratio for ratio in local_ratios if ratio not in (0, 4, 128)})
    if unsupported:
        raise ValueError(f"Unsupported DSV4 compression ratios: {unsupported}")

    c4_count = sum(ratio == 4 for ratio in local_ratios)
    c128_count = sum(ratio == 128 for ratio in local_ratios)
    return (
        (DSV4KVDescriptorSpec("main", 4),) * c4_count
        + (DSV4KVDescriptorSpec("indexer", 4),) * c4_count
        + (DSV4KVDescriptorSpec("main", 128),) * c128_count
    )


def build_dsv4_physical_row_transfer_plan(
    prefill_page_indices: npt.ArrayLike,
    dst_page_indices: npt.ArrayLike,
    *,
    physical_page_size: int,
    compression_ratio: int,
    dcp_size: int,
    dcp_rank: int,
) -> DSV4PhysicalRowTransferPlan:
    """Map full prefill compressed rows into one physical-DCP decode shard.

    Page arrays remain one entry per ``physical_page_size`` raw tokens on both
    sides.  Within a page, a compression ratio ``R`` produces
    ``physical_page_size / R`` rows.  Decode rank ``r`` owns global compressed
    row ``x`` iff ``x % dcp_size == r`` and stores it at ``x // dcp_size``.

    The returned indices are flat row indices relative to the compressed-region
    base pointers.  They intentionally exclude the replicated SWA prefix.
    """

    src_pages = np.asarray(prefill_page_indices, dtype=np.int64).reshape(-1)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64).reshape(-1)

    if src_pages.size != dst_pages.size:
        raise ValueError(
            "DSV4 physical DCP requires positionally aligned page arrays: "
            f"src={src_pages.size}, dst={dst_pages.size}"
        )
    if physical_page_size <= 0:
        raise ValueError(
            f"physical_page_size must be positive, got {physical_page_size}"
        )
    if compression_ratio <= 0 or physical_page_size % compression_ratio != 0:
        raise ValueError(
            "compression_ratio must divide physical_page_size, got "
            f"page={physical_page_size}, ratio={compression_ratio}"
        )
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"Invalid DCP geometry: size={dcp_size}, rank={dcp_rank}")
    if (src_pages < 0).any() or (dst_pages < 0).any():
        raise ValueError("KV page indices must be non-negative")
    if src_pages.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return DSV4PhysicalRowTransferPlan(empty, empty)

    rows_per_page = physical_page_size // compression_ratio
    row_offsets = np.arange(rows_per_page, dtype=np.int64)
    src_rows = (src_pages[:, None] * rows_per_page + row_offsets).reshape(-1)
    dst_global_rows = (dst_pages[:, None] * rows_per_page + row_offsets).reshape(-1)

    if dcp_size == 1:
        return DSV4PhysicalRowTransferPlan(src_rows, dst_global_rows)

    owned = (dst_global_rows % dcp_size) == dcp_rank
    return DSV4PhysicalRowTransferPlan(
        src_row_indices=src_rows[owned],
        dst_row_indices=dst_global_rows[owned] // dcp_size,
    )


def build_dsv4_replicated_page_transfer_plan(
    prefill_page_indices: npt.ArrayLike,
    dst_page_indices: npt.ArrayLike,
) -> DSV4ReplicatedPageTransferPlan:
    """Keep the C4 indexer page layout replicated on every decode rank."""

    src_pages = np.asarray(prefill_page_indices, dtype=np.int64).reshape(-1)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64).reshape(-1)
    if src_pages.size != dst_pages.size:
        raise ValueError(
            "DSV4 replicated indexer requires aligned page arrays: "
            f"src={src_pages.size}, dst={dst_pages.size}"
        )
    if (src_pages < 0).any() or (dst_pages < 0).any():
        raise ValueError("KV page indices must be non-negative")
    return DSV4ReplicatedPageTransferPlan(src_pages, dst_pages)


def dcp_scatter(
    dcp_size: int,
    dcp_rank: int,
    swa_pages: int,
    prefill_kv_indices: npt.NDArray[np.int32],
    dst_kv_indices: npt.NDArray[np.int32],
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Compatibility helper for callers whose indices already denote rows.

    Production DSV4 KV transfer uses ``build_dsv4_physical_row_transfer_plan``;
    this function preserves the former abstract row-slot mapping used by older
    diagnostics.
    """

    dcp = int(dcp_size)
    if dcp <= 1 or dst_kv_indices.size == 0:
        return prefill_kv_indices, dst_kv_indices

    dst = dst_kv_indices.astype(np.int64)
    is_swa = dst < swa_pages
    compressed_row = dst - swa_pages
    owned = is_swa | ((compressed_row % dcp) == int(dcp_rank))
    if not owned.any():
        empty = np.empty(0, dtype=dst_kv_indices.dtype)
        return empty, empty

    local = np.where(is_swa, dst, swa_pages + compressed_row // dcp)
    return prefill_kv_indices[owned], local[owned].astype(dst_kv_indices.dtype)
