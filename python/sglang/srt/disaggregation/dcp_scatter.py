"""Prefill-side scatter of DeepSeek-V4 unified_kv rows to their owning decode rank.

Sibling of ``common/utils.build_dcp_token_transfer_plan`` (sglang #32997 and its
predecessor), which does the same job for the GENERIC layout. The algebra is
identical -- ``owner = x % N``, ``local = x // N`` -- and only the index space
differs:

    generic : x = global TOKEN POSITION. Needs the chunk phase
              (src_page_offset / decode_prefix_len) to place the round robin,
              and requires decode_prefix_len to align to page_size * dcp_size.
    v4      : x = unified_kv STORAGE SLOT, which the decode allocator hands out,
              so there is no phase to track; and slots below swa_pages are the
              replicated SWA ring, which the generic plan has no notion of.

Keep the two side by side rather than forcing one to serve both: collapsing
them would mean teaching the generic builder about a replicated prefix region
that only DeepSeek-V4 has, on a path mooncake/nixl also depend on.

numpy-only and free of any transport dependency, so the mapping can be exercised
on its own (dcp/gpu/l5_pd_scatter.py drives this exact function across two
nodes). mori/conn.py is the only production caller.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


def dcp_scatter(
    dcp_size: int,
    dcp_rank: int,
    swa_pages: int,
    prefill_kv_indices: npt.NDArray[np.int32],
    dst_kv_indices: npt.NDArray[np.int32],
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Keep only the rows this decode rank owns, and remap them to its shard.

    Under DeepSeek-V4 unified_kv physical DCP the decode buffer is laid out
    as ``[ swa_pages (replicated) | ceil(compress/dcp) | DEAD ]`` and the
    owner of a compressed row is ``(slot - swa_pages) % dcp``. Prefill runs
    TP with a full (replicated) MLA KV on every rank, so every prefill rank
    can serve every decode rank directly -- no all-to-all, just a filter and
    an index remap, mirroring ``_dcp_row_owner(PHYSICAL=True)``.

    The SWA region ``[0, swa_pages)`` is replicated on the decode side and
    is passed through untouched (it is normally shipped separately as
    StateType.SWA_RING; handled here too so the mapping stays total).

    No-op when the peer reports ``dcp_size == 1``.
    """
    dcp = dcp_size
    if dcp <= 1 or dst_kv_indices.size == 0:
        return prefill_kv_indices, dst_kv_indices

    rank = dcp_rank
    dst = dst_kv_indices.astype(np.int64)

    is_swa = dst < swa_pages
    page = dst - swa_pages
    owned = is_swa | ((page % dcp) == rank)
    if not owned.any():
        empty = np.empty(0, dtype=dst_kv_indices.dtype)
        return empty, empty

    local = np.where(is_swa, dst, swa_pages + page // dcp)
    return (
        prefill_kv_indices[owned],
        local[owned].astype(dst_kv_indices.dtype),
    )

