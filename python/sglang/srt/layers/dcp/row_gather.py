"""DCP prefill: a batch's cache rows from all ranks in one pass.

Position p of a request lives on rank p % dcp_size as that rank's local row
p // dcp_size. Per layer: one indexed copy writes this rank's rows into the send
buffer, one all-gather, one indexed copy writes the gathered rows request after
request in position order. The row maps between them depend only on the batch's
lengths, so the layers of a forward share them.
"""

from collections import OrderedDict
from typing import Optional, Tuple

import msgspec
import torch

from sglang.srt.runtime_context import get_parallel, get_platform


class _RowGatherLayout(msgspec.Struct, frozen=True):
    # rows of the batch: sum of the request lengths
    total_rows: int
    # rows every rank sends: sum of ceil(length / dcp_size)
    rank_rows: int
    # rows this rank holds
    local_rows: int
    # [rank_rows] int32: this rank's local row per send row; a request whose share
    # is one short of its padded count repeats local row 0 there, a row no rank
    # reads. None when the local rows are the send rows as they are.
    send_select: Optional[torch.Tensor]
    # [total_rows] int32: row of the rank-major all-gather output per output row
    order: torch.Tensor


# A forward's layers ask for the same lengths (index-K rows and KV rows alike).
# Keyed by the lengths themselves, so an entry cannot go stale.
_LAYOUT_CACHE_SIZE = 4
_layout_cache: "OrderedDict[tuple, Optional[_RowGatherLayout]]" = OrderedDict()


def dcp_row_gather_enabled() -> bool:
    # Measured on gfx950 only; other platforms keep the multi-copy path.
    return get_platform().is_hip


def _build_layout(
    lens: Tuple[int, ...], world: int, rank: int, device: torch.device
) -> Optional[_RowGatherLayout]:
    padded = [-(-n // world) for n in lens]
    local = [(n - rank + world - 1) // world for n in lens]
    total_rows, rank_rows, local_rows = sum(lens), sum(padded), sum(local)
    if min(lens, default=0) < 0 or rank_rows == 0 or world * rank_rows >= 2**31:
        return None
    positions = torch.arange(max(lens), dtype=torch.int32, device=device)
    parts = []
    send_start = 0
    for n, n_padded in zip(lens, padded):
        if n:
            p = positions[:n]
            parts.append((p % world) * rank_rows + (p // world + send_start))
        send_start += n_padded
    order = parts[0] if len(parts) == 1 else torch.cat(parts)

    send_select = None
    if local_rows != rank_rows:
        send_select = torch.zeros(rank_rows, dtype=torch.int32, device=device)
        send_start = local_start = 0
        for n_padded, n_local in zip(padded, local):
            if n_local:
                send_select[send_start : send_start + n_local] = (
                    positions[:n_local] + local_start
                )
            send_start += n_padded
            local_start += n_local
    return _RowGatherLayout(
        total_rows=total_rows,
        rank_rows=rank_rows,
        local_rows=local_rows,
        send_select=send_select,
        order=order,
    )


def _get_layout(
    seq_lens_cpu: torch.Tensor, world: int, rank: int, device: torch.device
) -> Optional[_RowGatherLayout]:
    key = (tuple(seq_lens_cpu.tolist()), world, rank, device)
    if key in _layout_cache:
        _layout_cache.move_to_end(key)
        return _layout_cache[key]
    layout = _build_layout(lens=key[0], world=world, rank=rank, device=device)
    _layout_cache[key] = layout
    if len(_layout_cache) > _LAYOUT_CACHE_SIZE:
        _layout_cache.popitem(last=False)
    return layout


def _word_dtype(rows: torch.Tensor) -> torch.dtype:
    # Byte-packed rows copy several times faster as 8- or 4-byte words; the
    # conditions are those of Tensor.view(dtype) on [n, W] uint8.
    for dtype, size in ((torch.int64, 8), (torch.int32, 4)):
        if (
            rows.shape[1] % size == 0
            and rows.stride(0) % size == 0
            and rows.storage_offset() % size == 0
        ):
            return dtype
    return torch.uint8


def _send_rows(
    src: torch.Tensor,
    index: Optional[torch.Tensor],
    layout: _RowGatherLayout,
    words: torch.dtype,
) -> torch.Tensor:
    select = layout.send_select
    if index is None and select is None:
        return src.contiguous()
    send = src.new_empty((layout.rank_rows, src.shape[1]))
    if layout.local_rows == 0:
        # this rank holds none of the batch: its send rows are all padding
        return send
    if index is None:
        index = select
    elif select is not None:
        index = index.index_select(0, select)
    torch.index_select(src.view(words), 0, index, out=send.view(words))
    return send


def dcp_gather_rows_single_pass(
    src: torch.Tensor,
    index: Optional[torch.Tensor],
    seq_lens_cpu: torch.Tensor,
    pad_to: int = 1,
) -> Optional[torch.Tensor]:
    """Every request's rows from all ranks, request after request in position order.

    ``src`` [n, W] uint8. With ``index`` [local rows] it is this rank's cache pool
    and ``index`` its rows of the batch (request by request, local position
    order); without, ``src`` is those rows. Returns [sum(seq_lens) padded to a
    multiple of ``pad_to``, W] with the padding zeroed, or None when this path
    does not apply and the caller keeps the multi-copy one. The wire is one
    all-gather of [sum(ceil(len / dcp_size)), W] uint8 per rank.
    """
    parallel = get_parallel()
    if (
        not parallel.dcp_enabled
        or not dcp_row_gather_enabled()
        or src.dim() != 2
        or src.dtype != torch.uint8
        or not src.is_cuda
        or src.shape[1] == 0
        or src.stride(1) != 1
        or seq_lens_cpu.dim() != 1
    ):
        return None
    world = parallel.dcp_size
    layout = _get_layout(
        seq_lens_cpu=seq_lens_cpu,
        world=world,
        rank=parallel.dcp_rank,
        device=src.device,
    )
    if layout is None:
        return None
    if index is None:
        if src.shape[0] != layout.local_rows:
            return None
    elif index.dim() != 1 or index.shape[0] != layout.local_rows:
        return None
    elif index.dtype not in (torch.int32, torch.int64):
        index = index.long()

    width = src.shape[1]
    words = _word_dtype(src)
    send = _send_rows(src=src, index=index, layout=layout, words=words)
    # not use symmetric_memory unless torch mem_pool updated, see https://github.com/pytorch/pytorch/issues/178138
    gathered = src.new_empty((world * layout.rank_rows, width))
    parallel.dcp_group.all_gather_into_tensor(gathered, send)

    total = layout.total_rows
    out = src.new_empty((total + (-total % max(pad_to, 1)), width))
    out_words = out.view(words)
    torch.index_select(gathered.view(words), 0, layout.order, out=out_words[:total])
    if out.shape[0] > total:
        out_words[total:].zero_()
    return out
