"""Which device rows entered the tree while a forward could still write them.

A device-to-host backup has to wait for the forwards that produce the rows it
copies. Those are the forwards launched before the rows were handed to the
tree: once they finish, tree-owned rows no longer change. A backup therefore
waits for the forward stream only as far as the handover of its own rows, and
not at all when that point has already passed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.utils import get_device_module

device_module = get_device_module()

# Pools whose handed-over rows an insert names; a backup touching any other
# independent pool waits for every handover still pending.
_ATTRIBUTED_POOLS = (PoolName.KV, PoolName.MAMBA)

DeviceIndicesByPool = dict[PoolName, list[torch.Tensor]]


def _on_host(indices: list[torch.Tensor]) -> torch.Tensor:
    # Compared on the host: a device-side set test would queue kernels beside
    # the forward, and the read-back only waits for the indices' own stream.
    return torch.cat([i.reshape(-1).cpu() for i in indices])


def _intersect(indices: torch.Tensor, handed: torch.Tensor) -> bool:
    if indices.numel() == 0:
        return False
    # A request's handover can span its whole context; a backup's rows are few
    # and close together, so test only the handed rows inside their range.
    low, high = torch.aminmax(indices)
    near = handed[(handed >= low) & (handed <= high)]
    return near.numel() > 0 and bool(torch.isin(indices, near).any())


class DeviceHandover(msgspec.Struct):
    # None: the insert did not name its rows, so it covers every backup.
    indices_by_pool: Optional[DeviceIndicesByPool]
    # Recorded on the fence stream when the insert ends; None while it runs.
    producers_done: Optional[Any] = None
    # Host copies of indices_by_pool, made by the first backup that asks.
    host_indices_by_pool: Optional[dict[PoolName, torch.Tensor]] = None

    def holds_any(self, host_indices_by_pool: dict[PoolName, torch.Tensor]) -> bool:
        if self.host_indices_by_pool is None:
            self.host_indices_by_pool = {
                pool: _on_host(indices)
                for pool, indices in self.indices_by_pool.items()
                if indices
            }
        for pool, indices in host_indices_by_pool.items():
            handed = self.host_indices_by_pool.get(pool)
            if handed is not None and _intersect(indices, handed):
                return True
        return False


class WholeStream:
    """Fence marker: wait for everything queued on the fence stream."""


WHOLE_STREAM = WholeStream()


def _attributed_indices(
    device_indices: torch.Tensor, pool_transfers: Optional[list[PoolTransfer]]
) -> Optional[dict[PoolName, torch.Tensor]]:
    """The backup's device rows by pool, on the host; None when one of its
    pools is not attributed."""
    indices_by_pool: DeviceIndicesByPool = {}
    if device_indices.numel() > 0:
        indices_by_pool[PoolName.KV] = [device_indices]
    for transfer in pool_transfers or []:
        if transfer.indices_from_pool is not None:
            # A sidecar holds the same tokens or slots as its source pool.
            if transfer.indices_from_pool not in _ATTRIBUTED_POOLS:
                return None
            continue
        if transfer.device_indices is None:
            continue
        if transfer.name not in _ATTRIBUTED_POOLS:
            return None
        indices_by_pool.setdefault(transfer.name, []).append(transfer.device_indices)
    return {pool: _on_host(indices) for pool, indices in indices_by_pool.items()}


class DeviceHandoverLog:
    """Handovers whose producing forwards may still be running."""

    def __init__(self, cache_controller) -> None:
        # Read at each handover: the scheduler sets write_fence_stream later.
        self._cache_controller = cache_controller
        self._pending: list[DeviceHandover] = []

    def open(
        self, indices_by_pool: Optional[DeviceIndicesByPool]
    ) -> Optional[DeviceHandover]:
        """Start a handover; None when writes are not fenced on a stream."""
        if self._cache_controller.write_fence_stream is None:
            return None
        self._drop_finished()
        handover = DeviceHandover(indices_by_pool=indices_by_pool)
        self._pending.append(handover)
        return handover

    def close(self, handover: DeviceHandover) -> None:
        """Mark the forwards queued on the fence stream so far as the producers."""
        event = device_module.Event()
        with device_module.stream(self._cache_controller.write_fence_stream):
            event.record()
        handover.producers_done = event

    def fence_for(
        self,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]],
    ) -> Any:
        """What a backup of these rows waits for: None, an event, or WHOLE_STREAM."""
        self._drop_finished()
        if not self._pending:
            return None
        if any(handover.producers_done is None for handover in self._pending):
            return WHOLE_STREAM
        host_indices_by_pool = _attributed_indices(device_indices, pool_transfers)
        # Newest first: on one stream a later event covers the earlier ones.
        for handover in reversed(self._pending):
            if (
                host_indices_by_pool is None
                or handover.indices_by_pool is None
                or handover.holds_any(host_indices_by_pool)
            ):
                return handover.producers_done
        return None

    def _drop_finished(self) -> None:
        self._pending = [
            handover
            for handover in self._pending
            if handover.producers_done is None or not handover.producers_done.query()
        ]


@contextmanager
def device_handover(
    log: Optional[DeviceHandoverLog], indices_by_pool: Optional[DeviceIndicesByPool]
):
    """Log an insert that hands ``indices_by_pool`` to the tree."""
    handover = None if log is None else log.open(indices_by_pool)
    try:
        yield
    finally:
        if handover is not None:
            log.close(handover)
