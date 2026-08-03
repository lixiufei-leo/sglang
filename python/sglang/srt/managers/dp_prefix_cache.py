"""Lightweight rank-local prefix cache estimates for DP dispatch.

The controller sees every tokenized request before routing it, so it can keep a
bounded directory of the prefix pages that each rank is expected to own.  This
is intentionally an estimate: scheduler-side eviction and failed requests can
make it stale.  Bounded LRU tiers keep the estimate conservative over time.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from typing import NamedTuple, Sequence


class PrefixCacheHitEstimate(NamedTuple):
    device_tokens: int = 0
    host_tokens: int = 0
    storage_tokens: int = 0
    promised_tokens: int = 0

    @property
    def cached_context_tokens(self) -> int:
        return self.device_tokens + self.host_tokens + self.storage_tokens

    @property
    def host_transfer_tokens(self) -> int:
        return self.host_tokens + self.storage_tokens


class _LRUPrefixSet:
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError(f"prefix cache capacity cannot be negative: {capacity}")
        self.capacity = capacity
        self.entries: OrderedDict[bytes, None] = OrderedDict()

    def __contains__(self, key: bytes) -> bool:
        return key in self.entries

    def add(self, key: bytes) -> bytes | None:
        if self.capacity == 0:
            return key
        if key in self.entries:
            self.entries.move_to_end(key)
            return None
        self.entries[key] = None
        if len(self.entries) <= self.capacity:
            return None
        evicted, _ = self.entries.popitem(last=False)
        return evicted

    def discard(self, key: bytes) -> None:
        self.entries.pop(key, None)

    def clear(self) -> None:
        self.entries.clear()


class DPPrefixCacheTracker:
    """Estimate contiguous L1/L2/L3 prefix hits for every DP rank.

    Keys are cumulative 128-bit hashes at cache-page boundaries.  A cumulative
    key represents a radix-tree page, so shared prefixes occupy one entry and a
    match must remain contiguous from the first page.
    """

    _HASH_DOMAIN = b"sglang-dp-prefix-v1\0"

    def __init__(
        self,
        *,
        dp_size: int,
        page_size: int,
        device_pages_per_rank: int,
        host_pages_per_rank: int = 0,
        storage_pages_per_rank: int = 0,
        host_write_through: bool = True,
    ):
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self.dp_size = dp_size
        self.page_size = page_size
        self.host_write_through = host_write_through
        self._device = [_LRUPrefixSet(device_pages_per_rank) for _ in range(dp_size)]
        self._host = [_LRUPrefixSet(host_pages_per_rank) for _ in range(dp_size)]
        self._storage = [_LRUPrefixSet(storage_pages_per_rank) for _ in range(dp_size)]
        self._promised = [
            _LRUPrefixSet(device_pages_per_rank) for _ in range(dp_size)
        ]

    @staticmethod
    def _as_i64_array(input_ids: Sequence[int]) -> array:
        if isinstance(input_ids, array) and input_ids.typecode == "q":
            return input_ids
        return array("q", input_ids)

    def _prefix_keys(
        self,
        input_ids: Sequence[int],
        namespace: bytes,
    ) -> list[bytes]:
        cacheable_tokens = max(0, len(input_ids) - 1)
        cacheable_tokens -= cacheable_tokens % self.page_size
        if cacheable_tokens == 0:
            return []

        token_array = self._as_i64_array(input_ids)
        token_bytes = memoryview(token_array).cast("B")
        bytes_per_token = token_array.itemsize
        bytes_per_page = self.page_size * bytes_per_token
        cacheable_bytes = cacheable_tokens * bytes_per_token

        digest = hashlib.blake2b(digest_size=16)
        digest.update(self._HASH_DOMAIN)
        digest.update(len(namespace).to_bytes(8, "little"))
        digest.update(namespace)
        keys = []
        for start in range(0, cacheable_bytes, bytes_per_page):
            digest.update(token_bytes[start : start + bytes_per_page])
            keys.append(digest.digest())
        return keys

    def estimate(
        self,
        input_ids: Sequence[int],
        *,
        namespace: bytes = b"",
    ) -> list[PrefixCacheHitEstimate]:
        keys = self._prefix_keys(input_ids, namespace)
        return [self._estimate_rank(rank, keys) for rank in range(self.dp_size)]

    def _estimate_rank(
        self,
        rank: int,
        keys: list[bytes],
    ) -> PrefixCacheHitEstimate:
        device_end = 0
        host_end = 0
        storage_end = 0
        tier = 0
        resident_pages = 0
        for page_index, key in enumerate(keys, start=1):
            end = page_index * self.page_size
            if tier == 0 and key in self._device[rank]:
                device_end = end
                host_end = end
                storage_end = end
            elif tier <= 1 and key in self._host[rank]:
                tier = 1
                host_end = end
                storage_end = end
            elif key in self._storage[rank]:
                tier = 2
                storage_end = end
            else:
                break
            resident_pages = page_index

        promised_end = storage_end
        for page_index, key in enumerate(
            keys[resident_pages:],
            start=resident_pages + 1,
        ):
            if key not in self._promised[rank]:
                break
            promised_end = page_index * self.page_size

        return PrefixCacheHitEstimate(
            device_tokens=device_end,
            host_tokens=host_end - device_end,
            storage_tokens=storage_end - host_end,
            promised_tokens=promised_end - storage_end,
        )

    def insert(
        self,
        rank: int,
        input_ids: Sequence[int],
        *,
        namespace: bytes = b"",
    ) -> None:
        if rank < 0 or rank >= self.dp_size:
            raise ValueError(f"DP rank {rank} is outside [0, {self.dp_size})")
        for key in self._prefix_keys(input_ids, namespace):
            self._promised[rank].discard(key)
            if self.host_write_through:
                self._add_host(rank, key)
            evicted = self._device[rank].add(key)
            if evicted is not None and not self.host_write_through:
                self._add_host(rank, evicted)

    def promise(
        self,
        rank: int,
        input_ids: Sequence[int],
        *,
        namespace: bytes = b"",
    ) -> None:
        if rank < 0 or rank >= self.dp_size:
            raise ValueError(f"DP rank {rank} is outside [0, {self.dp_size})")
        for key in self._prefix_keys(input_ids, namespace):
            self._promised[rank].add(key)

    def _add_host(self, rank: int, key: bytes) -> None:
        evicted = self._host[rank].add(key)
        if evicted is not None:
            self._storage[rank].add(evicted)

    def clear(self) -> None:
        for tiers in zip(
            self._device,
            self._host,
            self._storage,
            self._promised,
        ):
            for tier in tiers:
                tier.clear()
