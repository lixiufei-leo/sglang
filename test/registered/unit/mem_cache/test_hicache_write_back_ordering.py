"""HiCache backups must wait for the checkpoint's KV producer."""

import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    node_device_value,
    node_host_value,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures

register_cuda_ci(est_time=13, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHiCacheWriteBackOrdering(UnifiedRadixCacheFixtures, unittest.TestCase):
    def _request(self, cache, allocator, req_pool, start, length=8):
        req = self._make_req(req_pool)
        req.origin_input_ids = array("q", range(start, start + 2 * length))
        req.full_untruncated_fill_ids = array("q", range(start, start + length))
        req.output_ids = array("q")
        req.pending_bootstrap = False
        req.extend_end = length
        ids = allocator.alloc(length)
        self.assertIsNotNone(ids)
        req_pool.write((req.kv.req_pool_idx, slice(0, length)), ids)
        req.kv.kv_committed_len = length
        req.last_node = cache.root_node_handle()
        req.lock_receipt = DecLockRefParams()
        return req

    def test_write_back_internal_swa_waits_for_checkpoint_producer(self):
        self.cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            page_size=1,
            sliding_window_size=4,
            num_layers=2,
            full_attention_layer_ids=(0,),
            kv_size=64,
            max_context_len=64,
            head_num=1,
            head_dim=64,
        )
        cache, allocator, req_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        controller = cache.cache_controller
        forward = torch.cuda.Stream()
        controller.write_fence_stream = forward
        warm = self._request(cache, allocator, req_pool, 100, length=12)
        cache.checkpoint(warm, up_to=12)
        warm_eviction = cache.evict(EvictParams(swa_num_tokens=1))
        self.assertGreaterEqual(warm_eviction.swa_num_tokens_evicted, 1)
        req = self._request(cache, allocator, req_pool, 0, length=12)
        kv = allocator.get_kvcache()
        buffers = [
            buf
            for pool in (kv.full_kv_pool, kv.swa_kv_pool)
            for buf in pool.k_buffer + pool.v_buffer
        ]
        for buf in buffers:
            buf.fill_(-1)
        read_done = torch.cuda.Event()
        read_done.record()
        torch.cuda.synchronize()
        engine = controller.l2_transfer_engine
        submit = engine.submit_device_to_host
        snapshots = []

        def record_copy(transfers):
            completion = submit(transfers)
            completion.finish_event.synchronize()
            for transfer in transfers:
                if transfer.host_indices.numel():
                    snapshots.append(
                        transfer.host_pool.get_data_page(
                            int(transfer.host_indices[0])
                        ).clone()
                    )
            return completion

        with torch.cuda.stream(forward):
            read_done.record()
            torch.cuda._sleep(600_000_000)
            for buf in buffers:
                buf.fill_(42)
        torch.cuda.current_stream().wait_event(read_done)
        try:
            with patch.object(engine, "submit_device_to_host", side_effect=record_copy):
                cache.checkpoint(req, up_to=12)
                self.assertEqual(controller.write_queue, [])
                # The active request pins Full and its final SWA window. An
                # older window from this same forward is still evictable.
                evicted = cache.evict(EvictParams(swa_num_tokens=1))
            self.assertGreaterEqual(evicted.swa_num_tokens_evicted, 1)
            self.assertTrue(snapshots)
            for snapshot in snapshots:
                self.assertTrue(
                    torch.equal(snapshot, torch.full_like(snapshot, 42)),
                    f"expected 42, copied {snapshot.unique().tolist()}",
                )
        finally:
            forward.synchronize()

    def test_write_through_stashed_chunk_waits_for_its_producer(self):
        # Prefill read-done lets the scheduler stash a chunk while its forward
        # still writes the KV; the write-through backup must copy the final KV.
        self.cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            page_size=1,
            sliding_window_size=4,
            num_layers=2,
            full_attention_layer_ids=(0,),
            kv_size=64,
            max_context_len=64,
            head_num=1,
            head_dim=64,
        )
        cache, allocator, req_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_through")
        cache.write_through_threshold = 1
        controller = cache.cache_controller
        forward = torch.cuda.Stream()
        controller.write_fence_stream = forward
        req = self._request(cache, allocator, req_pool, 0, length=12)
        kv = allocator.get_kvcache()
        buffers = [
            buf
            for pool in (kv.full_kv_pool, kv.swa_kv_pool)
            for buf in pool.k_buffer + pool.v_buffer
        ]
        for buf in buffers:
            buf.fill_(-1)
        read_done = torch.cuda.Event()
        torch.cuda.synchronize()
        engine = controller.l2_transfer_engine
        submit = engine.submit_device_to_host
        snapshots = []

        def record_copy(transfers):
            completion = submit(transfers)
            completion.finish_event.synchronize()
            for transfer in transfers:
                if transfer.host_indices.numel():
                    snapshots.append(
                        transfer.host_pool.get_data_page(
                            int(transfer.host_indices[0])
                        ).clone()
                    )
            return completion

        with torch.cuda.stream(forward):
            read_done.record()
            torch.cuda._sleep(600_000_000)
            for buf in buffers:
                buf.fill_(42)
        torch.cuda.current_stream().wait_event(read_done)
        try:
            with patch.object(engine, "submit_device_to_host", side_effect=record_copy):
                cache.checkpoint(req, up_to=12)
                cache.flush_pending_backups()
            self.assertTrue(snapshots)
            for snapshot in snapshots:
                self.assertTrue(
                    torch.equal(snapshot, torch.full_like(snapshot, 42)),
                    f"expected 42, copied {snapshot.unique().tolist()}",
                )
        finally:
            forward.synchronize()

    def test_write_back_waits_only_for_the_producer_of_its_rows(self):
        # A write-back copy starts after the forward that produces its rows, and
        # does not queue behind a forward launched after those rows were cached.
        self.cfg = CacheConfig(
            page_size=64,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            mamba_cache_size=40,
            kv_size=4096,
            max_num_reqs=20,
            max_context_len=1024,
        )
        cache, allocator, req_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        controller = cache.cache_controller
        forward = torch.cuda.Stream()
        controller.write_fence_stream = forward
        temporal = req_pool.mamba_pool.mamba_cache.temporal
        host_pool = controller.mem_pool_host.get_pool(PoolName.MAMBA)

        def cache_state(start, marker):
            """Fill a request's state with ``marker`` and cache it; (node, key)."""
            seq = self._make_seq(start, 2)
            req = self._make_req(req_pool)
            temporal[:, int(req.kv.mamba_pool_idx)] = marker
            torch.cuda.synchronize()
            return seq, req

        def insert(seq, req):
            key = RadixKey(array("q", seq))
            value = self._alloc(allocator, len(seq))
            cache.insert(
                InsertParams(
                    key=key,
                    value=value[: len(key)],
                    mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
                )
            )
            return cache.match_prefix(MatchPrefixParams(key=key)).best_match_node

        def host_state(node):
            self.assertIsNone(node_device_value(cache, node, ComponentType.MAMBA))
            host_value = node_host_value(cache, node, ComponentType.MAMBA)
            self.assertIsNotNone(host_value)
            return next(host_pool._iter_page_tensors(int(host_value[0])))

        # Load the transfer kernels before timing an eviction against a forward.
        insert(*cache_state(100000, 0))
        cache.evict(EvictParams(mamba_num=1))
        old = insert(*cache_state(200000, 1))
        fresh_seq, fresh_req = cache_state(300000, -1)
        fresh_slot = int(fresh_req.kv.mamba_pool_idx)
        torch.cuda.synchronize()

        forward_done = torch.cuda.Event()
        with torch.cuda.stream(forward):
            torch.cuda._sleep(600_000_000)
            temporal[:, fresh_slot] = 42
            forward_done.record()
        try:
            fresh = insert(fresh_seq, fresh_req)

            cache.evict(EvictParams(mamba_num=1))
            self.assertFalse(forward_done.query(), "old state waited for the forward")
            state = host_state(old)
            self.assertTrue(torch.equal(state, torch.full_like(state, 1)))

            cache.evict(EvictParams(mamba_num=1))
            state = host_state(fresh)
            self.assertTrue(
                torch.equal(state, torch.full_like(state, 42)),
                f"expected 42, copied {state.unique().tolist()}",
            )
        finally:
            forward.synchronize()


if __name__ == "__main__":
    unittest.main()
