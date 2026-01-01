"""Under lpm, short waiting requests share the forward of a chunked prompt's next chunk."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
    TreeLock,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

MAX_RUNNING = 20
CHUNK = 16384
PAGE = 64


class _BatchBuilt(Exception):
    pass


def _tree_cache() -> MagicMock:
    tree_cache = MagicMock()
    tree_cache.evictable_size.return_value = 0
    tree_cache.full_evictable_size.return_value = 0
    tree_cache.swa_evictable_size.return_value = 0
    tree_cache.disable = False
    tree_cache.supports_mamba.return_value = False
    tree_cache.supports_prefix_sharing.return_value = False
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()
    tree_cache.lock.side_effect = lambda node: TreeLock(
        node, tree_cache.inc_lock_ref(node).to_dec_params()
    )
    tree_cache.unlock.side_effect = lambda lock: (
        lock is not None and tree_cache.dec_lock_ref(lock.node, lock.receipt)
    )
    tree_cache.buffer_pipeline = None
    tree_cache.storage_prefetch_retries = None
    return tree_cache


def _allocator() -> MagicMock:
    allocator = MagicMock()
    allocator.has_window_pool = False
    allocator.available_size.return_value = 1 << 22
    allocator.full_available_size.return_value = 1 << 22
    allocator.swa_available_size.return_value = 0
    allocator.request_slot_reserve.return_value = 0
    allocator.page_size = PAGE
    allocator.state_slots = None
    allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
        PrefillBudget(allocator, tree_cache, **kwargs)
    )
    return allocator


def _req(rid, *, uncached: int, matched: int = 0, holds_row: bool = False):
    req = MagicMock(spec=Req)
    req.rid = str(rid)
    req.kv = ReqKvInfo(req_pool_idx=0 if holds_row else None)
    req.cache_request_handle = CacheRequestHandle(req.rid, 0)
    type(req).extend_len = Req.extend_len
    req.extend_end = None
    req.origin_input_ids = list(range(matched + uncached))
    req.full_untruncated_fill_ids = req.origin_input_ids
    req.output_ids = []
    req.prefix_len = matched
    req.num_matched_prefix_tokens = matched
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    req.retracted_stain = False
    req.beam_group = None
    req.token_indices_to_pool = None
    req.session = None
    req.last_node = MagicMock()
    for name in (
        "host_hit_length",
        "swa_host_hit_length",
        "auxiliary_swa_host_hit_length",
        "mamba_host_hit_length",
        "storage_hit_length",
        "host_loaded_length",
    ):
        setattr(req, name, 0)
    req.storage_hit_start = None
    req.host_hit_is_storage = False
    req.materialized_host_hit_len.return_value = 0
    req.fulfilled_storage_hit_len.return_value = 0
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    return req


class TestChunkReserveAdmission(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _scheduler(self, running, chunked_req, waiting) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = IndexedReqList(ListKind.WAITING)
        s.waiting_queue.join(waiting)
        s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        s.chunked_req = chunked_req
        s.min_free_slots_delayer = None
        s.policy = SchedulePolicy(
            policy="lpm",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # The queue is given in lpm order; matching needs a real tree.
        s.policy.calc_priority = lambda waiting, *args, **kwargs: list(waiting)
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = CHUNK
        s.dynamic_chunk_sizer = None
        s.pp_min_chunked_prefill_size = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: PrefillAdder
        s.page_size = PAGE
        s.tree_cache = _tree_cache()
        s.token_to_kv_pool_allocator = _allocator()
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = MAX_RUNNING
        s.dllm_config = None
        s.enable_lora = False
        s.disagg_prefill_bootstrap_queue = None
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.req_to_token_pool = SimpleNamespace(
            available_size=lambda: MAX_RUNNING - len(running) - 1
        )
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        s.running_reqs = KeyedReqList(ListKind.DECODING)
        return s

    def _admit(self, *, num_running: int, waiting):
        running = [
            _req(f"run{i}", uncached=1, holds_row=True) for i in range(num_running)
        ]
        chunked = _req("chunked", uncached=100_000, holds_row=True)
        chunked.prefix_len = 2 * CHUNK
        s = self._scheduler(running, chunked, waiting)

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=MAX_RUNNING),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None,
                    running_batch=ScheduleBatch(reqs=running),
                )
        return s, chunked, built.exception.args[0]

    def _waiting(self):
        # lpm order: by matched prefix, so a long request can lead short ones.
        return [
            _req("mid", uncached=8000, matched=90_000),
            _req("a", uncached=512, matched=60_000),
            _req("b", uncached=1000, matched=50_000),
            _req("c", uncached=2048, matched=40_000),
            _req("d", uncached=256, matched=30_000),
            _req("cold", uncached=50_000),
        ]

    def test_short_requests_ride_the_chunk_past_a_longer_one(self):
        waiting = self._waiting()
        mid, a, b, c, d, cold = waiting
        s, chunked, can_run_list = self._admit(num_running=10, waiting=waiting)

        self.assertEqual(can_run_list, [chunked, a, b, c, d])
        # 512 + 1024 + 2048 + 256 reserved, in pages.
        self.assertEqual(chunked.extend_len, CHUNK - 3840)
        self.assertIs(s.chunked_req, chunked)
        self.assertEqual(list(s.waiting_queue), [mid, cold])
        # The longer requests start no second chunked prefill.
        self.assertIsNone(mid.extend_end)
        self.assertIsNone(cold.extend_end)

    def test_reserve_counts_only_the_free_request_slots(self):
        waiting = self._waiting()
        mid, a, b, c, d, cold = waiting
        # 20 slots, 16 decoding, the chunked request: three are free.
        s, chunked, can_run_list = self._admit(num_running=16, waiting=waiting)

        self.assertEqual(can_run_list, [chunked, a, b, c])
        self.assertEqual(chunked.extend_len, CHUNK - 3584)
        self.assertEqual(list(s.waiting_queue), [mid, d, cold])

    def test_full_running_batch_leaves_the_chunk_whole(self):
        s, chunked, can_run_list = self._admit(
            num_running=MAX_RUNNING - 1, waiting=self._waiting()
        )
        self.assertEqual(can_run_list, [chunked])
        self.assertEqual(chunked.extend_len, CHUNK)

    def test_env_switch_keeps_the_whole_chunk_for_the_chunked_request(self):
        with envs.SGLANG_ENABLE_PREFILL_WORK_AWARE_ADMISSION.override(False):
            s, chunked, can_run_list = self._admit(
                num_running=10, waiting=self._waiting()
            )
        self.assertEqual(can_run_list, [chunked])
        self.assertEqual(chunked.extend_len, CHUNK)
        self.assertEqual(len(s.waiting_queue), 6)


if __name__ == "__main__":
    unittest.main()
