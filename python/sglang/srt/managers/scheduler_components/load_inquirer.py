from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.dp_cost_model import (
    DeepSeekV4PrefillCostModel,
    PrefillCostEstimate,
)
from sglang.srt.managers.load_snapshot import (
    DisaggregationMetrics,
    LoadSnapshot,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.runtime_context import get_lora

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    disaggregation_mode: DisaggregationMode
    ps: ParallelState
    server_args: ServerArgs
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: SchedulerPoolStatsObserver
    tp_worker: BaseTpWorker
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    spec_algorithm: SpeculativeAlgorithm
    prefill_cost_model: DeepSeekV4PrefillCostModel | None
    get_running_batch: Callable
    get_current_batch: Callable
    get_inflight_prefill_batch: Callable
    get_waiting_queue: Callable
    get_stats: Callable
    get_chunked_req: Callable
    get_disagg_prefill_bootstrap_queue: Callable
    get_disagg_prefill_inflight_queue: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_disagg_decode_transfer_queue: Callable
    get_spec_total_num_accept_tokens: Callable
    get_spec_total_num_forward_ct: Callable
    get_total_prefill_uncached_tokens: Callable
    get_total_prefill_busy_us: Callable
    get_decode_moment_totals: Callable
    get_last_accepted_dispatch_seq: Callable

    def _get_num_pending_tokens(self, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_indices`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_indices`` is already up-to-date, so the default
                0 is correct.
        """
        num_pending_tokens = sum(req.seqlen for req in self.get_waiting_queue())
        if self.get_chunked_req() is not None:
            req = self.get_chunked_req()
            num_pending_tokens += req.seqlen - len(req.prefix_indices) - chunk_deduct
        return num_pending_tokens

    def get_num_waiting_uncached_tokens(self) -> int:
        """Get uncached input tokens waiting for prefill compute."""
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return 0
        num_tokens = 0
        for req in self.get_waiting_queue():
            # if match-in-waiting-queue disabled, this metric returns seq_lens
            num_tokens += max(0, req.seqlen - req.num_matched_prefix_tokens)
        cr = self.get_chunked_req()
        if cr is not None:
            num_tokens += max(0, cr.seqlen - len(cr.prefix_indices))
        return num_tokens

    def get_prefill_cost_estimate(self) -> PrefillCostEstimate:
        """Estimate all waiting and overlap-inflight DSV4 prefill work."""
        model = self.prefill_cost_model
        if model is None or self.disaggregation_mode == DisaggregationMode.DECODE:
            return PrefillCostEstimate()

        cost = PrefillCostEstimate()
        seen: set[int] = set()
        for req in self.get_waiting_queue():
            seen.add(id(req))
            max_prefix_len = max(0, req.seqlen - 1)
            pending_storage_tokens = req.storage_prefetch_tokens
            cached_context_tokens = min(
                len(req.prefix_indices) + req.host_hit_length + pending_storage_tokens,
                max_prefix_len,
            )
            cost += model.estimate(
                input_tokens=req.seqlen,
                cached_context_tokens=cached_context_tokens,
                host_cache_tokens=req.host_hit_length + pending_storage_tokens,
                storage_cache_tokens=max(
                    req.storage_hit_length, pending_storage_tokens
                ),
                swa_host_cache_tokens=req.swa_host_hit_length,
            )

        active_prefill_reqs = []
        chunked_req = self.get_chunked_req()
        if chunked_req is not None:
            active_prefill_reqs.append(chunked_req)

        inflight_batch = self.get_inflight_prefill_batch()
        if (
            inflight_batch is not None
            and inflight_batch.forward_mode.is_extend_without_speculative()
        ):
            decoding_reqs = getattr(inflight_batch, "decoding_reqs", None) or []
            decoding_ids = {id(req) for req in decoding_reqs}
            active_prefill_reqs.extend(
                req for req in inflight_batch.reqs if id(req) not in decoding_ids
            )

        for req in active_prefill_reqs:
            if id(req) in seen:
                continue
            seen.add(id(req))
            # Once selected for a batch, HiCache load-back is complete and
            # prefix_indices is the materialized context. Charge the current
            # chunk plus any later chunks, but do not charge transfer twice.
            cost += model.estimate(
                input_tokens=req.seqlen,
                cached_context_tokens=min(
                    len(req.prefix_indices),
                    max(0, req.seqlen - 1),
                ),
            )
        return cost

    def get_next_prefill_step_cost_s(
        self,
        *,
        max_input_tokens: int,
        max_chunk_tokens: int,
        page_size: int,
        linear_tokens_per_second: float,
    ) -> float:
        """Estimate the work this rank would naturally admit on its next step."""
        if linear_tokens_per_second <= 0:
            raise ValueError("linear_tokens_per_second must be positive")
        model = self.prefill_cost_model
        if model is None or self.disaggregation_mode != DisaggregationMode.PREFILL:
            return 0.0

        rem_input_tokens = max(0, max_input_tokens)
        rem_chunk_tokens = max(0, max_chunk_tokens)
        cost_s = 0.0
        seen: set[int] = set()

        def add_req(req, *, materialized_prefix: bool) -> bool:
            nonlocal cost_s, rem_input_tokens, rem_chunk_tokens
            seen.add(id(req))
            max_prefix_len = max(0, req.seqlen - 1)
            if materialized_prefix:
                cached_context_tokens = min(
                    len(req.prefix_indices), max_prefix_len
                )
                host_cache_tokens = 0
                storage_cache_tokens = 0
                swa_host_cache_tokens = 0
            else:
                pending_storage_tokens = req.storage_prefetch_tokens
                cached_context_tokens = min(
                    len(req.prefix_indices)
                    + req.host_hit_length
                    + pending_storage_tokens,
                    max_prefix_len,
                )
                host_cache_tokens = req.host_hit_length + pending_storage_tokens
                storage_cache_tokens = max(
                    req.storage_hit_length, pending_storage_tokens
                )
                swa_host_cache_tokens = req.swa_host_hit_length

            remaining_tokens = req.seqlen - cached_context_tokens
            take_tokens = min(
                remaining_tokens, rem_input_tokens, rem_chunk_tokens
            )
            if take_tokens < remaining_tokens:
                take_tokens = take_tokens // page_size * page_size
            if take_tokens <= 0:
                return False

            # The shared DSV4 model intentionally omits request-linear
            # dense/MoE work, which is material when cache hits change the
            # number of new tokens admitted by each rank.
            attention_and_transfer_cost = model.estimate(
                input_tokens=cached_context_tokens + take_tokens,
                cached_context_tokens=cached_context_tokens,
                host_cache_tokens=host_cache_tokens,
                storage_cache_tokens=storage_cache_tokens,
                swa_host_cache_tokens=swa_host_cache_tokens,
            )
            cost_s += (
                attention_and_transfer_cost.total_seconds
                + take_tokens / linear_tokens_per_second
            )
            rem_input_tokens -= take_tokens
            rem_chunk_tokens -= take_tokens
            return (
                take_tokens == remaining_tokens
                and rem_input_tokens > 0
                and rem_chunk_tokens > 0
            )

        chunked_req = self.get_chunked_req()
        if chunked_req is not None and not add_req(
            chunked_req, materialized_prefix=True
        ):
            return cost_s

        for req in self.get_waiting_queue():
            if id(req) in seen:
                continue
            if not add_req(req, materialized_prefix=False):
                break
        return cost_s

    def get_num_running_reqs(self) -> int:
        """Count active requests across running and current batches once."""
        seen: set[int] = set()
        for batch in (self.get_running_batch(), self.get_current_batch()):
            if batch is None:
                continue
            for req in batch.reqs:
                req_id = id(req)
                if req_id in seen:
                    continue
                finished = getattr(req, "finished", None)
                if callable(finished) and finished():
                    continue
                seen.add(req_id)
        return len(seen)

    def get_loads(self) -> LoadSnapshot:
        """Build the per-DP-rank load snapshot for DP balancing and /v1/loads."""
        prefill_cost = self.get_prefill_cost_estimate()
        stats = self.get_stats()
        num_running_reqs = self.get_num_running_reqs()

        waiting_queues = [self.get_waiting_queue()]
        pending_token_queues = [self.get_waiting_queue()]
        awaiting_kv_tokens = 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            prefill_bootstrap_queue = self.get_disagg_prefill_bootstrap_queue().queue
            waiting_queues.append(prefill_bootstrap_queue)
            pending_token_queues.append(prefill_bootstrap_queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            decode_prealloc_queue = self.get_disagg_decode_prealloc_queue().queue
            decode_transfer_queue = self.get_disagg_decode_transfer_queue().queue
            decode_retracted_queue = (
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )
            waiting_queues.append(decode_prealloc_queue)
            waiting_queues.append(decode_transfer_queue)
            waiting_queues.append(decode_retracted_queue)
            # In disaggregated decode, transfer-queue requests and transferred
            # waiting-queue requests have already pre-allocated decode-side KV
            # slots, so they are already included in num_used_tokens.
            pending_token_queues = [decode_prealloc_queue, decode_retracted_queue]
            # KV not yet arrived from the prefill side.
            awaiting_kv_tokens = sum(
                req.seqlen
                for queue in (decode_prealloc_queue, decode_transfer_queue)
                for req in queue
            )

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in pending_token_queues for req in queue
        )
        num_active_tokens = max(0, num_total_tokens - awaiting_kv_tokens)

        memory = None
        try:
            memory = MemoryMetrics(
                weight_gb=round(self.tp_worker.model_runner.weight_load_mem_usage, 3),
                kv_cache_gb=round(
                    self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                ),
                graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                token_capacity=int(self.max_total_num_tokens),
            )
        except (AttributeError, TypeError) as e:
            logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if (
            not self.spec_algorithm.is_none()
            and self.get_spec_total_num_forward_ct() > 0
        ):
            speculative = SpeculativeMetrics(
                accept_length=(
                    self.get_spec_total_num_accept_tokens()
                    / self.get_spec_total_num_forward_ct()
                ),
                accept_rate=stats.spec_accept_rate,
            )

        lora = None
        if get_lora().enable_lora:
            lora = LoRAMetrics(
                slots_used=stats.lora_pool_slots_used,
                slots_total=stats.lora_pool_slots_total,
                utilization=stats.lora_pool_utilization,
            )

        mode_str = "null"
        prefill_bootstrap = prefill_inflight = 0
        decode_prealloc = decode_transfer = decode_retracted = 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            mode_str = "prefill"
            prefill_bootstrap = len(self.get_disagg_prefill_bootstrap_queue().queue)
            prefill_inflight = len(self.get_disagg_prefill_inflight_queue())
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            mode_str = "decode"
            decode_prealloc = len(self.get_disagg_decode_prealloc_queue().queue)
            decode_transfer = len(self.get_disagg_decode_transfer_queue().queue)
            decode_retracted = len(
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )
        disaggregation = DisaggregationMetrics(
            mode=mode_str,
            prefill_bootstrap_queue_reqs=prefill_bootstrap,
            prefill_inflight_queue_reqs=prefill_inflight,
            decode_prealloc_queue_reqs=decode_prealloc,
            decode_transfer_queue_reqs=decode_transfer,
            decode_retracted_queue_reqs=decode_retracted,
            kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
            kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
        )

        queues = QueueMetrics(
            waiting=len(self.get_waiting_queue()),
            grammar=stats.num_grammar_queue_reqs,
            paused=stats.num_paused_reqs,
            retracted=stats.num_retracted_reqs,
        )

        totals = self.get_decode_moment_totals()
        decode_moments = list(totals) if totals[0] > 0 else None

        return LoadSnapshot(
            dp_rank=int(self.ps.dp_rank) if self.ps.dp_rank is not None else 0,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_waiting_uncached_tokens=self.get_num_waiting_uncached_tokens(),
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            num_active_tokens=num_active_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            last_accepted_dispatch_seq=self.get_last_accepted_dispatch_seq(),
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(stats.gen_throughput, 2),
            cache_hit_rate=round(stats.cache_hit_rate, 4),
            utilization=round(stats.utilization, 4),
            prefill_cost_s=prefill_cost.total_seconds,
            prefill_csa_indexer_s=prefill_cost.csa_indexer_seconds,
            prefill_csa_attention_s=prefill_cost.csa_attention_seconds,
            prefill_hca_attention_s=prefill_cost.hca_attention_seconds,
            prefill_swa_attention_s=prefill_cost.swa_attention_seconds,
            prefill_h2d_s=prefill_cost.h2d_seconds,
            prefill_storage_prefetch_s=prefill_cost.storage_prefetch_seconds,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
            total_prefill_uncached_tokens=self.get_total_prefill_uncached_tokens(),
            total_prefill_busy_us=self.get_total_prefill_busy_us(),
            decode_moments=decode_moments,
        )
