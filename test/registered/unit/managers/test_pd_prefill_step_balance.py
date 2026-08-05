import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


def _req(
    *,
    seqlen: int,
    prefix_tokens: int = 0,
    host_hit_tokens: int = 0,
    storage_prefetch_tokens: int = 0,
):
    return SimpleNamespace(
        seqlen=seqlen,
        prefix_indices=list(range(prefix_tokens)),
        host_hit_length=host_hit_tokens,
        storage_prefetch_tokens=storage_prefetch_tokens,
    )


def _inquirer(
    *,
    waiting_queue=None,
    chunked_req=None,
    disaggregation_mode=DisaggregationMode.PREFILL,
):
    return SchedulerLoadInquirer(
        disaggregation_mode=disaggregation_mode,
        ps=None,
        server_args=None,
        max_total_num_tokens=0,
        max_running_requests=0,
        pool_stats_observer=None,
        tp_worker=None,
        token_to_kv_pool_allocator=None,
        spec_algorithm=None,
        get_running_batch=lambda: None,
        get_waiting_queue=lambda: waiting_queue or [],
        get_stats=lambda: None,
        get_chunked_req=lambda: chunked_req,
        get_disagg_prefill_bootstrap_queue=lambda: None,
        get_disagg_prefill_inflight_queue=lambda: None,
        get_disagg_decode_prealloc_queue=lambda: None,
        get_disagg_decode_transfer_queue=lambda: None,
        get_spec_total_num_accept_tokens=lambda: 0,
        get_spec_total_num_forward_ct=lambda: 0,
        get_total_prefill_uncached_tokens=lambda: 0,
        get_total_prefill_busy_us=lambda: 0,
        get_decode_moment_totals=lambda: [0.0] * 6,
    )


class TestPdPrefillStepCost(unittest.TestCase):
    def test_counts_uncached_tokens_after_hicache_hits(self):
        inquirer = _inquirer(
            waiting_queue=[
                _req(
                    seqlen=100,
                    prefix_tokens=20,
                    host_hit_tokens=10,
                    storage_prefetch_tokens=10,
                )
            ]
        )

        cost_s = inquirer.get_next_prefill_step_cost_s(
            max_input_tokens=70,
            max_chunk_tokens=70,
            page_size=10,
            linear_tokens_per_second=100.0,
        )

        self.assertAlmostEqual(cost_s, 0.6)

    def test_chunked_request_uses_materialized_prefix_and_page_alignment(self):
        inquirer = _inquirer(
            chunked_req=_req(seqlen=120, prefix_tokens=20),
            waiting_queue=[_req(seqlen=40)],
        )

        cost_s = inquirer.get_next_prefill_step_cost_s(
            max_input_tokens=70,
            max_chunk_tokens=70,
            page_size=16,
            linear_tokens_per_second=100.0,
        )

        self.assertAlmostEqual(cost_s, 0.64)

    def test_accumulates_requests_until_natural_budget_is_full(self):
        inquirer = _inquirer(
            waiting_queue=[
                _req(seqlen=30),
                _req(seqlen=100, prefix_tokens=40),
            ]
        )

        cost_s = inquirer.get_next_prefill_step_cost_s(
            max_input_tokens=70,
            max_chunk_tokens=70,
            page_size=10,
            linear_tokens_per_second=100.0,
        )

        self.assertAlmostEqual(cost_s, 0.7)

    def test_non_prefill_mode_has_no_step_cost(self):
        inquirer = _inquirer(
            waiting_queue=[_req(seqlen=100)],
            disaggregation_mode=DisaggregationMode.DECODE,
        )

        cost_s = inquirer.get_next_prefill_step_cost_s(
            max_input_tokens=100,
            max_chunk_tokens=100,
            page_size=10,
            linear_tokens_per_second=100.0,
        )

        self.assertEqual(cost_s, 0.0)

    def test_rejects_invalid_service_rate_and_page_size(self):
        inquirer = _inquirer()
        with self.assertRaisesRegex(ValueError, "must be positive"):
            inquirer.get_next_prefill_step_cost_s(
                max_input_tokens=10,
                max_chunk_tokens=10,
                page_size=1,
                linear_tokens_per_second=0.0,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            inquirer.get_next_prefill_step_cost_s(
                max_input_tokens=10,
                max_chunk_tokens=10,
                page_size=0,
                linear_tokens_per_second=100.0,
            )


if __name__ == "__main__":
    unittest.main()
