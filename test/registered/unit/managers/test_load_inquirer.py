import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.dp_cost_model import PrefillCostEstimate
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch(reqs, *, decoding_reqs=None, is_extend=True):
    return SimpleNamespace(
        reqs=reqs,
        decoding_reqs=decoding_reqs,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: is_extend),
    )


def _req(*, seqlen=128, prefix_tokens=0, finished=False):
    return SimpleNamespace(
        seqlen=seqlen,
        prefix_indices=[0] * prefix_tokens,
        storage_prefetch_tokens=0,
        host_hit_length=0,
        storage_hit_length=0,
        swa_host_hit_length=0,
        finished=lambda: finished,
    )


def _inquirer(
    *,
    model=None,
    running_batch=None,
    current_batch=None,
    inflight_batch=None,
    waiting_queue=None,
    chunked_req=None,
):
    empty_batch = _batch([])
    return SchedulerLoadInquirer(
        disaggregation_mode=DisaggregationMode.NULL,
        ps=None,
        server_args=None,
        max_total_num_tokens=0,
        max_running_requests=0,
        pool_stats_observer=None,
        tp_worker=None,
        token_to_kv_pool_allocator=None,
        spec_algorithm=None,
        prefill_cost_model=model,
        get_running_batch=lambda: running_batch or empty_batch,
        get_current_batch=lambda: current_batch,
        get_inflight_prefill_batch=lambda: inflight_batch,
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
        get_last_accepted_dispatch_seq=lambda: 0,
        get_retired_dp_prefill_cost_s=lambda: 0.0,
    )


class TestSchedulerLoadInquirerPrefillCost(CustomTestCase):
    def test_inflight_prefill_is_counted_once_when_also_chunked(self):
        model = MagicMock()
        model.estimate.return_value = PrefillCostEstimate(csa_indexer_seconds=0.5)
        req = _req(prefix_tokens=32)
        batch = _batch([req])
        inquirer = _inquirer(
            model=model,
            inflight_batch=batch,
            chunked_req=req,
        )

        cost = inquirer.get_prefill_cost_estimate()

        self.assertAlmostEqual(cost.total_seconds, 0.5)
        self.assertEqual(
            model.estimate.call_args_list,
            [call(input_tokens=128, cached_context_tokens=32)],
        )

    def test_mixed_batch_decode_requests_are_not_charged_as_prefill(self):
        model = MagicMock()
        model.estimate.return_value = PrefillCostEstimate(csa_indexer_seconds=0.5)
        prefill_req = _req(prefix_tokens=16)
        decode_req = _req(seqlen=256, prefix_tokens=255)
        batch = _batch(
            [prefill_req, decode_req],
            decoding_reqs=[decode_req],
        )
        inquirer = _inquirer(model=model, inflight_batch=batch)

        cost = inquirer.get_prefill_cost_estimate()

        self.assertAlmostEqual(cost.total_seconds, 0.5)
        model.estimate.assert_called_once_with(
            input_tokens=128,
            cached_context_tokens=16,
        )


class TestSchedulerLoadInquirerRunningCount(CustomTestCase):
    def test_current_batch_closes_prefill_count_gap_and_deduplicates(self):
        running_req = _req()
        current_prefill_req = _req()
        finished_req = _req(finished=True)
        inquirer = _inquirer(
            running_batch=_batch([running_req]),
            current_batch=_batch(
                [running_req, current_prefill_req, finished_req]
            ),
        )

        self.assertEqual(inquirer.get_num_running_reqs(), 2)


if __name__ == "__main__":
    unittest.main()