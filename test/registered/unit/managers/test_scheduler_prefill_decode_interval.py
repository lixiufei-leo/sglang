"""Tests for gating the prefill/decode interval on the running batch."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.prefill_work_interval import PrefillWorkInterval
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(
    *,
    interval: int,
    min_bs: int,
    require_mlp_sync: bool = False,
    chunk_tokens: int = None,
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_decode_interval_min_bs = min_bs
    scheduler.require_mlp_sync = require_mlp_sync
    scheduler.prefill_work_interval = (
        None
        if require_mlp_sync
        else PrefillWorkInterval.maybe_build(
            interval=interval, chunk_tokens=chunk_tokens
        )
    )
    return scheduler


def _arm(scheduler: Scheduler, new_tokens: int = None) -> None:
    scheduler._arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=True,
            extend_num_tokens=new_tokens,
        )
    )


def _make_running_batch(batch_size: int, *, prefill_only: bool = False):
    return SimpleNamespace(
        batch_size=lambda: batch_size,
        is_empty=lambda: batch_size == 0,
        is_prefill_only=prefill_only,
    )


class TestPrefillDecodeIntervalMinBs(unittest.TestCase):
    def test_small_running_batch_admits_prefill_and_clears_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(7)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        # The cleared interval does not resume once the batch grows.
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(8)))

    def test_large_running_batch_defers_prefill(self):
        scheduler = _make_scheduler(interval=2, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(9)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(9)))

    def test_running_batch_shrinking_below_threshold_ends_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(5)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


class TestPrefillDecodeIntervalNothingDecodes(unittest.TestCase):
    # These use the default min_bs=0, where the interval applies at any batch size.

    def test_empty_running_batch_does_not_defer(self):
        # The next chunk of a long prompt on an idle server runs right away.
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))

    def test_prefill_only_running_batch_does_not_defer(self):
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        running = _make_running_batch(3, prefill_only=True)
        self.assertFalse(scheduler._should_defer_prefill(running))

    def test_single_decoding_request_keeps_interval(self):
        scheduler = _make_scheduler(interval=2, min_bs=0)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(1)))

    def test_dp_defers_without_local_decode(self):
        # Under DP attention, peer ranks may still be decoding.
        scheduler = _make_scheduler(interval=2, min_bs=0, require_mlp_sync=True)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(0)))
        running = _make_running_batch(3, prefill_only=True)
        self.assertTrue(scheduler._should_defer_prefill(running))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))


class TestPrefillDecodeIntervalByWork(unittest.TestCase):
    # With chunked prefill, a prefill batch above 3/4 of a chunk holds longer.

    def test_hold_grows_with_the_batch_new_tokens_and_never_shrinks(self):
        scheduler = _make_scheduler(interval=32, min_bs=0, chunk_tokens=16384)
        held = []
        for new_tokens in (1, 64, 8192, 12288, 12289, 14336, 16384, 32768):
            _arm(scheduler, new_tokens)
            held.append(scheduler._prefill_decode_interval_remaining)
        self.assertEqual(held, [32, 32, 32, 32, 33, 38, 43, 86])

    def test_scaled_hold_defers_that_many_rounds(self):
        scheduler = _make_scheduler(interval=32, min_bs=0, chunk_tokens=16384)
        _arm(scheduler, 16384)
        rounds = 0
        while scheduler._should_defer_prefill(_make_running_batch(3)):
            rounds += 1
        self.assertEqual(rounds, 43)

    def test_min_bs_bypass_and_nothing_decoding_still_clear_a_scaled_hold(self):
        scheduler = _make_scheduler(interval=32, min_bs=18, chunk_tokens=16384)
        _arm(scheduler, 16384)
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(17)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        _arm(scheduler, 16384)
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))

    def test_no_chunking_or_interval_keeps_the_fixed_interval(self):
        self.assertIsNone(
            PrefillWorkInterval.maybe_build(interval=0, chunk_tokens=16384)
        )
        scheduler = _make_scheduler(interval=32, min_bs=0, chunk_tokens=None)
        self.assertIsNone(scheduler.prefill_work_interval)
        _arm(scheduler, 1 << 20)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 32)

    def test_dp_attention_keeps_the_synchronized_interval(self):
        scheduler = _make_scheduler(
            interval=32, min_bs=0, chunk_tokens=16384, require_mlp_sync=True
        )
        _arm(scheduler, 16384)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 32)

    def test_env_switch_keeps_the_fixed_interval(self):
        with envs.SGLANG_ENABLE_PREFILL_DECODE_INTERVAL_SCALING.override(False):
            scheduler = _make_scheduler(interval=32, min_bs=0, chunk_tokens=16384)
        self.assertIsNone(scheduler.prefill_work_interval)
        _arm(scheduler, 16384)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 32)

    def test_startup_line_names_the_base_and_full_chunk_holds(self):
        interval = PrefillWorkInterval.maybe_build(interval=32, chunk_tokens=16384)
        line = interval.describe(16384)
        self.assertIn("32 decode rounds", line)
        self.assertIn("12288 new tokens", line)
        self.assertIn("43 after a full 16384-token chunk", line)


if __name__ == "__main__":
    unittest.main()
