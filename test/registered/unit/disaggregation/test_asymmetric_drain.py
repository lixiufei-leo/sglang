"""Unit tests for the asymmetric-drain extension of the PD role switch.

Style mirrors the PR's own test/registered/unit/disaggregation/test_pd_role_switch.py:
no GPU, the heavy teardown/rebuild/migration is mocked, so this asserts the
control-plane contract of the two drain policies:

  - prefill -> decode: "graceful" — if busy, schedule a pending switch (do NOT
    flip yet); the flip fires later from _maybe_complete_pending_switch() once
    the instance drains to idle.
  - decode -> prefill: "migrate" — migrate in-flight decode requests to another
    node, then flip immediately.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    PdRoleSwitchReqInput,
    PdRoleSwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
except Exception:  # pragma: no cover - CI helper not present in all builds
    pass


def _scheduler(mode, *, enable=True, idle=True):
    s = Scheduler.__new__(Scheduler)
    s.disaggregation_mode = mode
    s.server_args = SimpleNamespace(
        enable_pd_role_switch=enable,
        disaggregation_mode=mode.value,
    )
    s.is_fully_idle = MagicMock(return_value=idle)
    s._do_flip = MagicMock()
    s._teardown_disaggregation = MagicMock()
    s.init_disaggregation = MagicMock()
    s._migrate_inflight_decode = MagicMock(return_value=(0, 0))
    s._migrate_kv_inflight_decode = MagicMock(return_value=(0, 0))
    s._event_loop_should_restart = False
    s._pd_switch_pending = None
    return s


class TestGracefulDrainPrefill(unittest.TestCase):
    def test_idle_prefill_flips_immediately(self):
        s = _scheduler(DisaggregationMode.PREFILL, idle=True)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode", drain_policy="graceful")
        )
        self.assertTrue(out.success)
        self.assertTrue(out.flipped)
        self.assertTrue(out.drained)
        s._do_flip.assert_called_once_with("decode")
        self.assertIsNone(s._pd_switch_pending)

    def test_busy_prefill_schedules_pending_without_flipping(self):
        s = _scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode", drain_policy="graceful")
        )
        # Accepted but NOT flipped yet: the loop will finish in-flight work first.
        self.assertTrue(out.success)
        self.assertFalse(out.flipped)
        self.assertFalse(out.drained)
        s._do_flip.assert_not_called()
        self.assertIsNotNone(s._pd_switch_pending)
        self.assertEqual(s._pd_switch_pending[0], "decode")
        self.assertEqual(s._pd_switch_pending[1], "graceful")

    def test_pending_completes_when_drained(self):
        s = _scheduler(DisaggregationMode.PREFILL, idle=False)
        Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode", drain_policy="graceful")
        )
        # Still busy -> no flip.
        Scheduler._maybe_complete_pending_switch(s)
        s._do_flip.assert_not_called()
        # Drains -> flip fires exactly once, pending cleared.
        s.is_fully_idle.return_value = True
        Scheduler._maybe_complete_pending_switch(s)
        s._do_flip.assert_called_once_with("decode")
        self.assertIsNone(s._pd_switch_pending)

    def test_maybe_complete_noop_without_pending(self):
        s = _scheduler(DisaggregationMode.PREFILL, idle=True)
        Scheduler._maybe_complete_pending_switch(s)
        s._do_flip.assert_not_called()


class TestMigrateDecode(unittest.TestCase):
    def test_busy_migrate_moves_then_schedules_flip(self):
        s = _scheduler(DisaggregationMode.DECODE, idle=False)
        s._migrate_inflight_decode = MagicMock(return_value=(3, 1))
        out = Scheduler.handle_pd_role_switch(
            s,
            PdRoleSwitchReqInput(
                new_role="prefill",
                drain_policy="migrate",
                migrate_url="http://router:8000",
            ),
        )
        # Migration happens synchronously in the call; the flip is scheduled and
        # fires once the forced drain reaches idle (near-immediate).
        self.assertTrue(out.success)
        self.assertFalse(out.flipped)
        self.assertEqual(out.migrated, 3)
        self.assertEqual(out.aborted, 1)
        s._migrate_inflight_decode.assert_called_once_with("http://router:8000")
        s._do_flip.assert_not_called()
        self.assertIsNotNone(s._pd_switch_pending)
        self.assertEqual(s._pd_switch_pending[0], "prefill")
        self.assertEqual(s._pd_switch_pending[1], "migrate")

    def test_migrate_completes_after_forced_drain(self):
        s = _scheduler(DisaggregationMode.DECODE, idle=False)
        s._migrate_inflight_decode = MagicMock(return_value=(2, 0))
        Scheduler.handle_pd_role_switch(
            s,
            PdRoleSwitchReqInput(
                new_role="prefill", drain_policy="migrate", migrate_url="http://r:8"
            ),
        )
        s.is_fully_idle.return_value = True
        Scheduler._maybe_complete_pending_switch(s)
        s._do_flip.assert_called_once_with("prefill")
        self.assertIsNone(s._pd_switch_pending)

    def test_idle_decode_migrate_flips_immediately(self):
        s = _scheduler(DisaggregationMode.DECODE, idle=True)
        out = Scheduler.handle_pd_role_switch(
            s,
            PdRoleSwitchReqInput(new_role="prefill", drain_policy="migrate"),
        )
        self.assertTrue(out.success)
        self.assertTrue(out.flipped)
        # Nothing in flight -> no migration attempted.
        s._migrate_inflight_decode.assert_not_called()
        s._do_flip.assert_called_once_with("prefill")


class TestAdmissionGate(unittest.TestCase):
    """While a drain is pending, _add_request_to_queue must refuse new work
    (bounce with a retryable abort) instead of enqueuing it, so the instance
    can reach idle."""

    def _sched(self, pending):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s._pd_switch_pending = pending
        s.waiting_queue = []
        s._set_or_validate_priority = MagicMock(return_value=True)
        s._abort_on_queued_limit = MagicMock(return_value=False)
        s._prefetch_kvcache = MagicMock()
        s.disagg_prefill_bootstrap_queue = MagicMock()
        s.model_config = SimpleNamespace(num_key_value_heads=1)
        s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        return s

    def test_gate_bounces_new_request_while_pending(self):
        s = self._sched(pending=("decode", "graceful", ""))
        req = SimpleNamespace(rid="r1", time_stats=MagicMock())
        Scheduler._add_request_to_queue(s, req)
        # Not enqueued anywhere; an abort was sent back to the tokenizer.
        self.assertEqual(len(s.waiting_queue), 0)
        s.disagg_prefill_bootstrap_queue.add.assert_not_called()
        s.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
        sent = s.ipc_channels.send_to_tokenizer.send_output.call_args[0][0]
        self.assertEqual(getattr(sent, "rid", None), "r1")

    def test_gate_open_when_not_pending(self):
        s = self._sched(pending=None)
        req = SimpleNamespace(rid="r2", time_stats=MagicMock())
        Scheduler._add_request_to_queue(s, req)
        # Normal path: routed into the prefill bootstrap queue, no abort sent.
        s.disagg_prefill_bootstrap_queue.add.assert_called_once()
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_gate_allows_retracted_even_while_pending(self):
        s = self._sched(pending=("decode", "graceful", ""))
        req = SimpleNamespace(rid="r3", time_stats=MagicMock())
        Scheduler._add_request_to_queue(s, req, is_retracted=True)
        # Retracted (already-owned) reqs are not new work; must not be bounced.
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


class TestMigrateKvDecode(unittest.TestCase):
    """drain_policy=migrate_kv ships existing KV to another decode node (no
    re-prefill) via _migrate_kv_inflight_decode, then schedules the flip."""

    def test_busy_migrate_kv_ships_then_schedules_flip(self):
        s = _scheduler(DisaggregationMode.DECODE, idle=False)
        s._migrate_kv_inflight_decode = MagicMock(return_value=(2, 1))
        out = Scheduler.handle_pd_role_switch(
            s,
            PdRoleSwitchReqInput(
                new_role="prefill",
                drain_policy="migrate_kv",
                migrate_url="http://decode-b:30030",
            ),
        )
        self.assertTrue(out.success)
        self.assertFalse(out.flipped)
        self.assertEqual(out.migrated, 2)
        self.assertEqual(out.aborted, 1)
        s._migrate_kv_inflight_decode.assert_called_once_with("http://decode-b:30030")
        # the re-prefill migrate path must NOT be used
        s._migrate_inflight_decode.assert_not_called()
        s._do_flip.assert_not_called()
        self.assertEqual(s._pd_switch_pending[1], "migrate_kv")

    def test_migrate_kv_completes_after_drain(self):
        s = _scheduler(DisaggregationMode.DECODE, idle=False)
        s._migrate_kv_inflight_decode = MagicMock(return_value=(1, 0))
        Scheduler.handle_pd_role_switch(
            s,
            PdRoleSwitchReqInput(
                new_role="prefill", drain_policy="migrate_kv", migrate_url="http://d:1"
            ),
        )
        s.is_fully_idle.return_value = True
        Scheduler._maybe_complete_pending_switch(s)
        s._do_flip.assert_called_once_with("prefill")


class TestRejectPolicyUnchanged(unittest.TestCase):
    def test_reject_busy_fails(self):
        s = _scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode", drain_policy="reject")
        )
        self.assertFalse(out.success)
        self.assertIn("not idle", out.message)
        s._do_flip.assert_not_called()

    def test_reject_is_default_policy(self):
        s = _scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        s._do_flip.assert_not_called()


class TestGuardsStillApply(unittest.TestCase):
    def test_flag_disabled(self):
        s = _scheduler(DisaggregationMode.PREFILL, enable=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode", drain_policy="graceful")
        )
        self.assertFalse(out.success)
        self.assertIn("enable-pd-role-switch", out.message)

    def test_invalid_role(self):
        s = _scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="both", drain_policy="graceful")
        )
        self.assertFalse(out.success)
        self.assertIn("invalid new_role", out.message)

    def test_same_role_noop(self):
        s = _scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill", drain_policy="graceful")
        )
        self.assertTrue(out.success)
        self.assertFalse(out.flipped)
        s._do_flip.assert_not_called()

    def test_invalid_drain_policy(self):
        s = _scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode", drain_policy="bogus")
        )
        self.assertFalse(out.success)
        self.assertIn("drain_policy", out.message)


if __name__ == "__main__":
    unittest.main()
