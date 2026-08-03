import unittest

from sglang.srt.managers.dp_prefix_cache import DPPrefixCacheTracker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _tracker(
    *,
    device_pages: int = 8,
    host_pages: int = 0,
    storage_pages: int = 0,
    host_write_through: bool = True,
):
    return DPPrefixCacheTracker(
        dp_size=2,
        page_size=4,
        device_pages_per_rank=device_pages,
        host_pages_per_rank=host_pages,
        storage_pages_per_rank=storage_pages,
        host_write_through=host_write_through,
    )


class TestDPPrefixCacheTracker(CustomTestCase):
    def test_hits_are_rank_specific_and_page_aligned(self):
        tracker = _tracker()
        tracker.insert(1, list(range(13)))

        hits = tracker.estimate(list(range(17)))
        self.assertEqual(hits[0].cached_context_tokens, 0)
        self.assertEqual(hits[1].device_tokens, 12)
        self.assertEqual(hits[1].host_tokens, 0)

    def test_last_token_is_not_assumed_reusable(self):
        tracker = _tracker()
        tracker.insert(0, list(range(8)))

        hit = tracker.estimate(list(range(8)))[0]
        self.assertEqual(hit.device_tokens, 4)

    def test_namespace_separates_cache_salts(self):
        tracker = _tracker()
        tracker.insert(0, list(range(9)), namespace=b"tenant-a")

        self.assertEqual(
            tracker.estimate(list(range(9)), namespace=b"tenant-a")[0].device_tokens,
            8,
        )
        self.assertEqual(
            tracker.estimate(list(range(9)), namespace=b"tenant-b")[
                0
            ].cached_context_tokens,
            0,
        )

    def test_write_through_falls_back_to_host_after_device_eviction(self):
        tracker = _tracker(device_pages=1, host_pages=4, storage_pages=4)
        tracker.insert(0, list(range(13)))

        hit = tracker.estimate(list(range(13)))[0]
        self.assertEqual(hit.device_tokens, 0)
        self.assertEqual(hit.host_tokens, 12)
        self.assertEqual(hit.storage_tokens, 0)

    def test_host_eviction_falls_back_to_storage(self):
        tracker = _tracker(device_pages=1, host_pages=1, storage_pages=4)
        tracker.insert(0, list(range(13)))

        hit = tracker.estimate(list(range(13)))[0]
        self.assertEqual(hit.device_tokens, 0)
        self.assertEqual(hit.host_tokens, 0)
        self.assertEqual(hit.storage_tokens, 8)

    def test_estimate_does_not_mutate_rank_lru_state(self):
        tracker = _tracker(device_pages=4)
        first = list(range(9))
        second = list(range(100, 109))
        tracker.insert(0, first)
        tracker.insert(0, second)

        tracker.estimate(first)
        tracker.insert(0, list(range(200, 205)))

        self.assertEqual(
            tracker.estimate(first)[0].cached_context_tokens,
            0,
        )
        self.assertEqual(
            tracker.estimate(second)[0].device_tokens,
            8,
        )

    def test_promised_pages_are_separate_from_resident_hits(self):
        tracker = _tracker()
        tracker.promise(0, list(range(13)))

        hit = tracker.estimate(list(range(13)))[0]
        self.assertEqual(hit.cached_context_tokens, 0)
        self.assertEqual(hit.promised_tokens, 12)

    def test_materialization_promotes_promised_pages_to_resident(self):
        tracker = _tracker()
        tokens = list(range(13))
        tracker.promise(0, tokens)
        tracker.insert(0, tokens)

        hit = tracker.estimate(tokens)[0]
        self.assertEqual(hit.device_tokens, 12)
        self.assertEqual(hit.promised_tokens, 0)

    def test_clear_discards_all_rank_estimates(self):
        tracker = _tracker()
        resident = list(range(9))
        promised = list(range(100, 109))
        tracker.insert(0, resident)
        tracker.promise(0, promised)
        tracker.clear()

        self.assertEqual(
            tracker.estimate(resident)[0].cached_context_tokens,
            0,
        )
        self.assertEqual(
            tracker.estimate(promised)[0].promised_tokens,
            0,
        )


if __name__ == "__main__":
    unittest.main()
