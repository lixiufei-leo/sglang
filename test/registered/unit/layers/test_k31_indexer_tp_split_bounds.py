"""Row bounds, KV floor and gather route of the Kimi-K3.1 prefill indexer TP split."""

import random
import unittest

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bounds(seq_lens, extend_lens, tp):
    return Indexer._indexer_tp_split_bounds(
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_lens=extend_lens,
        q_offset=sum(extend_lens),
        tp=tp,
    )


class TestIndexerTpSplitBounds(CustomTestCase):
    def test_every_rank_owns_a_contiguous_nonempty_slice(self):
        # The gather rebuilds rows [0, q_offset) in rank order, so the bounds must
        # tile the rows exactly with no empty rank, down to q_offset == tp.
        rng = random.Random(0)
        cases = [([143424], [64]), ([100], [8]), ([200000, 40, 40], [6, 1, 1])]
        for _ in range(200):
            n = rng.randint(1, 6)
            ext = [rng.randint(1, 300) for _ in range(n)]
            cases.append(([e + rng.randint(0, 300000) for e in ext], ext))
        for seq, ext in cases:
            if sum(ext) < 8:
                continue
            bounds = _bounds(seq, ext, tp=8)
            with self.subTest(seq=seq, ext=ext):
                self.assertEqual(len(bounds), 9)
                self.assertEqual(bounds[0], 0)
                self.assertEqual(bounds[-1], sum(ext))
                self.assertTrue(all(a < b for a, b in zip(bounds, bounds[1:])))

    def test_slices_balance_kv_length_not_row_count(self):
        # One long-context request next to short ones: the long rows cost the most
        # logits and top-k work, so they must be spread over more ranks.
        seq, ext = [140000, 600, 600], [64, 64, 64]
        bounds = _bounds(seq, ext, tp=8)
        rows_on_long_request = sum(1 for b in bounds[1:-1] if b < 64)
        self.assertGreaterEqual(rows_on_long_request, 6)


class TestIndexerTpSplitPlan(CustomTestCase):
    def _plan(self, seq, ext, min_kv_rows, tp=8):
        return dsa_indexer._indexer_tp_split_plan(
            seq_lens=tuple(seq),
            extend_lens=tuple(ext),
            q_offset=sum(ext),
            tp=tp,
            min_kv_rows=min_kv_rows,
        )

    def test_plan_is_the_bounds_once_the_batch_has_enough_kv(self):
        for seq, ext in [([160000], [729]), ([120000, 190000], [400, 329])]:
            with self.subTest(seq=seq, ext=ext):
                self.assertEqual(
                    list(self._plan(seq, ext, 16_000_000)), _bounds(seq, ext, tp=8)
                )
                self.assertEqual(list(self._plan(seq, ext, 0)), _bounds(seq, ext, tp=8))

    def test_too_little_kv_stays_unsplit(self):
        # 64 rows x 160k keys, a cold 4096-row extend, short rows next to one long row
        cases = [([160000], [64]), ([4096], [4096]), ([200000, 3000], [1, 300])]
        for seq, ext in cases:
            with self.subTest(seq=seq, ext=ext):
                self.assertIsNone(self._plan(seq, ext, 16_000_000))
                self.assertIsNotNone(self._plan(seq, ext, 0))

    def test_kv_rows_count_each_rows_own_kv_length(self):
        # rows see 9, 10 keys and 1, 2, 3 keys: 25 KV rows in all
        self.assertIsNotNone(self._plan([10, 3], [2, 3], 25, tp=2))
        self.assertIsNone(self._plan([10, 3], [2, 3], 26, tp=2))


class TestIndexerTpGatherSlot(CustomTestCase):
    def _slot(self, sizes, has_pynccl=True):
        return Indexer._indexer_tp_gather_slot(
            sizes=sizes, row_bytes=2048 * 4, has_pynccl=has_pynccl
        )

    def test_near_equal_sizes_gather_in_equal_strides(self):
        self.assertEqual(self._slot([91, 91, 91, 91, 92, 91, 91, 91]), 92)
        sizes = [2191, 2146, 2102, 2062, 2023, 1987, 1953, 1920]
        self.assertEqual(self._slot(sizes), 2191)

    def test_skewed_sizes_gather_exact_sizes_when_pynccl_can(self):
        skewed = [5792, 2399, 1841, 1553, 1367, 1236, 1137, 1059]
        self.assertIsNone(self._slot(skewed))
        self.assertEqual(self._slot(skewed, has_pynccl=False), 5792)

    def test_padding_bound_is_in_bytes(self):
        bound = dsa_indexer._INDEXER_TP_GATHER_MAX_PAD_BYTES // (2048 * 4)
        self.assertEqual(self._slot([bound + 1, 1]), bound + 1)
        self.assertIsNone(self._slot([bound + 2, 1]))


if __name__ == "__main__":
    unittest.main()
