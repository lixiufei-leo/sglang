import unittest

import torch

from sglang.srt.layers.moe.token_dispatcher.mori_lplb import (
    MoriLPLBSolver,
    dispatch_probability_torch,
)


class TestMoriLPLB(unittest.TestCase):
    def test_dispatch_probability_torch_uses_valid_replicas(self):
        topk_ids = torch.tensor([[0, 1, 0]], dtype=torch.int32)
        log2phy_prob = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        log2phy_map = torch.tensor(
            [
                [0, 4, -1],
                [1, -1, -1],
            ],
            dtype=torch.int64,
        )

        out = dispatch_probability_torch(topk_ids, log2phy_prob, log2phy_map)

        self.assertEqual(out.shape, topk_ids.shape)
        self.assertEqual(out.dtype, topk_ids.dtype)
        self.assertTrue(torch.equal(out[:, [0, 2]], torch.tensor([[4, 4]])))
        self.assertEqual(out[0, 1].item(), 1)

    def test_mori_lplb_solver_returns_logical_to_physical_probabilities(self):
        phy2log = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
        log2phy = torch.tensor(
            [
                [0, 2],
                [1, -1],
                [3, -1],
            ],
            dtype=torch.int64,
        )
        num_valid = torch.tensor([2, 1, 1], dtype=torch.int64)
        solver = MoriLPLBSolver(
            phy2log=phy2log,
            log2phy=log2phy,
            num_gpus=2,
            logical_to_all_physical_map_num_valid=num_valid,
        )

        out = solver.solve(torch.tensor([[0, 1], [2, 0]], dtype=torch.int32))

        self.assertEqual(out.shape, log2phy.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out >= 0).all())
        self.assertEqual(out[1, 1].item(), 0.0)
        self.assertEqual(out[2, 1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
