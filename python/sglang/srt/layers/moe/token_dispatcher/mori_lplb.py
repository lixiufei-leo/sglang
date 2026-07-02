"""Mori-specific LPLB helpers for ROCm.

The generic LPLB solver uses CUDA JIT kernels backed by cuBLASDx. Mori on AMD
needs the same logical-to-physical probability interface, but the decode CUDA
graph capture path cannot call torch.linalg on HIP. This module therefore keeps
the Mori path graph-capture friendly by using regular tensor/index operations to
produce per-batch physical-expert probabilities.
"""

from __future__ import annotations

import torch


class MoriLPLBSolver:
    """Torch implementation of LPLBSolver for Mori/HIP."""

    def __init__(
        self,
        phy2log: torch.Tensor,
        log2phy: torch.Tensor,
        num_gpus: int,
        ep_group=None,
        logical_to_all_physical_map_num_valid=None,
    ):
        device = phy2log.device
        self.num_gpus = num_gpus
        self.ep_group = ep_group
        self.num_logical = log2phy.shape[0]
        self.max_copies = log2phy.shape[1]
        self.num_phy = phy2log.shape[0]

        if self.num_phy % num_gpus != 0:
            raise ValueError(
                f"MoriLPLBSolver requires num_phy ({self.num_phy}) to be divisible "
                f"by num_gpus ({num_gpus}); per-rank-contiguous ownership is "
                "currently the only supported allocation."
            )
        num_phy_per_gpu = self.num_phy // num_gpus

        logcnt = torch.bincount(phy2log, minlength=self.num_logical)
        self.log_single = torch.nonzero(logcnt == 1).flatten().to(torch.int64)
        self.phy_single = log2phy[self.log_single, 0].to(torch.int64)
        self.log_replicated = torch.nonzero(logcnt > 1).flatten().to(torch.int64)
        self.phy_replicated = (
            torch.nonzero(logcnt[phy2log] > 1).flatten().to(torch.int64)
        )

        self.num_single = len(self.log_single)
        self.num_red_log = len(self.log_replicated)
        self.num_red_phy = len(self.phy_replicated)

        b_full = torch.zeros(
            (num_gpus, self.num_phy), dtype=torch.float32, device=device
        )
        for i in range(num_gpus):
            b_full[i, i * num_phy_per_gpu : (i + 1) * num_phy_per_gpu] = 1
        self.B1 = b_full[:, self.phy_single].contiguous()
        b2 = b_full[:, self.phy_replicated]

        c = torch.zeros(
            (self.num_red_log, self.num_red_phy), dtype=torch.float32, device=device
        )
        phy2log_rep = phy2log[self.phy_replicated]
        for i in range(self.num_red_log):
            c[i, phy2log_rep == self.log_replicated[i]] = 1.0

        zeros_top_g = torch.zeros(
            (self.num_red_log, num_gpus), dtype=torch.float32, device=device
        )
        zeros_top_1 = torch.zeros(
            (self.num_red_log, 1), dtype=torch.float32, device=device
        )
        eye_g = torch.eye(num_gpus, dtype=torch.float32, device=device)
        neg_ones = torch.full((num_gpus, 1), -1.0, dtype=torch.float32, device=device)

        a_top = torch.hstack([c, zeros_top_g, zeros_top_1])
        a_bottom = torch.hstack([b2, eye_g, neg_ones])
        self.A_base = torch.vstack([a_top, a_bottom]).contiguous()
        self._A_base_row_sum = self.A_base.sum(dim=1).contiguous()

        nv = self.A_base.shape[1] + 1
        self.c_vec = torch.zeros(nv, dtype=torch.float32, device=device)
        self.c_vec[-2] = 1.0
        self.c_vec[-1] = 1000.0

        self.log2phy = log2phy.to(torch.int64).contiguous()
        self._log2phy_valid = (self.log2phy >= 0).to(torch.float32).contiguous()
        self._log2phy_owner = self.log2phy.clamp(min=0).div(
            num_phy_per_gpu, rounding_mode="floor"
        )

        self._counts_norm = torch.empty(self.num_logical, dtype=torch.float32, device=device)
        self._t1 = torch.empty(self.num_single, dtype=torch.float32, device=device)
        self._gpu_load = torch.empty(num_gpus, dtype=torch.float32, device=device)
        self._candidate_load = torch.empty(
            log2phy.shape, dtype=torch.float32, device=device
        )
        self._log2phy_prob = torch.empty(
            log2phy.shape, dtype=torch.float32, device=device
        )

    def solve(self, topk_ids: torch.Tensor) -> torch.Tensor:
        device = topk_ids.device
        local_counts = torch.zeros(self.num_logical, dtype=torch.int32, device=device)
        flat_ids = topk_ids.flatten()
        local_counts.scatter_add_(
            0,
            flat_ids.long(),
            torch.ones_like(flat_ids, dtype=torch.int32),
        )

        global_counts = local_counts.float()
        if self.ep_group is not None:
            global_counts = self.ep_group.all_reduce(global_counts)

        return self._solve(global_counts)

    def _solve(self, global_counts: torch.Tensor) -> torch.Tensor:
        torch.div(
            global_counts,
            global_counts.sum().clamp(min=1.0),
            out=self._counts_norm,
        )
        torch.index_select(self._counts_norm, 0, self.log_single, out=self._t1)
        torch.mv(self.B1, self._t1, out=self._gpu_load)

        # HIP graph capture rejects the torch.linalg-based IPM solver. Use a
        # graph-safe pressure heuristic: prefer physical copies on GPUs with
        # lower single-expert load, and fall back to uniform weights when loads
        # are equal.
        torch.take(self._gpu_load, self._log2phy_owner, out=self._candidate_load)
        self._candidate_load.add_(1e-6)
        torch.reciprocal(self._candidate_load, out=self._log2phy_prob)
        self._log2phy_prob.mul_(self._log2phy_valid)
        return self._log2phy_prob


def dispatch_probability_torch(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
) -> torch.Tensor:
    """Sample physical expert ids from LP probabilities with torch ops."""

    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    n = flat_ids.shape[0]
    num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)

    probs = log2phy_prob[flat_ids]
    maps = log2phy_map[flat_ids]
    row_sum = probs.sum(dim=-1, keepdim=True)
    fallback_probs = (maps >= 0).to(probs.dtype)
    probs = torch.where(row_sum > 0, probs, fallback_probs)
    row_sum = probs.sum(dim=-1)

    random_vals = torch.rand(n, dtype=torch.float32, device=topk_ids.device)
    u = (random_vals * row_sum).unsqueeze(-1)
    cum = probs.cumsum(dim=-1)
    chosen = (cum <= u).sum(dim=-1).clamp(max=max_copies - 1)

    out = maps.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    return out.view(original_shape).to(topk_ids.dtype)
