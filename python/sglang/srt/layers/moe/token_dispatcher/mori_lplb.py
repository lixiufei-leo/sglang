"""Mori-specific LPLB helpers for ROCm.

The generic :class:`LPLBSolver` solves the load-balancing LP with a fused
JIT CUDA kernel backed by cuBLASDx (Hopper+ / Math-DX), which is not
available on HIP. :class:`MoriLPLBSolver` keeps the *exact same* LP —
same constraint matrices, same barrier-method interior-point solve, same
probability extraction — but runs it through torch ops so it works on
AMD. It is a drop-in subclass of :class:`LPLBSolver`: only the solve
backend differs, so the math stays bit-for-bit aligned with the NVIDIA
path (modulo the LU-vs-Cholesky KKT factorization inside the IPM, exactly
as the NV torch reference already differs from its own CUDA kernel).

Enablement is driven purely by ``--ep-dispatch-algorithm lp`` (see
``ModelRunner._init_lplb_solvers``); there is no prefill/decode
auto-detection here. The torch IPM uses ``torch.linalg.solve`` and so is
not CUDA-graph-capturable — that is fine because the LP solve runs eager,
outside any captured region, on both NV and ROCm (its EP all-reduce
cannot live inside a compiled/captured graph anyway).
"""

from __future__ import annotations

import torch

from sglang.srt.eplb.lplb_solver import LPLBSolver


class MoriLPLBSolver(LPLBSolver):
    """Torch-backed LPLBSolver for Mori/HIP — exact LP, no CUDA kernels."""

    def __init__(
        self,
        phy2log: torch.Tensor,
        log2phy: torch.Tensor,
        num_gpus: int,
        ep_group=None,
        logical_to_all_physical_map_num_valid=None,
    ):
        # Reuse the base solver's constraint-matrix construction verbatim so the
        # LP is identical to the NVIDIA path. Only the solve backend differs.
        super().__init__(
            phy2log=phy2log,
            log2phy=log2phy,
            num_gpus=num_gpus,
            ep_group=ep_group,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
        )

        device = phy2log.device

        # ``lp_post`` scatters into a (num_phy + 1)-wide vector whose last slot
        # is an always-zero "sink" for padded (-1) replicas. Pre-allocate it and
        # a gather index that maps log2phy's -1 entries to that sink slot, so the
        # per-solve extraction is a single index_select into a reused buffer.
        self._phy_prob = torch.zeros(
            self.num_phy + 1, dtype=torch.float32, device=device
        )
        self._log2phy_gather = torch.where(
            self.log2phy < 0,
            torch.full_like(self.log2phy, self.num_phy),
            self.log2phy,
        ).reshape(-1)

    def _warmup_solver(self, nc: int, nv: int, device) -> None:
        # Torch IPM backend: nothing to JIT-compile.
        return

    def _solve(self, global_counts: torch.Tensor) -> torch.Tensor:
        """Exact LP solve in torch — mirrors the NV prep/IPM/post kernels.

        Steps map 1:1 to ``csrc/lplb/lp_prep.cuh``, ``csrc/lplb/ipm.cuh``
        (via :func:`solve_ipm_torch_reference`), and ``csrc/lplb/lp_post.cuh``.
        """
        from sglang.jit_kernel.lplb.torch_solver import solve_ipm_torch_reference

        # ---- prep (lp_prep.cuh) ----
        #   counts_norm = global_counts / total.clamp(min=1.0)
        #   t1 = counts_norm[log_single]; b1 = counts_norm[log_replicated]
        #   b2 = -(B1 @ t1); b = cat(b1, b2)
        #   A_full[:, -1] = b - A_base_row_sum   (first NV-1 cols are A_base)
        total = global_counts.sum().clamp(min=1.0)
        counts_norm = global_counts / total
        torch.index_select(counts_norm, 0, self.log_single, out=self._t1)
        b1 = counts_norm[self.log_replicated]
        b2 = -(self.B1 @ self._t1)
        b = torch.cat([b1, b2])
        self._A_full[:, -1] = b - self._A_base_row_sum

        # ---- IPM (ipm.cuh) ----
        x = solve_ipm_torch_reference(self._A_full, b, self.c_vec, num_iters=5)

        # ---- post (lp_post.cuh) ----
        #   phy_prob = 0; phy_prob[phy_replicated] = clamp(x[:num_red_phy], min=0)
        #   phy_prob[phy_single] = t1
        #   log2phy_prob = phy_prob[log2phy]   (-1 -> zero sink slot)
        self._phy_prob.zero_()
        self._phy_prob[self.phy_replicated] = x[: self.num_red_phy].clamp(min=0.0)
        self._phy_prob[self.phy_single] = self._t1
        torch.index_select(
            self._phy_prob,
            0,
            self._log2phy_gather,
            out=self._log2phy_prob.view(-1),
        )
        return self._log2phy_prob


def dispatch_probability_torch(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
) -> torch.Tensor:
    """Sample physical expert ids from LP probabilities with torch ops.

    Faithful torch port of the NV ``dispatch_probability`` CUDA kernel
    (inverse-CDF sampling); see
    ``jit_kernel.lplb.cuda_solver.dispatch_probability_torch_reference``.
    """

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
