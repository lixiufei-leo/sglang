"""DCP8 MLA decode merge as one LL push + LSE combine kernel on gfx950 (see
csrc/kimi_k3/comm/dcp_ll_combine_hip.cuh), over communication/k3_ll_comm_hip's
scratch: every rank pushes each peer's heads of its partials into the peer's
scratch and combines its own heads by LSE weight, in place of the pack +
all-to-all + combine sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.communication.k3_ll_comm_hip import (
    DCP_REGION_OFFSET,
    MAX_ROWS,
    WORLD_SIZE,
    LLComm,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match kHeads / kHeadDim in the kernel.
HEADS_PER_RANK: int = 12
HEAD_DIM: int = 128


@cache_once
def _module() -> Module:
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "kimi_k3_dcp_ll_combine",
        cuda_files=["kimi_k3/comm/dcp_ll_combine_hip.cuh"],
        cuda_wrappers=[
            ("run", "k3_dcp_ll_combine::K3DcpLLCombineKernel::run"),
        ],
        # AITER's CustomAllreduce, whose meta buffers hold the scratch
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def combine_covered(v: torch.Tensor, lse: torch.Tensor) -> bool:
    """Whether dcp_ll_combine takes partials ``v`` with LSE ``lse``. Shapes and
    dtypes only: every rank must come to the same answer."""
    return (
        v.dtype == torch.bfloat16
        and v.dim() == 3
        and 0 < v.shape[0] <= MAX_ROWS
        and v.shape[1] == WORLD_SIZE * HEADS_PER_RANK
        and v.shape[2] == HEAD_DIM
        and lse.dtype == torch.float32
        and lse.shape == v.shape[:2]
    )


def _is_vector_aligned(v: torch.Tensor) -> bool:
    return (
        v.stride(2) == 1
        and v.stride(0) % 8 == 0
        and v.stride(1) % 8 == 0
        and v.data_ptr() % 16 == 0
    )


def dcp_ll_combine(
    v: torch.Tensor,
    lse: torch.Tensor,
    comm: LLComm,
    *,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """Combine DCP partials ``v`` [B, 96, 128] bf16 with LSE ``lse`` [B, 96] fp32
    across the 8 ranks of ``comm`` (rank r owns heads [12 r, 12 r + 12)); returns
    this rank's heads [B, 12, 128] bf16. Every rank calls it with the same B, in
    the same order. The caller checked combine_covered()."""
    if not _is_vector_aligned(v):
        # The kernel loads 16-byte vectors; a fresh allocation is aligned.
        v = v.clone(memory_format=torch.contiguous_format)
    out = v.new_empty((v.shape[0], HEADS_PER_RANK, HEAD_DIM))
    _module().run(
        comm.comm._ptr,
        DCP_REGION_OFFSET,
        v,
        lse,
        out,
        comm.dcp_epochs,
        is_lse_base_on_e,
    )
    return out
