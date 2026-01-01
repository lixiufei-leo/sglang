"""Multi-GPU test of the one-kernel DCP merge on gfx950 (dcp_ll_combine_hip):
every rank holds [B, 96, 128] partials of all heads with their LSE and gets back
the LSE combine of its own 12 heads, bit for bit what gathering every rank's
partials and running the Triton combine gives. Covers payload words that are
+0.0 (they travel marked, an all-zero line means "not arrived"), rows with no
local key (+inf / NaN LSE) or no key at all, both LSE bases, strided and
unaligned partials, and repeated calls and graph replays through both epochs of
the scratch.

Usage::

    python test/registered/kernels/ops/attention/test_dcp_ll_combine_hip.py   # relaunches under torchrun (8 GPUs)
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention import dcp_ll_combine_hip as ll
from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.communication.k3_ll_comm_hip import (
    MAX_ROWS,
    LLComm,
    create_ll_comm,
)
from sglang.srt.runtime_context import SpawnRanks, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.parallel_groups import publish

register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")

H = ll.HEADS_PER_RANK * 8


@cache_once
def _setup() -> tuple[ps.GroupCoordinator, LLComm]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    publish(
        ServerArgs(model_path="dummy", tp_size=world_size),
        role="test",
        ranks=SpawnRanks(world_rank=local_rank),
    )
    coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    ps._WORLD = coord
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    comm = create_ll_comm(coord.cpu_group, torch.device(f"cuda:{local_rank}"))
    assert comm is not None
    return coord, comm


def _reference(coord, v, lse, is_lse_base_on_e=True):
    n, rank = coord.world_size, coord.rank_in_group
    hpr = v.shape[1] // n
    all_v = [torch.empty_like(v) for _ in range(n)]
    all_lse = [torch.empty_like(lse) for _ in range(n)]
    dist.all_gather(all_v, v, group=coord.device_group)
    dist.all_gather(all_lse, lse, group=coord.device_group)
    recv_v = torch.stack([x[:, rank * hpr : (rank + 1) * hpr] for x in all_v])
    recv_lse = torch.stack([x[:, rank * hpr : (rank + 1) * hpr] for x in all_lse])
    out, _ = dcp_lse_combine_triton(
        recv_v.contiguous(), recv_lse.contiguous(), is_lse_base_on_e=is_lse_base_on_e
    )
    return out


def _assert_same_bits(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def _inputs(batch: int, g: torch.Generator, rank: int):
    v = torch.randn(batch, H, ll.HEAD_DIM, device="cuda", generator=g) * (rank + 1)
    v = v.to(torch.bfloat16)
    v[:, :, :4] = 0  # +0.0 payload words must not read as "not arrived"
    v[:, ::7, 4:8] = -0.0
    lse = torch.randn(batch, H, device="cuda", generator=g) * 4
    lse[::5, rank] = float("inf")  # no local key
    lse[::3, (rank + 1) % H] = float("nan")
    lse[batch // 2, ::2] = float("-inf")  # heads no rank holds a key for
    return v, lse


@pytest.mark.parametrize("batch", [1, 7, 100, MAX_ROWS])
@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_matches_gather_and_combine(batch, is_lse_base_on_e):
    coord, comm = _setup()
    rank = coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(batch * 131 + rank)
    for call in range(3):  # odd and even epochs, then a rewritten line
        v, lse = _inputs(batch, g, rank)
        if call == 1:
            lse[: min(batch, 3)] = 0.0
        assert ll.combine_covered(v, lse)
        out = ll.dcp_ll_combine(v, lse, comm, is_lse_base_on_e=is_lse_base_on_e)
        ref = _reference(coord, v, lse, is_lse_base_on_e)
        assert torch.isfinite(out).all()
        _assert_same_bits(out, ref)


def test_changing_row_counts():
    """Rows keep their own epoch: a call with fewer rows leaves the others' alone."""
    coord, comm = _setup()
    rank = coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(5 + rank)
    for batch in (14, 70, 5, 100, 14, 1, MAX_ROWS, 20):
        v, lse = _inputs(batch, g, rank)
        out = ll.dcp_ll_combine(v, lse, comm)
        _assert_same_bits(out, _reference(coord, v, lse))


def test_strided_and_unaligned_inputs():
    """Head-major partials, an LSE slice of a wider buffer, and partials that
    start 4 bytes into their storage."""
    coord, comm = _setup()
    rank = coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(11 + rank)
    batch = 64
    v = torch.randn(H, batch, ll.HEAD_DIM, device="cuda", generator=g)
    v = v.to(torch.bfloat16).transpose(0, 1)
    lse = (torch.randn(batch, H + 32, device="cuda", generator=g) * 4)[:, :H]
    out = ll.dcp_ll_combine(v, lse, comm)
    _assert_same_bits(out, _reference(coord, v.contiguous(), lse.contiguous()))

    flat = torch.zeros(v.numel() + 8, dtype=v.dtype, device="cuda")
    flat[2 : 2 + v.numel()] = v.reshape(-1)
    shifted = flat[2 : 2 + v.numel()].view(v.shape)
    assert shifted.data_ptr() % 16 != 0
    out = ll.dcp_ll_combine(shifted, lse, comm)
    _assert_same_bits(out, _reference(coord, v.contiguous(), lse.contiguous()))


def test_cuda_graph_replay():
    coord, comm = _setup()
    rank = coord.rank_in_group
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    v, lse = _inputs(64, g, rank)
    ll.dcp_ll_combine(v, lse, comm)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [ll.dcp_ll_combine(v, lse, comm) for _ in range(3)]
    for _ in range(5):  # an odd number of merges a replay: the epochs alternate
        new_v, new_lse = _inputs(64, g, rank)
        v.copy_(new_v)
        lse.copy_(new_lse)
        graph.replay()
        ref = _reference(coord, v, lse)
        for out in outs:
            _assert_same_bits(out, ref)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[8])
