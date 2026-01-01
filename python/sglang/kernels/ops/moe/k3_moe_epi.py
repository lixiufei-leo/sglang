# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode MoE front consumers on ROCm, regrouped by dependency (adapted
from sgl-project/sglang#46652's k3_moe_epi):

  before:  [SiTU + routed cast] -> [shared down GEMM] -> [router top-k]
  after:   [SiTU + routed cast] -> [shared down GEMM | router top-k]

The top-k reads only the fused front's router logits, so it rides the shared
down GEMM's launch as independent programs (one per token), next to the GEMM
tiles that read the SiTU activation.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _situ(g, u, beta, inv_beta, lin_beta, inv_lin_beta, HAS_LIN: tl.constexpr):
    gate = beta * libdevice.tanh(g * inv_beta) * (1.0 / (1.0 + tl.exp(-g)))
    if HAS_LIN:
        u = lin_beta * libdevice.tanh(u * inv_lin_beta)
    return gate * u


@triton.jit
def _aiter_sigmoid(x):
    # aiter topk_reg: __builtin_amdgcn_rcpf(1.0f + exp2f(-C_LOG2E * x)), where
    # C_LOG2E is a double literal (the product is formed in fp64)
    a = (x.to(tl.float64) * -1.44269504088896340736).to(tl.float32)
    ex = tl.inline_asm_elementwise(
        "v_exp_f32 $0, $1", "=v,v", [a], dtype=tl.float32, is_pure=True, pack=1
    )
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [1.0 + ex], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _aiter_tie_rank(idx):
    # aiter legacy_topk_tie_rank (wave64 DPP order): lower wins among equal scores
    vec4 = idx >> 2
    lane = vec4 & 63
    local = ((vec4 >> 6) << 2) | (idx & 3)
    low2 = (lane & 3) ^ tl.where((lane & 4) != 0, 3, 0)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    lane_rank = ((~lane) & 0x38) | bit2 | low2
    return (lane_rank << 5) | local


@triton.jit
def _aiter_biased_topk(
    logits_ptr,
    bias_ptr,
    offs_e,
    me,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    routed_scale,
):
    """aiter biased_grouped_topk (topk_reg_kernel, one group): experts by
    sigmoid(logit) + bias, weights sigmoid(logit) renormalized."""
    lg = tl.load(logits_ptr + offs_e, mask=me, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_e, mask=me, other=0.0)
    # biased_grouped_topk_gpu keeps an fp32 bias, else routes it in the logits' dtype
    if bias_ptr.dtype.element_ty != tl.float32:
        bias = bias.to(logits_ptr.dtype.element_ty)
    bias = bias.to(tl.float32)
    sig = _aiter_sigmoid(lg)
    sc = tl.where(me, sig + bias, float("-inf"))
    b = sc.to(tl.int32, bitcast=True)
    k = tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.int64)
    key = (
        (k << 32)
        | ((0xFFFF - _aiter_tie_rank(offs_e)).to(tl.int64) << 16)
        | offs_e.to(tl.int64)
    )
    top = tl.topk(key, TOPK, dim=0)  # descending
    ids = (top & 0xFFFF).to(tl.int32)
    w = _aiter_sigmoid(tl.load(logits_ptr + ids).to(tl.float32))
    offs_k = tl.arange(0, TOPK)
    if RENORM:
        total = 0.0
        for i in tl.static_range(TOPK):
            total += tl.sum(tl.where(offs_k == i, w, 0.0), axis=0)
        f = tl.where(total > 0.0, routed_scale / total, 0.0)
        w = w * f
    else:
        w = w * routed_scale
    return ids, w


@triton.jit
def _shared_down_tile(
    pid,
    act_ptr,  # [M, I] activation, dense
    wd_ptr,  # [N, I] contiguous
    out_ptr,  # [M, N], row stride stride_out
    stride_out,
    M,
    N: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # one (BM, BN) tile of out = act @ wd^T
    NT_N: tl.constexpr = N // BN
    pid_n = pid % NT_N
    pid_m = pid // NT_N
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, I, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(wd_ptr + offs_n[:, None] * I + offs_k[None, :])
        a = tl.load(
            act_ptr + offs_m[:, None] * I + offs_k[None, :], mask=mm[:, None], other=0.0
        )
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(
        out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mm[:, None],
    )


@triton.jit(
    do_not_specialize=["M", "stride_gu", "stride_out", "stride_src", "stride_lg"]
)
def _k3_moe_front_epi_kernel(
    gu_ptr,  # [M, 2*I] fp32/bf16 gate | up, row stride stride_gu (DO_SITU)
    stride_gu,
    act_ptr,  # [M, I] dense SiTU activation: written (DO_SITU) or read (DO_GEMM)
    wd_ptr,  # [N, I] shared down weight, contiguous
    out_ptr,  # [M, N] shared output, row stride stride_out
    stride_out,
    src_ptr,  # [M, C] routed input, row stride stride_src (DO_CAST)
    stride_src,
    dst_ptr,  # [M, C] dense cast of src (DO_CAST)
    lg_ptr,  # [M, E] router logits, row stride stride_lg (DO_TOPK)
    stride_lg,
    bias_ptr,  # [E] correction bias
    topk_ids_ptr,  # [M, TOPK] i32 out
    topk_w_ptr,  # [M, TOPK] fp32 out
    routed_scale,
    M,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    DO_TOPK: tl.constexpr,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    DO_GEMM: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DO_SITU: tl.constexpr,
    SITU_BLOCK: tl.constexpr,
    DO_CAST: tl.constexpr,
    C: tl.constexpr,
    CBLOCK: tl.constexpr,
):
    # independent programs, in order: top-k (one per token), GEMM tiles,
    # SiTU (token, block), cast (token, block)
    pid = tl.program_id(0)
    n_topk = M if DO_TOPK else 0
    n_gemm = (N // BN) * tl.cdiv(M, BM) if DO_GEMM else 0
    NB: tl.constexpr = (I + SITU_BLOCK - 1) // SITU_BLOCK
    n_situ = M * NB if DO_SITU else 0
    NCB: tl.constexpr = (C + CBLOCK - 1) // CBLOCK
    if pid < n_topk:
        offs_e = tl.arange(0, E_POW2)
        ids, wt = _aiter_biased_topk(
            lg_ptr + pid * stride_lg,
            bias_ptr,
            offs_e,
            offs_e < E,
            TOPK,
            RENORM,
            routed_scale,
        )
        offs_k = tl.arange(0, TOPK)
        tl.store(topk_ids_ptr + pid * TOPK + offs_k, ids)
        tl.store(topk_w_ptr + pid * TOPK + offs_k, wt)
    elif pid < n_topk + n_gemm:
        _shared_down_tile(
            pid - n_topk, act_ptr, wd_ptr, out_ptr, stride_out, M, N, I, BM, BN, BK
        )
    elif pid < n_topk + n_gemm + n_situ:
        q = pid - n_topk - n_gemm
        m = q // NB
        offs = (q % NB) * SITU_BLOCK + tl.arange(0, SITU_BLOCK)
        mk = offs < I
        g = tl.load(gu_ptr + m * stride_gu + offs, mask=mk, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + m * stride_gu + I + offs, mask=mk, other=0.0).to(
            tl.float32
        )
        a = _situ(g, u, beta, inv_beta, lin_beta, inv_lin_beta, HAS_LIN)
        tl.store(act_ptr + m * I + offs, a.to(act_ptr.dtype.element_ty), mask=mk)
    elif DO_CAST:
        qc = pid - n_topk - n_gemm - n_situ
        mc = qc // NCB
        offs_c = (qc % NCB) * CBLOCK + tl.arange(0, CBLOCK)
        mask_c = offs_c < C
        x = tl.load(src_ptr + mc * stride_src + offs_c, mask=mask_c, other=0.0)
        tl.store(
            dst_ptr + mc * C + offs_c, x.to(dst_ptr.dtype.element_ty), mask=mask_c
        )


class Router(NamedTuple):
    """Inputs and outputs of the sigmoid + correction-bias top-k."""

    logits: torch.Tensor  # [M, E], row-strided ok
    correction_bias: torch.Tensor  # [E]
    topk_ids: torch.Tensor  # [M, TOPK] i32 out
    topk_weights: torch.Tensor  # [M, TOPK] fp32 out
    renormalize: bool
    routed_scale: float


# (BM, BN, BK, num_warps, num_stages) of the shared down GEMM by largest M;
# MI355X sweep at N=7168, I=768 beside the top-k programs
_GEMM_CFG = {
    16: (16, 32, 256, 4, 2),
    32: (32, 32, 256, 4, 2),
    64: (32, 32, 256, 4, 2),
    128: (64, 32, 128, 4, 2),
}
MAX_TOKENS = max(_GEMM_CFG)


def _gemm_cfg(m: int) -> tuple[int, int, int, int, int]:
    return next(cfg for bm, cfg in _GEMM_CFG.items() if m <= bm)


def situ_cast(
    gate_up: torch.Tensor,
    act: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
    cast: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    """Launch A: act = SiTU(gate_up); cast (src, dst): dst = src in dst's dtype."""
    _launch(gate_up.shape[0], beta, linear_beta, gate_up=gate_up, act=act, cast=cast)


def shared_down_topk(
    act: torch.Tensor, w_down: torch.Tensor, out: torch.Tensor, router: Router
) -> None:
    """Launch B: out = act @ w_down^T, and router's top-k as aiter
    biased_grouped_topk with one group (same experts, order and weights)."""
    assert act.shape == (router.logits.shape[0], w_down.shape[1])
    assert shared_down_topk_fits(
        w_down, out, router.logits, router.correction_bias, router.topk_ids.shape[1]
    )
    _launch(act.shape[0], 1.0, None, act=act, w_down=w_down, out=out, router=router)


def topk_supported(num_experts: int, topk: int) -> bool:
    # expert ids ride the low 16 bits of the sort key; tl.topk takes a power of two
    return 0 < num_experts < 0xFFFF and topk > 0 and topk & (topk - 1) == 0


def shared_down_topk_fits(
    w_down: torch.Tensor,
    out: torch.Tensor,
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
) -> bool:
    """Whether shared_down_topk takes these operands (M = logits rows)."""
    m = logits.shape[0]
    if not 0 < m <= MAX_TOKENS:
        return False
    n, i = w_down.shape
    _, bn, bk, _, _ = _gemm_cfg(m)
    return (
        w_down.dtype == out.dtype
        and w_down.dtype in (torch.bfloat16, torch.float16)
        and w_down.is_contiguous()
        and out.shape == (m, n)
        and out.stride(1) == 1
        and n % bn == 0
        and i % bk == 0
        and logits.dim() == 2
        and logits.stride(1) == 1
        and topk_supported(logits.shape[1], top_k)
        and correction_bias.shape == (logits.shape[1],)
    )


def _launch(
    M: int,
    beta: float,
    linear_beta: Optional[float],
    *,
    gate_up: Optional[torch.Tensor] = None,
    act: torch.Tensor,
    cast: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    w_down: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    router: Optional[Router] = None,
) -> None:
    I = act.shape[1]
    assert act.is_contiguous() and act.shape[0] == M
    lin = linear_beta is not None
    do_situ = gate_up is not None
    situ_block = 256
    if do_situ:
        assert gate_up.stride(1) == 1 and gate_up.shape == (M, 2 * I)
        n_situ = M * triton.cdiv(I, situ_block)
    else:
        gate_up, n_situ = act, 0
    do_gemm = w_down is not None
    if do_gemm:
        bm, bn, bk, nw, ns = _gemm_cfg(M)
        n_gemm = (w_down.shape[0] // bn) * triton.cdiv(M, bm)
    else:
        bm, bn, bk, nw, ns = 16, 16, 16, 4, 2
        w_down, out, n_gemm = act, act, 0
    cblock = 1024
    if cast is not None:
        src, dst = cast
        assert src.stride(1) == 1 and dst.is_contiguous() and src.shape == dst.shape
        n_cols = src.shape[1]
        n_cast = M * triton.cdiv(n_cols, cblock)
    else:
        src = dst = act
        n_cols, n_cast = 1, 0
    if router is not None:
        lg, bias = router.logits, router.correction_bias
        ids, wts = router.topk_ids, router.topk_weights
        assert ids.is_contiguous() and wts.is_contiguous()
        E, topk, n_topk = lg.shape[1], ids.shape[1], M
    else:
        lg = bias = ids = wts = act
        E, topk, n_topk = 1, 1, 0
    _k3_moe_front_epi_kernel[(n_topk + n_gemm + n_situ + n_cast,)](
        gate_up,
        gate_up.stride(0),
        act,
        w_down,
        out,
        out.stride(0),
        src,
        src.stride(0),
        dst,
        lg,
        lg.stride(0),
        bias,
        ids,
        wts,
        float(router.routed_scale) if router is not None else 1.0,
        M,
        float(beta),
        1.0 / float(beta),
        float(linear_beta) if lin else 1.0,
        1.0 / float(linear_beta) if lin else 1.0,
        I=I,
        HAS_LIN=lin,
        DO_TOPK=router is not None,
        E=E,
        E_POW2=triton.next_power_of_2(E),
        TOPK=topk,
        RENORM=router.renormalize if router is not None else False,
        DO_GEMM=do_gemm,
        N=w_down.shape[0],
        BM=bm,
        BN=bn,
        BK=bk,
        DO_SITU=do_situ,
        SITU_BLOCK=situ_block,
        DO_CAST=cast is not None,
        C=n_cols,
        CBLOCK=cblock,
        num_warps=nw,
        num_stages=ns,
    )
