# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 KDA target verify in one FlyDSL launch (snapshot mode).

Host side of ROCm/aiter#7529's ``flydsl_kda_verify_with_f_b``: per draft token
the head-local ``f_b`` projection, the Q/K/V causal conv, the decay gate, the
gated delta rule and the gated RMSNorm. ``conv_state`` and ``state`` are only
read; step ``t`` writes its conv window and its recurrent-state snapshot.
"""

from __future__ import annotations

import functools

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .kernels.kimi_k3_kda_verify import (
    KDA_VERIFY_MAX_TOKENS,
    KDA_VERIFY_NUM_WARPS,
    create_kda_verify_kernel,
)
from .kimi_k3_kda_decode import _check_tensor, is_flydsl_kimi_k3_kda_decode_supported

_HEADS = 12
_DIM = 128
_CONV_CHANNELS = 3 * _HEADS * _DIM
_CONV_WIDTH = 4
_STATE_DTYPE_TAGS = {torch.float32: "f32", torch.bfloat16: "bf16"}
_INT32_MAX = 2**31 - 1

STATE_DTYPES = tuple(_STATE_DTYPE_TAGS)
MAX_TOKENS = KDA_VERIFY_MAX_TOKENS
NUM_WARPS = KDA_VERIFY_NUM_WARPS


def is_supported(device: torch.device | str | int | None = None) -> bool:
    return is_flydsl_kimi_k3_kda_decode_supported(device)


@functools.cache
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def num_warps_for(batch: int, device: torch.device) -> int:
    """aiter#7529's snapshot policy: 8 waves while batch x 12 workgroups <= 5/4 of the CUs."""
    return 8 if 4 * batch * _HEADS <= 5 * _num_cus(device) else 4


def _i32_span_ok(tensor: torch.Tensor, first_dim: int) -> bool:
    # Byte span from the tensor base; cache slots are rebased in int64,
    # so their slot dimension (and raw_beta/out's leading 1) is excluded.
    shape, strides = tensor.shape[first_dim:], tensor.stride()[first_dim:]
    if any(stride < 0 or stride > _INT32_MAX for stride in strides):
        return False
    span = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    return span * tensor.element_size() <= _INT32_MAX


def _dense_steps(tensor: torch.Tensor, inner: tuple[tuple[int, ...], ...]) -> bool:
    # Dense, non-overlapping slots; padding between slots is allowed.
    return (
        tuple(tensor.stride()[1:]) in inner
        and tensor.stride(0) >= tensor.shape[1:].numel()
    )


def flydsl_kimi_k3_kda_verify_with_f_b(
    *,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_states: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    num_tokens: int,
    num_warps: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify ``num_tokens`` draft tokens per sequence; returns [1, tokens, 12, 128] bf16.

    Sequence ``b`` owns rows ``b * num_tokens ..`` of ``f_a`` [tokens, 128],
    ``x`` [tokens, 4608], ``raw_beta`` [1, tokens, 12] and ``output_gate``
    [tokens, 12, 128]. ``cache_indices[b]`` selects the ``conv_state``
    [slots, 4608, 3] / ``state`` [slots, 12, 128, 128] slot;
    ``intermediate_indices[b]`` the line of ``intermediate_conv_window``
    [lines, steps, 4608, 3] and ``intermediate_states`` [lines, steps, 12, 128, 128].
    A negative or out-of-range index makes the sequence padding (zero output,
    no write). Live ``intermediate_indices`` must be unique.
    """
    if not 1 <= num_tokens <= KDA_VERIFY_MAX_TOKENS:
        raise ValueError(f"`num_tokens` must be in [1, {KDA_VERIFY_MAX_TOKENS}].")
    device = cache_indices.device
    batch = cache_indices.shape[0]
    tokens = batch * num_tokens
    if batch <= 0:
        raise ValueError("`cache_indices` must be non-empty.")
    if state.dtype not in _STATE_DTYPE_TAGS:
        raise ValueError(f"`state` dtype {state.dtype} is not supported.")
    for name, tensor, shape, dtype, inner in (
        ("f_a", f_a, (tokens, _DIM), torch.bfloat16, (1,)),
        ("f_b_weight", f_b_weight, (_HEADS, _DIM, _DIM), torch.bfloat16, (_DIM, 1)),
        ("x", x, (tokens, _CONV_CHANNELS), torch.bfloat16, (1,)),
        ("conv_weight", conv_weight, (_CONV_CHANNELS, _CONV_WIDTH), torch.float32, ()),
        ("raw_beta", raw_beta, (1, tokens, _HEADS), torch.bfloat16, (1,)),
        ("A_log", A_log, (_HEADS,), torch.float32, (1,)),
        ("dt_bias", dt_bias, (_HEADS * _DIM,), torch.float32, (1,)),
        ("cache_indices", cache_indices, (batch,), torch.int32, (1,)),
        ("intermediate_indices", intermediate_indices, (batch,), torch.int32, (1,)),
        ("output_gate", output_gate, (tokens, _HEADS, _DIM), torch.bfloat16, (1,)),
        ("norm_weight", norm_weight, (_DIM,), torch.bfloat16, (1,)),
    ):
        _check_tensor(name, tensor, shape=shape, dtype=dtype, device=device, inner_strides=inner)
    if 1 not in conv_weight.stride():
        raise ValueError("`conv_weight` must have an axis with stride 1.")
    if not (
        conv_state.ndim == 3
        and tuple(conv_state.shape[1:]) == (_CONV_CHANNELS, _CONV_WIDTH - 1)
        and conv_state.dtype == torch.bfloat16
        and conv_state.device == device
        and _dense_steps(conv_state, ((_CONV_WIDTH - 1, 1), (1, _CONV_CHANNELS)))
    ):
        raise ValueError("`conv_state` must be bf16 [slots, 4608, 3] with dense slots.")
    if not (
        state.ndim == 4
        and tuple(state.shape[1:]) == (_HEADS, _DIM, _DIM)
        and state.device == device
        and _dense_steps(state, ((_DIM * _DIM, _DIM, 1),))
    ):
        raise ValueError("`state` must be [slots, 12, 128, 128] with dense slots.")
    window_step = _CONV_CHANNELS * (_CONV_WIDTH - 1)
    if not (
        intermediate_conv_window.ndim == 4
        and intermediate_conv_window.shape[1] >= num_tokens
        and tuple(intermediate_conv_window.shape[2:])
        == (_CONV_CHANNELS, _CONV_WIDTH - 1)
        and intermediate_conv_window.dtype == torch.bfloat16
        and intermediate_conv_window.device == device
        and _dense_steps(
            intermediate_conv_window,
            ((window_step, _CONV_WIDTH - 1, 1), (window_step, 1, _CONV_CHANNELS)),
        )
    ):
        raise ValueError(
            "`intermediate_conv_window` must be bf16 [lines, >= tokens, 4608, 3], dense."
        )
    if not (
        intermediate_states.ndim == 5
        and intermediate_states.shape[1] >= num_tokens
        and tuple(intermediate_states.shape[2:]) == (_HEADS, _DIM, _DIM)
        and intermediate_states.dtype == state.dtype
        and intermediate_states.device == device
        and _dense_steps(
            intermediate_states, ((_HEADS * _DIM * _DIM, _DIM * _DIM, _DIM, 1),)
        )
    ):
        raise ValueError(
            "`intermediate_states` must be [lines, >= tokens, 12, 128, 128] "
            "in the state dtype, dense."
        )
    if out is None:
        out = torch.empty(
            (1, tokens, _HEADS, _DIM), dtype=torch.bfloat16, device=device
        )
    _check_tensor(
        "out",
        out,
        shape=(1, tokens, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    for name, tensor, first_dim in (
        ("f_a", f_a, 0),
        ("f_b_weight", f_b_weight, 0),
        ("x", x, 0),
        ("conv_weight", conv_weight, 0),
        ("raw_beta", raw_beta, 1),
        ("output_gate", output_gate, 0),
        ("out", out, 1),
        ("conv_state", conv_state, 1),
        ("state", state, 1),
        ("intermediate_conv_window", intermediate_conv_window, 1),
        ("intermediate_states", intermediate_states, 1),
    ):
        if not _i32_span_ok(tensor, first_dim):
            raise ValueError(f"`{name}` exceeds int32 addressing.")
    if num_warps is None:
        num_warps = num_warps_for(batch, device)
    if num_warps not in KDA_VERIFY_NUM_WARPS:
        raise ValueError(f"`num_warps` must be one of {KDA_VERIFY_NUM_WARPS}.")

    executable = create_kda_verify_kernel(
        float(norm_eps),
        float(lower_bound),
        num_tokens,
        _STATE_DTYPE_TAGS[state.dtype],
        False,
        num_warps,
    )
    num_slots = min(state.shape[0], conv_state.shape[0], _INT32_MAX)
    num_lines = min(
        intermediate_conv_window.shape[0], intermediate_states.shape[0], _INT32_MAX
    )
    # Ring operands are unused in snapshot mode; any tensor fills the slots.
    dummy = state
    window = intermediate_conv_window
    with torch.cuda.device(device):
        _run_compiled(
            executable,
            f_a,
            f_b_weight,
            x,
            conv_weight,
            conv_state,
            raw_beta,
            A_log,
            dt_bias,
            state,
            cache_indices,
            intermediate_indices,
            window,
            intermediate_states,
            dummy,
            dummy,
            dummy,
            dummy,
            output_gate,
            norm_weight,
            out,
            batch,
            num_slots,
            num_lines,
            f_a.stride(0),
            f_b_weight.stride(0),
            f_b_weight.stride(1),
            x.stride(0),
            conv_weight.stride(0),
            conv_weight.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            raw_beta.stride(1),
            state.stride(0),
            window.stride(0),
            window.stride(1),
            window.stride(2),
            window.stride(3),
            intermediate_states.stride(0),
            intermediate_states.stride(1),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            output_gate.stride(0),
            output_gate.stride(1),
            out.stride(1),
            out.stride(2),
            torch.cuda.current_stream(device),
        )
    return out


__all__ = [
    "MAX_TOKENS",
    "NUM_WARPS",
    "STATE_DTYPES",
    "flydsl_kimi_k3_kda_verify_with_f_b",
    "is_supported",
    "num_warps_for",
]
