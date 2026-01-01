"""Kimi-K3 KDA chain verify in one FlyDSL kernel on gfx950 (ROCm/aiter#7529).

The kernel fuses the f_b projection, the Q/K/V causal conv, the decay gate,
the gated delta rule with per-step snapshots and the gated o_norm. The model
defers f_b and offers the o_norm gate; any batch this adapter declines returns
None, and the caller materializes f_b and runs the HIP chain verify.
"""

from __future__ import annotations

import functools
import logging
from types import ModuleType
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import rank0_log

logger = logging.getLogger(__name__)

# Measured on MI355X (fp32 state, 5 draft tokens, per layer, graph replay):
# batch 1 fused 15.0 us vs f_b GEMM + HIP verify + o_norm 11.1 us; batch 4 16.6 vs 17.4;
# batch 14 20.5 vs 32.4; batch 20 23.2 vs 40.2. Smaller batches keep the HIP chain.
_MIN_BATCH = 4

_WARMED: set[tuple] = set()
_REJECTION_LOGGED = False


@functools.cache
def _op() -> Optional[ModuleType]:
    try:
        from sglang.kernels.ops.attention.kda_flydsl import kimi_k3_kda_verify
    except ImportError as error:
        logger.warning("Kimi-K3 FlyDSL KDA verify unavailable: %s", error)
        return None
    return kimi_k3_kda_verify


def available(device: torch.device) -> bool:
    if not is_hip() or envs.SGLANG_DISABLE_HIP_K3_KDA_FLYDSL_VERIFY.get():
        return False
    op = _op()
    return op is not None and op.is_supported(device)


def _warm_key(*, device, norm_eps, lower_bound, state_dtype, num_draft_tokens):
    return (device.index, float(norm_eps), float(lower_bound), state_dtype, num_draft_tokens)


def warmup(
    *,
    static_args: tuple,
    conv_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    state_dtype: torch.dtype,
    num_draft_tokens: int,
) -> bool:
    """JIT-build every wave count for this specialization; must run outside graph capture."""
    op = _op()
    f_b_weight, norm_weight, norm_eps, a_log = static_args
    device = f_b_weight.device
    key = _warm_key(
        device=device,
        norm_eps=norm_eps,
        lower_bound=lower_bound,
        state_dtype=state_dtype,
        num_draft_tokens=num_draft_tokens,
    )
    if key in _WARMED:
        return True
    if (
        op is None
        or state_dtype not in op.STATE_DTYPES
        or not 1 <= num_draft_tokens <= op.MAX_TOKENS
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    zeros = functools.partial(torch.zeros, device=device)
    channels = conv_weight.shape[0]
    heads, dim = f_b_weight.shape[0], f_b_weight.shape[1]
    tokens = num_draft_tokens
    index = torch.zeros(1, dtype=torch.int32, device=device)
    for num_warps in op.NUM_WARPS:
        op.flydsl_kimi_k3_kda_verify_with_f_b(
            f_a=zeros(tokens, dim, dtype=torch.bfloat16),
            f_b_weight=f_b_weight,
            x=zeros(tokens, channels, dtype=torch.bfloat16),
            conv_weight=conv_weight,
            conv_state=zeros(1, channels, 3, dtype=torch.bfloat16),
            raw_beta=zeros(1, tokens, heads, dtype=torch.bfloat16),
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            state=zeros(1, heads, dim, dim, dtype=state_dtype),
            cache_indices=index,
            intermediate_indices=index,
            intermediate_conv_window=zeros(1, tokens, channels, 3, dtype=torch.bfloat16),
            intermediate_states=zeros(1, tokens, heads, dim, dim, dtype=state_dtype),
            output_gate=zeros(tokens, heads, dim, dtype=torch.bfloat16),
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            num_tokens=tokens,
            num_warps=num_warps,
        )
    torch.cuda.synchronize(device)
    _WARMED.add(key)
    rank0_log(
        f"Kimi-K3 FlyDSL KDA verify armed: {num_draft_tokens} draft tokens, "
        f"{state_dtype} state snapshots"
    )
    return True


def _warn_rejected_once(reason: str) -> None:
    # Once per process; the first reason is enough to debug.
    global _REJECTION_LOGGED
    if not _REJECTION_LOGGED:
        _REJECTION_LOGGED = True
        rank0_log(
            f"Kimi-K3 FlyDSL KDA verify declined a batch ({reason}); "
            "it runs the HIP chain verify (logged once)"
        )


def run_layer(
    *,
    layer: torch.nn.Module,
    mixed_qkv: torch.Tensor,
    f_a: torch.Tensor,
    raw_beta: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    intermediate_state_cache: torch.Tensor,
    intermediate_conv_window_cache: torch.Tensor,
    num_draft_tokens: int,
) -> Optional[torch.Tensor]:
    """Dense chain verify of a layer whose f_b and o_norm were deferred, or None."""
    static_args = layer._k3_hip_fused_decode_args
    onorm_gate = layer._k3_onorm_gate
    tokens = mixed_qkv.shape[0]
    batch = tokens // num_draft_tokens
    if (
        static_args is None
        or onorm_gate is None
        or batch < _MIN_BATCH
        or tokens != batch * num_draft_tokens
        or cache_indices.shape[0] < batch
        or intermediate_state_indices.shape[0] < batch
        or not warmup(
            static_args=static_args,
            conv_weight=layer.conv_weights,
            dt_bias=layer.dt_bias,
            lower_bound=float(layer.lower_bound),
            state_dtype=ssm_states.dtype,
            num_draft_tokens=num_draft_tokens,
        )
    ):
        return None
    f_b_weight, norm_weight, norm_eps, a_log = static_args
    try:
        return _op().flydsl_kimi_k3_kda_verify_with_f_b(
            f_a=f_a,
            f_b_weight=f_b_weight,
            x=mixed_qkv,
            conv_weight=layer.conv_weights,
            conv_state=conv_states.transpose(-1, -2),
            raw_beta=raw_beta,
            A_log=a_log,
            dt_bias=layer.dt_bias,
            lower_bound=float(layer.lower_bound),
            state=ssm_states,
            cache_indices=cache_indices[:batch],
            intermediate_indices=intermediate_state_indices[:batch],
            intermediate_conv_window=intermediate_conv_window_cache.transpose(-1, -2),
            intermediate_states=intermediate_state_cache,
            output_gate=onorm_gate[:tokens].view(
                tokens, layer.num_v_heads, layer.head_v_dim
            ),
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            num_tokens=num_draft_tokens,
        )
    except ValueError as error:
        # The wrapper checks the layout contract before it launches anything.
        _warn_rejected_once(str(error))
        return None


def materialize_f_b(*, layer: torch.nn.Module, f_a: torch.Tensor) -> torch.Tensor:
    """The deferred f_b projection of f_a [tokens, K], as [1, tokens, H, K]."""
    # Same GEMM the model runs when it does not defer f_b.
    from sglang.srt.models.kimi_k3_amd.gemm import kda_f_b_gemm_hip

    f_b_weight = layer._k3_hip_fused_decode_args[0]
    gate = kda_f_b_gemm_hip(f_a, f_b_weight.flatten(0, 1))
    return gate.unflatten(-1, f_b_weight.shape[:2]).unsqueeze(0)
