"""Static per-tensor FP8 (amax / 448 weight scale, unit activation scale) for
Kimi-K3 BF16 dense linears on ROCm (port of sgl-project/sglang#46653).

``SGLANG_K3_TARGET_DENSE_FP8``: ``off`` | ``front`` (the row-sharded MoE front's
[router | latent] rows) | ``wide`` (front + KDA q/k/v/g).
"""

from __future__ import annotations

from functools import cache
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import rank0_log

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max
# aiter's (16, 16) preshuffle permutes within tiles of 16 rows x 32 columns.
TILE_ROWS = 16
_TILE_COLS = 32
# Rows a GEMM launch takes; larger batches run in row chunks.
_MAX_TOKENS = 1 << 15


def scope() -> str:
    value = envs.SGLANG_K3_TARGET_DENSE_FP8.get().lower()
    if value not in ("off", "front", "wide"):
        raise ValueError(
            f"SGLANG_K3_TARGET_DENSE_FP8={value!r}; expected off|front|wide"
        )
    # The AMD monokernel launches read these weights in bf16.
    if (
        not (is_hip() and envs.SGLANG_USE_AITER.get())
        or get_exec().kernel.enable_amd_monokernel
    ):
        return "off"
    return value


def front_enabled() -> bool:
    return scope() in ("front", "wide")


def qkvg_enabled() -> bool:
    return scope() == "wide"


@cache
def log_once(what: str) -> None:
    rank0_log(f"Kimi-K3 dense FP8 (SGLANG_K3_TARGET_DENSE_FP8={scope()}): {what}")


def convertible(weight: torch.Tensor) -> bool:
    """A plain [N, K] bf16 weight on the preshuffle's tile grid."""
    return (
        type(weight) is torch.Tensor
        and weight.dim() == 2
        and weight.dtype == torch.bfloat16
        and weight.shape[0] % TILE_ROWS == 0
        and weight.shape[1] % _TILE_COLS == 0
    )


@torch.no_grad()
def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16 -> ([N, K] E4M3, scalar fp32 scale) with scale = amax / 448."""
    w = weight.float()
    amax = w.abs().amax().clamp(min=1e-12)
    scale = (amax / _FP8_MAX).reshape(())
    q = (w / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).contiguous()
    return q, scale.to(torch.float32)


@cache
def _unit_scales(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The unit activation scale: the quant kernel's scalar and the GEMM's [M, 1]."""
    unit = torch.ones(1, dtype=torch.float32, device=device)
    x_scale = torch.ones(_MAX_TOKENS, 1, dtype=torch.float32, device=device)
    return unit, x_scale


class StaticFP8Weight:
    """E4M3 [N, K] weight for aiter.gemm_a8w8_bpreshuffle; ``__call__`` is x @ W^T."""

    def __init__(
        self,
        weight: torch.Tensor,
        release_source: Optional[Callable[[], None]] = None,
    ):
        """Quantize the bf16 ``weight``; release_source, if given, drops the
        caller's bf16 tensor once its E4M3 copy is on the host."""
        from aiter.ops.shuffle import shuffle_weight

        device = weight.device
        q, scale = quantize_weight(weight)
        self.n, self.k = q.shape
        host = shuffle_weight(q, layout=(16, 16)).cpu()
        del q, weight
        if release_source is not None:
            release_source()
        # The fp32 temporaries (and a released bf16 source) are larger than the E4M3
        # weight: placed in one of their cached blocks it would pin the whole block,
        # so return them to the device first. Device KV pools are sized after this.
        torch.cuda.empty_cache()
        self.weight = host.to(device)
        # [1, N] fp32, the one scale on every output column
        self.w_scale = scale.reshape(1, 1).expand(1, self.n).contiguous()
        # allocate the shared unit scales post-load, not inside a CUDA graph capture
        _unit_scales(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x @ W^T in x's dtype (bf16)."""
        if x.dtype != torch.bfloat16:
            raise ValueError(f"K3 ROCm FP8 linear: unsupported input dtype {x.dtype}")
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        m = x2.shape[0]
        if m == 0:
            y = x2.new_empty((0, self.n))
        elif m <= _MAX_TOKENS:
            y = self._gemm(x2)
        else:
            y = torch.cat(
                [
                    self._gemm(x2[i : i + _MAX_TOKENS])
                    for i in range(0, m, _MAX_TOKENS)
                ]
            )
        return y.reshape(*x.shape[:-1], self.n)

    def _gemm(self, x: torch.Tensor) -> torch.Tensor:
        import aiter

        unit, x_scale = _unit_scales(x.device)
        xq = torch.empty(x.shape, dtype=_FP8, device=x.device)
        # unit scale: a saturating cast to E4M3
        aiter.static_per_tensor_quant(xq, x, unit)
        return aiter.gemm_a8w8_bpreshuffle(
            xq, self.weight, x_scale[: x.shape[0]], self.w_scale, dtype=x.dtype
        )


class StaticFP8LinearMethod:
    """Drop-in ``quant_method`` for an already-loaded unquantized linear layer."""

    def __init__(self, fp8: StaticFP8Weight):
        self.fp8 = fp8

    def apply(
        self, layer, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.fp8(x)
        return y if bias is None else y + bias

    def process_weights_after_loading(self, layer) -> None:  # already converted
        return None


@torch.no_grad()
def convert_linear(layer: torch.nn.Module) -> StaticFP8Weight:
    """Convert ``layer.weight`` in place and route the layer through FP8."""
    weight = layer.weight

    def drop_bf16() -> None:  # nothing else reads it
        weight.data = torch.empty(0, dtype=_FP8, device=weight.device)

    fp8 = StaticFP8Weight(weight.data, release_source=drop_bf16)
    weight.data = fp8.weight
    layer.quant_method = StaticFP8LinearMethod(fp8)
    layer._derived_weight_cache_error = (
        "Online weight updates are not supported while SGLANG_K3_TARGET_DENSE_FP8 "
        "keeps this layer's weight in static FP8: in-place bf16 writes would not "
        "requantize it. Restart with SGLANG_K3_TARGET_DENSE_FP8=off to allow them."
    )
    return fp8
