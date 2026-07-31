"""Theoretical DeepSeek-V4 prefill cost model for DP request dispatch.

This module intentionally models only the request-dependent parts that differ
substantially across cached prefixes: CSA (indexer and sparse attention), HCA,
SWA, and HiCache data movement.  Dense/MoE work is left out of the first
version because it is approximately linear in uncached tokens and does not
depend on context length.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from sglang.srt.configs.model_config import is_deepseek_v4


class PrefillCostEstimate(NamedTuple):
    csa_indexer_seconds: float = 0.0
    csa_attention_seconds: float = 0.0
    hca_attention_seconds: float = 0.0
    swa_attention_seconds: float = 0.0
    h2d_seconds: float = 0.0
    storage_prefetch_seconds: float = 0.0

    @property
    def attention_seconds(self) -> float:
        return (
            self.csa_indexer_seconds
            + self.csa_attention_seconds
            + self.hca_attention_seconds
            + self.swa_attention_seconds
        )

    @property
    def prefetch_seconds(self) -> float:
        return self.h2d_seconds + self.storage_prefetch_seconds

    @property
    def total_seconds(self) -> float:
        return self.attention_seconds + self.prefetch_seconds

    def __add__(self, other: PrefillCostEstimate) -> PrefillCostEstimate:
        return PrefillCostEstimate(*(a + b for a, b in zip(self, other)))


class DeepSeekV4PrefillCostModel:
    """Estimate DSV4 attention and cache-transfer service time.

    ``attention_tflops_per_gpu`` is an effective attention throughput rather
    than a GEMM peak.  The two bandwidths are kept separate because an L3 hit
    first travels from storage to the host cache and then from host to device.
    """

    SPEC_VERSION = 2

    def __init__(
        self,
        *,
        num_attention_heads: int,
        num_hidden_layers: int,
        num_csa_layers: int,
        num_hca_layers: int,
        num_swa_only_layers: int,
        qk_head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        kv_cache_bytes_per_slot: int,
        attn_tp_size: int,
        attention_tflops_per_gpu: float,
        h2d_bandwidth_gbps: float,
        storage_bandwidth_gbps: float,
        fp4_indexer: bool = False,
    ):
        positive_ints = {
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "qk_head_dim": qk_head_dim,
            "v_head_dim": v_head_dim,
            "index_n_heads": index_n_heads,
            "index_head_dim": index_head_dim,
            "index_topk": index_topk,
            "window_size": window_size,
            "kv_cache_bytes_per_slot": kv_cache_bytes_per_slot,
            "attn_tp_size": attn_tp_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if min(num_csa_layers, num_hca_layers, num_swa_only_layers) < 0:
            raise ValueError("DSV4 layer counts cannot be negative")
        if num_csa_layers + num_hca_layers + num_swa_only_layers != num_hidden_layers:
            raise ValueError(
                "DSV4 compression ratios must cover every attention layer: "
                f"{num_csa_layers=} + {num_hca_layers=} + "
                f"{num_swa_only_layers=} != {num_hidden_layers=}"
            )
        for name, value in (
            ("attention_tflops_per_gpu", attention_tflops_per_gpu),
            ("h2d_bandwidth_gbps", h2d_bandwidth_gbps),
            ("storage_bandwidth_gbps", storage_bandwidth_gbps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_csa_layers = num_csa_layers
        self.num_hca_layers = num_hca_layers
        self.num_swa_only_layers = num_swa_only_layers
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.window_size = window_size
        self.kv_cache_bytes_per_slot = kv_cache_bytes_per_slot
        self.attn_tp_size = attn_tp_size
        self.attention_tflops_per_gpu = attention_tflops_per_gpu
        self.h2d_bandwidth_gbps = h2d_bandwidth_gbps
        self.storage_bandwidth_gbps = storage_bandwidth_gbps
        self.fp4_indexer = fp4_indexer

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        attn_tp_size: int,
        attention_tflops_per_gpu: float,
        h2d_bandwidth_gbps: float,
        storage_bandwidth_gbps: float,
        fp4_indexer: bool = False,
        unified_kv: bool = False,
    ) -> DeepSeekV4PrefillCostModel:
        hf_config = model_config.hf_text_config
        if not is_deepseek_v4(hf_config):
            architectures = getattr(hf_config, "architectures", None)
            raise ValueError(
                "load_balance_method=cost_aware currently supports only "
                f"DeepSeek-V4, got architectures={architectures}"
            )

        num_hidden_layers = int(model_config.num_hidden_layers)
        all_compression_ratios = tuple(int(r) for r in model_config.compress_ratios)
        if len(all_compression_ratios) < num_hidden_layers:
            raise ValueError(
                "DSV4 compression ratios do not cover every attention layer: "
                f"{len(all_compression_ratios)} < {num_hidden_layers}"
            )
        compression_ratios = all_compression_ratios[:num_hidden_layers]
        unsupported = sorted(set(compression_ratios) - {0, 4, 128})
        if unsupported:
            raise ValueError(
                "DSV4 cost model supports compression ratios 0, 4, and 128, "
                f"got unsupported ratios {unsupported}"
            )

        qk_head_dim = int(model_config.head_dim)
        qk_rope_head_dim = int(model_config.qk_rope_head_dim)
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim
        if qk_nope_head_dim <= 0:
            raise ValueError(
                "DSV4 qk_rope_head_dim must be smaller than head_dim, "
                f"got {qk_rope_head_dim=} and {qk_head_dim=}"
            )
        kv_cache_bytes_per_slot = (
            2 * qk_head_dim
            if unified_kv
            else (qk_nope_head_dim + 2 * qk_rope_head_dim + qk_nope_head_dim // 64 + 1)
        )

        return cls(
            num_attention_heads=int(model_config.num_attention_heads),
            num_hidden_layers=num_hidden_layers,
            num_csa_layers=compression_ratios.count(4),
            num_hca_layers=compression_ratios.count(128),
            num_swa_only_layers=compression_ratios.count(0),
            qk_head_dim=qk_head_dim,
            v_head_dim=int(model_config.v_head_dim),
            index_n_heads=int(hf_config.index_n_heads),
            index_head_dim=int(hf_config.index_head_dim),
            index_topk=int(hf_config.index_topk),
            window_size=int(model_config.window_size),
            kv_cache_bytes_per_slot=kv_cache_bytes_per_slot,
            attn_tp_size=attn_tp_size,
            attention_tflops_per_gpu=attention_tflops_per_gpu,
            h2d_bandwidth_gbps=h2d_bandwidth_gbps,
            storage_bandwidth_gbps=storage_bandwidth_gbps,
            fp4_indexer=fp4_indexer,
        )

    def to_spec(self) -> dict[str, int | float | bool]:
        """Return a pickle/msgpack-friendly spec for the scheduler handshake."""
        return {
            "version": self.SPEC_VERSION,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_csa_layers": self.num_csa_layers,
            "num_hca_layers": self.num_hca_layers,
            "num_swa_only_layers": self.num_swa_only_layers,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
            "window_size": self.window_size,
            "kv_cache_bytes_per_slot": self.kv_cache_bytes_per_slot,
            "attn_tp_size": self.attn_tp_size,
            "attention_tflops_per_gpu": self.attention_tflops_per_gpu,
            "h2d_bandwidth_gbps": self.h2d_bandwidth_gbps,
            "storage_bandwidth_gbps": self.storage_bandwidth_gbps,
            "fp4_indexer": self.fp4_indexer,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> DeepSeekV4PrefillCostModel:
        spec = dict(spec)
        version = spec.pop("version", None)
        if version != cls.SPEC_VERSION:
            raise ValueError(
                f"Unsupported DSV4 cost-model spec version {version}; "
                f"expected {cls.SPEC_VERSION}"
            )
        return cls(**spec)

    @staticmethod
    def _sum_capped_contexts(
        prefix_tokens: int,
        query_tokens: int,
        *,
        compression_ratio: int,
        cap: int | None = None,
    ) -> float:
        """Sum ``min((prefix + i) / ratio, cap)`` for causal queries i=1..q."""
        if query_tokens <= 0:
            return 0.0
        if cap is None:
            return (
                query_tokens * prefix_tokens + query_tokens * (query_tokens + 1) / 2
            ) / compression_ratio

        uncapped_queries = min(
            query_tokens,
            max(0, cap * compression_ratio - prefix_tokens),
        )
        uncapped_sum = (
            uncapped_queries * prefix_tokens
            + uncapped_queries * (uncapped_queries + 1) / 2
        ) / compression_ratio
        return uncapped_sum + (query_tokens - uncapped_queries) * cap

    @property
    def full_cache_bytes_per_input_token(self) -> float:
        """Logical HiCache bytes represented by one original input token.

        DSV4 stores one packed KV item per c4/c128 slot. The item is
        584 bytes in the separate FP8/BF16 layout and 1024 bytes in the
        unified BF16 layout. CSA also stores one quantized index vector plus
        a four-byte scale.
        """
        index_bytes = self.index_head_dim / (2 if self.fp4_indexer else 1) + 4
        return (
            self.num_csa_layers * (self.kv_cache_bytes_per_slot + index_bytes) / 4
            + self.num_hca_layers * self.kv_cache_bytes_per_slot / 128
        )

    @property
    def swa_cache_bytes_per_input_token(self) -> int:
        return self.num_hidden_layers * self.kv_cache_bytes_per_slot

    def estimate(
        self,
        *,
        input_tokens: int,
        cached_context_tokens: int = 0,
        host_cache_tokens: int = 0,
        storage_cache_tokens: int = 0,
        swa_host_cache_tokens: int = 0,
    ) -> PrefillCostEstimate:
        if (
            min(
                input_tokens,
                cached_context_tokens,
                host_cache_tokens,
                storage_cache_tokens,
                swa_host_cache_tokens,
            )
            < 0
        ):
            raise ValueError("Token counts in the DSV4 cost model cannot be negative")

        cached_context_tokens = min(cached_context_tokens, input_tokens)
        host_cache_tokens = min(host_cache_tokens, cached_context_tokens)
        storage_cache_tokens = min(storage_cache_tokens, host_cache_tokens)
        swa_host_cache_tokens = min(
            swa_host_cache_tokens, cached_context_tokens, self.window_size
        )
        query_tokens = input_tokens - cached_context_tokens

        csa_index_pairs = self._sum_capped_contexts(
            cached_context_tokens,
            query_tokens,
            compression_ratio=4,
        )
        csa_attention_pairs = self._sum_capped_contexts(
            cached_context_tokens,
            query_tokens,
            compression_ratio=4,
            cap=self.index_topk,
        )
        hca_attention_pairs = self._sum_capped_contexts(
            cached_context_tokens,
            query_tokens,
            compression_ratio=128,
        )
        swa_attention_pairs = self._sum_capped_contexts(
            cached_context_tokens,
            query_tokens,
            compression_ratio=1,
            cap=self.window_size,
        )

        attention_pair_flops = (
            2 * self.num_attention_heads * (self.qk_head_dim + self.v_head_dim)
        )
        index_pair_flops = 2 * self.index_n_heads * self.index_head_dim
        group_flops_per_second = (
            self.attention_tflops_per_gpu * 1e12 * self.attn_tp_size
        )

        csa_indexer_seconds = (
            self.num_csa_layers * csa_index_pairs * index_pair_flops
        ) / group_flops_per_second
        csa_attention_seconds = (
            self.num_csa_layers * csa_attention_pairs * attention_pair_flops
        ) / group_flops_per_second
        hca_attention_seconds = (
            self.num_hca_layers * hca_attention_pairs * attention_pair_flops
        ) / group_flops_per_second
        swa_attention_seconds = (
            self.num_hidden_layers * swa_attention_pairs * attention_pair_flops
        ) / group_flops_per_second

        h2d_bytes = (
            host_cache_tokens * self.full_cache_bytes_per_input_token
            + swa_host_cache_tokens * self.swa_cache_bytes_per_input_token
        )
        storage_bytes = storage_cache_tokens * self.full_cache_bytes_per_input_token

        return PrefillCostEstimate(
            csa_indexer_seconds=csa_indexer_seconds,
            csa_attention_seconds=csa_attention_seconds,
            hca_attention_seconds=hca_attention_seconds,
            swa_attention_seconds=swa_attention_seconds,
            h2d_seconds=h2d_bytes / (self.h2d_bandwidth_gbps * 1e9),
            storage_prefetch_seconds=storage_bytes
            / (self.storage_bandwidth_gbps * 1e9),
        )
