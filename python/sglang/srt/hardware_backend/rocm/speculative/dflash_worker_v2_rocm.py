"""DFlash V2 worker with the ROCm fast paths.

Same algorithm as DFlashWorkerV2; replays the post-verify draft-KV append from
a captured graph (eager on HIP otherwise: the base aux graph is CUDA-only) and
runs the verify argmax and the Mamba commit bookkeeping as single kernels.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import get_exec, get_parallel, mamba_track_grid
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)

# Eager rounds a tail signature runs before it is captured.
_TAIL_GRAPH_WARMUP = 3
# Arbitrary bounds; normally one signature per decode batch size.
_TAIL_GRAPH_MAX_GRAPHS = 64
_TAIL_GRAPH_MAX_PENDING = 256


class _TailGraph:
    """A captured draft-KV append and the static inputs it reads."""

    def __init__(
        self,
        *,
        graph: torch.cuda.CUDAGraph,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        self.graph = graph
        self.cache_loc = cache_loc
        self.positions = positions
        self.commit_lens = commit_lens

    def replay(
        self,
        *,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        self.cache_loc.copy_(cache_loc)
        self.positions.copy_(positions)
        self.commit_lens.copy_(commit_lens)
        self.graph.replay()


class DFlashWorkerV2ROCm(DFlashWorkerV2):
    def _init_tail_graph_state(self) -> None:
        # A collective inside the lazily captured graph would need the
        # communicator's capture registration, so a TP-split fc stays eager.
        graph_cfg = get_exec().graph.cuda_graph_config
        # Only an explicit --disable-cuda-graph turns it off; an unresolved
        # config (None) has no decode graph settings to honor.
        graphs_disabled = (
            graph_cfg is not None and graph_cfg.decode.backend == Backend.DISABLED
        )
        self._tail_graph_on = bool(
            self.lilicorr is None
            and not self._use_aux_graph
            and self._megakernel_aux is None
            and self.draft_model.fc_tp_columns is None
            and not get_parallel().attn_dp_enabled
            and not graphs_disabled
        )
        self._tail_graphs: Dict[tuple, _TailGraph] = {}
        self._tail_graph_pending: Dict[tuple, int] = {}
        self._tail_graph_failed: Set[tuple] = set()
        self._tail_graph_pool = None

    def _replay_tail_graph(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> bool:
        """Replay the post-verify draft-KV append (fc + hidden norm + per-layer
        kv-proj / k-norm / RoPE / prefix-valid KV write) as one graph.

        The graph reads ``target_hidden`` in place, so it is keyed by its
        address and layout (the decode graphs' shared aux output is stable per
        batch size); the small per-step inputs are staged into the graph's own
        buffers, so fresh ring locs or rotating commit buffers reuse it. A
        signature is captured after a few eager rounds; until then, and for any
        signature that cannot be captured, the caller runs the eager append."""
        if not self._tail_graph_on or bs <= 0:
            return False
        if (
            commit_lens is None
            or cache_loc_2d is None
            or target_hidden.dim() != 2
            or not target_hidden.is_cuda
            or cache_loc.device != target_hidden.device
            or positions.device != target_hidden.device
            or commit_lens.device != target_hidden.device
            or int(commit_lens.shape[0]) != bs
            or tuple(cache_loc_2d.shape) != (bs, cache_loc.numel() // bs)
            or cache_loc.numel() != positions.numel()
            or int(target_hidden.shape[0]) != cache_loc.numel()
        ):
            return False
        sig = (
            target_hidden.data_ptr(),
            tuple(target_hidden.shape),
            tuple(target_hidden.stride()),
            target_hidden.dtype,
            tuple(cache_loc_2d.shape),
            cache_loc.dtype,
            positions.dtype,
            commit_lens.dtype,
        )
        entry = self._tail_graphs.get(sig)
        if entry is None:
            entry = self._maybe_capture_tail(
                sig=sig,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                commit_lens=commit_lens,
                bs=bs,
            )
            if entry is None:
                return False
        entry.replay(
            cache_loc=cache_loc.reshape(-1),
            positions=positions.reshape(-1),
            commit_lens=commit_lens,
        )
        return True

    def _maybe_capture_tail(
        self,
        *,
        sig: tuple,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> Optional[_TailGraph]:
        if sig in self._tail_graph_failed:
            return None
        if len(self._tail_graphs) >= _TAIL_GRAPH_MAX_GRAPHS:
            return None
        warm = self._tail_graph_pending.get(sig, 0) + 1
        if warm <= _TAIL_GRAPH_WARMUP:
            if len(self._tail_graph_pending) >= _TAIL_GRAPH_MAX_PENDING:
                # One-off signatures (eager-verify hidden buffers) never warm up.
                self._tail_graph_pending.clear()
            self._tail_graph_pending[sig] = warm
            return None
        self._tail_graph_pending.pop(sig, None)

        static_loc = torch.empty_like(cache_loc.reshape(-1))
        static_pos = torch.empty_like(positions.reshape(-1))
        static_commit = torch.empty_like(commit_lens)
        cache_loc_2d_shape = (bs, static_loc.numel() // bs)
        if self._tail_graph_pool is None:
            self._tail_graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        try:
            torch.cuda.synchronize()
            with torch.cuda.graph(
                graph, pool=self._tail_graph_pool, capture_error_mode="thread_local"
            ):
                self._append_target_hidden_to_draft_kv_by_loc(
                    target_hidden=target_hidden,
                    cache_loc=static_loc,
                    cache_loc_2d=static_loc.view(cache_loc_2d_shape),
                    positions=static_pos,
                    commit_lens=static_commit,
                )
        except Exception:
            logger.warning(
                "DFLASH draft-KV tail graph capture failed for bs=%d; "
                "this signature stays eager.",
                bs,
                exc_info=True,
            )
            self._tail_graph_failed.add(sig)
            return None
        entry = _TailGraph(
            graph=graph,
            cache_loc=static_loc,
            positions=static_pos,
            commit_lens=static_commit,
        )
        self._tail_graphs[sig] = entry
        if get_parallel().tp_rank == 0:
            logger.info(
                "DFLASH draft-KV tail graph captured bs=%d rows=%d",
                bs,
                int(target_hidden.shape[0]),
            )
        return entry

    def _mamba_commit_steps(
        self,
        *,
        batch: ScheduleBatch,
        commit_lens: torch.Tensor,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not (
            commit_lens.is_cuda
            and seq_lens_pre_verify.is_cuda
            and seq_lens_post_verify.is_cuda
        ):
            return super()._mamba_commit_steps(
                batch=batch,
                commit_lens=commit_lens,
                seq_lens_pre_verify=seq_lens_pre_verify,
                seq_lens_post_verify=seq_lens_post_verify,
            )
        from sglang.kernels.ops.speculative.k3_dflash_post import mamba_track_steps

        has_track = batch.mamba_track_indices is not None
        return mamba_track_steps(
            commit_lens,
            seq_lens_pre_verify,
            seq_lens_post_verify,
            mamba_track_grid(batch.tree_cache.page_size) if has_track else 1,
            has_track,
        )

    def _verify_argmax(self, next_token_logits: torch.Tensor) -> torch.Tensor:
        if (
            next_token_logits.is_cuda
            and next_token_logits.dtype == torch.float32
            and next_token_logits.dim() == 2
            and next_token_logits.stride(1) == 1
        ):
            from sglang.kernels.ops.speculative.k3_dflash_post import verify_argmax

            return verify_argmax(next_token_logits)
        return super()._verify_argmax(next_token_logits)
