"""The in-graph accept's six results written into its static buffers in one launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _accept_outs_kernel(
    correct_len_ptr,
    bonus_ptr,
    cap_trim_ptr,
    prefix_lens_ptr,
    draft_tokens_ptr,
    correct_len_out_ptr,
    bonus_out_ptr,
    cap_trim_out_ptr,
    commit_lens_out_ptr,
    new_seq_lens_out_ptr,
    out_tokens_ptr,
    draft_row_stride,
    out_row_stride,
    bs,
    gamma,
    T,
    BLOCK_R: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = r < bs
    correct = tl.load(correct_len_ptr + r, mask=row_mask, other=0)
    bonus = tl.load(bonus_ptr + r, mask=row_mask, other=0)
    trim = tl.load(cap_trim_ptr + r, mask=row_mask, other=0).to(tl.int32)
    prefix = tl.load(prefix_lens_ptr + r, mask=row_mask, other=0)
    commit = correct.to(tl.int32) + 1
    new_seq = prefix + commit.to(prefix.dtype)

    tl.store(
        correct_len_out_ptr + r,
        correct.to(correct_len_out_ptr.dtype.element_ty),
        mask=row_mask,
    )
    tl.store(bonus_out_ptr + r, bonus.to(bonus_out_ptr.dtype.element_ty), mask=row_mask)
    tl.store(
        cap_trim_out_ptr + r, trim.to(cap_trim_out_ptr.dtype.element_ty), mask=row_mask
    )
    tl.store(
        commit_lens_out_ptr + r,
        commit.to(commit_lens_out_ptr.dtype.element_ty),
        mask=row_mask,
    )
    tl.store(
        new_seq_lens_out_ptr + r,
        new_seq.to(new_seq_lens_out_ptr.dtype.element_ty),
        mask=row_mask,
    )

    # Out tokens: the drafts, 0 past gamma, the bonus at column correct_len.
    k = tl.arange(0, BLOCK_T)[None, :]
    rows = r[:, None].to(tl.int64)
    mask = row_mask[:, None] & (k < T)
    draft = tl.load(
        draft_tokens_ptr + rows * draft_row_stride + k,
        mask=mask & (k < gamma),
        other=0,
    ).to(tl.int64)
    val = tl.where(
        k == correct[:, None].to(tl.int32),
        bonus[:, None].to(tl.int64),
        tl.where(k < gamma, draft, 0),
    )
    tl.store(out_tokens_ptr + rows * out_row_stride + k, val, mask=mask)


def write_accept_outs_triton(
    *,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_tokens: torch.Tensor,
    gamma: int,
    verify_num_draft_tokens: int,
    out,
) -> None:
    """``out`` (an ``AcceptOuts`` of ``[bs]`` / ``[bs, verify_num_draft_tokens]``
    buffers) gets what ``FinalizeAcceptLens`` and ``BuildOutTokens`` return, with
    ``correct_len`` and ``bonus`` copied through."""
    bs = correct_len.shape[0]
    T = verify_num_draft_tokens
    assert gamma <= T
    for t in (correct_len, bonus, cap_trim_lens, prefix_lens):
        assert t.shape == (bs,)
    assert draft_tokens.shape == (bs, gamma)
    for t in (
        out.correct_len,
        out.bonus,
        out.cap_trim_lens,
        out.commit_lens,
        out.new_seq_lens,
    ):
        assert t.shape == (bs,) and t.stride(0) == 1
    assert out.out_tokens.shape == (bs, T) and out.out_tokens.stride(1) == 1
    assert out.out_tokens.dtype == torch.int64
    if bs == 0:
        return
    if draft_tokens.stride(1) != 1:
        draft_tokens = draft_tokens.contiguous()
    block_r = min(64, triton.next_power_of_2(bs))
    _accept_outs_kernel[(triton.cdiv(bs, block_r),)](
        correct_len.contiguous(),
        bonus.contiguous(),
        cap_trim_lens.contiguous(),
        prefix_lens.contiguous(),
        draft_tokens,
        out.correct_len,
        out.bonus,
        out.cap_trim_lens,
        out.commit_lens,
        out.new_seq_lens,
        out.out_tokens,
        draft_tokens.stride(0),
        out.out_tokens.stride(0),
        bs,
        gamma,
        T,
        BLOCK_R=block_r,
        BLOCK_T=triton.next_power_of_2(T),
        num_warps=4,
    )
