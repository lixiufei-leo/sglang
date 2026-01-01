from typing import Optional

import msgspec

from sglang.srt.environ import envs

# A prefill forward's fixed cost is about 1/8 of a full chunk's: 66 ms of 756 ms
# measured for Kimi-K3.1 on MI355X at a 16384-token chunk.
_FIXED_COST_INTERVAL_DIVISOR = 8
# Arbitrary: an unfinished chunked prompt stops yielding to decode only below a
# third of the batch size at which the interval starts to apply.
_CHUNKED_MIN_BS_DIVISOR = 3


class PrefillWorkInterval(msgspec.Struct, frozen=True):
    """Sizes the prefill/decode interval by the prefill work that preceded it.

    Only used with --prefill-decode-interval-min-bs: there a running batch below
    the min admits prefill every round, so a prompt prefilled in chunks would
    run them back to back and stall decode for its whole prefill.
    """

    interval: int  # decode rounds held after a full chunk
    chunk_tokens: int
    min_rounds: int
    # Running batch size from which an unfinished chunked prompt keeps the interval.
    chunked_min_bs: int

    @classmethod
    def maybe_build(
        cls, *, interval: int, min_bs: int, chunk_tokens: Optional[int]
    ) -> Optional["PrefillWorkInterval"]:
        if (
            not envs.SGLANG_ENABLE_PREFILL_WORK_AWARE_ADMISSION.get()
            or interval <= 0
            or min_bs <= 0
            or chunk_tokens is None
        ):
            return None
        return cls(
            interval=interval,
            chunk_tokens=chunk_tokens,
            min_rounds=-(-interval // _FIXED_COST_INTERVAL_DIVISOR),
            chunked_min_bs=-(-min_bs // _CHUNKED_MIN_BS_DIVISOR),
        )

    def rounds_after(self, new_tokens: Optional[int]) -> int:
        """Decode rounds to hold after a prefill batch of new_tokens."""
        if new_tokens is None:
            return self.interval
        scaled = -(-self.interval * new_tokens // self.chunk_tokens)
        return min(self.interval, max(self.min_rounds, scaled))
