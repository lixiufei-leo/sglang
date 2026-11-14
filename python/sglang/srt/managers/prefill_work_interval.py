from typing import Optional

import msgspec

from sglang.srt.environ import envs

# Arbitrary: a batch of up to 3/4 of a chunk holds the configured interval, so
# a full chunk holds 4/3 of it.
_BASE_CHUNK_NUMERATOR = 3
_BASE_CHUNK_DENOMINATOR = 4


class PrefillWorkInterval(msgspec.Struct, frozen=True):
    """Lengthens the prefill/decode interval after a prefill batch with many new tokens.

    The interval is a count of decode rounds, whatever the prefill before it
    cost: back-to-back full chunks of a long prompt take a larger share of the
    wall clock than the same interval gives short batches. Holding in proportion
    to the batch's new tokens bounds that share. The interval is never shortened.
    """

    interval: int  # decode rounds held after a batch of up to base_tokens
    base_tokens: int  # new tokens in one prefill batch

    @classmethod
    def maybe_build(
        cls, *, interval: int, chunk_tokens: Optional[int]
    ) -> Optional["PrefillWorkInterval"]:
        if (
            not envs.SGLANG_ENABLE_PREFILL_DECODE_INTERVAL_SCALING.get()
            or interval <= 0
            or chunk_tokens is None
        ):
            return None
        base_tokens = chunk_tokens * _BASE_CHUNK_NUMERATOR // _BASE_CHUNK_DENOMINATOR
        if base_tokens <= 0:
            return None
        return cls(interval=interval, base_tokens=base_tokens)

    def rounds_after(self, new_tokens: Optional[int]) -> int:
        """Decode rounds to hold after a prefill batch of new_tokens."""
        if new_tokens is None:
            return self.interval
        scaled = -(-self.interval * new_tokens // self.base_tokens)
        return max(self.interval, scaled)

    def describe(self, chunk_tokens: int) -> str:
        return (
            "Prefill/decode interval scales with prefill work: "
            f"{self.interval} decode rounds after a prefill batch of up to "
            f"{self.base_tokens} new tokens, {self.rounds_after(chunk_tokens)} after "
            f"a full {chunk_tokens}-token chunk."
        )
