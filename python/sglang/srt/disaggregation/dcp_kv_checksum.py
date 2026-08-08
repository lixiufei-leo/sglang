"""Byte-level digests of DSV4 compressed KV rows for physical-DCP diagnosis.

Physical PD relays prefill KV over the network and re-lays it out across the
decode ranks.  DCP only changes the decode attention path, so a physical PD run
and a standalone DCP run should decode the *same* KV bytes; if they disagree the
relayout is not byte-exact.  Nothing else in the stack checks that: the row
mapping has unit tests and the transport reports completion, but the values
themselves are never compared.

Both sides digest their rows independently and log them.  The prefill side
digests the source rows it sends, the decode side digests the destination rows
it received, and the two logs are joined offline on
``(room, dcp_rank, descriptor)``.  Equal digests prove the relayout is
byte-exact end to end.

Device memory is read through the HIP runtime because the transport layer only
holds raw pointers, not the pool tensors.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_HIP_MEMCPY_DEVICE_TO_HOST = 2
_hip_lib: Optional[ctypes.CDLL] = None
_hip_lock = threading.Lock()


def _load_hip() -> Optional[ctypes.CDLL]:
    global _hip_lib
    if _hip_lib is not None:
        return _hip_lib
    with _hip_lock:
        if _hip_lib is not None:
            return _hip_lib
        for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            lib.hipMemcpy.restype = ctypes.c_int
            lib.hipMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _hip_lib = lib
            return _hip_lib
    logger.warning("[kv-checksum] no HIP runtime available; digests disabled")
    return None


def contiguous_runs(rows: npt.NDArray[np.int64]) -> List[Tuple[int, int]]:
    """Split sorted row indices into (start_row, count) runs."""

    if rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [rows.size]))
    return [(int(rows[s]), int(e - s)) for s, e in zip(starts, ends)]


def digest_rows(
    base_ptr: int,
    row_bytes: int,
    rows: npt.NDArray[np.int64],
) -> Optional[Dict[int, str]]:
    """Return ``{row_index: digest}`` for the given rows of a device buffer.

    ``base_ptr`` already points at the compressed region (the SWA ring is
    excluded by the descriptor), so row ``i`` lives at ``base_ptr + i*row_bytes``.
    """

    lib = _load_hip()
    if lib is None:
        return None

    ordered = np.sort(np.asarray(rows, dtype=np.int64))
    if ordered.size == 0:
        return {}

    digests: Dict[int, str] = {}
    for start_row, count in contiguous_runs(ordered):
        nbytes = row_bytes * count
        host = (ctypes.c_ubyte * nbytes)()
        rc = lib.hipMemcpy(
            ctypes.byref(host),
            ctypes.c_void_p(base_ptr + start_row * row_bytes),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(_HIP_MEMCPY_DEVICE_TO_HOST),
        )
        if rc != 0:
            logger.warning(
                "[kv-checksum] hipMemcpy failed rc=%d ptr=%#x rows=%d..%d",
                rc,
                base_ptr,
                start_row,
                start_row + count,
            )
            return None
        blob = bytes(host)
        for offset in range(count):
            row = start_row + offset
            chunk = blob[offset * row_bytes : (offset + 1) * row_bytes]
            digests[row] = hashlib.sha256(chunk).hexdigest()[:16]
    return digests


def combine(digests: Dict[int, str]) -> str:
    """Fold per-row digests into one order-independent-by-row-index digest."""

    accumulator = hashlib.sha256()
    for row in sorted(digests):
        accumulator.update(str(row).encode("ascii"))
        accumulator.update(digests[row].encode("ascii"))
    return accumulator.hexdigest()[:32]


def emit(
    side: str,
    room: int,
    dcp_rank: int,
    descriptor_id: int,
    digests: Dict[int, str],
) -> None:
    """Log one descriptor digest in a format the offline joiner can parse."""

    logger.info(
        "[kv-checksum] side=%s room=%d dcp_rank=%d desc=%d rows=%d digest=%s",
        side,
        room,
        dcp_rank,
        descriptor_id,
        len(digests),
        combine(digests),
    )
