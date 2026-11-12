# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The G2 index kept in each data pool's own header.

Each pool of a v2 data pool is a header followed by its block data. The header
is a 64-byte prefix (magic, version, key size, block size, entry count) and one
entry per block slot: a u64 flag word, then the key. The entry's position is the
slot id. The flag word holds the entry state in its low 32 bits (0 free, odd
being written, even committed) and the block's G2 order position as an int32
in its high 32 bits.
"""

import ctypes
import ctypes.util
import logging
import mmap
import struct
from functools import cache
from typing import Any

from .memmap import round_up

logger = logging.getLogger(__name__)

_ACQUIRE, _RELEASE = 2, 3


@cache
def _atomics() -> tuple[Any, Any]:
    """Bind libatomic's 8-byte acquire/release operations on first use.

    Late because libatomic is a system library, not a declared dependency:
    importing kvcr must not require what only recovery uses.
    """
    # Keep the GIL across these tiny atomics to avoid a handoff for every frame.
    library = ctypes.PyDLL(ctypes.util.find_library("atomic") or "libatomic.so.1")
    load, store = library["__atomic_load_8"], library["__atomic_store_8"]
    load.argtypes, load.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_uint64
    store.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    store.restype = None
    return load, store


class _AtomicU64:
    def __init__(self, address: int) -> None:
        self._address = address
        self._load, self._store = _atomics()

    def load_acquire(self) -> int:
        return int(self._load(self._address, _ACQUIRE))

    def store_release(self, value: int) -> None:
        self._store(self._address, value, _RELEASE)


_PREFIX = struct.Struct("<QIIQQ")  # magic, version, key size, block size, entries
_PREFIX_BYTES = 64
_MAGIC_BYTES = b"KVCRG2PH"
_MAGIC = int.from_bytes(_MAGIC_BYTES, "little")
_VERSION = 1
_STATE = 0xFFFFFFFF
_WRITING, _COMMITTED = 1, 2


def entry_bytes(key_size: int) -> int:
    return round_up(8 + key_size, 8)


def pool_ranges(
    size: int, pool_layouts: list[tuple[str, int]], key_size: int | None
) -> list[tuple[str, int, int, int]]:
    """Split a data pool of ``size`` bytes into its pools, the same on every side.

    Each pool holds the same number of blocks: its header (none without a key
    size), then its block data from the next page. Returns (name, header offset,
    data offset, data bytes) per pool; refuses a size the pools do not fill.
    """

    def layout(n: int) -> tuple[int, list[tuple[str, int, int, int]]]:
        offset, ranges = 0, []
        for name, block in pool_layouts:
            header = round_up(offset, 8)
            if key_size is not None:
                offset = round_up(
                    header + _PREFIX_BYTES + n * entry_bytes(key_size), mmap.PAGESIZE
                )
            ranges.append((name, header, offset, n * block))
            offset += n * block
        return offset, ranges

    low, high = 0, size // sum(block for _, block in pool_layouts)
    while low < high:  # the largest block count that fits
        middle = (low + high + 1) // 2
        low, high = (middle, high) if layout(middle)[0] <= size else (low, middle - 1)
    used, ranges = layout(low)
    if not low or used != size:
        raise ValueError(
            f"{size} bytes is not a whole number of blocks for {pool_layouts}, "
            f"key size {key_size}; {used} bytes holds {low} per pool"
        )
    return ranges


def pool_headers(
    address: int,
    ranges: list[tuple[str, int, int, int]],
    pool_layouts: list[tuple[str, int]],
    key_size: int | None,
    *,
    initialize: bool = False,
) -> "dict[str, PoolHeader]":
    """Each pool's verified header at the ``pool_ranges`` of a data pool mapped at
    address; ``initialize`` as for ``PoolHeader.verify``."""
    if key_size is None:
        return {}
    headers = {
        name: PoolHeader(address + header, key_size, block, size // block)
        for (name, header, _, size), (_, block) in zip(ranges, pool_layouts)
    }
    for header in headers.values():
        header.verify(initialize=initialize)
    return headers


def has_header(address: int) -> bool:
    """Whether the data pool mapped at ``address`` starts with a pool header."""
    return ctypes.string_at(address, len(_MAGIC_BYTES)) == _MAGIC_BYTES


class PoolHeader:
    """One pool's header in mapped memory. A slot's entry is cleared before the
    slot is reused, so a committed entry always names the slot's current key."""

    def __init__(self, address: int, key_size: int, block_size: int, n: int) -> None:
        self.address, self.key_size, self.n = address, key_size, n
        self._entry = entry_bytes(key_size)
        self._prefix = _PREFIX.pack(_MAGIC, _VERSION, key_size, block_size, n)
        self._load, self._store = _atomics()

    def verify(self, *, initialize: bool = False) -> None:
        """Refuse a header written for another key size, block size or count.

        ``initialize`` (the Guard, which maps the pool first) writes a blank one.
        """
        current = ctypes.string_at(self.address, _PREFIX.size)
        if initialize and not any(current):
            ctypes.memmove(self.address, self._prefix, _PREFIX.size)
        elif current != self._prefix:
            raise ValueError(
                "pool header does not match this claim's key size, block size "
                f"and block count ({self.key_size}, {self.n})"
            )

    def _flag(self, slot: int) -> int:
        return self.address + _PREFIX_BYTES + slot * self._entry

    def publish(self, slot: int, key: bytes, position: int) -> None:
        """Commit ``key`` at ``slot``; the same key committed only moves position."""
        if len(key) != self.key_size:
            raise ValueError(f"key is {len(key)} bytes; the pool's is {self.key_size}")
        flag = self._flag(slot)
        if (
            self._load(flag, _ACQUIRE) & _STATE != _COMMITTED
            or ctypes.string_at(flag + 8, self.key_size) != key
        ):
            self._store(flag, _WRITING, _RELEASE)
            ctypes.memmove(flag + 8, key, self.key_size)
        self._store(flag, (position & _STATE) << 32 | _COMMITTED, _RELEASE)

    def clear(self, slot: int) -> None:
        self._store(self._flag(slot), 0, _RELEASE)

    def scan(self) -> list[tuple[int, int, bytes]]:
        """(slot, flag word, key) of every entry that is not free.

        Only with the writer gone: one copy of the table, not a seqlock read.
        """
        table = ctypes.string_at(self.address + _PREFIX_BYTES, self.n * self._entry)
        entry = struct.Struct(f"<Q{self.key_size}s{self._entry - 8 - self.key_size}x")
        return [
            (slot, word, key)
            for slot, (word, key) in enumerate(entry.iter_unpack(table))
            if word
        ]


def rebuild(
    headers: dict[str, PoolHeader],
) -> dict[bytes, tuple[tuple[tuple[str, int], ...], int]]:
    """Keys committed in every pool, as (slots in pool order, position).

    The caller now owns the pools: every other entry -- torn, or a key missing
    from some pool -- is cleared, since its slot is about to be handed out free.
    """
    names = list(headers)
    scans = {name: header.scan() for name, header in headers.items()}
    found: dict[bytes, list[tuple[str, int, int]]] = {}
    torn = 0
    for name, entries in scans.items():
        for slot, word, key in entries:
            if word & 1:
                torn += 1
            else:
                position = ctypes.c_int32(word >> 32).value
                found.setdefault(key, []).append((name, slot, position))
    present = {
        key: (tuple((name, slot) for name, slot, _ in entries), entries[0][2])
        for key, entries in found.items()
        if [name for name, _, _ in entries] == names
    }
    kept = {(name, slot) for slots, _ in present.values() for name, slot in slots}
    for name, entries in scans.items():
        for slot, _, _ in entries:
            if (name, slot) not in kept:
                headers[name].clear(slot)
    logger.info(
        "KVCR_EVENT g2_index_rebuilt blocks=%d torn=%d partial=%d",
        len(present),
        torn,
        len(found) - len(present),
    )
    return present
