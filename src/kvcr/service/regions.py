# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Region pool: one memfd and a PROT_NONE reservation with MAP_FIXED regions."""

import ctypes
import mmap
import os
import types

from kvcr.service.memmap import round_up
from kvcr.service.protocol import PoolRegion

_MAP_FIXED = 0x10
_MAP_NORESERVE = 0x4000
_FALLOC_PUNCH = 0x01 | 0x02  # FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE
_DEFAULT_HUGE_PAGE = 2 << 20


class CapacityError(Exception):
    """A grow would exceed the effective maximum."""


class RegionBusy(Exception):
    """A shrink named a region the worker has not reported quiescent."""


class UnknownRegion(Exception):
    """A region index is not in the current layout."""


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
    return libc


def _check(result: int | None, what: str) -> int:
    if result is None or result == ctypes.c_void_p(-1).value:
        error = ctypes.get_errno()
        raise OSError(error, f"{what}: {os.strerror(error)}")
    return result


class RegionPool:
    """A memfd sized to the effective maximum; regions are mapped as the pool grows.

    Regions are appended at the high-water offset and mapped MAP_FIXED|MAP_SHARED at
    the same offset inside a PROT_NONE reservation, so the address of a region never
    changes. Shrink removes only the highest regions (a suffix of the layout), each
    already marked quiescent, so the high-water offset simply drops and the layout
    never has holes; the range is re-mapped PROT_NONE and its pages are punched out.
    Region indices are never reused. Pool data does not travel through the control
    socket: peers get ``pass_fd_list()`` over SCM_RIGHTS and map the regions.
    """

    def __init__(self, fd: int, maximum: int, page_size: int) -> None:
        self._libc = _libc()
        self._fd = fd
        self._max = maximum
        self._page = page_size
        self._regions: list[PoolRegion] = []
        self._quiescent: set[int] = set()
        self._next_index = 0
        self._generation = 0
        self._base = 0
        self._reserve = 0
        try:
            # Over-reserve so the base can be aligned to the page size (huge pages).
            self._reserve = maximum + page_size
            addr = self._libc.mmap(
                None,
                self._reserve,
                mmap.PROT_NONE,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE,
                -1,
                0,
            )
            self._raw_base = _check(addr, "pool reservation")
            self._base = round_up(self._raw_base, page_size)
        except BaseException:
            self.close()
            raise

    @classmethod
    def create(
        cls,
        effective_maximum_bytes: int,
        *,
        page_size: int | None = None,
        hugetlb: bool = False,
    ) -> "RegionPool":
        page = page_size or (_DEFAULT_HUGE_PAGE if hugetlb else mmap.PAGESIZE)
        maximum = round_up(effective_maximum_bytes, page)
        flags = os.MFD_CLOEXEC | (os.MFD_HUGETLB if hugetlb else 0)
        fd = os.memfd_create("kvcr-region-pool", flags)
        try:
            os.ftruncate(fd, maximum)
            return cls(fd, maximum, page)
        except BaseException:
            os.close(fd)
            raise

    @property
    def fd(self) -> int:
        return self._fd

    @property
    def regions(self) -> tuple[PoolRegion, ...]:
        return tuple(self._regions)

    @property
    def layout_generation(self) -> int:
        return self._generation

    @property
    def effective_maximum_bytes(self) -> int:
        return self._max

    @property
    def used_bytes(self) -> int:
        return sum(region.size for region in self._regions)

    def address(self, region: PoolRegion) -> int:
        """The process address of a mapped region."""
        return self._base + region.offset

    def pass_fd_list(self) -> list[int]:
        """The fds to send over SCM_RIGHTS; regions index into the memfd by offset."""
        return [self._fd]

    def grow(
        self, additional_bytes: int, split: int | None = None
    ) -> tuple[PoolRegion, ...]:
        """Append ``split`` (default one) equal page-aligned regions."""
        count = split or 1
        size = round_up(-(-additional_bytes // count), self._page)
        offset = self.used_bytes
        if offset + size * count > self._max:
            raise CapacityError(
                f"grow of {size * count} bytes exceeds maximum {self._max}"
                f" ({offset} used)"
            )
        added = []
        for _ in range(count):
            addr = self._libc.mmap(
                self._base + offset,
                size,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED | _MAP_FIXED,
                self._fd,
                offset,
            )
            _check(addr, "region mmap")
            region = PoolRegion(self._next_index, offset, size, 0)
            self._next_index += 1
            self._regions.append(region)
            added.append(region)
            offset += size
        self._generation += 1
        return tuple(added)

    def mark_quiescent(self, index: int) -> None:
        """Record that the worker no longer uses region ``index``."""
        if all(region.index != index for region in self._regions):
            raise UnknownRegion(index)
        self._quiescent.add(index)

    def shrink(self, region_indices: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        """Remove whole quiescent regions; only the highest regions may go.

        ``region_indices`` must be a suffix of the current layout, else RegionBusy
        (a lower region still stands above no hole). Unknown indices raise
        UnknownRegion. Nothing changes unless every named region is removable.
        """
        wanted = set(region_indices)
        known = {region.index for region in self._regions}
        if wanted - known:
            raise UnknownRegion(sorted(wanted - known))
        if wanted - self._quiescent:
            raise RegionBusy(sorted(wanted - self._quiescent))
        removed = [r for r in self._regions if r.index in wanted]
        if removed != self._regions[len(self._regions) - len(removed) :]:
            raise RegionBusy("only the highest regions can be removed")
        for region in reversed(removed):
            addr = self._libc.mmap(
                self._base + region.offset,
                region.size,
                mmap.PROT_NONE,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE | _MAP_FIXED,
                -1,
                0,
            )
            _check(addr, "region unmap")
            self._libc.fallocate(self._fd, _FALLOC_PUNCH, region.offset, region.size)
            self._regions.remove(region)
            self._quiescent.discard(region.index)
        if removed:
            self._generation += 1
        return tuple(region.index for region in removed)

    def close(self) -> None:
        if self._base:
            self._libc.munmap(self._raw_base, self._reserve)
            self._base = 0
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._regions.clear()

    def __enter__(self) -> "RegionPool":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()
