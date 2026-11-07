# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Region pool: memfd reservation with MAP_FIXED regions (Linux only)."""

import ctypes
import mmap
import os
import sys

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from kvcr.service.regions import CapacityError, RegionBusy, RegionPool, UnknownRegion

PAGE = mmap.PAGESIZE


@pytest.fixture
def pool():
    with RegionPool.create(8 * PAGE) as created:
        yield created


def test_grow_aliases_memfd(pool):
    (region,) = pool.grow(2 * PAGE)
    assert (region.offset, region.size) == (0, 2 * PAGE)
    buf = (ctypes.c_char * region.size).from_address(pool.address(region))
    buf[0:5] = b"hello"
    assert os.pread(pool.fd, 5, region.offset) == b"hello"
    os.pwrite(pool.fd, b"world", PAGE)
    assert bytes(buf[PAGE : PAGE + 5]) == b"world"
    assert pool.used_bytes == 2 * PAGE
    assert pool.pass_fd_list() == [pool.fd]


def test_grow_split_and_generation(pool):
    assert pool.layout_generation == 0
    regions = pool.grow(4 * PAGE, split=2)
    assert [(r.offset, r.size) for r in regions] == [(0, 2 * PAGE), (2 * PAGE, 2 * PAGE)]
    assert pool.layout_generation == 1
    pool.grow(PAGE)
    assert pool.layout_generation == 2
    assert pool.regions[-1].offset == 4 * PAGE


def test_grow_past_maximum(pool):
    pool.grow(6 * PAGE)
    with pytest.raises(CapacityError):
        pool.grow(3 * PAGE)
    assert pool.layout_generation == 1


def test_shrink_requires_quiescent(pool):
    (region,) = pool.grow(PAGE)
    with pytest.raises(RegionBusy):
        pool.shrink([region.index])
    with pytest.raises(UnknownRegion):
        pool.mark_quiescent(99)
    assert pool.regions == (region,)


def test_shrink_highest_only_and_reuse(pool):
    low, high = pool.grow(2 * PAGE, split=2)
    pool.mark_quiescent(low.index)
    with pytest.raises(RegionBusy):
        pool.shrink([low.index])
    pool.mark_quiescent(high.index)
    generation = pool.layout_generation
    assert pool.shrink([high.index]) == (high.index,)
    assert pool.regions == (low,)
    assert pool.layout_generation == generation + 1
    (again,) = pool.grow(PAGE)
    assert again.offset == high.offset
    assert again.index not in (low.index, high.index)


def test_close_frees_fd():
    pool = RegionPool.create(2 * PAGE)
    fd = pool.fd
    pool.grow(PAGE)
    pool.close()
    with pytest.raises(OSError):
        os.fstat(fd)
