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

from kvcr.service import regions as regions_module
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
    assert [(r.offset, r.size) for r in regions] == [
        (0, 2 * PAGE),
        (2 * PAGE, 2 * PAGE),
    ]
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


def test_shrink_punches_the_suffix_in_one_fallocate(pool, monkeypatch):
    first, second, third = pool.grow(3 * PAGE, split=3)
    for region in (second, third):
        pool.mark_quiescent(region.index)
    calls = []
    real = pool._libc.fallocate

    def fallocate(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(pool._libc, "fallocate", fallocate)
    pool.shrink([second.index, third.index])
    assert len(calls) == 1
    assert calls[0][2:] == (second.offset, second.size + third.size)
    assert pool.regions == (first,)


def test_close_frees_fd():
    pool = RegionPool.create(2 * PAGE)
    fd = pool.fd
    pool.grow(PAGE)
    pool.close()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_failed_reservation_closes_the_memfd_once(monkeypatch):
    made, closed = [], []
    real_memfd, real_close = os.memfd_create, os.close

    def memfd_create(*args, **kwargs):
        made.append(real_memfd(*args, **kwargs))
        return made[-1]

    def close(fd):
        closed.append(fd)
        real_close(fd)

    def failing_check(result, what):
        raise OSError(12, f"{what}: injected")

    monkeypatch.setattr(os, "memfd_create", memfd_create)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(regions_module, "_check", failing_check)
    with pytest.raises(OSError, match="injected"):
        RegionPool.create(2 * PAGE)
    assert len(made) == 1 and closed.count(made[0]) == 1


def test_shrink_fallocate_failure_changes_nothing(pool, monkeypatch):
    low, high = pool.grow(2 * PAGE, split=2)
    pool.mark_quiescent(high.index)
    generation = pool.layout_generation
    monkeypatch.setattr(pool._libc, "fallocate", lambda *args: -1)
    with pytest.raises(OSError):
        pool.shrink([high.index])
    assert pool.regions == (low, high) and pool.layout_generation == generation


def test_rollback_grow_restores_layout_and_ids(pool):
    pool.grow(PAGE)
    generation, next_index = pool.layout_generation, pool.next_index
    added = pool.grow(2 * PAGE, split=2)
    pool.rollback_grow(added, generation, next_index)
    assert len(pool.regions) == 1 and pool.layout_generation == generation
    assert pool.grow(PAGE)[0].index == next_index


def test_restore_remaps_a_persisted_layout(pool):
    regions = pool.grow(2 * PAGE, split=2)
    with RegionPool.create(8 * PAGE) as other:
        other.restore(list(regions), pool.next_index, pool.layout_generation)
        assert other.regions == regions
        assert other.grow(PAGE)[0].offset == 2 * PAGE


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_adoption_keeps_fd_ownership_until_retry(monkeypatch, fail_at):
    with (
        RegionPool.create(8 * PAGE, meta_bytes=PAGE) as pool,
        RegionPool.create(8 * PAGE, meta_bytes=PAGE) as held,
    ):
        pool.grow(2 * PAGE, split=2)
        held.grow(2 * PAGE, split=2)
        os.pwrite(held.fd, b"retained", PAGE)
        original = pool.pass_fd_list()
        layout = (pool.regions, pool.next_index, pool.layout_generation)
        received = [os.dup(fd) for fd in held.pass_fd_list()]
        real_mmap, calls = pool._libc.mmap, []

        def fail_mapping(*args):
            calls.append(args)
            return (
                ctypes.c_void_p(-1).value if len(calls) == fail_at else real_mmap(*args)
            )

        monkeypatch.setattr(pool._libc, "mmap", fail_mapping)
        with pytest.raises(OSError, match="region mmap"):
            pool.adopt(received)
        assert pool.pass_fd_list() == original
        assert (pool.regions, pool.next_index, pool.layout_generation) == layout
        for fd in original:
            os.fstat(fd)
        # The handback caller still owns and closes the failed inputs. Another
        # connection can immediately reuse those descriptor numbers.
        for fd in received:
            os.close(fd)
        unrelated = os.open(os.devnull, os.O_RDONLY)
        try:
            for fd in received:
                os.dup2(unrelated, fd)
            monkeypatch.setattr(pool._libc, "mmap", real_mmap)
            pool.adopt([os.dup(fd) for fd in held.pass_fd_list()])
            assert os.pread(pool.fd, 8, PAGE) == b"retained"
            assert (pool.regions, pool.next_index, pool.layout_generation) == layout
            pool.close()
            for fd in received:
                assert os.fstat(fd) == os.fstat(unrelated)
        finally:
            for fd in set(received + [unrelated]):
                os.close(fd)


def test_failed_split_grow_leaves_the_layout_untouched(pool, monkeypatch):
    pool.grow(PAGE)
    layout = (pool.regions, pool.layout_generation, pool.next_index)
    real, calls = pool._libc.mmap, []

    def mmap_fails_second_region(*args):
        calls.append(args)
        return ctypes.c_void_p(-1).value if len(calls) == 2 else real(*args)

    monkeypatch.setattr(pool._libc, "mmap", mmap_fails_second_region)
    with pytest.raises(OSError):
        pool.grow(4 * PAGE, split=2)
    monkeypatch.undo()
    assert (pool.regions, pool.layout_generation, pool.next_index) == layout
    assert pool.grow(PAGE)[0].index == layout[2]


def test_meta_memfd_is_the_pools_second_fd_and_survives_grow_and_shrink():
    with RegionPool.create(8 * PAGE, meta_bytes=4 * PAGE) as pool:
        assert pool.pass_fd_list() == [pool.fd, pool.meta_fd]
        assert os.fstat(pool.meta_fd).st_size == 4 * PAGE
        assert os.fstat(pool.meta_fd).st_mode & 0o777 == 0o600
        os.pwrite(pool.meta_fd, b"journal", 0)
        low, high = pool.grow(2 * PAGE, split=2)
        pool.mark_quiescent(high.index)
        pool.shrink([high.index])
        assert os.pread(pool.meta_fd, 7, 0) == b"journal"
        fds = (pool.fd, pool.meta_fd)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    with RegionPool.create(PAGE) as bare:  # no meta: only the region memfd
        assert bare.meta_fd == -1 and bare.pass_fd_list() == [bare.fd]
