# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The v2 data pool is the granted regions, in the client and in the Guard."""

import ctypes
import logging
import mmap
import os
import sys
import time
from unittest.mock import Mock, patch

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from test_claimant import (  # noqa: F401 - fixtures
    PAGE,
    _attach,
    _claim,
    _serve,
    tmp_path,
)
from test_kvcr_service import _channels_are_taken  # noqa: F401 - autouse fixture
from test_node_service_v2 import pools  # noqa: F401 - fixture

from kvcr.service import guard as guard_module
from kvcr.service.protocol import (
    DATA_POOL_LAYOUT,
    KVCRServiceReasonError,
    PoolRegion,
    region_table_digest,
)
from kvcr.service.regions import RegionMapping


def _memfd(pages: int) -> int:
    fd = os.memfd_create("data-pool-test")
    os.ftruncate(fd, pages * PAGE)
    return fd


def test_region_mapping_is_one_range_over_the_memfd():
    fd = _memfd(4)
    regions = (PoolRegion(0, 0, PAGE, 0), PoolRegion(1, PAGE, 3 * PAGE, 0))
    mapping = RegionMapping([fd], regions)
    try:
        assert mapping.size_bytes == 4 * PAGE
        with mmap.mmap(fd, 4 * PAGE) as direct:
            direct[PAGE - 1 : PAGE + 1] = b"ab"  # across the region boundary
            direct[4 * PAGE - 1] = ord("z")
        assert ctypes.string_at(mapping.address + PAGE - 1, 2) == b"ab"
        assert ctypes.string_at(mapping.address + 4 * PAGE - 1, 1) == b"z"
        ctypes.memset(mapping.address, 7, 1)  # writes land in the memfd
        assert os.pread(fd, 1, 0) == b"\x07"
    finally:
        mapping.close()
        os.close(fd)


@pytest.mark.parametrize(
    "regions, fds",
    [
        ((PoolRegion(0, PAGE, PAGE, 0),), 1),  # does not start at 0
        ((PoolRegion(0, 0, PAGE, 0), PoolRegion(1, 2 * PAGE, PAGE, 0)), 1),  # a hole
        ((PoolRegion(0, 0, PAGE, 1),), 1),  # fd_index past the fds
        ((), 1),
    ],
)
def test_region_mapping_refuses_a_layout_it_cannot_make_contiguous(regions, fds):
    fd = _memfd(4)
    try:
        with pytest.raises(ValueError):
            RegionMapping([fd] * fds, regions)
    finally:
        os.close(fd)


def test_client_pool_is_the_regions_and_the_journal_records_their_digest(
    tmp_path, pools  # noqa: F811
):
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        hold = _claim(harness)
        try:
            ((name, address, size),) = hold.local_dram.pools
            assert (name, size) == ("pool0", 4 * PAGE)
            with mmap.mmap(hold.region_fds[0], size, offset=0) as direct:
                direct[:5] = b"hello"
            assert ctypes.string_at(address, 5) == b"hello"
            assert hold._attachment.layout_digest() == region_table_digest(
                hold.regions
            )
        finally:
            hold.release()


def test_partial_grant_fails_closed_and_frees_the_slot(tmp_path, pools):  # noqa: F811
    server, table = _serve(tmp_path, pages=3)  # 3 pages free, 4 requested, 2 minimum
    with server as harness:
        _attach(harness, table)
        with pytest.raises(KVCRServiceReasonError) as refused:
            _claim(harness)
        assert refused.value.reason == DATA_POOL_LAYOUT
        assert "granted" in refused.value.message
        deadline = time.monotonic() + 5  # the service frees it on the release
        while table.get("t/l/0").held and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not table.get("t/l/0").held


def test_two_pool_tier_fails_closed(tmp_path, pools):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        from kvcr.config import GaaSSettings
        from test_kvcr_service import _TEST_DIGEST, _control_bind

        with pytest.raises(KVCRServiceReasonError) as refused:
            harness.client.claim(
                0,
                [("a", 1024), ("b", 1024)],
                _TEST_DIGEST,
                _control_bind(0),
                gaas=GaaSSettings("t", "l", 4 * PAGE, 4 * PAGE, 4 * PAGE),
                receive_timeout_s=10,
            )
        assert refused.value.reason == DATA_POOL_LAYOUT


def test_guard_registers_the_regions_not_pool_rank(tmp_path, pools):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            slot_pool = table.get("t/l/0").pool
            expected = (
                slot_pool.address(slot_pool.regions[0]),
                4 * PAGE,
                region_table_digest(slot_pool.regions),
            )
            assert guard._data_pool == expected
            # The real _prepare_core, with the core mocked: which range it registers.
            del guard._prepare_core
            with patch.object(guard_module, "_KVCRCore") as core:
                guard._prepare_core()
            dram = core.call_args.args[2].local_dram
            assert dram.pools == [("pool0", expected[0], 4 * PAGE)]
            # The same bytes the client's range shows: one memfd, two mappings.
            with mmap.mmap(hold.region_fds[0], PAGE) as direct:
                direct[:3] = b"abc"
            assert ctypes.string_at(expected[0], 3) == b"abc"
            assert ctypes.string_at(hold.local_dram.pools[0][1], 3) == b"abc"
        finally:
            hold.release()


@pytest.mark.parametrize("skew", [False, True])
def test_guard_refuses_records_written_under_another_layout(
    tmp_path, pools, caplog, skew  # noqa: F811
):
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            if skew:
                hold._attachment.set_layout_digest(b"\x01" * 32)
            with caplog.at_level(logging.WARNING):
                records = guard._recovery.take_for_promotion()
            assert records == {}
            refused = any("stale_layout" in m for m in caplog.messages)
            assert refused is skew
        finally:
            hold.release()
