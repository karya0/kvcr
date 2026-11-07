# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The v2 data pool is the granted regions, in the client and in the Guard."""

import ctypes
import logging
import mmap
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from test_claimant import (  # noqa: F401 - fixtures
    PAGE,
    _attach,
    _claim,
    _serve,
)
from test_kvcr_service import (  # noqa: F401 - _channels_are_taken is autouse
    _TEST_DIGEST,
    _channels_are_taken,
    _control_bind,
)
from test_node_service_v2 import pools  # noqa: F401 - used through pytestmark

from kvcr.backends.local_disk import _G3Residency
from kvcr.config import GaaSSettings
from kvcr.runtime.core import _BlockRecord
from kvcr.service import guard as guard_module
from kvcr.service.protocol import (
    DATA_POOL_LAYOUT,
    KVCRServiceReasonError,
    PoolRegion,
    region_table_digest,
)
from kvcr.service.recovery_journal import (
    _RECORD_BLOCK,
    _RECOVERY_ENCODER,
    _project_recovery_record,
)
from kvcr.service.regions import RegionMapping
from kvcr.types import BlockKey

pytestmark = pytest.mark.usefixtures("pools")


@pytest.fixture
def short_dir():  # a short path: the service sockets live under it
    with tempfile.TemporaryDirectory(prefix="gaas-dp-", dir="/tmp") as directory:
        yield Path(directory)


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
    short_dir,
):
    server, table = _serve(short_dir)
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
                hold.regions, hold.region_fds, hold.slot_name, hold.claim_generation
            )
        finally:
            hold.release()


def test_partial_grant_fails_closed_and_frees_the_slot(short_dir):
    server, table = _serve(short_dir, pages=3)  # 3 pages free, 4 requested, 2 minimum
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


def test_two_pool_tier_fails_closed(short_dir):
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        with pytest.raises(KVCRServiceReasonError, match="one pool") as refused:
            harness.client.claim(
                0,
                [("a", 1024), ("b", 1024)],
                _TEST_DIGEST,
                _control_bind(0),
                gaas=GaaSSettings("t", "l", 4 * PAGE, 4 * PAGE, 4 * PAGE),
                receive_timeout_s=10,
            )
        assert refused.value.reason == DATA_POOL_LAYOUT


def test_guard_registers_the_regions_not_pool_rank(short_dir):
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            slot_pool = table.get("t/l/0").pool
            expected = (slot_pool.address(slot_pool.regions[0]), 4 * PAGE)
            assert guard._data_pool[:2] == expected
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


def test_guard_logs_the_registered_data_pool_at_info(short_dir, caplog):
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            slot_pool = table.get("t/l/0").pool
            base = slot_pool.address(slot_pool.regions[0])
            del guard._prepare_core
            with caplog.at_level(logging.INFO, logger=guard_module.logger.name):
                with patch.object(guard_module, "_KVCRCore"):
                    guard._prepare_core()
            lines = [m for m in caplog.messages if "guard_data_pool_registered" in m]
            assert lines == [
                f"KVCR_EVENT guard_data_pool_registered guard=0 slot=t/l/0 "
                f"base=0x{base:x} bytes={4 * PAGE} "
                f"regions={len(slot_pool.regions)}"
            ]
            print(lines[0])
        finally:
            hold.release()


@pytest.mark.parametrize("skew", [False, True])
def test_guard_refuses_records_written_under_another_layout(
    short_dir,
    caplog,
    skew,
):
    server, table = _serve(short_dir)
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


def test_region_digest_names_the_backing_not_only_the_geometry():
    regions = (PoolRegion(0, 0, PAGE, 0),)
    first, second = _memfd(1), _memfd(1)
    try:
        same = region_table_digest(regions, [first])
        assert same == region_table_digest(regions, [first])
        assert same != region_table_digest(regions, [second])  # same geometry
        assert same != region_table_digest(regions, [first], "other-slot")
    finally:
        os.close(first)
        os.close(second)


def test_guard_drops_records_when_the_slot_backing_changed(
    short_dir,
    caplog,
):
    """Slot A's journal over slot B's memfd of identical geometry is cold."""
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            other = _memfd(4)
            try:
                hold._attachment.set_layout_digest(
                    region_table_digest(
                        hold.regions, [other], hold.slot_name, hold.claim_generation
                    )
                )
                with caplog.at_level(logging.WARNING):
                    assert guard._recovery.take_for_promotion() == {}
                assert any("stale_layout" in m for m in caplog.messages)
            finally:
                os.close(other)
        finally:
            hold.release()


def test_shrink_after_claim_changes_the_live_digest_and_drops_records(
    short_dir,
    caplog,
):
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            slot = table.get("t/l/0")
            claim_time = guard._recovery.layout_digest()
            assert hold._attachment.layout_digest() == claim_time
            slot.pool.mark_quiescent(slot.pool.regions[-1].index)
            slot.pool.shrink((slot.pool.regions[-1].index,))
            assert guard._recovery.layout_digest() != claim_time
            with caplog.at_level(logging.WARNING):
                assert guard._recovery.take_for_promotion() == {}
            assert any("stale_layout" in m for m in caplog.messages)
        finally:
            hold.release()


def test_an_unwritten_header_does_not_drop_the_mirror(
    short_dir,
    caplog,
):
    """A claim that failed before writing its digest leaves zeros and no records."""
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            hold._attachment.set_layout_digest(bytes(32))
            with caplog.at_level(logging.WARNING):
                guard._recovery.take_for_promotion()
            assert not any("stale_layout" in m for m in caplog.messages)
        finally:
            hold.release()


_KEY = BlockKey(b"slot-a-record")


def _seed(guard):
    """One record in the Guard's mirror, as a primary's journal would publish it."""
    record = _BlockRecord(g3=_G3Residency(("pool0",), (0,)))
    payload = _RECOVERY_ENCODER.encode(_project_recovery_record(record))
    guard._recovery.mirror.apply(_RECORD_BLOCK, bytes(_KEY), payload)


def _claim_layout(harness, layout):
    return harness.client.claim(
        0,
        [("pool0", 1024)],
        _TEST_DIGEST,
        _control_bind(0),
        gaas=GaaSSettings("t", layout, 4 * PAGE, 2 * PAGE, 8 * PAGE),
        receive_timeout_s=10,
    )


def _released_with_a_record(harness, guard, layout="l"):
    first = _claim_layout(harness, layout)
    _seed(guard)
    first.release()


@pytest.mark.parametrize("wrote_digest", [False, True])
def test_another_slots_claim_does_not_adopt_the_released_slots_records(
    short_dir, caplog, wrote_digest
):
    """Slot A's snapshot is its own: B on the same rank never inherits it."""
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        with caplog.at_level(logging.WARNING):
            _released_with_a_record(harness, guard)
            second = _claim_layout(harness, "other")
            try:
                if wrote_digest:
                    second._attachment.set_layout_digest(
                        region_table_digest(
                            second.regions,
                            second.region_fds,
                            second.slot_name,
                            second.claim_generation,
                        )
                    )
                else:
                    second._attachment.set_layout_digest(bytes(32))
                assert guard._recovery.take_for_promotion() == {}
            finally:
                second.release()
        assert any("stale_layout" in m for m in caplog.messages)


def test_reclaiming_the_same_backing_keeps_the_mirror(short_dir, caplog):
    """A claimant that dies before writing its digest resumes warm on its own slot."""
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        _released_with_a_record(harness, guard)
        again = _claim_layout(harness, "l")
        try:
            again._attachment.set_layout_digest(bytes(32))
            with caplog.at_level(logging.WARNING):
                assert list(guard._recovery.take_for_promotion()) == [_KEY]
            assert not any("stale_layout" in m for m in caplog.messages)
        finally:
            again.release()


def test_a_mirror_of_unknown_provenance_is_dropped(short_dir, caplog):
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim_layout(harness, "l")
        try:
            _seed(guard)
            guard._recovery._backing = None
            with caplog.at_level(logging.WARNING):
                assert guard._recovery.take_for_promotion() == {}
            assert any("stale_layout" in m for m in caplog.messages)
        finally:
            hold.release()
