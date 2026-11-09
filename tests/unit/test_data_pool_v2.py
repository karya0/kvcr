# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The v2 data pool is the granted regions, in the client and in the Guard."""

import ctypes
import errno
import logging
import mmap
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    _running_server,
)
from test_node_service_v2 import pools  # noqa: F401 - used through pytestmark

from kvcr.backends.local_disk import _G3Residency
from kvcr.config import GaaSSettings
from kvcr.runtime.core import _BlockRecord
from kvcr.service import guard as guard_module
from kvcr.service import memmap, node_service
from kvcr.service import memory as memory_module
from kvcr.service.node_service import SlotTable
from kvcr.service.protocol import (
    DATA_POOL_LAYOUT,
    KVCRServiceReasonError,
    PoolRegion,
    _map_data_pool,
    region_table_digest,
)
from kvcr.service.recovery_journal import (
    _RECORD_BLOCK,
    _RECOVERY_ENCODER,
    _project_recovery_record,
    read_handback,
    read_recovery_snapshot,
    write_recovery_snapshot,
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


def test_guard_logs_the_data_pool_at_promotion_not_at_claim(short_dir, caplog):
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        with caplog.at_level(logging.INFO, logger=guard_module.logger.name):
            hold = _claim(harness)
            try:
                # A claim released without promotion logs nothing.
                assert not [m for m in caplog.messages if "guard_data_pool" in m]
                slot_pool = table.get("t/l/0").pool
                base = slot_pool.address(slot_pool.regions[0])
                guard._core = MagicMock()
                name, slot_bytes = guard._configured.pool_layouts[0]
                address, size, *_ = guard._data_pool
                guard._core._local_dram._pools._regions = {
                    name: (address, size, slot_bytes)
                }
                with patch.object(guard._recovery, "prepare_to_serve", lambda r: r):
                    with patch.object(guard._recovery, "release_snapshot_region"):
                        guard._serve({}, time.monotonic(), time.monotonic())
                assert any("guard_promoted" in m for m in caplog.messages)
                lines = [m for m in caplog.messages if "guard_data_pool_" in m]
                assert lines == [
                    f"KVCR_EVENT guard_data_pool_serving guard=0 slot=t/l/0 "
                    f"base=0x{base:x} bytes={4 * PAGE} "
                    f"regions={len(slot_pool.regions)}"
                ]
            finally:
                guard._serving = False  # the mocked core has no state to hand back
                guard._core = None
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


def test_journal_lives_in_the_slot_meta_memfd_not_pool_rank(short_dir, caplog):
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        hold = _claim(harness)
        try:
            meta_fd = table.get("t/l/0").pool.meta_fd
            digest = bytes(range(32))
            hold._attachment.set_layout_digest(digest)
            # The client wrote through its mapping of the memfd the supervisor owns.
            assert os.pread(meta_fd, 32, 256) == digest
            # The Guard now reads the same bytes: its attachment is the meta memfd.
            assert guard._recovery.attachment.layout_digest() == digest
            spec = guard._recovery.attachment._spec
            assert (spec.device, spec.inode) == (
                os.fstat(meta_fd).st_dev,
                os.fstat(meta_fd).st_ino,
            )
            # pool_{rank} was left alone: nothing grew past it, no header written.
            pool_spec = guard._recovery._pool_spec
            assert os.stat(guard._spec.path).st_size == pool_spec.mapping_bytes
            with open(guard._spec.path, "rb") as pool_rank:
                pool_rank.seek(256)
                assert pool_rank.read(32) == bytes(32)
            assert len(hold.region_fds) == 1  # the meta fd is not a region fd
            with caplog.at_level(logging.INFO, logger=guard_module.logger.name):
                guard._core = MagicMock()
                name, slot_bytes = guard._configured.pool_layouts[0]
                address, size, *_ = guard._data_pool
                guard._core._local_dram._pools._regions = {
                    name: (address, size, slot_bytes)
                }
                with patch.object(guard._recovery, "prepare_to_serve", lambda r: r):
                    with patch.object(guard._recovery, "release_snapshot_region"):
                        guard._serve({}, time.monotonic(), time.monotonic())
            size = os.fstat(meta_fd).st_size
            offset = memmap.region_offset(meta_fd, spec.mapping_bytes)
            end = min(size, memory_module.meta_region_bytes(spec.journal_bytes))
            assert (
                f"KVCR_EVENT gaas_meta_region guard=0 slot=t/l/0 bytes={size} "
                f"journal_bytes={spec.journal_bytes} snapshot_budget={end - offset}"
            ) in caplog.messages
        finally:
            guard._serving = False
            guard._core = None
            hold.release()


def test_snapshot_round_trips_through_the_meta_memfd_and_reaches_a_replacement_client(
    short_dir,
):
    # The replacement is a new client on the same Guard process; a Guard process
    # restart is not a supported recovery path.
    server, table = _serve(short_dir)
    with server as harness:
        _attach(harness, table)
        guard = harness.server._server.registry._guards[0]
        _released_with_a_record(harness, guard)
        meta_fd = table.get("t/l/0").pool.meta_fd
        size = os.fstat(meta_fd).st_size  # the handback kept the memfd's size
        assert size == table._slot_meta_bytes(table.get("t/l/0"))
        assert os.stat(guard._spec.path).st_size == (
            guard._recovery._pool_spec.mapping_bytes
        )
        replacement = _claim_layout(harness, "l")
        try:
            # The replacement client reads what the Guard left, from the same memfd.
            recovered = read_handback(
                replacement._attachment, _TEST_DIGEST, replacement._pools
            )
            assert list(recovered.take_records()) == [_KEY]
            assert os.fstat(meta_fd).st_size == size
        finally:
            replacement.release()


def test_supervisor_pool_rank_is_a_sparse_stub_no_populate_touches(short_dir):
    # The supervisor cgroup pays for process memory only: journal and snapshot
    # live in the holder-charged meta memfd, so pool_{rank} is journal + one data
    # page (the smallest valid journal is a header plus one page) and unbacked.
    # This runs on tmpfs or a regular filesystem only; hugetlbfs page rounding of
    # the stub is out of scope here.
    populated = []
    real = memory_module._populate_pages

    def spy(fd, offset, length):
        populated.append((os.fstat(fd).st_ino, offset, length))
        return real(fd, offset, length)

    with patch.object(memory_module, "_populate_pages", spy):
        server, table = _serve(short_dir)
        with server as harness:
            _attach(harness, table)
            guard = harness.server._server.registry._guards[0]
            path = guard._recovery._pool_spec.path
            inode = guard._recovery._pool_spec.inode
            assert os.stat(path).st_size == 3 * PAGE  # 2 journal pages + 1 data
            assert os.stat(path).st_blocks == 0
            _released_with_a_record(harness, guard)  # a full v2 promotion and handback
            second = _claim_layout(harness, "l")
            second.release()
            assert os.stat(path).st_size == 3 * PAGE
            assert os.stat(path).st_blocks == 0  # nothing faulted a page in
            assert all(call[0] != inode for call in populated)
            # The grant still describes the configured journal, which the meta uses.
            assert guard._spec.mapping_bytes > 3 * PAGE


def test_meta_region_follows_the_configured_journal_bytes(short_dir):
    journal = 256 << 20  # the default constant would leave no room for this journal
    server, table = _serve(short_dir, journal_bytes=journal)
    with server as harness:
        _attach(harness, table)
        assert table.meta_bytes == node_service.meta_region_bytes(journal)
        hold = _claim(harness)
        try:
            attachment = hold._attachment
            assert attachment._spec.journal_bytes == journal
            assert attachment.snapshot_budget_bytes >= journal + 4096
        finally:
            hold.release()


def test_configured_journal_survives_a_restart_with_a_persisted_slot(short_dir):
    journal = 256 << 20
    server, table = _serve(short_dir, journal_bytes=journal)
    with server as harness:
        _attach(harness, table)
        _claim(harness).release()
    assert (short_dir / "slot-table.json").exists()
    # A new process: a default-sized table restores the slot, then the service binds.
    reborn = SlotTable(table.broker, table.cap_bytes, require_pod_identity=False)
    assert "t/l/0" in reborn._slots
    second = short_dir / "second"
    second.mkdir()
    with _running_server(second, slots=reborn, journal_bytes=journal) as harness:
        _attach(harness, reborn)
        assert reborn.meta_bytes == node_service.meta_region_bytes(journal)
        meta_fd = reborn._slots["t/l/0"].pool.meta_fd
        assert os.fstat(meta_fd).st_size == reborn._slot_meta_bytes(reborn.get("t/l/0"))
        hold = _claim(harness)
        try:
            assert hold._attachment._spec.journal_bytes == journal
        finally:
            hold.release()


def test_journal_bind_refusal_leaks_no_guard_or_pool(short_dir):
    _, table = _serve(short_dir)
    table.bind_journal_bytes(256 << 20)  # bound to a journal the service will not use
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(ValueError, match="bound to journal_bytes"):
        with _running_server(short_dir, slots=table, journal_bytes=1 << 20):
            pass
    assert {t.ident for t in threading.enumerate()} <= before
    assert list((short_dir / "pools").iterdir()) == []


def test_snapshot_beyond_the_meta_budget_is_refused_and_leaves_none(short_dir):
    # The journal is two pages in these services; the snapshot budget is one page.
    start = memmap.round_up(2 * PAGE + 4096, mmap.ALLOCATIONGRANULARITY)
    server, table = _serve(short_dir, meta_bytes=start + PAGE)
    with server as harness:
        _attach(harness, table)
        hold = _claim(harness)
        try:
            attachment = hold._attachment
            assert attachment.snapshot_budget_bytes == PAGE
            frame = (_RECORD_BLOCK, b"k", b"v" * 16)
            write_recovery_snapshot(attachment, b"terms", [frame])
            assert [f[1] for f in read_recovery_snapshot(attachment, b"terms")] == [
                b"k"
            ]
            big = (_RECORD_BLOCK, b"k", b"v" * 50000)
            with pytest.raises(OSError) as full:
                write_recovery_snapshot(attachment, b"terms", [big, big])
            assert full.value.errno == errno.ENOSPC
            assert list(read_recovery_snapshot(attachment, b"terms")) == []
        finally:
            hold.release()


@pytest.mark.parametrize("meta_bytes", [0, PAGE])
def test_claim_fails_closed_without_a_usable_meta_region(short_dir, meta_bytes):
    server, table = _serve(short_dir, meta_bytes=meta_bytes)
    with server as harness:
        _attach(harness, table)
        with pytest.raises(KVCRServiceReasonError, match="meta region") as refused:
            _claim(harness)
        assert refused.value.reason == DATA_POOL_LAYOUT
        deadline = time.monotonic() + 5
        while table.get("t/l/0").held and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not table.get("t/l/0").held


def test_meta_fd_identity_is_fixed_at_claim_and_fails_closed():
    recovery = object.__new__(guard_module._RecoveryState)
    recovery.meta_fd = recovery.meta_identity = None
    first = os.memfd_create("meta-a")
    second = os.memfd_create("meta-b")
    try:
        recovery.set_meta_fd(first)
        assert recovery.meta_fd not in (None, first)  # a dup, owned by the state
        assert recovery.meta_identity == (
            os.fstat(first).st_dev,
            os.fstat(first).st_ino,
        )
        os.dup2(second, recovery.meta_fd)  # the held descriptor is swapped
        with pytest.raises(ValueError, match="not the one"):
            recovery._attach_meta()
        recovery.set_meta_fd(None)
        assert recovery.meta_fd is None and recovery.meta_identity is None
    finally:
        os.close(first)
        os.close(second)


def test_data_pool_refuses_a_total_that_is_not_whole_blocks():
    # GaaS S4 r6: 50 GiB claimed, vLLM chunk 49 MiB (16 attention-sized pages of
    # 784 tokens), so the engine region is 1044 chunks and 44 MiB is left over.
    block = 49 << 20
    total = 50 << 30
    aligned = total // block * block
    assert (aligned, total - aligned) == (51156 << 20, 44 << 20)

    def grant(size):
        return SimpleNamespace(
            pools=(SimpleNamespace(block_size_bytes=block),),
            regions=(PoolRegion(0, 0, size, 0),),
        )

    with pytest.raises(KVCRServiceReasonError, match="whole number") as refused:
        _map_data_pool(grant(total), GaaSSettings("t", "l", total, total, total), [])
    assert refused.value.reason == DATA_POOL_LAYOUT

    fd = os.memfd_create("data-pool-aligned")  # sparse: nothing is touched
    try:
        os.ftruncate(fd, aligned)
        mapping = _map_data_pool(
            grant(aligned), GaaSSettings("t", "l", aligned, aligned, aligned), [fd]
        )
        try:
            assert mapping.size_bytes == aligned
        finally:
            mapping.close()
    finally:
        os.close(fd)
