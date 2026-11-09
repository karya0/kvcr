# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""H3: a reattached Guard leaves the live worker's pool headers and journal alone."""

import ctypes
import mmap
import os
import socket
import tempfile
from unittest.mock import Mock, call

import pytest
from test_guard import _PAGE_SPEC, _fake_attachment, _guard, _tier

from kvcr.service import guard as guard_module
from kvcr.service.g2_index import pool_headers, pool_ranges
from kvcr.service.guard import _Phase
from kvcr.service.protocol import DATA_POOL_LAYOUT, KVCRServiceReasonError
from kvcr.types import BlockKey

PAGE = mmap.PAGESIZE
KEY = 36
_MAPPINGS = []  # headers and the Guard's core hold raw addresses into these


def _recovery(monkeypatch, *, invalid=False):
    """A v2 Guard on its stub, as after a supervisor restart."""
    guard = _guard(_PAGE_SPEC, pool_spec=Mock())
    recovery = guard._recovery
    recovery.attachment = attachment = _fake_attachment()
    attachment._spec = recovery._pool_spec
    digest = b"d" * 32
    attachment.layout_digest.return_value = digest
    recovery.layout_digest = lambda **_kw: digest
    recovery._attach_meta = Mock()
    recovery.set_meta_fd = Mock()
    journal = Mock()
    journal.is_invalid.return_value = invalid
    journal.read_next.return_value = None
    journal.drain.return_value = iter(())
    recovery._journal = journal
    read = Mock(side_effect=AssertionError("handback read"))
    monkeypatch.setattr(guard_module, "read_handback", read)
    return guard, recovery, journal, read


def _data_pool(key_size=KEY, blocks=2):
    """A data pool of one ``""`` pool: its header page, then ``blocks`` blocks."""
    size = PAGE + blocks * PAGE
    buffer = mmap.mmap(-1, size)
    _MAPPINGS.append(buffer)
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    layouts = [("", PAGE)]
    ranges = pool_ranges(size, layouts, key_size)
    (header,) = pool_headers(
        address, ranges, layouts, key_size, initialize=True
    ).values()
    return address, size, header


@pytest.mark.parametrize("invalid", [False, True])
def test_reattach_reads_no_handback_and_keeps_the_journal(monkeypatch, invalid):
    _, recovery, journal, read = _recovery(monkeypatch, invalid=invalid)
    recovery.reattach(_tier(PAGE).pool_layouts)
    read.assert_not_called()
    # No reset and no cursor move: the live worker's journal is read from where it is.
    assert journal.mock_calls[0] == call.is_invalid()
    assert not [c for c in journal.mock_calls if c[0] in ("reset", "_consumed")]
    recovery.attachment.set_layout_digest.assert_not_called()
    if invalid:
        assert recovery.mirror is None
    else:
        assert recovery.mirror._records == {}


def _reattach(guard, recovery, monkeypatch, address, size, key_size, listener):
    monkeypatch.setattr(guard_module, "meta_spec", lambda *_args: None)
    monkeypatch.setattr(guard_module, "_KVCRCore", Mock(return_value=Mock()))
    return guard._reattach(
        Mock(incarnation="worker-1"),
        _tier(PAGE),
        listener.getsockname(),
        (address, size, recovery.layout_digest, "t/l/0", 1, 42, key_size),
        listener.fileno(),
    )


def test_reattach_leaves_the_header_and_promotion_rebuilds_it(monkeypatch):
    guard, recovery, journal, _ = _recovery(monkeypatch)
    address, size, header = _data_pool()
    committed, torn = b"c" * KEY, b"t" * KEY
    header.publish(0, committed, 5)
    flag = header._flag(1)  # a deposit in flight: odd state, key half written
    ctypes.memmove(flag, (1).to_bytes(8, "little") + torn[:10], 18)
    before = ctypes.string_at(address, PAGE)
    with (
        tempfile.TemporaryFile() as meta,
        socket.create_server(("127.0.0.1", 0)) as listener,
    ):
        meta.truncate(8192)
        recovery.attachment._file_descriptor = meta.fileno()
        recovery.attachment.snapshot_budget_bytes = 128
        _, _, granted_fd, lease = _reattach(
            guard, recovery, monkeypatch, address, size, KEY, listener
        )
        try:
            assert guard._phase is _Phase.PRIMARY
            assert ctypes.string_at(address, PAGE) == before
            assert list(recovery.headers) == [""]
            guard._promote_for(lease)
            records = guard._core.adopt_recovery_records.call_args.args[0]
            assert list(records) == [BlockKey(committed)]
            assert records[BlockKey(committed)].local_dram.slots == (("", 0),)
            assert records[BlockKey(committed)].position == 5
            assert [slot for slot, *_ in header.scan()] == [0]
            journal.reset.assert_not_called()
        finally:
            os.close(granted_fd)
            guard._control.close()
            guard._pool_lease.unbind()


@pytest.mark.parametrize("key_size", [KEY + 8, None])
def test_reattach_with_another_key_size_is_refused(monkeypatch, key_size):
    guard, recovery, _, _ = _recovery(monkeypatch)
    address, size, header = _data_pool()
    header.publish(0, b"c" * KEY, 0)
    before = ctypes.string_at(address, PAGE)
    with socket.create_server(("127.0.0.1", 0)) as listener:
        with pytest.raises(KVCRServiceReasonError) as refused:
            _reattach(guard, recovery, monkeypatch, address, size, key_size, listener)
    assert refused.value.reason == DATA_POOL_LAYOUT
    assert guard._phase is not _Phase.PRIMARY
    assert ctypes.string_at(address, PAGE) == before
