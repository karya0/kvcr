# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""With pool headers, G2 recovery comes from them; journal and snapshot carry G3."""

import ctypes
import mmap
from unittest.mock import Mock

from _kvcr_test_utils import _recovered_record
from test_recovery_journal import _Source

from kvcr.service.g2_index import _WRITING, PoolHeader
from kvcr.service.guard import _RecoveryState
from kvcr.service.protocol import _PoolDescriptor
from kvcr.service.recovery_journal import (
    _RECORD_BLOCK,
    _RECOVERY_DECODER,
    _RECOVERY_ENCODER,
    _attach_journal,
    _project_recovery_record,
    _recovery_frames,
    _RecoveryMirror,
)
from kvcr.types import BlockKey

_MAPPINGS = []


def _headers() -> dict[str, PoolHeader]:
    """Pools a and b. A is whole at slot 2, D at slot 3; E only in a; b's slot 5
    was cut mid-write."""
    memory = mmap.mmap(-1, 2 * mmap.PAGESIZE)
    _MAPPINGS.append(memory)
    base = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    headers = {
        name: PoolHeader(base + index * mmap.PAGESIZE, 4, 4096, 8)
        for index, name in enumerate("ab")
    }
    for header in headers.values():
        header.verify(initialize=True)
        header.publish(2, b"AAAA", 6)
        header.publish(3, b"DDDD", 7)
    headers["a"].publish(4, b"EEEE", 0)
    headers["b"]._store(headers["b"]._flag(5), _WRITING, 3)
    return headers


def _frame(key: bytes, **fields):
    record = _recovered_record(**fields)
    return (
        _RECORD_BLOCK,
        key,
        _RECOVERY_ENCODER.encode(_project_recovery_record(record)),
    )


def _expect_g2_from_headers(records, headers):
    assert {
        key: record.local_dram and record.local_dram.slots
        for key, record in records.items()
    } == {
        b"AAAA": (("a", 2), ("b", 2)),
        b"CCCC": None,  # its journal G2 is not consulted
        b"DDDD": (("a", 3), ("b", 3)),
    }
    assert records[BlockKey(b"AAAA")].g3.slots == (9,)
    assert records[BlockKey(b"DDDD")].position == 7
    for header in headers.values():  # the partial and the torn entry are cleared
        assert [slot for slot, _, _ in header.scan()] == [2, 3]


def test_promotion_rebuilds_g2_from_the_headers_and_g3_from_the_journal():
    headers = _headers()
    recovery = object.__new__(_RecoveryState)
    recovery.pools = (
        _PoolDescriptor("a", 4096, 4096),
        _PoolDescriptor("b", 4096, 4096),
    )
    recovery.attachment = Mock()
    recovery.layout_digest = None
    recovery.headers = headers
    recovery.mirror = _RecoveryMirror(("a", "b"))
    # What a Guard's own handback left: stale G2 slots, beside G3.
    recovery.mirror.adopt(
        {
            BlockKey(b"AAAA"): _recovered_record(
                g2=[("a", 0), ("b", 0)], g3=9, g3_pool="a"
            ),
            BlockKey(b"BBBB"): _recovered_record(g2=[("a", 1), ("b", 1)]),
        }
    )
    recovery._journal = Mock()
    recovery._journal.drain.return_value = [
        _frame(b"CCCC", g2=[("a", 7), ("b", 7)], g3=1, g3_pool="a")
    ]
    _expect_g2_from_headers(recovery.take_for_promotion(), headers)


def test_with_headers_the_journal_and_the_handback_carry_no_g2():
    local_dram, g3 = _Source(), _Source()
    local_dram._headers = {"a": object()}
    journal = Mock()
    _attach_journal(local_dram, journal, g3)
    local_dram.emit(BlockKey(b"AAAA"), _recovered_record(g2=[("a", 2)]))
    assert not journal.publish.called  # G2 alone: nothing for the journal
    both = _recovered_record(g2=[("a", 2)], g3=9, g3_pool="a")
    local_dram.emit(BlockKey(b"AAAA"), both)
    g3.emit(BlockKey(b"AAAA"), both)
    for call in journal.publish.call_args_list:
        assert _RECOVERY_DECODER.decode(call.args[2]).g2 is None

    records = {
        BlockKey(b"AAAA"): both,
        BlockKey(b"BBBB"): _recovered_record(g2=[("a", 3)]),
    }
    frames = list(_recovery_frames(records, g3_only=True))
    assert [key for _, key, _ in frames] == [b"AAAA"]
    assert _RECOVERY_DECODER.decode(frames[0][2]).g2 is None
