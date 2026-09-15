# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Higher-tier accesses refresh managed copies without loading or claiming them."""

import ctypes
import os

import pytest
from _kvcr_test_utils import (
    FakeNixlAgent,
    _mem_descriptor,
    _new_local_kvcr,
    _poll_until,
)
from test_g3 import _deposit as _g3_deposit
from test_g3 import _new_g3_kvcr

from kvcr.policy import FIFOPolicy, LRUPolicy
from kvcr.types import BlockKey, CacheTier, QueryStatus


class _RecordingLRU(LRUPolicy):
    def __init__(self):
        self.scored = []

    def eviction_score(self, meta, source):
        self.scored.append(meta)
        return super().eviction_score(meta, source)


class _ConstantPolicy(LRUPolicy):
    def eviction_score(self, meta, source):
        return 7.0


@pytest.fixture
def pool(monkeypatch):
    buffers = []
    instances = []

    def create(slots=2, policy=None):
        source = ctypes.create_string_buffer(b"x" * 16, 16)
        local = ctypes.create_string_buffer(slots * 16)
        buffers.extend((source, local))
        agent = FakeNixlAgent()
        agent.state = "DONE"
        events = []
        kvcr = _new_local_kvcr(agent, local, slots, events.append, policy=policy)
        instances.append(kvcr)
        clock = [0.0]
        monkeypatch.setattr(kvcr._core, "_clock", lambda: clock[0])

        def deposit(key):
            operation = kvcr.deposit({key: [_mem_descriptor(ctypes.addressof(source))]})
            result = dict(_poll_until(kvcr, bool))[operation][key]
            assert result.success

        return kvcr, clock, agent, events, deposit

    yield create
    for instance in instances:
        instance.close()


@pytest.mark.parametrize(
    "policy",
    [None, FIFOPolicy(), _ConstantPolicy()],
    ids=["default-lru", "fifo", "constant-custom"],
)
def test_touch_refreshes_lru_without_changing_fifo_or_copying(pool, policy):
    kvcr, clock, agent, events, deposit = pool(policy=policy)
    root, suffix, new = map(BlockKey, (b"root", b"suffix", b"new"))
    deposit(root)
    clock[0] = 1.0
    deposit(suffix)
    copies = len(agent.transfers)
    events.clear()
    clock[0] = 2.0
    kvcr.touch((root,))
    assert len(agent.transfers) == copies
    assert events == []
    assert list(kvcr.poll_completed()) == []
    clock[0] = 3.0
    deposit(new)
    expected = [QueryStatus.HIT, QueryStatus.MISS, QueryStatus.HIT]
    if policy is not None:
        expected = [QueryStatus.MISS, QueryStatus.HIT, QueryStatus.HIT]
    assert [status for status, _ in kvcr.query((root, suffix, new))] == expected


def test_touch_ignores_missing_and_filling_keys_and_deduplicates_ready_keys(pool):
    policy = _RecordingLRU()
    kvcr, clock, agent, _, deposit = pool(policy=policy)
    ready, filling, absent = map(BlockKey, (b"ready", b"filling", b"absent"))
    deposit(ready)
    source = ctypes.create_string_buffer(b"y" * 16, 16)
    agent.state = "PROC"
    operation = kvcr.deposit({filling: [_mem_descriptor(ctypes.addressof(source))]})
    policy.scored.clear()
    clock[0] = 0.5
    kvcr.touch((ready, absent, filling, ready))
    assert [
        (meta.block_key, meta.access_count, meta.last_access) for meta in policy.scored
    ] == [(ready, 1, 0.5)]
    assert kvcr.query((filling, absent)) == [
        (QueryStatus.FETCHING, CacheTier.LOCAL_G2),
        (QueryStatus.MISS, None),
    ]
    agent.state = "DONE"
    assert dict(_poll_until(kvcr, bool))[operation][filling].success


def test_touch_of_claimed_block_keeps_it_claimed_until_release(pool):
    kvcr, clock, _, _, deposit = pool()
    root, suffix, new, extra = map(BlockKey, (b"root", b"suffix", b"new", b"extra"))
    deposit(root)
    clock[0] = 1.0
    deposit(suffix)
    operation = kvcr.fetch((root,))
    root_claim = dict(_poll_until(kvcr, bool))[operation][root].release_handle
    assert root_claim is not None
    clock[0] = 2.0
    kvcr.touch((root,))
    clock[0] = 3.0
    deposit(new)
    assert kvcr.query((root, suffix, new)) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    # Root is now oldest, but must remain protected by the outstanding claim.
    clock[0] = 4.0
    deposit(extra)
    assert kvcr.query((root, new, extra)) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert kvcr.release((root_claim,)) == [(root_claim, True)]
    # Touching while claimed must not double-insert it when the claim releases.
    clock[0] = 5.0
    deposit(suffix)
    assert [status for status, _ in kvcr.query((root, extra, suffix))] == [
        QueryStatus.MISS,
        QueryStatus.HIT,
        QueryStatus.HIT,
    ]


def test_repeated_touch_keeps_eviction_accounting_and_heap_bounded(pool):
    kvcr, clock, _, _, deposit = pool()
    root, suffix = map(BlockKey, (b"root", b"suffix"))
    deposit(root)
    deposit(suffix)
    local = kvcr._core._local_dram
    initial = local.telemetry_state()
    for tick in range(1, 101):
        clock[0] = float(tick)
        kvcr.touch((root,))
        assert local.telemetry_state() == initial
        assert len(local._evictable) == 2
        assert len(local._evictable._heap) <= 2 * len(local._evictable)


@pytest.mark.parametrize("failure", ["raise", "nan"])
def test_invalid_touch_score_preserves_previous_eviction_candidate(pool, failure):
    class FailingPolicy(LRUPolicy):
        fail = False

        def eviction_score(self, meta, source):
            if self.fail:
                if failure == "raise":
                    raise RuntimeError("test score failure")
                return float("nan")
            return super().eviction_score(meta, source)

    policy = FailingPolicy()
    kvcr, clock, _, _, deposit = pool(policy=policy)
    root, suffix, new = map(BlockKey, (b"root", b"suffix", b"new"))
    deposit(root)
    clock[0] = 1.0
    deposit(suffix)
    before = kvcr._core._local_dram.telemetry_state()
    policy.fail = True
    clock[0] = 2.0
    kvcr.touch((root,))
    assert kvcr._core._local_dram.telemetry_state() == before
    policy.fail = False
    clock[0] = 3.0
    deposit(new)
    assert [status for status, _ in kvcr.query((root, suffix, new))] == [
        QueryStatus.MISS,
        QueryStatus.HIT,
        QueryStatus.HIT,
    ]


def test_touch_refreshes_g3_copy_without_loading_it(tmp_path, monkeypatch):
    page = os.sysconf("SC_PAGE_SIZE")
    source = ctypes.create_string_buffer(page)
    local = ctypes.create_string_buffer(page)
    kvcr = _new_g3_kvcr(tmp_path, local, g3_slot_count=2)
    clock = [0.0]
    monkeypatch.setattr(kvcr._core, "_clock", lambda: clock[0])
    keys = tuple(BlockKey(f"key{i}".encode()) for i in range(4))
    try:
        for index, key in enumerate(keys[:3]):
            clock[0] = float(index)
            assert _g3_deposit(kvcr, key, ctypes.addressof(source), page).success
        assert kvcr.query(keys[:2]) == [(QueryStatus.FETCHABLE, CacheTier.G3)] * 2
        clock[0] = 3.0
        kvcr.touch((keys[0],))
        assert kvcr.query((keys[0],)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
        clock[0] = 4.0
        assert _g3_deposit(kvcr, keys[3], ctypes.addressof(source), page).success
        assert kvcr.query(keys) == [
            (QueryStatus.FETCHABLE, CacheTier.G3),
            (QueryStatus.MISS, None),
            (QueryStatus.FETCHABLE, CacheTier.G3),
            (QueryStatus.HIT, CacheTier.LOCAL_G2),
        ]
    finally:
        kvcr.close()
