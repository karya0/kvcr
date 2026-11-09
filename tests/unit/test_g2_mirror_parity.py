# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cold recovery and promotion use the index, independently of G2 frames."""

import ctypes
import mmap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _recovered_record,
)

from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.service import recovery_journal as recovery
from kvcr.service.g2_index import G2Index
from kvcr.service.guard import _RecoveryState
from kvcr.service.memory import SOFT_KEY_MAX, index_slots
from kvcr.service.protocol import _PoolDescriptor
from kvcr.types import BlockKey, RegionDescriptor


def _index():
    data = bytearray(64 + 8 * (64 + 256))
    G2Index.initialize(data, 8, 256)
    return G2Index(data)


def _store(index, key, parts, position):
    assert index.reserve(key, parts)
    list(index.store_steps(key, parts, position))


def _projection(records):
    return {
        bytes(key): (list(record.local_dram.slots), record.position)
        for key, record in records.items()
        if record.local_dram is not None
    }


@pytest.mark.parametrize("has_g3", [False, True])
def test_cold_restart_ignores_g2_snapshot_and_preserves_mirror(monkeypatch, has_g3):
    index = _index()
    parts = [("pool:a", 1), ("pool:b", 2)]
    _store(index, b"warm" * 40, parts, 7)
    before = {b"warm" * 40: (parts, 7)}
    # Lose all writer/allocator objects, retaining only the mapped index bytes.
    cold = G2Index(bytearray(index.buffer))
    pool = Mock(g2_index=Mock(return_value=cold))
    pool._spec = SimpleNamespace()
    pools = (_PoolDescriptor("pool", 8192, 1024),)
    monkeypatch.setattr(recovery, "canonical_pool_terms", lambda *args: b"terms")
    poison = _recovered_record(g2=[("pool", 0)], g3=9, g3_pool="pool")
    frames = list(recovery._recovery_frames({BlockKey(b"poison"): poison}))
    read = (
        Mock(return_value=iter(frames)) if has_g3 else Mock(side_effect=AssertionError)
    )
    monkeypatch.setattr(recovery, "read_recovery_snapshot", read)
    pool.mapped_journal.side_effect = AssertionError("G2 read the journal")
    recovered = recovery.read_handback(pool, "digest", pools, has_g3=has_g3)
    assert _projection(recovered.take_records()) == before
    assert read.call_count == int(has_g3)
    pool.mapped_journal.assert_not_called()


def test_promotion_rescans_index_without_consulting_g2_journal():
    index = _index()
    pool = Mock(g2_index=Mock(return_value=index))
    state = _RecoveryState.__new__(_RecoveryState)
    state.attachment = pool
    state.has_g3 = False
    state.layout_digest = None
    state.pools = (_PoolDescriptor("pool", 8192, 1024),)
    state.mirror = recovery._RecoveryMirror(("pool",))
    # A later committed publication must be seen even though no journal frame arrived.
    _store(index, b"later", [("pool:a", 3)], 4)
    state._journal = Mock()
    state._journal.drain.side_effect = AssertionError("G2 replayed journal")
    records = state.take_for_promotion()
    assert _projection(records) == {b"later": ([("pool:a", 3)], 4)}
    state._journal.drain.assert_not_called()


def test_live_reattach_preserves_in_flight_store_for_later_promotion(monkeypatch):
    index = _index()
    parts = [("pool:a", 1)]
    assert index.reserve(b"key", parts)
    writer = index.store_steps(b"key", parts, 7)
    for _ in range(30):
        next(writer)
    # Attach to the same bytes while the primary is midway through publication.
    attached = G2Index(index.buffer)
    pool = Mock(g2_index=Mock(return_value=attached))
    pool._spec = SimpleNamespace()
    pools = (_PoolDescriptor("pool", 8192, 1024),)
    monkeypatch.setattr(
        recovery, "read_recovery_snapshot", Mock(side_effect=AssertionError)
    )
    before = bytes(index.buffer)
    mirror = recovery.read_handback(pool, "digest", pools, has_g3=False)
    assert _projection(mirror.take_records()) == {}
    assert bytes(index.buffer) == before
    # The primary can finish without having had its partially written payload cleared.
    list(writer)
    recovery.rebuild_g2_mirror(pool, ("pool",), mirror, promotion=True)
    assert _projection(mirror.take_records()) == {b"key": (parts, 7)}


def test_only_promotion_reclaims_an_abandoned_in_flight_slot():
    index = _index()
    parts = [("pool:a", 1)]
    assert index.reserve(b"key", parts)
    writer = index.store_steps(b"key", parts, 7)
    for _ in range(30):
        next(writer)
    pool = Mock(g2_index=Mock(return_value=G2Index(index.buffer)))
    mirror = recovery._RecoveryMirror(("pool",))
    before = bytes(index.buffer)
    recovery.rebuild_g2_mirror(pool, ("pool",), mirror)
    assert bytes(index.buffer) == before
    recovery.rebuild_g2_mirror(pool, ("pool",), mirror, promotion=True)
    assert index.buffer[64 + 64 : 64 + 2 * 64] == bytes(64)
    assert _projection(mirror.take_records()) == {}


@pytest.mark.parametrize("indexed", [False, True])
def test_g2_journal_depends_on_index_presence(monkeypatch, indexed):
    source = Mock()
    journal = Mock(_pool=None)
    # Orchestrator 12:00: retain v1 journal compatibility; suppress only indexed G2.
    monkeypatch.setattr(recovery, "_attach_g2_index", lambda *args: indexed)
    recovery._attach_journal(source, journal)
    observer = source.observe_residency.call_args.args[0]
    record = _recovered_record(g2=[("pool", 0)])
    observer(BlockKey(b"key"), record)
    source.publish_recovery.assert_called_once_with(BlockKey(b"key"), record)
    observer(BlockKey(b"key"), _recovered_record())
    assert journal.publish.call_count == (0 if indexed else 2)
    assert (
        list(recovery._recovery_frames({BlockKey(b"key"): record}, g3_only=True)) == []
    )


@pytest.mark.parametrize("key_bytes", [105, 233, 1024])
def test_empty_label_production_geometry_deposit_rebuild_parity(key_bytes):
    pool_bytes, data_slots = 53_640_953_856, 1044
    block_bytes = pool_bytes // data_slots
    # Sparse CPU backing retains production data/index geometry, not GPU transport.
    with mmap.mmap(-1, pool_bytes) as backing:
        address = ctypes.addressof(ctypes.c_char.from_buffer(backing))
        source = ctypes.create_string_buffer(block_bytes)
        agent = FakeNixlAgent()
        agent.state = "DONE"
        kvcr = _new_kvcr(
            agent,
            FakePrimaryPinning(),
            None,
            KVCRConfig(nixl_agent_name="target", pool_layouts=[("", block_bytes)]),
            local_dram=LocalDramOptions([("", address, pool_bytes)]),
            framework_regions=[
                RegionDescriptor(addr=ctypes.addressof(source), size=block_bytes)
            ],
        )
        try:
            slots = index_slots(pool_bytes)
            data = bytearray(64 + slots * (64 + SOFT_KEY_MAX))
            G2Index.initialize(data, slots, SOFT_KEY_MAX)
            index = G2Index(data)
            pools = kvcr._core._local_dram._pools
            pools._recovery_indexes = {"": index}
            pools.observe_residency(pools.publish_recovery)
            # vLLM make_offload_key: opaque hash bytes + four-byte big-endian group.
            block_hash = (bytes(range(256)) * 4)[: key_bytes - 4]
            key = BlockKey(block_hash + (7).to_bytes(4, "big"))
            op = kvcr.deposit({key: [_mem_descriptor()]})
            assert dict(_poll_until(kvcr, bool, timeout=5))[op][key].success
            before = _projection(kvcr._core._block_record_map)
            cold = G2Index(bytearray(data))
            pool = Mock(g2_index=Mock(return_value=cold))
            mirror = recovery._RecoveryMirror(("",))
            recovery.rebuild_g2_mirror(pool, ("",), mirror, promotion=True)
            records = mirror.take_records()
            assert _projection(records) == before
            assert list(records) == [key]
            assert bytes(next(iter(records)))[:-4] == block_hash
            assert int.from_bytes(next(iter(records))[-4:], "big") == 7
        finally:
            kvcr.close()
