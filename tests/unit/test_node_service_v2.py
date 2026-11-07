# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""v2 claim path: slots, capacity, SLOT_HELD, holder recreate, resize (Linux only)."""

import mmap
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from test_kvcr_service import _running_server
from test_node_service import _SPECS, FakeAPI, make_broker

from kvcr.runtime.control_channels import FramedConnection
from kvcr.service import node_service
from kvcr.service.node_service import SlotError, SlotTable, effective_grant
from kvcr.service.protocol import (
    _RESIZE_RESPONSE_DECODER,
    CAPACITY_BELOW_MINIMUM,
    IDENTITY_MISMATCH,
    INTERNAL,
    REGION_BUSY,
    SLOT_HELD,
    UNKNOWN_SLOT,
    _Error,
    _Grow,
    _Grown,
    _Shrink,
    _Shrunk,
)
from kvcr.service.regions import RegionPool

PAGE = mmap.PAGESIZE
HEADROOM = node_service._HEADROOM_BYTES
DIGEST = "model-v1"
ME = (os.getpid(), os.getuid())


@pytest.fixture
def tmp_path():
    # Short path: holder socket paths are bounded by AF_UNIX.
    with tempfile.TemporaryDirectory(prefix="gaas-v2-", dir="/tmp") as directory:
        yield Path(directory)


@pytest.mark.parametrize(
    ("total", "minimum", "maximum", "available", "expected"),
    [
        (8, 4, None, 100, 8),  # full
        (8, 4, 6, 100, 6),  # maximum caps the grant
        (8, 4, None, 5, 5),  # partial
        (8, 4, None, 4, 4),  # exactly the minimum
        (8, 4, None, 3, None),  # below minimum
    ],
)
def test_effective_grant(total, minimum, maximum, available, expected):
    assert effective_grant(total, minimum, maximum, available) == expected


@pytest.fixture
def pools(monkeypatch):
    created = []
    real = RegionPool.create

    def create(*args, **kwargs):
        created.append(real(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(RegionPool, "create", create)
    yield created
    for pool in created:
        pool.close()


def _table(tmp_path, api=None, pages=64, **kwargs):
    api = api or FakeAPI()
    broker = make_broker(tmp_path, api, **kwargs)
    return SlotTable(broker, HEADROOM + pages * PAGE), api


def _claim(table, claim_id="a", total=4, minimum=2, maximum=None, alive=lambda: True):
    return table.claim(
        "t",
        "l",
        total=total * PAGE,
        minimum=minimum * PAGE,
        maximum=maximum and maximum * PAGE,
        digest=DIGEST,
        claim_id=claim_id,
        peer=ME,
        owner_alive=alive,
    )


def test_full_grant(tmp_path, pools):
    table, api = _table(tmp_path)
    slot = _claim(table)
    assert slot.name == "t/l" and slot.effective_maximum_bytes == 4 * PAGE
    assert [r.size for r in slot.pool.regions] == [4 * PAGE]
    assert len(api.creates) == 1


def test_namespace_local_requests_extend_capacity(tmp_path, pools):
    table, _ = _table(tmp_path, pages=1)
    table.local_requests = lambda tenant: 8 * PAGE if tenant == "t" else 0
    assert _claim(table).effective_maximum_bytes == 4 * PAGE


def test_partial_grant(tmp_path, pools):
    table, _ = _table(tmp_path, pages=3)
    slot = _claim(table)
    assert slot.effective_maximum_bytes == 3 * PAGE
    assert [r.size for r in slot.pool.regions] == [3 * PAGE]


def test_other_slots_reduce_capacity(tmp_path, pools):
    table, _ = _table(tmp_path, pages=5)
    _claim(table)
    with pytest.raises(SlotError) as raised:
        table.claim(
            "u",
            "l",
            total=4 * PAGE,
            minimum=2 * PAGE,
            maximum=None,
            digest=DIGEST,
            claim_id="x",
            peer=ME,
            owner_alive=lambda: True,
        )
    assert raised.value.reason == CAPACITY_BELOW_MINIMUM


def test_below_minimum_leaves_nothing(tmp_path, pools):
    table, api = _table(tmp_path, pages=1)
    with pytest.raises(SlotError) as raised:
        _claim(table)
    assert raised.value.reason == CAPACITY_BELOW_MINIMUM
    assert not api.creates and not api.deletes and table.get("t/l") is None
    assert all(pool.fd == -1 for pool in pools)


def test_failed_holder_closes_pool_and_deletes_holder(tmp_path, pools):
    # A holder that never reports ready: the claim fails and nothing is retained.
    info = lambda _: {  # noqa: E731
        "state": "starting",
        "pool_specs": _SPECS,
        "compatibility_digest": DIGEST,
    }
    table, api = _table(tmp_path, info=info)
    table.ensure_timeout_s = 0.01
    with pytest.raises(SlotError) as raised:
        _claim(table)
    assert raised.value.reason == INTERNAL
    assert len(pools) == 1 and pools[0].fd == -1
    assert api.deletes == [(api.creates[0]["metadata"]["name"], "holder-uid-1")]
    assert table.get("t/l") is None


def test_slot_held_queue_and_release(tmp_path, pools):
    table, api = _table(tmp_path)
    alive = [True]
    _claim(table, "a", alive=lambda: alive[0])
    for waiter in ("b", "c", "b"):
        with pytest.raises(SlotError) as raised:
            _claim(table, waiter)
        assert raised.value.reason == SLOT_HELD
    assert table.waiters("t/l") == ["b", "c"]
    alive[0] = False  # the worker died: the slot is free, b is next
    with pytest.raises(SlotError):
        _claim(table, "c")
    slot = _claim(table, "b")
    assert table.waiters("t/l") == ["c"]
    assert slot.pool.fd >= 0 and len(api.creates) == 1  # pool and holder reused
    assert table.release("t/l") == ["c"]
    assert api.deletes and slot.pool.fd == -1
    assert _claim(table, "c").claim_id == "c"
    assert len(api.creates) == 2


def test_holder_death_recreates_once(tmp_path, pools):
    table, api = _table(tmp_path)
    slot = _claim(table)
    assert table.check_holder("t/l") is False
    del api.pods[api.creates[0]["metadata"]["name"]]
    assert table.check_holder("t/l") is True
    assert slot.incarnation == 1 and len(api.creates) == 2
    assert api.creates[1]["metadata"]["name"] != api.creates[0]["metadata"]["name"]
    assert table.check_holder("t/l") is False and len(api.creates) == 2


def test_terminating_holder_recreates(tmp_path, pools):
    table, api = _table(tmp_path)
    slot = _claim(table)
    api.pods[api.creates[0]["metadata"]["name"]]["metadata"]["deletionTimestamp"] = "t"
    assert table.check_holder("t/l") is True and slot.incarnation == 1


def test_resize_round_trip_and_identity(tmp_path, pools):
    table, _ = _table(tmp_path)
    _claim(table, total=2, minimum=2, maximum=8)
    with _running_server(tmp_path) as harness:
        harness.server._server.slots = table

        def send(request):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(harness.server.socket_path))
                channel = FramedConnection(connection)
                channel.send(request)
                return channel.receive(_RESIZE_RESPONSE_DECODER)

        grown = send(_Grow(1, "t/l", 2 * PAGE))
        assert isinstance(grown, _Grown) and grown.layout_generation == 2
        assert [r.index for r in grown.regions] == [1]
        assert send(_Grow(1, "t/l", 64 * PAGE)).reason == CAPACITY_BELOW_MINIMUM
        assert send(_Grow(1, "x/y", PAGE)).reason == UNKNOWN_SLOT
        assert send(_Shrink(1, "t/l", (0,))).reason == REGION_BUSY  # not a suffix
        assert send(_Shrink(1, "t/l", (7,))).reason == UNKNOWN_SLOT
        shrunk = send(_Shrink(1, "t/l", (1,)))
        assert isinstance(shrunk, _Shrunk) and shrunk.region_indices == (1,)
        assert shrunk.layout_generation == 3
        table.get("t/l").peer = (ME[0] + 1, ME[1])  # another peer owns the slot
        for request in (_Grow(1, "t/l", PAGE), _Shrink(1, "t/l", (0,))):
            error = send(request)
            assert isinstance(error, _Error) and error.reason == IDENTITY_MISMATCH
