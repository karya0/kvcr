# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""v2 claim path: slots, capacity, SLOT_HELD, holder recreate, resize (Linux only)."""

import copy
import errno
import json
import mmap
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from test_kvcr_service import _running_server
from test_node_service import _CLAIM, _SPECS, FakeAPI, make_broker

from kvcr.runtime.control_channels import FramedConnection
from kvcr.service import holder, node_service, server
from kvcr.service.node_service import SlotError, SlotTable
from kvcr.service.protocol import (
    _RESIZE_RESPONSE_DECODER,
    CAPACITY_BELOW_MINIMUM,
    IDENTITY_MISMATCH,
    INCOMPATIBLE_LINEAGE,
    INTERNAL,
    REGION_BUSY,
    SLOT_HELD,
    UNKNOWN_SLOT,
    _ErrorV2,
    _Grow,
    _Grown,
    _Shrink,
    _Shrunk,
)
from kvcr.service.regions import RegionPool

PAGE = mmap.PAGESIZE
DIGEST = "model-v1"
ME = (os.getpid(), os.getuid())


@pytest.fixture
def tmp_path():
    # Short path: holder socket paths are bounded by AF_UNIX.
    with tempfile.TemporaryDirectory(prefix="gaas-v2-", dir="/tmp") as directory:
        yield Path(directory)


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


def _table(tmp_path, api=None, **kwargs):
    api = api or FakeAPI()
    broker = make_broker(tmp_path, api, **kwargs)
    return SlotTable(broker, require_pod_identity=False), api


def _claim(
    table,
    claim_id="a",
    total=4,
    minimum=2,
    maximum=None,
    alive=lambda: True,
    waiter_alive=None,
):
    return table.claim(
        "t",
        "l",
        0,
        total=total * PAGE,
        minimum=minimum * PAGE,
        maximum=maximum and maximum * PAGE,
        digest=DIGEST,
        claim_id=claim_id,
        peer=ME,
        owner_alive=alive,
        waiter_alive=waiter_alive,
    )


def test_full_grant(tmp_path, pools):
    table, api = _table(tmp_path)
    slot = _claim(table)
    assert slot.name == "t/l/0" and slot.effective_maximum_bytes == 4 * PAGE
    assert [r.size for r in slot.pool.regions] == [4 * PAGE]
    assert len(api.creates) == 1


def test_grant_is_the_claimed_maximum_with_no_node_cap(tmp_path, pools):
    table, _ = _table(tmp_path)
    pages = (600 << 30) // PAGE  # 600 GiB: reserved address space, sparse memfd
    slot = _claim(table, total=4, minimum=2, maximum=pages)
    assert slot.effective_maximum_bytes == pages * PAGE
    assert [r.size for r in slot.pool.regions] == [4 * PAGE]  # sparse memfd


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
    assert table.get("t/l/0") is None


def test_slot_held_queue_and_release(tmp_path, pools):
    table, api = _table(tmp_path)
    alive = [True]
    _claim(table, "a", alive=lambda: alive[0])
    for waiter in ("b", "c", "b"):
        with pytest.raises(SlotError) as raised:
            _claim(table, waiter)
        assert raised.value.reason == SLOT_HELD
    assert table.waiters("t/l/0") == ["b", "c"]
    alive[0] = False  # the worker died: the slot is free, b is next
    with pytest.raises(SlotError):
        _claim(table, "c")
    slot = _claim(table, "b")
    assert table.waiters("t/l/0") == ["c"]
    assert slot.pool.fd >= 0 and len(api.creates) == 1  # pool and holder reused
    assert table.release("t/l/0") == ["c"]
    assert api.deletes and slot.pool.fd == -1
    assert _claim(table, "c").claim_id == "c"
    assert len(api.creates) == 2


def _queue_waiter(table, claim_id, fds):
    def open_fd():
        fds.append(os.dup(0))
        return fds[-1]

    with pytest.raises(SlotError) as raised:
        _claim_raw(table, claim_id=claim_id, waiter_fd=open_fd)
    assert raised.value.reason == SLOT_HELD


def _assert_closed(fds):
    assert fds
    for fd in fds:
        with pytest.raises(OSError) as raised:
            os.fstat(fd)
        assert raised.value.errno == errno.EBADF


def test_release_drops_queued_waiters(tmp_path, pools):
    table, _ = _table(tmp_path)
    _claim_raw(table)
    fds: list[int] = []
    _queue_waiter(table, "b", fds)
    assert table.release("t/l/0") == ["b"]
    _assert_closed(fds)
    assert table.waiters("t/l/0") == []
    assert _claim_raw(table, claim_id="d").claim_id == "d"  # no SLOT_HELD


def test_abort_claim_drops_queued_waiters(tmp_path, pools):
    table, _ = _table(tmp_path)
    slot, generation = table.claim_with_generation(
        "t",
        "l",
        0,
        total=4 * PAGE,
        minimum=2 * PAGE,
        maximum=None,
        digest=DIGEST,
        claim_id="a",
        peer=ME,
        owner_alive=lambda: True,
    )
    fds: list[int] = []
    _queue_waiter(table, "b", fds)
    table._abort_claim(slot, True, generation)
    _assert_closed(fds)
    assert table.waiters("t/l/0") == []
    assert _claim_raw(table, claim_id="d").claim_id == "d"


def test_holder_death_recreates_once(tmp_path, pools):
    table, api = _table(tmp_path)
    slot = _claim(table)
    assert table.check_holder("t/l/0") is False
    del api.pods[api.creates[0]["metadata"]["name"]]
    assert table.check_holder("t/l/0") is True
    assert slot.incarnation == 1 and len(api.creates) == 2
    assert api.creates[1]["metadata"]["name"] != api.creates[0]["metadata"]["name"]
    assert table.check_holder("t/l/0") is False and len(api.creates) == 2


def test_terminating_holder_recreates(tmp_path, pools):
    table, api = _table(tmp_path)
    slot = _claim(table)
    api.pods[api.creates[0]["metadata"]["name"]]["metadata"]["deletionTimestamp"] = "t"
    assert table.check_holder("t/l/0") is True and slot.incarnation == 1


def test_resize_round_trip_and_identity(tmp_path, pools):
    table, _ = _table(tmp_path)
    gen = _claim(table, total=2, minimum=2, maximum=8).claim_generation
    with _running_server(tmp_path) as harness:
        harness.server._server.slots = table

        def send(request):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(harness.server.socket_path))
                channel = FramedConnection(connection)
                channel.send(request)
                return channel.receive(_RESIZE_RESPONSE_DECODER)

        grown = send(_Grow(1, "t/l/0", 2 * PAGE, gen))
        assert isinstance(grown, _Grown) and grown.layout_generation == 2
        assert [r.index for r in grown.regions] == [1]
        assert send(_Grow(1, "t/l/0", 64 * PAGE, gen)).reason == CAPACITY_BELOW_MINIMUM
        assert send(_Grow(1, "x/y/0", PAGE, gen)).reason == UNKNOWN_SLOT
        assert (
            send(_Shrink(1, "t/l/0", (0,), gen)).reason == REGION_BUSY
        )  # not a suffix
        assert send(_Shrink(1, "t/l/0", (7,), gen)).reason == UNKNOWN_SLOT
        shrunk = send(_Shrink(1, "t/l/0", (1,), gen))
        assert isinstance(shrunk, _Shrunk) and shrunk.region_indices == (1,)
        assert shrunk.layout_generation == 3
        stale = send(_Grow(1, "t/l/0", PAGE, gen + 1))  # not the granted claim
        assert isinstance(stale, _ErrorV2) and stale.reason == IDENTITY_MISMATCH
        table.get("t/l/0").owner_alive = None  # released: no resize either
        assert send(_Shrink(1, "t/l/0", (0,), gen)).reason == IDENTITY_MISMATCH
        table.get("t/l/0").owner_alive = lambda: True
        table.get("t/l/0").peer = (ME[0] + 1, ME[1])  # another peer owns the slot
        for request in (_Grow(1, "t/l/0", PAGE, gen), _Shrink(1, "t/l/0", (0,), gen)):
            error = send(request)
            assert isinstance(error, _ErrorV2) and error.reason == IDENTITY_MISMATCH


def _claim_raw(table, *, peer=ME, claim_id="a", alive=lambda: True, **kwargs):
    return table.claim(
        "t",
        "l",
        0,
        total=4 * PAGE,
        minimum=2 * PAGE,
        maximum=8 * PAGE,
        digest=DIGEST,
        claim_id=claim_id,
        peer=peer,
        owner_alive=alive,
        **kwargs,
    )


def test_slot_key_includes_the_worker_position(tmp_path, pools):
    table, _ = _table(tmp_path)
    first = _claim_raw(table)
    second = table.claim(
        "t",
        "l",
        1,
        total=2 * PAGE,
        minimum=2 * PAGE,
        maximum=None,
        digest=DIGEST,
        claim_id="a",
        peer=ME,
        owner_alive=lambda: True,
    )
    assert (first.name, second.name) == ("t/l/0", "t/l/1")


def test_v2_fails_closed_without_pod_identity(tmp_path, pools):
    broker = make_broker(tmp_path, FakeAPI())
    table = SlotTable(broker)  # require_pod_identity=True
    for peer in (None, (ME[0], ME[1], None), (0, ME[1], "pod-a"), ME):
        with pytest.raises(SlotError) as raised:
            _claim_raw(table, peer=peer)
        assert raised.value.reason == IDENTITY_MISMATCH
    assert table.get("t/l/0") is None
    for call in (
        lambda: table.grow("t/l/0", (0, 1, "p"), PAGE, None),
        lambda: table.shrink("t/l/0", (1, 1, None), (0,)),
        lambda: table.vacate("t/l/0", (0, 1, "p"), 1),
    ):
        with pytest.raises(SlotError) as raised:
            call()
        assert raised.value.reason == IDENTITY_MISMATCH


class _LineageAPI(FakeAPI):
    def __init__(self, workers):
        super().__init__()
        self.workers = workers

    def get_by_uid(self, uid):
        return copy.deepcopy(self.workers.get(uid))


def _worker(lineage, namespace="ns"):
    labels = {
        node_service._LINEAGE_LABEL: lineage,
        node_service._DYNAMO_NAMESPACE_LABEL: namespace,
    }
    return {"metadata": {"uid": "u1", "labels": labels}}


@pytest.mark.parametrize(
    ("workers", "namespace", "expected"),
    [
        ({"u1": _worker("l")}, "ns", None),
        ({"u1": _worker("l")}, None, None),
        ({"u1": _worker("other")}, "ns", INCOMPATIBLE_LINEAGE),
        ({"u1": _worker("l", "elsewhere")}, "ns", INCOMPATIBLE_LINEAGE),
        ({"u1": {"metadata": {"uid": "u1"}}}, None, INCOMPATIBLE_LINEAGE),
        ({}, "ns", INCOMPATIBLE_LINEAGE),
    ],
)
def test_claim_proves_lineage_membership(tmp_path, pools, workers, namespace, expected):
    table = SlotTable(make_broker(tmp_path, _LineageAPI(workers)))
    peer = (ME[0], ME[1], "u1")
    if expected is None:
        assert _claim_raw(table, peer=peer, namespace=namespace).peer == peer
        return
    with pytest.raises(SlotError) as raised:
        _claim_raw(table, peer=peer, namespace=namespace)
    assert raised.value.reason == expected
    assert table.get("t/l/0") is None


def test_stale_release_does_not_free_the_new_owner(tmp_path, pools):
    table, _ = _table(tmp_path)
    alive = {"a": True, "b": True}
    first = _claim_raw(table, claim_id="a", alive=lambda: alive["a"])
    generation_a = first.claim_generation
    alive["a"] = False  # A died; its release is still in flight
    second = _claim_raw(table, claim_id="b", alive=lambda: alive["b"])
    assert second.claim_generation == generation_a + 1
    table.vacate("t/l/0", ME, generation_a)  # same peer, older claim: no-op
    assert second.held and second.claim_id == "b"
    table.vacate("t/l/0", ME, second.claim_generation)
    assert not second.held


def test_failed_holder_ack_rolls_the_grow_back(tmp_path, pools):
    table, api = _table(tmp_path)
    slot = _claim(table, total=2, minimum=2, maximum=8)
    before = (slot.pool.regions, slot.pool.layout_generation)
    supervisor_end, holder_end = socket.socketpair()
    holder_end.close()  # the holder is gone: EOF/EPIPE on the layout
    slot.holder_conn = supervisor_end
    for pod in api.pods.values():
        pod["status"]["containerStatuses"] = [
            {"lastState": {"terminated": {"reason": "OOMKilled"}}}
        ]
    with pytest.raises(SlotError) as raised:
        table.grow("t/l/0", ME, 2 * PAGE, None)
    assert raised.value.reason == INTERNAL
    assert str(raised.value).endswith("(OOMKilled)")
    assert (slot.pool.regions, slot.pool.layout_generation) == before
    assert slot.holder_conn is None
    regions, generation = table.grow("t/l/0", ME, 2 * PAGE, None)  # retry grows once
    assert [r.index for r in regions] == [1] and generation == before[1] + 1
    assert len(slot.pool.regions) == 2


def test_slow_exit_reason_lookup_is_bounded(tmp_path, pools, monkeypatch):
    table, api = _table(tmp_path)
    slot = _claim(table, total=2, minimum=2, maximum=8)
    monkeypatch.setattr(node_service, "_EXIT_REASON_DEADLINE_S", 0.2)
    real_get = api.get
    api.get = lambda name: (time.sleep(2), real_get(name))[1]
    supervisor_end, holder_end = socket.socketpair()
    holder_end.close()
    slot.holder_conn = supervisor_end
    start = time.monotonic()
    with pytest.raises(SlotError) as raised:
        table.grow("t/l/0", ME, 2 * PAGE, None)
    assert time.monotonic() - start < 1.5
    assert str(raised.value).endswith("holder closed before reporting")


def test_shrink_fallocate_failure_is_internal(tmp_path, pools, monkeypatch):
    table, _ = _table(tmp_path)
    slot = _claim(table, total=2, minimum=2, maximum=8)
    table.grow("t/l/0", ME, 2 * PAGE, None)
    layout = (slot.pool.regions, slot.pool.layout_generation)
    monkeypatch.setattr(slot.pool._libc, "fallocate", lambda *args: -1)
    with pytest.raises(SlotError) as raised:
        table.shrink("t/l/0", ME, (1,))
    assert raised.value.reason == INTERNAL
    assert (slot.pool.regions, slot.pool.layout_generation) == layout


def test_shrink_resends_the_reduced_layout(tmp_path, pools):
    table, _ = _table(tmp_path)
    slot = _claim(table, total=2, minimum=2, maximum=8)
    table.grow("t/l/0", ME, 2 * PAGE, None)
    supervisor_end, holder_end = socket.socketpair()
    slot.holder_conn = supervisor_end
    mappings = {}
    thread = threading.Thread(target=holder.handle_layout, args=(holder_end, mappings))
    thread.start()
    try:
        _, generation = table.shrink("t/l/0", ME, (1,))
        thread.join(10)
        assert slot.touched_bytes == 2 * PAGE
        assert slot.layout_generation == generation
        assert list(mappings) == [0, -1]  # the meta memfd stays mapped
    finally:
        for view, _size in mappings.values():
            view.close()
        supervisor_end.close()
        holder_end.close()


def test_dead_queue_head_is_dropped(tmp_path, pools):
    table, _ = _table(tmp_path)
    owner, head = [True], [True]
    _claim(table, "a", alive=lambda: owner[0])
    with pytest.raises(SlotError):
        _claim(table, "b", waiter_alive=lambda: head[0])
    with pytest.raises(SlotError):
        _claim(table, "c")
    owner[0] = head[0] = False  # the owner and the head both died
    assert _claim(table, "c").claim_id == "c"
    assert table.waiters("t/l/0") == []


def test_silent_queue_head_is_dropped_after_twice_the_ensure_timeout(tmp_path, pools):
    table, _ = _table(tmp_path)
    table.ensure_timeout_s = 10
    now = [0.0]
    table._clock = lambda: now[0]
    owner = [True]
    _claim(table, "a", alive=lambda: owner[0])
    for waiter in ("d", "e"):  # d heads the queue and never comes back
        with pytest.raises(SlotError):
            _claim(table, waiter)
    owner[0] = False
    now[0] = 5.0
    with pytest.raises(SlotError):  # d is within its window and is next
        _claim(table, "e")
    now[0] = 21.0  # d silent for 21 s > 2 x 10 s; e re-claimed at 5 s
    assert _claim(table, "e").claim_id == "e"


def test_slot_state_persists_across_supervisor_restart(tmp_path, pools):
    import uuid

    table, api = _table(tmp_path)
    slot = _claim(table, uuid.uuid4().hex, total=2, minimum=2, maximum=8)
    table.grow("t/l/0", ME, 2 * PAGE, None)
    restarted = SlotTable(table.broker, require_pod_identity=False)
    again = restarted.get("t/l/0")
    pools.append(again.pool)  # closed by the fixture
    assert again.pool.regions == slot.pool.regions
    assert again.pool.layout_generation == slot.pool.layout_generation
    assert (again.holder, again.digest) == (slot.holder, slot.digest)
    assert again.claim_generation == slot.claim_generation and again.held
    assert restarted._epochs == table._epochs == {"t/l/0": 0}
    # The live holder is kept and re-claimed, not deleted and recreated.
    creates = len(api.creates)
    now = [0.0]
    restarted._clock = lambda: now[0]
    restarted.start_quarantine()
    with pytest.raises(SlotError) as held:
        _claim(restarted, "b", total=2, minimum=2, maximum=8)
    assert held.value.reason == SLOT_HELD
    now[0] = restarted.lease_ttl_ms / 1000
    reclaimed = _claim(restarted, "b", total=2, minimum=2, maximum=8)
    assert (
        reclaimed is again and reclaimed.claim_generation == slot.claim_generation + 1
    )
    assert len(api.creates) == creates and not api.deletes
    # Epochs continue instead of restarting at 0.
    del api.pods[api.creates[0]["metadata"]["name"]]
    assert restarted.check_holder("t/l/0") is True
    assert restarted._epochs == {"t/l/0": 1}
    persisted = json.loads((tmp_path / "slot-table.json").read_text())
    assert persisted["epochs"] == {"t/l/0": 1}


def test_a_terminating_holder_counts_against_nothing_but_its_replacement(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    claims = [{**_CLAIM, "pool_id": f"pool-{name}"} for name in "abcd"]
    held = broker.ensure(claims[0])["holder_name"]
    terminating = copy.deepcopy(api.pods[held])
    assert broker.release(claims[0])["state"] == "deleted" and not api.pods
    terminating["metadata"]["deletionTimestamp"] = "now"
    api.pods[held] = terminating  # gone from the claim's view, not from the API
    broker.ensure(claims[1])
    broker.ensure(claims[2])
    api.pods[held]["metadata"]["uid"] = "replacement"  # same name, other UID
    with pytest.raises(RuntimeError, match="replaced"):
        broker.ensure(claims[3])
    del api.pods[held]
    broker.ensure(claims[3])


def _ids(regions):
    return [(r["index"], r["offset"], r["size"]) for r in regions]


def test_server_path_waiters_keep_fifo_after_slot_held_replies(tmp_path, pools):
    from types import SimpleNamespace

    from kvcr import KVCRServiceError
    from kvcr.service.protocol import PidfdLiveness, _ClaimV2, _TierConfig
    from kvcr.service.server import _ThreadingUnixServer

    table, _ = _table(tmp_path)
    owner = [True]
    _claim(table, "a", alive=lambda: owner[0])

    class Registry:
        def claimed_guard(self, index):
            return object()

        def claim(self, *args):
            raise KVCRServiceError("reached the guard")

    server = SimpleNamespace(
        compatibility_digest=DIGEST, slots=table, registry=Registry()
    )

    def server_claim(incarnation):
        request = _ClaimV2(
            0,
            DIGEST,
            _TierConfig([("pool", 4096)], None),
            "127.0.0.1",
            5555,
            1,
            "t",
            "l",
            incarnation,
            total_bytes=4 * PAGE,
            minimum_bytes=2 * PAGE,
        )
        liveness = PidfdLiveness(os.pidfd_open(os.getpid()))
        try:
            _ThreadingUnixServer.dispatch(server, request, liveness, ME)
        except SlotError as error:
            return error.reason
        except KVCRServiceError:
            return "admitted"
        finally:
            liveness.close()  # the handler closes it with every SLOT_HELD reply

    assert server_claim("b") == SLOT_HELD
    assert server_claim("c") == SLOT_HELD
    assert table.waiters("t/l/0") == ["b", "c"]
    owner[0] = False  # A releases
    assert server_claim("c") == SLOT_HELD  # B is still head
    assert server_claim("b") == "admitted"


def test_server_path_grant_advertises_a_reachable_guard_endpoint(tmp_path, pools):
    from types import SimpleNamespace

    from kvcr.service.protocol import PidfdLiveness, _ClaimV2, _TierConfig
    from kvcr.service.server import _ThreadingUnixServer

    table, _ = _table(tmp_path)
    guard = SimpleNamespace(dead_incarnations=(), heartbeat_enabled=False)

    class Registry:
        def claimed_guard(self, index):
            return guard

        def claim(self, *args):
            return None, None, -1, None

    def endpoint(advertise_host, incarnation, control_host="0.0.0.0"):
        server = SimpleNamespace(
            compatibility_digest=DIGEST,
            slots=table,
            registry=Registry(),
            advertise_host=advertise_host,
        )
        request = _ClaimV2(
            0,
            DIGEST,
            _TierConfig([("pool", 4096)], None),
            control_host,
            55001,
            1,
            "t",
            "l",
            incarnation,
            total_bytes=4 * PAGE,
            minimum_bytes=2 * PAGE,
        )
        liveness = PidfdLiveness(os.pidfd_open(os.getpid()))
        try:
            granted, _ = _ThreadingUnixServer.dispatch(server, request, liveness, ME)
        finally:
            liveness.close()
        return granted.guard_endpoint

    assert endpoint(None, "a") == "0.0.0.0:55001"
    table.vacate("t/l/0", ME, 1)
    assert endpoint("10.1.2.3", "a") == "10.1.2.3:55001"
    table.vacate("t/l/0", ME, 1)
    # A claimant that bound a specific address is advertised as bound.
    assert endpoint("10.1.2.3", "a", "10.9.9.9") == "10.9.9.9:55001"


def test_claim_returns_its_own_generation(tmp_path, pools):
    table, _ = _table(tmp_path)
    alive = {"a": True, "b": True}
    kwargs = dict(
        tenant="t",
        lineage="l",
        guard_index=0,
        total=4 * PAGE,
        minimum=2 * PAGE,
        maximum=None,
        digest=DIGEST,
        peer=ME,
    )
    slot, gen_a = table.claim_with_generation(
        claim_id="a", owner_alive=lambda: alive["a"], **kwargs
    )
    alive["a"] = False
    slot_b, gen_b = table.claim_with_generation(
        claim_id="b", owner_alive=lambda: alive["b"], **kwargs
    )
    assert slot_b is slot and gen_b == gen_a + 1
    table.vacate("t/l/0", ME, gen_a)  # A's late vacate with its own generation
    assert slot.held and slot.claim_id == "b"


def test_failed_grow_is_never_persisted(tmp_path, pools, monkeypatch):
    table, _ = _table(tmp_path)
    slot = _claim(table, total=2, minimum=2, maximum=8)
    path = Path(table._state_path)
    before = _ids(json.loads(path.read_text())["slots"]["t/l/0"]["regions"])
    seen = []

    def failing_send(_slot):
        with table._lock:  # another slot's claim persists the table mid-grow
            table._persist()
        seen.append(_ids(json.loads(path.read_text())["slots"]["t/l/0"]["regions"]))
        raise OSError("no ack")

    slot.holder_conn = socket.socketpair()[0]
    monkeypatch.setattr(table, "_send_layout", failing_send)
    with pytest.raises(SlotError):
        table.grow("t/l/0", ME, 2 * PAGE, None)
    with table._lock:
        table._persist()
    after = _ids(json.loads(path.read_text())["slots"]["t/l/0"]["regions"])
    assert seen == [before] and after == before


@pytest.mark.parametrize("op", ["shrink", "grow"])
def test_resize_rechecks_ownership_under_the_slot_lock(
    tmp_path, pools, monkeypatch, op
):
    table, _ = _table(tmp_path)
    alive = {"a": True, "b": True}
    slot = _claim(table, "a", total=2, minimum=2, maximum=8, alive=lambda: alive["a"])
    real = table._owned

    def owned_then_b_claims(name, peer, *args):
        result = real(name, peer, *args)
        alive["a"] = False  # A dies; B claims between validation and the lock
        _claim(table, "b", total=2, minimum=2, maximum=8, alive=lambda: alive["b"])
        return result

    monkeypatch.setattr(table, "_owned", owned_then_b_claims)
    before = (slot.pool.regions, slot.pool.layout_generation)
    with pytest.raises(SlotError) as raised:
        if op == "shrink":
            table.shrink("t/l/0", ME, (0,))
        else:
            table.grow("t/l/0", ME, PAGE, None)
    assert raised.value.reason == IDENTITY_MISMATCH
    assert (slot.pool.regions, slot.pool.layout_generation) == before


def test_v1_claim_is_refused_on_a_slot_service(tmp_path, pools):
    from types import SimpleNamespace

    from kvcr.service.protocol import NOT_IMPLEMENTED, _Claim, _TierConfig
    from kvcr.service.server import _ThreadingUnixServer

    table, _ = _table(tmp_path)
    server = SimpleNamespace(compatibility_digest=DIGEST, slots=table, registry=None)
    request = _Claim(
        0, DIGEST, _TierConfig([("pool", 4096)], None), "127.0.0.1", 5555, 1
    )
    with pytest.raises(SlotError) as raised:
        _ThreadingUnixServer.dispatch(server, request, SimpleNamespace())
    assert raised.value.reason == NOT_IMPLEMENTED


_SLOT_ARGS = (  # no --guard-count or --pool-sizes-gb: v2 does not read them
    "--socket-path s --pool-dir d --compatibility-digest g --namespace bench "
    "--node-name node-a --no-require-pod-identity"
).split()
_HOLDER = "reg/holder@sha256:" + "a" * 64


def test_service_main_builds_the_slot_table(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "InClusterPodAPI", lambda namespace: FakeAPI())
    args = server._parse_args(
        _SLOT_ARGS + ["--holder-image", _HOLDER, "--slot-root", str(tmp_path)]
    )
    args.socket_path = str(tmp_path / "service.sock")
    table = server._build_slots(args)
    assert isinstance(table, SlotTable)
    assert not table.require_pod_identity
    assert table.broker.image == _HOLDER and table.broker.node_name == "node-a"
    v1 = _SLOT_ARGS[:6] + "--guard-count 1 --pool-sizes-gb 1".split()
    assert server._build_slots(server._parse_args(v1)) is None
    # v2 slots must use the toucher holder, not a v1 service pod.
    assert table.broker.toucher is True
    claim = _Claim_pool("p1")
    state = {"pod_name": "n", "creation_id": "c", "socket_path": "/n/p/s.sock"}
    container = table.broker._pod(claim, state)["spec"]["containers"][0]
    assert container["command"] == ["python", "-m", "kvcr.service.holder"]


def _Claim_pool(name):
    return node_service.PoolClaim(name, 1, "g")


def test_service_main_passes_the_slot_table_to_the_service(tmp_path, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def serve_forever(self):
            raise server._ShutdownRequested

        def close(self):
            pass

    monkeypatch.setattr(server, "InClusterPodAPI", lambda namespace: FakeAPI())
    monkeypatch.setattr(server, "_KVCRService", Fake)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kvcr-service"]
        + _SLOT_ARGS
        + ["--holder-image", _HOLDER, "--slot-root", str(tmp_path)]
        + ["--advertise-host", "10.1.2.3"],
    )
    server.main()
    assert seen["advertise_host"] == "10.1.2.3"
    assert isinstance(seen["slots"], SlotTable)
    assert seen["slots"].broker.toucher is True


@pytest.mark.parametrize(
    "value", ["", " ", "a b", "10.1.2.3:55001", "http://h", "h/x", "0.0.0.0", "::"]
)
def test_advertise_host_rejects_non_hosts(tmp_path, value):
    with pytest.raises(SystemExit):
        server._parse_args(
            _SLOT_ARGS
            + ["--holder-image", _HOLDER, "--slot-root", str(tmp_path)]
            + ["--advertise-host", value]
        )


def test_without_a_holder_image_the_service_is_v1_and_needs_its_flags():
    with pytest.raises(SystemExit):
        server._parse_args(_SLOT_ARGS)


def test_resize_recheck_rejects_a_slot_vacated_after_the_check(tmp_path, pools):
    table, _ = _table(tmp_path)
    granted = _claim_raw(table)
    slot = table.get("t/l/0")
    table._recheck_owner(slot, ME, granted.claim_generation)  # still held: fine
    slot.owner_alive = None  # released between _owned and slot.lock
    with pytest.raises(SlotError) as error:
        table._recheck_owner(slot, ME, granted.claim_generation)
    assert error.value.reason == IDENTITY_MISMATCH


def test_holder_pod_limit_covers_data_meta_and_overhead(tmp_path):
    table, _ = _table(tmp_path)
    table.broker.toucher = True
    claim = node_service.PoolClaim("p46", 46, "g")  # a 46 GiB slot
    state = {"pod_name": "n", "creation_id": "c", "socket_path": "/n/p/s.sock"}
    container = table.broker._pod(claim, state)["spec"]["containers"][0]
    limit = container["resources"]["limits"]["memory"]
    assert limit.endswith("Mi")
    assert container["resources"]["requests"]["memory"] == limit
    charged = claim.data_bytes + table.meta_bytes
    floor = charged + node_service._TOUCHER_OVERHEAD_BYTES + charged // 200
    assert int(limit[:-2]) << 20 >= floor
    assert table.meta_bytes > (100 << 20)  # the meta memfd is real memory here


def test_meta_region_is_derived_from_the_journal_size():
    journal = 100 << 20
    size = node_service.meta_region_bytes(journal)
    offset = size - journal - node_service._JOURNAL_HEADER_BYTES
    assert offset >= journal + node_service._JOURNAL_HEADER_BYTES
    assert not hasattr(node_service, "_META_BYTES")  # the journal, not a constant


def test_holder_report_must_cover_the_meta_memfd(tmp_path, pools):
    table, _ = _table(tmp_path)
    slot = _claim(table)
    supervisor_end, holder_end = socket.socketpair()
    slot.holder_conn = supervisor_end

    def old_holder():  # an image that predates the meta memfd: no meta_bytes
        socket.recv_fds(holder_end, 65536, 4)
        report = {"touched_bytes": 4 * PAGE, "layout_generation": 0}
        holder_end.sendall(json.dumps(report).encode() + b"\n")

    thread = threading.Thread(target=old_holder)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="meta bytes"):
            table._send_layout(slot)
    finally:
        thread.join(10)
        supervisor_end.close()
        holder_end.close()


def test_restored_slot_adopts_the_holders_handback(tmp_path, pools, monkeypatch):
    table, _ = _table(tmp_path)
    slot = _claim(table)
    os.pwrite(slot.pool.fd, b"S", 0)
    restarted = SlotTable(table.broker, require_pod_identity=False)
    again = restarted.get("t/l/0")
    fresh = again.pool.pass_fd_list()
    sup, peer = socket.socketpair()
    again.holder_conn = sup
    held = {
        "layout": {
            "regions": [
                {"index": r.index, "offset": r.offset, "size": r.size, "fd_index": 0}
                for r in slot.pool.regions
            ],
            "layout_generation": slot.pool.layout_generation,
            "incarnation": slot.incarnation,
            "meta": {"fd_index": 1, "size": table.meta_bytes},
        },
        "fds": slot.pool.pass_fd_list(),
    }

    def no_create(*args, **kwargs):
        raise AssertionError("adoption must not create memfds")

    monkeypatch.setattr(os, "memfd_create", no_create)
    try:
        holder.hello(peer, held)
        restarted._adopt_handback(again)
        for adopted, original in zip(again.pool.pass_fd_list(), held["fds"]):
            assert os.fstat(adopted)[:3] == os.fstat(original)[:3]
            assert not os.get_inheritable(adopted)
        for fd in fresh:
            with pytest.raises(OSError) as error:
                os.fstat(fd)
            assert error.value.errno == errno.EBADF
        assert os.pread(again.pool.fd, 1, 0) == b"S"

        def report():
            layout, fds = holder.receive_line_with_fds(peer)
            for fd in fds:
                os.close(fd)
            peer.sendall(
                json.dumps(
                    {
                        "touched_bytes": 4 * PAGE,
                        "layout_generation": layout["layout_generation"],
                        "meta_bytes": layout["meta"]["size"],
                    }
                ).encode()
                + b"\n"
            )

        thread = threading.Thread(target=report)
        thread.start()
        restarted._send_layout(again)
        thread.join(10)
        assert not thread.is_alive()
        assert again.touched_bytes == 4 * PAGE
        identities = [os.fstat(fd)[:3] for fd in again.pool.pass_fd_list()]
        bad = {**held, "layout": {**held["layout"], "layout_generation": 999}}
        real_close = os.close
        closed = []

        def close(fd):
            closed.append(fd)
            real_close(fd)

        monkeypatch.setattr(os, "close", close)
        holder.hello(peer, bad)
        with pytest.raises(ValueError, match="handback"):
            restarted._adopt_handback(again)
        assert len(closed) == 2
        for fd in closed:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert [os.fstat(fd)[:3] for fd in again.pool.pass_fd_list()] == identities
    finally:
        sup.close()
        peer.close()


def _handback(slot, table):
    return {
        "layout": {
            "regions": [
                {"index": r.index, "offset": r.offset, "size": r.size, "fd_index": 0}
                for r in slot.pool.regions
            ],
            "layout_generation": slot.pool.layout_generation,
            "incarnation": slot.incarnation,
            "meta": {"fd_index": 1, "size": table.meta_bytes},
        },
        "fds": slot.pool.pass_fd_list(),
    }


def test_live_handback_does_not_remap(tmp_path, pools, monkeypatch):
    table, _ = _table(tmp_path)
    slot = _claim(table)
    sup, peer = socket.socketpair()
    slot.holder_conn = sup

    def unexpected_adopt(*args):
        pytest.fail("a live pool must not be remapped")

    monkeypatch.setattr(RegionPool, "adopt", unexpected_adopt)
    try:
        holder.hello(peer, _handback(slot, table))
        table._adopt_handback(slot)
    finally:
        sup.close()
        peer.close()


@pytest.mark.parametrize("bad", ["envelope", "incarnation", "missing", "size"])
def test_bad_restored_handback_is_quarantined(tmp_path, pools, monkeypatch, bad):
    table, _ = _table(tmp_path)
    slot = _claim(table)
    restarted = SlotTable(table.broker, require_pod_identity=False)
    again = restarted.get("t/l/0")
    sup, peer = socket.socketpair()
    again.holder_conn = sup
    held = _handback(slot, table)
    closed = []
    real_close = os.close

    def close(fd):
        closed.append(fd)
        real_close(fd)

    try:
        if bad == "envelope":
            socket.send_fds(peer, [b"[]\n"], held["fds"])
        elif bad == "missing":
            holder.hello(peer, None)
        else:
            if bad == "incarnation":
                held["layout"]["incarnation"] += 1
            else:
                os.ftruncate(held["fds"][0], 1)
            holder.hello(peer, held)
        monkeypatch.setattr(os, "close", close)
        with pytest.raises(ValueError, match="holder"):
            restarted._adopt_handback(again)
        assert again.restored_pool
        assert len(closed) == (0 if bad == "missing" else 2)
        for fd in closed:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        sup.close()
        peer.close()


def test_failed_feed_drops_connection(tmp_path, pools, monkeypatch):
    table, _ = _table(tmp_path)
    slot = _claim(table)
    table.broker.toucher = True
    path = table.broker.holder_socket_path(
        table.broker._load(node_service.PoolClaim.parse(slot.holder))
    )

    def bad_holder():
        with holder.connect(path, 10) as conn:
            conn.sendall(b"[]\n")
            assert conn.recv(1) == b""  # no fresh layout follows a rejected hello

    thread = threading.Thread(target=bad_holder)
    thread.start()
    try:
        with pytest.raises(ValueError, match="hello"):
            table._feed_holder(slot)
        assert slot.holder_conn is None
    finally:
        thread.join(10)
        assert not thread.is_alive()
