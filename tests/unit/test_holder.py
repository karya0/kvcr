# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""v2 holder: toucher pod spec, layout over SCM_RIGHTS, fd counts (Linux only)."""

import copy
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import msgspec
import pytest

if sys.platform != "linux":
    pytest.skip("memfd and SCM_RIGHTS regions are Linux-only", allow_module_level=True)

from test_node_service import FakeAPI, make_broker
from test_node_service_v2 import DIGEST, HEADROOM, PAGE, _claim

from kvcr.service import holder, protocol
from kvcr.service.memory import KVCRPoolSpec, _pool_filename
from kvcr.service.node_service import PoolClaim, SlotError, SlotTable
from kvcr.service.protocol import (
    IDENTITY_MISMATCH,
    SLOT_HELD,
    PoolRegion,
    _Error,
    _ErrorV2,
    _Granted,
    _GrantedV2,
    _PoolDescriptor,
    _response_fd_counts,
    _TierConfig,
)
from kvcr.service.server import pod_uid_from_cgroup


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="gaas-h-", dir="/tmp") as directory:
        yield Path(directory)


def _send_layout(conn, regions, generation, fds):
    layout = {"regions": regions, "layout_generation": generation}
    socket.send_fds(conn, [json.dumps(layout).encode() + b"\n"], fds)


def _read_report(conn):
    return json.loads(conn.makefile("rb").readline())


def test_holder_touches_and_reports(capsys):
    fd = os.memfd_create("t")
    os.ftruncate(fd, 2 * PAGE)
    os.pwrite(fd, b"keep", 0)
    supervisor, worker = socket.socketpair()
    mappings = {}
    region = {"index": 0, "offset": 0, "size": 2 * PAGE, "fd_index": 0}
    _send_layout(supervisor, [region], 3, [fd])
    assert holder.handle_layout(worker, mappings) is True
    assert _read_report(supervisor) == {
        "touched_bytes": 2 * PAGE,
        "layout_generation": 3,
    }
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines()]
    assert [x["holder"] for x in lines] == ["populate_start", "populate_end"]
    assert lines[1]["bytes"] == 2 * PAGE
    assert os.pread(fd, 4, 0) == b"keep"  # touching does not clobber data
    # A Grow re-sends the full layout; only the new region is mapped.
    os.ftruncate(fd, 3 * PAGE)
    grown = {"index": 1, "offset": 2 * PAGE, "size": PAGE, "fd_index": 0}
    _send_layout(supervisor, [region, grown], 4, [fd])
    assert holder.handle_layout(worker, mappings) is True
    assert _read_report(supervisor) == {
        "touched_bytes": 3 * PAGE,
        "layout_generation": 4,
    }
    supervisor.close()
    assert holder.handle_layout(worker, mappings) is False
    for view, _size in mappings.values():
        view.close()
    os.close(fd)
    worker.close()


def test_response_fd_counts():
    spec = KVCRPoolSpec(
        "p", "/x/" + _pool_filename("p", "0" * 32), "0" * 32, 1, 2, 16384, 8192
    )
    v2 = _GrantedV2(
        0,
        spec,
        _TierConfig([("pool", PAGE)], None),
        (_PoolDescriptor("pool", PAGE, PAGE, 0),),
        1,
        "t/l/0",
        PAGE,
        (PoolRegion(0, 0, PAGE, 0),),
        1,
        1,
        "h:1",
    )
    assert _response_fd_counts(v2) == (3,)
    two_fds = msgspec.structs.replace(
        v2, regions=(PoolRegion(0, 0, PAGE, 0), PoolRegion(1, PAGE, PAGE, 1))
    )
    assert _response_fd_counts(two_fds) == (4,)
    v1 = _Granted(0, v2.spec, v2.tier_config, v2.pools, 1)
    assert _response_fd_counts(v1) == (2,)
    assert _response_fd_counts(_Error("x", 2)) == (0,)
    assert _response_fd_counts(_ErrorV2("x", 2)) == (0,)


def test_holder_reconnects_after_eof_and_remaps(tmp_path):
    path = str(tmp_path / "h.sock")
    stop = threading.Event()
    mappings = {}
    thread = threading.Thread(
        target=holder.serve, args=(path, 10, stop, mappings), daemon=True
    )

    def feed(listener, size):
        conn, _ = listener.accept()
        fd = os.memfd_create("x")
        os.ftruncate(fd, size)
        layout = {
            "regions": [{"index": 0, "offset": 0, "size": size, "fd_index": 0}],
            "layout_generation": 1,
        }
        socket.send_fds(conn, [json.dumps(layout).encode() + b"\n"], [fd])
        os.close(fd)
        return conn, _read_report(conn)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(path)
        listener.listen(1)
        listener.settimeout(10)
        thread.start()
        conn, report = feed(listener, PAGE)
        assert report["touched_bytes"] == PAGE
        conn.close()  # the supervisor goes away
        listener.close()
        os.unlink(path)  # holder now retries with backoff, not sleeping forever
        time.sleep(0.5)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as second:
            second.bind(path)
            second.listen(1)
            second.settimeout(10)
            conn, report = feed(second, 2 * PAGE)
            # Remapped from the new memfd, not left on the stale mapping.
            assert report["touched_bytes"] == 2 * PAGE
            conn.close()
    stop.set()
    thread.join(10)
    assert not thread.is_alive()


class _ToucherAPI(FakeAPI):
    def create(self, pod):
        pod = copy.deepcopy(pod)
        pod["spec"]["containers"][0]["readinessProbe"] = {}
        return super().create(pod)


def test_toucher_pod_is_guaranteed_and_v1_unchanged(tmp_path):
    claim = PoolClaim("p1", 1, DIGEST)
    v1 = make_broker(tmp_path / "a", _ToucherAPI())
    v2 = make_broker(tmp_path / "b", _ToucherAPI())
    v2.toucher = True
    state = {
        "pod_name": "gaas-pool-x",
        "creation_id": "c",
        "socket_path": "/n/p/s.sock",
    }
    old, new = v1._pod(claim, state), v2._pod(claim, state)
    assert old["spec"]["containers"][0]["command"] == ["kvcr-service"]
    container = new["spec"]["containers"][0]
    assert container["command"] == ["python", "-m", "kvcr.service.holder"]
    assert container["args"] == ["--socket", "/n/p/holder.sock"]
    resources = container["resources"]
    assert resources["requests"] == resources["limits"]
    assert resources["requests"]["cpu"] == "1"
    assert resources["limits"]["cpu"] == "1"
    assert old["spec"]["containers"][0]["resources"]["requests"]["cpu"] == "100m"
    assert "readinessProbe" not in container


def test_toucher_memory_covers_page_tables(tmp_path):
    v2 = make_broker(tmp_path, _ToucherAPI())
    v2.toucher = True
    state = {
        "pod_name": "gaas-pool-x",
        "creation_id": "c",
        "socket_path": "/n/p/s.sock",
    }

    def limit_mib(gib):
        pod = v2._pod(PoolClaim("p1", gib, DIGEST), state)
        return int(pod["spec"]["containers"][0]["resources"]["limits"]["memory"][:-2])

    # Measured in c5-repro-memcg-20261107.txt: 46 GiB needs 47352 MiB (the 47324 MiB
    # sizing was OOM-killed); kernel charge is data / 235, holder anon about 46 MiB.
    assert limit_mib(46) >= 47352
    for gib in (46, 256):
        assert limit_mib(gib) - gib * 1024 >= gib * 1024 // 235 + 46 + 32
    assert limit_mib(1) <= 1024 + 136


def test_slot_claim_feeds_toucher(tmp_path):
    api = _ToucherAPI()
    broker = make_broker(tmp_path, api)
    broker.toucher = True
    table = SlotTable(
        broker, HEADROOM + 64 * PAGE, ensure_timeout_s=10, require_pod_identity=False
    )
    reports = []

    def run_holder():
        while not api.creates:
            time.sleep(0.01)
        args = api.creates[0]["spec"]["containers"][0]["args"]
        conn = holder.connect(args[1], 10)
        mappings = {}
        holder.handle_layout(conn, mappings)
        reports.append(sum(size for _v, size in mappings.values()))
        for view, _ in mappings.values():
            view.close()
        conn.close()

    thread = threading.Thread(target=run_holder)
    thread.start()
    try:
        slot = _claim(table)
        thread.join(10)
        assert slot.touched_bytes == 4 * PAGE == reports[0]
        assert slot.layout_generation == slot.pool.layout_generation
    finally:
        thread.join(10)
        table.release("t/l/0")


def test_slow_holder_wait_is_size_aware(tmp_path, monkeypatch):
    # 4 pages at this floor add 1 s to the 1 s base, so the report wait is 2 s.
    monkeypatch.setattr(protocol, "POPULATE_RATE_FLOOR_BYTES_PER_S", 4 * PAGE)
    api = _ToucherAPI()
    broker = make_broker(tmp_path, api)
    broker.toucher = True
    table = SlotTable(
        broker, HEADROOM + 64 * PAGE, ensure_timeout_s=1, require_pod_identity=False
    )
    done = threading.Event()

    def silent_holder():
        while not api.creates:
            time.sleep(0.01)
        args = api.creates[0]["spec"]["containers"][0]["args"]
        conn = holder.connect(args[1], 10)
        done.wait(10)  # takes the layout but never reports
        conn.close()

    thread = threading.Thread(target=silent_holder)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(Exception, match=r"holder did not report within 2 s"):
            _claim(table)
        assert time.monotonic() - started >= 1.9
    finally:
        done.set()
        thread.join(10)


def test_populate_falls_back_when_madvise_fails(capsys, monkeypatch):
    monkeypatch.setattr(holder, "_MADV_POPULATE_WRITE", 9999)  # EINVAL
    fd = os.memfd_create("t")
    os.ftruncate(fd, 2 * PAGE)
    supervisor, worker = socket.socketpair()
    mappings = {}
    region = {"index": 0, "offset": 0, "size": 2 * PAGE, "fd_index": 0}
    _send_layout(supervisor, [region], 1, [fd])
    assert holder.handle_layout(worker, mappings) is True
    assert _read_report(supervisor) == {
        "touched_bytes": 2 * PAGE,
        "layout_generation": 1,
    }
    events = [json.loads(x)["holder"] for x in capsys.readouterr().err.splitlines()]
    assert events == ["populate_start", "populate_fallback", "populate_end"]
    for view, _size in mappings.values():
        view.close()
    os.close(fd)
    supervisor.close()
    worker.close()


UID = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
UNDERSCORED = UID.replace("-", "_")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            f"0::/kubepods.slice/kubepods-burstable.slice/"
            f"kubepods-burstable-pod{UNDERSCORED}.slice/cri-containerd-ab12.scope\n",
            UID,
        ),
        (f"0::/kubepods.slice/kubepods-pod{UNDERSCORED}.slice/cri-o-ab.scope\n", UID),
        (f"12:memory:/kubepods/burstable/pod{UID}/ab12cd\n1:name=systemd:/x\n", UID),
        (f"1:cpu:/kubepods/pod{UID}\n", UID),
        ("0::/user.slice/user-1000.slice/session-3.scope\n", None),
        ("0::/system.slice/containerd.service\n", None),
        ("", None),
    ],
)
def test_pod_uid_from_cgroup(text, expected):
    assert pod_uid_from_cgroup(text) == expected


def _claim_as(table, peer, claim_id="a", alive=lambda: True):
    return table.claim(
        "t",
        "l",
        0,
        total=4 * PAGE,
        minimum=2 * PAGE,
        maximum=8 * PAGE,  # room above the grant to grow into
        digest=DIGEST,
        claim_id=claim_id,
        peer=peer,
        owner_alive=alive,
    )


def test_resize_identity_by_pod_uid(tmp_path):
    table = SlotTable(
        make_broker(tmp_path), HEADROOM + 64 * PAGE, require_pod_identity=False
    )
    owner = (100, 1000, "pod-a")
    try:
        _claim_as(table, owner)
        # Same pod from another process: allowed.
        table.grow("t/l/0", (101, 1000, "pod-a"), PAGE, None)
        # Different pod: refused, even with the same pid and uid.
        with pytest.raises(SlotError) as raised:
            table.grow("t/l/0", (100, 1000, "pod-b"), PAGE, None)
        assert raised.value.reason == IDENTITY_MISMATCH
        # No pod UID on one side: fall back to (pid, uid).
        table.grow("t/l/0", (100, 1000, None), PAGE, None)
        with pytest.raises(SlotError):
            table.grow("t/l/0", (101, 1000, None), PAGE, None)
    finally:
        table.release("t/l/0")


def test_successor_pod_claims_free_slot_not_held_slot(tmp_path):
    table = SlotTable(
        make_broker(tmp_path), HEADROOM + 64 * PAGE, require_pod_identity=False
    )
    alive = [True]
    try:
        _claim_as(table, (100, 1000, "pod-a"), alive=lambda: alive[0])
        with pytest.raises(SlotError) as raised:
            _claim_as(table, (200, 1000, "pod-b"), claim_id="b")
        assert raised.value.reason == SLOT_HELD
        alive[0] = False
        slot = _claim_as(table, (200, 1000, "pod-b"), claim_id="b")
        assert slot.peer[2] == "pod-b"
        with pytest.raises(SlotError):
            table.grow("t/l/0", (100, 1000, "pod-a"), PAGE, None)
    finally:
        table.release("t/l/0")


def test_receive_closes_fds_when_the_line_is_bad():
    supervisor, holder_end = socket.socketpair()
    fd = os.memfd_create("t")
    seen = []
    real_close = os.close
    try:
        socket.send_fds(supervisor, [b"not json\n"], [fd])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(os, "close", lambda f: (seen.append(f), real_close(f)))
            with pytest.raises(ValueError):
                holder._receive(holder_end)
        assert len(seen) == 1  # the received duplicate, not the original
    finally:
        os.close(fd)
        supervisor.close()
        holder_end.close()
