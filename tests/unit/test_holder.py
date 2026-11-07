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
from types import SimpleNamespace

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and SCM_RIGHTS regions are Linux-only", allow_module_level=True)

from test_node_service import FakeAPI, make_broker
from test_node_service_v2 import DIGEST, HEADROOM, PAGE, _claim

from kvcr.service import holder
from kvcr.service.node_service import PoolClaim, SlotError, SlotTable
from kvcr.service.protocol import (
    IDENTITY_MISMATCH,
    SLOT_HELD,
    PoolRegion,
    _Error,
    _response_fd_counts,
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


def test_holder_touches_and_reports():
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
    assert _response_fd_counts(SimpleNamespace(slot_name=None, regions=())) == (2,)
    v2 = SimpleNamespace(slot_name="t/l", regions=(PoolRegion(0, 0, PAGE, 0),))
    assert _response_fd_counts(v2) == (3,)
    two_fds = SimpleNamespace(
        slot_name="t/l",
        regions=(PoolRegion(0, 0, PAGE, 0), PoolRegion(1, PAGE, PAGE, 1)),
    )
    assert _response_fd_counts(two_fds) == (4,)
    assert _response_fd_counts(_Error("x", 2)) == (0,)


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
    assert resources["requests"]["cpu"] == "100m"
    assert "readinessProbe" not in container


def test_slot_claim_feeds_toucher(tmp_path):
    api = _ToucherAPI()
    broker = make_broker(tmp_path, api)
    broker.toucher = True
    table = SlotTable(broker, HEADROOM + 64 * PAGE, ensure_timeout_s=10)
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
        table.release("t/l")


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
        total=4 * PAGE,
        minimum=2 * PAGE,
        maximum=8 * PAGE,  # room above the grant to grow into
        digest=DIGEST,
        claim_id=claim_id,
        peer=peer,
        owner_alive=alive,
    )


def test_resize_identity_by_pod_uid(tmp_path):
    table = SlotTable(make_broker(tmp_path), HEADROOM + 64 * PAGE)
    owner = (100, 1000, "pod-a")
    try:
        _claim_as(table, owner)
        # Same pod from another process: allowed.
        table.grow("t/l", (101, 1000, "pod-a"), PAGE, None)
        # Different pod: refused, even with the same pid and uid.
        with pytest.raises(SlotError) as raised:
            table.grow("t/l", (100, 1000, "pod-b"), PAGE, None)
        assert raised.value.reason == IDENTITY_MISMATCH
        # No pod UID on one side: fall back to (pid, uid).
        table.grow("t/l", (100, 1000, None), PAGE, None)
        with pytest.raises(SlotError):
            table.grow("t/l", (101, 1000, None), PAGE, None)
    finally:
        table.release("t/l")


def test_successor_pod_claims_free_slot_not_held_slot(tmp_path):
    table = SlotTable(make_broker(tmp_path), HEADROOM + 64 * PAGE)
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
            table.grow("t/l", (100, 1000, "pod-a"), PAGE, None)
    finally:
        table.release("t/l")
