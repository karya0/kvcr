# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Q7 reconnect retains mappings until a terminal lease refusal."""

import contextlib
import logging
import os
import socket
import sys
import threading
import time
from unittest.mock import Mock

import pytest
from test_guard import _TEST_SPEC

from kvcr.config import LocalDramOptions
from kvcr.service import protocol
from kvcr.service.protocol import (
    INCOMPATIBLE_LINEAGE,
    INTERNAL,
    SLOT_STALE,
    KVCRPoolHold,
    KVCRServiceReasonError,
    _ClaimV2,
    _ErrorV2,
    _GrantedV2,
    _TierConfig,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux memfds/pidfds")
_Thread = threading.Thread  # the fixture patches threading.Thread


@pytest.fixture
def reconnect_hold(monkeypatch):
    thread = Mock()
    monkeypatch.setattr(protocol.threading, "Thread", thread)
    region, meta = os.memfd_create("data"), os.memfd_create("meta")
    os.ftruncate(region, 4096)
    os.ftruncate(meta, 8192)
    request = _ClaimV2(
        0,
        "digest",
        _TierConfig([("", 4096)], None),
        "127.0.0.1",
        1234,
        1,
        "tenant",
        "lineage",
        incarnation="a" * 32,
        total_bytes=4096,
    )
    listener = socket.create_server(("127.0.0.1", 0))
    old_connection = Mock()
    attachment = Mock(_file_descriptor=meta)
    mapping = Mock()
    hold = KVCRPoolHold(
        LocalDramOptions([]),
        (),
        attachment,
        old_connection,
        _control_listener_fd=os.dup(listener.fileno()),
        _incarnation=request.incarnation,
        _data_map=mapping,
        slot_name="tenant/lineage/0",
        claim_generation=7,
        region_fds=[region],
        _reconnect_request=request,
        _socket_path="/socket",
    )
    thread.reset_mock()
    try:
        yield hold, thread, listener, region, meta
    finally:
        with contextlib.suppress(OSError):
            hold.close_local()
        for fd in (region, meta):
            with contextlib.suppress(OSError):
                os.close(fd)
        listener.close()


def _response(hold):
    return _GrantedV2(
        0,
        _TEST_SPEC,
        hold._reconnect_request.tier_config,
        (),
        1,
        hold.slot_name,
        4096,
        (),
        0,
        hold.claim_generation,
        "tcp://127.0.0.1:1234",
        reattached=True,
    )


def test_q7_reconnect_sends_pair_pidfd_listener_preserves_maps_and_restarts_heartbeat(
    reconnect_hold, monkeypatch
):
    hold, thread, listener, region, meta = reconnect_hold
    old = hold._connection
    hold._heartbeat_enabled = True
    connection = Mock()
    received = [os.dup(region), os.dup(listener.fileno()), os.dup(region), os.dup(meta)]
    connection.receive_with_fd.return_value = (_response(hold), received)
    send = []

    def sent(request, pidfd, listener_fd):
        send.append(request)
        assert os.fstat(pidfd).st_mode
        assert os.fstat(listener_fd).st_ino == os.fstat(listener.fileno()).st_ino

    connection.send_with_fd.side_effect = sent
    monkeypatch.setattr(
        protocol.FramedConnection, "connect", Mock(return_value=connection)
    )
    hold.reconnect()
    assert len(send) == 1
    assert send[0].incarnation == hold._reconnect_request.incarnation
    assert send[0].claim_generation == 7
    assert hold._connection is connection
    old.close.assert_called_once()
    hold._attachment.close.assert_not_called()
    hold._data_map.close.assert_not_called()
    assert hold.region_fds == [region]
    thread.assert_called_once_with(target=hold._heartbeat, daemon=True)
    thread.return_value.start.assert_called_once()
    for fd in received:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("refusal", ["server-stale", "backing-changed"])
def test_q7_terminal_stale_drops_maps_and_surfaces_error(
    reconnect_hold, monkeypatch, refusal
):
    hold, _, listener, region, meta = reconnect_hold
    connection = Mock()
    if refusal == "server-stale":
        response, received = _ErrorV2("stale lease", 1, SLOT_STALE), []
    else:
        # A different meta memfd is a mismatch even at the same length.
        other = os.memfd_create("other-meta")
        os.ftruncate(other, os.fstat(meta).st_size)
        response = _response(hold)
        received = [os.dup(region), os.dup(listener.fileno()), os.dup(region), other]
    connection.receive_with_fd.return_value = response, received
    monkeypatch.setattr(
        protocol.FramedConnection, "connect", Mock(return_value=connection)
    )
    with pytest.raises(KVCRServiceReasonError) as refused:
        hold.reconnect()
    assert refused.value.reason == SLOT_STALE
    hold._data_map.close.assert_called_once()
    hold._attachment.close.assert_called_once()
    assert hold._heartbeat_stop.is_set()
    assert hold.region_fds == []
    with pytest.raises(KVCRServiceReasonError):
        hold.check_lease()
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    "ttl,delay", [(1000, 0.1), (5000, 0.5), (10000, 1.0), (20000, 1.0)]
)
def test_q7_eof_retries_at_lease_fraction_without_dropping_maps(
    reconnect_hold, monkeypatch, ttl, delay
):
    hold, _, _, _, _ = reconnect_hold
    hold._lease_ttl_ms = ttl
    stop = Mock()
    # Initial EOF observation; failed attempt; successful attempt; exit outer loop.
    stop.wait.side_effect = [False, False, False, True]
    hold._heartbeat_stop = stop
    hold._connection._connection.recv.return_value = b""
    monkeypatch.setattr(protocol.select, "select", lambda *_args: ([1], [], []))
    hold.reconnect = Mock(side_effect=[OSError("still down"), None])
    hold._watch_connection()
    assert hold.reconnect.call_count == 2
    assert [call.args[0] for call in stop.wait.call_args_list] == [
        protocol._HEARTBEAT_SECONDS,
        delay,
        delay,
        protocol._HEARTBEAT_SECONDS,
    ]
    hold._data_map.close.assert_not_called()
    hold._attachment.close.assert_not_called()


def test_q7_existing_caller_operations_surface_terminal_stale(reconnect_hold):
    from kvcr.api import KVCR

    hold, _, _, _, _ = reconnect_hold
    hold._lease_error = KVCRServiceReasonError("expired generation", SLOT_STALE)
    owner = object.__new__(KVCR)
    owner._pool_hold = hold
    owner._core = Mock()
    for operation, args in (
        (owner.query, ([],)),
        (owner.deliver, ({},)),
        (owner.deposit, ({},)),
        (owner.fetch, ([],)),
        (owner.poll_completed, ()),
    ):
        with pytest.raises(KVCRServiceReasonError) as refused:
            operation(*args)
        assert refused.value.reason == SLOT_STALE
    assert owner._core.mock_calls == []


def _clock(monkeypatch, hold):
    now = [100.0]
    monkeypatch.setattr(protocol, "time", Mock(monotonic=lambda: now[0]))
    hold._alive_at = now[0]
    return now


def _eof(hold, monkeypatch):
    monkeypatch.setattr(protocol.select, "select", lambda *_args: ([1], [], []))
    hold._connection._connection.recv.return_value = b""


def test_lease_window_passed_fences_and_stops_reconnecting(reconnect_hold, monkeypatch):
    hold, _, _, _, _ = reconnect_hold
    now = _clock(monkeypatch, hold)
    _eof(hold, monkeypatch)
    stop = Mock()
    stop.wait.return_value = False
    stop.is_set.return_value = False
    hold._heartbeat_stop = stop

    def down():
        now[0] += hold._lease_ttl_ms / 1000
        raise OSError("still down")

    hold.reconnect = Mock(side_effect=down)
    hold._watch_connection()
    assert hold.reconnect.call_count == 1
    hold._data_map.close.assert_called_once()
    with pytest.raises(KVCRServiceReasonError) as refused:
        hold.check_lease()
    assert refused.value.reason == SLOT_STALE


def test_frozen_worker_is_refused_by_its_next_operation(reconnect_hold, monkeypatch):
    hold, _, _, _, _ = reconnect_hold
    now = _clock(monkeypatch, hold)
    monkeypatch.setattr(protocol.select, "select", lambda *_args: ([], [], []))
    now[0] += hold._lease_ttl_ms / 1000
    hold.check_lease()  # the connection is still open: the window restarts
    assert hold._alive_at == now[0]
    _eof(hold, monkeypatch)
    now[0] += hold._lease_ttl_ms / 1000
    with pytest.raises(KVCRServiceReasonError) as refused:
        hold.check_lease()
    assert refused.value.reason == SLOT_STALE
    hold._data_map.close.assert_called_once()


def test_reattach_inside_the_window_clears_the_deadline(reconnect_hold, monkeypatch):
    hold, _, listener, region, meta = reconnect_hold
    now = _clock(monkeypatch, hold)
    _eof(hold, monkeypatch)
    connection = Mock()
    received = [os.dup(region), os.dup(listener.fileno()), os.dup(region), os.dup(meta)]
    connection.receive_with_fd.return_value = (_response(hold), received)
    monkeypatch.setattr(
        protocol.FramedConnection, "connect", Mock(return_value=connection)
    )
    now[0] += hold._lease_ttl_ms / 1000 - 1
    hold.reconnect()
    now[0] += 2  # past the deadline the lost connection set
    connection._connection.recv.return_value = b""
    hold.check_lease()
    hold._data_map.close.assert_not_called()


def test_transient_reattach_refusal_is_retried_not_fenced(reconnect_hold, monkeypatch):
    hold, _, listener, region, meta = reconnect_hold
    _eof(hold, monkeypatch)
    stop = Mock()
    # EOF observed; INTERNAL refusal; successful re-attach; exit outer loop.
    stop.wait.side_effect = [False, False, False, True]
    stop.is_set.return_value = False
    hold._heartbeat_stop = stop
    refused, granted = Mock(), Mock()
    refused.receive_with_fd.return_value = (
        _ErrorV2("pod lookup failed", 1, protocol.INTERNAL),
        [],
    )
    received = [os.dup(region), os.dup(listener.fileno()), os.dup(region), os.dup(meta)]
    granted.receive_with_fd.return_value = (_response(hold), received)
    monkeypatch.setattr(
        protocol.FramedConnection, "connect", Mock(side_effect=[refused, granted])
    )
    hold._watch_connection()
    assert hold._connection is granted
    hold.check_lease()
    hold._data_map.close.assert_not_called()
    hold._attachment.close.assert_not_called()


def _grant(hold, monkeypatch, listener, region, meta, *, on_receive=lambda: None):
    connection = Mock()
    received = [os.dup(region), os.dup(listener.fileno()), os.dup(region), os.dup(meta)]

    def receive(*_args):
        on_receive()
        return _response(hold), received

    connection.receive_with_fd.side_effect = receive
    monkeypatch.setattr(
        protocol.FramedConnection, "connect", Mock(return_value=connection)
    )
    return connection


def test_reattach_anchors_the_deadline_at_the_request_time(reconnect_hold, monkeypatch):
    hold, _, listener, region, meta = reconnect_hold
    now = _clock(monkeypatch, hold)
    now[0] += 5

    def slow():
        now[0] += 3

    _grant(hold, monkeypatch, listener, region, meta, on_receive=slow)
    hold.reconnect()
    assert hold._alive_at == 105  # sent at 105, granted at 108


def test_reattach_granted_after_the_deadline_does_not_revive(
    reconnect_hold, monkeypatch
):
    hold, _, listener, region, meta = reconnect_hold
    old = hold._connection
    now = _clock(monkeypatch, hold)
    now[0] += 5

    def late():
        now[0] += hold._lease_ttl_ms / 1000

    connection = _grant(hold, monkeypatch, listener, region, meta, on_receive=late)
    with pytest.raises(KVCRServiceReasonError) as refused:
        hold.reconnect()
    assert refused.value.reason == SLOT_STALE
    assert hold._connection is old
    connection.close.assert_called_once()
    hold._data_map.close.assert_called_once()
    with pytest.raises(KVCRServiceReasonError):
        hold.check_lease()


def test_check_lease_and_watcher_at_the_deadline_unmap_once(
    reconnect_hold, monkeypatch
):
    hold, _, _, _, _ = reconnect_hold
    now = _clock(monkeypatch, hold)
    _eof(hold, monkeypatch)
    now[0] += hold._lease_ttl_ms / 1000
    stop = Mock()
    stop.wait.side_effect = [False, False, True]
    stop.is_set.return_value = False
    hold._heartbeat_stop = stop
    hold._data_map.close.side_effect = lambda: time.sleep(0.05)
    caller = _Thread(target=lambda: pytest.raises(Exception, hold.check_lease))
    watcher = _Thread(target=hold._watch_connection)
    caller.start()
    watcher.start()
    caller.join()
    watcher.join()
    hold._fence_expired()
    hold._data_map.close.assert_called_once()
    hold._attachment.close.assert_called_once()


def test_release_racing_a_fence_waits_for_its_teardown(reconnect_hold):
    hold, _, _, _, _ = reconnect_hold
    entered, inside, overlapped = threading.Event(), threading.Lock(), []

    def unmap():
        if not inside.acquire(blocking=False):
            overlapped.append(True)
            return
        entered.set()
        time.sleep(0.05)
        inside.release()

    hold._data_map.close.side_effect = unmap
    error = KVCRServiceReasonError("lost", SLOT_STALE)
    fence = _Thread(target=hold._fence, args=(error,))
    fence.start()
    entered.wait()
    hold.release()
    fence.join()
    assert not overlapped
    assert hold.region_fds == [] and hold._control_listener_fd is None


def test_check_lease_waits_for_a_reattach_publishing_in_time(
    reconnect_hold, monkeypatch
):
    hold, _, listener, region, meta = reconnect_hold
    old = hold._connection
    now = _clock(monkeypatch, hold)
    _eof(hold, monkeypatch)
    _grant(hold, monkeypatch, listener, region, meta)
    now[0] += hold._lease_ttl_ms / 1000 - 1  # sent and granted inside the window
    checker = _Thread(target=hold.check_lease)

    def publishing():
        now[0] += 2  # past the lost connection's deadline, inside the new one
        checker.start()
        checker.join(0.2)
        assert checker.is_alive()  # the deadline check waits for the publication

    old.close.side_effect = publishing
    hold.reconnect()
    checker.join()
    hold.check_lease()
    assert hold._alive_at == 109
    hold._data_map.close.assert_not_called()


def _refusing(hold, monkeypatch, reason, now=None):
    refused = Mock()
    error = _ErrorV2("pod lookup failed", 1, reason)
    refused.receive_with_fd.return_value = error, []

    def connect(_path):
        if now is not None:
            now[0] += 4
        return refused

    connect = Mock(side_effect=connect)
    monkeypatch.setattr(protocol.FramedConnection, "connect", connect)
    stop = Mock()
    stop.wait.return_value = False
    stop.is_set.return_value = False
    hold._heartbeat_stop = stop
    return connect


def test_incompatible_lineage_on_reattach_is_terminal(reconnect_hold, monkeypatch):
    hold, _, _, _, _ = reconnect_hold
    _eof(hold, monkeypatch)
    connect = _refusing(hold, monkeypatch, INCOMPATIBLE_LINEAGE)
    hold._watch_connection()
    assert connect.call_count == 1
    hold._data_map.close.assert_called_once()
    with pytest.raises(KVCRServiceReasonError) as refused:
        hold.check_lease()
    assert refused.value.reason == INCOMPATIBLE_LINEAGE


def test_internal_refusals_retry_log_once_and_name_the_last_in_the_fence(
    reconnect_hold, monkeypatch, caplog
):
    hold, _, _, _, _ = reconnect_hold
    now = _clock(monkeypatch, hold)
    _eof(hold, monkeypatch)
    connect = _refusing(hold, monkeypatch, INTERNAL, now)
    with caplog.at_level(logging.WARNING, logger=protocol.logger.name):
        hold._watch_connection()
    assert connect.call_count == 3  # at 104, 108 and 112; the deadline was 110
    assert sum("retrying" in record.message for record in caplog.records) == 1
    with pytest.raises(KVCRServiceReasonError) as refused:
        hold.check_lease()
    assert refused.value.reason == SLOT_STALE
    assert "pod lookup failed" in str(refused.value)
