# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import concurrent.futures
import errno
import fcntl
import os
import sys
from pathlib import Path

import pytest
from test_kvcr_service import (
    _TEST_DIGEST,
    _TEST_JOURNAL_BYTES,
    _TEST_POOL_SIZES_BYTES,
    _TEST_TIER_CONFIG,
    _control_bind,
    _FakeLiveness,
    _running_server,
)

from kvcr.runtime.control_channels import KVCRServiceError
from kvcr.service import guard as guard_module
from kvcr.service import memory as memory_module
from kvcr.service.guard import _Phase
from kvcr.service.memory import KVCRPoolAttachment, _KVCRPoolOwner
from kvcr.service.protocol import drain_service, service_info
from kvcr.service.server import _PoolRegistry

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd/libatomic")


def _owner(tmp_path):
    return _KVCRPoolOwner.allocate(
        pool_id="drain", pool_size_bytes=16384, journal_bytes=8192, pool_dir=tmp_path
    )


def test_owner_retains_backing_until_external_mapping_closes(tmp_path):
    owner = _owner(tmp_path)
    attachment = KVCRPoolAttachment.attach(owner.spec)
    try:
        assert not owner.drain()
        assert Path(owner.spec.path).exists()
        attachment.close()
        assert owner.drain()
        assert owner.drain()  # an acknowledged drain is idempotent
        assert not Path(owner.spec.path).exists()
    finally:
        attachment.close()
        owner.close()


def test_owner_refuses_replaced_identity(tmp_path):
    owner = _owner(tmp_path)
    path = Path(owner.spec.path)
    old = path.with_name("original")
    path.rename(old)
    path.write_bytes(b"replacement")
    try:
        with pytest.raises(RuntimeError, match="identity changed"):
            owner.drain()
        assert path.read_bytes() == b"replacement"
        assert old.exists()
    finally:
        owner.close()


def test_owner_lock_error_preserves_file_and_restores_owner_lock(tmp_path, monkeypatch):
    owner = _owner(tmp_path)
    real_lock = memory_module._try_lock_pool

    def broken_lock(fd, *, exclusive):
        if exclusive:
            raise OSError(errno.EIO, "lock uncertain")
        return real_lock(fd, exclusive=exclusive)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(memory_module, "_try_lock_pool", broken_lock)
            with pytest.raises(OSError, match="lock uncertain"):
                owner.drain()
        assert Path(owner.spec.path).exists()
        with open(owner.spec.path, "r+b") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        owner.close()


def test_pending_grant_liveness_blocks_drain_even_without_an_attachment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(guard_module._Guard, "_prepare_core", lambda self: None)
    registry = _PoolRegistry(
        tmp_path, 1, _TEST_POOL_SIZES_BYTES, _TEST_JOURNAL_BYTES, _TEST_DIGEST
    )
    liveness = _FakeLiveness()
    spec, _, listener_fd, lease = registry.claim(
        0, _TEST_TIER_CONFIG, liveness, _control_bind()
    )
    os.close(listener_fd)
    try:
        # A successful send may have reached the worker before attach_fd locks it.
        assert registry.drain()["state"] == "busy"
        assert Path(spec.path).exists()
        assert registry._guards[0]._recovery.attachment is not None
        with pytest.raises(KVCRServiceError, match="closed"):
            registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind())
        registry.release(0, lease)
        assert registry.drain()["state"] == "drained"
        assert not Path(spec.path).exists()
    finally:
        if not liveness.closed:
            registry.release(0, lease)
        registry.close()


def test_claim_transition_and_failed_quiescence_never_authorize_deletion(
    tmp_path, monkeypatch
):
    registry = _PoolRegistry(
        tmp_path, 1, _TEST_POOL_SIZES_BYTES, _TEST_JOURNAL_BYTES, _TEST_DIGEST
    )
    guard = registry._guards[0]
    try:
        with guard._phase_lock:
            guard._reserved = _Phase.CLAIMING
        assert registry.drain()["state"] == "busy"
        assert Path(guard._spec.path).exists()
        with guard._phase_lock:
            guard._reserved = None
        with monkeypatch.context() as patch:
            patch.setattr(guard, "finish_close", lambda deadline: True)
            assert registry.drain()["state"] == "busy"
            assert Path(guard._spec.path).exists()
        assert registry.drain()["state"] == "drained"
    finally:
        registry.close()


def test_wire_drain_busy_then_closed_preserves_holder_liveness(tmp_path):
    with _running_server(tmp_path, guard_count=1) as harness:
        initial = service_info(harness.server.socket_path)
        assert initial["state"] == "ready"
        assert initial["compatibility_digest"] == _TEST_DIGEST
        assert initial["ready"]
        assert len(initial["pool_specs"]) == 1
        spec = harness.server._registry._guards[0]._spec
        attachment = KVCRPoolAttachment.attach(spec)
        try:
            assert drain_service(harness.server.socket_path)["state"] == "busy"
            assert Path(spec.path).exists()
            assert harness.server._health_status(10, ready=False)[0] == 200
            assert harness.server._health_status(10, ready=True)[0] == 503
            attachment.close()
            result = drain_service(harness.server.socket_path)
            assert result["state"] == "drained"
            assert result["pool_specs"] == initial["pool_specs"]
            assert result["compatibility_digest"] == initial["compatibility_digest"]
            assert not result["ready"]
            assert not Path(spec.path).exists()
            assert drain_service(harness.server.socket_path)["state"] == "drained"
            assert service_info(harness.server.socket_path)["state"] == "drained"
        finally:
            attachment.close()


def test_administrative_close_does_not_fall_back_to_unsafe_legacy_reclaim(tmp_path):
    registry = _PoolRegistry(
        tmp_path, 1, _TEST_POOL_SIZES_BYTES, _TEST_JOURNAL_BYTES, _TEST_DIGEST
    )
    spec = registry._guards[0]._spec
    attachment = KVCRPoolAttachment.attach(spec)
    try:
        assert registry.drain()["state"] == "busy"
        with pytest.raises(KVCRServiceError, match="undrained"):
            registry.close()
        assert Path(spec.path).exists()
        attachment.close()
        registry.close()
        assert not Path(spec.path).exists()
    finally:
        attachment.close()
        registry.close()


def test_drain_cancellation_keeps_backing_and_releases_the_admin_lock(
    tmp_path, monkeypatch
):
    registry = _PoolRegistry(
        tmp_path, 1, _TEST_POOL_SIZES_BYTES, _TEST_JOURNAL_BYTES, _TEST_DIGEST
    )
    guard = registry._guards[0]
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                guard,
                "finish_close",
                lambda deadline: (_ for _ in ()).throw(KeyboardInterrupt()),
            )
            with pytest.raises(KeyboardInterrupt):
                registry.drain()
        assert Path(guard._spec.path).exists()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            assert (
                executor.submit(registry.drain).result(timeout=3)["state"] == "drained"
            )
    finally:
        registry.close()


def test_uncertain_lock_is_an_error_not_a_reclamation_ack(tmp_path, monkeypatch):
    registry = _PoolRegistry(
        tmp_path, 1, _TEST_POOL_SIZES_BYTES, _TEST_JOURNAL_BYTES, _TEST_DIGEST
    )
    guard = registry._guards[0]
    real_lock = memory_module._try_lock_pool

    def broken_lock(fd, *, exclusive):
        if exclusive:
            raise OSError(errno.EIO, "uncertain lock")
        return real_lock(fd, exclusive=exclusive)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(memory_module, "_try_lock_pool", broken_lock)
            result = registry.drain()
            assert result["state"] == "error"
            assert "uncertain lock" in result["error"]
            assert Path(guard._spec.path).exists()
        assert registry.drain()["state"] == "drained"
    finally:
        registry.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_admin_request_rejects_unbounded_or_nonpositive_timeout(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        service_info("/not/a/socket", timeout_s=timeout)
