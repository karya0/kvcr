# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Q14: restarted Guard reattaches, promotes on pidfd death, serves real UCX."""

import ctypes
import os
import socket
import subprocess
import sys
import time

import pytest
from _kvcr_test_utils import FakePrimaryPinning, _router_hint, _wait_until, free_port
from test_guard_integration import (
    _DIGEST,
    _REAL_NIXL_TIMEOUT_SECONDS,
    _real_nixl_available,
    _real_nixl_refs,
    _real_nixl_regions,
    _zmq_context,  # noqa: F401 - imported pytest fixture
)

from kvcr import KVCR, KVCRBindings
from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWDramOptions
from kvcr.runtime import progress as kvcr_progress
from kvcr.runtime.control_channels import ZmqPeerControlChannel
from kvcr.service.memory import KVCRPoolAttachment
from kvcr.service.node_service import meta_region_bytes
from kvcr.service.protocol import PidfdLiveness, _TierConfig, region_table_digest
from kvcr.service.recovery_journal import RecoveryJournal
from kvcr.service.regions import RegionPool
from kvcr.service.server import _PoolRegistry
from kvcr.types import BlockKey

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux memfd and pidfd")

KEY = b"pre-restart".ljust(36, b"-")  # a 36-byte key, as a vLLM block hash


def test_q14_reattached_guard_serves_pre_restart_key_over_real_nixl(
    tmp_path,
    monkeypatch,
    _zmq_context,  # noqa: F811 - pytest fixture
):
    # This is a strict gate: unavailable native support is not evidence of PASS.
    assert _real_nixl_available(), "Q14 requires a runnable native NIXL UCX backend"
    monkeypatch.setattr(
        kvcr_progress, "_STARTUP_TIMEOUT_SECONDS", _REAL_NIXL_TIMEOUT_SECONDS
    )
    page = os.sysconf("SC_PAGE_SIZE")
    journal_bytes = 2 * page
    size = 2 * page  # the pool's header page, then one block
    pool = RegionPool.create(size, meta_bytes=meta_region_bytes(journal_bytes))
    pool.grow(size)
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    registry = _PoolRegistry(
        pool_dir,
        1,
        (page,),
        journal_bytes,
        _DIGEST,
        startup_timeout_seconds=_REAL_NIXL_TIMEOUT_SECONDS,
        v2_slots=True,
    )
    guard = registry._guards[0]
    attachment = KVCRPoolAttachment.attach_meta_fd(os.dup(pool.meta_fd), guard._spec)
    RecoveryJournal(attachment).reset()  # first writer initialization, before restart
    slot_name, generation = "tenant/lineage/0", 7

    def digest(*, backing_only=False):
        return region_table_digest(
            pool.regions, [pool.fd], slot_name, 0 if backing_only else generation
        )

    attachment.set_layout_digest(digest())
    listener = socket.create_server(("127.0.0.1", 0))
    endpoint = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
    # W1 owns the pool/meta/listener fds. It writes complete bytes, then commits
    # the pool header entry, then waits while a fresh Guard adopts the same fds.
    child_code = f"""
import ctypes, mmap, os, sys, time
from kvcr.service.g2_index import pool_headers, pool_ranges
fd, meta, listener, size, page = map(int, sys.argv[1:])
os.pwrite(fd, b"R" * page, page)
region = mmap.mmap(fd, size)
address = ctypes.addressof(ctypes.c_char.from_buffer(region))
layouts = [("", page)]
ranges = pool_ranges(size, layouts, {len(KEY)})
headers = pool_headers(address, ranges, layouts, {len(KEY)}, initialize=True)
headers[""].publish(0, {KEY!r}, 0)
print("ready", flush=True)
time.sleep(120)
"""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
            *(str(v) for v in (pool.fd, pool.meta_fd, listener.fileno(), size, page)),
        ],
        pass_fds=(pool.fd, pool.meta_fd, listener.fileno()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    target = None
    returned_fd = None
    lease = None
    try:
        marker = child.stdout.readline().strip()
        assert marker == "ready", (
            child.stderr.read() if child.poll() is not None else marker
        )
        lease = PidfdLiveness(os.pidfd_open(child.pid))
        lease.incarnation = "a" * 32
        _, _, returned_fd, returned_lease = guard.reattach(
            lease,
            _TierConfig([("", page)], None),
            listener.getsockname(),
            (
                pool.address(pool.regions[0]),
                size,
                digest,
                slot_name,
                1,
                pool.meta_fd,
                len(KEY),
            ),
            listener.fileno(),
        )
        assert returned_lease is lease
        assert guard._pool_lease.current is lease
        assert guard._core is not None
        assert guard._recovery.attachment.layout_digest() == digest()
        assert list(guard._recovery.headers) == [""]
        # Reattach starts with an empty mirror.
        assert guard._recovery.mirror._records == {}
        child.kill()
        child.wait(timeout=5)
        _wait_until(lambda: guard._serving, timeout=_REAL_NIXL_TIMEOUT_SECONDS)
        assert BlockKey(KEY) in guard._core._block_record_map

        destination = ctypes.create_string_buffer(page)
        pinning = FakePrimaryPinning()
        target = KVCR(
            KVCRConfig(
                nixl_agent_name="h3-target",
                pool_layouts=[("", page)],
                nixl_listen_port=0,
                operation_timeout_ms=_REAL_NIXL_TIMEOUT_SECONDS * 1000,
                abandon_timeout_ms=_REAL_NIXL_TIMEOUT_SECONDS * 2000,
            ),
            KVCRBindings(
                pinning.request_pin,
                pinning.poll_pin_results,
                pinning.release_pin,
                framework_control=ZmqPeerControlChannel(
                    "127.0.0.1", free_port(), "127.0.0.1"
                ),
            ),
            KVCRBackendConfigs(
                framework_regions=_real_nixl_regions(
                    ctypes.addressof(destination), [("", page)]
                ),
                remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
            ),
        )
        key = BlockKey(KEY)
        target.submit_hint(_router_hint(endpoint), request_id="h3-restarted-guard")
        operation = target.deliver(
            {key: _real_nixl_refs("h3-target", [("", page)])},
            request_id="h3-restarted-guard",
        )
        results = {}
        deadline = time.monotonic() + _REAL_NIXL_TIMEOUT_SECONDS
        while not results and time.monotonic() < deadline:
            results = dict(target.poll_completed())
            time.sleep(0.01)
        assert results, "reattached Guard never answered the real peer"
        assert results[operation][key].success
        assert destination.raw == b"R" * page
    finally:
        if target is not None:
            target.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdout.close()
        child.stderr.close()
        if returned_fd is not None:
            os.close(returned_fd)
        registry.close()
        if lease is not None:
            lease.close()
        listener.close()
        attachment.close()
        pool.close()
