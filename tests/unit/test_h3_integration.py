# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Q14: restarted Guard reattaches, promotes on pidfd death, serves real UCX."""

import ctypes
import mmap
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
from kvcr.service.g2_index import G2Index
from kvcr.service.memory import (
    KVCRPoolAttachment,
    index_region_bytes,
    index_slots,
    meta_region_bytes,
)
from kvcr.service.protocol import PidfdLiveness, _TierConfig, region_table_digest
from kvcr.service.recovery_journal import RecoveryJournal
from kvcr.service.regions import RegionPool
from kvcr.service.server import _PoolRegistry
from kvcr.types import BlockKey

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux memfd and pidfd")


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
    base, irb = meta_region_bytes(journal_bytes), index_region_bytes(page)
    pool = RegionPool.create(page, meta_bytes=base + irb)
    pool.grow(page)
    with mmap.mmap(pool.meta_fd, irb, offset=base) as index_map:
        G2Index.initialize(index_map, index_slots(page))
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
    # W1 owns the pool/meta/listener fds. It writes complete bytes, then publishes
    # the G2 index entry, then waits while a fresh Guard adopts the same fds.
    child_code = """
import mmap, os, sys, time
from kvcr.service.g2_index import G2Index
fd, meta, listener, base, size, page = map(int, sys.argv[1:])
os.pwrite(fd, b"R" * page, 0)
region = mmap.mmap(meta, size, offset=base)
index = G2Index(region)
parts = [("pool", 0)]
assert index.reserve(b"pre-restart", parts)
list(index.store_steps(b"pre-restart", parts, 1))
print("ready", flush=True)
time.sleep(120)
"""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
            *(
                str(v)
                for v in (pool.fd, pool.meta_fd, listener.fileno(), base, irb, page)
            ),
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
            _TierConfig([("pool", page)], None),
            listener.getsockname(),
            (pool.address(pool.regions[0]), page, digest, slot_name, 1, pool.meta_fd),
            listener.fileno(),
        )
        assert returned_lease is lease
        assert guard._pool_lease.current is lease
        assert guard._core is not None
        assert guard._recovery.attachment.layout_digest() == digest()
        assert BlockKey(b"pre-restart") in guard._recovery.mirror._records
        child.kill()
        child.wait(timeout=5)
        _wait_until(lambda: guard._serving, timeout=_REAL_NIXL_TIMEOUT_SECONDS)
        assert BlockKey(b"pre-restart") in guard._core._block_record_map

        destination = ctypes.create_string_buffer(page)
        pinning = FakePrimaryPinning()
        target = KVCR(
            KVCRConfig(
                nixl_agent_name="h3-target",
                pool_layouts=[("pool", page)],
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
                    ctypes.addressof(destination), [("pool", page)]
                ),
                remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
            ),
        )
        key = BlockKey(b"pre-restart")
        target.submit_hint(_router_hint(endpoint), request_id="h3-restarted-guard")
        operation = target.deliver(
            {key: _real_nixl_refs("h3-target", [("pool", page)])},
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
