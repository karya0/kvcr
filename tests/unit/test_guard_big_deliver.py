# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A promoted Guard serves one large deliver over real UCX/NIXL on host memory.

Opt-in measurement, skipped unless KVCR_BIG_DELIVER_BLOCKS is set:

  KVCR_BIG_DELIVER_BLOCKS       blocks in the one deliver op (required)
  KVCR_BIG_DELIVER_BLOCK_BYTES  bytes per block (default 51380224)
  KVCR_BIG_DELIVER_GUARD_CPUS   taskset -c list for the kvcr-service process

Three processes: the kvcr-service CLI (its Guard is the source, optionally
pinned with taskset), a primary child that fills the pool and is killed, and
this process as the target. Guard and target both run the KVCRConfig default
operation and abandon timeouts; nothing here overrides them.
"""

import ctypes
import logging
import os
import subprocess
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from _kvcr_test_utils import (
    FakePrimaryPinning,
    _mem_descriptor,
    _poll_until,
    _router_hint,
    free_port,
)
from test_guard_integration import (
    _DIGEST,
    _REAL_NIXL_TIMEOUT_SECONDS,
    _await_marker,
    _make_kvcr,
    _real_nixl_available,
    _real_nixl_regions,
)

from kvcr import KVCR, KVCRBindings
from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWDramOptions
from kvcr.runtime import progress as kvcr_progress
from kvcr.runtime.control_channels import ZmqPeerControlChannel
from kvcr.types import BlockKey

_DEFAULT_BLOCK_BYTES = 51_380_224
_GB = 1 << 30


def _fill(address: int, block_bytes: int, index: int) -> None:
    """A per-block byte with the block index stamped at both ends."""
    ctypes.memset(address, (index * 37 + 11) % 256, block_bytes)
    stamp = index.to_bytes(8, "little")
    ctypes.memmove(address, stamp, 8)
    ctypes.memmove(address + block_bytes - 8, stamp, 8)


def _check(address: int, block_bytes: int, index: int) -> None:
    stamp = index.to_bytes(8, "little")
    fill = bytes([(index * 37 + 11) % 256])
    assert ctypes.string_at(address, 8) == stamp, index
    assert ctypes.string_at(address + block_bytes - 8, 8) == stamp, index
    assert ctypes.string_at(address + block_bytes // 2, 1) == fill, index


def _big_deliver_primary_child(
    socket_path: str, control_port: str, blocks: str, block_bytes: str
) -> None:
    """Claim the pool, deposit every block one op at a time, hold until killed."""
    kvcr_progress._STARTUP_TIMEOUT_SECONDS = _REAL_NIXL_TIMEOUT_SECONDS
    size = int(block_bytes)
    framework = ctypes.create_string_buffer(size)
    kvcr = _make_kvcr(
        socket_path,
        None,
        control_port,
        "big-primary",
        framework=framework,
        pool_layouts=[("", size)],
    )
    for index in range(int(blocks)):
        _fill(ctypes.addressof(framework), size, index)
        key = BlockKey(b"big-%d" % index)
        operation = kvcr.deposit({key: [_mem_descriptor(end_point_name="big-primary")]})
        completed = _poll_until(kvcr, bool, timeout=_REAL_NIXL_TIMEOUT_SECONDS)
        assert dict(completed)[operation][key].success, index
    print("ready", flush=True)
    time.sleep(3600)


def _wait_for_log(path: Path, needle: str, process: subprocess.Popen) -> str:
    deadline = time.monotonic() + _REAL_NIXL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for line in path.read_text(errors="replace").splitlines():
            if needle in line:
                return line
        if process.poll() is not None:
            break
        time.sleep(0.05)
    tail = "\n".join(path.read_text(errors="replace").splitlines()[-40:])
    pytest.fail(f"service log never showed {needle!r}:\n{tail}")


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_promoted_guard_serves_one_big_deliver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if "KVCR_BIG_DELIVER_BLOCKS" not in os.environ:
        pytest.skip("set KVCR_BIG_DELIVER_BLOCKS to run the big-deliver measurement")
    if not _real_nixl_available():
        pytest.skip("no runnable NIXL agent on this machine")
    monkeypatch.setattr(
        kvcr_progress, "_STARTUP_TIMEOUT_SECONDS", _REAL_NIXL_TIMEOUT_SECONDS
    )
    blocks = int(os.environ["KVCR_BIG_DELIVER_BLOCKS"])
    block_bytes = int(
        os.environ.get("KVCR_BIG_DELIVER_BLOCK_BYTES", _DEFAULT_BLOCK_BYTES)
    )
    guard_cpus = os.environ.get("KVCR_BIG_DELIVER_GUARD_CPUS")
    page = os.sysconf("SC_PAGE_SIZE")
    pool_bytes = -(-blocks * block_bytes // page) * page
    with localcontext() as context:
        context.prec = 60
        pool_gb = str(Decimal(pool_bytes) / Decimal(_GB))

    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    socket_path = tmp_path / "service.sock"
    service_log = tmp_path / "service.log"
    command = [
        sys.executable,
        "-c",
        "from kvcr.service.server import main; main()",
        "--socket-path",
        str(socket_path),
        "--pool-dir",
        str(pool_dir),
        "--guard-count",
        "1",
        "--pool-sizes-gb",
        pool_gb,
        "--compatibility-digest",
        _DIGEST,
        "--disable-hugepages",
        "--startup-timeout-seconds",
        str(_REAL_NIXL_TIMEOUT_SECONDS * 2),
    ]
    if guard_cpus:
        command = ["taskset", "-c", guard_cpus, *command]
    control_port = free_port()
    with open(service_log, "w") as log:
        service = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    primary = None
    target = None
    try:
        _wait_for_log(service_log, "KVCR service ready", service)
        guard_affinity = next(
            line.split(":", 1)[1].strip()
            for line in Path(f"/proc/{service.pid}/status").read_text().splitlines()
            if line.startswith("Cpus_allowed_list:")
        )
        primary = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from test_guard_big_deliver import _big_deliver_primary_child; "
                "_big_deliver_primary_child(*sys.argv[2:])",
                str(Path(__file__).parent),
                str(socket_path),
                str(control_port),
                str(blocks),
                str(block_bytes),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _await_marker(primary, "ready", _REAL_NIXL_TIMEOUT_SECONDS * 4)
        primary.kill()
        primary.wait(timeout=_REAL_NIXL_TIMEOUT_SECONDS)
        promoted = _wait_for_log(service_log, "guard_promoted", service)
        assert f"recovered_blocks={blocks} " in promoted, promoted

        target_memory = ctypes.create_string_buffer(blocks * block_bytes)
        target_pinning = FakePrimaryPinning()
        target = KVCR(
            KVCRConfig(
                nixl_agent_name="big-target",
                pool_layouts=[("", block_bytes)],
                nixl_listen_port=0,
            ),
            KVCRBindings(
                target_pinning.request_pin,
                target_pinning.poll_pin_results,
                target_pinning.release_pin,
                framework_control=ZmqPeerControlChannel(
                    "127.0.0.1", free_port(), "127.0.0.1"
                ),
            ),
            KVCRBackendConfigs(
                framework_regions=_real_nixl_regions(
                    ctypes.addressof(target_memory),
                    [("", block_bytes)],
                    count=blocks,
                ),
                remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
            ),
        )
        guard_defaults = KVCRConfig(nixl_agent_name="", pool_layouts=[])
        keys = [BlockKey(b"big-%d" % index) for index in range(blocks)]
        target.submit_hint(
            _router_hint(f"tcp://127.0.0.1:{control_port}"), request_id="big"
        )
        # The target's per-transfer event is DEBUG; KVCR_LOG_LEVEL may also have
        # detached this logger from the root, so listen on it directly.
        core_logger = logging.getLogger("kvcr.runtime.core")
        events = _Lines()
        previous_level = core_logger.level
        core_logger.addHandler(events)
        core_logger.setLevel(logging.DEBUG)
        try:
            started = time.monotonic()
            operation = target.deliver(
                {
                    key: [_mem_descriptor(index, end_point_name="big-target")]
                    for index, key in enumerate(keys)
                },
                request_id="big",
            )
            results: dict = {}
            deadline = started + target.config.abandon_timeout_ms / 1000 + 30
            while time.monotonic() < deadline and not results:
                results = dict(target.poll_completed())
                time.sleep(0.001)
            elapsed = time.monotonic() - started
        finally:
            core_logger.setLevel(previous_level)
            core_logger.removeHandler(events)
        completed_lines = [
            line for line in events.lines if "target_transfer_completed" in line
        ]
        succeeded = sum(
            result.success for result in results.get(operation, {}).values()
        )
        total = blocks * block_bytes
        # Printed before the asserts so a failed run still reports its numbers.
        print(
            f"\nBIG_DELIVER blocks={blocks} succeeded={succeeded} bytes={total} "
            f"seconds={elapsed:.3f} GB/s={total / elapsed / 1e9:.3f} "
            f"guard_cpus={guard_affinity} "
            f"target_timeouts_ms={target.config.operation_timeout_ms}/"
            f"{target.config.abandon_timeout_ms} "
            f"guard_timeouts_ms={guard_defaults.operation_timeout_ms}/"
            f"{guard_defaults.abandon_timeout_ms} (KVCRConfig defaults)"
        )
        for line in completed_lines:
            print(line)
        assert results, "the Guard never answered the target"
        failed = [key for key in keys if not results[operation][key].success]
        assert not failed, f"{len(failed)}/{blocks} blocks failed; {completed_lines}"
        for index in range(blocks):
            _check(
                ctypes.addressof(target_memory) + index * block_bytes,
                block_bytes,
                index,
            )
        assert any(
            f"blocks={blocks} " in line and "result=success" in line
            for line in completed_lines
        ), completed_lines
    finally:
        if target is not None:
            target.close()
        if primary is not None:
            primary.kill()
            primary.wait(timeout=_REAL_NIXL_TIMEOUT_SECONDS)
        # SIGTERM lets the service release its pools; kill only if it hangs.
        service.terminate()
        try:
            service.wait(timeout=_REAL_NIXL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait(timeout=_REAL_NIXL_TIMEOUT_SECONDS)
        if primary is not None:
            for stream in (primary.stdout, primary.stderr):
                stream.close()
        print(f"service log: {service_log}")
