"""Review artifact: copy to KVCR tests/unit after applying the paired patch.

Real service/claim sockets, mappings and process deaths; fake and real NIXL.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from _kvcr_test_utils import _mem_descriptor, _poll_until, _wait_until, free_port
from kvcr.guard import _Guard, _Phase
from kvcr.kvcr_service import _KVCRService, _parse_args
from kvcr.types import BlockKey, CacheTier, QueryStatus
from test_guard_integration import (
    _DIGEST,
    _await_marker,
    _FileBackedNixlAgent,
    _make_kvcr,
    _real_nixl_available,
    _zmq_context,  # noqa: F401 - isolate the real service's ZMQ context
)


def _forbid_recovery(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("D-prime invoked a recovery operation")

    for target in (
        "kvcr.guard.RecoveryJournal",
        "kvcr.recovery_journal.RecoveryJournal",
        "kvcr.recovery_journal._attach_journal",
        "kvcr.recovery_journal.read_recovery_snapshot",
        "kvcr.recovery_journal.write_recovery_snapshot",
        "kvcr.guard.write_recovery_snapshot",
        "kvcr.memory.KVCRPoolAttachment.release_snapshot_region",
    ):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(_Guard, "_serve", forbidden)


def _cold_client(socket_path, port, real_nixl=False):
    framework = None
    agent = None
    if real_nixl:
        framework = ctypes.create_string_buffer(2 * os.sysconf("SC_PAGE_SIZE"))
    else:
        agent = _FileBackedNixlAgent()
        agent.state = "DONE"
    client = _make_kvcr(
        socket_path, None, port, "dprime-test", agent=agent, framework=framework
    )
    client._test_framework = framework
    return client


def _healthy_roundtrip(client):
    key = BlockKey(b"prior-generation")
    assert client.query([key]) == [(QueryStatus.MISS, None)]
    dram = client._core._local_dram
    assert len(dram._free_slots[""]) == 1
    page = os.sysconf("SC_PAGE_SIZE")
    framework = client._test_framework
    source = framework if framework is not None else ctypes.create_string_buffer(page)
    source_address = ctypes.addressof(source)
    ctypes.memset(source_address, ord("A"), page)
    op = client.deposit({key: [_mem_descriptor(source_address, page)]})
    assert dict(_poll_until(client, bool))[op][key].success
    assert client.query([key]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    target = ctypes.create_string_buffer(page)
    target_address = (
        source_address + page if framework is not None else ctypes.addressof(target)
    )
    op = client.deliver({key: [_mem_descriptor(target_address, page)]})
    assert dict(_poll_until(client, bool))[op][key].success
    if framework is not None:
        assert ctypes.string_at(target_address, page) == b"A" * page


def _child(socket_path, port, real_nixl):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _forbid_recovery(monkeypatch)
        client = _cold_client(socket_path, int(port), real_nixl == "True")
        _healthy_roundtrip(client)
        print("ready", flush=True)
        time.sleep(60)


def test_disable_flag_is_opt_in():
    args = [
        "--socket-path",
        "/tmp/review.sock",
        "--pool-dir",
        "/tmp",
        "--guard-count",
        "1",
        "--pool-sizes-gb",
        "1",
        "--compatibility-digest",
        _DIGEST,
    ]
    assert not _parse_args(args).disable_resiliency
    assert _parse_args(args + ["--disable-resiliency"]).disable_resiliency


@pytest.mark.usefixtures("_zmq_context")
@pytest.mark.parametrize("real_nixl", [False, True], ids=["fake-nixl", "real-nixl"])
def test_service_pool_is_cold_after_two_crashes_and_graceful_release(
    tmp_path, monkeypatch, real_nixl
):
    if real_nixl and not _real_nixl_available():
        pytest.skip("no runnable NIXL agent; real data-plane gate remains open")
    _forbid_recovery(monkeypatch)
    page = os.sysconf("SC_PAGE_SIZE")
    service = _KVCRService(
        tmp_path / "service.sock",
        tmp_path,
        guard_count=1,
        pool_sizes_bytes=(page,),
        compatibility_digest=_DIGEST,
        journal_bytes=2 * page,
        resiliency_enabled=False,
    )
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    guard = service._registry._guards[0]
    original_spec = guard._spec
    assert original_spec.resiliency_enabled is False
    port = free_port()
    children = []
    try:
        for generation in range(2):
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from test_service_no_resiliency import _child; "
                    "_child(*sys.argv[2:])",
                    str(Path(__file__).parent),
                    str(service.socket_path),
                    str(port),
                    str(real_nixl),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            children.append(child)
            _await_marker(child, "ready", 60 if real_nixl else 10)
            incarnation = guard._pool_lease.current.incarnation
            assert incarnation is not None
            assert guard._recovery._journal is None
            child.kill()
            child.wait(timeout=5)
            _wait_until(
                lambda: guard._phase is _Phase.IDLE or guard._failure, timeout=5
            )
            assert guard._failure is None
            assert guard._pool_lease.current is None
            assert guard._pool_lease.listener is None
            assert guard._core is None and not guard._serving
            assert guard._recovery.mirror is None
            assert incarnation in guard.dead_incarnations
            assert len(guard.dead_incarnations) == generation + 1
            assert guard._spec == original_spec
            assert (
                Path(original_spec.path).stat().st_size == original_spec.mapping_bytes
            )
            with open(original_spec.path, "rb") as pool:
                assert pool.read(original_spec.journal_bytes) == bytes(
                    original_spec.journal_bytes
                )

        for _ in range(2):
            client = _cold_client(str(service.socket_path), port, real_nixl)
            try:
                assert client._pool_hold._attachment._spec == original_spec
                assert set(client._pool_hold._dead_incarnations) == set(
                    guard.dead_incarnations
                )
                _healthy_roundtrip(client)
            finally:
                client.close()
            assert guard._phase is _Phase.IDLE
            assert guard._pool_lease.listener is None
            assert guard._recovery.mirror is None
            assert (
                Path(original_spec.path).stat().st_size == original_spec.mapping_bytes
            )
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()
        service.shutdown()
        thread.join(timeout=5)
        service.close()
        assert not thread.is_alive()
    assert not Path(original_spec.path).exists()
