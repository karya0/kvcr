# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic v2 claimant against an in-process service (Linux only)."""

import json
import logging
import mmap
import socket
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from test_kvcr_service import (  # noqa: F401 - _channels_are_taken is an autouse fixture
    _TEST_DIGEST,
    _channels_are_taken,
    _control_bind,
    _running_server,
)
from test_node_service import _SPECS, make_broker
from test_node_service_v2 import pools  # noqa: F401 - fixture

from kvcr.config import (
    GaaSSettings,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    RemoteFWDramOptions,
)
from kvcr.runtime.control_channels import (
    FramedConnection,
    KVCRMsgFramingError,
    KVCRServiceError,
)
from kvcr.service import claimant
from kvcr.service.node_service import SlotTable
from kvcr.service.protocol import (
    KVCRServiceReasonError,
    _Claim,
    _ClaimV2,
    _Error,
    _ErrorV2,
    _service_error,
)
from kvcr.service.recovery_journal import claim_guarded_pool

PAGE = mmap.PAGESIZE


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="gaas-cl-", dir="/tmp") as directory:
        yield Path(directory)


def _run(harness, capsys, *flags, pages=(4, 2, 8)):
    total, minimum, maximum = pages
    host, port = _control_bind(0)
    code = claimant.main(
        [
            "--socket", str(harness.server.socket_path),
            "--tenant", "t", "--lineage", "l",
            "--compatibility-digest", _TEST_DIGEST,
            "--tier-config", "pool0:1024",
            "--control-host", host, "--control-port", str(port),
            "--total-bytes", str(total * PAGE),
            "--minimum-bytes", str(minimum * PAGE),
            "--maximum-bytes", str(maximum * PAGE),
            "--grow-bytes", str(2 * PAGE),
            "--receive-timeout-s", "10",
            *flags,
        ]
    )  # fmt: skip
    steps = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return code, {step["step"]: step for step in steps}


def _serve(tmp_path, journal_bytes=None, **slot_table_args):
    info = {
        "state": "ready",
        "pool_specs": _SPECS,
        "compatibility_digest": _TEST_DIGEST,
    }
    broker = make_broker(tmp_path, info=lambda _: info, drain=lambda _: info)
    table = SlotTable(broker, require_pod_identity=False, **slot_table_args)
    extra = {} if journal_bytes is None else {"journal_bytes": journal_bytes}
    return _running_server(tmp_path, slots=table, **extra), table


def _attach(harness, table):
    harness.server._server.slots = table
    harness.server._server.registry.claimed_guard(0)  # what a first claim builds
    for guard in harness.server._server.registry._guards.values():
        # A real Guard would register the pool with NIXL, which the host cannot.
        guard._prepare_core = Mock()


def test_write_then_verify_across_runs(tmp_path, pools, capsys):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        code, steps = _run(harness, capsys, "--mode", "write", "--release")
        assert code == 0 and steps["written"]["bytes"] == 4 * PAGE
        code, steps = _run(harness, capsys, "--mode", "verify", "--release")
        assert code == 0 and steps["verified"]["header_generations"] == [1]


def test_grow_shrink_increments_generation_twice(tmp_path, pools, capsys):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        code, steps = _run(harness, capsys, "--mode", "grow-shrink", "--release")
        assert code == 0
        assert steps["granted"]["layout_generation"] == 1
        assert steps["grown"]["layout_generation"] == 2
        assert steps["shrunk"]["layout_generation"] == 3
        code, steps = _run(harness, capsys, "--mode", "verify", "--release")
        assert code == 0 and steps["verified"]["bytes"] == 4 * PAGE


def test_mapping_failure_frees_the_claim_and_exits_2(
    tmp_path,
    pools,  # noqa: F811
    capsys,
    monkeypatch,
):
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)

        def fail(*args, **kwargs):
            raise OSError(12, "injected mmap failure")

        monkeypatch.setattr(claimant, "_map", fail)
        code, steps = _run(harness, capsys)
        assert code == 2 and steps["error"]["reason"] == "transport"
        # This process is still alive, so only the explicit release frees it.
        deadline = time.monotonic() + 5
        while table.get("t/l/0").held and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not table.get("t/l/0").held


def test_receive_timeout_defaults_to_computed_wait(monkeypatch):
    seen = {}

    def capture(self, argv=None, namespace=None):
        seen["actions"] = {a.dest: a for a in self._actions}
        raise SystemExit(0)

    monkeypatch.setattr(claimant.argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        claimant.main([])
    assert seen["actions"]["receive_timeout_s"].default is None


@pytest.mark.parametrize("error", [EOFError("eof"), KVCRMsgFramingError("bad frame")])
def test_eof_and_framing_errors_exit_2_as_transport(error, capsys, monkeypatch):
    def boom(args):
        raise error

    monkeypatch.setattr(claimant, "run", boom)
    code = claimant.main(["--socket", "s", "--tenant", "t", "--lineage", "l"])
    assert code == 2
    assert '"reason": "transport"' in capsys.readouterr().out.replace('":"', '": "')


def test_receive_timeout_default_outlasts_the_server_wait():
    def args(**kw):
        base = {"receive_timeout_s": None, "total_bytes": 0, "maximum_bytes": 0}
        return claimant.argparse.Namespace(**{**base, **kw})

    gib = 1 << 30
    assert claimant._receive_timeout_s(args(maximum_bytes=150 * gib)) == 680
    assert claimant._receive_timeout_s(args(total_bytes=62 * gib)) == 504
    assert (
        claimant._receive_timeout_s(args(maximum_bytes=150 * gib, receive_timeout_s=7))
        == 7
    )


def test_readiness_wait_is_shared_by_service_and_protocol():
    from test_node_service import broker_module

    from kvcr.service.protocol import READINESS_WAIT_S

    assert broker_module.READINESS_WAIT_S == READINESS_WAIT_S == 180.0


def _claim(harness, pages=(4, 2, 8)):
    total, minimum, maximum = pages
    return harness.client.claim(
        0,
        [("pool0", 1024)],
        _TEST_DIGEST,
        _control_bind(0),
        gaas=GaaSSettings("t", "l", total * PAGE, minimum * PAGE, maximum * PAGE),
        receive_timeout_s=10,
    )


def test_client_v2_claim_keeps_the_slot_and_regions(tmp_path, pools):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        hold = _claim(harness)
        try:
            assert hold.slot_name == "t/l/0"
            assert (hold.claim_generation, hold.layout_generation) == (1, 1)
            assert hold.effective_maximum_bytes >= 8 * PAGE
            assert sum(r.size for r in hold.regions) == 4 * PAGE
            # The descriptors: one memfd per distinct fd_index.
            assert len(hold.region_fds) == len({r.fd_index or 0 for r in hold.regions})
            assert hold.guard_endpoint == "%s:%d" % _control_bind(0)
            # The same mapping path as v1: the pool is attached and described.
            assert [name for name, _, _ in hold.local_dram.pools] == ["pool0"]
        finally:
            hold.release()
        assert not table.get("t/l/0").held
        assert hold.region_fds == []


def test_client_without_tenant_and_lineage_sends_the_v1_claim(tmp_path):
    sent = []
    real = FramedConnection.send_with_fd

    def spy(self, message, *fds):
        if isinstance(message, (_Claim, _ClaimV2)):  # the service sends grants too
            sent.append(type(message))
        return real(self, message, *fds)

    with (
        _running_server(tmp_path) as harness,
        patch.object(FramedConnection, "send_with_fd", spy),
    ):
        for guard in harness.server._server.registry._guards.values():
            guard._prepare_core = Mock()  # NIXL registration is not under test
        for gaas in (None, GaaSSettings()):
            hold = harness.client.claim(
                0, [("pool0", 1024)], _TEST_DIGEST, _control_bind(0), gaas=gaas
            )
            assert hold.slot_name is None and hold.regions == ()
            hold.release()
    assert sent == [_Claim, _Claim]


def test_client_v2_claim_is_a_claim_v2_and_v1_server_refuses_it(tmp_path):
    sent = []
    real = FramedConnection.send_with_fd

    def spy(self, message, *fds):
        if isinstance(message, (_Claim, _ClaimV2)):  # the service sends grants too
            sent.append(type(message))
        return real(self, message, *fds)

    with (
        _running_server(tmp_path) as harness,
        patch.object(FramedConnection, "send_with_fd", spy),
    ):
        for guard in harness.server._server.registry._guards.values():
            guard._prepare_core = Mock()  # NIXL registration is not under test
        with pytest.raises(KVCRServiceReasonError) as refused:
            _claim(harness)
    assert sent == [_ClaimV2]
    assert refused.value.reason == "not_implemented"


def test_client_surfaces_the_error_reason(tmp_path, pools):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        hold = _claim(harness)
        try:
            with pytest.raises(KVCRServiceReasonError) as refused:
                _claim(harness)
        finally:
            hold.release()
    assert refused.value.reason == "slot_held"
    assert "slot_held" in str(refused.value)
    assert isinstance(refused.value, KVCRServiceError)


def test_service_error_keeps_v1_errors_plain():
    error = _service_error(_ErrorV2("m", 1, "slot_held"))
    assert (error.reason, error.message) == ("slot_held", "m")
    assert not isinstance(_service_error(_Error("m", 1)), KVCRServiceReasonError)


def test_client_grow_and_shrink_bump_the_layout_generation(tmp_path, pools):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        hold = _claim(harness)
        try:
            grown = harness.client.grow("t/l/0", 2 * PAGE, hold.claim_generation)
            assert grown.layout_generation == 2
            assert sum(r.size for r in grown.regions) == 2 * PAGE
            shrunk = harness.client.shrink(
                "t/l/0",
                tuple(r.index for r in grown.regions),
                hold.claim_generation,
            )
            assert shrunk.layout_generation == 3
            with pytest.raises(KVCRServiceReasonError) as refused:
                harness.client.grow("nobody/x/0", PAGE, 1)
            assert refused.value.reason == "unknown_slot"
        finally:
            hold.release()


def test_claim_guarded_pool_logs_the_v2_grant(tmp_path, pools, caplog):  # noqa: F811
    server, table = _serve(tmp_path)
    host, port = _control_bind(0)
    framework = Mock(control_bind_address=lambda: (host, port))
    with server as harness:
        _attach(harness, table)
        guard_config = KVCRGuardConfig(
            str(harness.server.socket_path),
            0,
            _TEST_DIGEST,
            gaas=GaaSSettings("t", "l", 4 * PAGE, 2 * PAGE, 8 * PAGE),
        )
        with (
            patch("kvcr.service.recovery_journal.read_handback"),
            caplog.at_level(logging.INFO, logger="kvcr.service.recovery_journal"),
        ):
            claimed = claim_guarded_pool(
                KVCRConfig("agent", [("pool0", 1024)]),
                guard_config,
                Mock(framework_control=framework),
                KVCRBackendConfigs(remote_fw_dram=RemoteFWDramOptions()),
            )
        try:
            assert claimed.hold.slot_name == "t/l/0"
        finally:
            claimed.hold.release()
    messages = [r.getMessage() for r in caplog.records]
    messages = [m for m in messages if "gaas claim" in m]
    assert len(messages) == 1
    assert messages[0].startswith(
        "gaas claim granted slot=t/l/0 claim_gen=1 layout_gen=1 effective_max="
    )
    assert messages[0].endswith(f"regions=1 guard={host}:{port}")


def test_invalid_claim_flags_emit_the_error_line_and_exit_2(
    tmp_path, pools, capsys  # noqa: F811
):
    server, table = _serve(tmp_path)
    with server as harness:
        _attach(harness, table)
        code, steps = _run(harness, capsys, "--total-bytes", "0")
    assert code == 2
    assert steps["error"]["reason"] == "error"
    assert "total_bytes" in steps["error"]["message"]


def test_silent_peer_costs_one_receive_timeout_not_two(tmp_path, capsys):
    path = tmp_path / "s.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)  # accepts at the kernel and never answers
    started = time.monotonic()
    code = claimant.main(
        [
            "--socket", str(path), "--tenant", "t", "--lineage", "l",
            "--receive-timeout-s", "1",
        ]
    )  # fmt: skip
    elapsed = time.monotonic() - started
    listener.close()
    steps = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 2 and steps[-1]["reason"] == "timeout"
    assert 1 <= elapsed < 1.8
