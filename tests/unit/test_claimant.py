# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic v2 claimant against an in-process service (Linux only)."""

import json
import mmap
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

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
from test_node_service_v2 import HEADROOM, pools  # noqa: F401 - fixture

from kvcr.runtime.control_channels import KVCRMsgFramingError
from kvcr.service import claimant
from kvcr.service.node_service import SlotTable

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


def _serve(tmp_path, pages=64):
    info = {
        "state": "ready",
        "pool_specs": _SPECS,
        "compatibility_digest": _TEST_DIGEST,
    }
    broker = make_broker(tmp_path, info=lambda _: info, drain=lambda _: info)
    table = SlotTable(broker, HEADROOM + pages * PAGE, require_pod_identity=False)
    return _running_server(tmp_path), table


def _attach(harness, table):
    harness.server._server.slots = table
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


def test_minimum_above_capacity_exits_2(tmp_path, pools, capsys):  # noqa: F811
    server, table = _serve(tmp_path, pages=3)
    with server as harness:
        _attach(harness, table)
        code, steps = _run(harness, capsys, pages=(8, 4, 8))
        assert code == 2 and steps["error"]["reason"] == "capacity_below_minimum"


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


def test_default_receive_timeout_covers_the_service_ensure(monkeypatch):
    seen = {}

    def capture(self, argv=None, namespace=None):
        seen["actions"] = {a.dest: a for a in self._actions}
        raise SystemExit(0)

    monkeypatch.setattr(claimant.argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        claimant.main([])
    assert seen["actions"]["receive_timeout_s"].default >= 180


@pytest.mark.parametrize("error", [EOFError("eof"), KVCRMsgFramingError("bad frame")])
def test_eof_and_framing_errors_exit_2_as_transport(error, capsys, monkeypatch):
    def boom(args):
        raise error

    monkeypatch.setattr(claimant, "run", boom)
    code = claimant.main(["--socket", "s", "--tenant", "t", "--lineage", "l"])
    assert code == 2
    assert '"reason": "transport"' in capsys.readouterr().out.replace('":"', '": "')
