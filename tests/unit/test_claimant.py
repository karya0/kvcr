# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic v2 claimant against an in-process service (Linux only)."""

import json
import mmap
import sys
import tempfile
from pathlib import Path

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from test_kvcr_service import _TEST_DIGEST, _control_bind, _running_server
from test_node_service import _SPECS, make_broker
from test_node_service_v2 import HEADROOM, pools  # noqa: F401 - fixture

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


def test_write_then_verify_across_runs(tmp_path, pools, capsys):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        harness.server._server.slots = table
        code, steps = _run(harness, capsys, "--mode", "write", "--release")
        assert code == 0 and steps["written"]["bytes"] == 4 * PAGE
        code, steps = _run(harness, capsys, "--mode", "verify", "--release")
        assert code == 0 and steps["verified"]["header_generations"] == [1]


def test_grow_shrink_increments_generation_twice(tmp_path, pools, capsys):  # noqa: F811
    server, table = _serve(tmp_path)
    with server as harness:
        harness.server._server.slots = table
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
        harness.server._server.slots = table
        code, steps = _run(harness, capsys, pages=(8, 4, 8))
        assert code == 2 and steps["error"]["reason"] == "capacity_below_minimum"
