# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A supervisor restart while a lineage is Retained (its DGD gone), no GPU."""

import copy
import json
import sys
import uuid
from pathlib import Path

import pytest

if sys.platform != "linux":
    pytest.skip("memfd slots require Linux", allow_module_level=True)

from test_node_service_v2 import (
    _lineage_claim,
    _retention_table,
)
from test_node_service_v2 import (
    pools as pools,
)
from test_node_service_v2 import (
    tmp_path as tmp_path,
)

from kvcr.service import node_service
from kvcr.service.node_service import SlotError, SlotTable
from kvcr.service.protocol import INTERNAL, SLOT_HELD, SLOT_STALE


def _restart(table, pools, **kwargs):
    """A new supervisor over the same socket root and slot table."""
    restarted = SlotTable(
        table.broker,
        table.cap_bytes,
        require_pod_identity=False,
        dgd_api=table.dgd_api,
        **kwargs,
    )
    pools.extend(slot.pool for slot in restarted._slots.values())
    return restarted


def _retain(table, dgd, lineage="l"):
    del dgd.crs[lineage]
    table.poll_dgds()
    assert table._lineages[f"t/{lineage}"]["retained_at"] is not None


@pytest.mark.parametrize("died_holding", [False, True])
def test_restart_keeps_a_retained_lineage_and_its_one_rebind(
    tmp_path, pools, died_holding
):
    wall, now = [1000], [0.0]
    clocks = {"wall_clock": lambda: wall[0], "clock": lambda: now[0]}
    table, dgd = _retention_table(tmp_path, **clocks)
    api = table.broker.api
    dgd.put("l", "X")
    # A UUID claim is persisted, so a worker that died holding it comes back
    # quarantined; any other claim id comes back free, as after a vacate.
    claim_id = uuid.uuid4().hex if died_holding else "a"
    slot = _lineage_claim(table, claim_id=claim_id)
    _retain(table, dgd)
    record = copy.deepcopy(table._lineages)
    assert record["t/l"]["retained_at"] == 1000 and not record["t/l"]["rebound"]
    pods = copy.deepcopy(api.pods)
    creates, deletes = len(api.creates), len(api.deletes)
    available = table._available("t", "t/new/0")
    wall[0] += 100
    restarted = _restart(table, pools, **clocks)
    # Still Retained, with the same identity and the original retention clock.
    assert restarted._lineages == record
    restarted.poll_dgds()  # the DGD is still gone: retained_at is not reset
    assert restarted._lineages == record
    again = restarted.get(slot.name)
    assert (again.holder, again.digest) == (slot.holder, slot.digest)
    assert again.pool.regions == slot.pool.regions
    assert again.quarantined is died_holding
    # The holder pod is kept, and the restored slot is charged once.
    assert api.pods == pods
    assert (len(api.creates), len(api.deletes)) == (creates, deletes)
    assert restarted._available("t", "t/new/0") == available
    # The predecessor UID seen again is not a rebind: fail closed.
    dgd.put("l", "X")
    with pytest.raises(SlotError) as raised:
        _lineage_claim(restarted, claim_id="old")
    assert raised.value.reason == INTERNAL
    dgd.put("l", "Y")
    if died_holding:
        restarted.start_quarantine()
        with pytest.raises(SlotError) as raised:
            _lineage_claim(restarted, claim_id="successor")  # inside the lease TTL
        assert raised.value.reason == SLOT_HELD
        now[0] = restarted.lease_ttl_ms / 1000
    # The successor's UID rebinds once, onto the restored slot and pool.
    assert _lineage_claim(restarted, claim_id="successor") is again
    rec = restarted._lineages["t/l"]
    assert (rec["dgd_uid"], rec["rebound"], rec["retained_at"]) == ("Y", True, None)
    assert (len(api.creates), len(api.deletes)) == (creates, deletes)
    if died_holding:
        assert claim_id in restarted.dead_incarnations
    # The spent rebind survives a second restart; dead incarnations do not.
    second = _restart(restarted, pools, **clocks)
    assert second._lineages["t/l"]["rebound"] is True
    assert second.dead_incarnations == set()
    assert _lineage_claim(second, claim_id="successor-again") is second.get(slot.name)
    dgd.put("l", "Z")
    with pytest.raises(SlotError) as raised:
        _lineage_claim(second, claim_id="third")
    assert raised.value.reason == SLOT_STALE


def test_restart_does_not_restart_the_retention_clock(tmp_path, pools):
    wall = [1000]
    clocks = {"wall_clock": lambda: wall[0], "retention_seconds": 600}
    table, dgd = _retention_table(tmp_path, pages=4, **clocks)
    dgd.put("l", "X")
    _lineage_claim(table)
    _retain(table, dgd)
    wall[0] = 1500
    restarted = _restart(table, pools, **clocks)
    stood_down = []
    restarted.stand_down = stood_down.append
    dgd.put("new", "N")
    with pytest.raises(SlotError) as raised:
        _lineage_claim(restarted, "new")  # 500 s of 600: not a reclaim victim
    assert raised.value.reason == SLOT_HELD
    assert restarted.get("t/l/0") is not None and not stood_down
    # 601 s after the original retained_at, 101 s after the restart.
    wall[0] = 1601
    assert _lineage_claim(restarted, "new").name == "t/new/0"
    assert restarted.get("t/l/0") is None and stood_down == ["t/l/0"]
    assert "t/l/0" not in json.loads(Path(restarted._state_path).read_text())["slots"]


def test_retiring_charge_beside_a_retained_lineage_is_counted_once(tmp_path, pools):
    table, dgd = _retention_table(tmp_path, pages=8, wall_clock=lambda: 1000)
    api = table.broker.api
    dgd.put("l", "X")
    _lineage_claim(table)
    _retain(table, dgd)
    dgd.put("r", "R")
    released = _lineage_claim(table, "r")
    pod = "gaas-pool-" + table.broker._key(
        node_service.PoolClaim.parse(released.holder)
    )
    api.keep_deleted = True  # the released holder pod lingers, terminating
    table.ensure_timeout_s = 0.01  # release's wait for the pod ends at once
    table.release(released.name)
    retiring = copy.deepcopy(table._retiring)
    assert list(retiring) == [pod]
    available = table._available("t", "t/new/0")
    restarted = _restart(table, pools, wall_clock=lambda: 1000)
    assert restarted._retiring == retiring
    assert restarted._available("t", "t/new/0") == available
    dgd.put("new", "N")
    with pytest.raises(SlotError) as raised:
        _lineage_claim(restarted, "new")  # fits only if the charge were lost
    assert raised.value.reason == SLOT_HELD
    del api.pods[pod]  # the holder finished terminating
    assert _lineage_claim(restarted, "new").name == "t/new/0"
    assert not restarted._retiring and restarted.get("t/l/0") is not None
    assert restarted._lineages["t/l"]["retained_at"] == 1000
    assert json.loads(Path(restarted._state_path).read_text())["retiring"] == {}
