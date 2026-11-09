# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""H3 slot admission gates with a supervisor clock, no GPU."""

import json
import sys
import uuid

import pytest

if sys.platform != "linux":
    pytest.skip("memfd slots require Linux", allow_module_level=True)

from test_node_service_v2 import (
    DIGEST,
    HEADROOM,
    ME,
    PAGE,
    _table,
)
from test_node_service_v2 import (
    pools as pools,
)
from test_node_service_v2 import (
    tmp_path as tmp_path,
)

from kvcr.service.node_service import SlotError, SlotTable
from kvcr.service.protocol import IDENTITY_MISMATCH, SLOT_HELD, SLOT_STALE


def _claim(table, claim_id, generation=None, peer=ME):
    return table.claim_with_generation(
        "t",
        "l",
        0,
        total=2 * PAGE,
        minimum=2 * PAGE,
        maximum=8 * PAGE,
        digest=DIGEST,
        claim_id=claim_id,
        peer=peer,
        owner_alive=lambda: True,
        claim_generation=generation,
        waiter_alive=lambda: True,
    )


def _restart(tmp_path, pools, monkeypatch, *, ttl=10000, peer=ME):
    table, api = _table(tmp_path)
    table.lease_ttl_ms = ttl
    claim_id = uuid.uuid4().hex
    original, generation = _claim(table, claim_id, peer=peer)
    now = [0.0]
    restarted = SlotTable(
        table.broker,
        HEADROOM + 64 * PAGE,
        require_pod_identity=False,
        clock=lambda: now[0],
    )
    slot = restarted.get(original.name)
    pools.append(slot.pool)
    monkeypatch.setattr(restarted, "_check_holder", lambda slot: False)
    restarted.start_quarantine()
    return restarted, slot, claim_id, generation, now, api


def _refused(reason, table, claim_id, generation=None, peer=ME):
    with pytest.raises(SlotError) as raised:
        _claim(table, claim_id, generation, peer)
    assert raised.value.reason == reason


def test_q1_q3_quarantine_rejects_foreign_and_stale_pairs(tmp_path, pools, monkeypatch):
    table, slot, claim_id, generation, now, _ = _restart(tmp_path, pools, monkeypatch)
    now[0] = 9.999
    _refused(SLOT_HELD, table, "b")
    _refused(SLOT_STALE, table, claim_id, generation - 1)
    _refused(SLOT_HELD, table, uuid.uuid4().hex, generation)
    assert slot.held and slot.quarantined
    saved = json.loads((tmp_path / "slot-table.json").read_text())["slots"][slot.name]
    assert (saved["claim_id"], saved["claim_generation"]) == (claim_id, generation)


def test_q4_expiry_bumps_once_and_persists_before_successor(
    tmp_path, pools, monkeypatch
):
    table, slot, claim_id, generation, now, _ = _restart(tmp_path, pools, monkeypatch)
    now[0] = 10.0
    successor, new_generation = _claim(table, "b")
    assert successor is slot and new_generation == generation + 1
    assert claim_id in table.dead_incarnations
    saved = json.loads((tmp_path / "slot-table.json").read_text())
    assert saved["slots"][slot.name]["claim_generation"] == generation + 1
    assert saved["generations"][slot.name] == generation + 1
    _refused(SLOT_STALE, table, claim_id, generation)


def test_q5_pod_gone_ends_quarantine_early(tmp_path, pools, monkeypatch):
    peer = (*ME, "worker-uid")
    table, slot, claim_id, generation, now, api = _restart(
        tmp_path,
        pools,
        monkeypatch,
        peer=peer,
    )
    monkeypatch.setattr(api, "get_by_uid", lambda uid: None, raising=False)
    now[0] = 0.01
    assert _claim(table, "b")[1] == generation + 1
    assert claim_id in table.dead_incarnations and not slot.quarantined


def test_q6_successors_keep_fifo(tmp_path, pools, monkeypatch):
    table, slot, _, generation, now, _ = _restart(tmp_path, pools, monkeypatch)
    _refused(SLOT_HELD, table, "b")
    _refused(SLOT_HELD, table, "c")
    now[0] = 10.0
    _refused(SLOT_HELD, table, "c")
    assert _claim(table, "b")[1] == generation + 1
    table.vacate(slot.name, ME, generation + 1)
    assert _claim(table, "c")[1] == generation + 2


def test_q8_q13_fallback_id_is_not_persisted(tmp_path, pools, monkeypatch):
    table, _ = _table(tmp_path)
    slot, _ = _claim(table, "127.0.0.1:5555")
    saved = json.loads((tmp_path / "slot-table.json").read_text())
    assert saved["slots"][slot.name]["claim_id"] is None
    restarted = SlotTable(
        table.broker, HEADROOM + 64 * PAGE, require_pod_identity=False
    )
    restored = restarted.get(slot.name)
    pools.append(restored.pool)
    assert not restored.held
    monkeypatch.setattr(restarted, "_check_holder", lambda slot: False)
    assert _claim(restarted, "b")[0] is restored


def test_q9_larger_persisted_ttl_wins(tmp_path, pools, monkeypatch):
    table, slot, claim_id, generation, now, _ = _restart(
        tmp_path,
        pools,
        monkeypatch,
        ttl=20000,
    )
    assert table.lease_ttl_ms == 20000
    now[0] = 10.0
    _refused(SLOT_HELD, table, "b")
    assert _claim(table, claim_id, generation)[1] == generation
    assert slot.held and not slot.quarantined


def test_q11_quarantine_uses_ttl_only_without_pidfd(tmp_path, pools, monkeypatch):
    """Restart loses pidfd: hold through G, fence/bump before W2; live pidfd wins."""
    peer = (*ME, "worker-uid")
    table, slot, claim_id, generation, now, api = _restart(
        tmp_path,
        pools,
        monkeypatch,
        peer=peer,
    )
    alive = [True]
    monkeypatch.setattr(
        api,
        "get_by_uid",
        lambda uid: {"metadata": {"uid": uid}} if alive[0] else None,
        raising=False,
    )
    now[0] = 9.0
    assert _claim(table, claim_id, generation, peer)[1] == generation
    now[0] = 100.0
    _refused(SLOT_HELD, table, "b")  # live pidfd never expires on wall time
    table._abort_claim(slot, False, generation)
    _refused(SLOT_STALE, table, claim_id, generation, peer)
    assert slot.claim_generation == generation + 1
    # A separate restart exercises pod-gone while G still runs.
    other = tmp_path / "other"
    other.mkdir()
    table2, slot2, old, gen2, clock2, api2 = _restart(
        other, pools, monkeypatch, peer=peer
    )
    monkeypatch.setattr(api2, "get_by_uid", lambda uid: None, raising=False)
    clock2[0] = 1.0
    assert _claim(table2, "b")[1] == gen2 + 1
    assert old in table2.dead_incarnations and not slot2.quarantined


def test_q12_abort_reconnect_requarantines(tmp_path, pools, monkeypatch):
    table, slot, claim_id, generation, _, _ = _restart(tmp_path, pools, monkeypatch)
    assert _claim(table, claim_id, generation)[1] == generation
    table._abort_claim(slot, False, generation)
    assert slot.quarantined and slot.held and slot.peer is None
    _refused(SLOT_HELD, table, "b")
    assert _claim(table, claim_id, generation)[1] == generation


def test_q13_restored_identity_and_reconnect_bypasses_waiter(
    tmp_path, pools, monkeypatch
):
    table, slot, claim_id, generation, _, _ = _restart(tmp_path, pools, monkeypatch)
    for action in (
        lambda: table._owned(slot.name, ME),
        lambda: table.vacate(slot.name, ME, generation),
    ):
        with pytest.raises(SlotError) as raised:
            action()
        assert raised.value.reason == IDENTITY_MISMATCH
    _refused(SLOT_HELD, table, "b")
    assert _claim(table, claim_id, generation)[0] is slot
    assert table.waiters(slot.name) == ["b"]
