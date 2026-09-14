"""Opt-in logging tests reuse existing service lifecycle scenarios."""

import json
import logging
from collections import Counter, deque
from types import SimpleNamespace

import pytest
import test_guard_integration as integration
import test_kvcr_remote_target as remote
import test_service_no_resiliency as cold
from kvcr import diagnostics
from test_guard_integration import _zmq_context, live_service  # noqa: F401


def rows(caplog):
    return [
        json.loads(r.getMessage().split("KVCR_DIAGNOSTICS ", 1)[1])
        for r in caplog.records
        if r.name == "kvcr.diagnostics"
    ]


def enable(monkeypatch, caplog):
    monkeypatch.setattr(diagnostics, "ENABLED", True)
    monkeypatch.setenv("KVCR_DIAGNOSTICS", "1")
    caplog.set_level(logging.INFO, logger="kvcr.diagnostics")


def test_disabled_does_not_read_clock_or_state(monkeypatch, caplog):
    monkeypatch.setattr(diagnostics, "ENABLED", False)

    def forbidden():
        raise AssertionError("disabled diagnostics read clock")

    monkeypatch.setattr(diagnostics.time, "monotonic_ns", forbidden)
    diagnostics.event("disabled")
    diagnostics.sample_g2(None)
    assert rows(caplog) == []


def test_keyset_digest_unambiguous_and_order_independent():
    assert diagnostics.keyset_digest([b"a", b"b", b"a"]) == diagnostics.keyset_digest(
        [b"b", b"a"]
    )
    assert diagnostics.keyset_digest([b"ab", b"c"]) != diagnostics.keyset_digest(
        [b"a", b"bc"]
    )


def test_occupancy_is_bounded_and_rate_limited(monkeypatch, caplog):
    enable(monkeypatch, caplog)
    tick = [10]
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: tick[0])
    dram = SimpleNamespace(
        _pools={"g2": (123, 4096, 1024)},
        _free_slots={"g2": deque([2, 3])},
        _evictable_slots=Counter(g2=1),
    )
    core = SimpleNamespace(
        _local_dram=dram, config=SimpleNamespace(nixl_agent_name="test")
    )
    diagnostics.sample_g2(core)
    diagnostics.sample_g2(core)
    tick[0] += 5_000_000_000
    diagnostics.sample_g2(core)
    assert len(rows(caplog)) == 2
    pool = rows(caplog)[0]["pools"][0]
    assert pool["allocated_bytes"] == 2048
    assert pool["free_slots"] == 2
    assert "epoch_ns" in rows(caplog)[0]


@pytest.mark.usefixtures("_zmq_context")
def test_existing_guard_lifecycle_event_order(tmp_path, monkeypatch, request, caplog):
    enable(monkeypatch, caplog)
    integration.test_replacement_primary_takes_the_cache_back_from_a_guard(
        tmp_path, monkeypatch, request.getfixturevalue("live_service")
    )
    names = [row["event"] for row in rows(caplog)]
    assert names.index("primary_death_detected") < names.index("guard_serving_started")
    assert names.index("guard_close_started") < names.index("guard_close_completed")
    assert names.index("guard_close_completed") < names.index(
        "guard_handback_completed"
    )
    assert "snapshot_written" in names
    assert "pool_claim_granted" in names
    assert "g2_occupancy" in names
    for index, name in enumerate(names):
        if name == "guard_handback_completed":
            closed = max(i for i in range(index) if names[i] == "guard_close_completed")
            assert any(
                n in ("snapshot_empty", "snapshot_written")
                for n in names[closed + 1 : index]
            )


@pytest.mark.usefixtures("_zmq_context")
def test_existing_cold_lifecycle_has_no_guard_events(tmp_path, monkeypatch, caplog):
    enable(monkeypatch, caplog)
    cold.test_service_pool_is_cold_after_two_crashes_and_graceful_release(
        tmp_path, monkeypatch, False
    )
    names = [row["event"] for row in rows(caplog)]
    assert names.count("primary_death_detected") == 2
    assert names.count("cold_reclaim") == 2
    assert "guard_serving_started" not in names
    assert "snapshot_written" not in names
    for row in rows(caplog):
        assert row["schema_version"] == 1 and row["pid"] > 0
        assert row["component"] == "kvcr"


def test_existing_partial_transfer_logs_joinable_outcomes(monkeypatch, caplog):
    enable(monkeypatch, caplog)
    remote.test_remote_framework_dram_transfers_available_keys(False, (0,))
    events = rows(caplog)
    target = next(r for r in events if r["event"] == "remote_target_completed")
    source = next(r for r in events if r["event"] == "remote_source_completed")
    assert target["request_id"] == "req"
    assert target["requested_keys"] == 3 and target["completed_keys"] == 2
    assert target["outcome"] == "partial"
    assert source["key_set_sha256"] == target["completed_key_set_sha256"]
    assert (
        source["completed_descriptor_bytes"]
        == target["completed_descriptor_bytes"]
        == 32
    )
    assert target["op_handle"] == source["op_handle"]


def test_existing_refusal_logs_reason_unknown(monkeypatch, caplog):
    enable(monkeypatch, caplog)
    remote.test_only_a_refusal_from_this_operation_s_source_finishes_it()
    refusals = [r for r in rows(caplog) if r["event"] == "remote_write_refused"]
    assert len(refusals) == 1
    assert refusals[0]["source_endpoint"] == "tcp://source:1"
    assert refusals[0]["cause"] == "peer_refusal_reason_not_sent"
