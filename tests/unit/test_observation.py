# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import ctypes
import json
import os
import threading
from types import SimpleNamespace

import msgspec
import pytest
from _kvcr_test_utils import FakeNixlAgent, _new_local_kvcr

from kvcr import KVCR
from kvcr.backends.local_dram_pools import _LocalDramResidency, _LocalDramState
from kvcr.runtime import observation
from kvcr.runtime.core import _BlockRecord


@pytest.fixture
def owner():
    data = ctypes.create_string_buffer(256)
    agent = FakeNixlAgent()
    kvcr = _new_local_kvcr(agent, data, 4)
    residency = _LocalDramResidency(
        slots=(("", 0), ("piece", 1)),
        state=_LocalDramState.READY,
        key_layout=("", "piece"),
        pending_labels={"piece"},
        writers=[0, 1],
        pending_deposits=1,
        remote_indices=(1,),
        remote_cancelled=True,
        claim_count=2,
        retire_on_release=True,
    )
    kvcr._core._block_record_map[b"raw-key"] = _BlockRecord(
        local_dram=residency,
        in_flight_ops={("deposit", 1)},
        last_access=12.5,
        access_count=3,
        position=9,
    )
    yield kvcr, data, agent


def fingerprint(kvcr, data):
    core = kvcr._core
    pools = core._local_dram._pools
    return (
        dict(kvcr.__dict__),
        dict(core.__dict__),
        msgspec.to_builtins(core.config),
        msgspec.to_builtins(core._block_record_map),
        copy.deepcopy(pools._free_slots),
        copy.deepcopy(pools._evictable.__dict__),
        dict(core._policy._policy.__dict__),
        bytes(data),
    )


def test_partial_ready_metadata_is_detached_bounded_and_not_a_data_proof(
    owner, monkeypatch
):
    kvcr, data, agent = owner
    core = kvcr._core

    def forbidden(*args, **kwargs):
        pytest.fail("observation invoked a state/native operation")

    for name in ("query", "poll_completed", "_clock"):
        monkeypatch.setattr(core, name, forbidden)
    monkeypatch.setattr(core._progress, "submit", forbidden)
    monkeypatch.setattr(core, "_state_lock", object())
    monkeypatch.setattr(core._local_dram._pools._source_state, "lock", object())
    registered = list(agent.registrations)
    before = fingerprint(kvcr, data)
    result = kvcr.observe_metadata()
    assert kvcr.observe_metadata() == result
    assert result["metadata_complete"] is True
    assert result["global_quiescence"] == "NOT_ESTABLISHED"
    assert result["native_progress"] == "MISSING_NOT_OBSERVED"
    assert result["source_reader_state"] == "MISSING_NOT_OBSERVED"
    assert result["data_proof"] == result["http_target_correlation"] == "MISSING"
    assert result["guarded_backing"] is None
    assert result["local_regions"] == [{"label": "", "length": 256, "slot_stride": 64}]
    entry = result["entries"][0]
    assert entry["key_hex"] == b"raw-key".hex()
    assert entry["request_id"] is None
    assert entry["in_flight_ops"] == 1
    assert entry["local_dram"] == {
        "raw_state": "READY",
        "slots": [["", 0], ["piece", 1]],
        "key_layout": ["", "piece"],
        "pending_labels": ["piece"],
        "writers": [0, 1],
        "pending_deposits": 1,
        "remote_indices": [1],
        "remote_cancelled": True,
        "claim_count": 2,
        "retire_on_release": True,
    }
    # Only detached JSON values are returned; the pool address is absent.
    encoded = json.dumps(result, separators=(",", ":"))
    assert str(ctypes.addressof(data)) not in encoded
    assert len(encoded) <= 1 << 20
    entry["local_dram"]["writers"][1] = 99
    entry["local_dram"]["slots"][0][1] = 99
    assert fingerprint(kvcr, data) == before
    monkeypatch.undo()
    assert agent.registrations == registered
    record = core._block_record_map[b"raw-key"].local_dram
    assert record._layout is record._pool_counts is record._cached_size is None


def test_held_grant_identity_is_metadata_not_fd_revalidation(owner, tmp_path):
    kvcr, data, _ = owner
    with (tmp_path / "journal").open("w+b") as backing:
        backing.write(b"journal cursors and data remain unchanged")
        backing.flush()
        spec = SimpleNamespace(
            pool_id="pool",
            generation="generation",
            device=11,
            inode=22,
            mapping_bytes=4096,
            journal_bytes=1024,
        )
        hold = SimpleNamespace(
            _attachment=SimpleNamespace(
                _spec=spec, _file_descriptor=backing.fileno(), _mapping=data
            )
        )
        kvcr._pool_hold = hold
        before = os.fstat(backing.fileno()), backing.tell(), fingerprint(kvcr, data)
        result = kvcr.observe_metadata()
        assert result["guarded_backing"] == vars(spec)
        assert result["backing_identity_verification"] == (
            "held_grant_metadata_only_no_revalidation"
        )
        assert (
            os.fstat(backing.fileno()),
            backing.tell(),
            fingerprint(kvcr, data),
        ) == before
        backing.seek(0)
        assert backing.read() == b"journal cursors and data remain unchanged"
        kvcr._pool_hold = None  # The synthetic hold owns no lease to release.


@pytest.mark.parametrize(
    "field", ["entries", "bytes", "unicode", "collection", "deadline"]
)
def test_limits_return_explicit_partial_metadata_without_mutation(
    owner, monkeypatch, field
):
    kvcr, data, _ = owner
    options = {}
    if field == "entries":
        options["max_entries"] = 0
    elif field == "bytes":
        kvcr._core._block_record_map[b"x" * 5000] = _BlockRecord()
        options["max_bytes"] = 4096
    elif field == "unicode":
        kvcr._core._block_record_map[b"raw-key"].local_dram.pending_labels = {
            "é" * 1000
        }
        options["max_bytes"] = 4096
    elif field == "collection":
        kvcr._core._block_record_map[b"raw-key"].local_dram.slots = (("", 0),) * 4097
    else:
        times = iter((0, 1))
        monkeypatch.setattr(
            observation, "time", SimpleNamespace(monotonic=lambda: next(times, 1))
        )
    before = fingerprint(kvcr, data)
    result = kvcr.observe_metadata(**options)
    assert result["metadata_complete"] is False
    assert result["truncated"] is True
    assert (
        result["reason"]
        == {
            "entries": "max_entries",
            "bytes": "max_bytes",
            "unicode": "max_bytes",
            "collection": "max_collection_items",
            "deadline": "deadline",
        }[field]
    )
    assert result["omitted_entries"] > 0
    assert len(json.dumps(result, separators=(",", ":"))) <= options.get(
        "max_bytes", 1 << 20
    )
    assert fingerprint(kvcr, data) == before


@pytest.mark.parametrize(
    "options",
    [
        {"max_entries": True},
        {"max_entries": 4097},
        {"max_bytes": 4095},
        {"max_bytes": (4 << 20) + 1},
        {"timeout_s": 0},
        {"timeout_s": float("nan")},
        {"timeout_s": True},
    ],
)
def test_invalid_limits_do_not_change_owner(owner, options):
    kvcr, data, _ = owner
    before = fingerprint(kvcr, data)
    with pytest.raises(ValueError):
        kvcr.observe_metadata(**options)
    assert fingerprint(kvcr, data) == before


def test_unsupported_key_metadata_fails_without_mutation(owner):
    kvcr, data, _ = owner
    kvcr._core._block_record_map["not-a-raw-block-key"] = _BlockRecord()
    before = fingerprint(kvcr, data)
    with pytest.raises(TypeError, match="raw bytes block keys"):
        kvcr.observe_metadata()
    assert fingerprint(kvcr, data) == before


def test_affinity_uninitialized_closed_and_fork_refused(owner, monkeypatch):
    kvcr, _, _ = owner
    errors = []

    def other_owner():
        try:
            kvcr.observe_metadata()
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=other_owner)
    worker.start()
    worker.join(2)
    assert errors == ["metadata observation requires the constructing owner"]
    with pytest.raises(RuntimeError, match="constructing owner"):
        object.__new__(KVCR).observe_metadata()
    with monkeypatch.context() as patch:
        patch.setattr(kvcr, "_metadata_owner_pid", -1)
        with pytest.raises(RuntimeError, match="constructing owner"):
            kvcr.observe_metadata()
    kvcr.close()
    with pytest.raises(RuntimeError, match="inactive"):
        kvcr.observe_metadata()


def test_failed_close_disables_observation_without_restarting(owner, monkeypatch):
    kvcr, _, _ = owner

    def failed_close():
        raise RuntimeError("native close failed")

    with monkeypatch.context() as patch:
        patch.setattr(kvcr._core, "close", failed_close)
        with pytest.raises(RuntimeError, match="native close failed"):
            kvcr.close()
        with pytest.raises(RuntimeError, match="inactive"):
            kvcr.observe_metadata()


def test_source_reader_release_continues_during_metadata_copy(owner, monkeypatch):
    kvcr, _, _ = owner
    pools = kvcr._core._local_dram._pools
    pools._source_state.operations = 1
    copying = threading.Event()
    released = threading.Event()
    original = observation._Budget.copy

    def copy_while_native_releases(self, value):
        copying.set()
        assert released.wait(1), "export gated source-reader release"
        return original(self, value)

    def reader_release():
        assert copying.wait(1)
        pools.release_readers((), (), success=True)
        released.set()

    worker = threading.Thread(target=reader_release)
    worker.start()
    monkeypatch.setattr(observation._Budget, "copy", copy_while_native_releases)
    result = kvcr.observe_metadata(timeout_s=0.25)
    worker.join(2)
    assert not worker.is_alive()
    assert result["metadata_complete"]
    assert pools._source_state.operations == 0
