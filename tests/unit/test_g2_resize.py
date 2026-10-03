"""Online allocator/registration contracts; physical pages tested on Linux."""

import ctypes
import hashlib
import threading
import time
from contextlib import closing
from unittest.mock import Mock

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _router_hint,
    _wait_until,
)

from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.core import _BlockRecord
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.recovery_journal import install_recovery_records
from kvcr.types import BlockKey, QueryStatus


def _controller(
    agent=None, chunk=32, percent=0, pressure=None, initial=64, telemetry=False
):
    memory = ctypes.create_string_buffer(64)
    callback = Mock()
    events = []
    controller = _new_kvcr(
        agent or FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(
            nixl_agent_name="resize",
            pool_layouts=[("", 16)],
            g2_resize_granularity_bytes=chunk,
            capacity_low_watermark_percent=percent,
            enable_telemetry=telemetry,
        ),
        local_dram=LocalDramOptions([("", ctypes.addressof(memory), initial)]),
        inventory_sink=events.append,
        capacity_needed_callback=pressure,
    )
    controller._core._resize_g2_memory = callback
    return controller, memory, callback, events


def _store(controller, memory, keys, no_evict=False):
    source = ctypes.create_string_buffer(b"x" * (16 * len(keys)))
    op = controller.deposit(
        {
            key: [_mem_descriptor(ctypes.addressof(source) + 16 * i)]
            for i, key in enumerate(keys)
        },
        no_evict=no_evict,
    )
    results = dict(_poll_until(controller, lambda done: op in dict(done)))[op]
    assert all(result.success for result in results.values())
    return results


def test_shrink_grow_retains_address_inventory_and_default_progress():
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, callback, events = _controller(agent, telemetry=True)
    keys = [BlockKey(bytes([i])) for i in range(4)]
    with closing(controller):
        _store(controller, memory, keys)
        head = controller._core._local_dram._descriptor("", 0).addr
        assert controller.resize_g2("", 32)
        assert controller.query(keys)[0][0] is QueryStatus.HIT
        assert controller.query(keys)[2][0] is QueryStatus.MISS
        assert controller._core._local_dram._descriptor("", 0).addr == head
        callback.assert_called_once_with("", 64, 32)
        assert events[-1].removed and set(events[-1].keys) == set(keys[2:])
        assert len(agent.deregistered) == 1
        assert controller.resize_g2("", 64)
        assert list(controller._core._local_dram._free_slots[""]) == [2, 3]
        _store(controller, memory, [BlockKey(b"new")])
        assert controller.resize_g2("", 32)
        assert controller.query([BlockKey(b"new")])[0][0] is QueryStatus.MISS
        waits = [
            value for kind, name, value, labels in controller.get_stats().records
            if kind == "histogram" and name == "kvcr_duration_seconds"
            and labels == ("resize_queue", "success")
        ]
        assert len(waits) == 3 and all(value >= 0 for value in waits)


def test_resize_tracks_recovered_slots():
    controller, _, callback, _ = _controller()
    keys = [BlockKey(b"head"), BlockKey(b"tail")]
    with closing(controller):
        install_recovery_records(
            controller._core,
            {
                key: _BlockRecord(
                    local_dram=_LocalDramResidency([("", slot)], _LocalDramState.READY)
                )
                for key, slot in zip(keys, (0, 3))
            },
        )
        assert controller.resize_g2("", 32)
        assert [status for status, _ in controller.query(keys)] == [
            QueryStatus.HIT,
            QueryStatus.MISS,
        ]
        callback.assert_called_once_with("", 64, 32)


def test_replacement_can_regrow_a_smaller_attached_pool():
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, _, _ = _controller(agent, initial=32)
    with closing(controller):
        # Service attachment restores the reservation ceiling, not active size.
        controller._core._local_dram._reservation_lengths[""] = 64
        assert controller.resize_g2("", 64)
        _store(controller, memory, [BlockKey(bytes([i])) for i in range(4)])
        assert controller.resize_g2("", 32)


def test_resize_advances_one_quantum_and_polls_between_steps():
    controller, _, callback, _ = _controller(chunk=16)
    observed_polls = []
    with closing(controller):
        poll = controller._core._progress._poll

        def tracked_poll(*args):
            observed_polls.append(True)
            return poll(*args)

        controller._core._progress._poll = tracked_poll
        polls_at_steps = []
        callback.side_effect = lambda *args: polls_at_steps.append(len(observed_polls))
        assert controller.resize_g2("", 16)
        assert [call.args for call in callback.call_args_list] == [
            ("", 64, 48),
            ("", 48, 32),
            ("", 32, 16),
        ]
        assert all(b > a for a, b in zip(polls_at_steps, polls_at_steps[1:]))
        callback.reset_mock()
        assert controller.resize_g2("", 64)
        assert [call.args for call in callback.call_args_list] == [
            ("", 16, 32),
            ("", 32, 48),
            ("", 48, 64),
        ]


def test_claimed_tail_refuses_shrink_then_retries():
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, callback, _ = _controller(agent)
    with closing(controller):
        results = _store(
            controller, memory, [BlockKey(bytes([i])) for i in range(4)], True
        )
        assert not controller.resize_g2("", 32)
        callback.assert_not_called()
        controller.release([result.release_handle for result in results.values()])
        assert controller.resize_g2("", 32)


def test_head_transfer_continues_while_tail_is_released():
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, callback, _ = _controller(agent)
    with closing(controller):
        _store(controller, memory, [BlockKey(b"head")])
        destination = ctypes.create_string_buffer(16)
        agent.state = "PROC"
        operation = controller.deliver(
            {BlockKey(b"head"): [_mem_descriptor(ctypes.addressof(destination))]}
        )
        _wait_until(lambda: len(agent.transfers) == 2)
        assert controller.resize_g2("", 32)
        assert list(controller.poll_completed()) == []
        agent.state = "DONE"
        _poll_until(controller, lambda done: operation in dict(done))
        assert destination.raw == b"x" * 16


def test_filling_tail_and_bad_targets_leave_capacity_intact():
    agent = FakeNixlAgent()
    controller, memory, callback, _ = _controller(agent)
    with closing(controller):
        source = ctypes.create_string_buffer(64)
        op = controller.deposit(
            {
                BlockKey(bytes([i])): [
                    _mem_descriptor(ctypes.addressof(source) + 16 * i)
                ]
                for i in range(4)
            }
        )
        _wait_until(lambda: bool(agent.transfers))
        assert not controller.resize_g2("", 32)
        for target in [0, 16, 96, True]:
            with pytest.raises(ValueError):
                controller.resize_g2("", target)
        callback.assert_not_called()
        agent.state = "DONE"
        _poll_until(controller, lambda done: op in dict(done))


def test_growth_registration_failure_keeps_old_capacity():
    agent = FakeNixlAgent()
    controller, memory, callback, _ = _controller(agent)
    with closing(controller):
        assert controller.resize_g2("", 32)
        register = agent.register_memory
        agent.register_memory = Mock(side_effect=RuntimeError("registration failure"))
        with pytest.raises(RuntimeError, match="registration failure"):
            controller.resize_g2("", 64)
        assert controller._core._local_dram._pools[""][1] == 32
        callback.assert_called_with("", 64, 32)
        agent.register_memory = register
        assert controller.resize_g2("", 64)


def test_growth_metadata_failure_never_admits_new_slots():
    agent = FakeNixlAgent()
    controller, memory, callback, _ = _controller(agent)
    with closing(controller):
        assert controller.resize_g2("", 32)
        capture = agent.get_agent_metadata
        agent.get_agent_metadata = Mock(side_effect=RuntimeError("metadata failure"))
        with pytest.raises(RuntimeError, match="metadata failure"):
            controller.resize_g2("", 64)
        assert controller._core._local_dram._pools[""][1] == 32
        assert list(controller._core._local_dram._free_slots[""]) == [0, 1]
        callback.assert_called_with("", 64, 32)
        agent.get_agent_metadata = capture
        assert controller.resize_g2("", 64)


def test_failed_physical_rollback_blocks_resize_but_keeps_serving():
    agent = FakeNixlAgent()
    controller, memory, callback, _ = _controller(agent)
    with closing(controller):
        assert controller.resize_g2("", 32)
        register = agent.register_memory
        agent.register_memory = Mock(side_effect=RuntimeError("registration failure"))
        callback.side_effect = [None, RuntimeError("rollback outcome unknown")]
        with pytest.raises(RuntimeError, match="rollback outcome unknown"):
            controller.resize_g2("", 64)
        agent.register_memory = register
        callback.side_effect = None
        with pytest.raises(RuntimeError, match="blocks further resizing"):
            controller.resize_g2("", 64)
        agent.state = "DONE"
        _store(controller, memory, [BlockKey(b"still-serving")])


def test_failed_shrink_keeps_tail_unallocatable_and_supports_retry():
    agent = FakeNixlAgent()
    controller, memory, callback, _ = _controller(agent)
    with closing(controller):
        deregister = agent.deregister_memory
        agent.deregister_memory = Mock(return_value=False)
        with pytest.raises(RuntimeError, match="retiring registration"):
            controller.resize_g2("", 32)
        assert list(controller._core._local_dram._free_slots[""]) == [0, 1]
        callback.assert_not_called()
        with pytest.raises(RuntimeError, match="unfinished shrink"):
            controller.resize_g2("", 64)
        agent.deregister_memory = deregister
        assert controller.resize_g2("", 32)
        callback.assert_called_once_with("", 64, 32)


def test_failed_growth_cleanup_never_releases_registered_pages():
    agent = FakeNixlAgent()
    controller, memory, callback, _ = _controller(agent, chunk=16)
    with closing(controller):
        assert controller.resize_g2("", 16)
        capture, deregister = agent.get_agent_metadata, agent.deregister_memory
        agent.get_agent_metadata = Mock(side_effect=RuntimeError("metadata failed"))
        agent.deregister_memory = Mock(return_value=False)
        with pytest.raises(RuntimeError, match="retiring registration"):
            controller.resize_g2("", 64)
        callback.assert_called_with("", 16, 32)  # No unsafe physical rollback.
        assert controller._core._local_dram._pools[""][1] == 16
        with pytest.raises(RuntimeError, match="blocks further resizing"):
            controller.resize_g2("", 64)
        agent.get_agent_metadata, agent.deregister_memory = capture, deregister
        agent.state = "DONE"
        _store(controller, memory, [BlockKey(b"still-serving")])


def test_multistep_resize_does_not_trip_watchdog_and_logs_stage_costs(caplog):
    controller, _, callback, _ = _controller(chunk=16)
    callback.side_effect = lambda *args: time.sleep(0.4)
    with closing(controller), caplog.at_level("INFO", logger="kvcr"):
        started = time.monotonic()
        assert controller.resize_g2("", 16)
        assert time.monotonic() - started >= 1.2
        assert controller._core._progress.call(
            controller._core._remote_fw_dram._dangling_ops.check_source_progress
        )
        assert "stage=backing" in caplog.text
        assert "stage=deregister" in caplog.text
        assert "steps=3 result=success" in caplog.text


def test_later_busy_chunk_reports_partial_capacity_and_allows_retry():
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, callback, _ = _controller(agent, chunk=16)
    with closing(controller):
        keys = [BlockKey(bytes([i])) for i in range(4)]
        results = _store(controller, memory, keys, True)
        controller.release([results[keys[3]].release_handle])
        assert not controller.resize_g2("", 16)
        assert controller._core._local_dram._pools[""][1] == 48
        callback.assert_called_once_with("", 64, 48)
        controller.release([results[key].release_handle for key in keys[:3]])
        assert controller.resize_g2("", 16)


def test_failed_intermediate_shrink_retries_original_requested_capacity():
    controller, _, callback, _ = _controller(chunk=16)
    with closing(controller):
        callback.side_effect = [None, OSError("release failed")]
        with pytest.raises(OSError, match="release failed"):
            controller.resize_g2("", 16)
        callback.side_effect = None
        assert controller.resize_g2("", 16)
        assert callback.call_args_list[-2].args == ("", 48, 32)
        assert callback.call_args_list[-1].args == ("", 32, 16)


def test_invalid_pool_does_not_leave_resize_command_locked():
    controller, _, _, _ = _controller()
    with closing(controller):
        with pytest.raises((KeyError, ValueError)):
            controller.resize_g2("unknown", 32)
        assert controller.resize_g2("", 32)


def test_delayed_ack_cannot_suppress_resized_registration_metadata():
    agent = FakeNixlAgent(metadata=b"old")
    controller, _, _, _ = _controller(agent)
    control = controller._core._remote_fw_dram._control
    endpoint = "tcp://source:1"
    with closing(controller):
        controller.submit_hint(_router_hint(endpoint), "before")
        _wait_until(lambda: bool(control.sent))
        agent.metadata = b"new"
        assert controller.resize_g2("", 32)
        ack = {
            "type": "target_metadata_ack",
            "sender_control_endpoint": endpoint,
            "target_metadata_digest": hashlib.sha256(b"old").digest(),
        }
        control.incoming.append(msgspec.msgpack.encode(ack))
        _wait_until(lambda: not control.incoming)
        control.sent.clear()
        controller.submit_hint(_router_hint(endpoint), "after")
        _wait_until(lambda: bool(control.sent))
        message = msgspec.msgpack.decode(control.sent[-1][1])
        assert message["target_agent_metadata"] == b"new"
        assert message["target_metadata_digest"] == hashlib.sha256(b"new").digest()
        ack["target_metadata_digest"] = message["target_metadata_digest"]
        control.incoming.append(msgspec.msgpack.encode(ack))
        _wait_until(
            lambda: endpoint in controller._core._remote_fw_dram._metadata_acked_sources
        )


def test_inventory_callback_reentry_is_rejected_without_deadlock():
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, _, events = _controller(agent)
    with closing(controller):
        _store(controller, memory, [BlockKey(bytes([i])) for i in range(4)])

        def callback(event):
            try:
                controller.resize_g2("", 32)
            except RuntimeError as error:
                events.append(str(error))

        controller._core._inventory_sink_callback = callback
        operator = threading.Thread(target=lambda: controller.resize_g2("", 32))
        operator.start()
        operator.join(timeout=1)
        blocked = operator.is_alive()
        if blocked:  # Release the old implementation's deadlock before teardown.
            queued = controller._core._progress._submissions.get_nowait()
            queued.future.set_exception(RuntimeError("test aborted deadlock"))
            operator.join(timeout=1)
        assert not blocked
        assert "state lock" in events[-1]


def test_fractional_watermark_still_requests_capacity_after_shrink():
    pressure = []
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, _, _ = _controller(
        agent, percent=50.25, pressure=pressure.append
    )
    with closing(controller):
        _store(controller, memory, [BlockKey(b"held-head")], True)
        assert controller.resize_g2("", 32)
        assert pressure == [[("", 2)]]


def test_caller_thread_capacity_callback_cannot_wait_with_state_lock():
    agent = FakeNixlAgent()
    agent.state = "DONE"
    controller, memory, _, events = _controller(agent, percent=50)
    with closing(controller):
        # Avoid hanging the unfixed implementation; detect whether it queues
        # synchronous work while the real caller-side callback holds the lock.
        controller._core._progress.call = Mock(return_value=True)

        def callback(request):
            try:
                controller.resize_g2("", 64)
            except RuntimeError as error:
                events.append(str(error))

        controller._core._capacity_needed_callback = callback
        _store(controller, memory, [BlockKey(bytes([i])) for i in range(3)], True)
        controller._core._progress.call.assert_not_called()
        assert any(isinstance(event, str) and "state lock" in event for event in events)


def test_shrink_metadata_failure_withholds_stale_snapshot_until_retry():
    agent = FakeNixlAgent()
    controller, _, callback, _ = _controller(agent)
    with closing(controller):
        capture = agent.get_agent_metadata
        agent.get_agent_metadata = Mock(side_effect=RuntimeError("metadata failure"))
        with pytest.raises(RuntimeError, match="metadata failure"):
            controller.resize_g2("", 32)
        assert controller._core._progress.nixl_agent_metadata is None
        controller.submit_hint(_router_hint("tcp://new-peer:1"), "failed-refresh")
        controller._core._progress.call(lambda: None)
        assert not controller._core._remote_fw_dram._control.sent
        with pytest.raises(RuntimeError, match="unfinished shrink"):
            controller.resize_g2("", 64)
        agent.get_agent_metadata = capture
        assert controller.resize_g2("", 32)
        assert controller._core._progress.nixl_agent_metadata == agent.metadata
        assert len(agent.deregistered) == 1
        callback.assert_called_once_with("", 64, 32)
