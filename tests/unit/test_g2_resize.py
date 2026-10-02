"""Online allocator/registration contracts; physical pages tested on Linux."""

import ctypes
from contextlib import closing
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _wait_until,
)

from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.types import BlockKey, QueryStatus


def _controller(agent=None, chunk=32):
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
        ),
        local_dram=LocalDramOptions([("", ctypes.addressof(memory), 64)]),
        inventory_sink=events.append,
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
    controller, memory, callback, events = _controller(agent)
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
        register, deregister = agent.register_memory, agent.deregister_memory
        agent.register_memory = Mock(side_effect=[99, RuntimeError("register failed")])
        agent.deregister_memory = Mock(return_value=False)
        with pytest.raises(RuntimeError, match="retiring registration"):
            controller.resize_g2("", 64)
        callback.assert_called_with("", 16, 64)  # No unsafe physical rollback.
        assert controller._core._local_dram._pools[""][1] == 16
        with pytest.raises(RuntimeError, match="blocks further resizing"):
            controller.resize_g2("", 64)
        agent.register_memory, agent.deregister_memory = register, deregister
        agent.state = "DONE"
        _store(controller, memory, [BlockKey(b"still-serving")])
