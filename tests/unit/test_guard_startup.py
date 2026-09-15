# SPDX-License-Identifier: Apache-2.0
"""Standby warming must not claim the live primary's endpoint or memory."""

import logging
from unittest.mock import Mock

import msgspec
import pytest
from test_guard import _guard, _tier
from test_progress import _transfer_progress, _TransferAgent

from kvcr import progress as progress_module


def test_claim_warms_once_without_serving_and_close_releases(monkeypatch):
    guard = _guard()
    guard._recovery = Mock(pools=())
    agent = Mock()
    create = Mock(return_value=agent)
    monkeypatch.setattr(progress_module, "nixl_agent", create)
    first, second = Mock(), Mock()
    guard._adopt(first, _tier(16))
    guard._adopt(second, _tier(16))
    create.assert_called_once()
    assert create.call_args.args[0].startswith("KVCR-Warm-")
    assert guard._warm_agent is agent
    assert guard._core is None and not guard._serving
    agent.register_memory.assert_not_called()
    first.initialize.assert_not_called()
    second.initialize.assert_not_called()
    guard._close_resources()
    assert guard._warm_agent is None


@pytest.mark.parametrize("resiliency,backend", [(False, "UCX"), (True, "REMOTE")])
def test_non_ucx_and_non_resilient_claims_do_not_warm(monkeypatch, resiliency, backend):
    guard = _guard()
    guard._spec = msgspec.structs.replace(guard._spec, resiliency_enabled=resiliency)
    guard._recovery = Mock(pools=())
    create = Mock()
    monkeypatch.setattr(progress_module, "nixl_agent", create)
    guard._adopt(Mock(), _tier(16, backend=backend))
    create.assert_not_called()
    guard._close_resources()


def test_prewarm_failure_refuses_claim_without_serving(monkeypatch):
    guard = _guard()
    guard._recovery = Mock(pools=())
    monkeypatch.setattr(
        progress_module, "nixl_agent", Mock(side_effect=RuntimeError("UCX failed"))
    )
    control = Mock()
    with pytest.raises(RuntimeError, match="UCX failed"):
        guard._adopt(control, _tier(16))
    control.close.assert_called_once()
    assert guard._configured is None and guard._warm_agent is None
    assert guard._core is None and not guard._serving
    guard._close_resources()


def test_disabled_timing_never_reads_clocks(monkeypatch):
    guard = _guard()
    progress = _transfer_progress(_TransferAgent())
    monkeypatch.setattr("kvcr.guard.logger.isEnabledFor", lambda _: False)
    monkeypatch.setattr("kvcr.progress.logger.isEnabledFor", lambda _: False)
    clock = Mock(side_effect=AssertionError("disabled timing read clock"))
    for name in ["time_ns", "monotonic_ns", "thread_time_ns"]:
        monkeypatch.setattr("time." + name, clock)
    guard._log_promotion_stage("test")
    progress._log_startup_stage("test")
    clock.assert_not_called()


def test_startup_failure_logs_phase_without_false_ready(monkeypatch, caplog):
    progress = _transfer_progress(_TransferAgent())
    monkeypatch.setattr(
        progress,
        "_register_memory_regions",
        Mock(side_effect=RuntimeError("register failed")),
    )
    with caplog.at_level(logging.DEBUG, logger="kvcr.progress"):
        with pytest.raises(RuntimeError, match="register failed"):
            progress.start()
        with pytest.raises(RuntimeError, match="register failed"):
            progress.close()
    messages = [r.getMessage() for r in caplog.records]
    assert any("stage=memory_registering " in m for m in messages)
    assert any("stage=failed:memory_registration " in m for m in messages)
    assert not any(
        "stage=memory_registered " in m or "stage=ready " in m for m in messages
    )
    assert all("monotonic_ns=" in m and "thread_cpu_ns=" in m for m in messages)


def test_death_observation_reports_lock_wait(monkeypatch, caplog):
    import time
    from contextlib import contextmanager

    from kvcr.guard import _Phase

    guard = _guard()
    guard._phase = _Phase.PRIMARY
    guard._pool_lease.current = Mock(incarnation="holder")
    monkeypatch.setattr(guard._pool_lease, "poll_pidfd", lambda _: None)

    @contextmanager
    def slow_lock():
        time.sleep(0.12)
        yield

    guard._phase_lock = slow_lock()
    with caplog.at_level(logging.DEBUG, logger="kvcr.guard"):
        guard._observe_holder()
    assert any("stage=death_check_lock_slow " in r.getMessage() for r in caplog.records)
    assert not any("stage=death_observed " in r.getMessage() for r in caplog.records)
