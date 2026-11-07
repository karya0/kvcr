# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import logging
import subprocess
import sys

from kvcr.runtime.logging_env import apply_env_log_level


def _saved():
    lg = logging.getLogger("kvcr")
    return lg, lg.level, list(lg.handlers), lg.propagate


def _restore(state):
    lg, level, handlers, propagate = state
    lg.setLevel(level)
    lg.handlers[:] = handlers
    lg.propagate = propagate


def test_level_applied_handler_once(monkeypatch):
    state = _saved()
    try:
        state[0].handlers[:] = []
        monkeypatch.setenv("KVCR_LOG_LEVEL", "debug")
        apply_env_log_level()
        apply_env_log_level()
        lg = state[0]
        assert lg.level == logging.DEBUG
        assert len(lg.handlers) == 1
        assert lg.propagate is False
    finally:
        _restore(state)


def test_invalid_value_ignored(monkeypatch):
    state = _saved()
    try:
        state[0].handlers[:] = []
        state[0].setLevel(logging.WARNING)
        monkeypatch.setenv("KVCR_LOG_LEVEL", "LOUD")
        apply_env_log_level()
        assert state[0].level == logging.WARNING
        assert state[0].handlers == []
    finally:
        _restore(state)


def test_unset_noop(monkeypatch):
    state = _saved()
    try:
        state[0].handlers[:] = []
        monkeypatch.delenv("KVCR_LOG_LEVEL", raising=False)
        before = state[0].level
        apply_env_log_level()
        assert state[0].level == before
        assert state[0].handlers == []
    finally:
        _restore(state)


def test_child_process_emits_debug(monkeypatch):
    monkeypatch.setenv("KVCR_LOG_LEVEL", "DEBUG")
    code = (
        "import logging;"
        "from kvcr.runtime.logging_env import apply_env_log_level;"
        "apply_env_log_level();"
        "logging.getLogger('kvcr.runtime.core').debug('marker-xyz')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "marker-xyz" in res.stderr
