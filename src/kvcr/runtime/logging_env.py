# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Apply ``KVCR_LOG_LEVEL`` to the ``kvcr`` logger of the current process."""

import logging
import os
import sys

_MARKER = "_kvcr_env_handler"
_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def apply_env_log_level() -> None:
    """Honor ``KVCR_LOG_LEVEL`` in embedded processes (e.g. vLLM EngineCore).

    Embedding frameworks often install no handler for ``kvcr`` in child
    processes, so DEBUG/INFO records would fall to ``logging.lastResort``.
    When the variable names a valid level, set it on the ``kvcr`` logger and,
    if the logger has no handlers, attach one stderr handler and stop
    propagation. Idempotent. The standalone service does not construct
    ``KVCR`` and configures the root logger itself, so it is unaffected.
    """
    raw = os.environ.get("KVCR_LOG_LEVEL")
    if not raw:
        return
    lg = logging.getLogger("kvcr")
    name = raw.strip().upper()
    mapping = getattr(logging, "getLevelNamesMapping", None)
    level = mapping().get(name) if mapping else getattr(logging, name, None)
    if not isinstance(level, int):
        lg.warning("ignoring invalid KVCR_LOG_LEVEL=%r", raw)
        return
    lg.setLevel(level)
    if lg.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    setattr(handler, _MARKER, True)
    lg.addHandler(handler)
    lg.propagate = False
