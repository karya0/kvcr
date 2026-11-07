# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""v2 holder: a Guaranteed pod that faults in a slot's pool so the kubelet charges it.

It connects to the supervisor's private socket, receives one JSON layout line
(``regions`` + ``layout_generation``) with the region memfd(s) over SCM_RIGHTS,
maps each region MAP_SHARED, touches every page, reports
``{"touched_bytes": N, "layout_generation": g}`` as one JSON line, then sleeps
until SIGTERM with the mappings kept so the pages stay resident. A later layout
line (after a Grow) maps and touches the new regions and reports again.

Run as ``python -m kvcr.service.holder --socket <path>``.
"""

from __future__ import annotations

import argparse
import array
import errno
import json
import mmap
import os
import select
import signal
import socket
import threading
import time
from typing import Any

_MAX_FDS = 8
_MAX_LINE = 1 << 20
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _touch(view: mmap.mmap, size: int) -> None:
    """Fault every page in without changing its contents."""
    try:
        view.madvise(_MADV_POPULATE_WRITE, 0, size)
        return
    except (OSError, ValueError):
        pass
    for offset in range(0, size, mmap.PAGESIZE):
        view[offset] = view[offset]


def _receive(connection: socket.socket) -> tuple[dict[str, Any], list[int]] | None:
    """One JSON line and its fds; None on EOF."""
    data = b""
    fds = array.array("i")
    while not data.endswith(b"\n"):
        chunk, ancillary, _flags, _ = connection.recvmsg(
            65536, socket.CMSG_SPACE(fds.itemsize * _MAX_FDS), socket.MSG_CMSG_CLOEXEC
        )
        for level, kind, payload in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(payload[: len(payload) - len(payload) % fds.itemsize])
        if not chunk:
            for fd in fds:
                os.close(fd)
            return None
        data += chunk
        if len(data) > _MAX_LINE:
            raise ValueError("holder layout line too large")
    return json.loads(data), list(fds)


def handle_layout(
    connection: socket.socket, mappings: dict[int, tuple[mmap.mmap, int]]
) -> bool:
    """Receive one layout, map and touch new regions, report. False on EOF.

    ``mappings`` maps region index to (mmap, size) and is kept by the caller so
    the pages stay resident; regions absent from the layout are unmapped.
    """
    received = _receive(connection)
    if received is None:
        return False
    layout, fds = received
    try:
        wanted = {region["index"]: region for region in layout["regions"]}
        for index in set(mappings) - set(wanted):
            mappings.pop(index)[0].close()
        for index, region in wanted.items():
            if index in mappings:
                continue
            view = mmap.mmap(
                fds[region.get("fd_index") or 0],
                region["size"],
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=region["offset"],
            )
            mappings[index] = (view, region["size"])
            _touch(view, region["size"])
    finally:
        for fd in fds:
            os.close(fd)
    report = {
        "touched_bytes": sum(size for _view, size in mappings.values()),
        "layout_generation": layout["layout_generation"],
    }
    connection.sendall(json.dumps(report).encode() + b"\n")
    return True


def connect(path: str, timeout_s: float) -> socket.socket:
    """Connect, retrying until the supervisor has bound its socket."""
    deadline = time.monotonic() + timeout_s
    while True:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(path)
            return connection
        except OSError as error:
            connection.close()
            if error.errno not in (errno.ENOENT, errno.ECONNREFUSED) or (
                time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=os.environ.get("KVCR_HOLDER_SOCKET"))
    parser.add_argument("--connect-timeout-s", type=float, default=180)
    args = parser.parse_args()
    if not args.socket:
        parser.error("--socket or KVCR_HOLDER_SOCKET is required")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    mappings: dict[int, tuple[mmap.mmap, int]] = {}
    connection = connect(args.socket, args.connect_timeout_s)
    listening = True
    while not stop.is_set():
        if listening and select.select([connection], [], [], 1.0)[0]:
            listening = handle_layout(connection, mappings)
        elif not listening:
            stop.wait(1.0)


if __name__ == "__main__":
    main()
