# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""v2 holder: a Guaranteed pod that faults in a slot's pool so the kubelet charges it.

It connects to the supervisor's private socket, receives one JSON layout line
(``regions`` + ``layout_generation``, plus ``meta``: the slot's journal and snapshot
memfd) with the memfd(s) over SCM_RIGHTS, maps each MAP_SHARED, touches every page,
reports ``{"touched_bytes": N, "layout_generation": g}`` (plus ``meta_bytes`` when
there is a meta memfd) as one JSON line, then sleeps
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
import sys
import threading
import time
from typing import Any

_MAX_FDS = 8
_META = -1  # the meta memfd's key among the mappings; region indices are >= 0
_MAX_LINE = 1 << 20
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"holder": event, **fields}), file=sys.stderr, flush=True)


def _touch(view: mmap.mmap, size: int, index: int = 0) -> None:
    """Fault every page in without changing its contents."""
    _log("populate_start", region=index, bytes=size)
    started = time.monotonic()
    try:
        view.madvise(_MADV_POPULATE_WRITE, 0, size)
        _log(
            "populate_end",
            region=index,
            bytes=size,
            elapsed_s=round(time.monotonic() - started, 3),
        )
        return
    except (OSError, ValueError):
        _log("populate_fallback", region=index, bytes=size)
    for offset in range(0, size, mmap.PAGESIZE):
        view[offset] = view[offset]
    _log(
        "populate_end",
        region=index,
        bytes=size,
        elapsed_s=round(time.monotonic() - started, 3),
    )


def receive_line_with_fds(
    connection: socket.socket,
) -> tuple[dict[str, Any], list[int]] | None:
    """One JSON line and its fds; None on EOF."""
    data = b""
    fds = array.array("i")
    try:
        while not data.endswith(b"\n"):
            chunk, ancillary, _flags, _ = connection.recvmsg(
                65536,
                socket.CMSG_SPACE(fds.itemsize * _MAX_FDS),
                socket.MSG_CMSG_CLOEXEC,
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
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


def handle_layout(
    connection: socket.socket,
    mappings: dict[int, tuple[mmap.mmap, int]],
    reset: bool = False,
    held: dict[str, Any] | None = None,
) -> bool:
    """Receive one layout, map and touch new regions, report. False on EOF.

    ``mappings`` maps region index (``_META`` for the meta memfd) to (mmap, size)
    and is kept by the caller so the pages stay resident; regions absent from the
    layout are unmapped. With ``reset`` (the first layout after a reconnect) every
    old mapping is dropped first: a restarted supervisor feeds a new memfd, so old
    mappings are stale. With ``held``, fd identity decides which mappings survive.
    """
    received = receive_line_with_fds(connection)
    if received is None:
        return False
    layout, fds = received
    try:
        if held is not None:
            previous = held.get("layout") or {"regions": []}
            old_regions = {r["index"]: r for r in previous["regions"]}
            if previous.get("meta"):
                old_regions[_META] = {"offset": 0, **previous["meta"]}
            new_regions = {r["index"]: r for r in layout["regions"]}
            if layout.get("meta"):
                new_regions[_META] = {"offset": 0, **layout["meta"]}
            old_fds = held.get("fds", [])
            for index in list(mappings):
                old, new = old_regions[index], new_regions.get(index)
                if (
                    new != old
                    or os.fstat(old_fds[old.get("fd_index") or 0])[:2]
                    != os.fstat(fds[new.get("fd_index") or 0])[:2]
                ):
                    mappings.pop(index)[0].close()
            for i, fd in enumerate(fds):
                if i < len(old_fds) and os.fstat(fd)[:2] == os.fstat(old_fds[i])[:2]:
                    os.close(fd)
                    fds[i] = old_fds[i]
                elif i < len(old_fds):
                    os.close(old_fds[i])
            for fd in old_fds[len(fds) :]:
                os.close(fd)
            held.update(layout=layout, fds=fds)
        elif reset:
            for view, _size in mappings.values():
                view.close()
            mappings.clear()
        wanted = {region["index"]: region for region in layout["regions"]}
        meta = layout.get("meta")  # the slot's journal and snapshot memfd
        if meta:
            wanted[_META] = {"offset": 0, **meta}
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
            _touch(view, region["size"], index)
    finally:
        if held is None:
            for fd in fds:
                os.close(fd)
    report = {
        "touched_bytes": sum(
            size for index, (_view, size) in mappings.items() if index != _META
        ),
        "layout_generation": layout["layout_generation"],
    }
    if _META in mappings:
        report["meta_bytes"] = mappings[_META][1]
    connection.sendall(json.dumps(report).encode() + b"\n")
    return True


def connect(
    path: str, timeout_s: float, stop: threading.Event | None = None
) -> socket.socket:
    """Connect, retrying until the supervisor has bound its socket.

    With ``stop`` the retry has no deadline and backs off up to 5 s, returning
    only when connected or raising ``InterruptedError`` once ``stop`` is set.
    """
    deadline = time.monotonic() + timeout_s
    delay = 0.2
    while True:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(path)
            return connection
        except OSError as error:
            connection.close()
            if error.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                raise
            if stop is None:
                if time.monotonic() >= deadline:
                    raise
            elif stop.is_set():
                raise InterruptedError from error
        if stop is None:
            time.sleep(0.2)
        else:
            stop.wait(delay)
            delay = min(delay * 2, 5.0)


def serve(
    path: str,
    connect_timeout_s: float,
    stop: threading.Event,
    mappings: dict[int, tuple[mmap.mmap, int]],
) -> None:
    """Hold the mappings; on EOF reconnect with backoff rather than sleeping forever."""
    held: dict[str, Any] = {"layout": None, "fds": []}
    connection = None
    try:
        try:
            connection = connect(path, connect_timeout_s, stop)
        except InterruptedError:
            return
        _log("connected", socket=path)
        hello(connection, held)
        while not stop.is_set():
            if select.select([connection], [], [], 1.0)[0]:
                try:
                    alive = handle_layout(connection, mappings, held=held)
                except OSError:
                    alive = False
                if alive:
                    continue
                connection.close()
                try:
                    connection = connect(path, 0, stop)
                except InterruptedError:
                    return
                hello(connection, held)
    finally:
        if connection is not None:
            connection.close()
        for fd in held["fds"]:
            os.close(fd)


def hello(connection: socket.socket, held: dict[str, Any] | None) -> None:
    """Offer the retained layout and fds before every supervisor feed."""
    socket.send_fds(
        connection,
        [json.dumps({"handback": held["layout"] if held else None}).encode() + b"\n"],
        held["fds"] if held else [],
    )


# Retain the private name used by existing receiver tests.
_receive = receive_line_with_fds


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
    serve(args.socket, args.connect_timeout_s, stop, {})


if __name__ == "__main__":
    main()
