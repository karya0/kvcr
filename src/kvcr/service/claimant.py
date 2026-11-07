# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic v2 claimant: claims a slot, fills or checks its regions with a pattern.

CPU-only stand-in for a worker, to exercise the node service socket.
``--mode write`` fills every granted region, ``verify`` checks the same
pattern, ``grow-shrink`` writes, grows, checks the original regions are intact,
writes the new region, then shrinks it away. One JSON line per step on stdout.
Exit 0 on success, 2 on a service error, 3 on a pattern mismatch. The claim is
not released unless ``--release`` is given: the slot is vacated when this
process dies, and its pool outlives it.

Run as ``python -m kvcr.service.claimant --socket <path> --tenant T --lineage L``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import mmap
import os
import signal
import struct
import sys
import uuid
from typing import Any

from kvcr.runtime.control_channels import FramedConnection
from kvcr.service.protocol import (
    _CLAIM_RESPONSE_DECODER,
    _PROTOCOL_VERSION,
    _RESIZE_RESPONSE_DECODER,
    _ClaimV2,
    _Grow,
    _Release,
    _response_fd_counts,
    _send_release,
    _Shrink,
    _TierConfig,
)

_STRIDE = 1 << 20
_HEADER = struct.Struct("<QQ")  # magic, layout_generation at the start of a region
_MAGIC = 0x4B5643525632  # "KVCRV2"
_GIB = 1 << 30
_SEED = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1


class _Failure(Exception):
    def __init__(self, code: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.code, self.reason = code, reason


def _step(step: str, **fields: Any) -> None:
    print(json.dumps({"step": step, **fields}), flush=True)


def _block(region: int, offset: int, size: int) -> bytes:
    """Expected bytes at ``offset`` of a region: its seed xor the offset, repeated."""
    word = ((region + 1) * _SEED ^ offset) & _MASK
    return (struct.pack("<Q", word) * (size // 8 + 1))[:size]


def _write(view: mmap.mmap, region: int, size: int, generation: int) -> None:
    for offset in range(0, size, _STRIDE):
        view[offset : offset + min(_STRIDE, size - offset)] = _block(
            region, offset, min(_STRIDE, size - offset)
        )
    view[: _HEADER.size] = _HEADER.pack(_MAGIC, generation)


def _verify(view: mmap.mmap, region: int, size: int) -> int:
    """The header's layout_generation; _Failure at the first mismatching byte."""
    magic, generation = _HEADER.unpack(view[: _HEADER.size])
    if magic != _MAGIC:
        raise _Failure(3, "mismatch", f"region {region} offset 0: bad header")
    for offset in range(0, size, _STRIDE):
        length = min(_STRIDE, size - offset)
        expected = _block(region, offset, length)
        if offset == 0:
            expected = view[: _HEADER.size] + expected[_HEADER.size :]
        actual = view[offset : offset + length]
        if actual != expected:
            at = next(i for i in range(length) if actual[i] != expected[i])
            raise _Failure(3, "mismatch", f"region {region} offset {offset + at}")
    return generation


def _map(fds: list[int], region: Any) -> mmap.mmap:
    return mmap.mmap(
        fds[region.fd_index or 0],
        region.size,
        mmap.MAP_SHARED,
        mmap.PROT_READ | mmap.PROT_WRITE,
        offset=region.offset,
    )


def _resize(socket_path: str, request: Any) -> Any:
    channel = FramedConnection.connect(socket_path)
    try:
        channel.send(request)
        response = channel.receive(_RESIZE_RESPONSE_DECODER)
    finally:
        channel.close()
    if hasattr(response, "message"):
        raise _Failure(2, getattr(response, "reason", "error"), response.message)
    return response


def _tier_config(text: str) -> _TierConfig:
    layouts = [(n, int(b)) for n, b in (p.split(":") for p in text.split(","))]
    return _TierConfig(layouts, None)


def run(args: argparse.Namespace) -> None:
    request = _ClaimV2(
        args.guard_index,
        args.compatibility_digest,
        _tier_config(args.tier_config),
        args.control_host,
        args.control_port,
        _PROTOCOL_VERSION,
        args.tenant,
        args.lineage,
        uuid.uuid4().hex,
        total_bytes=args.total_bytes,
        minimum_bytes=args.minimum_bytes,
        maximum_bytes=args.maximum_bytes,
    )
    connection = FramedConnection.connect(args.socket)
    # connect() leaves its 60 s timeout; a queued claim waits up to the
    # service's 180 s ensure, so the grant is awaited with a longer one (200 s).
    connection._connection.settimeout(args.receive_timeout_s)
    try:
        # The worker pidfd lets the service watch this process, as the v1 client does.
        pidfd = os.pidfd_open(os.getpid())
        try:
            connection.send_with_fd(request, pidfd)
        finally:
            os.close(pidfd)
        grant, fds = connection.receive_with_fd(
            _CLAIM_RESPONSE_DECODER, _response_fd_counts
        )
        if hasattr(grant, "message"):
            raise _Failure(2, getattr(grant, "reason", "error"), grant.message)
        # fds: pool, control listener (kept open until exit), then region memfds.
        region_fds = fds[2:]
        _step(
            "granted",
            slot_name=grant.slot_name,
            regions=len(grant.regions),
            bytes=sum(r.size for r in grant.regions),
            layout_generation=grant.layout_generation,
            claim_generation=grant.claim_generation,
        )
        views = {r.index: (_map(region_fds, r), r) for r in grant.regions}
        _mode(args, grant, region_fds, views)
        if args.release:
            _send_release(connection)
            _step("released")
    except BaseException:
        # Bare EOF frees nothing: the lease and slot would stay held until this
        # process dies. Say the claim was never served; best effort, no reply awaited.
        with contextlib.suppress(BaseException):
            connection.send(_Release(_PROTOCOL_VERSION, False))
        raise
    finally:
        connection.close()


def _mode(args: argparse.Namespace, grant: Any, fds: list[int], views: dict) -> None:
    sizes = sum(r.size for _v, r in views.values())
    if args.mode == "verify":
        generations = {i: _verify(v, i, r.size) for i, (v, r) in views.items()}
        _step(
            "verified",
            regions=len(views),
            bytes=sizes,
            header_generations=sorted(set(generations.values())),
        )
        return
    for index, (view, region) in views.items():
        _write(view, index, region.size, grant.layout_generation)
    _step("written", regions=len(views), bytes=sizes)
    if args.mode == "write":
        return
    grown = _resize(
        args.socket, _Grow(_PROTOCOL_VERSION, grant.slot_name, args.grow_bytes)
    )
    end = max(r.offset + r.size for _v, r in views.values())
    if grown.layout_generation != grant.layout_generation + 1 or any(
        r.offset < end for r in grown.regions
    ):
        raise _Failure(3, "mismatch", "grown layout moved or skipped a generation")
    for index, (view, region) in views.items():
        _verify(view, index, region.size)
    added = {r.index: (_map(fds, r), r) for r in grown.regions}
    for index, (view, region) in added.items():
        _write(view, index, region.size, grown.layout_generation)
    _step(
        "grown",
        regions=[r.index for r in grown.regions],
        bytes=sum(r.size for r in grown.regions),
        layout_generation=grown.layout_generation,
    )
    for view, _region in added.values():
        view.close()  # a region must be unmapped before it is shrunk away
    shrunk = _resize(
        args.socket, _Shrink(_PROTOCOL_VERSION, grant.slot_name, tuple(added))
    )
    if shrunk.layout_generation != grant.layout_generation + 2:
        raise _Failure(3, "mismatch", "layout_generation did not increment twice")
    _step(
        "shrunk",
        region_indices=list(shrunk.region_indices),
        layout_generation=shrunk.layout_generation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=os.environ.get("KVCR_SERVICE_SOCKET"))
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--minimum-bytes", type=int, default=4 * _GIB)
    parser.add_argument("--maximum-bytes", type=int, default=16 * _GIB)
    parser.add_argument("--total-bytes", type=int, default=8 * _GIB)
    parser.add_argument("--grow-bytes", type=int, default=4 * _GIB)
    parser.add_argument("--guard-index", type=int, default=0)
    parser.add_argument("--compatibility-digest", default="model-v1")
    parser.add_argument("--tier-config", default="pool0:1024", help="NAME:BLOCK,...")
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=5555)
    parser.add_argument("--receive-timeout-s", type=float, default=200)
    parser.add_argument(
        "--mode", choices=("write", "verify", "grow-shrink"), default="write"
    )
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args(argv)
    if not args.socket:
        parser.error("--socket or KVCR_SERVICE_SOCKET is required")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # release before dying
    try:
        run(args)
    except _Failure as failure:
        _step("error", reason=failure.reason, message=str(failure))
        return failure.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
