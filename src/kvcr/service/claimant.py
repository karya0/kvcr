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
import socket
import struct
import sys
from typing import Any

import msgspec

from kvcr.config import GaaSSettings
from kvcr.runtime.control_channels import (
    KVCRMsgFramingError,
    KVCRServiceError,
    KVCRSocketError,
)
from kvcr.service.protocol import (
    KVCRClient,
    KVCRPoolHold,
    KVCRServiceReasonError,
    claim_wait_s,
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


def _tier_layouts(text: str) -> list[tuple[str, int]]:
    return [(n, int(b)) for n, b in (p.split(":") for p in text.split(","))]


def _receive_timeout_s(args: argparse.Namespace) -> float:
    if args.receive_timeout_s is not None:
        return args.receive_timeout_s
    return claim_wait_s(args.maximum_bytes or args.total_bytes)


def run(args: argparse.Namespace) -> None:
    client = KVCRClient(args.socket)
    hold = client.claim(
        args.guard_index,
        _tier_layouts(args.tier_config),
        args.compatibility_digest,
        (args.control_host, args.control_port),
        gaas=GaaSSettings(
            tenant=args.tenant,
            lineage=args.lineage,
            total_bytes=args.total_bytes,
            minimum_bytes=args.minimum_bytes,
            maximum_bytes=args.maximum_bytes,
        ),
        receive_timeout_s=_receive_timeout_s(args),
    )
    views: dict = {}
    try:
        _step(
            "granted",
            slot_name=hold.slot_name,
            regions=len(hold.regions),
            bytes=sum(r.size for r in hold.regions),
            layout_generation=hold.layout_generation,
            claim_generation=hold.claim_generation,
        )
        for r in hold.regions:
            views[r.index] = (_map(hold.region_fds, r), r)
        _mode(args, client, hold, views)
        for view, _region in views.values():
            view.close()
        views.clear()
        if args.release:
            hold.release()
            _step("released")
    except BaseException:
        # Bare EOF frees nothing: the lease and slot would stay held until this
        # process dies. Say the claim was never served; best effort.
        for view, _region in views.values():
            view.close()
        with contextlib.suppress(BaseException):
            hold.release(activated=False)
        raise
    finally:
        hold.close_local()  # a no-op once released


def _mode(
    args: argparse.Namespace, client: KVCRClient, grant: KVCRPoolHold, views: dict
) -> None:
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
    grown = client.grow(grant.slot_name, args.grow_bytes, grant.claim_generation)
    end = max(r.offset + r.size for _v, r in views.values())
    if grown.layout_generation != grant.layout_generation + 1 or any(
        r.offset < end for r in grown.regions
    ):
        raise _Failure(3, "mismatch", "grown layout moved or skipped a generation")
    for index, (view, region) in views.items():
        _verify(view, index, region.size)
    added = {r.index: (_map(grant.region_fds, r), r) for r in grown.regions}
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
    shrunk = client.shrink(grant.slot_name, tuple(added), grant.claim_generation)
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
    parser.add_argument("--receive-timeout-s", type=float, default=None)
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
    except msgspec.ValidationError as error:  # a bad flag, caught before connecting
        _step("error", reason="error", message=str(error))
        return 2
    except KVCRServiceReasonError as error:
        _step("error", reason=error.reason, message=error.message)
        return 2
    except KVCRSocketError as error:  # the client wraps connect and claim I/O
        timed_out = isinstance(error.__cause__, (TimeoutError, socket.timeout))
        _step(
            "error", reason="timeout" if timed_out else "transport", message=str(error)
        )
        return 2
    except KVCRMsgFramingError as error:
        _step("error", reason="transport", message=str(error))
        return 2
    except KVCRServiceError as error:
        _step("error", reason="error", message=str(error))
        return 2
    except (TimeoutError, socket.timeout) as error:
        _step("error", reason="timeout", message=str(error))
        return 2
    except (OSError, EOFError) as error:  # incl. ConnectionError
        _step("error", reason="transport", message=str(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
