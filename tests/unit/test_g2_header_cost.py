# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""What the G2 pool-header index costs the hot path, against the G2 journal it
replaced and against neither. A micro-benchmark: ``pytest -s`` prints the table;
the asserts are loose bounds, not a performance gate."""

import ctypes
import mmap
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from _kvcr_test_utils import (
    FakeNixlAgent,
    _mem_descriptor,
    _new_local_kvcr,
    _poll_until,
    _recovered_record,
)
from test_recovery_journal import _attachment

from kvcr.config import LocalDramOptions
from kvcr.service import protocol
from kvcr.service.g2_index import PoolHeader, entry_bytes
from kvcr.service.protocol import KVCRPoolHold
from kvcr.service.recovery_journal import RecoveryJournal, _attach_journal
from kvcr.types import BlockKey, RegionDescriptor

KEY = 36  # a vLLM sha256 block hash plus its 4-byte group index
BLOCK = 16
RAW_OPS, WARMUP = 10_000, 1_000
BLOCKS = 1_000  # end to end: blocks per phase
_MAPPINGS = []  # kept for the session: headers hold raw addresses into them


def _memory(size):
    buffer = mmap.mmap(-1, size)
    _MAPPINGS.append(buffer)
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def _header(n):
    header = PoolHeader(_memory(64 + n * entry_bytes(KEY)), KEY, BLOCK, n)
    header.verify(initialize=True)
    return header


def _journal():
    journal_bytes = 8 << 20
    mapping = mmap.mmap(-1, journal_bytes + mmap.PAGESIZE)
    _MAPPINGS.append(mapping)
    journal = RecoveryJournal(_attachment(mapping, journal_bytes))
    journal.reset()
    return journal


def _key(i, tag=0):
    return BlockKey(i.to_bytes(4, "little") * (KEY // 4 - 1) + bytes([tag] * 4))


def _measure(op, count=RAW_OPS, after=None, warmup=WARMUP):
    """ns per call of op(i), after a warm-up; after(i) runs untimed."""
    clock, samples = time.perf_counter_ns, []
    for i in range(-warmup, count):
        start = clock()
        op(i)
        elapsed = clock() - start
        if after is not None:
            after(i)
        if i >= 0:
            samples.append(elapsed)
    return samples


def _stats(samples):
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[len(ordered) * 99 // 100]


def _timed(samples, method, count):
    """method, adding its time per item to samples; count(args, result) items."""

    def timed(*args, **kwargs):
        start = time.perf_counter_ns()
        result = method(*args, **kwargs)
        elapsed = time.perf_counter_ns() - start
        if items := count(args, result):
            samples.extend([elapsed // items] * items)
        return result

    return timed


def _deposit_all(kvcr, keys):
    handles = {kvcr.deposit({key: [_mem_descriptor(i)]}) for i, key in enumerate(keys)}
    done = dict(_poll_until(kvcr, lambda d: handles <= dict(d).keys(), timeout=30))
    assert all(entry.success for handle in handles for entry in done[handle].values())


def _end_to_end(setup):
    """put, align, lookup and evict, ns per block, with setup's G2 publication."""
    primary = ctypes.create_string_buffer(BLOCK * BLOCKS)
    local = ctypes.create_string_buffer(BLOCK * BLOCKS)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_local_kvcr(
        agent,
        local,
        BLOCKS,
        framework_regions=[
            RegionDescriptor(addr=ctypes.addressof(primary), size=BLOCK, count=BLOCKS)
        ],
    )
    dram = kvcr._core._local_dram
    pools = dram._pools
    journal = None
    if setup == "header":
        pools._headers = {"": _header(BLOCKS)}
    if setup != "baseline":  # with headers the journal takes G3 only
        journal = _journal()
        _attach_journal(pools, journal)
    keys = [_key(i) for i in range(BLOCKS)]
    put, evict = [], []
    dram._apply_fill_result = _timed(
        put, dram._apply_fill_result, lambda args, _: len(args[0])
    )
    _deposit_all(kvcr, keys)
    del dram._apply_fill_result
    # Each block once: its position moves from -1 to 1. The warm-up is a whole run.
    unknown = BlockKey(b"x" * KEY)
    align = _measure(
        lambda i: kvcr.align_sequence([unknown, keys[i]]), BLOCKS, warmup=0
    )
    lookup = _measure(lambda i: kvcr.query([keys[i]]), BLOCKS, warmup=0)
    pools._allocate_slots = _timed(
        evict, pools._allocate_slots, lambda _, result: len(result[1])
    )
    _deposit_all(kvcr, [_key(i, tag=1) for i in range(BLOCKS)])  # each evicts one
    assert len(evict) == BLOCKS
    assert journal is None or not journal.is_invalid()
    if setup == "header":
        assert len(pools._headers[""].scan()) == BLOCKS
    kvcr.close()
    return {"put": put, "evict": evict, "lookup": lookup, "align": align}


def test_g2_header_hot_path_cost():
    # Level 1: each operation alone.
    header, slots = _header(1024), 1024
    keys = (_key(1), _key(2))

    def publish_key(i):  # alternate keys per pass, so every call writes the key
        header.publish(i % slots, keys[(i // slots) % 2], 0)

    raw = {"publish key": _measure(publish_key)}
    raw["publish position"] = _measure(lambda i: header.publish(0, keys[0], i))
    raw["clear"] = _measure(lambda i: header.clear(i % slots))
    observers = []
    journal = _journal()
    source = SimpleNamespace(_headers={}, observe_residency=observers.append)
    _attach_journal(source, journal)  # no headers: G2 goes to the journal
    record = _recovered_record(g2=[("", 0)])
    raw["journal G2 record"] = _measure(
        lambda i: observers[0](keys[0], record), after=lambda i: journal.read_next()
    )
    assert not journal.is_invalid()
    # The per-API-call lease check of a hold that may reconnect, inside its TTL.
    with patch.object(protocol.threading, "Thread"):  # no connection watcher
        hold = KVCRPoolHold(
            LocalDramOptions([]), (), Mock(), Mock(), _reconnect_request=Mock()
        )
    raw["lease check"] = _measure(lambda i: hold.check_lease())

    # Level 2: through KVCR, the header path, the old G2 journal path, neither.
    _end_to_end("baseline")  # warm-up
    paths = {setup: _end_to_end(setup) for setup in ("baseline", "journal", "header")}

    def cell(samples):
        return "%d / %d" % _stats(samples) if samples else "-"

    columns = ("baseline", "journal path", "header path", "hdr-base", "hdr-jrnl")
    lines = [
        f"G2 hot-path cost, ns per op, median / p99 ({RAW_OPS} raw ops, "
        f"{BLOCKS} blocks end to end, key {KEY} bytes)",
        f"{'op':<22}" + "".join(f"{column:>16}" for column in columns),
    ]
    placement = {
        "publish key": "header path",
        "publish position": "header path",
        "clear": "header path",
        "journal G2 record": "journal path",
        "lease check": "baseline",
    }
    for name, samples in raw.items():
        cells = [cell(samples) if placement[name] == c else "-" for c in columns]
        lines.append(f"{'raw ' + name:<22}" + "".join(f"{c:>16}" for c in cells))
    for op in ("put", "evict", "lookup", "align"):
        medians = [_stats(paths[setup][op])[0] for setup in paths]
        cells = [cell(paths[setup][op]) for setup in paths] + [
            f"{medians[2] - medians[0]:+d}",
            f"{medians[2] - medians[1]:+d}",
        ]
        lines.append(f"{op:<22}" + "".join(f"{c:>16}" for c in cells))
    print("\n" + "\n".join(lines))

    for samples in [*raw.values(), *(s for p in paths.values() for s in p.values())]:
        assert _stats(samples)[0] < 1_000_000  # under 1 ms
    put = {setup: _stats(paths[setup]["put"])[0] for setup in paths}
    assert put["header"] <= 10 * put["baseline"]
