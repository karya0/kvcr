# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deterministic crash cuts for the iter004 G2 slot/arena representation."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "g2_index", Path(__file__).parents[2] / "src/kvcr/service/g2_index.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
G2Index, rebuild, reserve = _module.G2Index, _module.rebuild, _module.reserve


def make_index(slots=8, soft=256):
    data = bytearray(64 + slots * (64 + soft))
    G2Index.initialize(data, slots, soft)
    return G2Index(data)


def test_published_implies_indexed():
    index = make_index()
    parts = [("pool:a", 0), ("pool:b", 1), ("pool:c", 2)]
    key = b"K" * 200
    assert index.reserve(key, parts)
    steps = list(index.store_steps(key, parts, 5))
    assert len(steps) == 3 * 58
    assert rebuild({"pool": index}) == {key: (parts, 5)}
    writer = index.position_steps(parts, 17)
    for cut, _ in enumerate(writer, 1):
        truth = {key: (parts, 17)} if cut == 30 else {}
        assert rebuild({"pool": index}) == truth, ("position", cut)
    for cut, _ in enumerate(index.remove_steps(parts), 1):
        assert rebuild({"pool": index}) == {}, ("remove", cut)
    assert index.reserve(b"other", parts)
    for cut, _ in enumerate(index.store_steps(b"other", parts, -9), 1):
        truth = {b"other": (parts, -9)} if cut == 174 else {}
        assert rebuild({"pool": index}) == truth, ("reuse", cut)


def test_arena_fail_closed():
    index = make_index(2, 16)
    before = bytes(index.buffer)
    assert not index.reserve(b"k" * 33, [("pool:a", 0)])
    assert bytes(index.buffer) == before
    assert not index.reserve(b"k", [("x" * 65536, 0)])
    assert bytes(index.buffer) == before
    try:
        index.validate_slots(3)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized layout accepted")
    assert bytes(index.buffer) == before
    assert index.config_errors == 1
    assert index.reserve(b"k", [("pool:a", 0)])
    list(index.store_steps(b"k", [("pool:a", 0)], 1))
    assert rebuild({"pool": index}) == {b"k": ([("pool:a", 0)], 1)}


def test_crash_and_resume():
    index = make_index()
    a = [("pool:a", 0), ("pool:b", 1)]
    b = [("pool:a", 2), ("pool:b", 3)]
    assert index.reserve(b"K", a)
    list(index.store_steps(b"K", a, 7))
    assert index.reserve(b"new", b)
    writer = index.store_steps(b"new", b, 8)
    for _ in range(60):
        next(writer)
    assert rebuild({"pool": index}, promotion=True) == {b"K": (a, 7)}
    index = G2Index(index.buffer)
    c = [("pool:a", 4), ("pool:b", 5)]
    assert index.reserve(b"new", c)
    list(index.store_steps(b"new", c, 8))
    assert rebuild({"pool": index}, promotion=True) == {b"K": (a, 7), b"new": (c, 8)}


def test_torn_and_partial_never_surfaced():
    index = make_index()
    parts = [("pool:" + "a" * 256, 0), ("pool:b", 1)]
    assert index.reserve(b"K", parts)
    list(index.store_steps(b"K", parts, 4))
    index.buffer[index.arena_start] ^= 1
    assert rebuild({"pool": index}) == {}
    assert index.torn == 1
    assert index.partial > 0


def test_crash_and_resume_same_key():
    index = make_index()
    parts = [("pool:a", 0), ("pool:b", 1)]
    assert index.reserve(b"K", parts)
    writer = index.store_steps(b"K", parts, 3)
    for _ in range(60):
        next(writer)
    assert rebuild({"pool": index}, promotion=True) == {}
    index = G2Index(index.buffer)
    parts = [("pool:a", 4), ("pool:b", 5)]
    assert index.reserve(b"K", parts)
    list(index.store_steps(b"K", parts, 6))
    assert rebuild({"pool": index}, promotion=True) == {b"K": (parts, 6)}


def campaign(seed, count):
    import random

    rng = random.Random(seed)
    indexes = {name: make_index(3, 512) for name in ("p", "q", "r")}
    for op in range((count + 2) // 3):
        key = rng.randbytes(rng.randint(1, 200))
        pools = rng.choices(tuple(indexes), k=rng.randint(1, 3))
        # Force the historical three-parts-in-one-pool shape.
        if op % 11 == 0:
            pools = ["p"] * 3
        parts = [
            (pool + ":" + str(i) + "a" * (op % 254), i) for i, pool in enumerate(pools)
        ]
        position = rng.randint(-10000, 10000)
        assert reserve(indexes, key, parts, copy_step=lambda: assert_absent(indexes))
        steps = []
        for i, part in enumerate(parts):
            index = indexes[part[0].partition(":")[0]]
            steps.append(
                index.store_steps(
                    key, [part], position, n_parts=len(parts), part_indices=[i]
                )
            )
        for cut, _ in enumerate((x for step in steps for x in step), 1):
            truth = {key: (parts, position)} if cut == 58 * len(parts) else {}
            assert rebuild(indexes) == truth, ("P1/P2/PI store", seed, op, cut)
            for index in indexes.values():
                assert index.torn == 0
                assert index.dropped == {
                    s for s in range(index.n_slots) if index._seq(s) & 1
                }
        # A missing pool must prevent reconstruction of a multi-pool key.
        if len(set(pools)) > 1:
            subset = {p: i for p, i in indexes.items() if p != pools[0]}
            assert rebuild(subset) == {}
        new_position = position + 1
        steps = [
            indexes[label.partition(":")[0]].position_steps(
                [(label, slot)], new_position
            )
            for label, slot in parts
        ]
        for cut, _ in enumerate((x for step in steps for x in step), 1):
            truth = {key: (parts, new_position)} if cut == 10 * len(parts) else {}
            assert rebuild(indexes) == truth, ("P1/P2/PI position", seed, op, cut)
        steps = [
            indexes[label.partition(":")[0]].remove_steps([(label, slot)])
            for label, slot in parts
        ]
        for cut, _ in enumerate((x for step in steps for x in step), 1):
            assert rebuild(indexes) == {}, ("P1 remove", seed, op, cut)


def test_seeded_all_cuts():
    campaign(0, 30)


def test_scale():
    import time

    start = time.monotonic()
    for seed in range(3):
        campaign(seed, 2000)
    for seed in range(200):
        campaign(seed, 200)
    elapsed = time.monotonic() - start
    print(f"P6 campaign seconds={elapsed:.3f}")
    assert elapsed <= 60, elapsed


def test_uniform_churn_and_near_capacity():
    index = make_index(3, 64)
    parts = [("pool:a", 0), ("pool:b", 1), ("pool:c", 2)]
    for _ in range(30):
        assert index.reserve(b"k" * 32, parts)
        list(index.store_steps(b"k" * 32, parts, 1))
        assert rebuild({"pool": index}) == {b"k" * 32: (parts, 1)}
        list(index.remove_steps(parts))
    assert index.reserve(b"k" * 180, [("pool:a", 0)])
    list(index.store_steps(b"k" * 180, [("pool:a", 0)], 2))
    assert rebuild({"pool": index}) == {b"k" * 180: ([("pool:a", 0)], 2)}


def assert_absent(indexes):
    assert rebuild(indexes) == {}


def test_arena_fail_closed_across_pools():
    indexes = {"p": make_index(3, 256), "q": make_index(3, 16)}
    before = {p: bytes(i.buffer) for p, i in indexes.items()}
    assert not reserve(indexes, b"k" * 100, [("p:a", 0), ("q:b", 0)])
    assert {p: bytes(i.buffer) for p, i in indexes.items()} == before
    assert all(not i.reservations for i in indexes.values())
    assert reserve(indexes, b"k", [("p:a", 0), ("q:b", 0)])


def test_presence_rule_over_two_meta_memfds():
    import mmap
    import os
    import sys
    from contextlib import ExitStack

    import pytest

    if sys.platform != "linux":
        pytest.skip("meta memfds require Linux")
    base = mmap.PAGESIZE * 3
    size = mmap.PAGESIZE
    with ExitStack() as stack:
        indexes = {}
        for pool in ("a", "b"):
            fd = os.memfd_create(pool)
            stack.callback(os.close, fd)
            os.ftruncate(fd, base + size)
            region = stack.enter_context(mmap.mmap(fd, size, offset=base))
            G2Index.initialize(region, 8)
            indexes[pool] = G2Index(region)
        parts = [("a:x", 0), ("b:y", 0)]
        assert reserve(indexes, b"key", parts)
        for part_idx, pool in enumerate(("a", "b")):
            list(
                indexes[pool].store_steps(
                    b"key", [parts[part_idx]], 9, n_parts=2, part_indices=[part_idx]
                )
            )
        assert rebuild(indexes) == {b"key": (parts, 9)}
        assert rebuild({"a": indexes["a"]}) == {}
        assert rebuild({"b": indexes["b"]}) == {}
