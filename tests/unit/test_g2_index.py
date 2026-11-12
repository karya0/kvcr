# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The G2 index in each pool's header: layout, publication, and rebuild."""

import ctypes
import mmap
import random

import pytest

from kvcr.service.g2_index import (
    _PREFIX_BYTES,
    _WRITING,
    PoolHeader,
    entry_bytes,
    pool_ranges,
    rebuild,
)

PAGE = mmap.PAGESIZE
KEY = 36  # a vLLM sha256 block hash plus its 4-byte group index


_MAPPINGS = []  # kept for the session: headers hold raw addresses into them


def _memory(size):
    buffer = mmap.mmap(-1, size)
    _MAPPINGS.append(buffer)
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def _headers(n=8, names=("a", "b"), key_size=KEY, block=PAGE):
    base = _memory(len(names) * PAGE)
    headers = {
        name: PoolHeader(base + index * PAGE, key_size, block, n)
        for index, name in enumerate(names)
    }
    for header in headers.values():
        header.verify(initialize=True)
    return headers


def _key(i):
    return i.to_bytes(4, "little") * (KEY // 4)


def _deposit(headers, slot, key, position=0):
    for header in headers.values():
        header.publish(slot, key, position)


def test_one_pool_without_a_key_size_is_whole_blocks():
    assert pool_ranges(8 * PAGE, [("a", 2 * PAGE)], None) == [("a", 0, 0, 8 * PAGE)]
    with pytest.raises(ValueError, match="whole number"):
        pool_ranges(8 * PAGE, [("a", 3 * PAGE)], None)
    with pytest.raises(ValueError, match="whole number"):
        pool_ranges(PAGE, [("a", 2 * PAGE)], None)


def test_two_pools_with_headers_hold_equal_counts_and_fill_the_size():
    layouts = [("a", 2 * PAGE), ("b", PAGE)]
    n = 100
    header = -(-(_PREFIX_BYTES + n * entry_bytes(KEY)) // PAGE) * PAGE
    size = 2 * header + n * 3 * PAGE
    ranges = pool_ranges(size, layouts, KEY)
    a_data = header
    b_header = a_data + n * 2 * PAGE
    assert ranges == [
        ("a", 0, a_data, n * 2 * PAGE),
        ("b", b_header, b_header + header, n * PAGE),
    ]
    assert all(data % PAGE == 0 for _, _, data, _ in ranges)
    with pytest.raises(ValueError, match=f"{size} bytes holds {n} per pool"):
        pool_ranges(size + PAGE, layouts, KEY)


def test_entry_is_a_flag_word_and_the_key_rounded_to_eight():
    assert entry_bytes(36) == 48
    assert entry_bytes(12) == 24


def test_publish_then_rebuild_finds_the_key_in_every_pool():
    headers = _headers()
    _deposit(headers, 3, _key(1), position=7)
    assert rebuild(headers) == {_key(1): ((("a", 3), ("b", 3)), 7)}


def test_a_position_update_keeps_the_key():
    headers = _headers()
    _deposit(headers, 2, _key(1), position=5)
    _deposit(headers, 2, _key(1), position=-1)
    assert rebuild(headers) == {_key(1): ((("a", 2), ("b", 2)), -1)}


def test_cleared_slots_are_free():
    headers = _headers()
    _deposit(headers, 2, _key(1))
    for header in headers.values():
        header.clear(2)
    assert rebuild(headers) == {}
    assert all(header.scan() == [] for header in headers.values())


def test_a_key_missing_from_one_pool_is_partial_and_cleared(caplog):
    headers = _headers()
    headers["a"].publish(1, _key(1), 0)
    _deposit(headers, 4, _key(2))
    with caplog.at_level("INFO"):
        assert list(rebuild(headers)) == [_key(2)]
    assert "partial=1" in caplog.text
    assert [slot for slot, _, _ in headers["a"].scan()] == [4]


def test_a_write_cut_mid_key_is_torn_and_cleared(caplog):
    headers = _headers()
    _deposit(headers, 0, _key(1))
    for header in headers.values():  # the writer died after the first store
        flag = header._flag(5)
        header._store(flag, _WRITING, 3)
        ctypes.memmove(flag + 8, _key(2)[:10], 10)
    with caplog.at_level("INFO"):
        assert list(rebuild(headers)) == [_key(1)]
    assert "torn=2" in caplog.text
    assert [slot for slot, _, _ in headers["b"].scan()] == [0]


@pytest.mark.parametrize("seed", range(20))
def test_seeded_cuts_recover_exactly_the_keys_whole_in_every_pool(seed):
    rng = random.Random(seed)
    headers = _headers(n=16)
    slots = {}  # slot -> key, the allocator's view
    for i in range(rng.randrange(1, 40)):
        slot = rng.randrange(16)
        if slot in slots:
            for header in headers.values():
                header.clear(slot)
            del slots[slot]
        else:
            _deposit(headers, slot, _key(i), position=i)
            slots[slot] = _key(i)
    free = [slot for slot in range(16) if slot not in slots]
    expected = {key: ((("a", slot), ("b", slot))) for slot, key in slots.items()}
    if free:  # cut a deposit after some of its pools committed, or mid-entry
        slot = rng.choice(free)
        cut = rng.randrange(3)
        if cut:
            headers["a"].publish(slot, _key(99), 0)
        if cut == 2:
            headers["b"]._store(headers["b"]._flag(slot), _WRITING, 3)
    rebuilt = rebuild(headers)
    assert {key: value[0] for key, value in rebuilt.items()} == expected
    for name, header in headers.items():
        assert {s for s, _, _ in header.scan()} == set(slots)


def test_guard_initializes_a_blank_header_and_both_sides_verify():
    base = _memory(PAGE)
    PoolHeader(base, KEY, PAGE, 8).verify(initialize=True)
    PoolHeader(base, KEY, PAGE, 8).verify()
    PoolHeader(base, KEY, PAGE, 8).verify(initialize=True)  # already there


@pytest.mark.parametrize("other", [(12, PAGE, 8), (KEY, 2 * PAGE, 8), (KEY, PAGE, 9)])
def test_a_header_for_another_key_size_block_size_or_count_is_refused(other):
    base = _memory(PAGE)
    PoolHeader(base, KEY, PAGE, 8).verify(initialize=True)
    with pytest.raises(ValueError, match="does not match"):
        PoolHeader(base, *other).verify(initialize=True)


def test_a_worker_refuses_a_header_nobody_initialized():
    base = _memory(PAGE)
    with pytest.raises(ValueError, match="does not match"):
        PoolHeader(base, KEY, PAGE, 8).verify()


def test_a_key_of_another_length_raises():
    headers = _headers()
    with pytest.raises(ValueError, match="35 bytes"):
        headers["a"].publish(0, _key(1)[:-1], 0)


def test_a_position_outside_int32_raises():
    headers = _headers()
    with pytest.raises(ValueError, match="does not fit"):
        headers["a"].publish(0, _key(1), 1 << 31)
