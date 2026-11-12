# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The G2 index in each pool's header: layout, publication, and rebuild."""

import ctypes
import mmap

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
    assert pool_ranges(size + PAGE, layouts, KEY) == ranges  # the tail is unused


def test_headers_take_one_block_of_a_whole_block_pool():
    # Qwen3-32B: 1044 vLLM chunks of 49 MiB; without keys they fill it exactly.
    block = 49 << 20
    assert pool_ranges(1044 * block, [("", block)], None) == [
        ("", 0, 0, 1044 * block)
    ]
    header = -(-(_PREFIX_BYTES + 1043 * entry_bytes(KEY)) // PAGE) * PAGE
    assert pool_ranges(1044 * block, [("", block)], KEY) == [
        ("", 0, header, 1043 * block)
    ]
    with pytest.raises(ValueError, match="holds 0 per pool"):
        pool_ranges(block, [("", block)], KEY)


def test_a_position_update_keeps_the_key():
    headers = _headers()
    _deposit(headers, 2, _key(1), position=5)
    _deposit(headers, 2, _key(1), position=-1)
    assert rebuild(headers) == {_key(1): ((("a", 2), ("b", 2)), -1)}


def test_a_committed_entry_takes_a_new_key():
    headers = _headers()
    _deposit(headers, 3, _key(1))
    _deposit(headers, 3, _key(2))  # a slot reused without its entry cleared
    assert list(rebuild(headers)) == [_key(2)]


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
