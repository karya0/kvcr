# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot-addressed G2 recovery index and bounded, reconstructible key arena."""

import hashlib
import logging
import struct
from collections import defaultdict

HEADER = struct.Struct("<IHHIHHQH38x")
ENTRY = struct.Struct("<QHBBB BHqII32s")
MAGIC = 0x3249584B
GRANULE = 16


class G2Index:
    def __init__(self, buffer, atomic=None):
        self.buffer = buffer
        magic, version, width, self.n_slots, granule, _, self.arena_bytes, _ = (
            HEADER.unpack_from(buffer)
        )
        if (magic, version, width, granule) != (MAGIC, 1, 64, GRANULE):
            raise ValueError("invalid recovery index header")
        self.arena_start = 64 + self.n_slots * 64
        if len(buffer) < self.arena_start + self.arena_bytes:
            raise ValueError("truncated recovery index")
        self.atomic = atomic
        self.reservations = {}
        self.torn = 0
        self.config_errors = 0
        self.dropped = set()
        self.partial = 0
        self.rebuild_allocator()

    @staticmethod
    def initialize(buffer, n_slots, soft_key_max=64):
        HEADER.pack_into(
            buffer,
            0,
            MAGIC,
            1,
            64,
            n_slots,
            GRANULE,
            0,
            n_slots * soft_key_max,
            soft_key_max,
        )

    def _seq(self, slot):
        offset = 64 + slot * 64
        if self.atomic is not None:
            return self.atomic(offset).load_acquire()
        return struct.unpack_from("<Q", self.buffer, offset)[0]

    def _sequence(self, slot, value):
        offset = 64 + slot * 64
        if self.atomic is not None:
            self.atomic(offset).store_release(value)
        else:
            struct.pack_into("<Q", self.buffer, offset, value)

    def validate_slots(self, count):
        if count > self.n_slots:
            self.config_errors += 1
            logging.getLogger(__name__).error(
                "recovery index too small; raise index_bytes"
            )
            raise ValueError("recovery index too small; raise index_bytes")

    def candidates(self):
        self.torn = 0
        self.dropped = set()
        result = []
        for slot in range(self.n_slots):
            seq = self._seq(slot)
            if not seq:
                continue
            if seq & 1:
                self.dropped.add(slot)
                continue
            fields = ENTRY.unpack_from(self.buffer, 64 + slot * 64)
            (
                _,
                key_len,
                n_parts,
                part_idx,
                flags,
                pad,
                label_len,
                position,
                off,
                pad2,
                digest,
            ) = fields
            if not flags:
                continue
            start = self.arena_start + off * GRANULE
            end = start + key_len + label_len
            if (
                flags != 1
                or pad
                or pad2
                or not 1 <= key_len <= 65529
                or not label_len
                or not n_parts
                or part_idx >= n_parts
                or end > self.arena_start + self.arena_bytes
            ):
                self.torn += 1
                self.dropped.add(slot)
                continue
            data = bytes(self.buffer[start:end])
            if self._seq(slot) != seq:
                continue
            if hashlib.blake2b(data, digest_size=32).digest() != digest:
                self.torn += 1
                self.dropped.add(slot)
                continue
            result.append(
                (data[:key_len], data[key_len:], slot, n_parts, part_idx, position)
            )
        return result

    def rebuild_allocator(self):
        self.reservations = {}
        for key, label, slot, _, _, _ in self.candidates():
            off = ENTRY.unpack_from(self.buffer, 64 + slot * 64)[8] * GRANULE
            self.reservations[slot] = (off, (len(key) + len(label) + 15) // 16 * 16)

    def plan_reservation(self, key, parts):
        """Preflight without modifying allocator, index or arena bytes."""
        if not 1 <= len(key) <= 65529 or not 1 <= len(parts) <= 255:
            return None
        staged = dict(self.reservations)
        copies = []
        for label, slot in parts:
            label = label.encode()
            self.validate_slots(slot + 1)
            if not 1 <= len(label) <= 65535 or slot in staged:
                return None
            data = bytes(key) + label
            size = (len(data) + 15) // 16 * 16
            off = 0
            for used, length in sorted(staged.values()):
                if off + size <= used:
                    break
                off = max(off, used + length)
            if off + size > self.arena_bytes:
                return None
            staged[slot] = off, size
            copies.append((off, data))
        return staged, copies

    def commit_reservation(self, plan, *, copy_step=None):
        staged, copies = plan
        self.reservations = staged
        for off, data in copies:
            start = self.arena_start + off
            if copy_step is None:
                self.buffer[start : start + len(data)] = data
            else:
                for i, byte in enumerate(data):
                    self.buffer[start + i] = byte
                    copy_step()

    def reserve(self, key, parts, *, copy_step=None):
        plan = self.plan_reservation(key, parts)
        if plan is None:
            return False
        self.commit_reservation(plan, copy_step=copy_step)
        return True

    def store_steps(self, key, parts, position, *, n_parts=None, part_indices=None):
        """Yield after each atomic sequence store and each payload byte store."""
        if n_parts is None:
            n_parts = len(parts)
        if part_indices is None:
            part_indices = range(len(parts))
        for part_idx, (label, slot) in zip(part_indices, parts):
            off, _ = self.reservations[slot]
            data = bytes(key) + label.encode()
            seq = self._seq(slot)
            self._sequence(slot, seq + 1)
            yield
            entry = ENTRY.pack(
                seq + 1,
                len(key),
                n_parts,
                part_idx,
                1,
                0,
                len(label.encode()),
                position,
                off // 16,
                0,
                hashlib.blake2b(data, digest_size=32).digest(),
            )
            start = 64 + slot * 64
            for i in range(8, 64):
                self.buffer[start + i] = entry[i]
                yield
            self._sequence(slot, seq + 2)
            yield

    def position_steps(self, parts, position):
        for _, slot in parts:
            seq = self._seq(slot)
            self._sequence(slot, seq + 1)
            yield
            start = 64 + slot * 64 + 16
            for i, byte in enumerate(struct.pack("<q", position)):
                self.buffer[start + i] = byte
                yield
            self._sequence(slot, seq + 2)
            yield

    def remove_steps(self, parts):
        for _, slot in parts:
            seq = self._seq(slot)
            self._sequence(slot, seq + 1)
            yield
            start = 64 + slot * 64
            for i in range(8, 64):
                self.buffer[start + i] = 0
                yield
            self._sequence(slot, seq + 2)
            yield
            self.reservations.pop(slot, None)


def rebuild(indexes, *, promotion=False):
    """Rebuild whole keys across pools, preserving full labels and g2 order."""
    grouped = defaultdict(list)
    for pool, index in indexes.items():
        for key, label, slot, n_parts, part_idx, position in index.candidates():
            grouped[key].append((part_idx, n_parts, position, label, slot, pool))
    records = {}
    retained = set()
    for key, entries in grouped.items():
        entries.sort()
        n_parts, position = entries[0][1:3]
        if (
            len(entries) != n_parts
            or [e[0] for e in entries] != list(range(n_parts))
            or any(e[1:3] != (n_parts, position) for e in entries)
        ):
            for entry in entries:
                indexes[entry[5]].partial += 1
            continue
        records[key] = ([(e[3].decode(), e[4]) for e in entries], position)
        retained.update((e[5], e[4]) for e in entries)
    if promotion:
        for pool, index in indexes.items():
            for slot in range(index.n_slots):
                if (pool, slot) not in retained:
                    start = 64 + slot * 64
                    index.buffer[start : start + 64] = bytes(64)
            index.rebuild_allocator()
    return records


def reserve(indexes, key, parts, *, copy_step=None):
    """Reserve a whole key across all pools before writing any arena bytes."""
    if not 1 <= len(parts) <= 255:
        return False
    plans = []
    for pool, index in indexes.items():
        local = [
            (label, slot) for label, slot in parts if label.partition(":")[0] == pool
        ]
        if not local:
            continue
        plan = index.plan_reservation(key, local)
        if plan is None:
            return False
        plans.append((index, plan))
    if any(label.partition(":")[0] not in indexes for label, _ in parts):
        return False
    for index, plan in plans:
        index.commit_reservation(plan, copy_step=copy_step)
    return True
