# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded copies of framework-owner metadata; never a native/data snapshot."""

import json
import math
import time


class _Limit(Exception):
    pass


class _Budget:
    def __init__(self, remaining: int, deadline: float):
        self.remaining = remaining
        self.deadline = deadline

    def charge(self, size: int) -> None:
        if time.monotonic() >= self.deadline:
            raise _Limit("deadline")
        if size > self.remaining:
            raise _Limit("max_bytes")
        self.remaining -= size

    def copy(self, value):
        self.charge(0)
        if value is None or type(value) in (str, int, float, bool):
            # Bound temporary encoding as well as the eventual response.
            if isinstance(value, str) and len(value) > self.remaining:
                raise _Limit("max_bytes")
            if type(value) is int and value.bit_length() > self.remaining:
                raise _Limit("max_bytes")
            self.charge(len(json.dumps(value, allow_nan=False)))
            return value
        if type(value) in (tuple, list, set):
            self.charge(2)
            if len(value) > 4096:
                raise _Limit("max_collection_items")
            if len(value) > self.remaining:
                raise _Limit("max_bytes")
            result = []
            for item in sorted(value) if type(value) is set else value:
                self.charge(1)
                result.append(self.copy(item))
            return result
        if type(value) is dict:
            self.charge(2)
            result = {}
            for key, item in value.items():
                self.charge(2)
                result[self.copy(key)] = self.copy(item)
            return result
        raise TypeError("observation fields must be plain metadata values")


def _observe_metadata(kvcr, max_entries, max_bytes, timeout_s) -> dict:
    if type(max_entries) is not int or not 0 <= max_entries <= 4096:
        raise ValueError("max_entries must be an integer in [0, 4096]")
    if type(max_bytes) is not int or not 4096 <= max_bytes <= 4 << 20:
        raise ValueError("max_bytes must be an integer in [4096, 4194304]")
    if (
        type(timeout_s) not in (int, float)
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 0.25
    ):
        raise ValueError("timeout_s must be finite and in (0, 0.25]")
    deadline = time.monotonic() + timeout_s
    core = kvcr._core
    records = core._block_record_map
    result = {
        "format_version": 1,
        "scope": "framework_owner_metadata_only",
        "max_collection_items": 4096,
        "native_progress": "MISSING_NOT_OBSERVED",
        "source_reader_state": "MISSING_NOT_OBSERVED",
        "global_quiescence": "NOT_ESTABLISHED",
        "data_proof": "MISSING",
        "http_target_correlation": "MISSING",
        "backing_identity_verification": "held_grant_metadata_only_no_revalidation",
        "owner": None,
        "pool_layouts": None,
        "local_regions": [],
        "guarded_backing": None,
        "entries": [],
        "total_entries": len(records),
        "exported_entries": 0,
        "omitted_entries": len(records),
        "metadata_complete": False,
        "truncated": False,
        "reason": None,
    }
    # Leave room for final counts/reason and separators outside copied fields.
    budget = _Budget(max_bytes - len(json.dumps(result)) - 128, deadline)
    try:
        result["owner"] = budget.copy(
            {
                "pid": kvcr._metadata_owner_pid,
                "thread_id": kvcr._metadata_owner.ident,
                "nixl_agent_name": core.nixl_agent_name,
                "compatibility_digest": kvcr._metadata_compatibility_digest,
            }
        )
        result["pool_layouts"] = budget.copy(core.pool_layouts)
        local = core._local_dram
        if local is not None:
            for name, (_, length, stride) in local._pools._regions.items():
                result["local_regions"].append(
                    budget.copy(
                        {"label": name, "length": length, "slot_stride": stride}
                    )
                )
        hold = kvcr._pool_hold
        if hold is not None:
            spec = hold._attachment._spec
            result["guarded_backing"] = budget.copy(
                {
                    "pool_id": spec.pool_id,
                    "generation": spec.generation,
                    "device": spec.device,
                    "inode": spec.inode,
                    "mapping_bytes": spec.mapping_bytes,
                    "journal_bytes": spec.journal_bytes,
                }
            )
        for key, record in records.items():
            if len(result["entries"]) >= max_entries:
                raise _Limit("max_entries")
            if type(key) is not bytes:
                raise TypeError("observation requires raw bytes block keys")
            if len(key) * 2 > budget.remaining:
                raise _Limit("max_bytes")
            residency = record.local_dram
            result["entries"].append(
                budget.copy(
                    {
                        "key_hex": key.hex(),
                        "request_id": None,
                        "in_flight_ops": len(record.in_flight_ops or ()),
                        "framework_memory_present": record.fw_mem is not None,
                        "g3_present": record.g3 is not None,
                        "other_tier_locations": "MISSING_NOT_EXPORTED",
                        "local_dram": None
                        if residency is None
                        else {
                            "raw_state": residency.state.name,
                            "slots": residency.slots,
                            "key_layout": residency.key_layout,
                            "pending_labels": residency.pending_labels,
                            "writers": residency.writers,
                            "pending_deposits": residency.pending_deposits,
                            "remote_indices": residency.remote_indices,
                            "remote_cancelled": residency.remote_cancelled,
                            "claim_count": residency.claim_count,
                            "retire_on_release": residency.retire_on_release,
                        },
                    }
                )
            )
        budget.charge(0)
        result["metadata_complete"] = True
    except _Limit as error:
        result["truncated"] = True
        result["reason"] = str(error)
    result["exported_entries"] = len(result["entries"])
    result["omitted_entries"] -= result["exported_entries"]
    return result
