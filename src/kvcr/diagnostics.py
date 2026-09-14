"""Opt-in bounded diagnostics; no cache scans or competing stats consumption."""

import hashlib
import json
import logging
import os
import time

ENABLED = os.environ.get("KVCR_DIAGNOSTICS") == "1"
_LOGGER = logging.getLogger("kvcr.diagnostics")
_INTERVAL_NS = 5_000_000_000


def event(name: str, **fields) -> None:
    if not ENABLED:
        return
    _LOGGER.info(
        "KVCR_DIAGNOSTICS %s",
        json.dumps(
            {
                "schema_version": 1,
                "event": name,
                "epoch_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "component": "kvcr",
                **fields,
            },
            sort_keys=True,
        ),
    )


def keyset_digest(keys) -> str:
    digest = hashlib.sha256()
    for key in sorted(set(map(bytes, keys))):
        digest.update(len(key).to_bytes(8, "big"))
        digest.update(key)
    return digest.hexdigest()


def core_event(core, name: str, **fields) -> None:
    if ENABLED:
        event(
            name,
            agent_name=core.config.nixl_agent_name,
            **getattr(core, "_diagnostic_context", {"role": "primary"}),
            **fields,
        )


def sample_g2(core, *, force=False) -> None:
    if not ENABLED or core._local_dram is None:
        return
    now = time.monotonic_ns()
    if not force and now < getattr(core, "_diagnostic_next_ns", 0):
        return
    core._diagnostic_next_ns = now + _INTERVAL_NS
    dram = core._local_dram
    # Pool count is bounded by layout, not by resident key population.
    pools = []
    for name, (_, length, slot_size) in dram._pools.items():
        total = length // slot_size
        free = len(dram._free_slots[name])
        pools.append(
            {
                "pool_name": name,
                "slot_bytes": slot_size,
                "total_slots": total,
                "free_slots": free,
                "allocated_slots": total - free,
                "allocated_bytes": (total - free) * slot_size,
                "evictable_slots": dram._evictable_slots[name],
            }
        )
    core_event(
        core,
        "g2_occupancy",
        pools=pools,
        meaning="allocated includes in-flight; not proven usable KV",
    )
