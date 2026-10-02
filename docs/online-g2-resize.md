# Online G2 resizing prototype

This branch explores explicit physical resizing of KVCR-managed G2 while keeping the instance running and retained cache addresses stable. It does not resize the framework's native CPU tier. It is a research prototype, not merge-ready or production-safe under arbitrary concurrent traffic.

```python
config = KVCRConfig(..., g2_resize_granularity_bytes=8 * 1024**2)
done = kvcr.resize_g2("", 16 * 1024**2)
# False: a retiring block is filling, claimed, or in flight; retry later.
```

Worker-owned G2 provides `KVCRBindings.resize_g2_memory(name, old_bytes, new_bytes)`, which must back/release memory without moving the reservation. Service-owned G2 updates its backing and effective capacity over the primary's existing lease. Shrink removes retiring cache inventory, deregisters tail chunks, then releases physical tmpfs pages. Growth backs/registers chunks before admitting slots. The fixed maximum mapping remains; allocated pages change. Registration chunking is opt-in; existing default registration behavior is retained.

## Runtime tradeoffs

The intended benefit is to redistribute spare host RAM to a pressured cache, retaining its existing KV and potentially reducing evictions/recomputation. This model-serving benefit is not yet measured.

Resize runs synchronously on the NIXL progress thread under the state lock. Scanning cache records, evictions, service RPCs, page allocation/release, registration and metadata capture can temporarily delay serving and consume transfer deadlines. Shrink scans tracked records and rebuilds its free-slot deque. Growth and unchanged-size requests also scan records. Smaller chunks increase startup registration count and metadata size; the current chunk setting also applies to registered framework DRAM, not only G2. Cache removed during shrink may have to be recomputed. Large removal bursts may pressure the recovery journal.

A small Linux same-host CPU NIXL/UCX PoC released/reallocated 16 MiB between live instances. One initial pass measured roughly 5.3 ms shrink and 10.9 ms growth. Retained addresses/bytes, remote reads into grown chunks, and service-owned Guard delivery after a completed-resize primary crash passed. These are small operation timings, not TTFT/throughput results or predictions for large pools/RDMA. Independent repeat timings varied, particularly growth.

## Review blockers and limits

Snapshot-bound ACKs, capture-before-growth-admission, callback deadlock rejection, fractional watermarks and unresolved-rollback containment now have regression coverage. Shrink capture failure withholds stale metadata until retry. See the [review disposition](online-g2-resize-review.md).

Continuous rolling traffic exposed a separate blocker: changed metadata from the same process advances route generation and rejects queued writes. Full reloads also accumulate open descriptors and must not retire UCX rkeys still used by native transfers. A per-peer drain/refresh protocol is needed before arbitrary concurrent resizing. Crashes inside resize transitions remain unverified.

Supported scope: one pool; positive chunk-aligned sizes within the initial reservation; no automatic resizing policy; retry-based busy shrink. The ceiling is physically allocated once at bootstrap. An already-promoted Guard has no resize API. Framework integration still needs an invocation path, callback/event thread-safety validation, and a backing-memory binding for worker-owned G2. Existing routing need not change solely for resizing if eviction events are delivered correctly.

Unit tests cover allocator/registration ordering and selected failures. A separate stacked validation branch supplies the small native Linux PoC; neither is a production engine integration.
