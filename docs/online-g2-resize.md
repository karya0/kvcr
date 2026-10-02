# Online G2 resizing prototype

This branch explores explicit physical resizing of KVCR-managed G2 while keeping the instance running and retained cache addresses stable. It does not resize the framework's native CPU tier. It is a research prototype, not merge-ready or production-safe under arbitrary concurrent traffic.

```python
GiB = 1 << 30
# Add g2_resize_granularity_bytes=GiB to the existing KVCRConfig at construction.
done = kvcr.resize_g2("", 30 * GiB)
# False: a retiring block is filling, claimed, or in flight; retry later.
```

Worker-owned G2 provides `KVCRBindings.resize_g2_memory(name, old_bytes, new_bytes)`, which must back/release memory without moving the reservation. Service-owned G2 updates its backing and effective capacity over the primary's existing lease. Shrink removes retiring cache inventory, deregisters tail chunks, then releases physical tmpfs pages. Growth backs/registers chunks before admitting slots. The fixed maximum mapping remains; allocated pages change. Registration chunking is opt-in; existing default registration behavior is retained.

## Short usage guide

1. At construction, enable `g2_resize_granularity_bytes=1 << 30` for large pools. Targets are positive multiples of 1 GiB (1,073,741,824 bytes), within the startup reservation; each registration chunk must contain whole KV blocks. Verify the model's measured block geometry first. The smaller native tests intentionally retain 8 MiB chunks; the global default stays disabled (`0`).
2. Supply the worker-owned physical-memory binding above, or use the service-owned held lease. Call from the operator/framework thread, not a state-locked callback or the NIXL progress thread. There is no vLLM CLI/operator endpoint yet; the examples assume constructed KVCR objects and an integrated backing allocator.
3. For a 100 GiB combined budget, reserve at least 70 GiB per instance, then initially shrink each to 50 GiB. This prototype physically allocates ceilings at bootstrap: allow 140 GiB plus framework buffers, journals, registration and process headroom before reducing to the steady 100 GiB budget.
4. Shrink the donor, confirm physical release, then grow the recipient. `False` means a busy tail: retry later and do not grow the recipient yet. Exceptions are failures, not successful resize. `True` is local completion, not proof that every peer has refreshed or that new capacity is warm.

```python
# A and B are live KVCR instances already serving at 50 GiB each.
GiB = 1 << 30
if a.resize_g2("", 30 * GiB):
    # Operator verifies backing-page decrease before allocating to B.
    grown = b.resize_g2("", 70 * GiB)  # Check completion; exceptions are failures.

# Later, after demand switches to A:
if b.resize_g2("", 30 * GiB):
    # Verify the physical release again, then grow A.
    grown = a.resize_g2("", 70 * GiB)
```

Do not measure release by file length: the virtual reservation remains fixed. For the prototype's tmpfs files, use allocated blocks (`stat().st_blocks * 512`); also track cgroup/shmem memory and headroom. A failed grow can leave total G2 below budget; do not treat the two calls as an atomic cross-instance transaction.

```mermaid
flowchart TD
    A[Explicit resize request] --> B{Valid target and geometry?}
    B -->|No| X[Reject without reporting success]
    B -->|Yes| C[NIXL progress owner executes resize]
    C --> D{Shrink or grow?}
    D -->|Shrink| E{Retiring tail busy?}
    E -->|Yes| R[Return false; keep backing; retry later]
    E -->|No| F[Remove tail inventory and journal entries]
    F --> G[Deregister tail; release physical pages]
    D -->|Grow| H[Back pages; register chunks; capture metadata]
    H --> I[Admit new slots only after successful preparation]
    G --> J[Capture updated metadata; invalidate old ACKs]
    I --> K[Local resize complete]
    J --> K
    K --> L[Peers drain active writes and refresh registration]
    L --> M[New requests reuse retained KV; added capacity warms]
```

Peer refresh can overlap later traffic; it is not an additional synchronous step guaranteed complete at API return. The flow shows successful operations; failure/rollback limitations are described below.

## Runtime tradeoffs

The intended benefit is to redistribute spare host RAM to a pressured cache, retaining its existing KV and potentially reducing evictions/recomputation. This model-serving benefit is not yet measured.

Resize runs synchronously on the NIXL progress thread under the state lock. Scanning cache records, evictions, service RPCs, page allocation/release, registration and metadata capture can temporarily delay serving and consume transfer deadlines. Shrink scans tracked records and rebuilds its free-slot deque. Growth and unchanged-size requests also scan records. Smaller chunks increase startup registration count and metadata size; the current chunk setting also applies to registered framework DRAM, not only G2. Cache removed during shrink may have to be recomputed. Large removal bursts may pressure the recovery journal.

A small Linux same-host CPU NIXL/UCX PoC released/reallocated 16 MiB between live instances. One initial pass measured roughly 5.3 ms shrink and 10.9 ms growth. Retained addresses/bytes, remote reads into grown chunks, and service-owned Guard delivery after a completed-resize primary crash passed. These are small operation timings, not TTFT/throughput results or predictions for large pools/RDMA. Independent repeat timings varied, particularly growth.

## Review blockers and limits

Snapshot-bound ACKs, capture-before-growth-admission, callback deadlock rejection, fractional watermarks and unresolved-rollback containment now have regression coverage. Shrink capture failure withholds stale metadata until retry. See the [review disposition](online-g2-resize-review.md).

Peer refresh now defers registration-bearing messages and pauses new native writes only to that peer until its existing native handles complete and release. Queued retained writes keep their generation only for the same known process incarnation and unchanged native handle; replacement/unknown peers and failed reloads remain fenced. Deferred requests retain their original deadlines. Continuous C8/100-cycle native tests pass in both ownership modes, including physical backing changes and service-owned Guard/replacement checks.

Repeated full NIXL remove/reload still accumulates sockets in the pinned CPU UCX runtime, independently reproduced without KVCR or DMA. A higher test descriptor limit is a discriminator, not a production fix. Crashes inside resize transitions, large-pool performance and cross-node RDMA remain unverified.

Supported scope: one pool; positive chunk-aligned sizes within the initial reservation; no automatic resizing policy; retry-based busy shrink. The ceiling is physically allocated once at bootstrap. An already-promoted Guard has no resize API. Framework integration still needs an invocation path, callback/event thread-safety validation, and a backing-memory binding for worker-owned G2. Existing routing need not change solely for resizing if eviction events are delivered correctly.

Unit tests cover allocator/registration ordering and selected failures. A separate stacked validation branch supplies the small native Linux PoC; neither is a production engine integration.
