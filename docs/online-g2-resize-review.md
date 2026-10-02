# Online-resize prototype review disposition

Astra and Claude Opus 4.6 reviewed the initial prototype and follow-up corrections. The prototype remains **not merge-ready or production-safe under arbitrary concurrent traffic**.

Corrected with focused regressions:

1. ACKs identify the exact snapshot; delayed old ACKs cannot acknowledge newer registrations. Resizable instances require this identity. Older peers remain usable but cannot suppress metadata resends; static instances still accept legacy ACKs.
2. Growth captures metadata before admitting slots. Failed capture rolls back registrations/backing; failed cleanup blocks further resize. Shrink withholds stale metadata and supports retry. After successful physical release, retry only refreshes metadata, without repeating release.
3. Progress-thread operator reentry and caller-thread resize while holding KVCR's state lock are rejected rather than deadlocking. The lock check intentionally uses CPython's internal `RLock._is_owned()` for this PoC. Call resize outside locked callbacks.
4. Resize uses startup's `ceil` calculation, including fractional watermarks.
5. Ambiguous physical growth or unsuccessful rollback blocks further resize while retaining old-capacity serving. This is containment, not automatic accounting reconciliation.

The stale-content claim from cached **source** registration metadata was not established: transfers acquire current key residency/claims before constructing their source descriptors. This is distinct from stale **destination** registration metadata after growth.

Follow-up small native worker/service checks passed overlapping peer transfers, physical RAM release/reallocation, and completed-resize Guard/replacement recovery. Stronger rolling-concurrency stress exposed defects, followed by a reviewed transport correction:

- Corrected: same-process refresh preserves queued generations only for a known unchanged incarnation/handle. Real or unknown replacement and reload failure fence prior queued writes.
- Corrected: registration refresh waits for native DONE and successful handle release while progress continues; new writes to that peer pause and other peers continue. Deferred requests keep original deadlines. Focused tests cover release failure, expiry and peer isolation. Two consecutive C8/100-cycle CPU native passes completed all deliveries in both ownership modes.
- Remaining: repeated full reloads accumulate sockets. A standalone NIXL test without KVCR, resizing or DMA grew sockets70→870 across100 remove/reload cycles, while registration-only and peer-reuse controls stayed flat. Raising the test limit to8192 is not a production fix.

The native test uses CPU UCX on one host, not model serving or cross-node RDMA. Do not infer production readiness or model TTFT/throughput improvements. Crash-at-each-transition behavior remains unverified.

Pinned sources: [NIXL remote invalidation](https://github.com/ai-dynamo/nixl/blob/v1.3.2/src/core/nixl_agent.cpp#L1543), [UCX rkey lifetime](https://github.com/openucx/ucx/blob/v1.20.0/src/ucp/api/ucp.h#L3203).

See the [change/runtime summary](online-g2-resize.md). The stacked `poc/online-g2-resize-validation` branch adds the [native PoC instructions](https://github.com/karya0/kvcr/blob/poc/online-g2-resize-validation/docs/online-g2-resize-poc.md).
