# Online-resize prototype review disposition

Independent reviews by Astra and Claude Opus 4.6 covered the 10 production files and two unit-test files on top of base `6bcef5f35c104a4b3eac963ae95721c006bdfb48`. The prototype is publishable for review, **not merge-ready or production-safe under arbitrary concurrent traffic**. No correctness fixes were silently applied after review.

Accepted findings requiring further work:

1. Delayed old registration-metadata ACKs can mark a peer current after growth, suppressing the new snapshot. ACKs need a matching snapshot identity. A fake-native reproduction observed new local metadata with no metadata included in the next write request.
2. Growth admits new slots before metadata capture. A capture exception can leave larger capacity serving with stale metadata and ACKs. Admission/refresh failure handling needs ordering and reconciliation.
3. An inventory callback synchronously reentering resize can queue work to the progress thread and wait on itself. The fake-native reproduction stalled both calls. Reject self-thread operator calls; clarify callback dispatch rules.
4. Resize's integer-rounding watermark shortcut is incorrect for valid fractional percentages: four slots at 25.125% gives one instead of the startup calculation's two. A reproduced pressure callback was missed. Use the startup `ceil` expression.
5. Service growth followed by primary registration failure and failed service rollback can leave primary and service capacities different. A reply loss also makes the physical result ambiguous. This requires an unresolved-resize state/reconciliation; no byte corruption was demonstrated.

The stale-content claim from cached **source** registration metadata was not established: transfers acquire current key residency/claims before constructing their source descriptors. This is distinct from stale **destination** registration metadata after growth.

Sequential native CPU NIXL/UCX physical redistribution and completed-resize crash recovery passed. Fresh verification also passed 377 Linux unit tests and 61 selected local tests. These successes do not cover or resolve the findings above. Next tests must inject delayed ACKs, metadata-capture/rollback failures and callback reentry, then exercise continuous transfers through repeated resize before model-serving or RDMA trials.

See the [change/runtime summary](online-g2-resize.md). The stacked `poc/online-g2-resize-validation` branch adds the [native PoC instructions](https://github.com/karya0/kvcr/blob/poc/online-g2-resize-validation/docs/online-g2-resize-poc.md).
