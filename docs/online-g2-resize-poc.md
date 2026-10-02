# Small native online-resize validation

Use the stacked validation branch, which includes the online-resize implementation and a standalone Linux test. Prerequisites: Python with KVCR dependencies including NIXL 1.3.2, a working CPU UCX backend, and at least 256 MiB free tmpfs under `/dev/shm`. No GPUs or model weights are required. Run from the repository root using that environment:

```sh
python tests/integration/online_g2_resize_poc.py --mode worker --output /tmp/kvcr-resize-worker
python tests/integration/online_g2_resize_poc.py --mode service --output /tmp/kvcr-resize-service
# Continuous traffic across repeated resize, with exact completion accounting:
python tests/integration/online_g2_resize_poc.py --mode service --cycles 100 --output /tmp/kvcr-resize-stress
```

Choose unused output directories for each run; preserve prior evidence. The script imports this checkout's source, creates its own temporary pool directory and processes, and cleans them up on normal exit. It saves operation results in `result.json`; service mode also saves service/primary logs. Failure is an assertion, exception or nonzero exit, not a successful result.

Both modes start two live instances with 32 MiB reservations, 8 MiB registration chunks and 1 MiB synthetic blocks. They shrink A from 32 to 16 MiB and grow B from 8 to 24 MiB. Checks include actual 16 MiB backing-page release and reallocation (`st_blocks * 512`), unchanged retained addresses/bytes, remote reading into grown chunks, and filling the new capacity. Service mode subsequently kills the resized primary, verifies Guard remote delivery into a zeroed destination, then claims a replacement at 16 MiB and regrows it to 32 MiB.

With `--cycles`, eight rolling peer writes continue across repeated donor/recipient resizes. `continuous.json` records submitted/completed/unfinished operations, byte-check failures, operation states at physical resize, backing changes, latency samples and open descriptors. Service mode keeps its source primary in another process. Active overlap is asserted, not inferred from concurrent threads. Failures preserve this evidence before teardown; a failed result is not a passing run.

Repeated full metadata reloads can hit the default1024 descriptor limit in the pinned CPU UCX runtime. Tests use an experiment-only8192 limit; this is not a production leak fix. Queued-write fencing is corrected with peer drain/refresh. The [review disposition](online-g2-resize-review.md) separates verified transport behavior from the remaining backend limitation.

This does not test model TTFT, cross-node RDMA, or a crash inside each transition.

## Impact experiment design

Continuous cached-peer transfers, delayed ACKs, metadata-capture failure and claimed-tail checks now pass at small scale. Before scaling, resolve native socket retention, test crashes inside resize transitions, and integrate the backing/API with an actual engine.

Then compare three arms with identical total G2 and workload: a fixed donor/recipient split (32/16 GiB), live redistribution to 16/32 GiB, and 16/32 GiB from startup. Keep the donor's working set below its final capacity; pressure the recipient's G2 while holding native CPU/GPU cache budgets constant. Make the recipient's effective working set too large for its initial G2 but small enough for its final G2. Confirm tier demand with measured residency/hits, not prompt length alone.

Keep requests arriving on a fixed seeded schedule and continue serving throughout the intervention. Measure actual released/reallocated pages, resize duration, failures, cache-hit bytes, recomputation, TTFT P50/P90, ITL P50/P90 and output throughput. Separate the resize pause from subsequent cache warming. Use at least three paired repetitions with rotated arm order, and run ownership modes separately. Model/engine integration is a prerequisite; this script alone cannot deliver that comparison.
