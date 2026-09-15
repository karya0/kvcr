# Guard UCX prewarming and promotion timing

## Behavior

The first compatible resilient UCX pool claim constructs and retains one idle
NIXL agent on the guard actor thread before the claim is granted. Backend
configuration is supplied by that claim, so prewarming runs at pool
configuration rather than guessing a backend during pool allocation. It does
not register the pool, initialize the shared control endpoint, or serve requests.

Promotion still creates a fresh serving agent on its original progress thread.
The retained agent amortizes UCX's process-wide cold setup and is released on
guard shutdown. Subsequent claims reuse the warm-up agent. Non-UCX backends and
non-resilient service-owned pools skip prewarming.

Tradeoff: one additional NIXL agent/backend and its native resources per pool
group, plus cold-start latency paid on the first claim. A prewarming exception
refuses that claim; it does not publish a half-granted lease. No background
warm-up race or transfer of a thread-owned serving agent is introduced.

This is an intentionally small warm-up patch, not an eager serving-core redesign.
It does not move pool registration out of promotion or promise subsecond RDMA
promotion at all pool sizes.

## Evidence on two L40S GPUs, one node

Separate primary, guard service, and target processes; real NIXL 1.3.2; no
containers/model server. Primary on GPU 0, target on GPU 1. Guard owns only host
pool memory, but UCX must see the GPUs for this local host-to-CUDA path.

| Configuration | Signal → serving | Promotion NIXL init | Host registration |
|---|---:|---:|---:|
| Baseline, 1 GiB / 16 blocks | 1.471 s | 1.342 s | 2.35 ms |
| Baseline, 8 GiB / 880 blocks | 1.571 s | 1.342 s | 2.29 ms |
| Extra-agent experiment, 8 GiB / 880 blocks | 0.359 s | 0.067 s | 2.84 ms |
| This patch, 8 GiB / 880 blocks | 0.348 s | approximately 0.07 s | separately logged |

Each successful 880-block run verified 5,421,137,920 delivered bytes on GPU 1.
Sequential delivery plus GPU validation is not a bulk-transfer benchmark.
These are individual experiments, not percentile measurements. The initial
probe with no GPU visibility in the guard promoted but failed GPU delivery;
it is not counted as successful validation.

An isolated no-pool experiment separated agent/plugin discovery (2.4 ms) from
cold UCX backend creation (1.421 s). Later UCX backends in the same process took
50–58 ms. The 1.34-second local cost is therefore cold UCX setup, not snapshot
creation or pool registration. Individual native CUDA/UCX subcalls remain
unattributed. NIXL construction takes no pool address or size; registration is
a later operation and may have very different costs at 64 GiB on RDMA.

## Original 16-second campaign: unresolved intervals

Measured campaign image: sha256:380ba42ae008ebcd38feb3cad886aee67884105d1996f7ba47da99cc3cc1a4a2.
Software composition was D-prime 056a32c plus PR33 eca7c57 over main 94b4762.
The branch preserves the local staged composition in a separate base commit;
byte-for-byte equivalence to the deployed image is not established.

| Event, 2026-09-15 UTC | Timestamp | Relative to recorded signal |
|---|---|---:|
| SIGKILL issued | 00:35:13.633790 | 0 |
| Discovery attachment removed | 00:35:19.952951 | 6.319 s |
| First embedded UCX timestamp | 00:35:22.990281 | 9.356 s |
| NIXL initialized message (outer timestamp) | 00:35:24.058408 | 10.425 s |
| guard_promoted, 11,155 records | 00:35:29.728615 | 16.095 s |

There is **no guard pidfd/death-observation timestamp in the supplied log**.
Discovery removal is a separate Dynamo event, not a pidfd observation. The
9.356 seconds must not be labeled death-detection latency. Possibilities
include OS/GPU process teardown after SIGKILL, guard scheduling or throttling,
actor work delaying observation, journal work, and native setup before UCX's
first log. The existing log cannot rank these by measured duration.

The approximately 5.67 seconds after NIXL's initialized message also needs
measurement. In this source, NIXL initialization, backend initialization, host
registration, and metadata capture run inside core.start. UCX rows received
at 00:35:29.727 carry embedded epochs near 00:35:24.054: output buffering makes
outer timestamps unsuitable for operation attribution.

Phase locks do not enclose journal replay, snapshots, or NIXL initialization.
Most lock scopes only inspect/update lifecycle state. Claim commit does emit a
DEBUG log under the phase lock; INFO was used in the campaign. No long-held
lock is proven. The actor observes death only when its command queue is empty;
a long command or sustained command queue can defer observation without a
long-held mutex. Reserved/closing phases intentionally defer promotion.

Promotion transfers the in-memory recovery dictionary, not a snapshot of the
KV pool. Snapshot serialization occurs during handback or graceful release.
Old snapshot-tail release took 0.072 ms in the 880-record local baseline.

The 64-GiB pool/RDMA scenario remains untested here: only 58 GiB of tmpfs was
free at the capacity check, and allocation was rejected. The smaller accepted
run is not evidence against a 64-GiB RDMA registration cost. The local node
also lacks the campaign's shared 2-CPU service limit/state-agent load and B200
process teardown workload.

## Enable and interpret diagnostics

Use targeted Python logger levels before service startup:

```python
logging.getLogger("kvcr.guard").setLevel(logging.DEBUG)
logging.getLogger("kvcr.progress").setLevel(logging.DEBUG)
```

Ensure the installed handler accepts DEBUG. The standalone service also accepts
`--log-level DEBUG` or `KVCR_LOG_LEVEL=DEBUG`; that enables broader service logs.
Default INFO leaves timing logs off and does not read diagnostic clocks.

- `guard_promotion_stage`: prewarm, command begin/end, validated death observation,
  recovery start/end, construction start/end, record adoption, snapshot release
  boundary (`core_starting`), and serving.
- `progress_startup_stage`: NIXL, backend, registration, and metadata start/end,
  ready, or failed stage. Completion is not logged when the operation raises.
- Slow observations (100 ms or more): `death_check_lock_slow`,
  `promotion_lock_slow`, `pidfd_poll_slow`, `actor_poll_slow`, with `elapsed_ns`.
  These report only after the wait completes; they are not a deadlock watchdog.

Fields include epoch/monotonic/thread-CPU nanoseconds, PID and native TID.
Guard records carry pool/generation, holder incarnation and recovered count;
construction onward carries the serving agent name, which joins progress logs.
Progress records carry registered-region bytes. Counts default to -1 when not
known; the pre-filter recovery count may include G3-only records.

Subtract monotonic timestamps on the same host. Compare thread CPU with wall
time only for the same thread: native worker CPU is not included, so the gap
alone is not proof of cgroup throttling. Capture service cgroup cpu.stat and
memory events alongside the signal and actual victim-exit observation to test
scheduling/resource hypotheses. Do not infer per-thread stalls from cumulative
cgroup counters or compare monotonic epochs across nodes.

## Reproduce without containers

Use a CUDA-capable environment with the pinned dependencies and this source.
The probe uses test-only GPU registration on the owning progress thread; the
convenience framework_dram configuration registers host memory only.

```sh
PYTHONPATH="$PWD/src" python tests/manual/repro_two_gpu.py \
  --work /tmp/guard-probe-unique --pool-gib 8 --blocks 880 \
  > /tmp/guard-probe-controller.log 2>&1
```

Run outside a sandbox that hides device files. The script uses GPUs 0 and 1,
creates a temporary /dev/shm pool, requires 16 GiB remaining headroom, and kills
only its own primary via pidfd. It checks every target payload, closes its
processes, and removes the temporary pool on normal completion. Use a fresh
--work path; preserve failed-run logs. `--prewarm` is the earlier extra-agent
experiment and is unnecessary when validating this patch.

## Validation

- Full unit suite: 350 passed before the additional slow-lock diagnostic test.
- Final focused guard/progress/startup suite: 57 passed.
- Ruff 0.16.3 lint and format checks for changed files.
- Actual patched two-GPU 8-GiB/880-block run: passed, 348 ms signal to serving;
  fine-grained DEBUG records captured. Slow-lock markers were added afterward
  and checked by the focused suite.
- Exact final branch source and packaged harness: two-GPU 1-GiB/16-block smoke
  passed; 104 ms signal to death_observed, 211 ms signal to serving. Guard and
  progress stage ordering and monotonic timestamps verified from captured logs.
