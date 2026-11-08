# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""S4 end to end: a vLLM-shaped v2 claim against the supervisor S4 configures.

gaas S4 guard r7 (2026-11-08) failed every worker claim with
``KVCRServiceError: requested_bytes must hold at least one complete KV block of
51380224 bytes``: the supervisor ran with ``--pool-sizes-gb 0.00001`` (one 8192 B
page), and Guard.configure builds ``_PoolDescriptor(name, pool_sizes_bytes[i],
block)`` whose geometry needs at least one whole block. This drives the same claim
at that size (refused) and at ``0.0478515625`` GB = 51380224 B = one block (served).

Harness (no cluster, no GPU):
- Supervisor: the real ``_KVCRService`` that ``server.main`` builds, from
  ``server._parse_args`` over the r7 supervisor argv and ``server._build_slots``
  (SlotTable with journal 100 MiB, toucher holders). It serves in-process on a
  thread, as every service test here does. Two stand-ins, as in the existing
  harness: the Kubernetes pod API (``InClusterPodAPI``) and the Guard's NIXL
  registration (``_prepare_core``); ``ZmqPeerControlChannel.from_shared_listener``
  is patched by the shared autouse fixture.
- Holder: a real ``python -m kvcr.service.holder`` process per holder pod, run with
  the command and args of the pod spec the broker built. It connects to the
  supervisor's holder socket, receives the region and meta memfds and populates them.
- Local deviations from r7's argv: socket root under a fresh /dev/shm directory,
  ``--namespace``/``--node-name``/``--advertise-host 127.0.0.1`` (env in the pod),
  ``--no-require-pod-identity`` (no hostPID here).

Client: the vLLM KVCR tier path. manager.py (vllm-gaas-h7 397-427) builds
    KVCRGuardConfig(kvcr_service_socket_path=kvcr_service_socket_path,
                    guard_index=guard_index_base + dp_local_rank,
                    compatibility_digest=compatibility_digest,
                    gaas=GaaSSettings(**gaas) if gaas else None)
    KVCR(KVCRConfig(pool_layouts=[("", self.block_size_bytes)], ...),
         KVCRBindings(..., framework_control=control, ...),
         KVCRBackendConfigs(framework_regions=[...], local_dram=local_dram,
                            g3=g3_config, remote_fw_dram=RemoteFWDramOptions(...)),
         guard_config)
and api.py:104 calls ``claim_guarded_pool(config, guard_config, bindings,
backend_configs)`` -> ``KVCRClient(socket).claim(guard_index, pool_layouts,
digest, framework_control.control_bind_address(), g3, remote_fw_dram.backend,
gaas=gaas)``. Here: block_size_bytes = 51380224 (r7 kv_bytes_per_chunk),
guard_index 0, local_dram None, g3 None, gaas total = minimum = maximum = 4 blocks
(r7 claimed ~49.96 GiB; the refusal is a configuration check, so 4 blocks suffice).

Client-side consumers of the stub pool size (``response.pools``):
- protocol.py:97 ``_PoolDescriptor.__post_init__`` -> ``_compute_pool_geometry``:
  size_bytes must hold one block; runs on both Guard.configure and grant decode.
- protocol.py:855-873 ``_grant_layout``: name/block size match the request, offsets
  contiguous from journal_bytes, size_bytes page-multiple (v2 skips the fill check).
- protocol.py:814-820 ``_map_data_pool``: exactly one pool; its block_size_bytes
  must divide the granted regions. size_bytes itself is unused.
- protocol.py:698 v1 only: dram pool = attachment + offset, pool.size_bytes; v2
  uses the region mapping's size (713).
- recovery_journal.py:766-768/645 ``read_handback`` -> ``canonical_pool_terms``:
  pool names for the mirror; the full descriptors (size_bytes included) are
  msgpack'd into the snapshot terms a Guard and its replacement must agree on.

Two guard slots against the slot cap (gaas S4 guard r8, 2026-11-08): worker A
(guard_index 0) was served effective_maximum 53,640,953,856 B (1044 blocks); worker
B (guard_index 1, same tenant/lineage/total/minimum/maximum) was refused
``capacity below minimum for <tenant>/<lineage>/1 (reason: capacity_below_minimum)``.
SlotTable.claim_with_generation (node_service.py:1332-1338) grants
``effective_grant(size, minimum, maximum, _available(tenant, name))`` (:147-153),
and ``_available`` (:1220-1226) is cap_bytes + local_requests(tenant) (0: server.py
_build_slots passes none) - headroom_bytes - the other slots' effective_maximum_bytes.
headroom_bytes defaults to ``_HEADROOM_BYTES = 2 * _GIB`` (node_service.py:68).
The meta memfd does NOT count: a slot's effective_maximum_bytes is the data grant
(:1339, :1351); meta_bytes is charged only to the holder pod's memory (:459-461).
r8: 68,719,476,736 - 2,147,483,648 - 53,640,953,856 = 12,931,039,232 B (12.04 GiB)
< 53,640,953,856. Two S4 slots need cap >= 2,147,483,648 + 2 x 53,640,953,856 =
109,429,391,360 B (101.92 GiB); GAAS_SLOT_CAP_BYTES 120,259,084,288 (112 GiB)
leaves 10,829,692,928 B, which also covers the 2 x 209,723,392 B meta memfds if
one wants node RAM for them inside the cap. Scaled here with T = 4 blocks:
cap = headroom + 1.5 T refuses the second slot; headroom + 2 T + 2 meta + 64 MiB
serves both.
"""

import copy
import mmap
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

if sys.platform != "linux":
    pytest.skip("memfd and MAP_FIXED regions are Linux-only", allow_module_level=True)

from test_kvcr_service import (  # noqa: F401 - _channels_are_taken is autouse
    _channels_are_taken,
    _control_bind,
)
from test_node_service import FakeAPI, no_cluster  # noqa: F401 - autouse
from test_node_service_v2 import pools  # noqa: F401 - used through pytestmark

import kvcr
from kvcr.config import (
    GaaSSettings,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    RemoteFWDramOptions,
)
from kvcr.runtime.control_channels import KVCRServiceError
from kvcr.service import memmap, server
from kvcr.service.memory import _compute_pool_geometry
from kvcr.service.node_service import _HEADROOM_BYTES, meta_region_bytes
from kvcr.service.protocol import CAPACITY_BELOW_MINIMUM, KVCRServiceReasonError
from kvcr.service.recovery_journal import claim_guarded_pool

BLOCK = 51380224  # r7 vLLM kv_bytes_per_chunk
TOTAL = 4 * BLOCK  # 205,520,896
JOURNAL = 100 << 20
META = 209723392  # meta_region_bytes(JOURNAL): the slot's meta memfd
R7_SLOT_CAP = 68719476736  # GAAS_SLOT_CAP_BYTES in r7/r8 (64 GiB)
DIGEST = "db6b57b2823276ab9eb6a80c1ef5dfc8b295757bac26aa16e92205f8c2469dd7"
HOLDER_IMAGE = (
    "nvcr.io/0980761089281446/kapila-dev@sha256:"
    "82f0e57a6f01886cd635f6362750a6464b939f0aee442085b2cf40560b995ccb"
)
SRC = str(Path(kvcr.__file__).resolve().parent.parent)

pytestmark = pytest.mark.usefixtures("pools")  # closes the slot RegionPools


class _HolderProcessAPI(FakeAPI):
    """Pods are records; a holder pod also starts the real holder process."""

    def __init__(self):
        super().__init__()
        self.processes = {}

    def create(self, pod):
        pod = copy.deepcopy(pod)
        container = pod["spec"]["containers"][0]
        container["readinessProbe"] = {}  # FakeAPI defaults it; toucher pods have none
        created = super().create(pod)
        command = [*container["command"], *container["args"]]
        assert command[:3] == ["python", "-m", "kvcr.service.holder"]
        env = dict(os.environ, PYTHONPATH=SRC)
        self.processes[pod["metadata"]["name"]] = subprocess.Popen(
            [sys.executable, *command[1:]], env=env
        )
        return created

    def delete(self, name, uid):
        super().delete(name, uid)
        self._stop(name)

    def _stop(self, name):
        process = self.processes.pop(name, None)
        if process is not None:
            process.terminate()
            process.wait(10)

    def stop_all(self):
        for name in list(self.processes):
            self._stop(name)


@pytest.fixture
def tmp_path():  # the S4 socket root lives on /node-shm; here a fresh /dev/shm dir
    with tempfile.TemporaryDirectory(prefix="gs4e2e-", dir="/dev/shm") as directory:
        yield Path(directory)


def _supervisor_argv(root: Path, pool_sizes_gb: str, slot_cap: int) -> list[str]:
    return [
        "--socket-path", str(root / "service.sock"),
        "--pool-dir", str(root / "pools"),
        "--guard-count", "2",
        "--pool-sizes-gb", pool_sizes_gb,
        "--compatibility-digest", DIGEST,
        "--slot-cap-bytes", str(slot_cap),
        "--holder-image", HOLDER_IMAGE,
        "--slot-root", str(root / "slots"),
        "--manager-id", "gaas-s4-claim-e2e",
        "--disable-hugepages",
        "--namespace", "bench",
        "--node-name", "node-a",
        "--advertise-host", "127.0.0.1",
        "--no-require-pod-identity",
    ]  # fmt: skip


def _start_supervisor(root, pool_sizes_gb, monkeypatch, slot_cap=R7_SLOT_CAP):
    """server.main up to serve_forever, with the pod API and NIXL stood in."""
    (root / "pools").mkdir()
    (root / "slots").mkdir(mode=0o700)
    args = server._parse_args(_supervisor_argv(root, pool_sizes_gb, slot_cap))
    monkeypatch.setattr(memmap, "hugetlb_enabled", not args.disable_hugepages)
    api = _HolderProcessAPI()
    monkeypatch.setattr(server, "InClusterPodAPI", lambda namespace: api)
    service = server._KVCRService(
        args.socket_path,
        args.pool_dir,
        guard_count=args.guard_count,
        pool_sizes_bytes=args.pool_sizes_bytes,
        compatibility_digest=args.compatibility_digest,
        startup_timeout_seconds=args.startup_timeout_seconds,
        slots=server._build_slots(args),
        advertise_host=args.advertise_host,
    )
    for guard in service._server.registry._guards.values():
        guard._prepare_core = Mock()  # NIXL registration; the host has none
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    return args, service, thread, api


def _stop_supervisor(service, thread, api):
    try:
        service.shutdown()
        thread.join(10)
        service.close()
        assert not thread.is_alive()
    finally:
        api.stop_all()


def _claim_like_vllm(socket_path, guard_index=0, lineage="r7"):
    """api.py:104 -> recovery_journal.claim_guarded_pool, as the vLLM tier calls it."""
    framework = Mock(control_bind_address=lambda: _control_bind(guard_index))
    return claim_guarded_pool(
        KVCRConfig("KVCR-s4-e2e", [("", BLOCK)]),
        KVCRGuardConfig(
            kvcr_service_socket_path=str(socket_path),
            guard_index=guard_index,
            compatibility_digest=DIGEST,
            gaas=GaaSSettings("s4", lineage, TOTAL, TOTAL, TOTAL),
        ),
        Mock(framework_control=framework),
        KVCRBackendConfigs(remote_fw_dram=RemoteFWDramOptions()),
    )


def _pool_rank_files(pool_dir):
    """Each Guard rank's pool_{rank} file: (size, allocated bytes).

    The guard-daemonset.yaml comment sizes the supervisor at guard-count x (pool
    size + 100 MiB) of posix_fallocate'd /node-shm; with slots, pool_{rank} is a
    sparse 2-page-journal + 1-page stub whatever --pool-sizes-gb says.
    """
    files = {
        p.name: (p.stat().st_size, p.stat().st_blocks * 512)
        for p in sorted(Path(pool_dir).iterdir())
        if "pool_" in p.name and p.is_file()
    }
    assert len(files) == 2  # --guard-count 2
    assert set(files.values()) == {(3 * mmap.PAGESIZE, 0)}
    return files


def _vmas_covering(address, size):
    """The /proc/self/maps lines over [address, address + size), in order."""
    found = []
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            start, end = (int(x, 16) for x in fields[0].split("-"))
            if start < address + size and end > address:
                path = fields[5].strip() if len(fields) > 5 else ""
                found.append((start, end, int(fields[4]), path))
    return found


def _assert_slot_mapped(service, hold, slot_name):
    """The grant's data pool and journal are the slot's own memfds (S4b, S4c)."""
    # S4b: the data pool is the slot's regions, MAP_FIXED into one range.
    assert hold.slot_name == slot_name
    data_map = hold._data_map
    assert data_map is not None
    assert data_map.size_bytes == TOTAL == sum(r.size for r in hold.regions)
    assert len(hold.regions) == 1
    assert len(hold.region_fds) == 1
    ((name, base, size),) = hold.local_dram.pools
    assert (name, base, size) == ("", data_map.address, TOTAL)
    assert data_map._raw <= base and base + size <= (data_map._raw + data_map._reserve)
    vmas = _vmas_covering(base, size)
    assert vmas[0][0] == base and vmas[-1][1] == base + size
    assert all(a[1] == b[0] for a, b in zip(vmas, vmas[1:]))  # no holes
    region_inode = os.fstat(hold.region_fds[0]).st_ino
    assert all(
        inode == region_inode and path.startswith("/memfd:kvcr-region-pool")
        for _s, _e, inode, path in vmas
    )
    # S4c: the journal attachment is the slot's meta memfd, not pool_{rank}.
    meta_fd = service._server.slots.get(slot_name).pool.meta_fd
    attachment = hold._attachment
    spec = attachment._spec
    assert (spec.device, spec.inode) == (
        os.fstat(meta_fd).st_dev,
        os.fstat(meta_fd).st_ino,
    )
    assert os.fstat(attachment._file_descriptor).st_ino == spec.inode
    assert attachment._fixed_size is True
    assert spec.journal_bytes == JOURNAL
    return base, size, vmas, spec, meta_fd


def test_one_page_stub_pool_refuses_the_vllm_claim(tmp_path, monkeypatch):
    """r7: --pool-sizes-gb 0.00001 is 8192 B, less than one 51380224 B block."""
    args, service, thread, api = _start_supervisor(tmp_path, "0.00001", monkeypatch)
    try:
        assert args.pool_sizes_bytes == (8192,)
        print("S4 r7 supervisor pool_rank files:", _pool_rank_files(args.pool_dir))
        with pytest.raises(KVCRServiceError) as refused:
            _claim_like_vllm(args.socket_path)
        print("S4 r7 claim refused:", type(refused.value).__name__, refused.value)
        assert (
            "requested_bytes must hold at least one complete KV block of 51380224"
            in str(refused.value)
        )
    finally:
        _stop_supervisor(service, thread, api)


def test_one_block_stub_pool_serves_the_vllm_claim(tmp_path, monkeypatch):
    """0.0478515625 GB = 51380224 B: one block per rank, and the v2 claim is served."""
    args, service, thread, api = _start_supervisor(
        tmp_path, "0.0478515625", monkeypatch
    )
    try:
        assert args.pool_sizes_bytes == (BLOCK,)
        files = _pool_rank_files(args.pool_dir)
        print("one-block supervisor pool_rank files (size, allocated):", files)
        claimed = _claim_like_vllm(args.socket_path)
        hold = claimed.hold
        try:
            # The v1 stub descriptor in the grant now holds exactly one block.
            ((stub,),) = (hold._pools,)
            assert (stub.name, stub.size_bytes, stub.block_size_bytes) == (
                "",
                BLOCK,
                BLOCK,
            )
            assert _compute_pool_geometry(stub.size_bytes, stub.block_size_bytes) == (
                BLOCK,
                1,
            )
            base, size, vmas, spec, meta_fd = _assert_slot_mapped(
                service, hold, "s4/r7/0"
            )
            print(
                "one-block claim served:",
                f"slot={hold.slot_name} regions={len(hold.regions)}",
                f"base={base:#x} bytes={size} vmas={len(vmas)}",
                f"meta inode={spec.inode} meta bytes={os.fstat(meta_fd).st_size}",
                f"stub block_count={_compute_pool_geometry(BLOCK, BLOCK)[1]}",
            )
        finally:
            hold.release(activated=False)
        # The supervisor's pool_{rank} files are unchanged by the claim.
        assert _pool_rank_files(args.pool_dir) == files
    finally:
        _stop_supervisor(service, thread, api)


def test_second_guard_slot_below_the_slot_cap_is_refused(tmp_path, monkeypatch):
    """r8: cap = headroom + 1.5 T serves guard 0 and leaves 0.5 T < T for guard 1."""
    cap = _HEADROOM_BYTES + TOTAL + TOTAL // 2
    args, service, thread, api = _start_supervisor(
        tmp_path, "0.0478515625", monkeypatch, cap
    )
    try:
        table = service._server.slots
        assert (table.cap_bytes, table.headroom_bytes) == (cap, 2 << 30)
        first = _claim_like_vllm(args.socket_path, guard_index=0, lineage="r8")
        try:
            assert first.hold.slot_name == "s4/r8/0"
            assert table.get("s4/r8/0").effective_maximum_bytes == TOTAL
            available = table._available("s4", "s4/r8/1")
            assert available == cap - _HEADROOM_BYTES - TOTAL == TOTAL // 2 < TOTAL
            with pytest.raises(KVCRServiceReasonError) as refused:
                _claim_like_vllm(args.socket_path, guard_index=1, lineage="r8")
            print(
                "S4 r8 second slot refused:",
                type(refused.value).__name__,
                refused.value,
                f"cap={cap} headroom={table.headroom_bytes}",
                f"slot0 effective_maximum={TOTAL} available={available}",
            )
            assert refused.value.reason == CAPACITY_BELOW_MINIMUM
            assert refused.value.message == "capacity below minimum for s4/r8/1"
            assert str(refused.value) == (
                "capacity below minimum for s4/r8/1 (reason: capacity_below_minimum)"
            )
            assert table.get("s4/r8/1") is None
            assert len(api.processes) == 1  # no holder for the refused slot
        finally:
            first.hold.release(activated=False)
    finally:
        _stop_supervisor(service, thread, api)


def test_two_guard_slots_within_a_sized_slot_cap_are_served(tmp_path, monkeypatch):
    """cap = headroom + 2 T + 2 meta + 64 MiB: both slots served, each its own pool."""
    assert meta_region_bytes(JOURNAL) == META
    cap = _HEADROOM_BYTES + 2 * TOTAL + 2 * META + (64 << 20)
    args, service, thread, api = _start_supervisor(
        tmp_path, "0.0478515625", monkeypatch, cap
    )
    try:
        table = service._server.slots
        assert (table.cap_bytes, table.headroom_bytes) == (cap, 2 << 30)
        holds = []
        try:
            for index in (0, 1):
                holds.append(
                    _claim_like_vllm(args.socket_path, index, lineage="r8").hold
                )
            names = ["s4/r8/0", "s4/r8/1"]
            mapped = [
                _assert_slot_mapped(service, hold, name)
                for hold, name in zip(holds, names)
            ]
            slots = [table.get(name) for name in names]
            # Each slot's own holder populated its data and meta memfds.
            assert [s.touched_bytes for s in slots] == [TOTAL, TOTAL]
            assert len(api.processes) == 2
            assert all(p.poll() is None for p in api.processes.values())
            assert slots[0].holder != slots[1].holder
            # Distinct data and meta memfds, and disjoint MAP_FIXED data pools.
            data_inodes = {os.fstat(h.region_fds[0]).st_ino for h in holds}
            meta_inodes = {os.fstat(m[4]).st_ino for m in mapped}
            assert len(data_inodes) == len(meta_inodes) == 2
            assert not data_inodes & meta_inodes
            assert all(os.fstat(m[4]).st_size == META for m in mapped)
            (b0, s0, *_), (b1, s1, *_) = mapped
            assert b0 + s0 <= b1 or b1 + s1 <= b0
            # The slot cap holds both grants (meta is not charged against it).
            granted = sum(s.effective_maximum_bytes for s in slots)
            assert granted == 2 * TOTAL <= cap - table.headroom_bytes
            print(
                "two-slot claims served:",
                f"cap={cap} headroom={table.headroom_bytes} granted={granted}",
                f"left={cap - table.headroom_bytes - granted}",
                f"bases={b0:#x},{b1:#x} data inodes={sorted(data_inodes)}",
                f"meta inodes={sorted(meta_inodes)} meta bytes={META}",
                f"touched={[s.touched_bytes for s in slots]}",
            )
        finally:
            for hold in holds:
                hold.release(activated=False)
    finally:
        _stop_supervisor(service, thread, api)
