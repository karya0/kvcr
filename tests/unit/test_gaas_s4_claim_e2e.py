# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""S4 end to end: a vLLM-shaped v2 claim against the supervisor S4 configures.

gaas S4 guard r7 (2026-11-08) failed every worker claim with
``KVCRServiceError: requested_bytes must hold at least one complete KV block of
51380224 bytes``: the supervisor ran with ``--pool-sizes-gb 0.00001`` (one 8192 B
page), and Guard.configure built ``_PoolDescriptor(name, pool_sizes_bytes[i],
block)`` whose geometry needs at least one whole block. A slot service no longer
reads ``--guard-count`` or ``--pool-sizes-gb``: a Guard is built at its index's first
claim and its pools are the claim's data pool, split by the claim's layouts.

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

Two guard slots (gaas S4 guard r8, 2026-11-08, refused the second under the old
slot cap): with no node cap each slot is granted its claimed maximum.
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
from kvcr.service import memmap, server
from kvcr.service.memory import _compute_pool_geometry
from kvcr.service.node_service import meta_region_bytes
from kvcr.service.recovery_journal import claim_guarded_pool

BLOCK = 51380224  # r7 vLLM kv_bytes_per_chunk
TOTAL = 4 * BLOCK  # 205,520,896
JOURNAL = 100 << 20
META = 209723392  # meta_region_bytes(JOURNAL): the slot's meta memfd
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


def _supervisor_argv(root: Path) -> list[str]:
    return [
        "--socket-path", str(root / "service.sock"),
        "--pool-dir", str(root / "pools"),
        "--compatibility-digest", DIGEST,
        "--holder-image", HOLDER_IMAGE,
        "--slot-root", str(root / "slots"),
        "--manager-id", "gaas-s4-claim-e2e",
        "--disable-hugepages",
        "--namespace", "bench",
        "--node-name", "node-a",
        "--advertise-host", "127.0.0.1",
        "--no-require-pod-identity",
    ]  # fmt: skip


def _start_supervisor(root, monkeypatch):
    """server.main up to serve_forever, with the pod API and NIXL stood in."""
    (root / "pools").mkdir()
    (root / "slots").mkdir(mode=0o700)
    args = server._parse_args(_supervisor_argv(root))
    monkeypatch.setattr(memmap, "hugetlb_enabled", not args.disable_hugepages)
    api = _HolderProcessAPI()
    monkeypatch.setattr(server, "InClusterPodAPI", lambda namespace: api)
    # NIXL registration; the host has none. Guards are built at first claim.
    monkeypatch.setattr(server._Guard, "_prepare_core", Mock())
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


def _pool_rank_files(pool_dir, count):
    """Each Guard rank's pool_{rank} file: (size, allocated bytes).

    With slots, pool_{rank} is a sparse 2-page-journal + 1-page stub, one per
    Guard built so far.
    """
    files = {
        p.name: (p.stat().st_size, p.stat().st_blocks * 512)
        for p in sorted(Path(pool_dir).iterdir())
        if "pool_" in p.name and p.is_file()
    }
    assert len(files) == count
    assert set(files.values()) <= {(3 * mmap.PAGESIZE, 0)}
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


def test_the_vllm_claim_is_served_with_no_pool_sizes(tmp_path, monkeypatch):
    """r7's claim: the grant's pool is the claim's data pool, four whole blocks."""
    args, service, thread, api = _start_supervisor(tmp_path, monkeypatch)
    try:
        assert (args.guard_count, args.pool_sizes_bytes) == (None, None)
        assert _pool_rank_files(args.pool_dir, 0) == {}  # no Guard before a claim
        claimed = _claim_like_vllm(args.socket_path)
        hold = claimed.hold
        try:
            ((pool,),) = (hold._pools,)
            assert (pool.name, pool.size_bytes, pool.block_size_bytes) == (
                "",
                TOTAL,
                BLOCK,
            )
            assert _compute_pool_geometry(pool.size_bytes, BLOCK) == (TOTAL, 4)
            base, size, vmas, spec, meta_fd = _assert_slot_mapped(
                service, hold, "s4/r7/0"
            )
            print(
                "claim served:",
                f"slot={hold.slot_name} regions={len(hold.regions)}",
                f"base={base:#x} bytes={size} vmas={len(vmas)}",
                f"meta inode={spec.inode} meta bytes={os.fstat(meta_fd).st_size}",
            )
        finally:
            hold.release(activated=False)
        # One sparse stub, for the one Guard the claim built.
        _pool_rank_files(args.pool_dir, 1)
    finally:
        _stop_supervisor(service, thread, api)


def test_two_guard_slots_are_served(tmp_path, monkeypatch):
    """r8's two slots: both served at their claimed size, each its own pool."""
    assert meta_region_bytes(JOURNAL) == META
    args, service, thread, api = _start_supervisor(tmp_path, monkeypatch)
    try:
        table = service._server.slots
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
            granted = sum(s.effective_maximum_bytes for s in slots)
            assert granted == 2 * TOTAL
            assert sorted(service._server.registry._guards) == [0, 1]
            print(
                "two-slot claims served:",
                f"granted={granted}",
                f"bases={b0:#x},{b1:#x} data inodes={sorted(data_inodes)}",
                f"meta inodes={sorted(meta_inodes)} meta bytes={META}",
                f"touched={[s.touched_bytes for s in slots]}",
            )
        finally:
            for hold in holds:
                hold.release(activated=False)
    finally:
        _stop_supervisor(service, thread, api)
