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
"""

import copy
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
from test_node_service_v2 import pools  # noqa: F401 - fixture

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
from kvcr.service.recovery_journal import claim_guarded_pool

BLOCK = 51380224  # r7 vLLM kv_bytes_per_chunk
TOTAL = 4 * BLOCK  # 205,520,896
JOURNAL = 100 << 20
DIGEST = "db6b57b2823276ab9eb6a80c1ef5dfc8b295757bac26aa16e92205f8c2469dd7"
HOLDER_IMAGE = (
    "nvcr.io/0980761089281446/kapila-dev@sha256:"
    "82f0e57a6f01886cd635f6362750a6464b939f0aee442085b2cf40560b995ccb"
)
SRC = str(Path(kvcr.__file__).resolve().parent.parent)


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


def _supervisor_argv(root: Path, pool_sizes_gb: str) -> list[str]:
    return [
        "--socket-path", str(root / "service.sock"),
        "--pool-dir", str(root / "pools"),
        "--guard-count", "2",
        "--pool-sizes-gb", pool_sizes_gb,
        "--compatibility-digest", DIGEST,
        "--slot-cap-bytes", "68719476736",
        "--holder-image", HOLDER_IMAGE,
        "--slot-root", str(root / "slots"),
        "--manager-id", "gaas-s4-claim-e2e",
        "--disable-hugepages",
        "--namespace", "bench",
        "--node-name", "node-a",
        "--advertise-host", "127.0.0.1",
        "--no-require-pod-identity",
    ]  # fmt: skip


def _start_supervisor(root, pool_sizes_gb, monkeypatch):
    """server.main up to serve_forever, with the pod API and NIXL stood in."""
    (root / "pools").mkdir()
    (root / "slots").mkdir(mode=0o700)
    args = server._parse_args(_supervisor_argv(root, pool_sizes_gb))
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


def _claim_like_vllm(socket_path):
    """api.py:104 -> recovery_journal.claim_guarded_pool, as the vLLM tier calls it."""
    framework = Mock(control_bind_address=lambda: _control_bind(0))
    return claim_guarded_pool(
        KVCRConfig("KVCR-s4-e2e", [("", BLOCK)]),
        KVCRGuardConfig(
            kvcr_service_socket_path=str(socket_path),
            guard_index=0,
            compatibility_digest=DIGEST,
            gaas=GaaSSettings("s4", "r7", TOTAL, TOTAL, TOTAL),
        ),
        Mock(framework_control=framework),
        KVCRBackendConfigs(remote_fw_dram=RemoteFWDramOptions()),
    )


def _pool_rank_files(pool_dir):
    return {
        p.name: (p.stat().st_size, p.stat().st_blocks * 512)
        for p in sorted(Path(pool_dir).iterdir())
        if p.name.startswith("pool_")
    }


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


def test_one_page_stub_pool_refuses_the_vllm_claim(tmp_path, pools, monkeypatch):
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


def test_one_block_stub_pool_serves_the_vllm_claim(tmp_path, pools, monkeypatch):
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
            # S4b: the data pool is the slot's regions, MAP_FIXED into one range.
            assert hold.slot_name == "s4/r7/0"
            data_map = hold._data_map
            assert data_map is not None
            assert data_map.size_bytes == TOTAL == sum(r.size for r in hold.regions)
            assert len(hold.regions) == 1
            assert len(hold.region_fds) == 1
            ((name, base, size),) = hold.local_dram.pools
            assert (name, base, size) == ("", data_map.address, TOTAL)
            assert data_map._raw <= base and base + size <= (
                data_map._raw + data_map._reserve
            )
            vmas = _vmas_covering(base, size)
            assert vmas[0][0] == base and vmas[-1][1] == base + size
            assert all(a[1] == b[0] for a, b in zip(vmas, vmas[1:]))  # no holes
            region_inode = os.fstat(hold.region_fds[0]).st_ino
            assert all(
                inode == region_inode and path.startswith("/memfd:kvcr-region-pool")
                for _s, _e, inode, path in vmas
            )
            # S4c: the journal attachment is the slot's meta memfd, not pool_{rank}.
            meta_fd = service._server.slots.get("s4/r7/0").pool.meta_fd
            attachment = hold._attachment
            spec = attachment._spec
            assert (spec.device, spec.inode) == (
                os.fstat(meta_fd).st_dev,
                os.fstat(meta_fd).st_ino,
            )
            assert os.fstat(attachment._file_descriptor).st_ino == spec.inode
            assert attachment._fixed_size is True
            assert spec.journal_bytes == JOURNAL
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
