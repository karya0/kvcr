"""Standalone two-GPU promotion probe; run with the pinned KVCR on PYTHONPATH."""

import argparse
import functools
import json
import logging
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def event(name, **fields):
    sys.stdout.write(
        json.dumps(
            dict(
                event=name,
                pid=os.getpid(),
                monotonic=time.monotonic(),
                epoch=time.time(),
                **fields,
            )
        )
        + "\n"
    )
    sys.stdout.flush()


def instrument():
    import kvcr.progress as progress
    from kvcr.core import _KVCRCore
    from kvcr.guard import _Guard, _RecoveryState
    from kvcr.progress import _KVCRProgress

    progress._STARTUP_TIMEOUT_SECONDS = 120
    for cls, names in [
        (_Guard, ["_promote_for", "_promote", "_serve"]),
        (
            _RecoveryState,
            ["take_for_promotion", "prepare_to_serve", "release_snapshot_region"],
        ),
        (_KVCRCore, ["__init__", "adopt_recovery_records", "start"]),
        (
            _KVCRProgress,
            ["_initialize_nixl", "_register_memory_regions", "_capture_agent_metadata"],
        ),
    ]:
        for name in names:

            def wrap(original, label):
                @functools.wraps(original)
                def measured(*args, **kwargs):
                    start = time.monotonic()
                    event(label + ".begin")
                    result = original(*args, **kwargs)
                    event(label + ".end", seconds=time.monotonic() - start)
                    return result

                return measured

            setattr(cls, name, wrap(getattr(cls, name), cls.__name__ + "." + name))


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(check, label, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.001)
    raise TimeoutError(label)


def service(args):
    from kvcr.kvcr_service import _KVCRService

    root = Path(args.work)
    warm_agent = None
    if args.prewarm:
        from nixl import nixl_agent, nixl_agent_config

        start = time.monotonic()
        warm_agent = nixl_agent(
            "probe-prewarm",
            nixl_agent_config(
                num_threads=4,
                capture_telemetry=True,
                enable_listen_thread=True,
                listen_port=0,
                backends=["UCX"],
            ),
        )
        event("prewarm_complete", seconds=time.monotonic() - start)
    required = args.pool_gib * 1024**3 + 16 * 1024**2
    if shutil.disk_usage("/dev/shm").free < required + 16 * 1024**3:
        raise RuntimeError("pool would leave less than 16 GiB free in /dev/shm")
    with tempfile.TemporaryDirectory(prefix="kvcr-gpu-probe-", dir="/dev/shm") as pools:
        srv = _KVCRService(
            root / "service.sock",
            Path(pools),
            guard_count=1,
            pool_sizes_bytes=(args.pool_gib * 1024**3,),
            journal_bytes=16 * 1024**2,
            compatibility_digest="01" * 32,
        )
        thread = threading.Thread(target=srv.serve_forever)
        thread.start()
        try:
            (root / "service.ready").touch()
            guard = srv._registry._guards[0]
            wait_for(lambda: guard._serving, "guard serving")
            event("serving", recovered_blocks=len(guard._core._block_record_map))
            (root / "guard.ready").touch()
            wait_for(lambda: (root / "stop").exists(), "stop")
        finally:
            srv.shutdown()
            thread.join(10)
            srv.close()
            del warm_agent


def worker(args):
    import torch

    from kvcr import KVCR, KVCRBindings
    from kvcr.config import (
        KVCRBackendConfigs,
        KVCRConfig,
        KVCRGuardConfig,
        RemoteFWDramOptions,
    )
    from kvcr.control_channels import ZmqPeerControlChannel
    from kvcr.progress import _KVCRProgress
    from kvcr.types import BlockKey, MemDescriptor

    # Register the real GPU buffer on KVCR's owning progress thread, before
    # metadata capture. Product code supports descriptors but its convenience
    # framework_dram configuration only registers host memory.
    size = 6160384  # Campaign descriptor bytes per block.
    tensor = torch.full(
        (size,), 73 if args.role == "primary" else 0, dtype=torch.uint8, device="cuda:0"
    )
    torch.cuda.synchronize()
    original = _KVCRProgress._register_memory_regions

    def register(self):
        original(self)
        self._memory_registrations.append(
            self.nixl_agent.register_memory(
                [(tensor.data_ptr(), size, 0, "")], mem_type="VRAM"
            )
        )

    _KVCRProgress._register_memory_regions = register
    root = Path(args.work)
    control = args.port if args.role == "primary" else port()
    kv = KVCR(
        KVCRConfig(
            nixl_agent_name="probe-" + args.role,
            pool_layouts=[("", size)],
            nixl_listen_port=0,
            inventory_report_interval_ms=0,
            operation_timeout_ms=30000,
            abandon_timeout_ms=60000,
        ),
        KVCRBindings(
            lambda keys: 1,
            lambda: (),
            lambda handle: False,
            framework_control=ZmqPeerControlChannel("127.0.0.1", control, "127.0.0.1"),
        ),
        KVCRBackendConfigs(
            remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False)
        ),
        KVCRGuardConfig(
            kvcr_service_socket_path=str(root / "service.sock"),
            guard_index=0,
            compatibility_digest="01" * 32,
        )
        if args.role == "primary"
        else None,
    )

    def descriptors(i):
        return {
            BlockKey(str(i).encode()): [
                MemDescriptor("probe-" + args.role, "VRAM", tensor.data_ptr(), size, 0)
            ]
        }

    def complete(op):
        results = wait_for(lambda: dict(kv.poll_completed()), "transfer completion", 60)
        assert all(item.success for item in results[op].values()), results

    try:
        if args.role == "primary":
            for i in range(args.blocks):
                complete(kv.deposit(descriptors(i)))
            event("deposited", blocks=args.blocks, bytes=args.blocks * size)
            (root / "primary.ready").touch()
            wait_for(lambda: False, "primary awaiting SIGKILL")
        else:
            (root / "target.ready").touch()
            wait_for(lambda: (root / "guard.ready").exists(), "guard ready")
            hint = {
                "protocol_version": "0.1",
                "message_id": "probe",
                "actions": [
                    {
                        "action_id": "fetch",
                        "action_type": "kv.fetch",
                        "action_version": "1.0",
                        "payload": {
                            "source_control_endpoint": f"tcp://127.0.0.1:{args.port}",
                            "block_hashes": [123],
                        },
                    }
                ],
            }
            kv.submit_hint(hint, request_id="probe")
            started = time.monotonic()
            for i in range(args.blocks):
                tensor.zero_()
                torch.cuda.synchronize()
                complete(kv.deliver(descriptors(i), request_id="probe"))
                assert bool(torch.all(tensor == 73)), f"payload mismatch block {i}"
            event(
                "verified_gpu_delivery",
                blocks=args.blocks,
                bytes=args.blocks * size,
                seconds=time.monotonic() - started,
            )
    finally:
        kv.close()


def run(args):
    root = Path(args.work)
    root.mkdir(parents=True, exist_ok=False)
    args.port = port()
    children = []
    logs = []
    try:
        for role, gpu in [("service", "0,1"), ("primary", "0"), ("target", "1")]:
            log = (root / (role + ".log")).open("w")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            child = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--role",
                    role,
                    "--work",
                    str(root),
                    "--port",
                    str(args.port),
                    "--pool-gib",
                    str(args.pool_gib),
                    "--blocks",
                    str(args.blocks),
                ]
                + (["--prewarm"] if args.prewarm else []),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append(child)

            def ready():
                if child.poll() is not None:
                    raise RuntimeError(
                        f"{role} exited {child.returncode}; see {log.name}"
                    )
                return (root / (role + ".ready")).exists()

            wait_for(ready, role + " ready")
        primary = children[1]
        pidfd = os.pidfd_open(primary.pid)
        try:
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            event(
                "signal",
                victim_pid=primary.pid,
                pool_gib=args.pool_gib,
                blocks=args.blocks,
            )
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            event("signal_sent", victim_pid=primary.pid)
            observed = poller.poll(60000)
            if not observed or not observed[0][1] & select.POLLIN:
                raise RuntimeError(f"victim pidfd did not report exit: {observed}")
            event("pidfd_ready", victim_pid=primary.pid)
            primary.wait(60)
            event("exit_observed", victim_pid=primary.pid)
        finally:
            os.close(pidfd)
        assert children[2].wait(180) == 0, "target failed; see target.log"
        (root / "stop").touch()
        assert children[0].wait(30) == 0, "service failed; see service.log"
        event("passed")
    finally:
        (root / "stop").touch()
        if children and children[0].poll() is None:
            try:
                children[0].wait(10)
            except subprocess.TimeoutExpired:
                pass
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=["run", "service", "primary", "target"], default="run"
    )
    parser.add_argument("--work", required=True)
    parser.add_argument("--pool-gib", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--prewarm", action="store_true")
    args = parser.parse_args()
    if args.role == "run":
        run(args)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("kvcr.guard").setLevel(logging.DEBUG)
        logging.getLogger("kvcr.progress").setLevel(logging.DEBUG)
        instrument()
        (service if args.role == "service" else worker)(args)
