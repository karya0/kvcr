"""Small CPU-only Linux PoC using real KVCR, NIXL/UCX and tmpfs pages."""

import argparse
import ctypes
import json
import os
import selectors
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kvcr import KVCR, KVCRBindings
from kvcr.config import (
    FrameworkDramInput,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    LocalDramOptions,
)
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.kvcr_service import _KVCRService
from kvcr.memory import KVCRPoolAttachment, _KVCRPoolOwner
from kvcr.types import BlockKey, MemDescriptor

MiB = 1 << 20
MAX = 32 * MiB
CHUNK = 8 * MiB
ROOT = BlockKey(b"root")
DIGEST = "online-g2-resize-poc-v1"


class RootKeys:
    def encode(self, value):
        return BlockKey(b"root" + (str(value).encode() if value else b""))

    def decode(self, key):
        return int(key.removeprefix(b"root") or b"0")


def emit(value):
    print(json.dumps(value), flush=True)


def port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def completed(controller, operation):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for op, entries in controller.poll_completed():
            if op == operation:
                assert all(result.success for result in entries.values()), entries
                return entries
        time.sleep(0.001)
    raise TimeoutError(f"KVCR operation {operation} did not finish")


def make(stack, directory, rank, service=False, bind_port=None):
    source = ctypes.create_string_buffer(MAX)
    source.raw = b"a" * MAX
    control = ZmqPeerControlChannel("127.0.0.1", bind_port or port(), "127.0.0.1")
    stack.callback(control.close)
    local = None
    callback = None
    if not service:
        owner = _KVCRPoolOwner.allocate(
            pool_id=f"worker_{rank}",
            pool_size_bytes=MAX + 8192,
            journal_bytes=8192,
            pool_dir=directory,
        )
        stack.callback(owner.close)
        attachment = KVCRPoolAttachment.attach(owner.spec)
        stack.callback(attachment.close)
        local = LocalDramOptions([("", attachment.data_address, MAX)])
        def callback(name, old, new):
            attachment.resize_data(8192, old, new)
    controller = KVCR(
        KVCRConfig(
            nixl_agent_name=f"resize-{os.getpid()}-{rank}-{time.monotonic_ns()}",
            pool_layouts=[("", MiB)],
            nixl_listen_port=0,
            operation_timeout_ms=30000,
            abandon_timeout_ms=60000,
            g2_resize_granularity_bytes=CHUNK,
        ),
        KVCRBindings(
            lambda keys: 0,
            lambda: (),
            lambda handle: True,
            framework_control=control,
            key_adapter=RootKeys(),
            resize_g2_memory=callback,
        ),
        KVCRBackendConfigs(
            framework_dram=FrameworkDramInput(ctypes.addressof(source), MAX),
            local_dram=local,
        ),
        KVCRGuardConfig(str(directory / "memory.sock"), rank, DIGEST)
        if service
        else None,
    )
    stack.callback(controller.close)
    path = controller._pool_hold._attachment._spec.path if service else owner.spec.path
    return controller, source, Path(path), control


def fill(controller, source, blocks=32, tag=b"root"):
    keys = [BlockKey(tag if i == 0 else tag + str(i).encode()) for i in range(blocks)]
    operation = controller.deposit(
        {
            key: [
                MemDescriptor(
                    controller.config.nixl_agent_name,
                    "DRAM",
                    ctypes.addressof(source) + i * MiB,
                    MiB,
                    0,
                )
            ]
            for i, key in enumerate(keys)
        }
    )
    completed(controller, operation)
    return keys


def read(controller, key=ROOT, request_id=None):
    operation = controller.fetch([key], request_id=request_id)
    result = completed(controller, operation)[key]
    assert ctypes.string_at(result.descriptors[0].addr, MiB) == b"a" * MiB
    controller.release([result.release_handle])
    return result.descriptors[0].addr


def blocks(path):
    return path.stat().st_blocks * 512


def hint(controller, endpoint, request_id):
    controller.submit_hint(
        {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "block_hashes": [0, 1],
                        "source_control_endpoint": endpoint,
                    },
                }
            ],
        },
        request_id,
    )


def child_reply(process):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if not selector.select(timeout=1):
                continue
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"child exited {process.poll()}")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise TimeoutError("child did not report")


def command(process, payload):
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    return child_reply(process)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(mode, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with (
        tempfile.TemporaryDirectory(prefix="kvcr-resize-", dir="/dev/shm") as temp,
        ExitStack() as stack,
    ):
        directory = Path(temp)
        a_process = None
        if mode == "service":
            log = stack.enter_context((output / "service.log").open("w"))
            service = subprocess.Popen(
                [sys.executable, __file__, "--service", str(directory)],
                stdout=log,
                stderr=log,
            )
            stack.callback(stop, service)
            deadline = time.monotonic() + 30
            while not (directory / "memory.sock").exists():
                assert service.poll() is None, "service exited; inspect service.log"
                if time.monotonic() > deadline:
                    raise TimeoutError("service startup")
                time.sleep(0.05)
            a_log = stack.enter_context((output / "primary.log").open("w"))
            a_process = subprocess.Popen(
                [sys.executable, __file__, "--primary", str(directory)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=a_log,
                text=True,
                bufsize=1,
            )
            stack.callback(stop, a_process)
            initial = child_reply(a_process)
            a_path, a_port, address = (
                Path(initial["path"]),
                initial["port"],
                initial["address"],
            )
        else:
            a, a_source, a_path, a_control = make(stack, directory, 0)
            fill(a, a_source)
            address = read(a)
        b, b_source, b_path, b_control = make(stack, directory, 1, mode == "service")
        assert b.resize_g2("", 8 * MiB)
        fill(b, b_source, blocks=7, tag=b"second")
        source_endpoint = (
            f"tcp://127.0.0.1:{a_port}" if a_process else a_control.endpoint
        )
        hint(b, source_endpoint, "before-grow")
        read(b, ROOT, "before-grow")
        before = blocks(a_path) + blocks(b_path)
        stamp = time.monotonic()
        if a_process:
            resized = command(a_process, {"resize": 16 * MiB})
            assert resized["ok"] and resized["address"] == address
        else:
            assert a.resize_g2("", 16 * MiB)
            assert read(a) == address
        shrink_ms = (time.monotonic() - stamp) * 1000
        after_shrink = blocks(a_path) + blocks(b_path)
        assert before - after_shrink == 16 * MiB, (before, after_shrink)
        stamp = time.monotonic()
        assert b.resize_g2("", 24 * MiB)
        grow_ms = (time.monotonic() - stamp) * 1000
        hint(b, source_endpoint, "after-grow")
        grown_remote_address = read(b, BlockKey(b"root1"), "after-grow")
        assert grown_remote_address >= b._core._local_dram._pools[""][0] + 8 * MiB
        fill(b, b_source, blocks=24, tag=b"second")
        read(b, BlockKey(b"second23"))
        after_grow = blocks(a_path) + blocks(b_path)
        assert after_grow == before, (before, after_grow)
        records.append(
            {
                "case": mode,
                "before_bytes": before,
                "after_shrink_bytes": after_shrink,
                "after_grow_bytes": after_grow,
                "returned_bytes": before - after_shrink,
                "shrink_ms": shrink_ms,
                "grow_ms": grow_ms,
                "retained_address": address,
                "grown_tail_read": True,
                "remote_read_into_grown_chunk": True,
            }
        )
        if a_process:
            a_process.kill()
            a_process.wait(timeout=10)
            # Survivor requests the old endpoint after confirmed death. Retry a
            # refusal while Guard promotes; each retry is a fresh operation.
            deadline = time.monotonic() + 30
            served = False
            ctypes.memset(ctypes.addressof(b_source), 0, MiB)
            while time.monotonic() < deadline and not served:
                b.submit_hint(
                    {
                        "protocol_version": "0.1",
                        "actions": [
                            {
                                "action_type": "kv.fetch",
                                "action_version": "1.0",
                                "payload": {
                                    "block_hashes": [123],
                                    "source_control_endpoint": f"tcp://127.0.0.1:{a_port}",
                                },
                            }
                        ],
                    },
                    "guard-resized",
                )
                op = b.deliver(
                    {
                        ROOT: [
                            MemDescriptor(
                                b.config.nixl_agent_name,
                                "DRAM",
                                ctypes.addressof(b_source),
                                MiB,
                                0,
                            )
                        ]
                    },
                    "guard-resized",
                )
                try:
                    completed(b, op)
                    assert b_source.raw[:MiB] == b"a" * MiB
                    served = True
                except AssertionError:
                    time.sleep(0.05)
            assert served, "Guard did not serve retained block"
            replacement, replacement_source, replacement_path, _ = make(
                stack, directory, 0, True, a_port
            )
            assert replacement._core._local_dram._pools[""][1] == 16 * MiB
            read(replacement)
            assert replacement.resize_g2("", MAX)
            fill(replacement, replacement_source)
            read(replacement, BlockKey(b"root31"))
            records.append(
                {
                    "case": "guard-after-SIGKILL",
                    "served": True,
                    "replacement_bytes": 16 * MiB,
                    "replacement_regrow_bytes": MAX,
                }
            )
    result = {
        "mode": mode,
        "records": records,
        "elapsed_s": time.monotonic() - started,
        "gpu_hours": 0,
        "build_hours": 0,
        "paid_model_calls": 0,
        "session_token_cost": None,
        "transport": "CPU NIXL/UCX, same-host; not cross-node RDMA",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    emit(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--mode", choices=["worker", "service"], default="worker")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.service:
        service = _KVCRService(
            args.service / "memory.sock",
            args.service,
            guard_count=2,
            pool_sizes_bytes=(MAX,),
            journal_bytes=8192,
            compatibility_digest=DIGEST,
        )
        service.serve_forever()
    elif args.primary:
        with ExitStack() as stack:
            controller, source, path, control = make(stack, args.primary, 0, True)
            fill(controller, source)
            emit(
                {
                    "path": str(path),
                    "port": control.control_bind_address()[1],
                    "address": read(controller),
                }
            )
            for line in sys.stdin:
                payload = json.loads(line)
                emit(
                    {
                        "ok": controller.resize_g2("", payload["resize"]),
                        "address": read(controller),
                    }
                )
    else:
        run(args.mode, args.output)
