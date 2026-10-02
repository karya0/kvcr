"""Small CPU-only Linux PoC using real KVCR, NIXL/UCX and tmpfs pages."""

import argparse
import ctypes
import json
import logging
import os
import selectors
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
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
from kvcr.remote_fw_dram import _SourceWriteOp
from kvcr.types import BlockKey, MemDescriptor

MiB = 1 << 20
MAX = 32 * MiB
CHUNK = 8 * MiB
ROOT = BlockKey(b"root")
DIGEST = "online-g2-resize-poc-v1"

# Research-only discriminator: an unchanged process must not fence a retained
# destination merely because its registration snapshot changed during resize.
_source_progress = _SourceWriteOp.progress


def traced_source_progress(operation, progress, event):
    name, generation = operation.route
    current = operation._backend._route_generation.get(name, 0)
    if operation.transfer_id is None and name and generation != current:
        logging.warning(
            "RESIZE_ROUTE_FENCE op=%s target=%s queued=%s current=%s",
            operation.op_id,
            name,
            generation,
            current,
        )
    return _source_progress(operation, progress, event)


_SourceWriteOp.progress = traced_source_progress


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


def fd_counts():
    counts = Counter()
    for path in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        kind = target.split(":", 1)[0] if ":" in target else "file"
        counts[kind] += 1
    return dict(counts)


def continuous(a, a_process, b, destination, endpoint, a_path, b_path, cycles, output):
    """Keep byte-checked peer writes active across registration changes."""
    stop_event = threading.Event()
    samples, errors, overlaps, states, resizes = [], [], [], [], []
    accounting = {"submitted": 0, "unfinished": []}
    owner = b._pool_hold if b._pool_hold is not None else b._core
    attribute = "resize_g2" if b._pool_hold is not None else "_resize_g2_memory"
    original = getattr(owner, attribute)

    def backing(name, old, new):
        overlaps.append(len(b._core._progress._in_flight_ops))
        states.append(
            [
                getattr(op.state, "name", str(op.state))
                for op in b._core._progress._in_flight_ops.values()
            ]
        )
        original(name, old, new)

    setattr(owner, attribute, backing)

    def traffic():
        pending = {}
        try:
            free_slots = list(range(8))
            while pending or not stop_event.is_set():
                while free_slots and not stop_event.is_set():
                    slot = free_slots.pop()
                    request = f"continuous-{accounting['submitted']}"
                    hint(b, endpoint, request)
                    address = ctypes.addressof(destination) + slot * MiB
                    ctypes.memset(address, 0, MiB)
                    operation = b.deliver(
                        {
                            ROOT: [
                                MemDescriptor(
                                    b.config.nixl_agent_name, "DRAM", address, MiB, 0
                                )
                            ]
                        },
                        request,
                    )
                    accounting["submitted"] += 1
                    pending[operation] = (slot, address, time.monotonic())
                for operation, entries in b.poll_completed():
                    slot, address, stamp = pending.pop(operation)
                    assert all(result.success for result in entries.values()), entries
                    assert ctypes.string_at(address, MiB) == b"a" * MiB
                    samples.append(
                        (time.monotonic(), (time.monotonic() - stamp) * 1000)
                    )
                    free_slots.append(slot)
                if any(time.monotonic() - item[2] > 30 for item in pending.values()):
                    raise TimeoutError(f"continuous transfers stuck: {list(pending)}")
                time.sleep(0.001)
        except Exception as error:
            errors.append(repr(error))
            stop_event.set()
        finally:
            accounting["unfinished"] = list(pending)

    def wait_requests(count):
        deadline = time.monotonic() + 30
        while len(samples) < count:
            assert not errors, errors
            if time.monotonic() > deadline:
                raise TimeoutError("continuous traffic made no progress")
            time.sleep(0.001)

    thread = threading.Thread(target=traffic)
    thread.start()
    baseline = 20
    try:
        wait_requests(baseline)
        for cycle in range(cycles):
            before_count = len(samples)
            target = (8 if cycle % 2 == 0 else 24) * MiB
            before_bytes = blocks(b_path)
            stamp = time.monotonic()
            assert b.resize_g2("", target)
            after_bytes = blocks(b_path)
            resizes.append(
                {
                    "target_bytes": target,
                    "backed_bytes": after_bytes,
                    "delta_bytes": after_bytes - before_bytes,
                    "latency_ms": (time.monotonic() - stamp) * 1000,
                }
            )
            donor_target = (32 if cycle % 2 == 0 else 16) * MiB
            if a_process:
                donor = command(a_process, {"resize": donor_target})
                assert donor["ok"]
                resizes[-1]["donor_open_fds"] = donor["open_fds"]
                resizes[-1]["donor_fd_types"] = donor["fd_types"]
            else:
                assert a.resize_g2("", donor_target)
            resizes[-1]["driver_open_fds"] = len(list(Path("/proc/self/fd").iterdir()))
            resizes[-1]["driver_fd_types"] = fd_counts()
            assert blocks(a_path) == donor_target + 8192
            assert after_bytes == target + 8192
            wait_requests(before_count + 5)
    finally:
        stop_event.set()
        thread.join(timeout=35)
        setattr(owner, attribute, original)
        evidence = {
            "case": "continuous-cached-peer",
            "cycles": cycles,
            "completed": len(samples),
            **accounting,
            "errors": errors,
            "thread_stopped": not thread.is_alive(),
            "live_ops_at_backing_change": overlaps,
            "live_op_states_at_backing_change": states,
            "requests": samples,
            "resizes": resizes,
        }
        evidence["idle_fd_samples"] = []
        for delay in (0.1, 0.9, 2.0):
            time.sleep(delay)
            row = {"driver": len(list(Path("/proc/self/fd").iterdir()))}
            row["driver_fd_types"] = fd_counts()
            if a_process and a_process.poll() is None:
                donor = command(a_process, {})
                row["donor"] = donor["open_fds"]
                row["donor_fd_types"] = donor["fd_types"]
            evidence["idle_fd_samples"].append(row)
        (output / "continuous.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert not thread.is_alive() and not errors, evidence
    assert len(samples) >= baseline + cycles * 5
    assert accounting["submitted"] == len(samples) and not accounting["unfinished"]
    assert any(overlaps), "no active peer operation observed during resizing"
    return evidence


def run(mode, output, cycles=0):
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
        if cycles:
            records.append(
                continuous(
                    a if not a_process else None,
                    a_process,
                    b,
                    b_source,
                    source_endpoint,
                    a_path,
                    b_path,
                    cycles,
                    output,
                )
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
    parser.add_argument("--cycles", type=int, default=0)
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
                        "ok": controller.resize_g2("", payload["resize"])
                        if "resize" in payload
                        else True,
                        "address": read(controller),
                        "open_fds": len(list(Path("/proc/self/fd").iterdir())),
                        "fd_types": fd_counts(),
                    }
                )
    else:
        run(args.mode, args.output, args.cycles)
