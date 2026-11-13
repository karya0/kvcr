# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restart proof: a real v2 kvcr-service with no sizing flags, killed and restarted.

Linux only, no GPU, no cluster. From the repo root:

    UCX_TLS='^ib,rc,ud,dc' KVCR_LOG_LEVEL=INFO PYTHONPATH=src \\
        python tests/e2e/restart_proof.py --workdir /dev/shm/<new> --logdir <new>

Derived from tests/e2e/three_holder_proof.py (wip/gaas-guard-h7-budget-proof-b).

Phase ab (A, B): the supervisor starts with no --guard-count, --pool-sizes-gb or
--slot-cap-bytes. It has no Guard and /ready answers 200. Workers claim 161025622016 B
(3134 blocks) at guard_index 0, 1 and 2, then 8 blocks at index 3: all four are
granted, each index's Guard appears at its first claim, /ready stays 200.

Phase cd (C, D): a fresh supervisor; two workers claim 8 blocks at indices 0 and 2
and write a pattern at both ends of their data pool through their own mapping. The
supervisor process is SIGKILLed (holders and workers live on) and started again with
the same argv. It must rebuild Guards 0 and 2 and no other (D: 1 and 3 have none),
answer /ready 200, re-attach both workers on the memfds the holders hand back (the
client refuses a re-attach whose backing inode changed), and serve fresh claims
whose readers see the bytes written before the kill.

Real: kvcr.service.server.main in its own process (argv parser, _build_slots:
SlotTable + NodeBroker with toucher holders, Guards with NIXL agents, health HTTP);
the holders (kvcr.service.holder's main, one process per holder pod); the claimants
(KVCRClient.claim v2 and KVCRPoolHold's own re-attach, one process per worker).
Stood in:
  - Kubernetes pod API: tests/unit/test_node_service.py's FakeAPI, kept in a file so
    it outlives the supervisor as the API server would. A pod's phase is Failed once
    its holder process is gone; creating a pod starts the holder with the pod's args.
  - Holder populate of data regions above 16 GiB is skipped (sparse memfds; the host
    cannot back 3 x 150 GiB). Smaller regions and the meta memfd are populated.
  - --namespace/--node-name/--advertise-host 127.0.0.1 and --no-require-pod-identity.
  - The workers write and read the data pool through their own mapping, not through
    a KVCR core or a NIXL transfer.
"""

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve()
SRC = SCRIPT.parents[2] / "src"
UNIT_TESTS = SCRIPT.parents[2] / "tests" / "unit"

BLOCK = 51380224  # one KV block of the r3 runs
BIG = 3134  # blocks: 161025622016 B, each r3 worker's claim
SMALL = 8  # blocks: 411041792 B
DIGEST = "db6b57b2823276ab9eb6a80c1ef5dfc8b295757bac26aa16e92205f8c2469dd7"
KEY_SIZE = 36  # what vLLM bcf62995 sends
NAME = "gaas-grfull-b-20261112-r3"  # tenant = lineage, as in r3 arm B
IMAGE = "9edc24983698e9c25d13b7978ace39e71108f593303aba0b70a17376641c4ea3"
SPARSE_ABOVE = 16 << 30
PATTERN_BYTES = 4 << 20


def _stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _say(log: Path, text: str) -> None:
    line = f"{_stamp()} {text}"
    with log.open("a") as out:
        out.write(line + "\n")
    print(line, flush=True)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def _pattern(index: int) -> bytes:
    import hashlib

    return hashlib.shake_256(f"s47 restart proof slot {index}".encode()).digest(
        PATTERN_BYTES
    )


# -- service process ----------------------------------------------------------


def _serve(work: Path, logdir: Path, argv: list[str]) -> None:
    """kvcr-service's main() with the pod API stood in."""
    import copy

    sys.path.insert(0, str(UNIT_TESTS))
    import kvcr
    from kvcr.service import server
    from test_node_service import FakeAPI

    state = work / "pod-api.json"

    class Pods(FakeAPI):
        """Pods kept in a file (they outlive the supervisor); a pod is Failed once
        its holder process is gone; creating one starts the holder process."""

        def __init__(self) -> None:
            super().__init__()
            self.pids: dict[str, int] = {}
            self.processes: dict[str, subprocess.Popen] = {}
            if state.exists():
                saved = json.loads(state.read_text())
                self.pods, self.pids = saved["pods"], saved["pids"]
            print(
                f"RESTART_PROOF pod API at start: {sorted(self.pods)} pids {self.pids}",
                file=sys.stderr,
                flush=True,
            )

        def _save(self) -> None:
            temporary = state.with_suffix(".tmp")
            temporary.write_text(json.dumps({"pods": self.pods, "pids": self.pids}))
            os.replace(temporary, state)

        def get(self, name):
            pod = super().get(name)
            if pod is not None and not _alive(self.pids.get(name)):
                pod["status"] = {"phase": "Failed"}
            return pod

        def create(self, pod):
            pod = copy.deepcopy(pod)
            name = pod["metadata"]["name"]
            container = pod["spec"]["containers"][0]
            container["readinessProbe"] = {}  # FakeAPI sets it; toucher pods have none
            command = [*container["command"], *container["args"]]
            print(
                f"RESTART_PROOF pod create {name} command {command} "
                f"resources {json.dumps(container.get('resources'))}",
                file=sys.stderr,
                flush=True,
            )
            assert command[:3] == ["python", "-m", "kvcr.service.holder"], command
            created = super().create(pod)
            uid = str(uuid.uuid4())  # FakeAPI gives every pod the same UID
            self.pods[name]["metadata"]["uid"] = created["metadata"]["uid"] = uid
            with (logdir / f"holder-{name}.log").open("ab") as out:
                process = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "_holder", *command[3:]],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
            self.processes[name] = process
            self.pids[name] = process.pid
            self._save()
            return created

        def delete(self, name, uid):
            super().delete(name, uid)
            pid = self.pids.pop(name, None)
            if _alive(pid):
                os.kill(pid, signal.SIGTERM)
            self._save()

    print(f"RESTART_PROOF kvcr from {kvcr.__file__}", file=sys.stderr, flush=True)
    if os.environ.get("RESTART_PROOF_STUB_NIXL") == "1":
        # --stub-nixl: the Guard's NIXL registration of its pool is not run.
        server._Guard._prepare_core = lambda self: None
        print("RESTART_PROOF Guard._prepare_core stubbed", file=sys.stderr, flush=True)
    server.InClusterPodAPI = lambda namespace: Pods()
    sys.argv = ["kvcr-service", *argv]
    server.main()


def _service_argv(work: Path, health_port: int) -> list[str]:
    """r3 arm B's supervisor argv without the removed sizing flags."""
    return [
        "--socket-path", str(work / "service.sock"),
        "--pool-dir", str(work / "pools"),
        "--compatibility-digest", DIGEST,
        "--holder-image", "nvcr.io/0980761089281446/kapila-dev@sha256:" + IMAGE,
        "--slot-root", str(work / "slots"),
        "--manager-id", NAME,
        "--disable-hugepages",
        "--worker-lease-ttl-ms", "60000",
        "--health-port", str(health_port),
        # Local only: pod env on the cluster, and no pods to identify here.
        "--namespace", "bench",
        "--node-name", "node-a",  # where FakeAPI places holders
        "--advertise-host", "127.0.0.1",
        "--no-require-pod-identity",
    ]  # fmt: skip


# -- holder process -----------------------------------------------------------


def _holder(argv: list[str]) -> None:
    """kvcr.service.holder's main(), data regions above 16 GiB left sparse."""
    from kvcr.service import holder

    populate = holder._touch

    def touch(view, size, index=0):
        if index < 0 or size <= SPARSE_ABOVE:
            return populate(view, size, index)
        holder._log("populate_skipped_sparse", region=index, bytes=size)

    holder._touch = touch
    sys.argv = ["kvcr.service.holder", *argv]
    holder.main()


# -- claimant process ---------------------------------------------------------


def _backing(hold) -> dict:
    def identity(fd):
        st = os.fstat(fd)
        return [st.st_dev, st.st_ino, st.st_size]

    return {
        "regions": [identity(fd) for fd in hold.region_fds],
        "meta": identity(hold._attachment._file_descriptor),
    }


def _pool_bytes(hold) -> tuple[dict, bytes]:
    import ctypes

    pool, address, size = hold.local_dram.pools[0]
    offsets = (0, size - PATTERN_BYTES)
    data = b"".join(ctypes.string_at(address + o, PATTERN_BYTES) for o in offsets)
    where = {
        "pool": pool,
        "data_offset_in_region": address - hold._data_map.address,
        "pool_bytes": size,
        "pattern_offsets": offsets,
    }
    return where, data


def _claim(
    socket_path: str, index: int, blocks: int, mode: str, work: Path, port: int
) -> int:
    """One v2 claim as an r3 worker sends it; prints RESULT.

    ``port`` is the worker's control port: a Guard keeps the address its first
    claimant asked for, so a later worker of that index asks for the same one.

    hold: keep the lease. write: write the pattern, keep the lease, report the
    client's own re-attach, release when the driver asks. read: read, release.
    """
    import ctypes
    import hashlib
    import logging

    from kvcr.config import GaaSSettings
    from kvcr.service.protocol import KVCRClient

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    size = blocks * BLOCK
    gaas = {
        "tenant": NAME,
        "lineage": NAME,
        "total_bytes": size,
        "minimum_bytes": size,
        "maximum_bytes": size,
        "key_size_bytes": KEY_SIZE,
    }
    started = time.monotonic()
    try:
        hold = KVCRClient(socket_path).claim(
            index,
            [("", BLOCK)],
            DIGEST,
            ("127.0.0.1", port),
            gaas=GaaSSettings(**gaas),
        )
    except Exception as error:  # noqa: BLE001 - the refusal is the result
        result = {
            "guard_index": index,
            "refused": type(error).__name__,
            "reason": getattr(error, "reason", None),
            "message": str(error),
        }
        print("RESULT " + json.dumps(result), flush=True)
        return 2
    result = {
        "guard_index": index,
        "pid": os.getpid(),
        "granted": hold.slot_name,
        "effective_maximum_bytes": hold.effective_maximum_bytes,
        "region_bytes": sum(region.size for region in hold.regions),
        "claim_generation": hold.claim_generation,
        "backing": _backing(hold),
        "after_s": round(time.monotonic() - started, 3),
    }
    if mode == "write":
        pattern = _pattern(index)
        _, address, pool_size = hold.local_dram.pools[0]
        for offset in (0, pool_size - PATTERN_BYTES):
            ctypes.memmove(address + offset, pattern, PATTERN_BYTES)
    if mode in ("write", "read"):
        where, data = _pool_bytes(hold)
        result.update(where)
        result["sha256"] = hashlib.sha256(data).hexdigest()
        result["pattern_sha256"] = hashlib.sha256(_pattern(index) * 2).hexdigest()
    print("RESULT " + json.dumps(result), flush=True)
    if mode == "hold":
        signal.pause()  # hold the lease until the driver kills this worker
        return 0
    if mode == "read":
        hold.release()
        print("RELEASED", flush=True)
        return 0
    first, reported = hold._connection, False
    while True:
        time.sleep(0.1)
        if hold._lease_error is not None:
            print(f"LEASE_ERROR {hold._lease_error!r}", flush=True)
            return 3
        if not reported and hold._connection is not first:
            _, data = _pool_bytes(hold)
            reattached = {
                "guard_index": index,
                "slot": hold.slot_name,
                "claim_generation": hold.claim_generation,
                "backing": _backing(hold),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            print("REATTACHED " + json.dumps(reattached), flush=True)
            reported = True
        if (work / f"release-{index}").exists():
            hold.release()
            print("RELEASED", flush=True)
            return 0


# -- driver -------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _line(path: Path, needle: str) -> str | None:
    return next((x for x in _read(path).splitlines() if needle in x), None)


def _wait(what: str, predicate, timeout_s: float, *processes):
    deadline = time.monotonic() + timeout_s
    while True:
        value = predicate()
        if value:
            return value
        for process in processes:
            if process.poll() is not None:
                value = predicate()
                if value:
                    return value
                raise RuntimeError(f"process exited {process.returncode} ({what})")
        if time.monotonic() > deadline:
            raise RuntimeError(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.1)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _guards(work: Path) -> list[int]:
    """The Guards the running service reports (service_info pool_specs)."""
    from kvcr.service.protocol import service_info

    info = service_info(str(work / "service.sock"), timeout_s=10)
    return sorted(spec["guard_index"] for spec in info["pool_specs"])


def _health(port: int) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=10) as r:
            return r.status, r.read().decode(errors="replace").strip()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace").strip()


def _probe(function, *args):
    """The value, or the error as text: a stalled service is evidence too."""
    try:
        return function(*args)
    except Exception as error:  # noqa: BLE001 - recorded
        return f"{type(error).__name__}: {error}"


def _memfds(pid: int) -> list[list]:
    """(fd, name, dev, inode, size) of each memfd the process holds."""
    found = []
    for entry in sorted(Path(f"/proc/{pid}/fd").iterdir(), key=lambda p: int(p.name)):
        with contextlib.suppress(OSError):
            link = os.readlink(entry)
            if link.startswith("/memfd:"):
                st = os.stat(entry)
                found.append([int(entry.name), link, st.st_dev, st.st_ino, st.st_size])
    return found


class _Run:
    """One workdir: the service, its claimants, their logs."""

    def __init__(
        self, log: Path, work: Path, logdir: Path, timeout_s: float, stub_nixl=False
    ):
        self.log, self.work, self.logdir, self.timeout_s = log, work, logdir, timeout_s
        (work / "pools").mkdir(parents=True)
        (work / "slots").mkdir(mode=0o700)
        logdir.mkdir(parents=True, exist_ok=True)
        path = [str(SRC), os.environ.get("PYTHONPATH", "")]
        self.env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, path)))
        if stub_nixl:
            self.env["RESTART_PROOF_STUB_NIXL"] = "1"
        self.port = _free_port()
        self.argv = _service_argv(work, self.port)
        self.services: list[subprocess.Popen] = []
        self.claimants: list[subprocess.Popen] = []
        self.ports: dict[int, int] = {}  # per index, as a worker's config fixes it

    def say(self, text: str) -> None:
        _say(self.log, text)

    def start_service(self, log_name: str) -> subprocess.Popen:
        service_log = self.logdir / log_name
        with service_log.open("wb") as out:
            service = subprocess.Popen(
                [sys.executable, str(SCRIPT), "_serve", str(self.work)]
                + [str(self.logdir), *self.argv],
                stdout=out,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,  # holders join its process group
            )
        self.services.append(service)
        ready = _wait(
            "service ready",
            lambda: _line(service_log, "KVCR service ready"),
            self.timeout_s,
            service,
        )
        self.say(f"service pid {service.pid} ({log_name}): {ready.strip()}")
        return service

    def claim(self, index: int, blocks: int, mode: str, tag: str) -> tuple:
        claim_log = self.logdir / f"claim-{tag}.log"
        with claim_log.open("wb") as out:
            claimant = subprocess.Popen(
                [sys.executable, str(SCRIPT), "_claim", str(self.work / "service.sock")]
                + [str(index), str(blocks), mode, str(self.work)]
                + [str(self.ports.setdefault(index, _free_port()))],
                stdout=out,
                stderr=subprocess.STDOUT,
                env=self.env,
            )
        self.claimants.append(claimant)
        found = _wait(
            f"claim {tag}",
            lambda: _line(claim_log, "RESULT "),
            self.timeout_s,
            self.services[-1],
            claimant,
        )
        return claimant, claim_log, json.loads(found.split("RESULT ", 1)[1]), found

    def state(self) -> dict:
        return json.loads((self.work / "pod-api.json").read_text())

    def close(self) -> None:
        for claimant in self.claimants:
            claimant.kill()
            claimant.wait()
        holders = self.state()["pids"] if (self.work / "pod-api.json").exists() else {}
        for service in self.services:
            if service.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(service.pid, signal.SIGTERM)
                try:
                    service.wait(15)
                except subprocess.TimeoutExpired:
                    pass
            with contextlib.suppress(ProcessLookupError):
                os.killpg(service.pid, signal.SIGKILL)  # holders left in its group
            service.wait()
        for pid in holders.values():
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        left = {name: pid for name, pid in holders.items() if _alive(pid)}
        self.say(
            f"teardown: services exited {[s.returncode for s in self.services]}; "
            f"claimants killed; holders {holders} killed; still alive: {left or 'none'}"
        )


def _phase_ab(
    log: Path, work: Path, logdir: Path, timeout_s: float, stub_nixl: bool
) -> dict:
    run = _Run(log, work, logdir, timeout_s, stub_nixl)
    verdict = {"A": False, "B": False}
    try:
        run.say(f"[ab] service argv: kvcr-service {' '.join(run.argv)}")
        sizing = [
            flag
            for flag in ("--guard-count", "--pool-sizes-gb", "--slot-cap-bytes")
            if flag in run.argv
        ]
        run.say(f"[ab] sizing flags in argv: {sizing or 'none'}")
        run.say(f"[ab] Guard NIXL registration: {'STUBBED' if stub_nixl else 'real'}")
        run.start_service("service-ab.log")
        guards, health = _guards(work), _health(run.port)
        run.say(f"[ab] A: before any claim: guards {guards}, /ready {health}")
        verdict["A"] = not sizing and guards == [] and health[0] == 200
        ok = True
        for index, blocks in ((0, BIG), (1, BIG), (2, BIG), (3, SMALL)):
            before = _probe(_guards, work)
            claimant, _, result, found = run.claim(index, blocks, "hold", f"ab-{index}")
            after, health = _probe(_guards, work), _probe(_health, run.port)
            run.say(
                f"[ab] B: claim index {index} ({blocks} blocks = {blocks * BLOCK} B, "
                f"worker pid {claimant.pid}) {found.strip()}"
            )
            run.say(
                f"[ab] B: index {index}: guards before {before} after {after}, "
                f"/ready {health}"
            )
            ok = ok and (
                result.get("granted") == f"{NAME}/{NAME}/{index}"
                and result.get("effective_maximum_bytes") == blocks * BLOCK
                and result.get("region_bytes") == blocks * BLOCK
                and isinstance(before, list)
                and index not in before
                and after == sorted([*before, index])
                and health == (200, "ok")
            )
        table = json.loads((work / "slots" / "slot-table.json").read_text())
        for name, slot in sorted(table.get("slots", {}).items()):
            run.say(
                f"[ab] slot-table.json {name}: effective_maximum_bytes "
                f"{slot['effective_maximum_bytes']}"
            )
        run.say(f"[ab] pod API pods: {sorted(run.state()['pods'])}")
        verdict["B"] = ok and len(table.get("slots", {})) == 4
    except Exception as error:  # noqa: BLE001 - reported, then cleaned up
        run.say(f"[ab] driver error: {type(error).__name__}: {error}")
    finally:
        run.close()
    return verdict


def _phase_cd(log: Path, work: Path, logdir: Path, timeout_s: float) -> dict:
    run = _Run(log, work, logdir, timeout_s)
    verdict = {"C": False, "D": False}
    claimed = (0, 2)
    never = (1, 3)
    try:
        run.say(f"[cd] service argv: kvcr-service {' '.join(run.argv)}")
        first = run.start_service("service-cd-before.log")
        run.say(
            f"[cd] before claims: guards {_guards(work)}, /ready {_health(run.port)}"
        )
        writers = {}
        for index in claimed:
            claimant, claim_log, result, found = run.claim(
                index, SMALL, "write", f"cd-write-{index}"
            )
            writers[index] = (claimant, claim_log, result)
            run.say(f"[cd] write index {index} (worker pid {claimant.pid}): {found}")
        guards, health = _guards(work), _health(run.port)
        run.say(f"[cd] before kill: guards {guards}, /ready {health}")
        written_ok = all(
            r["sha256"] == r["pattern_sha256"] for _, _, r in writers.values()
        )
        holders = run.state()["pids"]
        run.say(f"[cd] before kill: holder pods/pids {holders}")
        for name, pid in holders.items():
            run.say(f"[cd] before kill: holder {name} pid {pid} memfds {_memfds(pid)}")
        run.say(
            f"[cd] before kill: supervisor pid {first.pid} memfds {_memfds(first.pid)}"
        )
        table = _read(work / "slots" / "slot-table.json")
        run.say(f"[cd] before kill: slot-table.json {table}")
        uids = {n: p["metadata"]["uid"] for n, p in run.state()["pods"].items()}

        os.kill(first.pid, signal.SIGKILL)
        first.wait()
        run.say(f"[cd] SIGKILL supervisor pid {first.pid}: exited {first.returncode}")
        survivors = {
            "holders": {pid: _alive(pid) for pid in holders.values()},
            "workers": {w.pid: w.poll() is None for w, _, _ in writers.values()},
        }
        run.say(f"[cd] after kill, alive: {survivors}")

        second = run.start_service("service-cd-after.log")
        guards_restart, health_restart = _guards(work), _health(run.port)
        run.say(
            f"[cd] C: after restart, before any re-claim: guards {guards_restart}, "
            f"/ready {health_restart}"
        )
        fresh = _memfds(second.pid)
        run.say(f"[cd] after restart: supervisor pid {second.pid} memfds {fresh}")

        reattached = {}
        for index, (claimant, claim_log, result) in writers.items():
            found = _wait(
                f"re-attach {index}",
                lambda log=claim_log: (
                    _line(log, "REATTACHED ") or _line(log, "LEASE_ERROR")
                ),
                timeout_s,
                claimant,
            )
            run.say(f"[cd] C: worker pid {claimant.pid}: {found}")
            if "REATTACHED " not in found:
                raise RuntimeError(f"worker {index} did not re-attach: {found}")
            reattached[index] = json.loads(found.split("REATTACHED ", 1)[1])
        guards_reclaim, health_reclaim = _guards(work), _health(run.port)
        adopted = _memfds(second.pid)
        run.say(f"[cd] after re-attach: supervisor pid {second.pid} memfds {adopted}")
        run.say(
            f"[cd] C: after re-attach: guards {guards_reclaim}, /ready {health_reclaim}"
        )
        original = {  # (dev, inode) of each slot's region and meta memfds
            tuple(ident[:2])
            for _, _, r in writers.values()
            for ident in [*r["backing"]["regions"], r["backing"]["meta"]]
        }
        held_now = {(m[2], m[3]) for m in adopted}
        held_fresh = {(m[2], m[3]) for m in fresh}
        run.say(
            f"[cd] C: original memfds (dev, inode) {sorted(original)}; held by the "
            f"restarted supervisor after re-attach: {sorted(original & held_now)}; "
            f"right after restart (before handback): {sorted(original & held_fresh)}"
        )

        for index in claimed:
            (work / f"release-{index}").touch()
        for index, (claimant, claim_log, _) in writers.items():
            _wait(f"release {index}", lambda log=claim_log: _line(log, "RELEASED"), 30)
            claimant.wait(30)
        readers = {}
        for index in claimed:
            claimant, _, result, found = run.claim(
                index, SMALL, "read", f"cd-read-{index}"
            )
            readers[index] = result
            run.say(
                f"[cd] C: fresh read claim index {index} (pid {claimant.pid}): {found}"
            )
        guards_final, health_final = _guards(work), _health(run.port)
        run.say(
            f"[cd] after fresh claims: guards {guards_final}, /ready {health_final}"
        )
        uids_after = {n: p["metadata"]["uid"] for n, p in run.state()["pods"].items()}
        run.say(f"[cd] pod UIDs before kill {uids}, at end {uids_after}")

        checks = {
            "written": written_ok,
            "survived": all(survivors["holders"].values())
            and all(survivors["workers"].values()),
            "guards_restart": guards_restart == list(claimed),
            "health": all(
                h[0] == 200
                for h in (health, health_restart, health_reclaim, health_final)
            ),
            "reattach_same_backing": all(
                reattached[i]["backing"] == writers[i][2]["backing"]
                and reattached[i]["claim_generation"]
                == writers[i][2]["claim_generation"]
                for i in claimed
            ),
            "supervisor_adopted": original <= held_now and not original & held_fresh,
            "read_same_backing": all(
                readers[i]["backing"] == writers[i][2]["backing"] for i in claimed
            ),
            "read_same_bytes": all(
                readers[i]["sha256"] == writers[i][2]["sha256"] for i in claimed
            ),
            "guards_stay": guards_reclaim == guards_final == list(claimed),
            "no_new_holder": uids_after == uids,
        }
        run.say(f"[cd] C checks: {checks}")
        verdict["C"] = all(checks.values())
        absent = [
            all(i not in g for g in (guards_restart, guards_reclaim, guards_final))
            for i in never
        ]
        run.say(
            f"[cd] D: never-claimed indices {list(never)} absent from every guard list "
            f"after restart: {absent}"
        )
        verdict["D"] = all(absent) and guards_restart != []
    except Exception as error:  # noqa: BLE001 - reported, then cleaned up
        run.say(f"[cd] driver error: {type(error).__name__}: {error}")
    finally:
        run.close()
    return verdict


def main() -> int:
    if sys.argv[1:2] == ["_serve"]:
        _serve(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
        return 0
    if sys.argv[1:2] == ["_holder"]:
        _holder(sys.argv[2:])
        return 0
    if sys.argv[1:2] == ["_claim"]:
        socket_path, index, blocks, mode, work, port = sys.argv[2:8]
        return _claim(socket_path, int(index), int(blocks), mode, Path(work), int(port))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workdir", type=Path, required=True, help="must not exist")
    parser.add_argument("--logdir", type=Path, required=True, help="logs, created")
    parser.add_argument("--phase", choices=("ab", "cd", "all"), default="all")
    parser.add_argument("--timeout-s", type=float, default=300, help="per wait")
    parser.add_argument(
        "--stub-nixl",
        action="store_true",
        help="phase ab only: skip each Guard's NIXL pool registration",
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux only: memfd, pidfd and /proc")
    args.workdir.mkdir(parents=True)
    args.logdir.mkdir(parents=True, exist_ok=True)
    log = args.logdir / "driver.log"
    sys.path.insert(0, str(SRC))
    _say(
        log,
        f"restart proof: phase {args.phase}, workdir {args.workdir}, python "
        f"{sys.executable}, UCX_TLS={os.environ.get('UCX_TLS')!r}, "
        f"KVCR_LOG_LEVEL={os.environ.get('KVCR_LOG_LEVEL')!r}",
    )
    verdict = {}
    if args.phase in ("ab", "all"):
        verdict |= _phase_ab(
            log, args.workdir / "ab", args.logdir, args.timeout_s, args.stub_nixl
        )
    if args.phase in ("cd", "all"):
        verdict |= _phase_cd(log, args.workdir / "cd", args.logdir, args.timeout_s)
    for claim, ok in verdict.items():
        _say(log, f"VERDICT {claim} {'PASS' if ok else 'FAIL'}")
    ok = bool(verdict) and all(verdict.values())
    _say(log, f"RESULT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
