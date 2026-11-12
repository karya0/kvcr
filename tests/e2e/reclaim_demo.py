# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reclaim demonstration: a real kvcr-service reclaims an expired retained slot.

Linux only, no GPU, no cluster. From the repo root:

    UCX_TLS='^ib,rc,ud,dc' PYTHONPATH=src python tests/e2e/reclaim_demo.py \\
        --log /tmp/reclaim-demo.log

Exit 0 only if every step below holds; one timestamped line per step goes to --log.
Each phase runs its own fresh service process; both run the same sequence and
the same waits, and differ only in --retention-seconds.

positive (--retention-seconds 2):
  1. Claim A (lineage a) is served. Its worker is killed and DGD a is deleted, so
     the service logs "lineage retained"; the wait then passes A's TTL.
  2. Claim B (lineage b) does not fit beside A's slot: the slot cap is
     headroom + 1.5 slots.
  3. The service logs "slot reclaimed slot=t/a/0 reason=expired".
  4. slot-table.json keeps A's bytes under "retiring" while A's holder process
     still runs, and drops them only after it has exited.
  5. Claim B is served.
negative (--retention-seconds 3600): B is refused slot_held "capacity retained by
  live or unexpired lineage; retry", nothing is reclaimed, A's slot and holder stay.

Real: the kvcr-service entry point (kvcr.service.server.main: its argv parser,
_build_slots, SlotTable persisting slot-table.json, the Guards with NIXL), one
process per phase; the holders (python -m kvcr.service.holder, a process per
holder pod, populating the slot's memfds); the claimants (KVCRClient.claim v2, a
process per worker, liveness by pidfd).
Stand-ins (inside the service process unless noted):
  - Kubernetes pod API (InClusterPodAPI): tests/unit/test_gaas_s4_claim_e2e.py's
    _HolderProcessAPI, which records pods and starts the real holder process for
    each. Here a deleted pod stays, terminating, until its holder process has
    exited, and the holder is sent SIGTERM --holder-exit-delay-s after the
    delete (a holder slow to exit).
  - DGD API (InClusterDGDAPI): a JSON file of DGD CRs that this driver rewrites;
    removing a lineage's entry is "DGD deleted".
  - State agent (python -m dynamo.kv_state_agent, which the supervisor spawns): a
    module, written to the work directory, that sleeps until SIGTERM.
  - Host hugetlbfs mounts are hidden from the service, as in
    test_kvcr_service_workflow's daemon; --no-require-pod-identity, as in the S4
    harness (no pods on the host).
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
SRC = SCRIPT.parents[2] / "src"
UNIT_TESTS = SCRIPT.parents[2] / "tests" / "unit"

BLOCK = 51380224  # one KV block, as in test_gaas_s4_claim_e2e
TOTAL = 4 * BLOCK  # every claim's total = minimum = maximum
HEADROOM = 2 << 30  # node_service._HEADROOM_BYTES
CAP = HEADROOM + TOTAL + TOTAL // 2  # one slot fits, two do not
DIGEST = "reclaim-demo-v1"
HOLDER_IMAGE = "example.test/holder@sha256:" + "a" * 64  # never pulled
TTL_S = {"positive": 2, "negative": 3600}  # --retention-seconds per phase
TTL_WAIT_S = 4  # both phases claim B this long after A's lineage was retained
RECLAIMED = "slot reclaimed slot=t/a/0 reason=expired"
REFUSED = "capacity retained by live or unexpired lineage; retry"
STATE_AGENT = '''"""reclaim_demo.py stand-in for dynamo.kv_state_agent."""
import signal
import sys
import time

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(60)
'''


class StepFailed(Exception):
    def __init__(self, message: str, logged: bool = False) -> None:
        super().__init__(message)
        self.logged = logged


class Report:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.failed = False

    def line(self, text: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self.path.open("a") as log:
            log.write(f"{stamp} {text}\n")
        print(f"{stamp} {text}", flush=True)

    def step(self, phase: str, step: int, ok: bool, evidence: str) -> None:
        self.line(f"{phase} step {step} {'PASS' if ok else 'FAIL'}: {evidence}")
        if not ok:
            raise StepFailed(f"{phase} step {step} failed", logged=True)


# -- service process ----------------------------------------------------------


def _serve(work: Path, exit_delay_s: float, argv: list[str]) -> None:
    """kvcr-service's main() with the pod, DGD and hugetlbfs stand-ins."""
    sys.path.insert(0, str(UNIT_TESTS))
    from test_gaas_s4_claim_e2e import FakeAPI, _HolderProcessAPI

    from kvcr.service import memmap, server

    class Pods(_HolderProcessAPI):
        """A deleted holder pod stays, terminating, until its holder has exited."""

        def __init__(self) -> None:
            super().__init__()
            self.keep_deleted = True
            self.lock = threading.RLock()

        def create(self, pod):
            with self.lock:
                return super().create(pod)

        def list(self, manager):
            with self.lock:
                return super().list(manager)

        def delete(self, name, uid):
            with self.lock:
                FakeAPI.delete(self, name, uid)  # marks the pod terminating
                process = self.processes.get(name)
            if process is not None:
                timer = threading.Timer(exit_delay_s, process.terminate)
                timer.daemon = True
                timer.start()

        def get(self, name):
            with self.lock:
                pod, process = self.pods.get(name), self.processes.get(name)
                if (
                    pod is not None
                    and pod["metadata"].get("deletionTimestamp")
                    and (process is None or process.poll() is not None)
                ):
                    del self.pods[name]
                    self.processes.pop(name, None)
                return super().get(name)

    class DGDs:
        """DGD CRs by name, from the JSON file the driver rewrites."""

        def get(self, name):
            return json.loads((work / "dgds.json").read_text()).get(name)

    memmap._hugetlbfs_mounts = lambda: []
    server.InClusterPodAPI = lambda namespace: Pods()
    server.InClusterDGDAPI = lambda namespace: DGDs()
    sys.argv = ["kvcr-service", *argv]
    server.main()


def _service_argv(work: Path, ttl_s: float) -> list[str]:
    return [
        "--socket-path", str(work / "service.sock"),
        "--pool-dir", str(work / "pools"),
        "--guard-count", "1",
        "--pool-sizes-gb", "0.0478515625",  # one BLOCK
        "--compatibility-digest", DIGEST,
        "--slot-cap-bytes", str(CAP),
        "--holder-image", HOLDER_IMAGE,
        "--slot-root", str(work / "slots"),
        "--manager-id", "gaas-reclaim-demo",
        "--disable-hugepages",
        "--namespace", "bench",
        "--node-name", "node-a",  # where the stand-in pod API places holders
        "--advertise-host", "127.0.0.1",
        "--no-require-pod-identity",
        "--state-agent-mode", "supervisor",  # the mode that reads DGDs
        "--retention-seconds", str(ttl_s),
        "--dgd-poll-seconds", "0.5",
        "--log-level", "INFO",
    ]  # fmt: skip


# -- claimant process ---------------------------------------------------------


def _claim(socket_path: str, lineage: str, namespace: str, port: int) -> int:
    """One v2 claim; prints RESULT, then holds the lease until killed."""
    from kvcr.config import GaaSSettings
    from kvcr.service.protocol import KVCRClient, KVCRServiceReasonError

    started = time.monotonic()
    try:
        hold = KVCRClient(socket_path).claim(
            0,
            [("", BLOCK)],
            DIGEST,
            ("127.0.0.1", port),
            gaas=GaaSSettings(
                tenant="t",
                lineage=lineage,
                total_bytes=TOTAL,
                minimum_bytes=TOTAL,
                maximum_bytes=TOTAL,
                state_agent={"namespace": namespace},
            ),
        )
    except KVCRServiceReasonError as error:
        result = {"refused": error.reason, "message": error.message}
        code = 2
    else:
        result = {
            "granted": hold.slot_name,
            "bytes": hold.effective_maximum_bytes,
            "claim_generation": hold.claim_generation,
        }
        code = 0
    result["after_s"] = round(time.monotonic() - started, 3)
    print("RESULT " + json.dumps(result), flush=True)
    if code == 0:
        threading.Event().wait()
    return code


# -- driver -------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _log_line(path: Path, needle: str) -> str | None:
    return next((x.strip() for x in _read(path).splitlines() if needle in x), None)


def _result(path: Path) -> dict | None:
    line = _log_line(path, "RESULT ")
    return json.loads(line.split("RESULT ", 1)[1]) if line else None


def _table(work: Path) -> dict:
    return json.loads((work / "slots" / "slot-table.json").read_text())


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def _wait(what: str, predicate, timeout_s: float, *processes):
    deadline = time.monotonic() + timeout_s
    while True:
        value = predicate()
        if value:
            return value
        for process in processes:
            if process.poll() is not None:
                value = predicate()  # it may have answered just before exiting
                if value:
                    return value
                raise StepFailed(
                    f"{process.args[2]} process exited {process.returncode} "
                    f"while waiting for {what}"
                )
        if time.monotonic() > deadline:
            raise StepFailed(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.1)


def _holder(work: Path, holder_claim: dict) -> tuple[str, int | None]:
    """A holder's pod name (broker state) and its process ID (/proc)."""
    for path in (work / "slots" / "state").glob("*.json"):
        state = json.loads(path.read_text())
        if state["claim"]["pool_id"] == holder_claim["pool_id"]:
            holder_socket = str(Path(state["socket_path"]).with_name("holder.sock"))
            for proc in Path("/proc").iterdir():
                with contextlib.suppress(OSError):
                    args = (proc / "cmdline").read_bytes().split(b"\0")
                    if (
                        b"kvcr.service.holder" in args
                        and os.fsencode(holder_socket) in args
                    ):
                        return state["pod_name"], int(proc.name)
            return state["pod_name"], None
    raise StepFailed(f"no broker state for holder {holder_claim}")


def _holder_rss_kb(pid: int | None) -> int | None:
    """The holder's resident shared memory in kB; None once it has exited."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except (OSError, TypeError):
        return None
    fields = dict(x.split(":", 1) for x in status.splitlines() if ":" in x)
    if fields["State"].split()[0] in ("Z", "X"):
        return None
    return int(fields.get("RssShmem", "0 kB").split()[0])


def _stamp(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


def _start_claim(work, lineage, namespace, env, claimants) -> subprocess.Popen:
    from _kvcr_test_utils import free_port

    with (work / f"claim-{lineage}.log").open("wb") as out:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "_claim", str(work / "service.sock")]
            + [lineage, namespace, str(free_port())],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
        )
    claimants.append(process)
    return process


def _run(report, name, work, service, claimants, env) -> None:
    log = work / "service.log"
    _wait("service ready", lambda: _log_line(log, "KVCR service ready"), 120, service)

    # 1. A served; its worker dies and its DGD goes: retained, then past the TTL.
    a = _start_claim(work, "a", "ns-a", env, claimants)
    granted = _wait("claim A", lambda: _result(work / "claim-a.log"), 240, service, a)
    slot_a = _table(work)["slots"].get("t/a/0") or {}
    pod_a, pid_a = _holder(work, slot_a["holder"]) if slot_a else (None, None)
    agent_a = _log_line(log, "lineage agent started lineage=t/a")
    a.kill()
    a.wait()
    _write_json(work / "dgds.json", {})
    retained = _wait(
        "lineage retained",
        lambda: _log_line(log, "lineage retained lineage=t/a slots=1"),
        30,
        service,
    )

    def persisted() -> dict | None:  # poll_dgds logs the retention, then persists
        rec = _table(work)["lineages"]["t/a"]
        return rec if rec["retained_at"] is not None else None

    rec = _wait("slot-table lineages[t/a].retained_at", persisted, 30, service)
    time.sleep(max(0.0, rec["retained_at"] + TTL_WAIT_S - time.time()))
    age = time.time() - rec["retained_at"]
    report.step(
        name,
        1,
        granted.get("granted") == "t/a/0"
        and slot_a.get("effective_maximum_bytes") == TOTAL
        and pid_a is not None
        and agent_a is not None
        and rec["retention_s"] == TTL_S[name],
        f"claim A RESULT {granted}; slot-table slots[t/a/0].effective_maximum_bytes="
        f"{slot_a.get('effective_maximum_bytes')}, holder pod {pod_a} pid {pid_a}; "
        f"log {agent_a!r}; claimant A pid {a.pid} killed (SIGKILL) and DGD a "
        f"deleted; log {retained!r}; slot-table lineages[t/a]={rec}; "
        f"retained {age:.1f} s ago, retention_s {rec['retention_s']}",
    )

    # 2. B does not fit beside A's charge.
    table = _table(work)
    charged = sum(s["effective_maximum_bytes"] for s in table["slots"].values())
    charged += sum(c for c, _ in table["retiring"].values())
    available = CAP - HEADROOM - charged
    report.step(
        name,
        2,
        "t/a/0" in table["slots"] and available < TOTAL,
        f"claim B (lineage b) needs {TOTAL} B; slot cap {CAP} - headroom {HEADROOM}"
        f" - charged {charged} (slot-table slots {sorted(table['slots'])}, "
        f"retiring {table['retiring']}) = {available} B available",
    )
    _write_json(work / "dgds.json", {"b": {"metadata": {"uid": f"{name}-dgd-b"}}})
    b = _start_claim(work, "b", "ns-b", env, claimants)
    snapshots, last = [], None
    deadline = time.monotonic() + 240
    while _result(work / "claim-b.log") is None:
        if b.poll() is not None or service.poll() is not None:
            break
        if time.monotonic() > deadline:
            raise StepFailed("timed out after 240 s waiting for claim B")
        table = _table(work)
        slots = tuple(sorted(table["slots"]))
        retiring = {pod: c for pod, (c, _) in table["retiring"].items()}
        rss = _holder_rss_kb(pid_a)
        if (slots, retiring, rss is None) != last:
            snapshots.append((time.time(), slots, retiring, rss))
            last = (slots, retiring, rss is None)
        time.sleep(0.02)
    result_b = _result(work / "claim-b.log")
    if result_b is None:
        raise StepFailed(f"claim B gave no RESULT (see {work / 'claim-b.log'})")
    lines = _read(log).splitlines()
    table = _table(work)

    def index(needle: str) -> int | None:
        return next((i for i, x in enumerate(lines) if needle in x), None)

    if name == "negative":
        report.step(
            name,
            3,
            index("slot reclaimed") is None,
            f"no 'slot reclaimed' line in {len(lines)} lines of {log}",
        )
        rss = _holder_rss_kb(pid_a)
        bytes_a = table["slots"].get("t/a/0", {}).get("effective_maximum_bytes")
        report.step(
            name,
            4,
            bytes_a == TOTAL and not table["retiring"] and rss is not None,
            f"slot-table slots {sorted(table['slots'])}, slots[t/a/0]."
            f"effective_maximum_bytes={bytes_a}, retiring {table['retiring']}; "
            f"holder pid {pid_a} running, RssShmem {rss} kB",
        )
        refused = _log_line(log, f"claim refused reason=slot_held lineage=b: {REFUSED}")
        report.step(
            name,
            5,
            result_b.get("refused") == "slot_held"
            and result_b.get("message") == REFUSED
            and refused is not None
            and "t/b/0" not in table["slots"],
            f"claim B RESULT {result_b}; log {refused!r}; slot-table slots "
            f"{sorted(table['slots'])}",
        )
        return

    stopped = index("lineage agent stopped lineage=t/a")
    reclaimed = index(RECLAIMED)
    agent_b = index("lineage agent started lineage=t/b")
    report.step(
        name,
        3,
        None not in (stopped, reclaimed, agent_b) and stopped < reclaimed < agent_b,
        f"log {lines[reclaimed] if reclaimed is not None else None!r} at line "
        f"{reclaimed}, after 'lineage agent stopped lineage=t/a' (line {stopped}) "
        f"and before 'lineage agent started lineage=t/b' (line {agent_b}) of {log}",
    )

    # 4. A's bytes stay charged while its holder runs; uncharged once it exited.
    held = next(
        (
            s
            for s in snapshots
            if "t/a/0" not in s[1] and s[2].get(pod_a) == TOTAL and s[3] is not None
        ),
        None,
    )
    freed = held and next(
        (s for s in snapshots[snapshots.index(held) :] if pod_a not in s[2]), None
    )
    leaked = [
        s
        for s in snapshots
        if "t/a/0" not in s[1] and pod_a not in s[2] and s[3] is not None
    ]
    report.step(
        name,
        4,
        bool(held and freed and freed[3] is None and not leaked),
        (
            f"{_stamp(held[0])} slot-table slots {list(held[1])} retiring "
            f"{held[2]} while holder pid {pid_a} runs (RssShmem {held[3]} kB); "
            if held
            else "no snapshot with A released, charged and its holder running; "
        )
        + (
            f"{_stamp(freed[0])} slots {list(freed[1])} retiring {freed[2]}, holder "
            f"pid {pid_a} {'exited' if freed[3] is None else 'running'}, "
            f"{freed[0] - held[0]:.2f} s later; "
            if freed
            else "never uncharged; "
        )
        + f"{len(leaked)} snapshots uncharged with the holder running "
        f"({len(snapshots)} distinct snapshots)",
    )
    report.step(
        name,
        5,
        result_b.get("granted") == "t/b/0"
        and table["slots"].get("t/b/0", {}).get("effective_maximum_bytes") == TOTAL
        and "t/a/0" not in table["slots"]
        and not table["retiring"],
        f"claim B RESULT {result_b} (its wait includes the reclaim); slot-table "
        f"slots {sorted(table['slots'])} retiring {table['retiring']}; log "
        f"{lines[agent_b].strip()!r}",
    )


def _phase(report: Report, root: Path, name: str, exit_delay_s: float) -> None:
    work = root / name
    (work / "pools").mkdir(parents=True)
    (work / "slots").mkdir(mode=0o700)
    agent = work / "standin" / "dynamo"
    agent.mkdir(parents=True)
    (agent / "__init__.py").write_text("")
    (agent / "kv_state_agent.py").write_text(STATE_AGENT)
    _write_json(work / "dgds.json", {"a": {"metadata": {"uid": f"{name}-dgd-a"}}})
    path = [str(work / "standin"), str(SRC), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, path)))
    report.line(
        f"{name} start: retention_s {TTL_S[name]}, slot cap {CAP}, claim {TOTAL} B, "
        f"service log {work / 'service.log'}"
    )
    with (work / "service.log").open("wb") as out:
        service = subprocess.Popen(
            [sys.executable, str(SCRIPT), "_serve", str(work), str(exit_delay_s)]
            + _service_argv(work, TTL_S[name]),
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # holders and the agent stand-in join its group
        )
    claimants: list[subprocess.Popen] = []
    try:
        _run(report, name, work, service, claimants, env)
    except StepFailed as error:
        report.failed = True
        if not error.logged:
            report.line(f"{name} FAIL: {error}; see {work}")
    except Exception as error:  # noqa: BLE001 - reported, then the next phase runs
        report.failed = True
        traceback.print_exc()
        report.line(f"{name} FAIL: {type(error).__name__}: {error}; see {work}")
    finally:
        for process in claimants:
            process.kill()
            process.wait()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(service.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            service.wait(15)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(service.pid, signal.SIGKILL)
        service.wait()


def main() -> int:
    if sys.argv[1:2] == ["_serve"]:
        _serve(Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
        return 0
    if sys.argv[1:2] == ["_claim"]:
        return _claim(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, required=True, help="step log (appended)")
    parser.add_argument(
        "--workdir", type=Path, help="kept for inspection (default: new under /tmp)"
    )
    parser.add_argument(
        "--holder-exit-delay-s",
        type=float,
        default=2.0,
        help="stand-in pod API: SIGTERM a deleted holder this long after the delete",
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux only: memfd, pidfd and /proc")
    root = args.workdir or Path(tempfile.mkdtemp(prefix="rcl-", dir="/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(UNIT_TESTS))  # free_port, for the claimants' Guard binds
    report = Report(args.log)
    report.line(
        f"reclaim demo: workdir {root}, python {sys.executable}, "
        f"UCX_TLS={os.environ.get('UCX_TLS')!r}"
    )
    for name in ("positive", "negative"):
        _phase(report, root, name, args.holder_exit_delay_s)
    report.line(f"RESULT {'FAIL' if report.failed else 'PASS'}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
