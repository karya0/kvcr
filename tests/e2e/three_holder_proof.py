# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three-holder proof: a real kvcr-service takes three 161025622016 B v2 claims.

Linux only, no GPU, no cluster. From the repo root, inside the run's image:

    UCX_TLS='^ib,rc,ud,dc' KVCR_LOG_LEVEL=DEBUG PYTHONPATH=src \\
        python tests/e2e/three_holder_proof.py --arm a --expect granted \\
        --workdir /w --log /w/driver.log

Replays the gaas-grfull-<arm>-20261112-r3 supervisor argv (--guard-count 3,
--slot-cap-bytes 485378490368, its digest, manager id and holder image) and its three
workers' claims: tenant = lineage = the experiment name, guard_index 0, 1, 2, total =
minimum = maximum = 161025622016 B (3134 blocks of 51380224 B), one claim at a time.

--expect granted: all three are granted with effective_maximum_bytes 161025622016.
--expect h7: claim 0 is granted, claims 1 and 2 are refused with "H7 holder count or
capacity budget exceeded" (NodeBroker._check_budget before the budget change).
Exit 0 only if the expectation holds.

Real: kvcr.service.server.main (argv parser, _build_slots: SlotTable + NodeBroker with
toucher holders, the Guards with NIXL) in its own process; the holders (python -m
kvcr.service.holder's main, one process per holder pod, receiving the slot's memfds);
the claimants (KVCRClient.claim v2, one process per worker, liveness by pidfd).
Stood in:
  - Kubernetes pod API (InClusterPodAPI): tests/unit/test_node_service.py's FakeAPI
    records the pod (Running, node-a), and creating a holder pod starts the holder
    process with the pod's own args.
  - Holder populate of the data regions: holder._touch is skipped for region indices
    >= 0 (sparse memfds; the host cannot back 3 x 150 GiB). The meta memfd is still
    populated, and the holder still maps every region and reports touched_bytes.
  - --namespace/--node-name/--advertise-host 127.0.0.1 (pod env on the cluster) and
    --no-require-pod-identity (no pods here), as in test_gaas_s4_claim_e2e.
"""

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
SRC = SCRIPT.parents[2] / "src"
UNIT_TESTS = SCRIPT.parents[2] / "tests" / "unit"

BLOCK = 51380224  # --pool-sizes-gb 0.0478515625, one KV block
SIZE = 161025622016  # each r3 worker's claim, 3134 blocks
SLOT_CAP = 485378490368
DIGEST = "db6b57b2823276ab9eb6a80c1ef5dfc8b295757bac26aa16e92205f8c2469dd7"
KEY_SIZE = 36  # vLLM wip/gaas-g2-key-size supplies it when GaaSSettings has the field
H7 = "H7 holder count or capacity budget exceeded"
ARMS = {
    "a": {
        "name": "gaas-grfull-a-20261112-r3",
        "image": "7758b73b340fe8d63c1d3bcb7a515260d92b0d2418effa9dafb4e129c84b13bb",
        "extra": [],
    },
    "b": {
        "name": "gaas-grfull-b-20261112-r3",
        "image": "9edc24983698e9c25d13b7978ace39e71108f593303aba0b70a17376641c4ea3",
        "extra": ["--worker-lease-ttl-ms", "60000"],
    },
}


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _say(log: Path, text: str) -> None:
    line = f"{_stamp()} {text}"
    with log.open("a") as out:
        out.write(line + "\n")
    print(line, flush=True)


# -- service process ----------------------------------------------------------


def _serve(work: Path, argv: list[str]) -> None:
    """kvcr-service's main() with the pod API stood in."""
    import copy

    sys.path.insert(0, str(UNIT_TESTS))
    from test_node_service import FakeAPI

    import kvcr
    from kvcr.service import server

    class Pods(FakeAPI):
        """A holder pod is a record plus a holder process run with the pod's args."""

        def __init__(self) -> None:
            super().__init__()
            self.processes = {}

        def create(self, pod):
            pod = copy.deepcopy(pod)
            name = pod["metadata"]["name"]
            container = pod["spec"]["containers"][0]
            container["readinessProbe"] = {}  # FakeAPI sets it; toucher pods have none
            command = [*container["command"], *container["args"]]
            print(
                f"THREE_HOLDER_PROOF pod create {name} command {command} "
                f"resources {json.dumps(container.get('resources'))}",
                file=sys.stderr,
                flush=True,
            )
            assert command[:3] == ["python", "-m", "kvcr.service.holder"], command
            created = super().create(pod)
            with (work / f"holder-{name}.log").open("ab") as out:
                self.processes[name] = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "_holder", *command[3:]],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
            return created

        def delete(self, name, uid):
            super().delete(name, uid)
            process = self.processes.pop(name, None)
            if process is not None:
                process.terminate()

    print(f"THREE_HOLDER_PROOF kvcr from {kvcr.__file__}", file=sys.stderr, flush=True)
    server.InClusterPodAPI = lambda namespace: Pods()
    sys.argv = ["kvcr-service", *argv]
    server.main()


def _service_argv(work: Path, arm: dict) -> list[str]:
    return [
        "--socket-path", str(work / "service.sock"),
        "--pool-dir", str(work / "pools"),
        "--guard-count", "3",
        "--pool-sizes-gb", "0.0478515625",
        "--compatibility-digest", DIGEST,
        "--slot-cap-bytes", str(SLOT_CAP),
        "--holder-image", "nvcr.io/0980761089281446/kapila-dev@sha256:" + arm["image"],
        "--slot-root", str(work / "slots"),
        "--manager-id", arm["name"],
        "--disable-hugepages",
        *arm["extra"],
        # Local only: pod env on the cluster, and no pods to identify here.
        "--namespace", "bench",
        "--node-name", "node-a",  # where FakeAPI places holders
        "--advertise-host", "127.0.0.1",
        "--no-require-pod-identity",
    ]  # fmt: skip


# -- holder process -----------------------------------------------------------


def _holder(argv: list[str]) -> None:
    """kvcr.service.holder's main(), data regions left sparse."""
    from kvcr.service import holder

    populate = holder._touch

    def touch(view, size, index=0):
        if index < 0:  # the meta memfd: populated as on the cluster
            return populate(view, size, index)
        holder._log("populate_skipped_sparse", region=index, bytes=size)

    holder._touch = touch
    sys.argv = ["kvcr.service.holder", *argv]
    holder.main()


# -- claimant process ---------------------------------------------------------


def _claim(socket_path: str, name: str, guard_index: int) -> int:
    """One v2 claim as an r3 worker sends it; prints RESULT, then holds the lease."""
    from kvcr.config import GaaSSettings
    from kvcr.service.protocol import KVCRClient

    gaas = {
        "tenant": name,
        "lineage": name,
        "total_bytes": SIZE,
        "minimum_bytes": SIZE,
        "maximum_bytes": SIZE,
    }
    if "key_size_bytes" in GaaSSettings.__struct_fields__:
        gaas["key_size_bytes"] = KEY_SIZE
    with socket.socket() as probe:  # the Guard binds the claimant's control address
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    started = time.monotonic()
    try:
        hold = KVCRClient(socket_path).claim(
            guard_index,
            [("", BLOCK)],
            DIGEST,
            ("127.0.0.1", port),
            gaas=GaaSSettings(**gaas),
        )
    except Exception as error:  # noqa: BLE001 - the refusal is the result
        result = {
            "guard_index": guard_index,
            "refused": type(error).__name__,
            "reason": getattr(error, "reason", None),
            "message": str(error),
        }
        code = 2
    else:
        result = {
            "guard_index": guard_index,
            "granted": hold.slot_name,
            "effective_maximum_bytes": hold.effective_maximum_bytes,
            "region_bytes": sum(region.size for region in hold.regions),
            "claim_generation": hold.claim_generation,
        }
        code = 0
    result["gaas"] = gaas
    result["after_s"] = round(time.monotonic() - started, 3)
    print("RESULT " + json.dumps(result), flush=True)
    if code == 0:
        signal.pause()  # hold the lease until the driver kills this worker
    return code


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


def _budget_lines() -> list[str]:
    """The budget check's limit lines in the node_service.py this run imports."""
    text = (SRC / "kvcr" / "service" / "node_service.py").read_text().splitlines()
    pattern = re.compile(r"len\(counted\) >=|\* _GIB$|raise ValueError\(\"H7")
    return [
        f"node_service.py:{i}: {x.strip()}"
        for i, x in enumerate(text, 1)
        if pattern.search(x)
    ]


def _run(log: Path, work: Path, arm: dict, expect: str, timeout_s: float) -> bool:
    (work / "pools").mkdir(parents=True)
    (work / "slots").mkdir(mode=0o700)
    path = [str(SRC), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, path)))
    service_log = work / "service.log"
    argv = _service_argv(work, arm)
    _say(log, f"service argv: kvcr-service {' '.join(argv)}")
    for line in _budget_lines():
        _say(log, f"budget check under test: {line}")
    with service_log.open("wb") as out:
        service = subprocess.Popen(
            [sys.executable, str(SCRIPT), "_serve", str(work), *argv],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # holders join its process group
        )
    claimants: list[subprocess.Popen] = []
    results: list[dict] = []
    try:
        ready = _wait(
            "service ready",
            lambda: _line(service_log, "KVCR service ready"),
            timeout_s,
            service,
        )
        _say(log, f"service pid {service.pid}: {ready.strip()}")
        for guard_index in (0, 1, 2):
            claim_log = work / f"claim-{guard_index}.log"
            with claim_log.open("wb") as out:
                claimant = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "_claim", str(work / "service.sock")]
                    + [arm["name"], str(guard_index)],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            claimants.append(claimant)
            found = _wait(
                f"claim {guard_index}",
                lambda: _line(claim_log, "RESULT "),
                timeout_s,
                service,
                claimant,
            )
            result = json.loads(found.split("RESULT ", 1)[1])
            results.append(result)
            _say(log, f"claim {guard_index + 1} (worker pid {claimant.pid}) {found}")
        table = json.loads((work / "slots" / "slot-table.json").read_text())
        for name, slot in sorted(table.get("slots", {}).items()):
            _say(
                log,
                f"slot-table.json slots[{name}].effective_maximum_bytes="
                f"{slot.get('effective_maximum_bytes')}",
            )
        for path in sorted((work / "slots" / "state").glob("*.json")):
            state = json.loads(path.read_text())
            _say(
                log,
                f"broker state {path.name}: pod {state.get('pod_name')} phase "
                f"{state.get('phase')} claim {json.dumps(state.get('claim'))}",
            )
        for path in sorted(work.glob("holder-*.log")):
            for line in _read(path).splitlines():
                _say(log, f"{path.name}: {line}")
    except Exception as error:  # noqa: BLE001 - reported, then cleaned up
        _say(log, f"driver error: {type(error).__name__}: {error}")
        return False
    finally:
        for claimant in claimants:
            claimant.kill()
            claimant.wait()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(service.pid, signal.SIGTERM)
        try:
            service.wait(30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(service.pid, signal.SIGKILL)
            service.wait()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(service.pid, signal.SIGKILL)  # holders left in the group
        _say(log, f"service exited {service.returncode}; claimants and holders killed")

    if expect == "granted":
        return len(results) == 3 and all(
            r.get("granted") == f"{arm['name']}/{arm['name']}/{r['guard_index']}"
            and r.get("effective_maximum_bytes") == SIZE
            and r.get("region_bytes") == SIZE
            for r in results
        )
    return (
        len(results) == 3
        and results[0].get("effective_maximum_bytes") == SIZE
        and all(H7 in (r.get("message") or "") for r in results[1:])
    )


def main() -> int:
    if sys.argv[1:2] == ["_serve"]:
        _serve(Path(sys.argv[2]), sys.argv[3:])
        return 0
    if sys.argv[1:2] == ["_holder"]:
        _holder(sys.argv[2:])
        return 0
    if sys.argv[1:2] == ["_claim"]:
        return _claim(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--expect", choices=("granted", "h7"), required=True)
    parser.add_argument("--workdir", type=Path, required=True, help="must not exist")
    parser.add_argument("--log", type=Path, required=True, help="driver log (appended)")
    parser.add_argument("--timeout-s", type=float, default=300, help="per wait")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux only: memfd, pidfd and /proc")
    arm = ARMS[args.arm]
    _say(
        args.log,
        f"three-holder proof: arm {args.arm} ({arm['name']}), expect {args.expect}, "
        f"workdir {args.workdir}, python {sys.executable}, "
        f"UCX_TLS={os.environ.get('UCX_TLS')!r}, "
        f"KVCR_LOG_LEVEL={os.environ.get('KVCR_LOG_LEVEL')!r}",
    )
    ok = _run(args.log, args.workdir, arm, args.expect, args.timeout_s)
    _say(args.log, f"RESULT {'PASS' if ok else 'FAIL'} (expect {args.expect})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
