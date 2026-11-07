# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Namespace-scoped holder broker. Pool data never travels through this socket.

The holder runs KVCR-Service; clients connect to it directly. A worker disappearing
does not release its pool. Explicit release is durable and deletion requires a
terminal drain acknowledgment from that same holder.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import select
import signal
import socket
import socketserver
import ssl
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

_GIB = 1 << 30
_JOURNAL_BYTES = 100 << 20
_HEADROOM_BYTES = 2 * _GIB
_LABEL = "kvcr.nvidia.com/manager"
_ANNOTATION = "kvcr.nvidia.com/"
_MAX_MESSAGE = 65536
_TOUCHER_OVERHEAD_BYTES = 128 << 20
_EXIT_REASON_DEADLINE_S = 5.0


def _toucher_overhead_bytes(data_bytes: int) -> int:
    """Fixed holder RSS plus kernel memory charged to the holder's memcg.

    Populating N bytes of 4 KiB shmem pages charges about N / 235 of page tables
    and page-cache slab (210 MB for 46 GiB, c5-repro-memcg-20261107.txt); N / 200
    leaves margin. The fixed 128 MiB covers the holder's own RSS (~46 MiB).
    """
    return _TOUCHER_OVERHEAD_BYTES + -(-data_bytes // 200)


_LINEAGE_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_DYNAMO_NAMESPACE_LABEL = "nvidia.com/dynamo-namespace"
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolClaim:
    pool_id: str
    size_gb: Decimal
    compatibility_digest: str

    @classmethod
    def parse(cls, value: dict[str, Any]) -> PoolClaim:
        if not isinstance(value, dict) or set(value) != {
            "pool_id",
            "size_gb",
            "compatibility_digest",
        }:
            raise ValueError("claim requires pool_id, size_gb and compatibility_digest")
        name, digest = value["pool_id"], value["compatibility_digest"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("pool_id must be a DNS label")
        if not isinstance(digest, str) or not 0 < len(digest) <= 4096:
            raise ValueError("compatibility_digest must be a nonempty string")
        try:
            size = Decimal(str(value["size_gb"]))
        except InvalidOperation as error:
            raise ValueError("invalid size_gb") from error
        if (
            not size.is_finite()
            or not Decimal("0.5") <= size <= 256
            or size != Decimal(int(size * 1024)) / 1024
        ):
            raise ValueError("holder size must be 0.5 to 256 GiB in whole MiB")
        return cls(name, size.normalize(), digest)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "size_gb": float(self.size_gb),
            "compatibility_digest": self.compatibility_digest,
        }

    @property
    def data_bytes(self) -> int:
        return int(self.size_gb * _GIB)

    def resources(self) -> dict[str, int]:
        # H7 bounds the snapshot tail to one additional mapping. Metadata has
        # no universal size bound; this is an explicit experiment capacity cap,
        # enforced by tmpfs and fully covered by the Pod reservation.
        mapping = self.data_bytes + _JOURNAL_BYTES
        return {
            "mapping_bytes": mapping,
            "snapshot_budget_bytes": mapping,
            "backing_limit_bytes": 2 * mapping,
            "memory_request_bytes": 2 * mapping + _HEADROOM_BYTES,
        }


def effective_grant(
    total: int, minimum: int, maximum: int | None, available: int
) -> int | None:
    """Bytes to grant a claim, or None when less than its minimum is available."""
    if available < minimum:
        return None
    return min(maximum or total, available)


def _matching_tolerations(actual: Any, intended: list[dict[str, Any]]) -> bool:
    """Accept only intent and the two standard DefaultTolerationSeconds entries."""

    def normalize(value: Any) -> tuple[Any, ...] | None:
        if not isinstance(value, dict) or set(value) - {
            "key",
            "operator",
            "value",
            "effect",
            "tolerationSeconds",
        }:
            return None
        return (
            value.get("key", ""),
            value.get("operator") or "Equal",
            value.get("value") or "",
            value.get("effect") or "",
            value.get("tolerationSeconds"),
        )

    if not isinstance(actual, list):
        return False
    expected = [normalize(value) for value in intended]
    allowed_defaults = {
        (f"node.kubernetes.io/{condition}", "Exists", "", "NoExecute", 300)
        for condition in ("not-ready", "unreachable")
    }
    observed = [normalize(value) for value in actual]
    if any(value is None for value in observed):
        return False
    # Consume rather than use a set: duplicate tolerations are not our template.
    for value in expected:
        if value not in observed:
            return False
        observed.remove(value)
    for value in observed:
        if value not in allowed_defaults:
            return False
        allowed_defaults.remove(value)
    return True


class PodAPI(Protocol):
    def get(self, name: str) -> dict[str, Any] | None: ...
    def get_by_uid(self, uid: str) -> dict[str, Any] | None: ...
    def list(self, manager: str) -> list[dict[str, Any]]: ...
    def create(self, pod: dict[str, Any]) -> dict[str, Any]: ...
    def delete(self, name: str, uid: str) -> None: ...


class PodAPIError(RuntimeError):
    def __init__(self, status: int | None, operation: str):
        self.status = status
        super().__init__(f"Pod API {operation} failed (status={status})")


class InClusterPodAPI:
    """Only namespace Pod endpoints; tokens and server error bodies are not logged."""

    def __init__(self, namespace: str):
        if not _NAME.fullmatch(namespace):
            raise ValueError("invalid namespace")
        host = os.environ["KUBERNETES_SERVICE_HOST"]
        port = int(os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443"))
        if any(c in host for c in "/?#@") or not 1 <= port <= 65535:
            raise ValueError("invalid Kubernetes service address")
        host = f"[{host}]" if ":" in host else host
        self._url = f"https://{host}:{port}/api/v1/namespaces/{namespace}/pods"
        service_account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
        self._token_path = service_account / "token"
        self._context = ssl.create_default_context(cafile=service_account / "ca.crt")

    def _request(self, method: str, suffix: str = "", body=None):
        request = urllib.request.Request(
            self._url + suffix,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                # Projected service-account credentials rotate while the
                # broker lives; resolve the mounted file on every request.
                "Authorization": f"Bearer {self._token_path.read_text().strip()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, context=self._context, timeout=10
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise PodAPIError(error.code, method) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise PodAPIError(None, method) from None

    def get(self, name: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", "/" + urllib.parse.quote(name, safe=""))
        except PodAPIError as error:
            if error.status == 404:
                return None
            raise

    def get_by_uid(self, uid: str) -> dict[str, Any] | None:
        # The API cannot select pods by UID; read the namespace and filter.
        for pod in self._request("GET")["items"]:
            if pod.get("metadata", {}).get("uid") == uid:
                return pod
        return None

    def list(self, manager: str) -> list[dict[str, Any]]:
        selector = urllib.parse.urlencode({"labelSelector": f"{_LABEL}={manager}"})
        return self._request("GET", "?" + selector)["items"]

    def create(self, pod: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", body=pod)

    def delete(self, name: str, uid: str) -> None:
        self._request(
            "DELETE",
            "/" + urllib.parse.quote(name, safe=""),
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {"uid": uid},
                "gracePeriodSeconds": 5,
            },
        )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".state-")
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(value, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as source:
        return json.load(source)


def _contains(actual: Any, expected: Any) -> bool:
    """Allow API-server default fields, but not changed intended values."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_contains(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return actual == expected


class NodeBroker:
    def __init__(
        self,
        *,
        namespace: str,
        node_name: str,
        manager_id: str,
        image: str,
        socket_root: str | Path,
        api: PodAPI,
        info=None,
        drain=None,
        poll_interval: float = 0.25,
        toucher: bool = False,
    ):
        if not _NAME.fullmatch(namespace) or not _NAME.fullmatch(manager_id):
            raise ValueError("invalid namespace or manager ID")
        if not node_name or len(node_name) > 253 or "/" in node_name:
            raise ValueError("invalid node name")
        if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", image):
            raise ValueError("holder image must be digest pinned")
        self.namespace, self.node_name = namespace, node_name
        self.manager_id, self.image, self.api = manager_id, image, api
        self.root = Path(socket_root)
        if not self.root.is_absolute():
            raise ValueError("socket root must be absolute")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = self.root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.geteuid():
            raise ValueError("socket root must be an owned directory, not a symlink")
        if root_stat.st_mode & 0o077:
            raise ValueError("socket root must be private (0700)")
        self.state_dir = self.root / "state"
        self.state_dir.mkdir(mode=0o700, exist_ok=True)
        if self.state_dir.is_symlink():
            raise ValueError("state directory is a symlink")
        identity = {"namespace": namespace, "node": node_name, "manager": manager_id}
        self.manager = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:32]
        identity_path = self.root / "manager.json"
        if identity_path.exists():
            if _read_json(identity_path) != identity:
                raise ValueError("socket root belongs to another manager")
        else:
            _atomic_json(identity_path, identity)
        self._info = info or self._service_info
        self._drain = drain or self._drain_service
        self._lock = threading.RLock()
        self.poll_interval = poll_interval
        # v2: the holder is a Guaranteed toucher fed by the supervisor, not a
        # kvcr-service pool holder.
        self.toucher = toucher

    @staticmethod
    def _service_info(path: str) -> dict[str, Any]:
        from .protocol import service_info

        return service_info(path, timeout_s=10)

    @staticmethod
    def _drain_service(path: str) -> dict[str, Any]:
        from .protocol import drain_service

        return drain_service(path, timeout_s=10)

    def _key(self, claim: PoolClaim) -> str:
        return hashlib.sha256(f"{self.manager}:{claim.pool_id}".encode()).hexdigest()[
            :24
        ]

    def _state_path(self, claim: PoolClaim) -> Path:
        return self.state_dir / f"{self._key(claim)}.json"

    def _save(self, claim: PoolClaim, state: dict[str, Any]) -> None:
        _atomic_json(self._state_path(claim), state)
        logger.info(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "holder_state",
                    "namespace": self.namespace,
                    "node_name": self.node_name,
                    "manager_identity": self.manager,
                    "pool_id": claim.pool_id,
                    "holder_uid": state["holder_uid"],
                    "holder_name": state["pod_name"],
                    "state": state["phase"],
                }
            )
        )

    def _load(self, claim: PoolClaim) -> dict[str, Any] | None:
        path = self._state_path(claim)
        if not path.exists():
            return None
        state = _read_json(path)
        if state["claim"] != claim.as_dict() or state["manager"] != self.manager:
            raise ValueError("pool identity or capacity differs from durable claim")
        if state["pod_name"] != "gaas-pool-" + self._key(claim) or state[
            "socket_path"
        ] != str(self.root / f"p-{self._key(claim)}" / "service.sock"):
            raise ValueError("durable holder path identity differs")
        return state

    def _record_info(
        self, claim: PoolClaim, state: dict[str, Any], info: dict[str, Any]
    ) -> None:
        if not info.get("pool_specs"):
            raise RuntimeError("holder did not provide pool identity")
        if info.get("compatibility_digest") != claim.compatibility_digest:
            raise RuntimeError("holder compatibility digest changed")
        if state.get("pool_specs") not in (None, info["pool_specs"]):
            raise RuntimeError("holder pool identity changed")
        state["pool_specs"] = info["pool_specs"]

    @staticmethod
    def holder_socket_path(state: dict[str, Any]) -> str:
        return str(Path(state["socket_path"]).with_name("holder.sock"))

    def _toucher_pod(
        self, pod: dict[str, Any], claim: PoolClaim, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Guaranteed QoS: one CPU, memory = pool maximum + overhead.

        Population is single-threaded and CPU-bound, so a fractional CPU would
        throttle it far below the ensure window.

        The overhead is a fixed part plus page tables, which the holder's memcg is
        charged for; sized any lower, populating a full pool is OOM-killed.
        """
        total = claim.data_bytes + _toucher_overhead_bytes(claim.data_bytes)
        memory = f"{-(-total // (1 << 20))}Mi"
        spec = pod["spec"]
        container = spec["containers"][0]
        container["command"] = ["python", "-m", "kvcr.service.holder"]
        container["args"] = ["--socket", self.holder_socket_path(state)]
        container.pop("readinessProbe")
        container["volumeMounts"] = container["volumeMounts"][:1]
        spec["volumes"] = spec["volumes"][:1]
        resources = {"cpu": "1", "memory": memory}
        container["resources"] = {"requests": dict(resources), "limits": resources}
        return pod

    def _pod(self, claim: PoolClaim, state: dict[str, Any]) -> dict[str, Any]:
        pod = self._service_pod(claim, state)
        return self._toucher_pod(pod, claim, state) if self.toucher else pod

    def _service_pod(self, claim: PoolClaim, state: dict[str, Any]) -> dict[str, Any]:
        resources = claim.resources()
        memory = f"{resources['memory_request_bytes'] >> 20}Mi"
        backing_size = f"{resources['backing_limit_bytes'] >> 20}Mi"
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": state["pod_name"],
                "namespace": self.namespace,
                "labels": {_LABEL: self.manager, "kvcr.nvidia.com/pool": claim.pool_id},
                "annotations": {
                    _ANNOTATION + "claim": json.dumps(claim.as_dict(), sort_keys=True),
                    _ANNOTATION + "creation-id": state["creation_id"],
                },
            },
            "spec": {
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchFields": [
                                        {
                                            "key": "metadata.name",
                                            "operator": "In",
                                            "values": [self.node_name],
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
                "restartPolicy": "Never",
                "automountServiceAccountToken": False,
                "securityContext": {
                    "runAsUser": 1000,
                    "runAsGroup": 1000,
                    "runAsNonRoot": True,
                    "fsGroup": 1000,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "imagePullSecrets": [{"name": "nvcrimagepullsecret"}],
                "tolerations": [
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Equal",
                        "value": "true",
                        "effect": "NoSchedule",
                    }
                ],
                "containers": [
                    {
                        "name": "holder",
                        "image": self.image,
                        "command": ["kvcr-service"],
                        "args": [
                            "--socket-path",
                            state["socket_path"],
                            "--pool-dir",
                            "/pool",
                            "--guard-count",
                            "1",
                            "--pool-sizes-gb",
                            str(claim.size_gb),
                            "--compatibility-digest",
                            claim.compatibility_digest,
                            "--disable-hugepages",
                            "--health-host",
                            "0.0.0.0",
                            "--health-port",
                            "18080",
                        ],
                        "env": [
                            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
                            {"name": "CUDA_VISIBLE_DEVICES", "value": "-1"},
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": memory},
                            "limits": {"cpu": "1", "memory": memory},
                        },
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                        },
                        "volumeMounts": [
                            {"name": "sockets", "mountPath": "/node-shm"},
                            {"name": "pool", "mountPath": "/pool"},
                        ],
                        "readinessProbe": {
                            "httpGet": {
                                "path": "/ready",
                                "port": 18080,
                                "scheme": "HTTP",
                            }
                        },
                    }
                ],
                "volumes": [
                    {
                        "name": "sockets",
                        "hostPath": {"path": "/dev/shm", "type": "Directory"},
                    },
                    {
                        "name": "pool",
                        "emptyDir": {
                            "medium": "Memory",
                            "sizeLimit": backing_size,
                        },
                    },
                ],
            },
        }

    def _verify_pod(
        self,
        claim: PoolClaim,
        state: dict[str, Any],
        pod: dict[str, Any],
        *,
        allow_terminating: bool = False,
    ) -> str:
        expected = self._pod(claim, state)
        metadata, spec = pod["metadata"], pod["spec"]
        if metadata.get("deletionTimestamp") and not allow_terminating:
            raise ValueError("holder is terminating outside acknowledged deletion")
        if (
            metadata.get("namespace") != self.namespace
            or metadata.get("name") != state["pod_name"]
        ):
            raise ValueError("holder namespace or name mismatch")
        if metadata.get("ownerReferences") or any(
            metadata.get("labels", {}).get(key) != value
            for key, value in expected["metadata"]["labels"].items()
        ):
            raise ValueError("holder ownership mismatch")
        for key, value in expected["metadata"]["annotations"].items():
            if metadata.get("annotations", {}).get(key) != value:
                raise ValueError("holder creation identity mismatch")
        for key in (
            "affinity",
            "restartPolicy",
            "automountServiceAccountToken",
            "securityContext",
            "imagePullSecrets",
            "volumes",
        ):
            if not _contains(spec.get(key), expected["spec"][key]):
                raise ValueError(f"holder {key} mismatch")
        if not _matching_tolerations(
            spec.get("tolerations", []), expected["spec"]["tolerations"]
        ):
            raise ValueError("holder tolerations mismatch")
        if spec.get("nodeName") not in (None, "", self.node_name):
            raise ValueError("holder assigned to another node")
        for key in (
            "hostNetwork",
            "hostPID",
            "hostIPC",
            "shareProcessNamespace",
            "initContainers",
            "ephemeralContainers",
        ):
            if spec.get(key):
                raise ValueError(f"unexpected holder {key}")
        # Reject injected containers: no unrelated process may own this pool.
        containers = spec.get("containers", [])
        if len(containers) != 1:
            raise ValueError("holder container count mismatch")
        for key, value in expected["spec"]["containers"][0].items():
            if not _contains(containers[0].get(key), value):
                raise ValueError(f"holder container {key} mismatch")
        security = containers[0].get("securityContext", {})
        if security.get("privileged") or security.get("capabilities", {}).get("add"):
            raise ValueError("unexpected holder privilege")
        if any(security.get(key, 1000) != 1000 for key in ("runAsUser", "runAsGroup")):
            raise ValueError("unexpected holder user")
        uid = metadata.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("holder has no UID")
        if state.get("holder_uid") not in (None, uid):
            raise ValueError("holder UID changed; refusing replacement")
        return uid

    def _check_budget(self, claim: PoolClaim) -> None:
        states = [_read_json(path) for path in self.state_dir.glob("*.json")]
        pods = {pod["metadata"]["name"]: pod for pod in self.api.list(self.manager)}
        known = {state["pod_name"] for state in states}
        if any(name not in known for name in pods):
            raise RuntimeError(
                "unreconciled holder exists; retain it and refuse allocation"
            )
        counted = []
        for state in states:
            pod = pods.get(state["pod_name"])
            if pod is not None and state.get("holder_uid") not in (
                None,
                pod["metadata"].get("uid"),
            ):
                raise RuntimeError(
                    "a holder was replaced under the same name; refusing allocation"
                )
            # A deleted-phase holder still in the API is terminating and keeps
            # its memory: it counts until the pod is gone.
            if state["phase"] != "deleted" or pod is not None:
                counted.append(state)
        if (
            len(counted) >= 2
            or sum(PoolClaim.parse(s["claim"]).data_bytes for s in counted)
            + claim.data_bytes
            > 256 * _GIB
        ):
            raise ValueError("H7 holder count or capacity budget exceeded")

    def ensure(self, value: dict[str, Any], timeout_s: float = 180) -> dict[str, Any]:
        claim = PoolClaim.parse(value)
        if not 0 < timeout_s <= 180:
            raise ValueError("ensure timeout must be in (0, 180]")
        with self._lock:
            state = self._load(claim)
            if state is None:
                self._check_budget(claim)
                directory = self.root / f"p-{self._key(claim)}"
                directory.mkdir(mode=0o700, exist_ok=True)
                if directory.is_symlink():
                    raise ValueError("pool socket directory is a symlink")
                socket_path = str(directory / "service.sock")
                if len(os.fsencode(socket_path)) > 107:
                    raise ValueError("pool socket path exceeds AF_UNIX limit")
                state = {
                    "claim": claim.as_dict(),
                    "manager": self.manager,
                    "creation_id": str(uuid.uuid4()),
                    "pod_name": "gaas-pool-" + self._key(claim),
                    "holder_uid": None,
                    "socket_path": socket_path,
                    "phase": "creating",
                }
                self._save(claim, state)  # creation identity survives an uncertain POST
            if state["phase"] != "creating" and state["phase"] != "ready":
                raise RuntimeError("pool has a release intent or is terminal")
            pod = self.api.get(state["pod_name"])
            if pod is None:
                if state["holder_uid"] is not None:
                    raise RuntimeError(
                        "holder disappeared; refusing silent reallocation"
                    )
                try:
                    pod = self.api.create(self._pod(claim, state))
                except PodAPIError:
                    pod = self.api.get(state["pod_name"])
                    if pod is None:
                        raise
            state["holder_uid"] = self._verify_pod(claim, state, pod)
            self._save(claim, state)
            deadline = time.monotonic() + timeout_s
            while True:
                if pod is None:
                    raise RuntimeError("holder disappeared while becoming ready")
                self._verify_pod(claim, state, pod)
                status = pod.get("status", {})
                if status.get("phase") in ("Failed", "Succeeded"):
                    raise RuntimeError(
                        "holder terminated; reservation and cache are not reusable"
                    )
                if status.get("phase") == "Running" and status.get("podIP"):
                    if pod["spec"].get("nodeName") != self.node_name:
                        raise ValueError(
                            "running holder has not been assigned to its node"
                        )
                    if self.toucher:
                        state.update(phase="ready", holder_pod_ip=status["podIP"])
                        self._save(claim, state)
                        return self._descriptor(state)
                    try:
                        info = self._info(state["socket_path"])
                    except (OSError, RuntimeError):
                        info = None
                    if info and info.get("state") == "ready":
                        self._record_info(claim, state, info)
                        state.update(
                            phase="ready",
                            pool_specs=info["pool_specs"],
                            holder_pod_ip=status["podIP"],
                        )
                        self._save(claim, state)
                        return self._descriptor(state)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "holder readiness timed out; reservation retained"
                    )
                time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
                pod = self.api.get(state["pod_name"])

    def _descriptor(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            **state["claim"],
            **PoolClaim.parse(state["claim"]).resources(),
            "namespace": self.namespace,
            "node_name": self.node_name,
            "manager_id": self.manager_id,
            "manager_identity": self.manager,
            "holder_name": state["pod_name"],
            "holder_uid": state["holder_uid"],
            "service_socket_path": state["socket_path"],
            "holder_pod_ip": state.get("holder_pod_ip"),
            "control_port": 55001,
            "guard_index": 0,
            "state": state["phase"],
            "pool_specs": state.get("pool_specs", []),
            "deleted": state["phase"] == "deleted",
            "drain": state.get("drain_ack"),
        }

    def inspect(self, value: dict[str, Any]) -> dict[str, Any]:
        claim = PoolClaim.parse(value)
        with self._lock:
            state = self._load(claim)
            if state is None:
                raise ValueError("unknown pool")
            if state["phase"] != "deleted":
                pod = self.api.get(state["pod_name"])
                if pod is None:
                    if not state.get("drain_ack"):
                        raise RuntimeError(
                            "holder disappeared without acknowledged drain"
                        )
                    state["phase"] = "deleted"
                    self._save(claim, state)
                    return self._descriptor(state)
                uid = self._verify_pod(
                    claim, state, pod, allow_terminating=bool(state.get("drain_ack"))
                )
                if state["holder_uid"] is None:
                    # Recover only the verified creation, without promoting
                    # readiness or allocating a replacement holder.
                    if str(uuid.UUID(uid)) != uid:
                        raise ValueError("holder UID must be a canonical UUID")
                    state["holder_uid"] = uid
                    self._save(claim, state)
            return self._descriptor(state)

    def release(self, value: dict[str, Any]) -> dict[str, Any]:
        claim = PoolClaim.parse(value)
        with self._lock:
            state = self._load(claim)
            if state is None:
                raise ValueError("unknown pool")
            if state["phase"] == "deleted":
                return self._descriptor(state)
            if state["phase"] not in ("release_requested", "drained", "deleting"):
                state["phase"] = "release_requested"
                self._save(claim, state)  # never roll this back on busy/timeout
            return self._finish_release(claim, state)

    def _finish_release(
        self, claim: PoolClaim, state: dict[str, Any]
    ) -> dict[str, Any]:
        pod = self.api.get(state["pod_name"])
        if pod is None:
            if not state.get("drain_ack"):
                raise RuntimeError("holder disappeared before drain was acknowledged")
            state["phase"] = "deleted"
            self._save(claim, state)
            return self._descriptor(state)
        uid = self._verify_pod(
            claim, state, pod, allow_terminating=bool(state.get("drain_ack"))
        )
        if state["holder_uid"] is None:
            state["holder_uid"] = uid
            self._save(claim, state)
        if self.toucher and not state.get("drain_ack"):
            state.update(phase="drained", drain_ack={"state": "drained"})
            self._save(claim, state)
        if not state.get("drain_ack"):
            if not state.get("pool_specs"):
                self._record_info(claim, state, self._info(state["socket_path"]))
                self._save(claim, state)
            result = self._drain(state["socket_path"])
            if result.get("state") != "drained":
                return {**self._descriptor(state), "drain": result}
            if (
                result.get("pool_specs") != state.get("pool_specs")
                or result.get("compatibility_digest") != claim.compatibility_digest
            ):
                raise RuntimeError("drain acknowledgment pool identity differs")
            state.update(phase="drained", drain_ack=result)
            self._save(claim, state)
        state["phase"] = "deleting"
        self._save(claim, state)
        self.api.delete(state["pod_name"], state["holder_uid"])
        remaining = self.api.get(state["pod_name"])
        if remaining is None:
            state["phase"] = "deleted"
            self._save(claim, state)
        else:
            self._verify_pod(claim, state, remaining, allow_terminating=True)
        return self._descriptor(state)

    def abandon(self, value: dict[str, Any]) -> None:
        """Retire a holder that died (gone, terminating or finished) so a
        replacement can be created under a new claim identity."""
        claim = PoolClaim.parse(value)
        with self._lock:
            state = self._load(claim)
            if state is None or state["phase"] == "deleted":
                return
            pod = self.api.get(state["pod_name"])
            if pod is not None:
                terminating = bool(pod["metadata"].get("deletionTimestamp"))
                if pod.get("status", {}).get("phase") not in ("Failed", "Succeeded"):
                    if not terminating:
                        raise RuntimeError("holder is alive; refusing to abandon it")
                if not terminating and pod["metadata"].get("uid"):
                    self.api.delete(state["pod_name"], pod["metadata"]["uid"])
            state["phase"] = "deleted"
            self._save(claim, state)

    def reconcile_releases(self) -> list[str]:
        errors = []
        with self._lock:
            for path in self.state_dir.glob("*.json"):
                state = _read_json(path)
                if state["phase"] in ("release_requested", "drained", "deleting"):
                    try:
                        claim = PoolClaim.parse(state["claim"])
                        verified = self._load(claim)
                        if verified != state or self._state_path(claim) != path:
                            raise ValueError("release state path mismatch")
                        self._finish_release(claim, state)
                    except (OSError, RuntimeError, ValueError) as error:
                        errors.append(f"{state['pod_name']}: {type(error).__name__}")
        return errors


def same_peer(a: Any, b: Any) -> bool:
    """Same pod when both know their pod UID, else the same (pid, uid)."""
    if len(a) > 2 and len(b) > 2 and a[2] and b[2]:
        return a[2] == b[2]
    return tuple(a[:2]) == tuple(b[:2])


class SlotError(Exception):
    """A v2 claim or resize refused with a protocol reason code."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


_POD_CACHE_TTL_S = 5.0  # claimant pod labels, per pod UID


def _open_peer_pidfd(peer: Any) -> int | None:
    """A pidfd the waiter owns; None where pidfd_open is unavailable."""
    try:
        return os.pidfd_open(int(peer[0]))
    except (AttributeError, OSError, TypeError, ValueError, IndexError):
        return None


@dataclass
class _Waiter:
    """A queued claim. It owns its liveness fd: the connection's own liveness
    is closed with the SLOT_HELD reply, so it cannot outlive that reply."""

    claim_id: str
    seen: float
    pidfd: int | None = None
    alive_hook: Callable[[], bool] | None = None  # test override

    def alive(self) -> bool:
        if self.alive_hook is not None:
            return self.alive_hook()
        if self.pidfd is None:  # no pidfd: alive until the silence expiry
            return True
        try:
            return not select.select([self.pidfd], [], [], 0)[0]
        except (OSError, ValueError):
            return False

    def close(self) -> None:
        fd, self.pidfd = self.pidfd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)


@dataclass
class _Slot:
    name: str
    digest: str
    pool: Any  # RegionPool
    effective_maximum_bytes: int
    holder: dict[str, Any]
    incarnation: int = 0
    peer: Any = None
    owner_alive: Callable[[], bool] | None = None
    claim_id: str | None = None
    holder_conn: socket.socket | None = None
    touched_bytes: int = 0
    layout_generation: int = -1
    # Incremented on every successful claim; a release from an older claim is stale.
    claim_generation: int = 0
    # Last layout that is durable: updated only after a holder ack (or rollback).
    # Persistence reads this, never the live pool, so a grow in flight is invisible.
    committed: tuple | None = None
    # Guards the pool layout and the holder socket; never taken while waiting
    # on the table lock.
    lock: Any = field(default_factory=threading.RLock, repr=False)

    @property
    def held(self) -> bool:
        return self.owner_alive is not None and self.owner_alive()


class SlotTable:
    """v2 slots keyed ``tenant/lineage/guard_index``: capacity, queueing, holder upkeep.

    A slot outlives its worker: death frees the claim (``owner_alive`` turns
    false) but keeps the region pool and holder so a successor reuses them.
    ``release`` is the teardown. ``_waiters`` is the per-slot queue of pending
    claims; the head is next to be granted, others get SLOT_HELD, and a head
    that is dead or silent for ``2 * ensure_timeout_s`` is dropped.

    ``_lock`` covers table mutations only. Holder creation and the holder
    socket are under the slot's own lock, so one slow holder never stalls the
    others. Minimal slot state is persisted so a restarted supervisor keeps its
    epochs and claim generations and re-feeds a live holder instead of deleting it.
    """

    def __init__(
        self,
        broker: NodeBroker,
        cap_bytes: int,
        local_requests: Callable[[str], int] = lambda tenant: 0,
        headroom_bytes: int = _HEADROOM_BYTES,
        ensure_timeout_s: float = 180,
        *,
        require_pod_identity: bool = True,
        clock: Callable[[], float] = time.monotonic,
        state_path: Path | None = None,
    ):
        self.broker, self.cap_bytes = broker, cap_bytes
        self.local_requests, self.headroom_bytes = local_requests, headroom_bytes
        self.ensure_timeout_s = ensure_timeout_s
        self.require_pod_identity = require_pod_identity
        self._clock = clock
        self._slots: dict[str, _Slot] = {}
        self._waiters: dict[str, list[_Waiter]] = {}
        self._pods: dict[str, tuple[float, dict[str, Any]]] = {}
        self._epochs: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._lock = threading.RLock()
        root = getattr(broker, "root", None)
        self._state_path = state_path or (root and Path(root) / "slot-table.json")
        self._restore()

    @staticmethod
    def slot_name(tenant: str, lineage: str, guard_index: int) -> str:
        return f"{tenant}/{lineage}/{guard_index}"

    def waiters(self, name: str) -> list[str]:
        return [waiter.claim_id for waiter in self._waiters.get(name, ())]

    def get(self, name: str) -> _Slot | None:
        return self._slots.get(name)

    def pool_fds(self, name: str) -> list[int]:
        return self._slots[name].pool.pass_fd_list()

    # -- persistence -------------------------------------------------------

    def _persist(self) -> None:
        """Caller holds ``_lock``."""
        if not self._state_path:
            return
        _atomic_json(
            Path(self._state_path),
            {
                "epochs": self._epochs,
                "generations": self._generations,
                "slots": {
                    name: {
                        "digest": slot.digest,
                        "effective_maximum_bytes": slot.effective_maximum_bytes,
                        "holder": slot.holder,
                        "incarnation": slot.incarnation,
                        "claim_generation": slot.claim_generation,
                        "next_index": layout[1],
                        "layout_generation": layout[2],
                        "regions": [
                            {"index": r.index, "offset": r.offset, "size": r.size}
                            for r in layout[0]
                        ],
                    }
                    for name, slot in self._slots.items()
                    for layout in (slot.committed or self._layout(slot),)
                },
            },
        )

    @staticmethod
    def _layout(slot: _Slot) -> tuple:
        return (
            tuple(slot.pool.regions),
            slot.pool.next_index,
            slot.pool.layout_generation,
        )

    def _restore(self) -> None:
        from .protocol import PoolRegion
        from .regions import RegionPool

        if not self._state_path or not Path(self._state_path).exists():
            return
        data = _read_json(Path(self._state_path))
        self._epochs = {k: int(v) for k, v in data.get("epochs", {}).items()}
        self._generations = {k: int(v) for k, v in data.get("generations", {}).items()}
        for name, rec in data.get("slots", {}).items():
            pool = RegionPool.create(rec["effective_maximum_bytes"])
            try:
                pool.restore(
                    [
                        PoolRegion(r["index"], r["offset"], r["size"], 0)
                        for r in rec["regions"]
                    ],
                    rec["next_index"],
                    rec["layout_generation"],
                )
            except BaseException:
                pool.close()
                raise
            self._slots[name] = _Slot(
                name,
                rec["digest"],
                pool,
                rec["effective_maximum_bytes"],
                rec["holder"],
                incarnation=rec["incarnation"],
                claim_generation=rec["claim_generation"],
            )
            self._slots[name].committed = self._layout(self._slots[name])

    # -- admission ---------------------------------------------------------

    def _check_identity(self, peer: Any) -> None:
        from .protocol import IDENTITY_MISMATCH

        if self.require_pod_identity and (
            peer is None or len(peer) < 3 or not peer[0] or peer[2] is None
        ):
            raise SlotError(
                IDENTITY_MISMATCH, "peer pod identity is unavailable; refusing"
            )

    def _check_lineage(self, lineage: str, namespace: str | None, peer: Any) -> None:
        """Require the claimant's pod to carry this lineage (and namespace) label."""
        from .protocol import INCOMPATIBLE_LINEAGE, INTERNAL

        if not self.require_pod_identity:
            return
        now = self._clock()
        cached = self._pods.get(peer[2])
        if cached and now - cached[0] < _POD_CACHE_TTL_S:
            pod = cached[1]  # retries of a held claim must not list the namespace
        else:
            try:
                pod = self.broker.api.get_by_uid(peer[2])
            except PodAPIError as error:
                raise SlotError(INTERNAL, f"pod lookup failed: {error}") from error
            if pod is not None:
                self._pods[peer[2]] = (now, pod)
            else:
                self._pods.pop(peer[2], None)
        if pod is None:
            raise SlotError(INCOMPATIBLE_LINEAGE, "claimant pod is not visible")
        labels = pod.get("metadata", {}).get("labels") or {}
        if labels.get(_LINEAGE_LABEL) != lineage:
            raise SlotError(
                INCOMPATIBLE_LINEAGE, f"claimant pod is not in lineage {lineage}"
            )
        if namespace is not None and labels.get(_DYNAMO_NAMESPACE_LABEL) != namespace:
            raise SlotError(
                INCOMPATIBLE_LINEAGE, f"claimant pod is not in namespace {namespace}"
            )

    def _expire_waiters(self, name: str) -> None:
        """Caller holds ``_lock``. Drop waiters that died or stopped re-claiming."""
        queue = self._waiters.get(name)
        if not queue:
            return
        now = self._clock()
        limit = 2 * self.ensure_timeout_s
        keep = []
        for waiter in queue:
            if now - waiter.seen > limit or not waiter.alive():
                logger.warning(
                    "dropping stale waiter %s for slot %s", waiter.claim_id, name
                )
                waiter.close()
            else:
                keep.append(waiter)
        queue[:] = keep

    def _available(self, tenant: str, name: str) -> int:
        others = sum(
            s.effective_maximum_bytes for n, s in self._slots.items() if n != name
        )
        return (
            self.cap_bytes + self.local_requests(tenant) - self.headroom_bytes - others
        )

    def _new_holder(self, name: str, effective: int, digest: str) -> dict[str, Any]:
        # A fresh pool_id per holder: the broker keeps a deleted claim terminal.
        with self._lock:
            epoch = self._epochs[name] = self._epochs.get(name, -1) + 1
            self._persist()  # the epoch reaches disk before its holder pod exists
        stem = "s" + hashlib.sha256(name.encode()).hexdigest()[:12]
        mib = max(512, -(-effective // (1 << 20)))
        return PoolClaim(f"{stem}-{epoch}", Decimal(mib) / 1024, digest).as_dict()

    def claim(
        self,
        tenant: str,
        lineage: str,
        guard_index: int,
        *,
        total: int | None,
        minimum: int | None,
        maximum: int | None,
        digest: str,
        claim_id: str,
        peer: Any,
        owner_alive: Callable[[], bool],
        namespace: str | None = None,
        waiter_fd: Callable[[], int | None] | None = None,
        waiter_alive: Callable[[], bool] | None = None,
    ) -> _Slot:
        """Claim a slot; returns it. See ``claim_with_generation``."""
        return self.claim_with_generation(
            tenant,
            lineage,
            guard_index,
            total=total,
            minimum=minimum,
            maximum=maximum,
            digest=digest,
            claim_id=claim_id,
            peer=peer,
            owner_alive=owner_alive,
            namespace=namespace,
            waiter_fd=waiter_fd,
            waiter_alive=waiter_alive,
        )[0]

    def claim_with_generation(
        self,
        tenant: str,
        lineage: str,
        guard_index: int,
        *,
        total: int | None,
        minimum: int | None,
        maximum: int | None,
        digest: str,
        claim_id: str,
        peer: Any,
        owner_alive: Callable[[], bool],
        namespace: str | None = None,
        waiter_fd: Callable[[], int | None] | None = None,
        waiter_alive: Callable[[], bool] | None = None,
    ) -> tuple[_Slot, int]:
        """Claim a slot; returns it and the generation this claim was assigned.

        ``waiter_fd`` opens a pidfd the queued waiter owns (default: the peer's
        pid); callers must use the returned generation, never re-read the slot.
        """
        from .protocol import (
            CAPACITY_BELOW_MINIMUM,
            INCOMPATIBLE_LINEAGE,
            INTERNAL,
            SLOT_HELD,
        )
        from .regions import CapacityError, RegionPool

        name = self.slot_name(tenant, lineage, guard_index)
        self._check_identity(peer)
        self._check_lineage(lineage, namespace, peer)
        created = False
        with self._lock:
            self._expire_waiters(name)
            slot = self._slots.get(name)
            queue = self._waiters.setdefault(name, [])
            if (slot is not None and slot.held) or (
                queue and queue[0].claim_id != claim_id
            ):
                now = self._clock()
                for waiter in queue:
                    if waiter.claim_id == claim_id:
                        waiter.seen = now
                        break
                else:
                    fd = waiter_fd() if waiter_fd else _open_peer_pidfd(peer)
                    queue.append(_Waiter(claim_id, now, fd, waiter_alive))
                raise SlotError(SLOT_HELD, f"slot {name} is held")
            if queue:
                queue.pop(0).close()
            if slot is not None:
                if slot.digest != digest:
                    raise SlotError(
                        INCOMPATIBLE_LINEAGE, f"slot {name} lineage differs"
                    )
            else:
                size = total or maximum or minimum
                if not size:
                    raise SlotError(INTERNAL, "claim carries no size")
                effective = effective_grant(
                    size, minimum or size, maximum, self._available(tenant, name)
                )
                if effective is None:
                    raise SlotError(
                        CAPACITY_BELOW_MINIMUM, f"capacity below minimum for {name}"
                    )
                pool = RegionPool.create(effective)
                try:
                    pool.grow(min(size, effective))
                except BaseException as error:
                    pool.close()
                    if isinstance(error, CapacityError):
                        raise SlotError(CAPACITY_BELOW_MINIMUM, str(error)) from error
                    raise
                slot = _Slot(
                    name,
                    digest,
                    pool,
                    effective,
                    self._new_holder(name, effective, digest),
                )
                # Reserved now: capacity is counted and the slot reads as held
                # while the holder comes up outside this lock.
                slot.committed = self._layout(slot)
                self._slots[name] = slot
                created = True
            generation = self._generations[name] = self._generations.get(name, 0) + 1
            slot.claim_generation = generation
            slot.peer, slot.owner_alive, slot.claim_id = peer, owner_alive, claim_id
            self._persist()
        try:
            with slot.lock:
                if created:
                    self.broker.ensure(slot.holder, self.ensure_timeout_s)
                    if self.broker.toucher:
                        self._feed_holder(slot)
                else:
                    self._check_holder(slot)
        except BaseException as error:
            self._abort_claim(slot, created, generation)
            if isinstance(error, SlotError):
                raise
            if isinstance(error, CapacityError):
                raise SlotError(CAPACITY_BELOW_MINIMUM, str(error)) from error
            if isinstance(error, (OSError, RuntimeError, ValueError, TimeoutError)):
                raise SlotError(INTERNAL, f"holder for {name}: {error}") from error
            raise
        return slot, generation

    def _abort_claim(self, slot: _Slot, created: bool, generation: int) -> None:
        """A claim whose holder never came up: forget a new slot, free an old one."""
        with self._lock:
            if not created:
                if slot.claim_generation == generation:
                    slot.owner_alive = None
                return
            if self._slots.get(slot.name) is slot:
                del self._slots[slot.name]
            for waiter in self._waiters.get(slot.name, ()):
                waiter.close()
            self._waiters.pop(slot.name, None)
            self._persist()
        with slot.lock:
            if slot.holder_conn is not None:
                slot.holder_conn.close()
                slot.holder_conn = None
            slot.pool.close()
            with contextlib.suppress(Exception):
                self.broker.release(slot.holder)

    def vacate(self, name: str, peer: Any, generation: int) -> None:
        """The claimant released its lease: free the slot, keep pool and holder.

        ``generation`` is the claim generation captured at grant; a vacate from
        an earlier claim than the current one must not free the new owner.
        """
        self._check_identity(peer)
        with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                return
            if generation != slot.claim_generation:
                logger.warning(
                    "ignoring stale vacate of slot %s: claim generation %s, current %s",
                    name,
                    generation,
                    slot.claim_generation,
                )
                return
            if same_peer(slot.peer, peer):
                slot.owner_alive = None

    # -- holder upkeep (caller holds slot.lock) -----------------------------

    def _send_layout(self, slot: _Slot) -> None:
        """Send the whole layout plus the pool fds; store the holder's report."""
        conn = slot.holder_conn
        layout = {
            "regions": [
                {"index": r.index, "offset": r.offset, "size": r.size, "fd_index": 0}
                for r in slot.pool.regions
            ],
            "layout_generation": slot.pool.layout_generation,
        }
        conn.settimeout(self.ensure_timeout_s)
        line = b""
        try:
            socket.send_fds(
                conn, [json.dumps(layout).encode() + b"\n"], slot.pool.pass_fd_list()
            )
            while not line.endswith(b"\n"):
                chunk = conn.recv(_MAX_MESSAGE)
                if not chunk or len(line) > _MAX_MESSAGE:
                    raise EOFError
                line += chunk
        except TimeoutError as error:
            raise RuntimeError(
                f"holder did not report within {self.ensure_timeout_s:.0f} s"
            ) from error
        except (EOFError, OSError) as error:
            reason = self._holder_exit_reason(slot)
            raise RuntimeError(
                f"holder closed before reporting{f' ({reason})' if reason else ''}"
            ) from error
        report = json.loads(line)
        slot.touched_bytes = report["touched_bytes"]
        slot.layout_generation = report["layout_generation"]

    def _holder_exit_reason(self, slot: _Slot) -> str | None:
        """The holder container's terminated reason (e.g. OOMKilled), if readable."""
        # urlopen's timeout is per socket operation, so bound the whole lookup.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            name = "gaas-pool-" + self.broker._key(PoolClaim.parse(slot.holder))
            pod = pool.submit(self.broker.api.get, name).result(
                timeout=_EXIT_REASON_DEADLINE_S
            )
            for status in (pod or {}).get("status", {}).get("containerStatuses", []):
                for key in ("state", "lastState"):
                    reason = (status.get(key) or {}).get("terminated", {}).get("reason")
                    if reason:
                        return str(reason)
        except Exception:  # best effort; never mask the original failure
            pass
        finally:
            pool.shutdown(wait=False)
        return None

    def _feed_holder(self, slot: _Slot) -> None:
        """Accept the toucher on its private socket and hand it the layout."""
        path = self.broker.holder_socket_path(
            self.broker._load(PoolClaim.parse(slot.holder))
        )
        self._drop_holder_conn(slot)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            listener.listen(1)
            listener.settimeout(self.ensure_timeout_s)
            try:
                slot.holder_conn, _ = listener.accept()
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
        self._send_layout(slot)

    @staticmethod
    def _drop_holder_conn(slot: _Slot) -> None:
        if slot.holder_conn is not None:
            with contextlib.suppress(OSError):
                slot.holder_conn.close()
            slot.holder_conn = None

    def _holder_dead(self, slot: _Slot) -> bool:
        pod = self.broker.api.get(
            "gaas-pool-" + self.broker._key(PoolClaim.parse(slot.holder))
        )
        return (
            pod is None
            or bool(pod["metadata"].get("deletionTimestamp"))
            or pod.get("status", {}).get("phase") in ("Failed", "Succeeded")
        )

    def _check_holder(self, slot: _Slot) -> bool:
        """Recreate a dead holder once, or re-feed a live one that lost its
        connection (a restarted supervisor); True if it was recreated."""
        from .protocol import INTERNAL

        if not self._holder_dead(slot):
            if self.broker.toucher and slot.holder_conn is None:
                self._feed_holder(slot)
            return False
        self.broker.abandon(slot.holder)
        self._drop_holder_conn(slot)
        slot.incarnation += 1
        slot.holder = self._new_holder(
            slot.name, slot.effective_maximum_bytes, slot.digest
        )
        with self._lock:
            self._persist()
        try:
            self.broker.ensure(slot.holder, self.ensure_timeout_s)
            if self.broker.toucher:
                self._feed_holder(slot)
        except (OSError, RuntimeError, ValueError, TimeoutError) as error:
            raise SlotError(
                INTERNAL, f"holder recreate for {slot.name}: {error}"
            ) from error
        return True

    def check_holder(self, name: str) -> bool:
        """Recreate the holder once if it is gone or terminating; True if it was."""
        from .protocol import UNKNOWN_SLOT

        with self._lock:
            slot = self._slots.get(name)
        if slot is None:
            raise SlotError(UNKNOWN_SLOT, f"unknown slot {name}")
        with slot.lock:
            return self._check_holder(slot)

    # -- resize -------------------------------------------------------------

    def _recheck_owner(self, slot: _Slot, peer: Any, generation: int) -> None:
        """Under ``slot.lock``: the validated claim must still own the slot."""
        from .protocol import IDENTITY_MISMATCH

        if (
            slot.claim_generation != generation
            or not slot.held
            or not same_peer(slot.peer, peer)
        ):
            raise SlotError(
                IDENTITY_MISMATCH, f"slot {slot.name} changed owner during the resize"
            )

    def _owned(
        self, name: str, peer: Any, claim_generation: int | None = None
    ) -> tuple[_Slot, int]:
        from .protocol import IDENTITY_MISMATCH, UNKNOWN_SLOT

        self._check_identity(peer)
        with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                raise SlotError(UNKNOWN_SLOT, f"unknown slot {name}")
            if not same_peer(slot.peer, peer):
                raise SlotError(
                    IDENTITY_MISMATCH, f"slot {name} belongs to another peer"
                )
            # A released slot, or a stale claim of the same pod, may not resize.
            if not slot.held or claim_generation not in (None, slot.claim_generation):
                raise SlotError(
                    IDENTITY_MISMATCH, f"slot {name} is not held by this claim"
                )
            return slot, slot.claim_generation

    def grow(
        self,
        name: str,
        peer: Any,
        additional_bytes: int,
        split: int | None,
        claim_generation: int | None = None,
    ):
        from .protocol import CAPACITY_BELOW_MINIMUM, INTERNAL
        from .regions import CapacityError

        slot, generation_seen = self._owned(name, peer, claim_generation)
        with slot.lock:
            self._recheck_owner(slot, peer, generation_seen)
            before = (slot.pool.layout_generation, slot.pool.next_index)
            try:
                regions = slot.pool.grow(additional_bytes, split)
            except CapacityError as error:
                raise SlotError(CAPACITY_BELOW_MINIMUM, str(error)) from error
            if slot.holder_conn is not None or self.broker.toucher:
                try:
                    if slot.holder_conn is None:
                        self._feed_holder(slot)
                    else:
                        self._send_layout(slot)
                except (OSError, RuntimeError, ValueError, KeyError) as error:
                    # The holder never acknowledged: take the regions back so a
                    # retry grows once, and make the holder reconnect for a re-feed.
                    self._drop_holder_conn(slot)
                    slot.pool.rollback_grow(regions, *before)
                    raise SlotError(INTERNAL, f"holder layout: {error}") from error
            generation = slot.pool.layout_generation
            slot.committed = self._layout(slot)  # acknowledged: now durable
        with self._lock:
            self._persist()
        return regions, generation

    def shrink(
        self,
        name: str,
        peer: Any,
        indices: tuple[int, ...],
        claim_generation: int | None = None,
    ):
        from .protocol import INTERNAL, REGION_BUSY, UNKNOWN_SLOT
        from .regions import RegionBusy, UnknownRegion

        slot, generation_seen = self._owned(name, peer, claim_generation)
        with slot.lock:
            self._recheck_owner(slot, peer, generation_seen)
            try:
                for index in indices:
                    slot.pool.mark_quiescent(index)
                removed = slot.pool.shrink(indices)
            except RegionBusy as error:
                raise SlotError(REGION_BUSY, str(error)) from error
            except UnknownRegion as error:
                raise SlotError(UNKNOWN_SLOT, f"unknown region {error}") from error
            except OSError as error:  # the punch failed; the layout is untouched
                raise SlotError(INTERNAL, str(error)) from error
            if slot.holder_conn is not None:
                try:
                    self._send_layout(slot)
                except (OSError, RuntimeError, ValueError, KeyError):
                    # The regions are gone either way; the holder is re-fed in
                    # full after it reconnects.
                    logger.warning("holder did not take the reduced layout of %s", name)
                    self._drop_holder_conn(slot)
            generation = slot.pool.layout_generation
            slot.committed = self._layout(slot)
        with self._lock:
            self._persist()
        return removed, generation

    def release(self, name: str) -> list[str]:
        """PoC teardown: close pool, drain and delete holder; returns waiters."""
        from .protocol import UNKNOWN_SLOT

        with self._lock:
            slot = self._slots.pop(name, None)
            if slot is None:
                raise SlotError(UNKNOWN_SLOT, f"unknown slot {name}")
            self._persist()
            waiters = self.waiters(name)
            for waiter in self._waiters.get(name, ()):
                waiter.close()
            self._waiters.pop(name, None)
        with slot.lock:
            self._drop_holder_conn(slot)
            slot.pool.close()
            if self._holder_dead(slot):
                self.broker.abandon(slot.holder)
            else:
                self.broker.release(slot.holder)
        return waiters


def _rpc(
    socket_path: str, op: str, claim: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    PoolClaim.parse(claim)
    if not 0 < timeout_s <= 180:
        raise ValueError("timeout must be in (0, 180]")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_s)
        connection.connect(socket_path)
        connection.sendall(
            json.dumps({"op": op, "claim": claim, "timeout_s": timeout_s}).encode()
            + b"\n"
        )
        with connection.makefile("rb") as source:
            line = source.readline(_MAX_MESSAGE + 1)
        if not line.endswith(b"\n") or len(line) > _MAX_MESSAGE:
            raise RuntimeError("invalid broker response size")
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error", "broker request failed"))
        return response["result"]


def ensure_pool(
    socket_path: str, claim: dict[str, Any], timeout_s: float = 180
) -> dict[str, Any]:
    return _rpc(socket_path, "ensure", claim, timeout_s)


def inspect_pool(
    socket_path: str, claim: dict[str, Any], timeout_s: float = 180
) -> dict[str, Any]:
    return _rpc(socket_path, "inspect", claim, timeout_s)


def release_pool(
    socket_path: str, claim: dict[str, Any], timeout_s: float = 180
) -> dict[str, Any]:
    return _rpc(socket_path, "release", claim, timeout_s)


def serve(broker: NodeBroker, socket_path: str, stop: threading.Event) -> None:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(185)
            try:
                line = self.rfile.readline(_MAX_MESSAGE + 1)
                if len(line) > _MAX_MESSAGE or not line.endswith(b"\n"):
                    raise ValueError("invalid broker request size")
                request = json.loads(line)
                if request["op"] == "ensure":
                    result = broker.ensure(
                        request["claim"], request.get("timeout_s", 180)
                    )
                elif request["op"] == "inspect":
                    result = broker.inspect(request["claim"])
                elif request["op"] == "release":
                    result = broker.release(request["claim"])
                else:
                    raise ValueError("unknown broker operation")
                response = {"ok": True, "result": result}
            except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
                response = {"ok": False, "error": str(error)}
            self.wfile.write(json.dumps(response).encode() + b"\n")

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = False

    # A process death can leave its socket inode. Only the exclusive manager
    # may remove that stale inode; a second live broker must not replace it.
    lock_fd = os.open(
        broker.root / "broker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _read_json(broker.root / "manager.json") != {
            "namespace": broker.namespace,
            "node": broker.node_name,
            "manager": broker.manager_id,
        }:
            raise ValueError("manager identity changed before lock acquisition")
        with contextlib.suppress(FileNotFoundError):
            if not stat.S_ISSOCK(os.lstat(socket_path).st_mode):
                raise ValueError("broker path is not a socket")
            os.unlink(socket_path)
        with Server(socket_path, Handler) as server:
            inode = os.lstat(socket_path).st_ino
            os.chmod(socket_path, 0o600)
            server.timeout = 0.25
            previous_errors = []
            try:
                while not stop.is_set():
                    server.handle_request()
                    errors = broker.reconcile_releases()
                    if errors and errors != previous_errors:
                        logger.warning(
                            json.dumps(
                                {"event": "holder_reconcile_error", "errors": errors}
                            )
                        )
                    previous_errors = errors
            finally:
                with contextlib.suppress(FileNotFoundError):
                    if os.lstat(socket_path).st_ino == inode:
                        os.unlink(socket_path)
    finally:
        os.close(lock_fd)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=("serve", "ensure", "inspect", "release"),
        default="serve",
    )
    parser.add_argument("--socket-root", default="/node-shm/gaas-h7")
    parser.add_argument("--broker-socket")
    parser.add_argument("--claim-json")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--namespace", default=os.environ.get("POD_NAMESPACE"))
    parser.add_argument("--node-name", default=os.environ.get("NODE_NAME"))
    parser.add_argument("--manager-id", default="gaas-h7")
    parser.add_argument("--holder-image")
    args = parser.parse_args()
    if args.operation != "serve":
        if args.claim_json is None:
            parser.error("client operation requires --claim-json")
        function = {
            "ensure": ensure_pool,
            "inspect": inspect_pool,
            "release": release_pool,
        }[args.operation]
        print(
            json.dumps(
                function(
                    args.broker_socket or str(Path(args.socket_root) / "broker.sock"),
                    json.loads(args.claim_json),
                    args.timeout_seconds,
                ),
                sort_keys=True,
            )
        )
        return
    if not args.namespace or not args.node_name or not args.holder_image:
        parser.error("serve requires namespace, node-name and holder-image")
    broker = NodeBroker(
        namespace=args.namespace,
        node_name=args.node_name,
        manager_id=args.manager_id,
        image=args.holder_image,
        socket_root=args.socket_root,
        api=InClusterPodAPI(args.namespace),
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    serve(broker, str(broker.root / "broker.sock"), stop)


if __name__ == "__main__":
    main()
