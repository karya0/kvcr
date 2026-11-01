# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Namespace-scoped holder broker. Pool data never travels through this socket.

The holder runs KVCR-Service; clients connect to it directly. A worker disappearing
does not release its pool. Explicit release is durable and deletion requires a
terminal drain acknowledgment from that same holder.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
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
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

_GIB = 1 << 30
_JOURNAL_BYTES = 100 << 20
_HEADROOM_BYTES = 2 * _GIB
_LABEL = "kvcr.nvidia.com/manager"
_ANNOTATION = "kvcr.nvidia.com/"
_MAX_MESSAGE = 65536
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
        if size not in (Decimal("0.5"), Decimal(1)):
            raise ValueError("H7 holder size must be 0.5 or 1.0 GiB")
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

    def _pod(self, claim: PoolClaim, state: dict[str, Any]) -> dict[str, Any]:
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
        active = [state for state in states if state["phase"] != "deleted"]
        pods = self.api.list(self.manager)
        known = {state["pod_name"] for state in active}
        if any(pod["metadata"]["name"] not in known for pod in pods):
            raise RuntimeError(
                "unreconciled holder exists; retain it and refuse allocation"
            )
        if (
            len(active) >= 2
            or sum(PoolClaim.parse(s["claim"]).data_bytes for s in active)
            + claim.data_bytes
            > 3 * _GIB // 2
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
                self._verify_pod(
                    claim, state, pod, allow_terminating=bool(state.get("drain_ack"))
                )
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
