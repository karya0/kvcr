# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import importlib.util
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# This module has no framework dependencies. Loading it directly also permits
# its ownership tests on macOS, where the optional NIXL wheel is unavailable.
_checkout_source = Path(__file__).parents[2] / "src/kvcr/service/node_service.py"
_source = _checkout_source
if not _source.is_file():
    # Mounted validation tests must exercise the installed image's bytes.
    # Finding the top-level package does not import its framework dependencies.
    _package_spec = importlib.util.find_spec("kvcr")
    if _package_spec is None or _package_spec.origin is None:
        raise ImportError("cannot resolve installed kvcr package origin")
    _source = (
        Path(_package_spec.origin).resolve().parent / "service/node_service.py"
    ).resolve(strict=True)
    if not _source.is_file():
        raise ImportError("installed kvcr node service is not a file")
_spec = importlib.util.spec_from_file_location("_node_service_test_target", _source)
broker_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = broker_module
_spec.loader.exec_module(broker_module)
NodeBroker = broker_module.NodeBroker
PoolClaim = broker_module.PoolClaim
PodAPIError = broker_module.PodAPIError

_IMAGE = "example.test/consumer@sha256:" + "a" * 64
_CLAIM = {"pool_id": "pool-a", "size_gb": 1.0, "compatibility_digest": "model-v1"}
_SPECS = [
    {"pool_id": "allocation-a", "generation": "generation-a", "device": 1, "inode": 2}
]


def test_broker_module_origin(record_property):
    actual = Path(broker_module.__file__).resolve()
    assert actual == _source.resolve()
    if not _checkout_source.is_file():
        installed = importlib.util.find_spec("kvcr")
        expected = Path(installed.origin).resolve().parent / "service/node_service.py"
        assert actual == expected.resolve(strict=True)
    record_property("broker_module_origin", str(actual))


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="gaas-node-", dir="/tmp") as directory:
        yield Path(directory)


class FakeAPI:
    def __init__(self):
        self.pods = {}
        self.creates = []
        self.deletes = []
        self.lose_create_response = False
        self.hide_created_once = False
        self._hide_next_read = False
        self.keep_deleted = False
        self.lose_delete_response = False

    def get(self, name):
        if self._hide_next_read:
            self._hide_next_read = False
            return None
        return copy.deepcopy(self.pods.get(name))

    def list(self, manager):
        return [
            copy.deepcopy(p)
            for p in self.pods.values()
            if p["metadata"]["labels"][broker_module._LABEL] == manager
        ]

    def create(self, pod):
        self.creates.append(copy.deepcopy(pod))
        name = pod["metadata"]["name"]
        if name in self.pods:
            raise PodAPIError(409, "POST")
        actual = copy.deepcopy(pod)
        actual["metadata"]["uid"] = "holder-uid-1"
        actual["status"] = {"phase": "Running", "podIP": "10.1.2.3"}
        actual["spec"]["nodeName"] = "node-a"
        # Kubernetes adds defaults which must not invalidate the intended spec.
        actual["spec"]["securityContext"]["supplementalGroupsPolicy"] = "Merge"
        actual["spec"]["containers"][0]["readinessProbe"]["timeoutSeconds"] = 1
        actual["spec"]["tolerations"] += [
            {
                "key": "node.kubernetes.io/not-ready",
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 300,
            },
            {
                "key": "node.kubernetes.io/unreachable",
                "operator": "Exists",
                "value": "",
                "effect": "NoExecute",
                "tolerationSeconds": 300,
            },
        ]
        self.pods[name] = actual
        if self.lose_create_response:
            self._hide_next_read = self.hide_created_once
            raise PodAPIError(None, "POST")
        return copy.deepcopy(actual)

    def delete(self, name, uid):
        self.deletes.append((name, uid))
        assert self.pods[name]["metadata"]["uid"] == uid
        if not self.keep_deleted:
            del self.pods[name]
        else:
            self.pods[name]["metadata"]["deletionTimestamp"] = "2026-11-01T22:00:00Z"
        if self.lose_delete_response:
            raise PodAPIError(None, "DELETE")


@pytest.fixture(autouse=True)
def no_cluster(monkeypatch, tmp_path):
    config = tmp_path / "empty-kubeconfig"
    config.write_text("")
    monkeypatch.setenv("KUBECONFIG", str(config))

    def denied(*args, **kwargs):
        pytest.fail("unmocked Kubernetes HTTP access")

    monkeypatch.setattr(broker_module.urllib.request, "urlopen", denied)


def make_broker(root, api=None, info=None, drain=None):
    return NodeBroker(
        namespace="bench",
        node_name="node-a",
        manager_id="gaas",
        image=_IMAGE,
        socket_root=root,
        api=api or FakeAPI(),
        info=info
        or (
            lambda _: {
                "state": "ready",
                "pool_specs": _SPECS,
                "compatibility_digest": "model-v1",
            }
        ),
        drain=drain
        or (
            lambda _: {
                "state": "drained",
                "pool_specs": _SPECS,
                "compatibility_digest": "model-v1",
            }
        ),
        poll_interval=0,
    )


def test_create_idempotent_reattach_and_resources(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    first = broker.ensure(_CLAIM)
    second = broker.ensure(dict(_CLAIM))
    assert first == second
    assert len(api.creates) == 1
    assert first["holder_uid"] == "holder-uid-1"
    assert first["holder_pod_ip"] == "10.1.2.3"
    assert first["control_port"] == 55001
    pod = api.creates[0]
    assert not pod["metadata"].get("ownerReferences")
    spec = pod["spec"]
    assert "nodeName" not in spec
    affinity = spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    assert affinity["nodeSelectorTerms"][0]["matchFields"] == [
        {"key": "metadata.name", "operator": "In", "values": ["node-a"]}
    ]
    assert spec["restartPolicy"] == "Never"
    holder = spec["containers"][0]
    assert holder["resources"]["requests"]["memory"] == "4296Mi"
    assert holder["resources"]["limits"]["memory"] == "4296Mi"
    assert "livenessProbe" not in holder
    assert not any("gpu" in key for key in holder["resources"]["requests"])
    assert spec["volumes"][1]["emptyDir"] == {"medium": "Memory", "sizeLimit": "2248Mi"}


def test_lost_create_response_reconciles_once(tmp_path):
    api = FakeAPI()
    api.lose_create_response = True
    result = make_broker(tmp_path, api).ensure(_CLAIM)
    assert result["holder_uid"] == "holder-uid-1"
    assert len(api.creates) == 1


def test_uncertain_create_can_be_released_once_uid_becomes_observable(tmp_path):
    api = FakeAPI()
    api.lose_create_response = True
    api.hide_created_once = True
    broker = make_broker(tmp_path, api)
    with pytest.raises(PodAPIError):
        broker.ensure(_CLAIM)
    result = broker.release(_CLAIM)
    assert result["deleted"] is True
    assert result["holder_uid"] == "holder-uid-1"
    assert api.deletes[0][1] == "holder-uid-1"


@pytest.fixture
def unbound_holder(tmp_path, monkeypatch):
    api = FakeAPI()
    create = api.create

    def crash_after_accept(pod):
        created = create(pod)
        api.pods[created["metadata"]["name"]]["metadata"]["uid"] = (
            "12345678-1234-4234-8234-123456789abc"
        )
        raise SystemExit("broker died after API accepted create")

    with monkeypatch.context() as patch:
        patch.setattr(api, "create", crash_after_accept)
        with pytest.raises(SystemExit):
            make_broker(tmp_path, api).ensure(_CLAIM)
    broker = make_broker(tmp_path, api)
    assert broker._load(PoolClaim.parse(_CLAIM))["holder_uid"] is None
    return api, broker


def test_inspect_binds_accepted_create_durably_without_marking_ready(unbound_holder):
    api, broker = unbound_holder
    stop = threading.Event()
    path = str(broker.root / "broker.sock")
    worker = threading.Thread(target=broker_module.serve, args=(broker, path, stop))
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while not Path(path).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        first = broker_module.inspect_pool(path, _CLAIM, 2)
        assert broker_module.inspect_pool(path, _CLAIM, 2) == first
    finally:
        stop.set()
        worker.join(3)
        assert not worker.is_alive()
    assert first["holder_uid"] == "12345678-1234-4234-8234-123456789abc"
    assert first["state"] == "creating"
    assert first["pool_specs"] == []
    assert first["holder_pod_ip"] is None
    restarted = make_broker(broker.root, api)
    assert restarted.inspect(_CLAIM) == first
    result = restarted.release(_CLAIM)
    assert result["state"] == "deleted"
    assert result["pool_specs"] == _SPECS
    assert api.deletes == [(first["holder_name"], first["holder_uid"])]
    assert len(api.creates) == 1


@pytest.mark.parametrize(
    "mutation", ["missing", "nonce", "owner", "container", "claim"]
)
def test_inspect_unbound_holder_refuses_missing_or_foreign_pod(
    unbound_holder, mutation
):
    api, broker = unbound_holder
    pod = next(iter(api.pods.values()))
    if mutation == "missing":
        api.pods.clear()
    elif mutation == "nonce":
        pod["metadata"]["annotations"][broker_module._ANNOTATION + "creation-id"] = (
            "foreign"
        )
    elif mutation == "owner":
        pod["metadata"]["ownerReferences"] = [{"uid": "foreign"}]
    elif mutation == "container":
        pod["spec"]["containers"][0]["image"] = "foreign"
    else:
        pod["metadata"]["annotations"][broker_module._ANNOTATION + "claim"] = "{}"
    with pytest.raises((RuntimeError, ValueError)):
        broker.inspect(_CLAIM)
    assert broker._load(PoolClaim.parse(_CLAIM))["holder_uid"] is None
    assert len(api.creates) == 1
    assert not api.deletes


@pytest.mark.parametrize(
    "uid",
    [
        None,
        1,
        "",
        "malformed",
        "12345678123442348234123456789abc",
        "12345678-1234-4234-8234-123456789ABC",
    ],
)
def test_inspect_unbound_holder_refuses_malformed_uid(unbound_holder, uid):
    api, broker = unbound_holder
    next(iter(api.pods.values()))["metadata"]["uid"] = uid
    with pytest.raises(ValueError):
        broker.inspect(_CLAIM)
    assert broker._load(PoolClaim.parse(_CLAIM))["holder_uid"] is None
    assert len(api.creates) == 1
    assert not api.deletes


@pytest.mark.parametrize("failure", ["read", "replace"])
def test_inspect_bind_uncertainty_fails_closed_and_can_retry(
    unbound_holder, monkeypatch, failure
):
    api, broker = unbound_holder

    def fail(*args):
        if failure == "read":
            raise PodAPIError(None, "GET")
        raise OSError("state persistence failed")

    with monkeypatch.context() as patch:
        patch.setattr(
            api if failure == "read" else broker_module.os,
            "get" if failure == "read" else "replace",
            fail,
        )
        with pytest.raises((PodAPIError, OSError)):
            broker.inspect(_CLAIM)
    assert broker._load(PoolClaim.parse(_CLAIM))["holder_uid"] is None
    assert make_broker(broker.root, api).inspect(_CLAIM)["holder_uid"] is not None
    assert len(api.creates) == 1
    assert not api.deletes


def test_inspect_snapshot_bind_never_accepts_later_replacement(
    unbound_holder, monkeypatch
):
    api, broker = unbound_holder
    get = api.get

    def replace_after_read(name):
        snapshot = get(name)
        api.pods[name]["metadata"]["uid"] = "87654321-4321-4321-8321-cba987654321"
        return snapshot

    with monkeypatch.context() as patch:
        patch.setattr(api, "get", replace_after_read)
        original = broker.inspect(_CLAIM)
    restarted = make_broker(broker.root, api)
    for operation in (restarted.inspect, restarted.release):
        with pytest.raises(ValueError, match="UID changed"):
            operation(_CLAIM)
        assert (
            restarted._load(PoolClaim.parse(_CLAIM))["holder_uid"]
            == original["holder_uid"]
        )
    assert len(api.creates) == 1
    assert not api.deletes


def test_integer_and_float_capacities_are_canonical(tmp_path):
    broker = make_broker(tmp_path)
    first = broker.ensure({**_CLAIM, "size_gb": 1})
    assert broker.ensure(_CLAIM) == first


def test_broker_restart_preserves_claim_and_pool_identity(tmp_path):
    api = FakeAPI()
    first = make_broker(tmp_path, api).ensure(_CLAIM)
    second = make_broker(tmp_path, api).ensure(_CLAIM)
    assert first == second
    assert len(api.creates) == 1
    changed = make_broker(
        tmp_path,
        api,
        info=lambda _: {
            "state": "ready",
            "pool_specs": [{"generation": "reallocated"}],
            "compatibility_digest": "model-v1",
        },
    )
    with pytest.raises(RuntimeError, match="pool identity changed"):
        changed.ensure(_CLAIM)


@pytest.mark.parametrize(
    "field,value", [("size_gb", 0.5), ("compatibility_digest", "other")]
)
def test_claim_drift_refuses_reallocation(tmp_path, field, value):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    broker.ensure(_CLAIM)
    with pytest.raises(ValueError, match="differs"):
        broker.ensure({**_CLAIM, field: value})
    assert len(api.creates) == 1


@pytest.mark.parametrize(
    "mutation", ["uid", "node", "owner", "nonce", "container", "host-network"]
)
def test_adoption_refuses_foreign_or_mutated_pod(tmp_path, mutation):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    descriptor = broker.ensure(_CLAIM)
    pod = api.pods[descriptor["holder_name"]]
    if mutation == "uid":
        pod["metadata"]["uid"] = "foreign"
    elif mutation == "node":
        pod["spec"]["nodeName"] = "other"
    elif mutation == "owner":
        pod["metadata"]["ownerReferences"] = [{"uid": "daemonset"}]
    elif mutation == "nonce":
        pod["metadata"]["annotations"][broker_module._ANNOTATION + "creation-id"] = (
            "other"
        )
    elif mutation == "container":
        pod["spec"]["containers"].insert(0, {"name": "injected"})
    else:
        pod["spec"]["hostNetwork"] = True
    with pytest.raises(ValueError):
        broker.inspect(_CLAIM)
    assert not api.deletes


def test_no_worker_or_timeout_ttl_deletes_holder(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    broker.ensure(_CLAIM)
    for _ in range(4):
        assert broker.reconcile_releases() == []
    assert not api.deletes
    assert broker.inspect(_CLAIM)["state"] == "ready"


def test_busy_release_is_durable_then_collected_after_last_reference(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api, drain=lambda _: {"state": "busy"})
    original = broker.ensure(_CLAIM)
    assert broker.release(_CLAIM)["state"] == "release_requested"
    assert not api.deletes
    with pytest.raises(RuntimeError, match="release intent"):
        broker.ensure(_CLAIM)
    restarted = make_broker(tmp_path, api)
    assert restarted.reconcile_releases() == []
    assert restarted.inspect(_CLAIM)["state"] == "deleted"
    assert api.deletes == [(original["holder_name"], original["holder_uid"])]
    assert restarted.release(_CLAIM)["state"] == "deleted"


@pytest.mark.parametrize("result", [{}, {"state": "error"}, {"state": "draining"}])
def test_unknown_drain_never_authorizes_delete(tmp_path, result):
    api = FakeAPI()
    broker = make_broker(tmp_path, api, drain=lambda _: result)
    broker.ensure(_CLAIM)
    broker.release(_CLAIM)
    assert not api.deletes


def test_drain_transport_error_retains_intent_and_reservation(tmp_path):
    def failed(_):
        raise TimeoutError("uncertain")

    api = FakeAPI()
    broker = make_broker(tmp_path, api, drain=failed)
    broker.ensure(_CLAIM)
    with pytest.raises(TimeoutError):
        broker.release(_CLAIM)
    state = json.loads(next((tmp_path / "state").glob("*.json")).read_text())
    assert state["phase"] == "release_requested"
    assert not api.deletes


def test_acknowledged_delete_with_lost_response_is_observed_not_repeated(tmp_path):
    api = FakeAPI()
    api.lose_delete_response = True
    broker = make_broker(tmp_path, api)
    result = broker.ensure(_CLAIM)
    with pytest.raises(PodAPIError):
        broker.release(_CLAIM)
    observed = broker.inspect(_CLAIM)
    assert observed["state"] == "deleted"
    assert observed["deleted"] is True
    assert observed["holder_uid"] == result["holder_uid"]
    assert observed["drain"]["state"] == "drained"
    assert len(api.deletes) == 1


def test_abandoned_ensure_can_be_released_after_readiness_recovers(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api, info=lambda _: {"state": "starting"})
    with pytest.raises(TimeoutError):
        broker.ensure(_CLAIM, 0.001)
    restarted = make_broker(tmp_path, api)
    result = restarted.release(_CLAIM)
    assert result["state"] == "deleted"
    assert result["holder_uid"] == "holder-uid-1"


def test_wrong_drain_generation_never_deletes_holder(tmp_path):
    api = FakeAPI()
    broker = make_broker(
        tmp_path,
        api,
        drain=lambda _: {
            "state": "drained",
            "pool_specs": [{"generation": "other"}],
            "compatibility_digest": "model-v1",
        },
    )
    broker.ensure(_CLAIM)
    with pytest.raises(RuntimeError, match="identity differs"):
        broker.release(_CLAIM)
    assert not api.deletes


def test_replacement_during_release_is_preserved(tmp_path):
    api = FakeAPI()
    api.keep_deleted = True
    broker = make_broker(tmp_path, api)
    result = broker.ensure(_CLAIM)
    assert broker.release(_CLAIM)["state"] == "deleting"
    api.pods[result["holder_name"]]["metadata"]["uid"] = "replacement"
    assert broker.reconcile_releases()
    assert len(api.deletes) == 1
    assert api.pods[result["holder_name"]]["metadata"]["uid"] == "replacement"


def test_missing_holder_does_not_silently_replace_cache(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    result = broker.ensure(_CLAIM)
    del api.pods[result["holder_name"]]
    with pytest.raises(RuntimeError, match="disappeared"):
        broker.ensure(_CLAIM)
    assert len(api.creates) == 1


def test_terminating_holder_cannot_be_reattached(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    result = broker.ensure(_CLAIM)
    api.pods[result["holder_name"]]["metadata"]["deletionTimestamp"] = "now"
    with pytest.raises(ValueError, match="terminating"):
        broker.ensure(_CLAIM)
    assert not api.deletes


def test_capacity_and_count_budget(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    large_claim = {**_CLAIM, "size_gb": 255.5}
    broker.ensure(large_claim)
    with pytest.raises(ValueError, match="budget"):
        broker.ensure({**_CLAIM, "pool_id": "pool-b"})
    broker.ensure({**_CLAIM, "pool_id": "pool-b", "size_gb": 0.5})
    with pytest.raises(ValueError, match="budget"):
        broker.ensure({**_CLAIM, "pool_id": "pool-c", "size_gb": 0.5})
    broker.release(large_claim)
    broker.ensure({**large_claim, "pool_id": "pool-c"})


def test_holder_count_limit_independent_of_capacity(tmp_path):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    broker.ensure({**_CLAIM, "size_gb": 0.5})
    broker.ensure({**_CLAIM, "pool_id": "pool-b", "size_gb": 0.5})
    with pytest.raises(ValueError, match="budget"):
        broker.ensure({**_CLAIM, "pool_id": "pool-c", "size_gb": 0.5})
    assert len(api.creates) == 2


def test_client_server_roundtrip(tmp_path):
    # macOS's default pytest temp directory can exceed AF_UNIX's path limit.
    with tempfile.TemporaryDirectory(prefix="gaas-", dir="/tmp") as short:
        broker = make_broker(Path(short))
        stop = threading.Event()
        path = str(Path(short) / "broker.sock")
        worker = threading.Thread(target=broker_module.serve, args=(broker, path, stop))
        worker.start()
        try:
            deadline = time.monotonic() + 2
            while not Path(path).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            first = broker_module.ensure_pool(path, _CLAIM, 2)
            assert broker_module.inspect_pool(path, _CLAIM, 2) == first
            assert broker_module.release_pool(path, _CLAIM, 2)["state"] == "deleted"
        finally:
            stop.set()
            worker.join(3)
            assert not worker.is_alive()
        assert not Path(path).exists()


def test_uid_precondition_is_in_delete_request(monkeypatch):
    api = object.__new__(broker_module.InClusterPodAPI)
    captured = []
    monkeypatch.setattr(
        api, "_request", lambda *args, **kwargs: captured.append((args, kwargs))
    )
    api.delete("holder-a", "owned-uid")
    assert captured[0][0][2]["preconditions"] == {"uid": "owned-uid"}


def test_release_cli_needs_no_service_account_or_server_configuration(
    monkeypatch, capsys
):
    result = {"state": "deleted", "deleted": True, "holder_uid": "owned-uid"}
    captured = []
    monkeypatch.setattr(
        broker_module,
        "release_pool",
        lambda *args: captured.append(args) or result,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kvcr-node-service",
            "--operation",
            "release",
            "--broker-socket",
            "/tmp/broker.sock",
            "--claim-json",
            json.dumps(_CLAIM),
            "--timeout-seconds",
            "3",
        ],
    )
    broker_module.main()
    assert json.loads(capsys.readouterr().out) == result
    assert captured == [("/tmp/broker.sock", _CLAIM, 3)]


@pytest.mark.parametrize(
    "claim",
    [
        {**_CLAIM, "pool_id": "../escape"},
        {**_CLAIM, "size_gb": 257},
        {**_CLAIM, "size_gb": True},
        {**_CLAIM, "compatibility_digest": ""},
        {**_CLAIM, "node": "other"},
    ],
)
def test_invalid_claim(claim):
    with pytest.raises(ValueError):
        PoolClaim.parse(claim)


@pytest.mark.parametrize(
    "size",
    [
        0,
        False,
        -1,
        0.499,
        12.001,
        256 + 1 / 1024,
        0.5001,
        "1.00000000000000000000000000001",
        "NaN",
        "sNaN",
        "Infinity",
        "-Infinity",
        "",
        "twelve",
        "12 GiB",
        None,
        [],
        {},
    ],
)
def test_claim_size_rejects_invalid_capacity(size):
    with pytest.raises(ValueError):
        PoolClaim.parse({**_CLAIM, "size_gb": size})


@pytest.mark.parametrize("size", ["0.5", "1", "12", "150", "150.0009765625", "256"])
def test_numeric_string_claim_size_remains_supported(size):
    claim = PoolClaim.parse({**_CLAIM, "size_gb": size})
    assert claim.as_dict()["size_gb"] == float(size)
    assert PoolClaim.parse(claim.as_dict()) == claim


@pytest.mark.parametrize(
    "size,mapping,backing,memory",
    [
        (1, 1124, 2248, 4296),
        (0.5, 612, 1224, 3272),
        (513 / 1024, 613, 1226, 3274),
        (1.25, 1380, 2760, 4808),
        (12, 12388, 24776, 26824),
        (150, 153700, 307400, 309448),
        (256, 262244, 524488, 526536),
    ],
)
def test_snapshot_budget_is_reserved_and_described(
    tmp_path, size, mapping, backing, memory
):
    api = FakeAPI()
    descriptor = make_broker(tmp_path, api).ensure({**_CLAIM, "size_gb": size})
    assert {
        key: descriptor[key] >> 20
        for key in (
            "mapping_bytes",
            "snapshot_budget_bytes",
            "backing_limit_bytes",
            "memory_request_bytes",
        )
    } == dict(
        mapping_bytes=mapping,
        snapshot_budget_bytes=mapping,
        backing_limit_bytes=backing,
        memory_request_bytes=memory,
    )
    pod = api.creates[0]["spec"]
    assert pod["volumes"][1]["emptyDir"]["sizeLimit"] == f"{backing}Mi"
    assert pod["containers"][0]["resources"]["requests"]["memory"] == f"{memory}Mi"
    assert pod["containers"][0]["resources"]["limits"]["memory"] == f"{memory}Mi"
    args = pod["containers"][0]["args"]
    assert float(args[args.index("--pool-sizes-gb") + 1]) == size
    claim = PoolClaim.parse({**_CLAIM, "size_gb": size})
    assert PoolClaim.parse(claim.as_dict()).size_gb == size


@pytest.mark.parametrize(
    "mutation", ["unknown", "changed-default", "duplicate", "intent"]
)
def test_unexpected_tolerations_rejected(tmp_path, mutation):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    broker.ensure(_CLAIM)
    tolerations = next(iter(api.pods.values()))["spec"]["tolerations"]
    if mutation == "unknown":
        tolerations.append(
            {"key": "other", "operator": "Exists", "effect": "NoExecute"}
        )
    elif mutation == "changed-default":
        tolerations[1]["tolerationSeconds"] = 301
    elif mutation == "duplicate":
        tolerations.append(copy.deepcopy(tolerations[1]))
    else:
        tolerations[0]["value"] = "false"
    with pytest.raises(ValueError, match="tolerations"):
        broker.inspect(_CLAIM)
    assert api.deletes == []


@pytest.mark.parametrize("target", ["memory", "volume"])
def test_snapshot_reservation_drift_rejected(tmp_path, target):
    api = FakeAPI()
    broker = make_broker(tmp_path, api)
    broker.ensure(_CLAIM)
    spec = next(iter(api.pods.values()))["spec"]
    if target == "memory":
        spec["containers"][0]["resources"]["requests"]["memory"] = "3172Mi"
    else:
        spec["volumes"][1]["emptyDir"]["sizeLimit"] = "1124Mi"
    with pytest.raises(ValueError, match="mismatch"):
        broker.inspect(_CLAIM)
    assert api.deletes == []


def test_toleration_intent_accepts_absent_default_operator():
    expected = [
        {
            "key": "nvidia.com/gpu",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]
    actual = [{"key": "nvidia.com/gpu", "value": "true", "effect": "NoSchedule"}]
    assert broker_module._matching_tolerations(actual, expected)


def test_pod_api_reads_rotated_projected_token_each_request(tmp_path, monkeypatch):
    import io

    service_account = tmp_path / "service-account"
    service_account.mkdir()
    token = service_account / "token"
    token.write_text("first-fixture-token\n")
    original_path = Path
    monkeypatch.setattr(
        broker_module,
        "Path",
        lambda path: service_account
        if str(path) == "/var/run/secrets/kubernetes.io/serviceaccount"
        else original_path(path),
    )
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "api.fixture.invalid")
    monkeypatch.setattr(
        broker_module.ssl, "create_default_context", lambda **kwargs: object()
    )
    headers = []

    def fake_open(request, **kwargs):
        headers.append(request.get_header("Authorization"))
        return io.BytesIO(b'{"items": []}')

    monkeypatch.setattr(broker_module.urllib.request, "urlopen", fake_open)
    api = broker_module.InClusterPodAPI("bench")
    assert api.list("gaas") == []
    replacement = service_account / "next-token"
    replacement.write_text("rotated-fixture-token\n")
    replacement.replace(token)
    assert api.list("gaas") == []
    assert headers == ["Bearer first-fixture-token", "Bearer rotated-fixture-token"]
