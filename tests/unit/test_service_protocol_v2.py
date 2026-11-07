# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import msgspec
import pytest
from test_kvcr_service import _running_server, _send_raw_request

from kvcr.service import protocol
from kvcr.service.memory import KVCRPoolSpec, _pool_filename
from kvcr.service.protocol import (
    _CLAIM_DECODER,
    _CLAIM_RESPONSE_DECODER,
    _RESIZE_RESPONSE_DECODER,
    CAPACITY_BELOW_MINIMUM,
    ERROR_REASONS,
    IDENTITY_MISMATCH,
    INCOMPATIBLE_LINEAGE,
    INTERNAL,
    NOT_IMPLEMENTED,
    REGION_BUSY,
    SLOT_HELD,
    UNKNOWN_SLOT,
    PlacementHints,
    PoolRegion,
    StateAgentSettings,
    _Claim,
    _Error,
    _Granted,
    _Grow,
    _Grown,
    _PoolDescriptor,
    _reason_error,
    _Shrink,
    _Shrunk,
    _TierConfig,
)

_TIER = _TierConfig([("pool", 4096)], None)  # the declared type is a list
_REGIONS = (PoolRegion(0, 0, 4096, None), PoolRegion(1, 4096, 8192, 1))


def _roundtrip(value, decoder):
    return decoder.decode(msgspec.msgpack.encode(value))


def test_v2_claim_roundtrip():
    claim = _Claim(
        0,
        "digest",
        _TIER,
        "127.0.0.1",
        5555,
        1,
        "inc",
        tenant="t",
        lineage="l",
        total_bytes=100,
        minimum_bytes=10,
        maximum_bytes=200,
        placement=PlacementHints("n0", 1, 2 << 20),
        state_agent=StateAgentSettings("ns", "c", "e", False),
    )
    assert _roundtrip(claim, _CLAIM_DECODER) == claim


def test_v1_claim_wire_is_unchanged_and_decodes_with_defaults():
    claim = _Claim(0, "digest", _TIER, "127.0.0.1", 5555, 1)
    wire = msgspec.to_builtins(claim)
    assert set(wire) == {
        "type",
        "guard_index",
        "compatibility_digest",
        "tier_config",
        "control_host",
        "control_port",
        "version",
    }
    decoded = _CLAIM_DECODER.decode(msgspec.msgpack.encode(wire))
    assert decoded.tenant is None and decoded.placement is None
    assert decoded.state_agent is None and decoded.total_bytes is None


def test_state_agent_and_placement_defaults():
    assert StateAgentSettings().advertise is True
    assert PlacementHints() == PlacementHints(None, None, None)


def _granted_wire() -> dict:
    spec = KVCRPoolSpec(
        "p", "/x/" + _pool_filename("p", "0" * 32), "0" * 32, 1, 2, 16384, 8192
    )
    return {
        "type": "granted",
        "guard_index": 0,
        "spec": msgspec.to_builtins(spec),
        "tier_config": msgspec.to_builtins(_TIER),
        "pools": msgspec.to_builtins((_PoolDescriptor("p", 4096, 4096),)),
        "version": 1,
    }


def test_v1_granted_decodes_with_defaults():
    granted = _CLAIM_RESPONSE_DECODER.decode(msgspec.msgpack.encode(_granted_wire()))
    assert granted.slot_name is None
    assert granted.effective_maximum_bytes is None
    assert granted.regions == ()
    assert granted.layout_generation == 0
    assert granted.guard_endpoint is None
    # Defaults stay off the wire, so a v1 reader sees the v1 shape.
    assert set(msgspec.to_builtins(granted)) == set(_granted_wire())


def test_v2_granted_roundtrip():
    wire = _granted_wire() | {
        "slot_name": "slot-a",
        "effective_maximum_bytes": 1 << 30,
        "regions": msgspec.to_builtins(_REGIONS),
        "layout_generation": 2,
        "guard_endpoint": "10.0.0.1:5555",
    }
    granted = _CLAIM_RESPONSE_DECODER.decode(msgspec.msgpack.encode(wire))
    assert isinstance(granted, _Granted)
    assert granted.regions == _REGIONS and granted.layout_generation == 2
    assert _roundtrip(granted, _CLAIM_RESPONSE_DECODER) == granted


def test_pool_region_roundtrip():
    for region in _REGIONS:
        assert _roundtrip(region, msgspec.msgpack.Decoder(PoolRegion)) == region
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"index": 0, "offset": 0, "size": 0}, PoolRegion)


def test_grow_shrink_and_responses_roundtrip():
    grow = _Grow(1, "slot-a", 4096, 2)
    assert _roundtrip(grow, _CLAIM_DECODER) == grow
    assert _roundtrip(_Grow(1, "slot-a", 4096), _CLAIM_DECODER).split is None
    shrink = _Shrink(1, "slot-a", (1, 2))
    assert _roundtrip(shrink, _CLAIM_DECODER) == shrink
    grown = _Grown(1, _REGIONS, 3)
    assert _roundtrip(grown, _RESIZE_RESPONSE_DECODER) == grown
    shrunk = _Shrunk(1, (1,), 4)
    assert _roundtrip(shrunk, _RESIZE_RESPONSE_DECODER) == shrunk


def test_error_reason_default_and_codes():
    assert _Error("m", 1).reason == INTERNAL
    # The v1 wire form of an error is unchanged.
    assert msgspec.to_builtins(_Error("m", 1)) == {
        "type": "error",
        "message": "m",
        "version": 1,
    }
    old = msgspec.msgpack.encode({"type": "error", "message": "m", "version": 1})
    assert _CLAIM_RESPONSE_DECODER.decode(old).reason == INTERNAL
    codes = {
        CAPACITY_BELOW_MINIMUM,
        SLOT_HELD,
        IDENTITY_MISMATCH,
        INCOMPATIBLE_LINEAGE,
        UNKNOWN_SLOT,
        REGION_BUSY,
        NOT_IMPLEMENTED,
        INTERNAL,
    }
    assert codes == ERROR_REASONS and len(codes) == 8
    for code in codes:
        error = _reason_error(code, "why")
        assert _roundtrip(error, _RESIZE_RESPONSE_DECODER) == error
        assert error.reason == code and error.version == protocol._PROTOCOL_VERSION
    with pytest.raises(ValueError):
        _reason_error("bogus", "why")


@pytest.mark.parametrize(
    "request_",
    [_Grow(1, "slot-a", 4096), _Shrink(1, "slot-a", (0,))],
)
def test_server_rejects_resize_with_a_reason(tmp_path, request_):
    with _running_server(tmp_path) as harness:
        response = _send_raw_request(harness.server.socket_path, request_)
    assert isinstance(response, _Error)
    assert response.reason == NOT_IMPLEMENTED
