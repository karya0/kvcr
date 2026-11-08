# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import socket

import msgspec
import pytest
from test_kvcr_service import _running_server, _send_raw_request

from kvcr.runtime.control_channels import FramedConnection, KVCRMsgFramingError
from kvcr.service import protocol
from kvcr.service.memory import KVCRPoolSpec, _pool_filename
from kvcr.service.protocol import (
    _CLAIM_DECODER,
    _CLAIM_RESPONSE_DECODER,
    _RESIZE_RESPONSE_DECODER,
    CAPACITY_BELOW_MINIMUM,
    DATA_POOL_LAYOUT,
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
    _ClaimV2,
    _Error,
    _ErrorV2,
    _Granted,
    _GrantedV2,
    _Grow,
    _Grown,
    _PoolDescriptor,
    _reason_error,
    _response_fd_counts,
    _Shrink,
    _Shrunk,
    _TierConfig,
)

_TIER = _TierConfig([("pool", 4096)], None)  # the declared type is a list
_REGIONS = (PoolRegion(0, 0, 4096, None), PoolRegion(1, 4096, 8192, 1))


def _roundtrip(value, decoder):
    return decoder.decode(msgspec.msgpack.encode(value))


def test_v2_claim_roundtrip():
    claim = _ClaimV2(
        0,
        "digest",
        _TIER,
        "127.0.0.1",
        5555,
        1,
        "t",
        "l",
        "inc",
        total_bytes=100,
        minimum_bytes=10,
        maximum_bytes=200,
        placement=PlacementHints("n0", 1, 2 << 20),
        state_agent=StateAgentSettings("ns", "c", "e", False),
    )
    assert _roundtrip(claim, _CLAIM_DECODER) == claim


# Captured by encoding the same values with src/kvcr/service/protocol.py of the
# prototype commit 4bf23cd, before any v2 type existed.
_V1_CLAIM_BYTES = (
    b"\x88\xa4type\xa5claim\xabguard_index\x00\xb4compatibility_digest\xa6digest"
    b"\xabtier_config\x83\xacpool_layouts\x91\x92\xa4pool\xcd\x10\x00\xa2g3\xc0"
    b"\xb6remote_fw_dram_backend\xa3UCX\xaccontrol_host\xa9127.0.0.1"
    b"\xaccontrol_port\xcd\x15\xb3\xa7version\x01\xabincarnation\xa3inc"
)
_V1_GRANTED_BYTES = (
    b"\x88\xa4type\xa7granted\xabguard_index\x00\xa4spec\x87\xa7pool_id\xa1p"
    b"\xa4path\xd9*/x/kvcr-p-00000000000000000000000000000000"
    b"\xaageneration\xd9 00000000000000000000000000000000\xa6device\x01\xa5inode\x02"
    b"\xadmapping_bytes\xcd@\x00\xadjournal_bytes\xcd \x00\xabtier_config\x83"
    b"\xacpool_layouts\x91\x92\xa4pool\xcd\x10\x00\xa2g3\xc0"
    b"\xb6remote_fw_dram_backend\xa3UCX\xa5pools\x91\x84\xa4name\xa1p"
    b"\xaasize_bytes\xcd\x10\x00\xb0block_size_bytes\xcd\x10\x00\xacoffset_bytes\x00"
    b"\xa7version\x01\xb1dead_incarnations\x91\xa1d\xb1heartbeat_enabled\xc3"
)
_V1_ERROR_BYTES = b"\x83\xa4type\xa5error\xa7message\xa1m\xa7version\x01"


def test_v1_wire_bytes_are_identical_to_the_prototype():
    spec = KVCRPoolSpec(
        "p", "/x/" + _pool_filename("p", "0" * 32), "0" * 32, 1, 2, 16384, 8192
    )
    claim = _Claim(0, "digest", _TIER, "127.0.0.1", 5555, 1, "inc")
    granted = _Granted(
        0, spec, _TIER, (_PoolDescriptor("p", 4096, 4096),), 1, ("d",), True
    )
    assert msgspec.msgpack.encode(claim) == _V1_CLAIM_BYTES
    assert msgspec.msgpack.encode(granted) == _V1_GRANTED_BYTES
    assert msgspec.msgpack.encode(_Error("m", 1)) == _V1_ERROR_BYTES
    assert _CLAIM_DECODER.decode(_V1_CLAIM_BYTES) == claim
    assert _CLAIM_RESPONSE_DECODER.decode(_V1_GRANTED_BYTES) == granted
    assert _CLAIM_RESPONSE_DECODER.decode(_V1_ERROR_BYTES) == _Error("m", 1)


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


def test_v2_granted_roundtrip():
    wire = _granted_wire() | {
        "type": "granted2",
        "slot_name": "slot-a",
        "effective_maximum_bytes": 1 << 30,
        "regions": msgspec.to_builtins(_REGIONS),
        "layout_generation": 2,
        "claim_generation": 5,
        "guard_endpoint": "10.0.0.1:5555",
    }
    granted = _CLAIM_RESPONSE_DECODER.decode(msgspec.msgpack.encode(wire))
    assert isinstance(granted, _GrantedV2)
    assert granted.regions == _REGIONS and granted.layout_generation == 2
    assert granted.claim_generation == 5
    assert _roundtrip(granted, _CLAIM_RESPONSE_DECODER) == granted
    # pool, listener, two region fds, the meta fd
    assert _response_fd_counts(granted) == (5,)


def test_response_receives_four_fds_over_a_socketpair():
    # A v2 grant with one region memfd carries four descriptors (the meta memfd
    # last); the receive buffer must hold them (it once held two: MSG_CTRUNC).
    wire = _granted_wire() | {
        "type": "granted2",
        "slot_name": "slot-a",
        "effective_maximum_bytes": 4096,
        "regions": msgspec.to_builtins((PoolRegion(0, 0, 4096, 0),)),
        "layout_generation": 1,
        "claim_generation": 1,
        "guard_endpoint": "10.0.0.1:5555",
    }
    granted = _CLAIM_RESPONSE_DECODER.decode(msgspec.msgpack.encode(wire))
    sender, receiver = socket.socketpair()
    owned = [os.open("/dev/null", os.O_RDONLY) for _ in range(4)]
    try:
        FramedConnection(sender).send_with_fd(granted, *owned)
        response, received = FramedConnection(receiver).receive_with_fd(
            _CLAIM_RESPONSE_DECODER, _response_fd_counts
        )
        assert response == granted and len(received) == 4
        for fd in received:
            os.close(fd)
    finally:
        for fd in owned:
            os.close(fd)
        sender.close()
        receiver.close()


def test_v1_response_with_extra_fds_is_still_refused():
    wire = _granted_wire()
    granted = _CLAIM_RESPONSE_DECODER.decode(msgspec.msgpack.encode(wire))
    sender, receiver = socket.socketpair()
    owned = [os.open("/dev/null", os.O_RDONLY) for _ in range(3)]
    try:
        FramedConnection(sender).send_with_fd(granted, *owned)
        with pytest.raises(KVCRMsgFramingError):
            FramedConnection(receiver).receive_with_fd(
                _CLAIM_RESPONSE_DECODER, _response_fd_counts
            )
    finally:
        for fd in owned:
            os.close(fd)
        sender.close()
        receiver.close()


def test_pool_region_roundtrip():
    for region in _REGIONS:
        assert _roundtrip(region, msgspec.msgpack.Decoder(PoolRegion)) == region
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"index": 0, "offset": 0, "size": 0}, PoolRegion)


def test_grow_shrink_and_responses_roundtrip():
    grow = _Grow(1, "slot-a", 4096, 7, 2)
    assert _roundtrip(grow, _CLAIM_DECODER) == grow
    assert _roundtrip(_Grow(1, "slot-a", 4096, 7), _CLAIM_DECODER).split is None
    shrink = _Shrink(1, "slot-a", (1, 2), 7)
    assert _roundtrip(shrink, _CLAIM_DECODER) == shrink
    grown = _Grown(1, _REGIONS, 3)
    assert _roundtrip(grown, _RESIZE_RESPONSE_DECODER) == grown
    shrunk = _Shrunk(1, (1,), 4)
    assert _roundtrip(shrunk, _RESIZE_RESPONSE_DECODER) == shrunk


def test_error_reason_default_and_codes():
    assert _ErrorV2("m", 1).reason == INTERNAL
    # v1 keeps its own error; the reason travels only in the tagged v2 error.
    assert msgspec.to_builtins(_Error("m", 1)) == {
        "type": "error",
        "message": "m",
        "version": 1,
    }
    assert isinstance(_reason_error(SLOT_HELD, "m"), _ErrorV2)
    codes = {
        CAPACITY_BELOW_MINIMUM,
        SLOT_HELD,
        IDENTITY_MISMATCH,
        INCOMPATIBLE_LINEAGE,
        UNKNOWN_SLOT,
        REGION_BUSY,
        NOT_IMPLEMENTED,
        INTERNAL,
        DATA_POOL_LAYOUT,
    }
    assert codes == ERROR_REASONS and len(codes) == 9
    for code in codes:
        error = _reason_error(code, "why")
        assert _roundtrip(error, _RESIZE_RESPONSE_DECODER) == error
        assert error.reason == code and error.version == protocol._PROTOCOL_VERSION
        assert msgspec.to_builtins(error)["type"] == "error2"
    with pytest.raises(ValueError):
        _reason_error("bogus", "why")


@pytest.mark.parametrize(
    "request_",
    [_Grow(1, "slot-a", 4096, 7), _Shrink(1, "slot-a", (0,), 7)],
)
def test_server_rejects_resize_with_a_reason(tmp_path, request_):
    with _running_server(tmp_path) as harness:
        response = _send_raw_request(harness.server.socket_path, request_)
    assert isinstance(response, _ErrorV2)
    assert response.reason == NOT_IMPLEMENTED


@pytest.mark.parametrize(
    ("gib", "seconds"), [(0, 180), (62, 304), (150, 480), (256, 692)]
)
def test_populate_wait_s(gib, seconds):
    from kvcr.service.protocol import populate_wait_s

    assert populate_wait_s(gib << 30) == seconds
    assert populate_wait_s(gib << 30, 10) == seconds - 170
