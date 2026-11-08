# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The wire between a KVCR worker and KVCR-Service, and the client half."""

import contextlib
import errno
import hashlib
import logging
import mmap
import os
import socket
import struct
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from math import gcd
from typing import TYPE_CHECKING, Annotated, Any, Literal

import msgspec

from ..backends.local_disk import _normalize_g3_backend_options
from ..config import G3Options, GaaSSettings, LocalDramOptions, _validate_pool_layouts
from ..runtime.control_channels import (
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRServiceError,
    KVCRSocketError,
)
from ..types import PoolBlockLayouts
from .memory import KVCRPoolAttachment, KVCRPoolSpec, _compute_pool_geometry

if TYPE_CHECKING:
    from .regions import RegionMapping

logger = logging.getLogger(__name__)

ProtocolVersion = Literal[1]
_PROTOCOL_VERSION: ProtocolVersion = 1

# SO_PEERPIDFD requires Linux 6.5 or later.
_SO_PEERPIDFD_FALLBACK = 77
_SO_PEERPIDFD = getattr(socket, "SO_PEERPIDFD", _SO_PEERPIDFD_FALLBACK)
_HEARTBEAT_SECONDS = 0.05
# A missed deadline SIGKILLs the primary, so keep the floor clear of GC pauses.
_MIN_HEARTBEAT_TIMEOUT_SECONDS = 20 * _HEARTBEAT_SECONDS

READINESS_WAIT_S = 180.0
POPULATE_RATE_FLOOR_BYTES_PER_S = 512 << 20


def populate_wait_s(data_bytes: int, base_s: float = READINESS_WAIT_S) -> float:
    """Seconds to wait for a holder to populate ``data_bytes`` of pool memory.

    ``base_s`` covers pod readiness and handshake; the rest assumes the holder
    populates (MADV_POPULATE_WRITE) at no less than 0.5 GiB/s. Measured in
    c5-repro-cpuquota-20261107.txt: 1.23 GiB/s at 1 CPU, 0.115 GiB/s at 0.1 CPU,
    so the floor is about 40% of the 1-CPU rate.
    """
    return base_s + data_bytes / POPULATE_RATE_FLOOR_BYTES_PER_S


def claim_wait_s(data_bytes: int) -> float:
    """How long a v2 claimant waits for its grant.

    The service spends up to READINESS_WAIT_S on the holder pod, then
    populate_wait_s() on its report; wait 20 s longer than both.
    """
    return READINESS_WAIT_S + populate_wait_s(data_bytes) + 20


class _G3Config(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    # Constrained here because the claimant that opens G3 under these terms is a
    # replacement started long after this claim was accepted.
    paths: Annotated[tuple[str, ...], msgspec.Meta(min_length=1)]
    capacity_bytes_per_file: Annotated[int, msgspec.Meta(gt=0)]
    backend: Annotated[str, msgspec.Meta(min_length=1)]
    backend_options: dict[str, str]

    def __post_init__(self) -> None:
        if not all(os.path.isabs(path) for path in self.paths):
            raise ValueError("G3 paths must be absolute")
        resolved = {os.path.realpath(path) for path in self.paths}
        if len(resolved) != len(self.paths):
            raise ValueError("G3 file paths must be unique")


class _PoolDescriptor(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One ordered pool region within a Guard-owned allocation."""

    name: str
    size_bytes: Annotated[int, msgspec.Meta(gt=0)]
    block_size_bytes: Annotated[int, msgspec.Meta(gt=0)]
    offset_bytes: Annotated[int, msgspec.Meta(ge=0)] = 0

    def __post_init__(self) -> None:
        _compute_pool_geometry(self.size_bytes, self.block_size_bytes)


class _TierConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    pool_layouts: PoolBlockLayouts
    g3: _G3Config | None
    remote_fw_dram_backend: Annotated[str, msgspec.Meta(min_length=1)] = "UCX"

    def __post_init__(self) -> None:
        _validate_pool_layouts(self.pool_layouts)
        if self.g3 is not None:
            block_size_bytes = gcd(*(size for _, size in self.pool_layouts))
            if block_size_bytes % mmap.PAGESIZE:
                raise ValueError("G3 requires page-aligned slots")
            if self.g3.capacity_bytes_per_file < block_size_bytes:
                raise ValueError(
                    "G3 file capacity must contain at least one complete slot"
                )


class PlacementHints(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    node: str | None = None
    numa_node: Annotated[int, msgspec.Meta(ge=0)] | None = None
    hugepage_size: Annotated[int, msgspec.Meta(gt=0)] | None = None


class StateAgentSettings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    namespace: str | None = None
    component: str | None = None
    endpoint: str | None = None
    advertise: bool = True


class PoolRegion(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One region of a slot's pool: a byte range, optionally in its own fd."""

    index: Annotated[int, msgspec.Meta(ge=0)]
    offset: Annotated[int, msgspec.Meta(ge=0)]
    size: Annotated[int, msgspec.Meta(gt=0)]
    fd_index: Annotated[int, msgspec.Meta(ge=0)] | None = None


def region_table_digest(
    regions: tuple[PoolRegion, ...],
    fds: list[int],
    slot_name: str = "",
    claim_generation: int = 0,
) -> bytes:
    """Names the layout and backing of a v2 data pool; the claimant records it in
    the journal header and the Guard refuses to adopt records written under
    another one. ``fds`` are the region memfds by fd_index: their (st_dev, st_ino)
    tell one slot's memory from another's of the same geometry."""
    backing = []
    for fd in fds:
        stat = os.fstat(fd)
        backing.append((stat.st_dev, stat.st_ino))
    table = [(r.index, r.offset, r.size, r.fd_index or 0) for r in regions]
    packed = msgspec.msgpack.encode((table, backing, slot_name, claim_generation))
    return hashlib.sha256(packed).digest()


# Reason codes carried by _Error; a client branches on these, not on message.
CAPACITY_BELOW_MINIMUM = "capacity_below_minimum"
SLOT_HELD = "slot_held"
IDENTITY_MISMATCH = "identity_mismatch"
INCOMPATIBLE_LINEAGE = "incompatible_lineage"
UNKNOWN_SLOT = "unknown_slot"
REGION_BUSY = "region_busy"
NOT_IMPLEMENTED = "not_implemented"
INTERNAL = "internal"
DATA_POOL_LAYOUT = "data_pool_layout"
ERROR_REASONS = frozenset(
    {
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
)


def _require_literal_ipv4(host: str) -> None:
    # A literal address, because the service binds this holding the lock
    # every other pool needs, and resolving a name there would stall all
    # of them for as long as the resolver takes.
    try:
        socket.inet_pton(socket.AF_INET, host)
    except OSError as error:
        raise ValueError(f"not a literal IPv4 address: {host}") from error


class _Claim(msgspec.Struct, frozen=True, tag="claim"):
    guard_index: Annotated[int, msgspec.Meta(ge=0)]
    compatibility_digest: str
    tier_config: _TierConfig
    control_host: str
    control_port: Annotated[int, msgspec.Meta(ge=1, le=65535)]
    version: ProtocolVersion
    incarnation: Annotated[str, msgspec.Meta(min_length=1)] | None = None

    def __post_init__(self) -> None:
        _require_literal_ipv4(self.control_host)


class _ClaimV2(msgspec.Struct, frozen=True, tag="claim2"):
    """A GaaS claim: a tenant/lineage slot with a byte budget. v1 stays _Claim."""

    guard_index: Annotated[int, msgspec.Meta(ge=0)]
    compatibility_digest: str
    tier_config: _TierConfig
    control_host: str
    control_port: Annotated[int, msgspec.Meta(ge=1, le=65535)]
    version: ProtocolVersion
    tenant: Annotated[str, msgspec.Meta(min_length=1)]
    lineage: Annotated[str, msgspec.Meta(min_length=1)]
    incarnation: Annotated[str, msgspec.Meta(min_length=1)] | None = None
    total_bytes: Annotated[int, msgspec.Meta(gt=0)] | None = None
    minimum_bytes: Annotated[int, msgspec.Meta(gt=0)] | None = None
    maximum_bytes: Annotated[int, msgspec.Meta(gt=0)] | None = None
    placement: PlacementHints | None = None
    state_agent: StateAgentSettings | None = None

    def __post_init__(self) -> None:
        _require_literal_ipv4(self.control_host)


class _Release(msgspec.Struct, frozen=True, tag="release"):
    version: ProtocolVersion
    # False when the claimant never served this lease: the grant arrived but
    # mapping, adoption, or startup failed. Local access has already stopped
    # by the time any release is sent, so the service may act on it.
    activated: bool = True


class _Heartbeat(msgspec.Struct, frozen=True, tag="heartbeat"):
    version: ProtocolVersion


class _Granted(msgspec.Struct, frozen=True, tag="granted"):
    guard_index: int
    spec: KVCRPoolSpec
    tier_config: _TierConfig
    pools: tuple[_PoolDescriptor, ...]
    version: ProtocolVersion
    dead_incarnations: tuple[str, ...] = ()
    heartbeat_enabled: bool = False


class _GrantedV2(msgspec.Struct, frozen=True, tag="granted2"):
    guard_index: int
    spec: KVCRPoolSpec
    tier_config: _TierConfig
    pools: tuple[_PoolDescriptor, ...]
    version: ProtocolVersion
    slot_name: str
    effective_maximum_bytes: int
    regions: tuple[PoolRegion, ...]
    layout_generation: int
    claim_generation: int
    guard_endpoint: str
    dead_incarnations: tuple[str, ...] = ()
    heartbeat_enabled: bool = False


class _Grow(msgspec.Struct, frozen=True, tag="grow"):
    version: ProtocolVersion
    slot_name: str
    additional_bytes: Annotated[int, msgspec.Meta(gt=0)]
    # From the grant: a resize by a stale claim of the same pod is refused.
    claim_generation: int
    # Number of regions to add the bytes as; None lets the service choose.
    split: Annotated[int, msgspec.Meta(gt=0)] | None = None


class _Shrink(msgspec.Struct, frozen=True, tag="shrink"):
    version: ProtocolVersion
    slot_name: str
    region_indices: tuple[int, ...]
    claim_generation: int


class _Grown(msgspec.Struct, frozen=True, tag="grown"):
    version: ProtocolVersion
    regions: tuple[PoolRegion, ...]
    layout_generation: int


class _Shrunk(msgspec.Struct, frozen=True, tag="shrunk"):
    version: ProtocolVersion
    region_indices: tuple[int, ...]
    layout_generation: int


class _Released(msgspec.Struct, frozen=True, tag="released"):
    version: ProtocolVersion


class _Error(msgspec.Struct, frozen=True, tag="error"):
    message: str
    version: ProtocolVersion


class _ErrorV2(msgspec.Struct, frozen=True, tag="error2"):
    """An error with a reason code, sent only on v2 paths; v1 keeps _Error."""

    message: str
    version: ProtocolVersion
    reason: str = INTERNAL


_ERRORS = (_Error, _ErrorV2)


class KVCRServiceReasonError(KVCRServiceError):
    """A v2 refusal; ``reason`` is one of ERROR_REASONS, ``message`` the service's."""

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(f"{message} (reason: {reason})")
        self.message, self.reason = message, reason


def _service_error(response: "_Error | _ErrorV2") -> KVCRServiceError:
    if isinstance(response, _ErrorV2):
        return KVCRServiceReasonError(response.message, response.reason)
    return KVCRServiceError(response.message)


def _reason_error(reason: str, message: str) -> _ErrorV2:
    if reason not in ERROR_REASONS:
        raise ValueError(f"unknown error reason: {reason}")
    return _ErrorV2(message, _PROTOCOL_VERSION, reason)


class _ServiceInfo(msgspec.Struct, frozen=True, tag="service_info"):
    version: ProtocolVersion


class _Drain(msgspec.Struct, frozen=True, tag="drain"):
    version: ProtocolVersion


class _ServiceState(msgspec.Struct, frozen=True, tag="service_state"):
    info: dict[str, Any]
    version: ProtocolVersion


def _claim_fd_counts(
    request: "_Claim | _ClaimV2 | _ServiceInfo | _Drain | _Grow | _Shrink",
) -> tuple[int, ...]:
    return (0, 1) if isinstance(request, (_Claim, _ClaimV2)) else (0,)


def _response_fd_counts(response: "_Granted | _GrantedV2 | _Error | _ErrorV2"):
    """An error carries none; a grant the pool, then the control listener.

    A v2 grant appends one region memfd per distinct fd_index, then the meta memfd.
    """
    if isinstance(response, _ERRORS):
        return (0,)
    if isinstance(response, _Granted):
        return (2,)
    return (3 + len({region.fd_index or 0 for region in response.regions} or {0}),)


_CLAIM_DECODER = msgspec.msgpack.Decoder(
    _Claim | _ClaimV2 | _ServiceInfo | _Drain | _Grow | _Shrink
)
_HOLD_DECODER = msgspec.msgpack.Decoder(_Heartbeat | _Release)
_CLAIM_RESPONSE_DECODER = msgspec.msgpack.Decoder(
    _Granted | _GrantedV2 | _Error | _ErrorV2
)
_RESIZE_RESPONSE_DECODER = msgspec.msgpack.Decoder(_Grown | _Shrunk | _Error | _ErrorV2)
_RELEASE_RESPONSE_DECODER = msgspec.msgpack.Decoder(_Released | _Error | _ErrorV2)
_SERVICE_STATE_DECODER = msgspec.msgpack.Decoder(_ServiceState | _Error | _ErrorV2)


def _service_request(
    socket_path: str | os.PathLike[str], request: Any, timeout_s: float
) -> dict:
    if not 0 < timeout_s < float("inf"):
        raise ValueError("service request timeout must be positive and finite")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_s)
        connection.connect(os.fspath(socket_path))
        channel = FramedConnection(connection)
        channel.send(request)
        response = channel.receive(_SERVICE_STATE_DECODER)
    if isinstance(response, _ERRORS):
        raise KVCRServiceError(response.message)
    return response.info


def service_info(socket_path: str | os.PathLike[str], timeout_s: float = 30.0) -> dict:
    """Inspect a holder without acquiring a worker lease or changing its pools."""
    return _service_request(socket_path, _ServiceInfo(_PROTOCOL_VERSION), timeout_s)


def drain_service(socket_path: str | os.PathLike[str], timeout_s: float = 30.0) -> dict:
    """Request irreversible drain; only ``state == 'drained'`` permits deletion.

    Busy, errors and transport uncertainty retain the holder reservation. The
    caller may inspect or repeat this idempotent request after references close.
    """
    return _service_request(socket_path, _Drain(_PROTOCOL_VERSION), timeout_s)


def _verify_peer_pidfd(connection: socket.socket, worker_pidfd: int) -> None:
    """Refuse a descriptor that is not a pidfd for this socket's peer.

    With /proc mounted for the service's PID namespace, SO_PEERCRED and fdinfo
    report comparable PIDs. A non-pidfd has no Pid: line and a peer outside
    the namespace reads as 0; both fail closed.
    """
    try:
        peer_pid = struct.unpack(
            "3i",
            connection.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            ),
        )[0]
        with open(f"/proc/self/fdinfo/{worker_pidfd}") as info:
            worker_pid = next(
                (int(line.split()[1]) for line in info if line.startswith("Pid:")),
                None,
            )
    except (OSError, ValueError) as error:
        raise KVCRServiceError("cannot verify worker pidfd") from error
    if peer_pid <= 0 or worker_pid != peer_pid:
        raise KVCRServiceError("worker pidfd does not match socket peer")


class PidfdLiveness:
    """Own the pidfd returned for an accepted Unix-socket peer."""

    def __init__(self, pidfd: int) -> None:
        self._pidfd = pidfd
        self.incarnation: str | None = None
        self._close_lock = threading.Lock()

    @classmethod
    def from_peer_socket(
        cls, connection: socket.socket, worker_pidfd: int | None = None
    ) -> "PidfdLiveness":
        """Prefer the kernel's peer pidfd; fall back to a verified transferred one."""
        with contextlib.ExitStack() as transferred:
            if worker_pidfd is not None:
                transferred.callback(os.close, worker_pidfd)
            try:
                pidfd = connection.getsockopt(socket.SOL_SOCKET, _SO_PEERPIDFD)
            except OSError as error:
                if error.errno != errno.ENOPROTOOPT:
                    raise
                if worker_pidfd is None:
                    message = "worker pidfd is required without SO_PEERPIDFD"
                    logger.warning(message)
                    raise KVCRServiceError(message) from error
                _verify_peer_pidfd(connection, worker_pidfd)
                liveness = cls(worker_pidfd)
                transferred.pop_all()
                return liveness
            return cls(pidfd)

    def fileno(self) -> int:
        if self._pidfd < 0:
            raise ValueError("pidfd is closed")
        return self._pidfd

    def close(self) -> None:
        """Give the descriptor up either way; a close failure is only logged."""
        # Under a lock: shutdown can race a failed claim's cleanup here, and
        # an unsynchronized swap lets both threads close -- the second close
        # can reach a descriptor number the process has since reused.
        with self._close_lock:
            pidfd, self._pidfd = self._pidfd, -1
        if pidfd >= 0:
            try:
                os.close(pidfd)
            except OSError:
                # EBADF means this number is already someone else's, so the
                # close may have reached an unrelated descriptor.
                logger.warning("Failed to close a KVCR pidfd", exc_info=True)


@dataclass
class KVCRPoolHold:
    """One mapped pool group and the connection holding its lease."""

    local_dram: LocalDramOptions
    _pools: tuple[_PoolDescriptor, ...]
    _attachment: KVCRPoolAttachment
    _connection: FramedConnection
    _control_listener_fd: int | None = None
    _incarnation: str | None = None
    _dead_incarnations: tuple[str, ...] = ()
    _heartbeat_enabled: bool = False
    _data_map: "RegionMapping | None" = None
    # v2 grants only: the slot and its regions; region fds are owned by the hold.
    slot_name: str | None = None
    claim_generation: int | None = None
    layout_generation: int | None = None
    effective_maximum_bytes: int | None = None
    regions: tuple[PoolRegion, ...] = ()
    region_fds: list[int] = field(default_factory=list, repr=False)
    guard_endpoint: str | None = None
    _release_attempted: bool = field(default=False, init=False, repr=False)
    _heartbeat_stop: threading.Event = field(
        default_factory=threading.Event, init=False
    )
    _send_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self._heartbeat_enabled:
            threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self) -> None:
        while not self._heartbeat_stop.wait(_HEARTBEAT_SECONDS):
            with self._send_lock:
                if self._heartbeat_stop.is_set():
                    return
                try:
                    self._connection.send(_Heartbeat(_PROTOCOL_VERSION))
                except OSError:
                    return

    def hand_listener_to(self, adopt: Callable[[int], None]) -> None:
        """Adopt-then-disown: a failed adoption leaves this hold owning the fd,
        which is what closes it on release."""
        if self._control_listener_fd is None:
            return
        adopt(self._control_listener_fd)
        self._control_listener_fd = None

    def _stop_local_access(self) -> None:
        if self._data_map is not None:
            self._data_map.close()
        self._attachment.close()
        self._heartbeat_stop.set()
        for fd in (self._control_listener_fd, *self.region_fds):
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._control_listener_fd = None
        self.region_fds.clear()

    def close_local(self) -> None:
        """Stop local access and drop the connection without releasing the lease.

        The service frees the lease when this process dies, as on a crash.
        """
        if self._release_attempted:
            return
        self._stop_local_access()
        self._release_attempted = True
        _close_quietly(self._connection)

    def release(self, *, activated: bool = True) -> None:
        """Stop local access before releasing the connection-scoped lease.

        ``activated=False`` tells the service this lease never served: the
        Guard it stood down may resume instead of leaving the pool group idle.
        """
        if self._release_attempted:
            return
        self._stop_local_access()
        self._release_attempted = True

        try:
            with self._send_lock:
                _send_release(self._connection, activated=activated)
            self._connection.close()
        except BaseException as error:
            _close_quietly(self._connection)
            if isinstance(error, (OSError, EOFError)):
                raise KVCRSocketError(
                    f"KVCR-Service release failed: {error}"
                ) from error
            raise


class KVCRClient:
    """Synchronous client for a standalone KVCR service."""

    def __init__(self, socket_path: str | os.PathLike[str]) -> None:
        self._socket_path = os.fspath(socket_path)

    def claim(
        self,
        guard_index: int,
        pool_layouts: PoolBlockLayouts,
        compatibility_digest: str,
        control_bind: tuple[str, int],
        g3: G3Options | None = None,
        remote_fw_dram_backend: str = "UCX",
        *,
        gaas: GaaSSettings | None = None,
        receive_timeout_s: float | None = None,
    ) -> KVCRPoolHold:
        """Claim and map one Guard-owned pool group.

        With ``gaas`` carrying tenant and lineage this sends the v2 claim and the
        returned hold also names the slot and its regions; otherwise the v1 claim
        goes out unchanged. ``receive_timeout_s`` overrides the v2 grant wait.
        """
        g3_config = g3 and {
            "paths": [str(path.expanduser().resolve()) for path in g3.paths],
            "capacity_bytes_per_file": g3.capacity_bytes_per_file,
            "backend": g3.backend,
            "backend_options": _normalize_g3_backend_options(g3),
        }
        # msgspec.convert validates where __init__ would not: bad pool layout,
        # port or G3 path fails here, not at the service.
        fields: dict[str, Any] = {
            "guard_index": guard_index,
            "compatibility_digest": compatibility_digest,
            "tier_config": {
                "pool_layouts": pool_layouts,
                "g3": g3_config,
                "remote_fw_dram_backend": remote_fw_dram_backend,
            },
            "control_host": control_bind[0],
            "control_port": control_bind[1],
            "version": _PROTOCOL_VERSION,
            "incarnation": uuid.uuid4().hex,
        }
        v2 = gaas is not None and gaas.is_v2
        if v2:
            fields.update(
                {
                    name: getattr(gaas, name)
                    for name in (
                        "tenant",
                        "lineage",
                        "total_bytes",
                        "minimum_bytes",
                        "maximum_bytes",
                        "state_agent",
                        "placement",
                    )
                }
            )
        request = msgspec.convert(fields, type=_ClaimV2 if v2 else _Claim)
        connection = FramedConnection.connect(self._socket_path)
        if v2:
            # connect() leaves its short timeout.
            connection._connection.settimeout(
                receive_timeout_s
                if receive_timeout_s is not None
                else claim_wait_s(gaas.maximum_bytes or gaas.total_bytes or 0)
            )
        attachment: KVCRPoolAttachment | None = None
        data_map: RegionMapping | None = None
        listener_fd: int | None = None
        pool_fd: int | None = None
        meta_fd: int | None = None
        region_fds: list[int] = []
        grant_received = False
        try:
            try:
                worker_pidfd = os.pidfd_open(os.getpid())
            except (AttributeError, OSError):
                connection.send(request)
            else:
                try:
                    connection.send_with_fd(request, worker_pidfd)
                finally:
                    os.close(worker_pidfd)
            response, received_fds = connection.receive_with_fd(
                _CLAIM_RESPONSE_DECODER, _response_fd_counts
            )
            if isinstance(response, _ERRORS):
                raise _service_error(response)
            pool_fd, listener_fd, *region_fds = received_fds
            if not isinstance(response, _GrantedV2):  # v1 maps none of them
                for region_fd in region_fds:
                    os.close(region_fd)
                region_fds = []
            else:
                meta_fd = region_fds.pop()  # the journal and snapshot memfd, last
            grant_received = True
            spec, granted_pools = _grant_layout(
                response, guard_index, request.tier_config
            )
            fd, pool_fd = pool_fd, None
            if meta_fd is None:
                attachment = KVCRPoolAttachment.attach_fd(fd, spec)
            else:
                os.close(fd)  # v2 journals in the meta memfd, not pool_{rank}
                fd, meta_fd = meta_fd, None
                try:
                    attachment = KVCRPoolAttachment.attach_meta_fd(fd, spec)
                except (OSError, ValueError) as error:
                    raise KVCRServiceReasonError(
                        f"v2 meta region refused: {error}", DATA_POOL_LAYOUT
                    ) from error
            if not v2:
                dram_pools = [
                    (pool.name, attachment.address + pool.offset_bytes, pool.size_bytes)
                    for pool in granted_pools
                ]
            else:
                # The data pool is the regions; the meta memfd keeps the journal.
                data_map = _map_data_pool(response, gaas, region_fds)
                attachment.set_layout_digest(
                    region_table_digest(
                        response.regions,
                        region_fds,
                        response.slot_name,
                        response.claim_generation,
                    )
                )
                dram_pools = [
                    (granted_pools[0].name, data_map.address, data_map.size_bytes)
                ]
            return KVCRPoolHold(
                local_dram=LocalDramOptions(
                    dram_pools, request.tier_config.remote_fw_dram_backend
                ),
                _data_map=data_map,
                _pools=granted_pools,
                _attachment=attachment,
                _connection=connection,
                _control_listener_fd=listener_fd,
                _incarnation=request.incarnation,
                _dead_incarnations=response.dead_incarnations,
                _heartbeat_enabled=response.heartbeat_enabled,
                **(_slot_fields(response, region_fds) if v2 else {}),
            )
        except BaseException as error:
            # Release the lease only after local access has stopped, or the
            # next claimant could map bytes this process can still reach.
            for leftover in (listener_fd, pool_fd, meta_fd, *region_fds):
                if leftover is not None:
                    with contextlib.suppress(OSError):
                        os.close(leftover)
            local_access_stopped = True
            if data_map is not None:
                data_map.close()
            if attachment is not None:
                try:
                    attachment.close()
                except BaseException:
                    # Even an interrupt mid-close must gate the release.
                    local_access_stopped = False
            if local_access_stopped:
                # Sent even without a grant seen: the service may hold a lease
                # this claimant never learned it won; only this message or death
                # -- never bare EOF -- rolls it back. Must not mask the claim error.
                with contextlib.suppress(BaseException):
                    if v2:
                        # Do not await a reply: after a receive timeout the peer
                        # is silent and a second wait would double the delay.
                        connection.send(_Release(_PROTOCOL_VERSION, False))
                    else:
                        _send_release(connection, activated=False)
            _close_quietly(connection)
            if not grant_received and isinstance(error, (OSError, EOFError)):
                raise KVCRSocketError(f"KVCR-Service claim failed: {error}") from error
            raise

    def grow(
        self,
        slot_name: str,
        additional_bytes: int,
        claim_generation: int,
        split: int | None = None,
    ) -> _Grown:
        """Add bytes to a slot as new regions; the new regions' fds are not sent."""
        return self._resize(
            _Grow(
                _PROTOCOL_VERSION,
                slot_name,
                additional_bytes,
                claim_generation,
                split,
            )
        )

    def shrink(
        self, slot_name: str, region_indices: tuple[int, ...], claim_generation: int
    ) -> _Shrunk:
        """Drop regions from a slot; each must be unmapped by the caller first."""
        return self._resize(
            _Shrink(
                _PROTOCOL_VERSION, slot_name, tuple(region_indices), claim_generation
            )
        )

    def _resize(self, request: "_Grow | _Shrink") -> Any:
        # A fresh connection per call: the claim's own connection carries only
        # the lease.
        connection = FramedConnection.connect(self._socket_path)
        try:
            connection.send(request)
            response = connection.receive(_RESIZE_RESPONSE_DECODER)
        finally:
            connection.close()
        if isinstance(response, _ERRORS):
            raise _service_error(response)
        return response


def _map_data_pool(
    response: "_GrantedV2", gaas: GaaSSettings, region_fds: list[int]
) -> "RegionMapping":
    """Map the granted regions as the one data pool, or fail the claim closed.

    The engine sized its block pool before claiming, so a partial grant cannot
    be used: specs set minimum = maximum = total and anything else is refused.
    """
    from .regions import RegionMapping

    granted = sum(region.size for region in response.regions)
    if len(response.pools) != 1:
        problem = f"{len(response.pools)} pools; a v2 data pool is one pool"
    elif gaas.total_bytes is None:
        problem = "no total_bytes requested"
    elif granted != gaas.total_bytes:
        problem = f"granted {granted} bytes, requested total {gaas.total_bytes}"
    elif granted % response.pools[0].block_size_bytes:
        problem = f"{granted} bytes is not a whole number of blocks"
    else:
        try:
            return RegionMapping(region_fds, response.regions)
        except (OSError, ValueError) as error:
            problem = f"cannot map the regions: {error}"
    raise KVCRServiceReasonError(f"v2 data pool refused: {problem}", DATA_POOL_LAYOUT)


def _slot_fields(response: _GrantedV2, region_fds: list[int]) -> dict[str, Any]:
    return {
        "slot_name": response.slot_name,
        "claim_generation": response.claim_generation,
        "layout_generation": response.layout_generation,
        "effective_maximum_bytes": response.effective_maximum_bytes,
        "regions": response.regions,
        "region_fds": region_fds,
        "guard_endpoint": response.guard_endpoint,
    }


def _grant_layout(
    response: "_Granted | _GrantedV2",
    requested_guard_index: int,
    requested_tier_config: _TierConfig,
) -> tuple[KVCRPoolSpec, tuple[_PoolDescriptor, ...]]:
    """Take the grant apart, refusing one that answers a different request."""
    if response.guard_index != requested_guard_index:
        raise KVCRGuardProtocolError(
            "claim Guard mismatch: "
            f"requested {requested_guard_index}, got {response.guard_index}"
        )
    if response.tier_config != requested_tier_config:
        raise KVCRGuardProtocolError("claim tier configuration mismatch")
    pools = response.pools
    if len(pools) != len(requested_tier_config.pool_layouts):
        raise KVCRGuardProtocolError("claim pool count mismatch")
    expected_offset = response.spec.journal_bytes
    for index, (pool, layout) in enumerate(
        zip(pools, requested_tier_config.pool_layouts)
    ):
        if (
            (pool.name, pool.block_size_bytes) != layout
            or pool.offset_bytes != expected_offset
            or pool.size_bytes % mmap.PAGESIZE
        ):
            raise KVCRGuardProtocolError(f"claim pool {index} layout mismatch")
        expected_offset += pool.size_bytes
    # A v2 data pool is the regions, not pool_{rank}'s own pools.
    if expected_offset != response.spec.mapping_bytes and not isinstance(
        response, _GrantedV2
    ):
        raise KVCRGuardProtocolError("claim pool sizes do not fill the allocation")
    return response.spec, pools


def _send_release(connection: FramedConnection, *, activated: bool = True) -> None:
    connection.send(_Release(_PROTOCOL_VERSION, activated))
    response = connection.receive(_RELEASE_RESPONSE_DECODER)
    if isinstance(response, _ERRORS):
        raise KVCRServiceError(response.message)


def _close_quietly(connection: FramedConnection) -> None:
    # Only called while an original error is unwinding; a close failure
    # (even an interrupt) must not mask it.
    with contextlib.suppress(BaseException):
        connection.close()
