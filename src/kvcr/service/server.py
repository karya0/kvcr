# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR-Service owns Guarded shared-memory pool groups beyond a worker's life."""

import argparse
import contextlib
import functools
import logging
import mmap
import os
import re
import select
import signal
import socket
import socketserver
import stat
import struct
import sys
import threading
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import urlsplit

import msgspec

from ..runtime.control_channels import (
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    KVCRServiceError,
)
from ..runtime.progress import _STARTUP_TIMEOUT_SECONDS
from . import memmap
from .guard import _Guard, _Lease, _Phase
from .health import _serve_health
from .memory import (
    _OPEN_FLAGS,
    _POOL_PREFIX,
    KVCRPoolSpec,
    _KVCRPoolOwner,
    _pool_dir_guard,
    _reclaim_pool_if_orphaned,
    _unlink_if_identity,
)
from .node_service import _JOURNAL_BYTES as _NODE_JOURNAL_BYTES
from .node_service import InClusterPodAPI, NodeBroker, SlotError, SlotTable
from .protocol import (
    _CLAIM_DECODER,
    _HOLD_DECODER,
    _MIN_HEARTBEAT_TIMEOUT_SECONDS,
    _PROTOCOL_VERSION,
    INTERNAL,
    NOT_IMPLEMENTED,
    KVCRServiceReasonError,
    PidfdLiveness,
    _Claim,
    _claim_fd_counts,
    _ClaimV2,
    _Drain,
    _Error,
    _Granted,
    _GrantedV2,
    _Grow,
    _Grown,
    _Heartbeat,
    _PoolDescriptor,
    _reason_error,
    _Released,
    _ServiceInfo,
    _ServiceState,
    _Shrink,
    _Shrunk,
    _TierConfig,
    region_table_digest,
)

logger = logging.getLogger(__name__)

_PRIVATE_SOCKET_UMASK = 0o177
_CLIENT_IDLE_TIMEOUT_SECONDS = 30.0
_REGISTRY_TRANSITION_TIMEOUT_SECONDS = 30.0
_STALE_SOCKET_PROBE_SECONDS = 1.0
_DEFAULT_JOURNAL_BYTES = _NODE_JOURNAL_BYTES  # one source: node_service's
_V2_STUB_JOURNAL_BYTES = 2 * 4096  # smallest valid journal: header plus one page
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# Matches names produced by memory._pool_filename: "kvcr-<pool_id>-<32 hex>".
_ORPHANED_POOL_NAME = re.compile(rf"{re.escape(_POOL_PREFIX)}-.+-[0-9a-f]{{32}}")
_GB = 1 << 30


class _PoolRegistry:
    """A directory of pool groups, each owned by one Guard thread.

    No locks: each Guard's mailbox orders its claims, releases and deaths;
    Guards share nothing but the refusal flag.
    """

    def __init__(
        self,
        pool_dir: str | os.PathLike[str],
        guard_count: int,
        pool_sizes_bytes: tuple[int, ...],
        journal_bytes: int,
        compatibility_digest: str,
        heartbeat_timeout_seconds: float | None = None,
        startup_timeout_seconds: int | None = None,
        v2_slots: bool = False,
    ) -> None:
        self._pool_dir = Path(pool_dir).resolve()
        if not self._pool_dir.is_dir():
            raise ValueError(f"KVCR pool directory does not exist: {self._pool_dir}")
        self._guard_count = guard_count
        self._compatibility_digest = compatibility_digest
        self._guards: dict[int, _Guard] = {}
        self._refusing = threading.Event()
        self._draining = threading.Event()
        self._drained = False
        self._legacy_closing = False
        self._drain_lock = threading.RLock()
        # Stand-in until the owning server takes over; only it can stop the service.
        self.on_uncontained_failure = functools.partial(
            logger.critical, "Uncontained KVCR pool failure: %s"
        )
        self._purge_orphaned_pools()
        mapping_bytes = journal_bytes + sum(pool_sizes_bytes)
        for rank in range(guard_count):
            try:
                # v2 journals and snapshots in the slot's holder-charged meta memfd
                # and serves the slot's regions, so pool_{rank} is a sparse stub the
                # supervisor is not charged for. The grant still describes the full
                # layout (journal_bytes sizes the meta journal); only the file is small.
                owner = _KVCRPoolOwner.allocate(
                    pool_id=f"pool_{rank}",
                    pool_size_bytes=(
                        _V2_STUB_JOURNAL_BYTES + mmap.PAGESIZE
                        if v2_slots
                        else mapping_bytes
                    ),
                    journal_bytes=(
                        _V2_STUB_JOURNAL_BYTES if v2_slots else journal_bytes
                    ),
                    pool_dir=self._pool_dir,
                    populate=not v2_slots,
                )
                grant_spec = (
                    msgspec.structs.replace(
                        owner.spec,
                        mapping_bytes=mapping_bytes,
                        journal_bytes=journal_bytes,
                    )
                    if v2_slots
                    else owner.spec
                )
                # Built with the group, not a claim: a Guard that cannot attach its
                # allocation is better discovered at startup than when a worker dies.
                try:
                    guard = _Guard(
                        grant_spec,
                        functools.partial(self._guard_failed, rank),
                        pool_spec=owner.spec,
                        compatibility_digest=compatibility_digest,
                        guard_index=rank,
                        pool_sizes_bytes=pool_sizes_bytes,
                        owner=owner,
                        refusing=self._refusing.is_set,
                        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                        startup_timeout_seconds=startup_timeout_seconds,
                    )
                except BaseException:
                    # Nothing has recorded this pool group yet, so the sweep
                    # below cannot reach it and its file would outlive the process.
                    owner.close()
                    raise
                # Recorded before it starts, so a failed preparation is rolled back by
                # the sweep below.
                self._guards[rank] = guard
                guard.start()
            except BaseException:
                self._release_pools()
                raise

    def _release_pools(self) -> None:
        """Give back every pool group built so far after a failed startup."""
        for guard_index, guard in list(self._guards.items()):
            # Even an interrupt must not stop the sweep: the startup failure is
            # already propagating, and every group left behind is committed RAM.
            try:
                guard.close()
            except BaseException:
                # The Guard's thread may still hold this group's mapping, and
                # unlinking under it would fault the process. Leave the file
                # for the next start's purge, and keep the pool visible.
                logger.warning(
                    "Failed to close KVCR Guard for %s; leaving its pool in place",
                    guard._spec.path,
                    exc_info=True,
                )
                continue
            del self._guards[guard_index]

    def _purge_orphaned_pools(self) -> None:
        """Remove pool files no live daemon owns.

        Crash leftovers fill the fixed-size directory and fail the next eager
        allocation. Use is decided by the shared flock, never the filename.
        """
        self._purge_orphaned_pools_in(self._pool_dir)
        for directory in memmap.pool_dirs():
            try:
                self._purge_orphaned_pools_in(directory)
            except OSError as error:  # optional cleanup must not stop the service
                logger.warning("Purge skipped for %s: %s", directory, error)

    def _purge_orphaned_pools_in(self, directory: Path) -> None:
        with _pool_dir_guard(directory, exclusive=True):
            for path in sorted(directory.glob(f"{_POOL_PREFIX}-*")):
                recognized = _ORPHANED_POOL_NAME.fullmatch(path.name) is not None
                if not recognized or path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                    reclaimed = _reclaim_pool_if_orphaned(path)
                except OSError:
                    logger.warning(
                        "Failed to inspect candidate KVCR pool: %s", path, exc_info=True
                    )
                    continue
                if reclaimed:
                    logger.warning(
                        "Reclaimed orphaned KVCR pool from a dead instance: "
                        "%s (%d bytes)",
                        path,
                        size,
                    )
                else:
                    logger.info("Leaving KVCR pool still in use: %s", path)

    def _guard(self, guard_index: int) -> _Guard:
        if not (0 <= guard_index < self._guard_count):
            raise KVCRServiceError(
                f"guard_index {guard_index} is out of range [0, {self._guard_count})"
            )
        guard = self._guards.get(guard_index)
        if guard is None:
            raise KVCRServiceError(f"no claimable KVCR Guard {guard_index}")
        return guard

    def claim(
        self,
        guard_index: int,
        tier_config: _TierConfig,
        liveness: PidfdLiveness,
        control_bind: tuple[str, int],
        data_pool: tuple[int, int, Callable[..., bytes], str, int, int] | None = None,
    ) -> "tuple[KVCRPoolSpec, tuple[_PoolDescriptor, ...], int, _Lease]":
        """Give a pool group to a primary and return its Guard endpoint.

        The refusal check is a fast path only; the grant commits on the pool's
        actor under the same lock refuse_claims reads, so no grant follows it.
        """
        if self._refusing.is_set():
            raise KVCRServiceError("KVCR pool registry is closed")
        return self._guard(guard_index).claim(
            liveness, tier_config, control_bind, data_pool
        )

    def reattach(
        self, guard_index, tier_config, liveness, control_bind, data_pool, listener_fd
    ):
        if self._refusing.is_set():
            raise KVCRServiceError("KVCR pool registry is closed")
        return self._guard(guard_index).reattach(
            liveness, tier_config, control_bind, data_pool, listener_fd
        )

    def release(self, guard_index: int, lease: "_Lease") -> None:
        self._guard(guard_index).release(lease)

    def heartbeat(self, guard_index: int, lease: "_Lease") -> None:
        self._guard(guard_index).heartbeat(lease)

    def abort_grant(self, guard_index: int, lease: "_Lease") -> None:
        """Take back a grant its claimant declared it never served."""
        self._guard(guard_index).abort_grant(lease)

    def refuse_claims(self) -> None:
        """Stop granting pool groups without waiting for the close path to run.

        Each Guard's phase lock is the barrier: after this returns, no grant can
        commit. Pre-barrier grants may still deliver; those leases are fenced.
        """
        self._refusing.set()
        # Snapshot: close() deletes groups from the dict on other threads.
        for guard in list(self._guards.values()):
            with guard._phase_lock:
                pass

    def info(self) -> dict[str, Any]:
        healthy = all(
            guard._failure is None and guard._thread.is_alive()
            for guard in self._guards.values()
        )
        return {
            "version": _PROTOCOL_VERSION,
            "compatibility_digest": self._compatibility_digest,
            "state": "drained"
            if self._drained
            else (
                "draining"
                if self._draining.is_set()
                else ("error" if self._legacy_closing or not healthy else "ready")
            ),
            "ready": not self._refusing.is_set() and healthy,
            "guard_count": self._guard_count,
            "pool_specs": [
                {
                    "guard_index": index,
                    **{
                        name: getattr(guard._spec, name)
                        for name in (
                            "pool_id",
                            "path",
                            "generation",
                            "device",
                            "inode",
                            "mapping_bytes",
                            "journal_bytes",
                        )
                    },
                }
                for index, guard in sorted(self._guards.items())
            ],
        }

    def drain(self) -> dict[str, Any]:
        """Refuse claims, quiesce actors, then prove external attachments gone."""
        self._draining.set()
        self.refuse_claims()
        if not self._drain_lock.acquire(blocking=False):
            return {**self.info(), "state": "busy"}
        try:
            if self._drained:
                return self.info()
            if self._legacy_closing:
                raise KVCRServiceError("service shutdown is not a verified drain")
            guards = list(self._guards.values())
            for guard in guards:
                if not guard.begin_drain():
                    return {**self.info(), "state": "busy"}
            deadline = time.monotonic() + _REGISTRY_TRANSITION_TIMEOUT_SECONDS
            for guard in guards:
                if guard.finish_close(deadline):
                    return {**self.info(), "state": "busy"}
            for guard in guards:
                if not guard.drain_pool():
                    return {**self.info(), "state": "busy"}
            self._drained = True
            return self.info()
        except Exception as error:
            logger.warning("KVCR administrative drain retained its pools: %s", error)
            return {**self.info(), "state": "error", "error": str(error)}
        finally:
            self._drain_lock.release()

    def close(self) -> None:
        with self._drain_lock:
            if self._draining.is_set():
                result = self.drain()
                if result["state"] != "drained":
                    raise KVCRServiceError("KVCR holder still has undrained references")
                return
            self._legacy_closing = True
            self._close_legacy()

    def _close_legacy(self) -> None:
        """Give every pool group back, keeping the first reason one would not go.

        A group that will not close keeps only its own file and endpoint and
        stays listed, so a later close can try it again; failing that, the
        flock dies with the process and the next start reclaims.
        """
        self._refusing.set()
        failure: BaseException | None = None
        # Tell all Guards before waiting on any: a wedged one must not block the rest.
        for guard in self._guards.values():
            try:
                guard.begin_close()
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
        deadline = time.monotonic() + _REGISTRY_TRANSITION_TIMEOUT_SECONDS
        wedged: list[int] = []
        kept: set[int] = set()
        for guard_index in sorted(self._guards):
            try:
                if self._guards[guard_index].finish_close(deadline):
                    wedged.append(guard_index)
                    kept.add(guard_index)
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
                kept.add(guard_index)
        # Wedged groups stay visible; drained ones stay listed until the whole
        # drain finished, so a release racing shutdown is absorbed.
        # Reclaim only after all joins: unlink must not consume their deadline.
        for guard_index in sorted(self._guards):
            if self._guards[guard_index]._thread.is_alive():
                continue
            try:
                self._guards[guard_index].reclaim_pool()
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
                kept.add(guard_index)
        for guard_index in [index for index in self._guards if index not in kept]:
            del self._guards[guard_index]
        if failure is not None:
            raise failure
        if wedged:
            raise TimeoutError(
                f"timed out waiting for KVCR Guard transitions: guards {wedged} leaked"
            )

    def _guard_failed(
        self, guard_index: int, _guard: "_Guard", error: BaseException
    ) -> None:
        """A Guard has stopped being one, which the service cannot survive.

        TODO: no per-Guard containment. Its group can no longer be recovered and may
        still hold an endpoint the service cannot reach. One Guard takes the others'
        workers with it; add isolation back if that stops being acceptable.
        """
        logger.critical("KVCR Guard %d failed", guard_index)
        self.on_uncontained_failure(error)


def _live_layout_digest(
    slot: Any, claim_generation: int, backing_only: bool = False
) -> bytes:
    """The slot's layout now, so a Grow or Shrink after the claim changes it.

    ``backing_only`` drops the claim generation: it names the memory alone."""
    pool = slot.pool
    try:
        return region_table_digest(
            pool.regions, [pool.fd], slot.name, 0 if backing_only else claim_generation
        )
    except OSError:  # the pool is closed: nothing matches
        return b""


def _open_pool_fd(spec: KVCRPoolSpec) -> int:
    """A fresh open-file-description of the pool: flock belongs to it, so the
    claimant's lock stays independent of the Guard's, as with a path open."""
    pool_fd = os.open(spec.path, _OPEN_FLAGS)
    try:
        pool_stat = os.fstat(pool_fd)
        if (pool_stat.st_dev, pool_stat.st_ino) != (spec.device, spec.inode):
            raise KVCRServiceError("KVCR pool file does not match its grant")
    except BaseException:
        os.close(pool_fd)
        raise
    return pool_fd


_POD_UID = re.compile(
    r"(?:^|[/-])pod([0-9a-f]{8}(?:[-_][0-9a-f]{4}){3}[-_][0-9a-f]{12})(?:\.slice|/|$)"
)


def pod_uid_from_cgroup(text: str) -> str | None:
    """The pod UID in a /proc/<pid>/cgroup body, or None for a non-pod cgroup.

    Handles the systemd form ``kubepods-<qos>-pod<uid_with_underscores>.slice``
    and the cgroupfs form ``/kubepods/<qos>/pod<uid>``.
    """
    for line in text.splitlines():
        path = line.rsplit(":", 1)[-1]
        if "kubepods" in path and (match := _POD_UID.search(path)):
            return match.group(1).replace("_", "-")
    return None


def _peer_identity(connection: socket.socket) -> tuple[int, int, str | None]:
    """(pid, uid, pod_uid) of the peer: SO_PEERCRED at connect, then the pod UID
    from the pid's cgroup (needs hostPID; None when unreadable or not a pod)."""
    pid, uid, _gid = struct.unpack(
        "3i",
        connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        ),
    )
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as cgroup:
            pod_uid = pod_uid_from_cgroup(cgroup.read())
    except OSError:
        pod_uid = None
    return pid, uid, pod_uid


def _dup_liveness_fd(liveness: PidfdLiveness) -> int | None:
    """An independent pidfd for a queued waiter (the connection's closes at reply)."""
    try:
        return os.dup(liveness.fileno())
    except (OSError, ValueError):
        return None


def _pidfd_alive(liveness: PidfdLiveness) -> bool:
    try:
        return not select.select([liveness], [], [], 0)[0]
    except (OSError, ValueError):  # closed pidfd: the lease is gone
        return False


class _RequestHandler(socketserver.BaseRequestHandler):
    server: "_ThreadingUnixServer"
    request: socket.socket

    _slot_name: str | None = None
    _peer: tuple[int, int, str | None] | None = None
    _claim_generation: int = 0

    def setup(self) -> None:
        self.channel = FramedConnection(self.request)

    def _resize(self, request: "_Grow | _Shrink") -> "_Grown | _Shrunk | _Error":
        slots = self.server.slots
        if slots is None:
            return _reason_error(NOT_IMPLEMENTED, "resize is not implemented")
        try:
            peer = _peer_identity(self.request)
            if isinstance(request, _Grow):
                regions, generation = slots.grow(
                    request.slot_name,
                    peer,
                    request.additional_bytes,
                    request.split,
                    request.claim_generation,
                )
                return _Grown(_PROTOCOL_VERSION, regions, generation)
            removed, generation = slots.shrink(
                request.slot_name,
                peer,
                request.region_indices,
                request.claim_generation,
            )
            return _Shrunk(_PROTOCOL_VERSION, removed, generation)
        except SlotError as error:
            return _reason_error(error.reason, str(error))
        except Exception:  # noqa: BLE001 - report internal resize failures
            logger.exception("Unexpected failure while handling KVCR resize")
            return _reason_error(INTERNAL, "internal KVCR service error")

    def handle(self) -> None:
        try:
            request, worker_fds = self.channel.receive_with_fd(
                _CLAIM_DECODER, _claim_fd_counts
            )
            worker_pidfd = worker_fds[0] if worker_fds else None
            listener_fd = worker_fds[1] if len(worker_fds) > 1 else None
        except (EOFError, OSError):
            return
        except (KVCRGuardProtocolError, KVCRMsgFramingError) as error:
            self._send_error(error)
            return

        if isinstance(request, (_ServiceInfo, _Drain)):
            info = (
                self.server.registry.drain()
                if isinstance(request, _Drain)
                else self.server.registry.info()
            )
            with contextlib.suppress(OSError):
                self.channel.send(_ServiceState(info, _PROTOCOL_VERSION))
            return

        if isinstance(request, (_Grow, _Shrink)):
            with contextlib.suppress(OSError):
                self.channel.send(self._resize(request))
            return

        liveness: PidfdLiveness | None = None
        grant: "tuple[int, int, _Lease] | None" = None
        try:
            liveness = PidfdLiveness.from_peer_socket(self.request, worker_pidfd)
            if isinstance(request, _ClaimV2):  # v2 claim: slot + peer identity
                self._peer = _peer_identity(self.request)
                response, grant = self.server.dispatch(
                    request, liveness, self._peer, listener_fd=listener_fd
                )
            else:  # v1 claim: untouched path
                response, grant = self.server.dispatch(request, liveness)
        except SlotError as error:
            response = _reason_error(error.reason, str(error))
        except KVCRServiceReasonError as error:
            response = _reason_error(error.reason, error.message)
        except KVCRServiceError as error:
            response = _Error(str(error), _PROTOCOL_VERSION)
        except Exception:  # noqa: BLE001 - report internal claim failures
            logger.exception("Unexpected failure while handling KVCR claim")
            response = _Error("internal KVCR service error", _PROTOCOL_VERSION)

        if listener_fd is not None:
            os.close(listener_fd)
        if grant is None:
            if liveness is not None:
                liveness.close()
            with contextlib.suppress(OSError):
                self.channel.send(response)
            return

        guard_index, listener_fd, lease = grant
        if isinstance(response, _GrantedV2):
            self._slot_name = response.slot_name
            self._claim_generation = response.claim_generation
        try:
            pool_fd = _open_pool_fd(response.spec)
        except Exception:  # noqa: BLE001 - nothing sent yet: take the grant back
            logger.exception("KVCR could not open the pool for a claimant")
            os.close(listener_fd)
            self._release_or_fail(guard_index, lease, activated=False)
            self._vacate()
            self._send_error(KVCRServiceError("KVCR could not open the pool"))
            return
        try:
            # No timeout: a held connection must wait as long as the lease lives.
            # Inside the try: shutdown may have landed; listener_fd closes either way.
            self.request.settimeout(None)
            region_fds = (
                []
                if self._slot_name is None or self.server.slots is None
                else self.server.slots.pool_fds(self._slot_name)
            )
            self.channel.send_with_fd(response, pool_fd, listener_fd, *region_fds)
            if isinstance(response, _GrantedV2) and response.reattached:
                logger.info(
                    "event=worker_reattached worker_id=%s incarnation=%s "
                    "claim_generation=%s pool=%s slot=%s reattached=true",
                    response.guard_index,
                    request.incarnation,
                    response.claim_generation,
                    response.spec.pool_id,
                    response.slot_name,
                )
        # The lease is already granted and the send may have partially
        # crossed: releasing could double-grant a pool a claimant just mapped.
        # Hold it instead, under the same fencing as any lease: the claimant's
        # death frees it, and a claimant that never saw the grant says so with
        # an unactivated release. EOF alone frees nothing -- a claimant that
        # DID map the pool can drop its connection while still alive.
        except BaseException:
            logger.exception("KVCR grant delivery failed; holding the lease")
        finally:
            for file_descriptor in (pool_fd, listener_fd):
                with contextlib.suppress(OSError):
                    os.close(file_descriptor)
        self._await_release(guard_index, lease)

    def _vacate(self) -> None:
        """Free this connection's slot claim; a stale or foreign release is a no-op."""
        slots = getattr(self.server, "slots", None)
        if self._slot_name is None or slots is None:
            return
        try:
            slots.vacate(self._slot_name, self._peer, self._claim_generation)
        except SlotError:
            logger.warning(
                "KVCR slot %s vacate refused", self._slot_name, exc_info=True
            )

    def _await_release(self, guard_index: int, lease: "_Lease") -> None:
        """Keep the lease alive with heartbeats until its release.

        The Guard actor watches the pidfd, not this thread. EOF only ends the
        connection; the lease outlives it, and a death still promotes.
        """
        while True:
            try:
                message = self.channel.receive(_HOLD_DECODER)
            except (EOFError, OSError):
                return
            except (KVCRGuardProtocolError, KVCRMsgFramingError) as error:
                self._send_error(error)
                continue
            if isinstance(message, _Heartbeat):
                self.server.registry.heartbeat(guard_index, lease)
                continue
            if self._release_or_fail(guard_index, lease, message.activated):
                self._vacate()
                with contextlib.suppress(OSError):
                    self.channel.send(_Released(_PROTOCOL_VERSION))
            return

    def _release_or_fail(
        self, guard_index: int, lease: "_Lease", activated: bool = True
    ) -> bool:
        try:
            if activated:
                self.server.registry.release(guard_index, lease)
            else:
                # The claimant declared it never served this lease; the Guard
                # it stood down may resume serving.
                self.server.registry.abort_grant(guard_index, lease)
        except KVCRServiceError as error:
            # Registry closing: claimant learns the release did not commit; not fatal.
            self._send_error(error)
            return False
        except BaseException as error:  # noqa: BLE001 - post-grant failure
            self.server.fail(error)
            return False
        return True

    def finish(self) -> None:
        self.server.remove_connection(self.request)

    def _send_error(self, error: Exception) -> None:
        with contextlib.suppress(OSError):
            self.channel.send(_Error(str(error), _PROTOCOL_VERSION))


class _ThreadingUnixServer(
    socketserver.ThreadingMixIn,
    socketserver.UnixStreamServer,
):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        registry: _PoolRegistry,
        compatibility_digest: str,
        slots: "SlotTable | None" = None,
        advertise_host: str | None = None,
    ) -> None:
        self.registry = registry
        self.slots = slots
        self.advertise_host = advertise_host
        registry.on_uncontained_failure = self.fail
        self.compatibility_digest = compatibility_digest
        self._fatal_error: BaseException | None = None
        self._last_progress: float | None = None
        self._fatal_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        previous_umask = os.umask(_PRIVATE_SOCKET_UMASK)
        try:
            super().__init__(os.fspath(socket_path), _RequestHandler)
        finally:
            os.umask(previous_umask)

    def get_request(self) -> tuple[socket.socket, Any]:
        connection, address = super().get_request()
        connection.settimeout(_CLIENT_IDLE_TIMEOUT_SECONDS)
        self.add_connection(connection)
        if self.slots is not None:
            self.slots.start_quarantine()
        return connection, address

    def service_actions(self) -> None:
        self._last_progress = time.monotonic()
        if self.slots is not None and self.slots.agents is not None:
            with self.slots._lock:
                self.slots.agents.upkeep()

    def dispatch(
        self,
        request: "_Claim | _ClaimV2",
        liveness: PidfdLiveness,
        peer: tuple[int, int, str | None] | None = None,
        *,
        listener_fd: int | None = None,
    ) -> "tuple[_Granted | _GrantedV2, tuple[int, int, _Lease]]":
        if request.compatibility_digest != self.compatibility_digest:
            raise KVCRServiceError(
                "KVCR compatibility digest does not match the service"
            )
        liveness.incarnation = request.incarnation
        # v2 claims take a slot first; v1 claims skip it.
        slot = generation = None
        slots = getattr(self, "slots", None)  # absent on v1-only servers
        if isinstance(request, _ClaimV2):
            if slots is None:
                raise SlotError(NOT_IMPLEMENTED, "this service has no slot table")
            slot, generation = slots.claim_with_generation(
                request.tenant,
                request.lineage,
                request.guard_index,
                total=request.total_bytes,
                minimum=request.minimum_bytes,
                maximum=request.maximum_bytes,
                digest=request.compatibility_digest,
                claim_id=request.incarnation
                or f"{request.control_host}:{request.control_port}",
                peer=peer,
                owner_alive=lambda: _pidfd_alive(liveness),
                claim_generation=request.claim_generation,
                waiter_fd=lambda: _dup_liveness_fd(liveness),
                namespace=request.state_agent.namespace
                if request.state_agent
                else None,
            )
        elif slots is not None:  # a v1 claim must not take the Guard from a slot
            raise SlotError(NOT_IMPLEMENTED, "this service takes v2 claims only")
        try:
            # Retain the Guard across a concurrent close; claim validates the request.
            guard = self.registry._guards.get(request.guard_index)
            if slot is not None and slots.dead_incarnations:
                guard.fence_incarnations(slots.dead_incarnations)
            reattached = slot is not None and slot.reconnecting
            incoming_listener = listener_fd
            if reattached and incoming_listener is None:
                raise KVCRServiceError("reconnect requires the retained listener")
            grant_method = self.registry.reattach if reattached else self.registry.claim
            extra = {"listener_fd": incoming_listener} if reattached else {}
            spec, pools, listener_fd, lease = grant_method(
                request.guard_index,
                request.tier_config,
                liveness,
                (request.control_host, request.control_port),
                # The Guard serves this slot's regions, already mapped here.
                None
                if slot is None
                else (
                    slot.pool.address(slot.pool.regions[0]),
                    slot.pool.used_bytes,
                    lambda backing_only=False: _live_layout_digest(
                        slot, generation, backing_only
                    ),
                    slot.name,
                    len(slot.pool.regions),
                    slot.pool.meta_fd,
                ),
                **extra,
            )
            if guard is None:
                raise KVCRServiceError("KVCR claim succeeded without a Guard")
        except BaseException:
            if slot is not None:
                if slot.reconnecting:
                    slots._abort_claim(slot, False, generation)
                else:
                    slots.vacate(slot.name, peer, generation)
            raise
        common = (
            request.guard_index,
            spec,
            request.tier_config,
            pools,
            _PROTOCOL_VERSION,
        )
        if slot is None:
            granted: "_Granted | _GrantedV2" = _Granted(
                *common,
                dead_incarnations=guard.dead_incarnations,
                heartbeat_enabled=guard.heartbeat_enabled,
            )
        else:
            host = request.control_host
            if host == "0.0.0.0" and self.advertise_host:
                host = self.advertise_host
            granted = _GrantedV2(
                *common,
                reattached=reattached,
                lease_ttl_ms=slots.lease_ttl_ms,
                slot_name=slot.name,
                effective_maximum_bytes=slot.effective_maximum_bytes,
                regions=slot.pool.regions,
                layout_generation=slot.pool.layout_generation,
                claim_generation=generation,
                # The Guard binds the claimant-requested control address. Only a
                # wildcard bind is replaced by the configured reachable host; a
                # claimant that bound a specific address is advertised as bound.
                guard_endpoint=f"{host}:{request.control_port}",
                dead_incarnations=guard.dead_incarnations,
                heartbeat_enabled=guard.heartbeat_enabled,
            )
        return granted, (request.guard_index, listener_fd, lease)

    def fail(self, error: BaseException) -> None:
        """Stop the service, keeping only the first failure.

        Locked: Guard and request threads race here; the first explains the rest.
        """
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = error
        # Before shutdown, not as part of it: a handler already inside claim() would
        # otherwise be granted a pool by a service on its way out.
        self.registry.refuse_claims()
        self.shutdown()

    def add_connection(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.add(connection)

    def remove_connection(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def close_connections(self) -> None:
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                connection.close()


class _KVCRService:
    """Lifecycle wrapper used by the CLI and focused tests."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        pool_dir: str | os.PathLike[str],
        guard_count: int,
        pool_sizes_bytes: tuple[int, ...],
        compatibility_digest: str,
        journal_bytes: int = _DEFAULT_JOURNAL_BYTES,
        heartbeat_timeout_seconds: float | None = None,
        startup_timeout_seconds: int | None = None,
        slots: SlotTable | None = None,
        advertise_host: str | None = None,
    ) -> None:
        try:
            probe_pidfd = os.pidfd_open(os.getpid())
        except (AttributeError, OSError) as error:
            raise KVCRServiceError(
                "KVCR-Service requires pidfd_open (Linux 5.3+)"
            ) from error
        os.close(probe_pidfd)
        self.socket_path = Path(socket_path).resolve()
        if not self.socket_path.parent.is_dir():
            raise ValueError(
                f"KVCR socket directory does not exist: {self.socket_path.parent}"
            )
        # One daemon owns a socket path: the deployment runs a single service
        # per pod and clears the path before starting it.
        _unlink_stale_socket(self.socket_path)
        if slots is not None:
            # Before any pool or Guard thread exists, so a refusal leaks nothing.
            slots.bind_journal_bytes(journal_bytes)
        self._registry = _PoolRegistry(
            pool_dir,
            guard_count,
            pool_sizes_bytes,
            journal_bytes,
            compatibility_digest,
            heartbeat_timeout_seconds,
            startup_timeout_seconds,
            v2_slots=slots is not None,
        )
        try:
            self._server = _ThreadingUnixServer(
                self.socket_path,
                self._registry,
                compatibility_digest,
                slots,
                advertise_host,
            )
            socket_stat = self.socket_path.stat(follow_symlinks=False)
            self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
        except BaseException:
            self._release_partial_construction()
            raise
        self._closed = False

    def _release_partial_construction(self) -> None:
        """Release whatever __init__ managed to create before it failed."""
        server = getattr(self, "_server", None)
        if server is not None:
            with contextlib.suppress(OSError):
                server.server_close()
        with contextlib.suppress(BufferError, OSError):
            self._registry.close()

    def _health_status(self, timeout: float, *, ready: bool = True) -> tuple[int, str]:
        """Observe progress without claiming pools or entering Guard actor locks."""
        if self._closed:
            return 503, "service stopping"
        if self._server._fatal_error is not None:
            return 503, "service failed"
        now = time.monotonic()
        progress = self._server._last_progress
        if progress is not None and now - progress > timeout:
            return 503, "service stalled"
        if self._registry._draining.is_set():
            # Readiness refuses clients; liveness must not restart a holder
            # whose reservation still protects busy external references.
            return (503 if ready else 200), "service draining"
        if self._registry._refusing.is_set():
            return 503, "service stopping"
        guards = list(self._registry._guards.values())
        if len(guards) != self._registry._guard_count:
            return 503, "guards unavailable"
        for guard in guards:
            if guard._failure is not None or not guard._thread.is_alive():
                return 503, "guard failed"
            if ready and guard._reserved is _Phase.PROMOTING:
                return 503, "guard promoting"
            if ready and now - guard._last_progress > timeout:
                return 503, "guard stalled"
        if progress is None:
            return (503 if ready else 200), "service starting"
        return 200, "ok"

    def serve_forever(self) -> None:
        try:
            self._server.serve_forever()
        finally:
            self._server._last_progress = None
        if self._server._fatal_error is not None:
            raise self._server._fatal_error

    def shutdown(self) -> None:
        # Before the accept loop stops, for the same reason.
        self._registry.refuse_claims()
        self._server.shutdown()

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        try:
            self._server.close_connections()
        except OSError as error:
            first_error = error
        try:
            self._server.server_close()
        except OSError as error:
            if first_error is None:
                first_error = error
        try:
            self._registry.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        if self._server.slots is not None and self._server.slots.agents is not None:
            with self._server.slots._lock:
                self._server.slots.agents.close()
        try:
            _unlink_if_identity(self.socket_path, self._socket_identity)
        except OSError as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error
        self._closed = True


def _unlink_stale_socket(path: Path) -> None:
    """Remove a socket file left behind by a crashed daemon.

    A live daemon accepts the connection, so only a refused connect is stale.
    The unlink is identity-guarded: another daemon may have replaced the path
    between the probe and here, and that socket is live.
    """
    try:
        stat_result = path.lstat()
    except OSError:
        return
    if not stat.S_ISSOCK(stat_result.st_mode):
        return
    identity = (stat_result.st_dev, stat_result.st_ino)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(_STALE_SOCKET_PROBE_SECONDS)
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        logger.warning("Removing stale KVCR service socket: %s", path)
        _unlink_if_identity(path, identity)
        return
    except OSError as error:
        raise OSError(f"KVCR service socket is not usable: {path}: {error}") from error
    finally:
        probe.close()
    raise OSError(f"another KVCR service is listening on {path}")


class _ShutdownRequested(Exception):
    pass


def _handle_shutdown_signal(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise _ShutdownRequested


def _parse_pool_sizes_gb(value: str) -> tuple[int, ...]:
    try:
        sizes_gb = tuple(Decimal(item) for item in value.split(","))
        if not all(size.is_finite() and size > 0 for size in sizes_gb):
            raise InvalidOperation
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb must contain positive, finite numbers"
        ) from error
    if any(size > sys.maxsize for size in sizes_gb):
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb describes an allocation that is too large"
        )
    page = mmap.PAGESIZE
    with localcontext() as context:
        context.prec = max(len(size.as_tuple().digits) for size in sizes_gb) + 10
        sizes_bytes = tuple(int(size * _GB) // page * page for size in sizes_gb)
    if not all(size >= page for size in sizes_bytes):
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb values must be at least one memory page"
        )
    if _DEFAULT_JOURNAL_BYTES + sum(sizes_bytes) > sys.maxsize:
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb describes an allocation that is too large"
        )
    return sizes_bytes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the service's command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--guard-count", type=int, required=True)
    parser.add_argument(
        "--pool-sizes-gb",
        type=_parse_pool_sizes_gb,
        dest="pool_sizes_bytes",
        required=True,
    )
    parser.add_argument("--compatibility-digest", required=True)
    parser.add_argument("--health-host", default="127.0.0.1")
    parser.add_argument(
        "--health-port", type=int, help="enable /live and /ready HTTP probes"
    )
    parser.add_argument("--health-timeout-seconds", type=float, default=30)
    parser.add_argument(
        "--health-dependency-url",
        help="optional HTTP readiness dependency; /ready requires it to return 200",
    )
    parser.add_argument("--worker-lease-ttl-ms", type=int, default=10000)
    parser.add_argument(
        "--heartbeat-timeout-ms",
        type=float,
        help="opt in to early Guard promotion (minimum 1000 ms; default: pidfd-only)",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=int,
        default=int(_STARTUP_TIMEOUT_SECONDS),
        help="Guard core preparation and startup budget (default: %(default)s s)",
    )
    parser.add_argument(
        "--disable-hugepages",
        action="store_true",
        help="never place pools on hugetlbfs",
    )
    parser.add_argument(
        "--slot-cap-bytes",
        type=int,
        help="enable the v2 slot server with this node capacity (bytes)",
    )
    parser.add_argument("--holder-image", help="digest-pinned holder image")
    parser.add_argument("--namespace", default=os.environ.get("POD_NAMESPACE"))
    parser.add_argument("--node-name", default=os.environ.get("NODE_NAME"))
    parser.add_argument(
        "--advertise-host",
        default=os.environ.get("KVCR_ADVERTISE_HOST"),
        help="host in v2 grants' guard_endpoint (default: the claim's control host)",
    )
    parser.add_argument("--manager-id", default="gaas")
    parser.add_argument(
        "--state-agent-mode", choices=("dgd", "supervisor"), default="dgd"
    )
    parser.add_argument(
        "--slot-root",
        help="private holder socket/state dir (default: the socket's directory)",
    )
    parser.add_argument(
        "--require-pod-identity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="slots: pod UID from /proc/<pid>/cgroup (needs hostPID)",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=_LOG_LEVELS,
        default=os.environ.get("KVCR_LOG_LEVEL", "INFO"),
    )
    args = parser.parse_args(argv)
    if args.advertise_host is not None and (
        not args.advertise_host.strip()
        or args.advertise_host != args.advertise_host.strip()
        or any(c in args.advertise_host for c in ":/ \t")
        or args.advertise_host in ("0.0.0.0", "::")
    ):
        parser.error(
            "--advertise-host must be a bare reachable host name or address "
            "(no port, URL, whitespace or wildcard)"
        )
    if args.guard_count < 1:
        parser.error("--guard-count must be at least 1")
    if args.health_port is not None and not 1 <= args.health_port <= 65535:
        parser.error("--health-port must be between 1 and 65535")
    if not 1 < args.health_timeout_seconds < float("inf"):
        parser.error("--health-timeout-seconds must be finite and greater than 1")
    if args.health_dependency_url is not None:
        try:
            dependency = urlsplit(args.health_dependency_url)
            valid_dependency = (
                dependency.scheme in ("http", "https")
                and bool(dependency.hostname)
                and (dependency.port is None or dependency.port > 0)
            )
        except ValueError:
            valid_dependency = False
        if args.health_port is None or not valid_dependency:
            parser.error(
                "--health-dependency-url requires --health-port and a valid HTTP(S) URL"
            )
    if args.worker_lease_ttl_ms < 1000:
        parser.error("--worker-lease-ttl-ms must be at least 1000")
    if args.heartbeat_timeout_ms is not None and not (
        _MIN_HEARTBEAT_TIMEOUT_SECONDS * 1000
        <= args.heartbeat_timeout_ms
        < float("inf")
    ):
        parser.error("--heartbeat-timeout-ms must be at least 1000 and finite")
    if args.startup_timeout_seconds is not None and args.startup_timeout_seconds <= 0:
        parser.error("--startup-timeout-seconds must be positive")
    if args.slot_cap_bytes is not None:
        if args.slot_cap_bytes <= 0:
            parser.error("--slot-cap-bytes must be positive")
        if not (args.holder_image and args.namespace and args.node_name):
            parser.error("slots require --holder-image, --namespace and --node-name")
        if args.require_pod_identity and not _host_proc_visible():
            parser.error(
                "pod identity needs hostPID (/proc/1/cgroup is not the host's); "
                "set hostPID or pass --no-require-pod-identity"
            )
    return args


def _host_proc_visible() -> bool:
    """True when /proc shows host processes: pid 1 is outside our own cgroup."""
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as init:
            with open("/proc/self/cgroup", encoding="utf-8") as own:
                return init.read() != own.read()
    except OSError:
        return False


def _build_slots(args: argparse.Namespace) -> SlotTable | None:
    if args.slot_cap_bytes is None:
        return None
    broker = NodeBroker(
        namespace=args.namespace,
        node_name=args.node_name,
        manager_id=args.manager_id,
        image=args.holder_image,
        socket_root=args.slot_root or Path(args.socket_path).resolve().parent,
        api=InClusterPodAPI(args.namespace),
        toucher=True,
    )
    return SlotTable(
        broker,
        args.slot_cap_bytes,
        require_pod_identity=args.require_pod_identity,
        journal_bytes=_DEFAULT_JOURNAL_BYTES,
        lease_ttl_ms=args.worker_lease_ttl_ms,
        state_agent_mode=args.state_agent_mode,
        state_agent_max_slots=args.guard_count,
    )


def main() -> None:
    """Run the standalone KVCR service daemon."""
    args = _parse_args()
    logging.basicConfig(level=args.log_level)
    memmap.hugetlb_enabled = not args.disable_hugepages

    shutdown_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, shutdown_signals)
    old_sigint = signal.signal(signal.SIGINT, _handle_shutdown_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    server: _KVCRService | None = None
    stopping = False

    def health_status(ready: bool) -> tuple[int, str]:
        if stopping:
            return 503, "service stopping"
        if server is None:
            return (503 if ready else 200), "service starting"
        return server._health_status(args.health_timeout_seconds, ready=ready)

    try:
        # Bind before allocating pools. Health threads inherit the blocked signals;
        # the main thread handles shutdown once construction can be cleaned up.
        with _serve_health(
            (args.health_host, args.health_port)
            if args.health_port is not None
            else None,
            health_status,
            args.health_dependency_url,
        ):
            try:
                server = _KVCRService(
                    args.socket_path,
                    args.pool_dir,
                    guard_count=args.guard_count,
                    pool_sizes_bytes=args.pool_sizes_bytes,
                    compatibility_digest=args.compatibility_digest,
                    heartbeat_timeout_seconds=(
                        args.heartbeat_timeout_ms / 1000
                        if args.heartbeat_timeout_ms is not None
                        else None
                    ),
                    startup_timeout_seconds=args.startup_timeout_seconds,
                    slots=_build_slots(args),
                    advertise_host=args.advertise_host,
                )
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                logger.info(
                    "KVCR service ready: socket=%s guards=%d pool_sizes_bytes=%s "
                    "journal_bytes=%d advertise_host=%s",
                    args.socket_path,
                    args.guard_count,
                    args.pool_sizes_bytes,
                    _DEFAULT_JOURNAL_BYTES,
                    args.advertise_host,
                )
                server.serve_forever()
            except _ShutdownRequested:
                pass
            finally:
                stopping = True
                signal.pthread_sigmask(signal.SIG_BLOCK, shutdown_signals)
                if server is not None:
                    server.close()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


if __name__ == "__main__":
    main()
