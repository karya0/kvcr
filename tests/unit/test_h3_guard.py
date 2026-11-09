# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""H3 Guard reconnect ordering and promotion gates."""

import os
import socket
import tempfile
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import _recovered_record
from test_guard import (
    _PAGE_BLOCK_SIZE_BYTES,
    _PAGE_SPEC,
    _TEST_SPEC,
    _fake_attachment,
    _guard,
    _tier,
)

from kvcr.service import guard as guard_module
from kvcr.service.guard import _Phase
from kvcr.service.protocol import _G3Config
from kvcr.service.recovery_journal import _RecoveryMirror
from kvcr.types import BlockKey


def _recovery(monkeypatch, *, invalid=False, spec=_TEST_SPEC):
    guard = _guard(spec)
    recovery = guard._recovery
    recovery.attachment = attachment = _fake_attachment()
    digest = b"d" * 32
    attachment.layout_digest.return_value = digest
    recovery.layout_digest = lambda **_kw: digest
    recovery._attach_meta = Mock()
    published, consumed = Mock(), Mock()
    published.load_acquire.return_value = 17
    journal = Mock(_published=published, _consumed=consumed)
    journal.is_invalid.return_value = invalid
    journal.read_next.return_value = None
    journal.drain.return_value = iter(())
    recovery._journal = journal
    key = BlockKey(b"pre-restart")
    mirror = _RecoveryMirror(("",))
    mirror.adopt({key: _recovered_record(g2=[("", 0)])})

    def rebuild(*_args, has_g3):
        assert has_g3 is True
        consumed.store_release.assert_called_once_with(17)
        assert journal._consumed_local == 17
        return mirror

    read = Mock(side_effect=rebuild)
    monkeypatch.setattr(guard_module, "read_handback", read)
    return guard, recovery, journal, read, key


def test_q2_reattach_preserves_digest_and_orders_p0_before_rebuild(monkeypatch):
    guard, recovery, journal, read, key = _recovery(monkeypatch)
    recovery.reattach(_tier(16).pool_layouts)
    read.assert_called_once()
    journal.reset.assert_not_called()
    recovery.attachment.set_layout_digest.assert_not_called()
    assert key in recovery.mirror._records
    assert recovery.poll() is False
    journal.read_next.assert_called_once()


def test_q12_invalid_reattach_is_cold_without_rebuild(monkeypatch, caplog):
    _, recovery, journal, read, _ = _recovery(monkeypatch, invalid=True)
    recovery.reattach(_tier(16).pool_layouts)
    assert recovery.mirror is None
    read.assert_not_called()
    journal.reset.assert_not_called()
    assert "recovery disabled" in caplog.text


@pytest.mark.parametrize("invalid", [False, True])
def test_q10_q14_reattach_seats_lease_prepares_core_and_promotes(monkeypatch, invalid):
    guard, recovery, journal, read, key = _recovery(
        monkeypatch, invalid=invalid, spec=_PAGE_SPEC
    )
    # These cases exercise journal validity, which applies only to G3 tiers.
    tier = _tier(
        _PAGE_BLOCK_SIZE_BYTES,
        g3=_G3Config(("/g3",), _PAGE_BLOCK_SIZE_BYTES, "MOCK", {}),
    )
    recovery.set_meta_fd = Mock()
    guard._refuse_incompatible = Mock()
    guard._pool_lease.bind = Mock(side_effect=AssertionError("second bind"))
    guard._adopt = Mock(side_effect=AssertionError("fresh adoption"))
    monkeypatch.setattr(guard_module, "meta_spec", lambda *_args: None)
    core = Mock(nixl_agent_name="restarted-guard")
    # Exercise _prepare_core and _serve themselves, with the external transport
    # replaced by a core that records installation and startup.
    monkeypatch.setattr(guard_module, "_KVCRCore", Mock(return_value=core))
    monkeypatch.setattr(guard_module, "_attach_g2_index", Mock())
    lease = Mock(incarnation="worker-1")
    with (
        tempfile.TemporaryFile() as meta,
        socket.create_server(("127.0.0.1", 0)) as listener,
    ):
        meta.truncate(8192)
        recovery.attachment._file_descriptor = meta.fileno()
        recovery.attachment.snapshot_budget_bytes = 128
        bind = listener.getsockname()
        _, _, returned_fd, returned_lease = guard._reattach(
            lease,
            tier,
            bind,
            (1234, 32, recovery.layout_digest, "t/l/0", 1, 42),
            listener.fileno(),
        )
        try:
            assert returned_lease is lease
            assert guard._pool_lease.current is lease
            assert guard._phase is _Phase.PRIMARY
            assert guard._core is core
            assert guard._configured == tier
            assert guard._control is not None
            assert os.fstat(returned_fd).st_ino == os.fstat(listener.fileno()).st_ino
            guard._promote_for(lease)
            records = core.adopt_recovery_records.call_args.args[0]
            assert (key in records) is (not invalid)
            core.start.assert_called_once()
            assert guard._serving
            lease.close.assert_called_once()
            guard._adopt.assert_not_called()
            guard._pool_lease.bind.assert_not_called()
            journal.reset.assert_not_called()
            if invalid:
                read.assert_not_called()
        finally:
            os.close(returned_fd)
            guard._control.close()
            guard._pool_lease.unbind()
