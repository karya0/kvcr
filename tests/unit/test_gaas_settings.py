# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GaaS settings on the guard config: v2 is selected iff tenant and lineage are set."""

import pytest

from kvcr.config import GaaSSettings, KVCRGuardConfig


def test_settings_absent_select_v1():
    assert KVCRGuardConfig("/s", 0, "d").gaas is None
    assert not GaaSSettings().is_v2


def test_tenant_and_lineage_select_v2():
    settings = GaaSSettings(
        "t",
        "l",
        total_bytes=8,
        minimum_bytes=4,
        maximum_bytes=16,
        state_agent={"namespace": "ns", "advertise": False},
        placement={"node": "n0"},
    )
    assert settings.is_v2
    assert KVCRGuardConfig("/s", 0, "d", gaas=settings).gaas is settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tenant": "t"},
        {"lineage": "l"},
        {"total_bytes": 8},
        {"state_agent": {"namespace": "ns"}},
    ],
)
def test_partial_settings_are_an_error(kwargs):
    with pytest.raises(ValueError, match="GaaS"):
        GaaSSettings(**kwargs)
