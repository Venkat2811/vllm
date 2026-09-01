# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DBO builds list[dict] slot mappings; the drafter must normalize them."""

from vllm.v1.spec_decode.dflash import _normalize_slot_mappings


def test_plain_dict_passthrough():
    sm = {"layer.0": object()}
    assert _normalize_slot_mappings(sm) is sm


def test_none_passthrough():
    assert _normalize_slot_mappings(None) is None


def test_ubatched_list_collapses_to_first():
    first, second = {"layer.0": object()}, {"layer.0": object()}
    assert _normalize_slot_mappings([first, second]) is first


def test_empty_list_becomes_none():
    assert _normalize_slot_mappings([]) is None
