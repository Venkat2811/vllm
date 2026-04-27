# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Provider-abstraction + instrumentation smoke tests.

These don't require GPU, model weights, or the Rust binding to pass — they
exercise the provider factory and the instrumentation wrapper in isolation.
The Myelon path is gated behind the optional `myelon_objstore` PyO3 module
and is skipped when that's not importable, so this file can run on any
host where vLLM itself imports.

Run:
    .venv/bin/python -m pytest tests/multimodal/test_shm_object_storage_provider.py -v
"""
from __future__ import annotations

import importlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from vllm.distributed.device_communicators import shm_object_storage_provider as p


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_default_provider_is_vllm(monkeypatch):
    monkeypatch.delenv(p.ENV_PROVIDER, raising=False)
    assert p.get_provider() == p.PROVIDER_VLLM


def test_provider_vllm_explicit(monkeypatch):
    monkeypatch.setenv(p.ENV_PROVIDER, "vllm")
    assert p.get_provider() == p.PROVIDER_VLLM


def test_provider_myelon_explicit(monkeypatch):
    monkeypatch.setenv(p.ENV_PROVIDER, "myelon")
    assert p.get_provider() == p.PROVIDER_MYELON


def test_provider_invalid_raises(monkeypatch):
    monkeypatch.setenv(p.ENV_PROVIDER, "nonsense")
    with pytest.raises(ValueError, match="not supported"):
        p.get_provider()


def test_myelon_without_binding_fails_loudly(monkeypatch):
    # Confirm the user gets a clear error when they ask for the Myelon
    # provider but the Rust binding isn't installed. We can only prove this
    # behaviour when myelon_objstore is genuinely not importable.
    if importlib.util.find_spec("myelon_objstore") is not None:
        pytest.skip("myelon_objstore is installed; cannot test failure mode")
    monkeypatch.setenv(p.ENV_PROVIDER, "myelon")
    with pytest.raises(RuntimeError, match="myelon_objstore"):
        p.make_writer(
            data_buffer_size=1 << 20,
            name="/objs-vllm-test",
            max_object_size=4096,
            n_readers=1,
        )


# ---------------------------------------------------------------------------
# Default vllm provider — round-trip put/get with instrumentation
# ---------------------------------------------------------------------------


def test_vllm_provider_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv(p.ENV_PROVIDER, "vllm")
    monkeypatch.delenv(p.ENV_INSTRUMENT, raising=False)
    monkeypatch.delenv(p.ENV_INSTRUMENT_FILE, raising=False)
    name = "/objs-vllm-rt"
    store = p.make_writer(
        data_buffer_size=1 << 20,
        name=name,
        max_object_size=8192,
        n_readers=1,
    )
    try:
        addr, mid = store.put("k1", b"hello")
        # vLLM's MsgpackSerde wraps bytes in pickle; round-trip via store.get.
        out = store.get(addr, mid)
        # Default MsgpackSerde returns bytes for bytes input.
        assert out == b"hello"
    finally:
        store.close()


def test_instrumentation_emits_jsonl(monkeypatch, tmp_path):
    """Every put + get should emit a JSONL line with the documented schema."""
    out_path = tmp_path / "instr.jsonl"
    monkeypatch.setenv(p.ENV_PROVIDER, "vllm")
    monkeypatch.setenv(p.ENV_INSTRUMENT, "1")
    monkeypatch.setenv(p.ENV_INSTRUMENT_FILE, str(out_path))
    # The sink is a singleton; resetting it between tests by reaching into
    # private state is a deliberate test-only escape hatch.
    p._InstrumentSink._instance = None  # type: ignore[attr-defined]
    name = "/objs-vllm-instr"
    store = p.make_writer(
        data_buffer_size=1 << 20,
        name=name,
        max_object_size=8192,
        n_readers=1,
    )
    try:
        addr, mid = store.put("alpha", b"x" * 256)
        out = store.get(addr, mid)
        assert out == b"x" * 256
        store.touch("alpha")
    finally:
        store.close()
    # Reset singleton so other tests see clean state.
    p._InstrumentSink._instance = None  # type: ignore[attr-defined]

    text = out_path.read_text().strip().splitlines()
    assert len(text) >= 3, f"expected ≥3 records (put,get,touch), got {len(text)}"
    by_op: dict[str, dict] = {}
    for line in text:
        rec = json.loads(line)
        for required in ("ts", "op", "provider", "latency_ns", "success"):
            assert required in rec, f"missing field {required} in {rec}"
        assert rec["provider"] == "vllm"
        assert rec["latency_ns"] >= 0
        by_op.setdefault(rec["op"], rec)
    assert {"put", "get", "touch"}.issubset(by_op.keys())
    assert by_op["put"]["payload_bytes"] == 256
    assert by_op["put"]["success"] is True
    assert by_op["get"]["success"] is True


def test_instrumentation_off_by_default(monkeypatch, tmp_path):
    """Without VLLM_MM_CACHE_INSTRUMENT, the sink is not opened and no
    JSONL file is created."""
    out_path = tmp_path / "instr_off.jsonl"
    monkeypatch.setenv(p.ENV_PROVIDER, "vllm")
    monkeypatch.delenv(p.ENV_INSTRUMENT, raising=False)
    monkeypatch.setenv(p.ENV_INSTRUMENT_FILE, str(out_path))
    p._InstrumentSink._instance = None  # type: ignore[attr-defined]
    name = "/objs-vllm-instr-off"
    store = p.make_writer(
        data_buffer_size=1 << 20,
        name=name,
        max_object_size=8192,
        n_readers=1,
    )
    try:
        addr, mid = store.put("alpha", b"x" * 64)
        store.get(addr, mid)
    finally:
        store.close()
    assert not out_path.exists(), "instrumentation should not have written when disabled"
