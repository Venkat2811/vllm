# SPDX-License-Identifier: Apache-2.0
"""Tests for the narrow output sender adapter
(`vllm/utils/myelon_hot_path_loader.py:NarrowOutputSender`).

Mirrors the substantive Python-side coverage from
`myelon-zmq-py/tests/test_hot_path.py` for the narrow path.
End-to-end roundtrip tests are deferred to integration suites that
spawn the rust frontend receiver — the unit layer here exercises only
the Python wrapper logic using a mocked `NarrowSendSocket`.
"""
from unittest.mock import MagicMock, patch

import pytest

import vllm.utils.myelon_hot_path_loader as loader


# ---------- env-gating ----------


def test_is_narrow_enabled_respects_env(monkeypatch):
    monkeypatch.delenv("USE_MYELON_NARROW", raising=False)
    assert loader.is_narrow_enabled() is False
    monkeypatch.setenv("USE_MYELON_NARROW", "1")
    assert loader.is_narrow_enabled() is True
    monkeypatch.setenv("USE_MYELON_NARROW", "")
    assert loader.is_narrow_enabled() is False


def test_is_enabled_returns_true_when_narrow_only(monkeypatch):
    monkeypatch.delenv("USE_MYELON_HOT_PATH", raising=False)
    monkeypatch.setenv("USE_MYELON_NARROW", "1")
    assert loader.is_enabled() is True


# ---------- segment-name derivation (shared with multipart) ----------


def test_segment_from_endpoint_ipc_basic():
    name = loader._segment_from_endpoint("ipc:///tmp/sock_abc123.sock")
    assert name == "sock_abc123.sock"


def test_segment_from_endpoint_ipc_long_truncated_to_40():
    name = loader._segment_from_endpoint("ipc:///tmp/" + "x" * 80)
    assert len(name) == 40
    assert name.endswith("x" * 40)


def test_segment_from_endpoint_tcp_deterministic():
    a = loader._segment_from_endpoint("tcp://127.0.0.1:5555")
    b = loader._segment_from_endpoint("tcp://127.0.0.1:5555")
    assert a == b
    assert a.startswith("tcp_")


def test_segment_from_endpoint_tcp_differs_per_endpoint():
    a = loader._segment_from_endpoint("tcp://127.0.0.1:5555")
    b = loader._segment_from_endpoint("tcp://127.0.0.1:5556")
    assert a != b


# ---------- NarrowOutputSender construction ----------


def _make_sender(monkeypatch, segment="seg_test") -> loader.NarrowOutputSender:
    """Construct a NarrowOutputSender with the underlying NarrowSendSocket mocked.

    Returns the sender; tests can access the mock via `sender._send`.
    """
    fake_socket = MagicMock(name="NarrowSendSocket")
    fake_socket.warm_discovery.return_value = True
    monkeypatch.setattr(loader, "_ensure_narrow_importable", lambda: None)

    # Patch the dynamic `from myelon_narrow_py import NarrowSendSocket` so we
    # never actually load the PyO3 binding.
    import sys

    fake_module = MagicMock(name="myelon_narrow_py")
    fake_module.NarrowSendSocket.return_value = fake_socket
    monkeypatch.setitem(sys.modules, "myelon_narrow_py", fake_module)

    return loader.NarrowOutputSender(f"ipc:///tmp/{segment}.sock")


def test_sender_constructs_with_endpoint(monkeypatch):
    sender = _make_sender(monkeypatch)
    assert sender._endpoint == "ipc:///tmp/seg_test.sock"
    assert sender._segment == "seg_test.sock"
    assert sender._send is not None
    assert sender._consumer_discovered is False
    assert sender._discovery_retry_exhausted is False


# ---------- payload-type acceptance (the P2 fix surface) ----------


@pytest.mark.parametrize(
    "payload",
    [b"hello", bytearray(b"hello"), memoryview(b"hello")],
)
def test_sender_accepts_buffer_protocol_payloads(monkeypatch, payload):
    sender = _make_sender(monkeypatch)
    fake = sender._send
    # First send triggers warm_discovery
    sender.send(payload)
    fake.warm_discovery.assert_called_once()
    fake.send.assert_called_once_with(payload)


def test_sender_passes_payload_unchanged(monkeypatch):
    """Critical for the P2 fix: the bytes(payload) cast was removed.
    Verify the payload object identity reaches the Rust binding intact.
    """
    sender = _make_sender(monkeypatch)
    buf = bytearray(b"abc")
    sender.send(buf)
    # The send mock was called with the SAME bytearray object, not a copy.
    args, _ = sender._send.send.call_args
    assert args[0] is buf


# ---------- discovery state machine ----------


def test_sender_marks_consumer_discovered_on_success(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = True
    sender.send(b"x")
    assert sender._consumer_discovered is True
    assert sender._discovery_retry_exhausted is False


def test_sender_exhausts_retries_after_max_attempts(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = False
    for _ in range(loader._MAX_DISCOVERY_ATTEMPTS):
        sender.send(b"x")
    assert sender._consumer_discovered is False
    assert sender._discovery_retry_exhausted is True
    assert sender._discovery_attempts == loader._MAX_DISCOVERY_ATTEMPTS


def test_sender_skips_discovery_after_first_success(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = True
    sender.send(b"first")
    sender._send.warm_discovery.reset_mock()
    sender.send(b"second")
    sender.send(b"third")
    sender._send.warm_discovery.assert_not_called()


def test_sender_skips_discovery_after_retry_exhausted(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = False
    for _ in range(loader._MAX_DISCOVERY_ATTEMPTS):
        sender.send(b"x")
    sender._send.warm_discovery.reset_mock()
    sender.send(b"after-exhaustion")
    sender._send.warm_discovery.assert_not_called()


# ---------- close / lifecycle ----------


def test_close_marks_send_none(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender.close()
    assert sender._send is None


def test_close_is_idempotent(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender.close()
    # Calling close again must not raise even though _send is already None.
    sender.close()
    assert sender._send is None


def test_send_after_close_raises(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender.close()
    with pytest.raises(RuntimeError, match="myelon narrow: send called after close"):
        sender.send(b"x")


def test_close_resets_discovery_state(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = True
    sender.send(b"x")
    assert sender._consumer_discovered is True
    sender.close()
    assert sender._consumer_discovered is False
    assert sender._discovery_retry_exhausted is False


# ---------- pyzmq-compat surface ----------


def test_getsockopt_identity_returns_endpoint_bytes(monkeypatch):
    sender = _make_sender(monkeypatch)
    # zmq.IDENTITY is 32 — should return the endpoint as bytes.
    assert sender.getsockopt(32) == b"ipc:///tmp/seg_test.sock"


def test_getsockopt_unknown_returns_zero(monkeypatch):
    sender = _make_sender(monkeypatch)
    assert sender.getsockopt(99) == 0


def test_setsockopt_is_no_op(monkeypatch):
    sender = _make_sender(monkeypatch)
    # Must not raise.
    sender.setsockopt(123, b"whatever")


def test_send_ignores_copy_and_track_kwargs(monkeypatch):
    """The pyzmq drop-in surface accepts copy/track; narrow must too."""
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = True
    # Must not raise; must not pass kwargs to underlying Rust send.
    sender.send(b"x", copy=False, track=True, extra_unused=42)
    args, kwargs = sender._send.send.call_args
    assert args == (b"x",)
    assert kwargs == {}


def test_send_returns_tracker_when_track_true(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = True
    tracker = sender.send(b"x", track=True)
    assert tracker is not None
    assert tracker.done is True


def test_send_returns_none_when_track_false(monkeypatch):
    sender = _make_sender(monkeypatch)
    sender._send.warm_discovery.return_value = True
    assert sender.send(b"x") is None
    assert sender.send(b"x", track=False) is None


# ---------- make_output_sender dispatch ----------


def test_make_output_sender_returns_narrow_when_enabled(monkeypatch):
    monkeypatch.setenv("USE_MYELON_NARROW", "1")
    monkeypatch.setattr(loader, "_ensure_narrow_importable", lambda: None)
    import sys

    fake_socket = MagicMock()
    fake_socket.warm_discovery.return_value = True
    fake_module = MagicMock()
    fake_module.NarrowSendSocket.return_value = fake_socket
    monkeypatch.setitem(sys.modules, "myelon_narrow_py", fake_module)

    sender = loader.make_output_sender("ipc:///tmp/x.sock")
    assert isinstance(sender, loader.NarrowOutputSender)


def test_make_output_receiver_raises_for_narrow(monkeypatch):
    monkeypatch.setenv("USE_MYELON_NARROW", "1")
    with pytest.raises(RuntimeError, match=r"(?i)rust frontend"):
        loader.make_output_receiver("ipc:///tmp/x.sock")
