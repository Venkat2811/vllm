# SPDX-License-Identifier: Apache-2.0
"""Loader for myelon SHM hot-path adapter.

Gated by `USE_MYELON_HOT_PATH=1`. When enabled, vLLM swaps the
engine→frontend output socket (and only that socket) from pyzmq
PUSH/PULL to a pure-myelon SHM ring. All other zmq users —
handshake, coordinator, input_socket (ROUTER/DEALER), stats,
distributed-TP, KV-transfer — stay on pyzmq unchanged.

Architecture rationale:
- output_socket is the hottest path (per-token streaming, 256+
  concurrent requests). Pure unidirectional PUSH/PULL maps cleanly
  to a SHM ring with no semantic emulation needed.
- input_socket is bidirectional (engine DEALER sends a ready msg
  at startup then receives requests; frontend ROUTER routes by
  identity). Bidirectional emulation on SHM is the trap the
  earlier USE_MYELON_ZMQ drop-in fell into. Keep it on pyzmq.
- Handshake / coord / stats are one-shot or low-frequency — pyzmq's
  per-call cost is irrelevant there.

Both ends of every myelon ring created via this loader are pure-myelon;
zmq-client interop is explicitly NOT a goal.
"""
import os
import sys
from pathlib import Path


def is_narrow_enabled() -> bool:
    return bool(os.environ.get("USE_MYELON_NARROW"))


def is_enabled() -> bool:
    return bool(os.environ.get("USE_MYELON_HOT_PATH")) or is_narrow_enabled()


def _ensure_shim_importable() -> None:
    """Add the myelon-zmq-py source path to sys.path when running
    against a checked-out repo rather than an installed wheel.

    Set MYELON_HOT_PATH_SRC to point at the shim directory if it
    isn't on the default Python path.
    """
    candidates = []
    src = os.environ.get("MYELON_HOT_PATH_SRC")
    if src:
        candidates.append(Path(src))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root.parent / "myelon-src" / "crates" / "myelon-zmq-py",
        repo_root.parent / "myelon-playground" / "crates" / "myelon-zmq-py",
    ])
    for candidate in candidates:
        candidate = candidate.resolve()
        if not (candidate / "myelon_pyzmq_shim" / "__init__.py").exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return


def _ensure_narrow_importable() -> None:
    candidates = []
    src = os.environ.get("MYELON_NARROW_SRC")
    if src:
        candidates.append(Path(src))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "tools" / "myelon_narrow_py")
    for candidate in candidates:
        candidate = candidate.resolve()
        if not (candidate / "pyproject.toml").exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return


_RING_SLOTS = int(os.environ.get("MYELON_HOT_PATH_RING_SLOTS", "256"))
_DISCOVERY_MS = int(os.environ.get("MYELON_HOT_PATH_DISCOVERY_MS", "200"))
_MAX_DISCOVERY_ATTEMPTS = int(
    os.environ.get("MYELON_HOT_PATH_DISCOVERY_RETRIES", "3")
)


def _segment_from_endpoint(endpoint: str) -> str:
    if endpoint.startswith("ipc://"):
        path = endpoint[6:]
        base = path.rsplit("/", 1)[-1]
        return base[-40:] if len(base) > 40 else base
    if endpoint.startswith("tcp://"):
        mask = 0xFFFFFFFFFFFFFFFF
        acc = 0
        for b in endpoint.encode("utf-8"):
            acc = ((acc * 31) + b) & mask
        return f"tcp_{acc % (10**8)}"
    return endpoint[-40:]


class _AlwaysDoneTracker:
    done = True

    def wait(self, timeout=None):
        return True


class NarrowOutputSender:
    """Raw-payload myelon sender used only with the Rust frontend.

    This bypasses the multipart envelope entirely. It assumes the
    engine output is a single msgpack buffer. If the workload starts
    producing aux buffers, the caller must fall back to the multipart
    path instead of trying to mix wire formats on one ring.
    """

    def __init__(self, endpoint: str, ring_slots: int = _RING_SLOTS):
        _ensure_narrow_importable()
        from myelon_narrow_py import NarrowSendSocket

        self._endpoint = endpoint
        self._segment = _segment_from_endpoint(endpoint)
        self._send = NarrowSendSocket(self._segment, ring_slots, "vlm_0")
        self._consumer_discovered = False
        self._discovery_retry_exhausted = False
        self._discovery_attempts = 0

    def send(self, payload, copy: bool = False, track: bool = False, **kwargs):
        if self._send is None:
            raise RuntimeError(
                "myelon narrow: send called after close() on NarrowOutputSender"
            )
        # NarrowSendSocket.send now accepts any C-contiguous u8 buffer-protocol
        # object (bytes, bytearray, memoryview). Skip the bytes(payload) cast
        # so encode_into()'s reused bytearray reaches Rust without realloc.
        if (
            not self._consumer_discovered
            and not self._discovery_retry_exhausted
        ):
            discovered_now = self._send.warm_discovery(_DISCOVERY_MS)
            if discovered_now:
                self._consumer_discovered = True
            else:
                self._discovery_attempts += 1
                if self._discovery_attempts >= _MAX_DISCOVERY_ATTEMPTS:
                    self._discovery_retry_exhausted = True
                    sys.stderr.write(
                        "[myelon narrow] WARN: consumer discovery timed out "
                        f"after {self._discovery_attempts} attempts on "
                        f"segment={self._segment}; subsequent sends will "
                        "proceed fail-closed if the ring fills.\n"
                    )
                    sys.stderr.flush()
        self._send.send(payload)
        return _AlwaysDoneTracker() if track else None

    def getsockopt(self, opt: int):
        if opt == 32:
            return self._endpoint.encode()
        return 0

    def setsockopt(self, opt: int, value):
        pass

    def close(self, *args, **kwargs):
        if self._send is not None:
            self._send.close()
        self._send = None
        self._consumer_discovered = False
        self._discovery_retry_exhausted = False


def make_output_sender(endpoint: str):
    """Engine side: PUSH-equivalent SHM ring producer."""
    if is_narrow_enabled():
        return NarrowOutputSender(endpoint)
    _ensure_shim_importable()
    from myelon_pyzmq_shim.hot_path import OutputSender
    return OutputSender(endpoint)


def make_output_receiver(endpoint: str):
    """Frontend side: PULL-equivalent SHM ring consumer."""
    if is_narrow_enabled():
        raise RuntimeError(
            "USE_MYELON_NARROW is only supported with the Rust frontend. "
            "The Python frontend still expects the multipart receiver."
        )
    _ensure_shim_importable()
    from myelon_pyzmq_shim.hot_path import OutputReceiver
    return OutputReceiver(endpoint)
