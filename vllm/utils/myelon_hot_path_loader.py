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


def is_enabled() -> bool:
    return bool(os.environ.get("USE_MYELON_HOT_PATH"))


def _ensure_shim_importable() -> None:
    """Add the myelon-zmq-py source path to sys.path when running
    against a checked-out repo rather than an installed wheel.

    Set MYELON_HOT_PATH_SRC to point at the shim directory if it
    isn't on the default Python path.
    """
    src = os.environ.get("MYELON_HOT_PATH_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)


def make_output_sender(endpoint: str):
    """Engine side: PUSH-equivalent SHM ring producer."""
    _ensure_shim_importable()
    from myelon_pyzmq_shim.hot_path import OutputSender
    return OutputSender(endpoint)


def make_output_receiver(endpoint: str):
    """Frontend side: PULL-equivalent SHM ring consumer."""
    _ensure_shim_importable()
    from myelon_pyzmq_shim.hot_path import OutputReceiver
    return OutputReceiver(endpoint)
