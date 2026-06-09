# SPDX-License-Identifier: Apache-2.0
"""Conditional zmq loader, gated by USE_MYELON_ZMQ env var.

Returns either pyzmq or the myelon SHM shim. ONLY intended for the
engine↔frontend ZMQ hot path. Other zmq users (e.g. shm_broadcast.py,
distributed coord) must keep their own `import zmq` direct so they
stay on pyzmq regardless of env var — myelon is intra-node only and
the broadcast path may cross hosts.

Usage:
    from vllm.utils.myelon_zmq_loader import zmq, zmq_asyncio
"""
import os
import sys

if os.environ.get("USE_MYELON_ZMQ"):
    # Path to the shim source on this host. Adjust the prepend if
    # the shim moves into vllm's wheel proper.
    _SHIM_PATH = "/home/venkat/vllm-rust-bench/myelon-src/crates/myelon-zmq-py"
    if _SHIM_PATH not in sys.path:
        sys.path.insert(0, _SHIM_PATH)
    import myelon_pyzmq_shim as zmq
    zmq_asyncio = zmq.asyncio
    sys.stderr.write(
        "[USE_MYELON_ZMQ=1] engine↔frontend zmq → myelon SHM shim\n"
    )
else:
    import zmq
    import zmq.asyncio as zmq_asyncio
