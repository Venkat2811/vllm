#!/usr/bin/env python3
"""
Standalone micro-benchmark for shm_broadcast MessageQueue fast-path.

Runs entirely in one process (writer + reader) and provides lightweight stubs
for torch/torch.distributed if they are not installed, so no ML stack is needed.

Usage:
  PYTHONPATH=. python benchmarks/shm_broadcast_fastpath_bench_standalone.py \
      --payload 512 --iters 50000 --chunks 8

Fast-path toggle:
  VLLM_MQ_BINARY_FASTPATH=0  # force pickle path
  VLLM_MQ_BINARY_FASTPATH=1  # bytes fast-path (default when no remote readers)
"""

import argparse
import os
import importlib.util
import sys
import time
import mmap
import tempfile
import uuid
import pathlib
import collections


def _install_torch_stubs():
    """Install minimal torch / torch.distributed stubs for import."""

    class _StubTensor:
        def __init__(self, mv):
            self._mv = mv

        def fill_(self, val):
            if isinstance(self._mv, memoryview):
                self._mv[:] = bytes([val]) * len(self._mv)
            return self

    def frombuffer(buf, dtype=None):
        return _StubTensor(memoryview(buf))

    # torch.distributed stubs
    class _StubPG:
        pass

    class _StubDist:
        ProcessGroup = _StubPG

        def get_rank(self, pg=None):
            return 0

        def get_world_size(self, pg=None):
            return 1

        def get_process_group_ranks(self, pg=None):
            return [0]

    torch_stub = type("torch", (), {})()
    torch_stub.frombuffer = frombuffer
    torch_stub.uint8 = "uint8"
    torch_stub.distributed = _StubDist()

    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("torch.distributed", torch_stub.distributed)
    sys.modules.setdefault("torch.distributed.distributed_c10d", torch_stub.distributed)
    sys.modules.setdefault("torch.distributed.c10d", torch_stub.distributed)
    sys.modules.setdefault("torch.distributed._tools", torch_stub.distributed)


try:
    import torch  # type: ignore
except ImportError:
    _install_torch_stubs()

# File-backed SharedMemory shim (avoids posix_shm permission issues).
def _install_shared_memory_file_backed():
    from multiprocessing import shared_memory as sm_mod

    class FileBackedSharedMemory:
        def __init__(self, name=None, create=False, size=0):
            if create:
                if name is None:
                    name = f"fbshm-{uuid.uuid4().hex}"
                self._path = pathlib.Path(tempfile.gettempdir()) / name
                self._owner = True
                fd = os.open(self._path, os.O_CREAT | os.O_RDWR)
                os.ftruncate(fd, size)
                self._mmap = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
                os.close(fd)
            else:
                if name is None:
                    raise FileNotFoundError("name required when create=False")
                self._path = pathlib.Path(tempfile.gettempdir()) / name
                self._owner = False
                fd = os.open(self._path, os.O_RDWR)
                self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE)
                os.close(fd)
            self.buf = memoryview(self._mmap)
            self.size = len(self.buf)
            self.name = str(self._path.name)

        def close(self):
            try:
                self._mmap.close()
            except Exception:
                pass

        def unlink(self):
            if self._owner:
                try:
                    os.remove(self._path)
                except FileNotFoundError:
                    pass

    sm_mod.SharedMemory = FileBackedSharedMemory  # type: ignore

_install_shared_memory_file_backed()

# Minimal pyzmq stub to avoid real sockets/binds.
def _install_zmq_stub():
    channels = {}
    class _StubSocket:
        def __init__(self):
            self._addr = None
        def bind(self, addr): self._addr = addr; channels.setdefault(addr, collections.deque())
        def connect(self, addr): self._addr = addr; channels.setdefault(addr, collections.deque())
        def setsockopt(self, *args, **kwargs): return
        def setsockopt_string(self, *args, **kwargs): return
        def send(self, data): channels[self._addr].append(bytes(data))
        def send_multipart(self, parts, copy=False): channels[self._addr].append(b"".join([bytes(p) for p in parts]))
        def recv(self, copy=False): return channels[self._addr].popleft() if channels[self._addr] else b""
        def recv_multipart(self, *args, **kwargs): return [channels[self._addr].popleft()] if channels[self._addr] else [b""]
        def poll(self, timeout=None): return 1 if channels.get(self._addr) else 0
        def close(self, linger=0): channels.get(self._addr, collections.deque()).clear()
    class _StubContext:
        def socket(self, _type=None): return _StubSocket()
    stub = type("zmq", (), {})()
    stub.Context = _StubContext
    stub.Socket = _StubSocket
    stub.XPUB = 1
    stub.SUB = 2
    stub.XPUB_VERBOSE = 3
    stub.SUBSCRIBE = b""
    stub.IPV6 = 4
    sys.modules["zmq"] = stub
_install_zmq_stub()

# Minimal vllm stubs to satisfy shm_broadcast imports.
import types
import logging
import socket

envs_mod = types.ModuleType("vllm.envs")
envs_mod.VLLM_RINGBUFFER_WARNING_INTERVAL = 60
envs_mod.VLLM_SLEEP_WHEN_IDLE = False
sys.modules["vllm.envs"] = envs_mod

def _sched_yield():
    time.sleep(0)

dist_utils = types.ModuleType("vllm.distributed.utils")
class _StatelessProcessGroup:
    def __init__(self, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank
    def broadcast_obj(self, obj, src):
        return obj
dist_utils.StatelessProcessGroup = _StatelessProcessGroup
dist_utils.sched_yield = _sched_yield
sys.modules["vllm.distributed.utils"] = dist_utils

logger_mod = types.ModuleType("vllm.logger")
def _init_logger(name):
    return logging.getLogger(name)
logger_mod.init_logger = _init_logger
sys.modules["vllm.logger"] = logger_mod

platforms_mod = types.ModuleType("vllm.platforms")
def current_platform():
    return "cpu"
platforms_mod.current_platform = current_platform
sys.modules["vllm.platforms"] = platforms_mod

net_mod = types.ModuleType("vllm.utils.network_utils")
def get_ip():
    return "127.0.0.1"
def get_open_port():
    get_open_port._ctr = getattr(get_open_port, "_ctr", 40000) + 1
    return get_open_port._ctr
def get_open_zmq_ipc_path():
    return f"tcp://127.0.0.1:{get_open_port()}"
def is_valid_ipv6_address(addr):
    return ":" in addr
net_mod.get_ip = get_ip
net_mod.get_open_port = get_open_port
net_mod.get_open_zmq_ipc_path = get_open_zmq_ipc_path
net_mod.is_valid_ipv6_address = is_valid_ipv6_address
sys.modules["vllm.utils.network_utils"] = net_mod

# shim vllm package root to avoid __init__ side effects
vllm_root = types.ModuleType("vllm")
sys.modules["vllm"] = vllm_root

# Load shm_broadcast directly to avoid package-level deps.
_SHM_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "vllm",
        "distributed",
        "device_communicators",
        "shm_broadcast.py",
    )
)
_spec = importlib.util.spec_from_file_location("shm_broadcast_standalone", _SHM_PATH)
shm_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(shm_mod)  # type: ignore
MessageQueue = shm_mod.MessageQueue


def bench(payload_bytes: int, iters: int, max_chunks: int) -> None:
    payload = b"x" * payload_bytes

    mq = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=payload_bytes + 16,  # keep inline
        max_chunks=max_chunks,
    )
    handle = mq.export_handle()
    reader = MessageQueue.create_from_handle(handle, rank=0)
    reader._binary_fastpath = True

    # Skip READY handshakes; sockets are stubbed.
    mq.wait_until_ready = lambda: None
    reader.wait_until_ready = lambda: None

    # Warmup
    for _ in range(500):
        mq.enqueue(payload)
        reader.dequeue()

    start = time.perf_counter()
    for _ in range(iters):
        mq.enqueue(payload)
        reader.dequeue()
    elapsed = time.perf_counter() - start

    throughput = iters / elapsed
    avg_us = (elapsed / iters) * 1e6

    print(f"payload={payload_bytes}B iters={iters} chunks={max_chunks}")
    print(f"fastpath={os.environ.get('VLLM_MQ_BINARY_FASTPATH', '1')}")
    print(f"throughput={throughput:,.0f} msg/s  avg_lat={avg_us:.3f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=int, default=512, help="payload size in bytes")
    ap.add_argument("--iters", type=int, default=50000, help="iterations")
    ap.add_argument("--chunks", type=int, default=8, help="ring depth")
    args = ap.parse_args()
    bench(args.payload, args.iters, args.chunks)


if __name__ == "__main__":
    main()
