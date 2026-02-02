#!/usr/bin/env python3
"""
Minimal shared-memory broadcast microbench with no ML or ZMQ deps.

It compares:
  - pickle mode   (pickle dumps/loads per message)
  - binary mode   (raw bytes length-prefixed, no pickle)

Single process, one writer + one reader over a shared memory ring.
Uses only Python stdlib (multiprocessing.shared_memory).

Run:
  python benchmarks/simple_shm_broadcast_nopickle_bench.py --payload 512 --iters 50000
"""

import argparse
import pickle
import time
from multiprocessing import shared_memory
import mmap
import os
import tempfile
import uuid


class SimpleRing:
    """Single-writer, single-reader ring for fixed-size slots."""

    def __init__(self, slots: int, slot_bytes: int):
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.meta_bytes = slots  # 1 byte flag per slot (0 = free, 1 = full)
        self.data_bytes = slots * slot_bytes
        self.total_bytes = self.meta_bytes + self.data_bytes
        self._file_backed = False
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_bytes)
            self.raw = self.shm.buf
            mv = memoryview(self.raw)
            self.meta = mv[: self.meta_bytes]
            self.data = mv[self.meta_bytes :]
        except PermissionError:
            # Fallback: file-backed mmap
            self._file_backed = True
            self._path = tempfile.NamedTemporaryFile(prefix="shm_fallback_", delete=False)
            os.unlink(self._path.name)  # ensure cleanup by us
            fd = os.open(self._path.name, os.O_CREAT | os.O_RDWR)
            os.ftruncate(fd, self.total_bytes)
            self._mmap = mmap.mmap(fd, self.total_bytes, access=mmap.ACCESS_WRITE)
            os.close(fd)
            self.raw = self._mmap
            mv = memoryview(self.raw)
            self.meta = mv[: self.meta_bytes]
            self.data = mv[self.meta_bytes :]
        for i in range(self.meta_bytes):
            self.meta[i] = 0
        self.widx = 0
        self.ridx = 0

    def close(self):
        self.meta.release()
        self.data.release()
        if self._file_backed:
            try:
                self._mmap.close()
            except Exception:
                pass
            try:
                os.remove(self._path.name)
            except Exception:
                pass
        else:
            self.shm.close()
            self.shm.unlink()

    def acquire_write(self):
        while self.meta[self.widx] == 1:
            pass
        offset = self.widx * self.slot_bytes
        self.widx = (self.widx + 1) % self.slots
        return offset

    def release_write(self, slot):
        self.meta[slot // self.slot_bytes] = 1

    def acquire_read(self):
        while self.meta[self.ridx] == 0:
            pass
        offset = self.ridx * self.slot_bytes
        self.ridx = (self.ridx + 1) % self.slots
        return offset

    def release_read(self, slot):
        self.meta[slot // self.slot_bytes] = 0


def bench(payload_bytes: int, iters: int, use_pickle: bool):
    payload = b"x" * payload_bytes
    pickled_len = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    # Reserve enough for both modes: 4B length + max(payload, pickled_len)
    slot_bytes = 4 + max(payload_bytes, pickled_len)
    ring = SimpleRing(slots=8, slot_bytes=slot_bytes)

    # Warmup
    for _ in range(100):
        _write_one(ring, payload, use_pickle)
        _read_one(ring, use_pickle)

    start = time.perf_counter()
    for _ in range(iters):
        _write_one(ring, payload, use_pickle)
        _read_one(ring, use_pickle)
    elapsed = time.perf_counter() - start

    ring.close()

    throughput = iters / elapsed
    avg_us = (elapsed / iters) * 1e6
    mode = "pickle" if use_pickle else "binary"
    print(f"mode={mode} payload={payload_bytes}B iters={iters}")
    print(f"throughput={throughput:,.0f} msg/s  avg_lat={avg_us:.3f} us")


def _write_one(ring: SimpleRing, payload: bytes, use_pickle: bool):
    slot = ring.acquire_write()
    if use_pickle:
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        ln = len(blob)
        ring.raw[ring.meta_bytes + slot + 0 : ring.meta_bytes + slot + 4] = ln.to_bytes(
            4, "little", signed=False
        )
        ring.raw[ring.meta_bytes + slot + 4 : ring.meta_bytes + slot + 4 + ln] = blob
    else:
        ln = len(payload)
        ring.raw[ring.meta_bytes + slot + 0 : ring.meta_bytes + slot + 4] = ln.to_bytes(
            4, "little", signed=False
        )
        ring.raw[ring.meta_bytes + slot + 4 : ring.meta_bytes + slot + 4 + ln] = payload
    ring.release_write(slot)


def _read_one(ring: SimpleRing, use_pickle: bool):
    slot = ring.acquire_read()
    raw_view = memoryview(ring.raw)[
        ring.meta_bytes + slot : ring.meta_bytes + slot + ring.slot_bytes
    ].cast("B")
    if use_pickle:
        ln = int.from_bytes(raw_view[0:4], "little", signed=False)
        _ = pickle.loads(raw_view[4 : 4 + ln])
    else:
        ln = int.from_bytes(raw_view[0:4], "little", signed=False)
        _ = bytes(raw_view[4 : 4 + ln])
    ring.release_read(slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50000)
    args = ap.parse_args()
    bench(args.payload, args.iters, use_pickle=True)
    bench(args.payload, args.iters, use_pickle=False)


if __name__ == "__main__":
    main()
