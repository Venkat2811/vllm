# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Provider abstraction + per-op instrumentation harness for the SHM object
storage that backs MultiModalProcessorCache (multimodal kwargs cache).

This module is the *only* place where the choice of backend is made. Cache
call sites construct backends through `make_writer()` / `make_reader()`
factories and stay backend-agnostic.

Backends:

    VLLM_MM_CACHE_PROVIDER=vllm    (default)
        Returns the in-tree SingleWriterShmObjectStorage. Identical
        behaviour to before this module was added.

    VLLM_MM_CACHE_PROVIDER=myelon
        Returns a thin shim around `myelon_objstore.MyelonShmObjectStorage`
        (a Rust-backed PyO3 binding). The shim translates the vLLM
        SingleWriterShmObjectStorage API surface 1:1.
        If `myelon_objstore` is not importable, this factory fails fast
        with a clear error so a misconfigured deployment never silently
        falls back.

Per-op instrumentation:

    VLLM_MM_CACHE_INSTRUMENT=1
        Wraps put/get/touch with a wall-time measurement. Every operation
        emits a JSONL record with:
            ts, op, provider, payload_bytes, latency_ns, success, error

    VLLM_MM_CACHE_INSTRUMENT_FILE=/path/to/file.jsonl   (optional)
        Append destination. If unset, records go to stderr.

The instrumentation wrapper sits *outside* the chosen backend. Both backends
see the same per-op record shape so vllm-vs-myelon comparisons are
apples-to-apples.

Design note: this module only exposes factories, not the wrapper classes
themselves. Callers cannot construct a wrapper directly — they go through
`make_writer(...)` or `make_reader(...)`. This keeps the instrumentation
contract single-sourced.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing.synchronize import Lock as LockType
from typing import Any

from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde,
    ObjectSerde,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Env var helpers (kept local; not added to vllm/envs.py to keep this change
# self-contained until upstream review).
# ---------------------------------------------------------------------------

ENV_PROVIDER = "VLLM_MM_CACHE_PROVIDER"
ENV_INSTRUMENT = "VLLM_MM_CACHE_INSTRUMENT"
ENV_INSTRUMENT_FILE = "VLLM_MM_CACHE_INSTRUMENT_FILE"

PROVIDER_VLLM = "vllm"
PROVIDER_MYELON = "myelon"


def _read_provider() -> str:
    raw = os.environ.get(ENV_PROVIDER, PROVIDER_VLLM).strip().lower()
    if raw in (PROVIDER_VLLM, PROVIDER_MYELON):
        return raw
    raise ValueError(
        f"{ENV_PROVIDER}={raw!r} is not supported. "
        f"Allowed: {PROVIDER_VLLM!r}, {PROVIDER_MYELON!r}."
    )


def _instrument_enabled() -> bool:
    return os.environ.get(ENV_INSTRUMENT, "0").strip() in ("1", "true", "True", "yes")


# ---------------------------------------------------------------------------
# Per-op record sink. Lazily opens the file (or stderr) on first use, closes
# at process exit via atexit. Single sink per process.
# ---------------------------------------------------------------------------


class _InstrumentSink:
    _instance: "_InstrumentSink | None" = None

    def __init__(self) -> None:
        path = os.environ.get(ENV_INSTRUMENT_FILE)
        if path:
            try:
                self._fp = open(path, "a", buffering=1)  # line-buffered
                self._owns_fp = True
            except OSError as e:
                logger.warning(
                    "Could not open %s=%s for writing (%s); falling back to stderr",
                    ENV_INSTRUMENT_FILE,
                    path,
                    e,
                )
                self._fp = sys.stderr
                self._owns_fp = False
        else:
            self._fp = sys.stderr
            self._owns_fp = False

    @classmethod
    def get(cls) -> "_InstrumentSink":
        if cls._instance is None:
            cls._instance = cls()
            import atexit

            atexit.register(cls._instance.close)
        return cls._instance

    def emit(self, record: dict) -> None:
        try:
            self._fp.write(json.dumps(record, separators=(",", ":")))
            self._fp.write("\n")
        except (OSError, ValueError):
            # Never let instrumentation kill the cache hot path.
            pass

    def close(self) -> None:
        if self._owns_fp:
            try:
                self._fp.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Instrumentation wrapper. Forwards every method to the underlying store and
# emits a JSONL record per put/get/touch.
# ---------------------------------------------------------------------------


class _InstrumentedShmObjectStorage:
    """Wrap any SingleWriterShmObjectStorage-shaped object and time every
    put/get/touch. Forwards `key_index`, `clear`, `is_cached`, `get_cached`,
    `close`, `handle`, etc. unchanged."""

    def __init__(self, inner: Any, provider: str) -> None:
        self._inner = inner
        self._provider = provider
        self._sink = _InstrumentSink.get()

    # --------------- timed methods (cache hot path) ---------------------

    def put(self, key: str, value: Any) -> tuple[int, int]:
        start = time.perf_counter_ns()
        try:
            address, mid = self._inner.put(key, value)
        except Exception as e:
            self._sink.emit(
                {
                    "ts": time.time(),
                    "op": "put",
                    "provider": self._provider,
                    "payload_bytes": _payload_bytes(value),
                    "latency_ns": time.perf_counter_ns() - start,
                    "success": False,
                    "error": type(e).__name__ + ": " + str(e),
                }
            )
            raise
        self._sink.emit(
            {
                "ts": time.time(),
                "op": "put",
                "provider": self._provider,
                "payload_bytes": _payload_bytes(value),
                "latency_ns": time.perf_counter_ns() - start,
                "success": True,
                "address": address,
                "monotonic_id": mid,
            }
        )
        return address, mid

    def get(self, address: int, monotonic_id: int) -> Any:
        start = time.perf_counter_ns()
        try:
            obj = self._inner.get(address, monotonic_id)
        except Exception as e:
            self._sink.emit(
                {
                    "ts": time.time(),
                    "op": "get",
                    "provider": self._provider,
                    "address": address,
                    "monotonic_id": monotonic_id,
                    "latency_ns": time.perf_counter_ns() - start,
                    "success": False,
                    "error": type(e).__name__ + ": " + str(e),
                }
            )
            raise
        self._sink.emit(
            {
                "ts": time.time(),
                "op": "get",
                "provider": self._provider,
                "address": address,
                "monotonic_id": monotonic_id,
                "payload_bytes": _payload_bytes(obj),
                "latency_ns": time.perf_counter_ns() - start,
                "success": True,
            }
        )
        return obj

    def touch(self, key: str, address: int = 0, monotonic_id: int = 0) -> None:
        start = time.perf_counter_ns()
        try:
            self._inner.touch(key, address=address, monotonic_id=monotonic_id)
        except Exception as e:
            self._sink.emit(
                {
                    "ts": time.time(),
                    "op": "touch",
                    "provider": self._provider,
                    "latency_ns": time.perf_counter_ns() - start,
                    "success": False,
                    "error": type(e).__name__ + ": " + str(e),
                }
            )
            raise
        self._sink.emit(
            {
                "ts": time.time(),
                "op": "touch",
                "provider": self._provider,
                "latency_ns": time.perf_counter_ns() - start,
                "success": True,
            }
        )

    # --------------- pass-through (untimed) -----------------------------

    def __getattr__(self, name: str) -> Any:
        # All other attributes (key_index, clear, is_cached, get_cached,
        # close, handle, n_readers, max_object_size, ring_buffer, ...) go
        # straight to the underlying store. Critically this preserves
        # `_shm_cache.key_index` access from `remove_dangling_items()` in
        # `ShmObjectStoreSenderCache`.
        return getattr(self._inner, name)


def _payload_bytes(value: Any) -> int:
    """Best-effort byte count for an instrumentation record. We never want
    instrumentation itself to mutate the payload, so this is conservative."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__len__"):
        try:
            return int(len(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    return -1  # unknown


# ---------------------------------------------------------------------------
# Myelon backend — thin shim that imports the PyO3 binding lazily so that
# vLLM continues to import on a host where myelon_objstore is not installed.
# ---------------------------------------------------------------------------


def _import_myelon_or_raise():
    try:
        import myelon_objstore  # noqa: F401  (per-RFC 0035 binding)
    except ImportError as e:
        raise RuntimeError(
            f"{ENV_PROVIDER}={PROVIDER_MYELON} requested but the "
            "`myelon_objstore` PyO3 binding is not importable. "
            "Build it from `myelon-playground/crates/myelon-playground-py` "
            "(per RFC 0035 §5) or unset the env var to fall back to "
            f"the default {PROVIDER_VLLM!r} provider."
        ) from e
    return myelon_objstore


class _MyelonStoreShim:
    """API-compatible shim around myelon_objstore.MyelonShmObjectStorage.

    Translates the vLLM `SingleWriterShmObjectStorage` surface 1:1 so
    `MultiModalProcessorCache` cannot tell the difference. The Rust binding
    enforces its own internal invariants; this shim is purely an interface
    adapter.
    """

    def __init__(self, *, max_object_size: int, n_readers: int, ring_bytes: int,
                 name: str | None, serde_class: type[ObjectSerde],
                 reader_lock: LockType | None, is_writer: bool) -> None:
        myelon_objstore = _import_myelon_or_raise()
        self._serde = serde_class()
        self._reader_lock = reader_lock
        self._is_writer = is_writer
        # The PyO3 binding constructs differently for writer vs reader.
        if is_writer:
            self._inner = myelon_objstore.MyelonShmObjectStorage.create_writer(
                max_object_size=max_object_size,
                n_readers=n_readers,
                ring_bytes=ring_bytes,
                name=name,
            )
        else:
            # Reader path: the receiver-side init in cache.py constructs a
            # ring buffer with create=False and an existing name. We mirror
            # that by calling open_reader on the binding side. The vLLM
            # MultiModalProcessorCache passes the same name via env, so the
            # binding can attach to it.
            self._inner = myelon_objstore.MyelonShmObjectStorage.open_reader(
                name=name,
                ring_bytes=ring_bytes,
                n_readers=n_readers,
            )
        # vLLM accesses `.key_index` directly in remove_dangling_items().
        # Provide a writer-side dict that the shim updates on put.
        self.key_index: dict[str, tuple[int, int]] = {}
        self.max_object_size = max_object_size
        self.n_readers = n_readers

    # ------- API surface mirroring SingleWriterShmObjectStorage ---------

    def put(self, key: str, value: Any) -> tuple[int, int]:
        # vLLM's MsgpackSerde returns (data_or_list, data_bytes,
        # serialized_metadata, md_bytes). The Rust binding accepts a single
        # bytes payload, so we serialize here and pass the concatenated
        # payload. The binding does no further serialization.
        data, data_bytes, metadata, md_bytes = self._serde.serialize(value)
        if isinstance(data, list):
            payload = b"".join(data)
        else:
            payload = bytes(data)
        # On the wire we keep the vLLM layout: [metadata][data]. The Rust
        # binding stores `payload` opaquely; we reconstruct on the read
        # side using the inverse split below.
        wire = bytes(metadata) + payload
        address, monotonic_id = self._inner.put(key, wire)
        self.key_index[key] = (address, monotonic_id)
        return address, monotonic_id

    def get(self, address: int, monotonic_id: int) -> Any:
        wire = self._inner.get(address, monotonic_id)
        # Reverse the split done in put. The serde's deserialize takes a
        # memoryview of the full record (metadata + data) the same way
        # vLLM's MsgpackSerde does.
        return self._serde.deserialize(memoryview(wire))

    def is_cached(self, key: str) -> bool:
        return self._inner.is_cached(key)

    def get_cached(self, key: str) -> tuple[int, int]:
        return self._inner.get_cached(key)

    def touch(self, key: str, address: int = 0, monotonic_id: int = 0) -> None:
        self._inner.touch(key, address=address, monotonic_id=monotonic_id)

    def clear(self) -> None:
        self.key_index.clear()
        self._inner.clear()

    def close(self) -> None:
        self._inner.close()

    def handle(self):
        return self._inner.handle()


# ---------------------------------------------------------------------------
# Public factories.
# ---------------------------------------------------------------------------


def make_writer(
    *,
    data_buffer_size: int,
    name: str | None,
    max_object_size: int,
    n_readers: int,
    serde_class: type[ObjectSerde] = MsgpackSerde,
) -> Any:
    """Construct a writer-side cache backend. Returns an object that
    presents the SingleWriterShmObjectStorage API."""

    provider = _read_provider()
    if provider == PROVIDER_VLLM:
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=data_buffer_size,
            name=name,
            create=True,
        )
        store: Any = SingleWriterShmObjectStorage(
            max_object_size=max_object_size,
            n_readers=n_readers,
            ring_buffer=ring_buffer,
            serde_class=serde_class,
        )
    elif provider == PROVIDER_MYELON:
        store = _MyelonStoreShim(
            max_object_size=max_object_size,
            n_readers=n_readers,
            ring_bytes=data_buffer_size,
            name=name,
            serde_class=serde_class,
            reader_lock=None,
            is_writer=True,
        )
    else:  # unreachable; _read_provider validates
        raise AssertionError(provider)

    if _instrument_enabled():
        store = _InstrumentedShmObjectStorage(store, provider=provider)
    logger.info(
        "MM cache provider=%s instrumentation=%s (writer)",
        provider,
        _instrument_enabled(),
    )
    return store


def make_reader(
    *,
    data_buffer_size: int,
    name: str | None,
    max_object_size: int,
    n_readers: int,
    reader_lock: LockType,
    serde_class: type[ObjectSerde] = MsgpackSerde,
) -> Any:
    """Construct a reader-side cache backend. Returns an object that
    presents the SingleWriterShmObjectStorage API."""

    provider = _read_provider()
    if provider == PROVIDER_VLLM:
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=data_buffer_size,
            name=name,
            create=False,
        )
        store: Any = SingleWriterShmObjectStorage(
            max_object_size=max_object_size,
            n_readers=n_readers,
            ring_buffer=ring_buffer,
            serde_class=serde_class,
            reader_lock=reader_lock,
        )
    elif provider == PROVIDER_MYELON:
        store = _MyelonStoreShim(
            max_object_size=max_object_size,
            n_readers=n_readers,
            ring_bytes=data_buffer_size,
            name=name,
            serde_class=serde_class,
            reader_lock=reader_lock,
            is_writer=False,
        )
    else:  # unreachable
        raise AssertionError(provider)

    if _instrument_enabled():
        store = _InstrumentedShmObjectStorage(store, provider=provider)
    logger.info(
        "MM cache provider=%s instrumentation=%s (reader)",
        provider,
        _instrument_enabled(),
    )
    return store


def get_provider() -> str:
    """Public accessor — used by tests and harness scripts."""
    return _read_provider()
