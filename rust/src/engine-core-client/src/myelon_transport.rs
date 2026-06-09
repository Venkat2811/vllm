//! Myelon SHM transport for the rust frontend's engine→frontend output path.
//!
//! Drop-in addition to vllm-rs at `rust/src/engine-core-client/src/`. Compiled
//! in alongside the default `zeromq::PullSocket` when the binary is built with
//! `--features myelon_hot_path`, selected at runtime via `USE_MYELON_HOT_PATH=1`.
//! Both ends of the channel must agree on the choice — when this is enabled,
//! the Python engine subprocess must also have `USE_MYELON_HOT_PATH=1` so it
//! binds its OutputSender to a myelon SHM segment instead of a zmq PUSH bind.
//!
//! # Why
//!
//! The Python frontend benchmark already showed myelon hot-path wins over
//! pyzmq on the dispatch-dominated regime (see
//! `crates/myelon-zmq-py/myelon_pyzmq_shim/HOT_PATH_TUNING.md`). The Python
//! win comes from freeing the asyncio loop from libzmq's internal
//! scheduling. The rust frontend uses `tokio` instead; the analogous
//! question is whether `tokio`'s task scheduler under the `zeromq` crate
//! has the same scheduling-fairness ceiling at high concurrency. If yes,
//! swapping in myelon here amplifies the win the same way it did for
//! asyncio.
//!
//! Architectural target (per `feedback_myelon_competitor_is_rust`):
//! rust+myelon must beat rust+pyzmq by ≥25% on the benchmark that
//! matters — likely the LMCache or Mooncake KV-cache offloading
//! scenarios where SHM bandwidth (≥10 GB/s) clearly beats pyzmq IPC
//! (~1-2 GB/s) at the protocol layer.

use std::collections::VecDeque;
use std::time::Duration;

use bytes::Bytes;
use myelon::transport::{
    FixedFrame, FramedTransportConsumer, MyelonWaitStrategy, ReassemblyBuffer,
};
use myelon_zmq::RecvSocket;

use crate::protocol::{EngineCoreOutputs, decode_msgpack};
use crate::transport::ENGINE_CORE_DEAD_SENTINEL;

/// Per-frame SHM payload capacity. Must match `FRAME_BYTES_SHM` in
/// `crates/myelon-zmq-py/src/lib.rs:140`. Both ends use the same
/// compile-time const since the ring layout is fixed at bind time.
pub const FRAME_BYTES: usize = 2 * 1024 * 1024;

/// Ring depth. Matches the Python adapter's `_RING_SLOTS` default of 256.
/// 256 slots × 2 MiB = 512 MiB SHM segment (sparse-allocated; only
/// touched pages are resident).
pub const RING_SLOTS: usize = 256;

/// Per-cycle drain cap. Audit #88: the original `recv()` did
/// single-message polling + `tokio::task::yield_now()` per message,
/// paying one tokio yield per RTT. The Python adapter already does
/// batched draining via `recv_multipart_drain_spin(timeout_us,
/// max_batch=64)` — that's why py+myelon hit the wins it did. Mirror
/// the pattern in Rust: drain up to `MAX_DRAIN_BATCH` messages per
/// cycle into a local VecDeque, then return them one at a time
/// without yielding between. Amortizes yield + spin-loop cost across
/// the burst, materially closing the gap with the substrate ceiling
/// in dispatch-bound workloads.
pub const MAX_DRAIN_BATCH: usize = 64;

/// Errors raised by `MyelonOutputSocket`.
#[derive(Debug, thiserror::Error)]
pub enum MyelonOutputError {
    #[error("myelon attach: {0}")]
    Attach(String),
    #[error("myelon recv: {0}")]
    Recv(String),
    #[error("myelon decode: {0}")]
    Decode(String),
}

/// Rust-frontend-side receiver for engine outputs over a myelon SHM
/// ring. Mirrors the `zeromq::PullSocket` API shape used by
/// `run_output_loop` so the call site can dispatch via an enum.
///
/// Audit #88: the receiver maintains a small in-memory drain buffer
/// so a single `recv()` returns one message at a time but only pays
/// the spin + tokio-yield tax once per `MAX_DRAIN_BATCH` messages
/// drained. Matches the Python adapter's batched drain pattern; the
/// gap previously made rust+myelon vs rust+pyzmq look like parity
/// when it may have been measuring an under-optimized receiver.
pub struct MyelonOutputSocket {
    inner: RecvSocket<FRAME_BYTES>,
    segment: String,
    /// Pre-drained messages awaiting per-call return. Spin/yield is
    /// paid only when this is empty; non-empty calls return
    /// synchronously from the front.
    pending: VecDeque<Vec<Bytes>>,
}

impl MyelonOutputSocket {
    /// Attach to a segment whose name is derived from the IPC
    /// endpoint string passed by the engine — must use the same
    /// derivation as the Python adapter's `_segment_from_endpoint`.
    pub fn connect(endpoint: &str) -> Result<Self, MyelonOutputError> {
        let segment = segment_from_endpoint(endpoint);
        Self::connect_to_segment(&segment)
    }

    /// Attach directly to a known segment name.
    ///
    /// This is mainly for local replay/tests where both ends of the
    /// transport are under our control and we do not need to mirror
    /// vLLM's `ipc://...` endpoint-derived naming.
    pub fn connect_to_segment(segment: &str) -> Result<Self, MyelonOutputError> {
        // The Python OutputReceiver attaches eagerly in a daemon
        // thread for up to MYELON_HOT_PATH_ATTACH_TIMEOUT seconds.
        // The rust counterpart does the same retry — the engine may
        // not have bound its SHM ring yet when this is called.
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        loop {
            match RecvSocket::connect(segment, RING_SLOTS, "vlm_0") {
                Ok(inner) => {
                    return Ok(Self {
                        inner,
                        segment: segment.to_string(),
                        pending: VecDeque::with_capacity(MAX_DRAIN_BATCH),
                    });
                }
                Err(_) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(e) => return Err(MyelonOutputError::Attach(e.to_string())),
            }
        }
    }

    /// Receive one multipart message.
    ///
    /// Audit #88: batched-drain pattern. If the local pending buffer
    /// has a message, return it synchronously (no spin, no yield).
    /// Otherwise spin until at least one message arrives, then drain
    /// up to `MAX_DRAIN_BATCH` more messages from the ring into
    /// `pending` BEFORE returning the first. Amortizes the
    /// `tokio::task::yield_now()` cost across the burst — the Python
    /// adapter has done this via `recv_multipart_drain_spin` since
    /// the original GIL-fairness fix; the Rust side was paying the
    /// per-message yield tax until now.
    pub async fn recv(&mut self) -> Result<Vec<Bytes>, MyelonOutputError> {
        // Fast path: pre-drained message ready.
        if let Some(msg) = self.pending.pop_front() {
            return Ok(msg);
        }
        // Slow path: spin until we drain at least one. Then opportunistically
        // drain more (up to MAX_DRAIN_BATCH-1 extras) so subsequent recv()
        // calls return from `pending` without spinning.
        loop {
            // First, try to grab one message from the ring.
            let mut got_one: Option<Vec<Bytes>> = None;
            self.inner.try_recv_multipart_with(|res| {
                if let Ok(frames) = res {
                    got_one = Some(
                        frames.iter().map(|s| Bytes::copy_from_slice(s)).collect(),
                    );
                }
            });
            if let Some(first) = got_one {
                // We have one — opportunistically drain more without
                // yielding between. Each try_recv_multipart_with is a
                // single non-blocking probe; we stop as soon as the
                // ring is empty.
                let mut extras_drained = 0;
                while extras_drained < MAX_DRAIN_BATCH - 1 {
                    let mut more: Option<Vec<Bytes>> = None;
                    self.inner.try_recv_multipart_with(|res| {
                        if let Ok(frames) = res {
                            more = Some(
                                frames.iter().map(|s| Bytes::copy_from_slice(s)).collect(),
                            );
                        }
                    });
                    match more {
                        Some(msg) => {
                            self.pending.push_back(msg);
                            extras_drained += 1;
                        }
                        None => break,
                    }
                }
                return Ok(first);
            }
            // No message yet; CPU-friendly pause + tokio yield so
            // other tasks (HTTP request handlers serving 256+
            // concurrent clients) can run.
            std::hint::spin_loop();
            tokio::task::yield_now().await;
        }
    }

    /// Segment name we're attached to (for logging/metrics).
    pub fn segment(&self) -> &str {
        &self.segment
    }
}

/// Narrow-path message returned from [`MyelonNarrowOutputSocket`].
///
/// The sender publishes the raw msgpack payload directly into the SHM
/// slot. That lets the receiver decode from borrowed slot bytes without
/// first rebuilding a multipart frame vector.
pub enum NarrowOutputMessage {
    Outputs(EngineCoreOutputs),
    EngineDead,
}

/// Rust-frontend-side receiver for the true narrow path.
///
/// Unlike [`MyelonOutputSocket`], this receiver does not unpack a
/// multipart envelope and does not copy the message bytes into
/// `Vec<Bytes>`. It decodes `EngineCoreOutputs` directly from the
/// leased SHM slot bytes and only allocates the resulting protocol
/// objects.
pub struct MyelonNarrowOutputSocket {
    inner: FramedTransportConsumer<FixedFrame<FRAME_BYTES>>,
    reassembly_buf: ReassemblyBuffer,
    segment: String,
    pending: VecDeque<NarrowOutputMessage>,
}

impl MyelonNarrowOutputSocket {
    pub fn connect(endpoint: &str) -> Result<Self, MyelonOutputError> {
        let segment = segment_from_endpoint(endpoint);
        Self::connect_to_segment(&segment)
    }

    /// Attach directly to a known segment name.
    ///
    /// Used by the replay/example path where we control both ends and
    /// want the substrate's portable name helper instead of vLLM's
    /// endpoint-derived name.
    pub fn connect_to_segment(segment: &str) -> Result<Self, MyelonOutputError> {
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        // Wait strategy gated by MYELON_NARROW_WAIT_STRATEGY env var:
        //   "block" (default) - park the thread, signal-based wake; fair to other tokio tasks
        //   "spin"            - tight CPU spin; lowest median, can hurt P99 tail under contention
        // The frontend profile showed receiver active work is <1us/iter, so the wait strategy
        // dominates tail behavior. Default Block aligns with the upstream myelon default.
        let wait_strategy = match std::env::var("MYELON_NARROW_WAIT_STRATEGY").as_deref() {
            Ok("spin") | Ok("busy") | Ok("busyspin") | Ok("BusySpin") => MyelonWaitStrategy::BusySpin,
            _ => MyelonWaitStrategy::Block,
        };
        loop {
            match FramedTransportConsumer::attach_with_consumer_id(
                segment,
                RING_SLOTS,
                "vlm_0",
                wait_strategy,
            ) {
                Ok(inner) => {
                    return Ok(Self {
                        inner,
                        reassembly_buf: ReassemblyBuffer::new(64),
                        segment: segment.to_string(),
                        pending: VecDeque::with_capacity(MAX_DRAIN_BATCH),
                    });
                }
                Err(_) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(e) => return Err(MyelonOutputError::Attach(e.to_string())),
            }
        }
    }

    pub async fn recv(&mut self) -> Result<NarrowOutputMessage, MyelonOutputError> {
        if let Some(msg) = self.pending.pop_front() {
            return Ok(msg);
        }
        loop {
            let mut got_one: Option<Result<NarrowOutputMessage, MyelonOutputError>> = None;
            self.inner
                .try_recv_message_leased(&mut self.reassembly_buf, |_kind, bytes| {
                    got_one = Some(decode_narrow_message(bytes));
                });
            if let Some(first) = got_one {
                let first = first?;
                let mut extras_drained = 0;
                while extras_drained < MAX_DRAIN_BATCH - 1 {
                    let mut more: Option<Result<NarrowOutputMessage, MyelonOutputError>> = None;
                    self.inner
                        .try_recv_message_leased(&mut self.reassembly_buf, |_kind, bytes| {
                            more = Some(decode_narrow_message(bytes));
                        });
                    match more {
                        Some(Ok(msg)) => {
                            self.pending.push_back(msg);
                            extras_drained += 1;
                        }
                        Some(Err(err)) => return Err(err),
                        None => break,
                    }
                }
                return Ok(first);
            }
            std::hint::spin_loop();
            tokio::task::yield_now().await;
        }
    }

    pub fn segment(&self) -> &str {
        &self.segment
    }
}

fn decode_narrow_message(bytes: &[u8]) -> Result<NarrowOutputMessage, MyelonOutputError> {
    if bytes == ENGINE_CORE_DEAD_SENTINEL {
        return Ok(NarrowOutputMessage::EngineDead);
    }
    let decoded = decode_msgpack(bytes).map_err(|e| MyelonOutputError::Decode(e.to_string()))?;
    Ok(NarrowOutputMessage::Outputs(decoded))
}

/// Mirror of the Python adapter's `_segment_from_endpoint`. Both
/// sides MUST produce the same string for the same input or the
/// attach fails — keep this in sync with the Python implementation
/// in `crates/myelon-zmq-py/myelon_pyzmq_shim/hot_path.py`.
fn segment_from_endpoint(endpoint: &str) -> String {
    if let Some(path) = endpoint.strip_prefix("ipc://") {
        let base = path.rsplit('/').next().unwrap_or(path);
        // macOS PSHMNAMLEN budget; mirror python's `base[-40:]`.
        if base.len() > 40 {
            base[base.len() - 40..].to_string()
        } else {
            base.to_string()
        }
    } else if endpoint.starts_with("tcp://") {
        // tcp:// makes no sense for SHM — fall back to a hashed name.
        // Cross-host fall-back to pyzmq happens at the gate in
        // transport.rs by NOT picking the myelon variant when the
        // endpoint isn't ipc://.
        format!(
            "tcp_{}",
            endpoint.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)) % 100_000_000
        )
    } else {
        let last_chars = endpoint.len().saturating_sub(40);
        endpoint[last_chars..].to_string()
    }
}

/// Test-visible wrapper around the private `segment_from_endpoint`
/// so the integration tests under `tests/` can reproduce the exact
/// segment derivation without making the function public. Kept as a
/// thin pass-through for now; if the test stops using it the wrapper
/// can be removed.
pub fn segment_from_endpoint_for_test(endpoint: &str) -> String {
    segment_from_endpoint(endpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use myelon_zmq::SendSocket;
    use std::time::Duration;

    /// Deterministic check: segment names produced from `ipc://` URIs
    /// match what the Python adapter's `_segment_from_endpoint` would
    /// produce. If this drifts, the Rust frontend will fail to attach
    /// to the Python engine's SHM segment.
    #[test]
    fn segment_from_endpoint_matches_python_logic() {
        assert_eq!(
            segment_from_endpoint("ipc:///tmp/abc-123-def"),
            "abc-123-def"
        );
        let long = "ipc:///tmp/very-long-uuid-1234567890-1234567890-1234567890-1234567890";
        let seg = segment_from_endpoint(long);
        assert!(seg.len() <= 40, "expected ≤40 chars, got {}", seg.len());
    }

    /// Verify the endpoint with no scheme falls back to a sane suffix.
    #[test]
    fn segment_from_endpoint_no_scheme_falls_back_to_suffix() {
        let s = segment_from_endpoint("just-some-name");
        assert_eq!(s, "just-some-name");
    }

    /// tcp:// endpoints aren't SHM-attachable — confirm the hash
    /// fall-back at least produces a deterministic string.
    #[test]
    fn segment_from_endpoint_tcp_is_deterministic_hash() {
        let s1 = segment_from_endpoint("tcp://127.0.0.1:5555");
        let s2 = segment_from_endpoint("tcp://127.0.0.1:5555");
        assert_eq!(s1, s2);
        assert!(s1.starts_with("tcp_"));
    }

    // End-to-end SHM round-trip and segment-accessor integration
    // checks live as tests/myelon_transport_e2e.rs at the crate root
    // — putting them in `mod tests {}` causes a stack overflow because
    // `RecvSocket<2 MiB>` is constructed on the test thread's stack,
    // and tokio's default worker stack is only ~512 KB. As integration
    // tests they run with the OS default 8 MiB stack.
}
