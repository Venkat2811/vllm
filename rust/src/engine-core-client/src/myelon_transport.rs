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

use std::time::Duration;

use bytes::Bytes;
use myelon_zmq::RecvSocket;

/// Per-frame SHM payload capacity. Must match `FRAME_BYTES_SHM` in
/// `crates/myelon-zmq-py/src/lib.rs:140`. Both ends use the same
/// compile-time const since the ring layout is fixed at bind time.
pub const FRAME_BYTES: usize = 2 * 1024 * 1024;

/// Ring depth. Matches the Python adapter's `_RING_SLOTS` default of 256.
/// 256 slots × 2 MiB = 512 MiB SHM segment (sparse-allocated; only
/// touched pages are resident).
pub const RING_SLOTS: usize = 256;

/// Errors raised by `MyelonOutputSocket`.
#[derive(Debug, thiserror::Error)]
pub enum MyelonOutputError {
    #[error("myelon attach: {0}")]
    Attach(String),
    #[error("myelon recv: {0}")]
    Recv(String),
}

/// Rust-frontend-side receiver for engine outputs over a myelon SHM
/// ring. Mirrors the `zeromq::PullSocket` API shape used by
/// `run_output_loop` so the call site can dispatch via an enum.
pub struct MyelonOutputSocket {
    inner: RecvSocket<FRAME_BYTES>,
    segment: String,
}

impl MyelonOutputSocket {
    /// Attach to a segment whose name is derived from the IPC
    /// endpoint string passed by the engine — must use the same
    /// derivation as the Python adapter's `_segment_from_endpoint`.
    pub fn connect(endpoint: &str) -> Result<Self, MyelonOutputError> {
        let segment = segment_from_endpoint(endpoint);
        // The Python OutputReceiver attaches eagerly in a daemon
        // thread for up to MYELON_HOT_PATH_ATTACH_TIMEOUT seconds.
        // The rust counterpart does the same retry — the engine may
        // not have bound its SHM ring yet when this is called.
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        loop {
            match RecvSocket::connect(&segment, RING_SLOTS, "vlm_0") {
                Ok(inner) => return Ok(Self { inner, segment }),
                Err(_) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(e) => return Err(MyelonOutputError::Attach(e.to_string())),
            }
        }
    }

    /// Receive one multipart message. Spins in a tight loop with
    /// `std::hint::spin_loop()`, yielding to the tokio scheduler
    /// between probes so other tokio tasks (the HTTP request
    /// handlers serving 256+ concurrent clients) can run. This
    /// preserves the same asyncio-fairness property that powers the
    /// Python adapter's wins, just translated into tokio.
    pub async fn recv(&mut self) -> Result<Vec<Bytes>, MyelonOutputError> {
        loop {
            let mut got: Option<Vec<Bytes>> = None;
            self.inner.try_recv_multipart_with(|res| {
                if let Ok(frames) = res {
                    let owned: Vec<Bytes> = frames
                        .iter()
                        .map(|s| Bytes::copy_from_slice(s))
                        .collect();
                    got = Some(owned);
                }
            });
            if let Some(msg) = got {
                return Ok(msg);
            }
            // CPU-cache-friendly pause hint before yielding.
            std::hint::spin_loop();
            tokio::task::yield_now().await;
        }
    }

    /// Segment name we're attached to (for logging/metrics).
    pub fn segment(&self) -> &str {
        &self.segment
    }
}

/// Mirror of the Python adapter's `_segment_from_endpoint`. Both
/// sides MUST produce the same string for the same input or the
/// attach fails — keep this in sync with the Python implementation
/// in `crates/myelon-zmq-py/myelon_pyzmq_shim/hot_path.py`.
pub fn segment_from_endpoint_for_test(endpoint: &str) -> String { segment_from_endpoint(endpoint) }

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
