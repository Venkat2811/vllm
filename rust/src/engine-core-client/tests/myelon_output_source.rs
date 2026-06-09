//! Integration tests for the rust frontend's `OutputSource` enum
//! dispatch path (`engine-core-client/src/transport.rs`).
//!
//! Lives in `tests/` so it runs with the OS-default integration-test
//! stack (8 MiB on Linux). The myelon variant requires a larger stack
//! (`MyelonOutputSocket::FRAME_BYTES = 2 MiB`), so we wrap construction
//! in a 16 MiB-stack std::thread and pass the resulting `recv` end back
//! into a tokio runtime.

#![cfg(feature = "myelon_hot_path")]

use std::time::Duration;

use bytes::Bytes;
use myelon_zmq::SendSocket;
use vllm_engine_core_client::myelon_transport::{
    FRAME_BYTES, MyelonOutputSocket, RING_SLOTS, segment_from_endpoint_for_test,
};

// `OutputSource` is in a private module — exercising it via integration
// tests requires either re-exporting from lib.rs OR walking the public
// `myelon_transport` surface directly. We do the latter so this test
// stays out of the way of the crate's public-API contract: the
// `MyelonOutputSocket.recv()` shape is what `OutputSource::Myelon`
// dispatches to, so testing it directly exercises the same end-to-end
// pathway from the outside.

fn run_with_big_stack<F>(test_body: F)
where
    F: FnOnce() + Send + 'static,
{
    let thread = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(test_body)
        .expect("spawn thread");
    thread.join().expect("test thread panicked");
}

fn build_rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(16 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("rt")
}

/// Sanity: the `OutputSource::Myelon` variant's `recv_frames` returns
/// `Bytes` frames (the same shape the Zmq variant returns), and the
/// frames byte-match what was sent. This is the minimum guarantee
/// the rest of `run_output_loop` depends on.
#[test]
fn myelon_variant_recv_frames_matches_send() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-output-src-test-{}-{}",
                std::process::id(),
                line!(),
            );
            let segment = segment_from_endpoint_for_test(&endpoint);
            let mut send: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&segment, RING_SLOTS, "vlm_0").expect("producer bind");

            let endpoint_clone = endpoint.clone();
            let recv_handle = tokio::task::spawn_blocking(move || {
                MyelonOutputSocket::connect(&endpoint_clone).expect("attach")
            });
            let recv = recv_handle.await.expect("attach join");

            // Test `MyelonOutputSocket::recv` directly — this is the
            // exact call dispatched by `OutputSource::Myelon` in the
            // production frontend, just without the enum wrapper.
            let mut recv = recv;
            send.warm_discovery(Duration::from_millis(200));

            // Produce a 3-frame multipart message — the same shape vLLM's
            // engine emits (engine_id + msgpack_envelope + maybe tracker).
            send.send_multipart(&[b"engine_id_42", b"msgpack-bytes", b"trailer"])
                .expect("send_multipart");

            let frames = tokio::time::timeout(
                Duration::from_secs(2),
                recv.recv(),
            )
            .await
            .expect("recv timed out")
            .expect("recv err");

            assert_eq!(frames.len(), 3, "expected 3 frames");
            assert_eq!(&frames[0][..], b"engine_id_42");
            assert_eq!(&frames[1][..], b"msgpack-bytes");
            assert_eq!(&frames[2][..], b"trailer");
            // Verify they are `Bytes` (cheap-clone + offset-supporting).
            let _typed: &Vec<Bytes> = &frames;
        });
    });
}

/// Sequence + payload integrity over the OutputSource dispatch path —
/// 64 messages, varied payloads, every message must arrive in order.
/// Exercises the same code path `run_output_loop` calls under steady-
/// state vLLM bench traffic.
#[test]
fn myelon_output_source_64_msg_burst_in_order() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-output-src-burst-{}-{}",
                std::process::id(),
                line!(),
            );
            let segment = segment_from_endpoint_for_test(&endpoint);
            let mut send: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&segment, RING_SLOTS, "vlm_0").expect("bind");
            let endpoint_clone = endpoint.clone();
            let recv_handle = tokio::task::spawn_blocking(move || {
                MyelonOutputSocket::connect(&endpoint_clone).expect("attach")
            });
            let mut recv = recv_handle.await.expect("attach join");
            send.warm_discovery(Duration::from_millis(200));

            for i in 0u32..64 {
                let tag = i.to_le_bytes();
                let body = vec![(i & 0xFF) as u8; 1024];
                send.send_multipart(&[&tag, &body]).expect("send");
            }
            for i in 0u32..64 {
                let frames = tokio::time::timeout(
                    Duration::from_secs(2),
                    recv.recv(),
                )
                .await
                .expect("recv timed out")
                .expect("recv err");
                let got_tag =
                    u32::from_le_bytes(<[u8; 4]>::try_from(&frames[0][..]).unwrap());
                assert_eq!(got_tag, i, "out-of-order at i={i}");
                assert_eq!(frames[1].len(), 1024);
                assert_eq!(frames[1][0], (i & 0xFF) as u8);
            }
        });
    });
}

/// Error from Myelon variant becomes `Error::MyelonTransport` so the
/// caller can distinguish it from a Zmq error and react accordingly.
///
/// We can't easily *force* a Myelon recv error without invasive mocks,
/// but we can at least verify the type compiles + the error variant is
/// reachable through the recv_frames signature.
#[test]
fn myelon_output_source_error_variant_compiles() {
    use vllm_engine_core_client::Error;
    // Construct a dummy error to confirm the variant + its message
    // field round-trip cleanly.
    let e = Error::MyelonTransport {
        message: "test error reaches caller".to_string(),
    };
    let s = format!("{e:?}");
    assert!(
        s.contains("MyelonTransport") && s.contains("test error"),
        "MyelonTransport error should carry its message; got {s}",
    );
}
