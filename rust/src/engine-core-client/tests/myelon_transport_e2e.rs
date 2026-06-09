//! End-to-end integration tests for `myelon_transport::MyelonOutputSocket`.
//!
//! `RecvSocket<2 MiB>` is too big for the default thread stack (even
//! the OS-default 8 MiB integration-test stack overflows on some
//! systems), so each test spawns a dedicated thread with a 16 MiB
//! stack and runs a small tokio runtime on it.

#![cfg(feature = "myelon_hot_path")]

use std::time::Duration;

use myelon_zmq::SendSocket;
use vllm_engine_core_client::myelon_transport::{
    FRAME_BYTES, MyelonOutputSocket, RING_SLOTS, segment_from_endpoint_for_test,
};

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

/// Producer binds, consumer attaches, single multipart message
/// round-trips through the async `recv`. Closest we can get to a
/// DST-style integration test without spinning up a real engine.
#[test]
fn end_to_end_roundtrip_single_message() {
    run_with_big_stack(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_stack_size(16 * 1024 * 1024)
            .enable_all()
            .build()
            .expect("rt");

        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-dst-{}-{}",
                std::process::id(),
                line!(),
            );
            let segment = segment_from_endpoint_for_test(&endpoint);

            let mut send: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&segment, RING_SLOTS, "vlm_0").expect("producer bind");

            let endpoint_clone = endpoint.clone();
            let recv_handle = tokio::task::spawn_blocking(move || {
                MyelonOutputSocket::connect(&endpoint_clone).expect("receiver attach")
            });
            let mut recv = recv_handle.await.expect("attach join");

            send.warm_discovery(Duration::from_millis(200));

            let payload_a: &[u8] = b"alpha";
            let payload_b: &[u8] = &[0u8; 4096];
            send.send_multipart(&[payload_a, payload_b])
                .expect("send_multipart");

            let frames = tokio::time::timeout(Duration::from_secs(2), recv.recv())
                .await
                .expect("recv timed out — myelon attach/spin path broken")
                .expect("recv error");

            assert_eq!(frames.len(), 2, "expected 2 frames");
            assert_eq!(&frames[0][..], payload_a, "frame 0 mismatch");
            assert_eq!(frames[1].len(), 4096, "frame 1 length mismatch");
            assert!(
                frames[1].iter().all(|b| *b == 0),
                "frame 1 content mismatch"
            );
        });
    });
}

/// 16 sequential messages — exercises ring-slot rotation and the
/// spin-then-yield path under steady producer rate.
#[test]
fn end_to_end_sequential_burst_drains_in_order() {
    run_with_big_stack(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_stack_size(16 * 1024 * 1024)
            .enable_all()
            .build()
            .expect("rt");

        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-dst-burst-{}-{}",
                std::process::id(),
                line!(),
            );
            let segment = segment_from_endpoint_for_test(&endpoint);
            let mut send: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&segment, RING_SLOTS, "vlm_0").expect("producer bind");

            let endpoint_clone = endpoint.clone();
            let recv_handle = tokio::task::spawn_blocking(move || {
                MyelonOutputSocket::connect(&endpoint_clone).expect("receiver attach")
            });
            let mut recv = recv_handle.await.expect("attach join");

            send.warm_discovery(Duration::from_millis(200));

            for i in 0u32..16 {
                let tag = i.to_le_bytes();
                let payload = vec![i as u8; 256];
                send.send_multipart(&[&tag, &payload])
                    .expect("send_multipart");
            }

            for i in 0u32..16 {
                let frames = tokio::time::timeout(Duration::from_secs(2), recv.recv())
                    .await
                    .expect("recv timed out at i")
                    .expect("recv error at i");
                assert_eq!(frames.len(), 2, "msg {i}: expected 2 frames");
                let tag = u32::from_le_bytes(<[u8; 4]>::try_from(&frames[0][..]).unwrap());
                assert_eq!(tag, i, "msg {i}: out-of-order tag");
                assert!(
                    frames[1].iter().all(|b| *b == i as u8),
                    "msg {i}: payload content mismatch"
                );
            }
        });
    });
}

/// Connect retries: receiver attaches BEFORE the producer binds.
/// The eager retry in `MyelonOutputSocket::connect` should absorb the
/// race and complete the attach when the producer shows up.
#[test]
fn end_to_end_connect_retries_before_producer_binds() {
    run_with_big_stack(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_stack_size(16 * 1024 * 1024)
            .enable_all()
            .build()
            .expect("rt");

        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-dst-race-{}-{}",
                std::process::id(),
                line!(),
            );
            let segment = segment_from_endpoint_for_test(&endpoint);

            // Spawn the receiver FIRST — connect must retry.
            let endpoint_clone = endpoint.clone();
            let recv_handle = tokio::task::spawn_blocking(move || {
                MyelonOutputSocket::connect(&endpoint_clone).expect("receiver attach")
            });

            // Give the receiver time to start retrying.
            tokio::time::sleep(Duration::from_millis(150)).await;

            // Now bind the producer — the receiver's retry should pick it up.
            let mut send: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&segment, RING_SLOTS, "vlm_0").expect("producer bind");

            let mut recv = recv_handle.await.expect("attach join");
            send.warm_discovery(Duration::from_millis(200));

            let payload: &[u8] = b"after-retry";
            send.send_multipart(&[payload]).expect("send_multipart");

            let frames = tokio::time::timeout(Duration::from_secs(2), recv.recv())
                .await
                .expect("recv timed out after retry attach")
                .expect("recv error");
            assert_eq!(frames.len(), 1);
            assert_eq!(&frames[0][..], payload);
        });
    });
}
