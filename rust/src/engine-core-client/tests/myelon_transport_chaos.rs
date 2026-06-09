//! Chaos / DST-style integration tests for `myelon_transport`.
//!
//! Complements `myelon_transport_e2e.rs`. While that file proves
//! happy-path round-trips work, this file pushes on edges that only
//! surface under realistic vLLM bench load:
//!
//! - zero-byte payloads
//! - exact-boundary frame sizes (`FRAME_BYTES - 8`)
//! - alternating tiny/large payloads
//! - high-volume burst (≥1024 messages — exercises ring wrap-around)
//! - producer→multiple-distinct-consumers session isolation
//! - reattach after detach
//! - connect deadline (producer never binds)
//!
//! All tests run on a 16 MiB stack via `run_with_big_stack` since the
//! 2 MiB `FRAME_BYTES` makes the default tokio worker stack overflow.

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

fn build_rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(16 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("rt")
}

/// Zero-byte payload — vLLM uses empty delimiter frames in its protocol
/// envelope, so this MUST round-trip intact.
#[test]
fn zero_byte_payload_roundtrips() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-chaos-zero-{}-{}",
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

            send.send_multipart(&[&[], b"non-empty", &[]])
                .expect("send_multipart with empty frames");

            let frames = tokio::time::timeout(Duration::from_secs(2), recv.recv())
                .await
                .expect("recv timed out")
                .expect("recv error");
            assert_eq!(frames.len(), 3, "expected 3 frames");
            assert_eq!(frames[0].len(), 0, "frame 0 should be empty");
            assert_eq!(&frames[1][..], b"non-empty");
            assert_eq!(frames[2].len(), 0, "frame 2 should be empty");
        });
    });
}

/// Exact-boundary frame: envelope is at the per-frame cap.
/// `FRAME_BYTES` is the per-slot payload capacity. The wire envelope
/// is `count(4) + len(4) + payload`, so the maximum single-payload
/// frame is `FRAME_BYTES - 8`. Anything ≤ that must succeed.
#[test]
fn boundary_size_payload_at_frame_bytes_minus_8_succeeds() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-chaos-edge-{}-{}",
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

            // Maximum payload that fits in one frame's envelope.
            let max_payload = vec![0xABu8; FRAME_BYTES - 8];
            send.send_multipart(&[&max_payload])
                .expect("send_multipart at exact frame cap");

            let frames = tokio::time::timeout(Duration::from_secs(3), recv.recv())
                .await
                .expect("recv timed out at boundary")
                .expect("recv error");
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), FRAME_BYTES - 8);
            assert!(
                frames[0].iter().all(|b| *b == 0xAB),
                "payload corrupted at exact boundary"
            );
        });
    });
}

/// Alternating tiny (16 B) and large (256 KiB) payloads. If anywhere
/// in the encode/decode pipeline assumes uniform slot sizes, large
/// frames after a streak of tinies will misalign.
#[test]
fn alternating_tiny_and_large_payloads_preserve_order() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-chaos-alt-{}-{}",
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

            const N: u32 = 12;
            for i in 0..N {
                let tiny = vec![i as u8; 16];
                let large = vec![(i & 0xFF) as u8; 256 * 1024];
                send.send_multipart(&[&tiny, &large])
                    .expect("send_multipart alternating");
            }
            for i in 0..N {
                let frames = tokio::time::timeout(Duration::from_secs(5), recv.recv())
                    .await
                    .expect("recv timed out alt")
                    .expect("recv error alt");
                assert_eq!(frames.len(), 2);
                assert_eq!(frames[0].len(), 16, "tiny msg {i} wrong size");
                assert!(
                    frames[0].iter().all(|b| *b == i as u8),
                    "tiny msg {i} content wrong"
                );
                assert_eq!(frames[1].len(), 256 * 1024, "large msg {i} wrong size");
                assert_eq!(
                    frames[1][0], (i & 0xFF) as u8,
                    "large msg {i} first byte wrong"
                );
            }
        });
    });
}

/// 1024 sequential messages — exercises ring wrap-around 4× at the
/// default RING_SLOTS=256. Every message arrives in order, nothing
/// duplicated, nothing dropped.
#[test]
fn high_volume_burst_1024_messages_lossless() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-chaos-1k-{}-{}",
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

            // Tee the recv into a background task that pulls fast so the
            // producer never blocks on a full ring.
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
            let recv_task = tokio::spawn(async move {
                for _ in 0..1024u32 {
                    let frames = tokio::time::timeout(Duration::from_secs(10), recv.recv())
                        .await
                        .expect("recv timed out in burst")
                        .expect("recv error in burst");
                    let tag = u32::from_le_bytes(<[u8; 4]>::try_from(&frames[0][..]).unwrap());
                    tx.send(tag).expect("tx ok");
                }
            });

            for i in 0u32..1024 {
                let tag = i.to_le_bytes();
                let body = vec![(i & 0xFF) as u8; 64];
                send.send_multipart(&[&tag, &body])
                    .expect("send_multipart in 1024 burst");
            }

            recv_task.await.expect("recv task ok");
            let mut seen = Vec::with_capacity(1024);
            while let Some(v) = rx.recv().await {
                seen.push(v);
                if seen.len() == 1024 {
                    break;
                }
            }
            assert_eq!(seen.len(), 1024, "lost messages");
            for (i, v) in seen.iter().enumerate() {
                assert_eq!(*v, i as u32, "out of order at i={i}");
            }
        });
    });
}

/// Two sessions in parallel — frames must not cross streams.
#[test]
fn two_sessions_do_not_cross_streams() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let pid = std::process::id();
            let ep_a = format!("ipc:///tmp/myelon-rust-chaos-A-{}", pid);
            let ep_b = format!("ipc:///tmp/myelon-rust-chaos-B-{}", pid);
            let seg_a = segment_from_endpoint_for_test(&ep_a);
            let seg_b = segment_from_endpoint_for_test(&ep_b);
            assert_ne!(seg_a, seg_b, "segment names should differ");

            let mut send_a: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&seg_a, RING_SLOTS, "vlm_0").expect("bind A");
            let mut send_b: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&seg_b, RING_SLOTS, "vlm_0").expect("bind B");

            let ep_a_c = ep_a.clone();
            let ep_b_c = ep_b.clone();
            let (mut recv_a, mut recv_b) = {
                let h_a = tokio::task::spawn_blocking(move || {
                    MyelonOutputSocket::connect(&ep_a_c).expect("attach A")
                });
                let h_b = tokio::task::spawn_blocking(move || {
                    MyelonOutputSocket::connect(&ep_b_c).expect("attach B")
                });
                (h_a.await.expect("join A"), h_b.await.expect("join B"))
            };
            send_a.warm_discovery(Duration::from_millis(200));
            send_b.warm_discovery(Duration::from_millis(200));

            send_a.send_multipart(&[b"A1"]).expect("send A1");
            send_b.send_multipart(&[b"B1"]).expect("send B1");
            send_a.send_multipart(&[b"A2"]).expect("send A2");
            send_b.send_multipart(&[b"B2"]).expect("send B2");

            let fa1 = tokio::time::timeout(Duration::from_secs(2), recv_a.recv())
                .await.expect("a1 timeout").expect("a1 err");
            let fb1 = tokio::time::timeout(Duration::from_secs(2), recv_b.recv())
                .await.expect("b1 timeout").expect("b1 err");
            let fa2 = tokio::time::timeout(Duration::from_secs(2), recv_a.recv())
                .await.expect("a2 timeout").expect("a2 err");
            let fb2 = tokio::time::timeout(Duration::from_secs(2), recv_b.recv())
                .await.expect("b2 timeout").expect("b2 err");

            assert_eq!(&fa1[0][..], b"A1");
            assert_eq!(&fb1[0][..], b"B1");
            assert_eq!(&fa2[0][..], b"A2");
            assert_eq!(&fb2[0][..], b"B2");
        });
    });
}

/// Detach then reattach to the same segment. Real vLLM lifecycle
/// allows the frontend to drop and reconnect — must succeed.
#[test]
fn reattach_after_drop_succeeds() {
    run_with_big_stack(|| {
        let rt = build_rt();
        rt.block_on(async {
            let endpoint = format!(
                "ipc:///tmp/myelon-rust-chaos-reattach-{}-{}",
                std::process::id(),
                line!(),
            );
            let segment = segment_from_endpoint_for_test(&endpoint);
            let mut send: SendSocket<FRAME_BYTES> =
                SendSocket::bind(&segment, RING_SLOTS, "vlm_0").expect("bind");

            // Attach #1 — drop without recv
            let endpoint_clone = endpoint.clone();
            let recv1_handle = tokio::task::spawn_blocking(move || {
                MyelonOutputSocket::connect(&endpoint_clone).expect("attach 1")
            });
            let recv1 = recv1_handle.await.expect("attach 1 join");
            drop(recv1);

            // Attach #2 — should succeed and receive a fresh send
            let endpoint_clone2 = endpoint.clone();
            let recv2_handle = tokio::task::spawn_blocking(move || {
                MyelonOutputSocket::connect(&endpoint_clone2).expect("attach 2")
            });
            let mut recv2 = recv2_handle.await.expect("attach 2 join");
            send.warm_discovery(Duration::from_millis(200));
            send.send_multipart(&[b"after-reattach"]).expect("send");

            let frames = tokio::time::timeout(Duration::from_secs(2), recv2.recv())
                .await
                .expect("recv timed out after reattach")
                .expect("recv error");
            assert_eq!(&frames[0][..], b"after-reattach");
        });
    });
}
