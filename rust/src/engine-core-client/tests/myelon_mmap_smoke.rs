//! Minimal Rust-only repro for the MMAP segfault seen from PyO3.
//!
//! Mimics the Python smoke exactly:
//! - tempdir root
//! - segment name
//! - slots=16
//! - send a tiny payload via `MmapSendSocket::send_multipart`
//! - drain via `MmapRecvSocket::try_recv_multipart_with` until non-None
//!
//! If this passes, the bug is in the PyO3 binding (refcount, GIL,
//! transmute lifetime). If this crashes, the bug is in the MMAP
//! backend.

#![cfg(feature = "myelon_hot_path")]

use myelon_zmq::{MmapRecvSocket, MmapSendSocket};

const FRAME_BYTES: usize = 16 * 1024 * 1024;

fn run_with_big_stack<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("join");
}

#[test]
fn mmap_smoke_minimal_roundtrip() {
    run_with_big_stack(|| {
        let root = tempfile::tempdir().expect("tempdir");
        let segment = format!("smoke-{}", std::process::id());
        let mut send: MmapSendSocket<FRAME_BYTES> =
            MmapSendSocket::create(root.path(), &segment, 16, "vlm_0")
                .expect("mmap send create");
        let mut recv: MmapRecvSocket<FRAME_BYTES> =
            MmapRecvSocket::attach(root.path(), &segment, 16, "vlm_0")
                .expect("mmap recv attach");
        let ok = send.warm_discovery(std::time::Duration::from_millis(200));
        eprintln!("warm_discovery -> {}", ok);
        send.send_multipart(&[b"mmap-hi"]).expect("send_multipart");
        eprintln!("sent");

        for i in 0..1_000_000u64 {
            let got = recv.try_recv_multipart_with(|res| {
                let frames = res.expect("unpack");
                let owned: Vec<Vec<u8>> =
                    frames.iter().map(|s| s.to_vec()).collect();
                owned
            });
            if let Some(g) = got {
                eprintln!("got at spin={}, frames={:?}", i, g);
                assert_eq!(g.len(), 1);
                assert_eq!(&g[0][..], b"mmap-hi");
                return;
            }
        }
        panic!("did not receive after 1M spins");
    });
}
