#![allow(unsafe_op_in_unsafe_fn)]

use std::time::Duration;

use myelon::transport::{FixedFrame, FramedTransportProducer};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const FRAME_BYTES: usize = 2 * 1024 * 1024;

type InnerSend = FramedTransportProducer<FixedFrame<FRAME_BYTES>>;

#[pyclass]
struct NarrowSendSocket {
    inner: Option<InnerSend>,
    expected_consumer_id: String,
}

#[pymethods]
impl NarrowSendSocket {
    #[new]
    fn new(segment: &str, slots: usize, expected_consumer_id: &str) -> PyResult<Self> {
        let inner = FramedTransportProducer::create(segment, slots)
            .map_err(|e| PyRuntimeError::new_err(format!("bind: {e}")))?;
        Ok(Self {
            inner: Some(inner),
            expected_consumer_id: expected_consumer_id.to_string(),
        })
    }

    fn warm_discovery(&mut self, timeout_ms: u64) -> PyResult<bool> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("warm_discovery on closed NarrowSendSocket"))?;
        Ok(inner.discover_consumer_id(
            &self.expected_consumer_id,
            Duration::from_millis(timeout_ms),
        ))
    }

    /// Accept any contiguous u8 buffer-protocol object: bytes, bytearray,
    /// memoryview, numpy arrays. Avoids the bytes() copy at the Python edge
    /// so the engine can publish straight out of a reused encode_into() buf.
    /// publish() memcpy's into the SHM ring before returning, so the caller
    /// is free to reuse the source buffer immediately afterwards.
    fn send(&mut self, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("send on closed NarrowSendSocket"))?;
        let buf: PyBuffer<u8> = PyBuffer::get(payload)?;
        if !buf.is_c_contiguous() {
            return Err(PyRuntimeError::new_err(
                "NarrowSendSocket.send requires a C-contiguous buffer",
            ));
        }
        let len = buf.item_count();
        let slice = unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const u8, len) };
        inner.publish(slice, 0);
        Ok(())
    }

    fn close(&mut self) {
        self.inner = None;
    }
}

#[pymodule]
fn myelon_narrow_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NarrowSendSocket>()?;
    Ok(())
}
