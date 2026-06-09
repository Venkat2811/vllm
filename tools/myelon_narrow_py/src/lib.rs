#![allow(unsafe_op_in_unsafe_fn)]

use std::time::Duration;

use myelon::transport::{FixedFrame, FramedTransportProducer};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

    fn send(&mut self, payload: &Bound<'_, PyBytes>) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("send on closed NarrowSendSocket"))?;
        inner.publish(payload.as_bytes(), 0);
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
