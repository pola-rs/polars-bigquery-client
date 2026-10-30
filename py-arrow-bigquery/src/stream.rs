use arrow_bigquery_lib::{
    record_batch_into_arrow_array_stream, ArrowArrayStream, ArrowStreamOptions,
    BigQueryRecordBatchReceiver, RecordBatch, SchemaRef,
};
use pyo3::prelude::*;

enum StreamSource {
    Receiver(SchemaRef, BigQueryRecordBatchReceiver),
    Batch(RecordBatch),
}

/// A Python-exposed class that implements the Arrow C Stream interface.
///
/// This class acts as a bridge between the Rust BigQuery reader and Python Arrow
/// consumers (such as Polars and nanoarrow), allowing them to consume the data stream
/// directly via the Arrow C Stream Interface (`__arrow_c_stream__`) without copying data.
#[pyclass]
pub struct ArrowStreamExporter {
    /// The underlying BigQuery record batch source, wrapped in a mutex.
    /// It is an `Option` because the stream can only be consumed once.
    source: std::sync::Mutex<Option<StreamSource>>,
}

impl ArrowStreamExporter {
    pub(crate) fn new(schema: SchemaRef, receiver: BigQueryRecordBatchReceiver) -> Self {
        Self {
            source: std::sync::Mutex::new(Some(StreamSource::Receiver(schema, receiver))),
        }
    }

    pub(crate) fn from_record_batch(batch: RecordBatch) -> Self {
        Self {
            source: std::sync::Mutex::new(Some(StreamSource::Batch(batch))),
        }
    }
}

#[pymethods]
impl ArrowStreamExporter {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = requested_schema;
        let mut source_guard = self.source.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Stream receiver lock is poisoned")
        })?;
        let source = source_guard
            .take()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Stream already consumed"))?;

        let stream: ArrowArrayStream = match source {
            StreamSource::Receiver(schema, rx) => {
                let rt = pyo3_async_runtimes::tokio::get_runtime();
                let options =
                    ArrowStreamOptions::new(rt.handle().clone()).with_poll_wrapper(|wait_fn| {
                        Python::attach(|py| {
                            // We need to be able to stop if the Python side decides to, so
                            // occasionally check to see if there were any interrupts.
                            if let Err(py_err) = py.check_signals() {
                                py_err.restore(py);
                                return Err("Python interrupt");
                            }

                            py.detach(wait_fn);
                            Ok(())
                        })
                    });
                rx.into_arrow_array_stream(schema, options)
            },
            StreamSource::Batch(batch) => record_batch_into_arrow_array_stream(batch),
        };

        let capsule = pyo3::types::PyCapsule::new(py, stream, Some(c"arrow_array_stream".into()))?;
        Ok(capsule.into())
    }
}
