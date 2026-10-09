use polars_arrow::datatypes::ArrowSchemaRef;
use pyo3::prelude::*;

/// A Python-exposed class that implements the Arrow C Stream interface.
///
/// This class acts as a bridge between the Rust BigQuery reader and Python Polars,
/// allowing Polars to consume the data stream directly via the Arrow C Data Interface
/// (`__arrow_c_stream__`) without copying data.
#[pyclass]
pub struct ArrowStreamExporter {
    /// The schema of the Arrow stream.
    schema: ArrowSchemaRef,
    /// The underlying BigQuery record batch receiver, wrapped in a mutex.
    /// It is an `Option` because the stream can only be consumed once.
    receiver: std::sync::Mutex<Option<arrow_bigquery_lib::BigQueryRecordBatchReceiver>>,
}

impl ArrowStreamExporter {
    pub(crate) fn new(
        schema: ArrowSchemaRef,
        receiver: arrow_bigquery_lib::BigQueryRecordBatchReceiver,
    ) -> Self {
        Self {
            schema,
            receiver: std::sync::Mutex::new(Some(receiver)),
        }
    }
}

/// An iterator that adapts the asynchronous [`arrow_bigquery_lib::BigQueryRecordBatchReceiver`] into
/// a synchronous iterator yielding Arrow arrays.
///
/// This is used internally by [`ArrowStreamExporter`] to feed the Arrow C Stream.
/// Each iteration blocks on the Tokio runtime to receive the next batch.
struct ReceiverIterator {
    /// The receiver yielding record batches from the BigQuery Storage Read API.
    /// Held in an `Option` so it can be dropped immediately upon fatal error or interrupt
    /// to abort sibling background stream tasks without waiting for iterator GC.
    rx: Option<arrow_bigquery_lib::BigQueryRecordBatchReceiver>,
    /// The Arrow datatype (specifically a `Struct` type) matching the schema of the batches.
    dtype: polars_arrow::datatypes::ArrowDataType,
}

impl Iterator for ReceiverIterator {
    type Item =
        pyo3_polars::export::polars_error::PolarsResult<Box<dyn polars_arrow::array::Array>>;

    fn next(&mut self) -> Option<Self::Item> {
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let timeout_duration = std::time::Duration::from_millis(100);

        loop {
            let rx = self.rx.as_mut()?;

            let step = Python::attach(|py| {
                // We need to be able to stop if the Python side decides to, so
                // occasionally check to see if there were any interrupts.
                if let Err(py_err) = py.check_signals() {
                    py_err.restore(py);
                    return Err(());
                }

                Ok(py.detach(|| {
                    rt.block_on(async {
                        tokio::time::timeout(timeout_duration, rx.recv()).await
                    })
                }))
            });

            let result = match step {
                Ok(res) => res,
                Err(()) => {
                    self.rx = None;
                    return Some(Err(
                        pyo3_polars::export::polars_error::PolarsError::ComputeError(
                            "Python interrupt".into(),
                        ),
                    ));
                },
            };

            match result {
                Ok(Some(Ok(batch))) => {
                    let len = batch.len();
                    let (_, arrays) = batch.into_schema_and_arrays();
                    let struct_array = polars_arrow::array::StructArray::new(
                        self.dtype.clone(),
                        len,
                        arrays,
                        None,
                    );
                    return Some(Ok(
                        Box::new(struct_array) as Box<dyn polars_arrow::array::Array>
                    ));
                },
                Ok(Some(Err(err))) => {
                    // Drop the receiver immediately so sibling background streams are aborted now.
                    self.rx = None;
                    return Some(Err(
                        pyo3_polars::export::polars_error::PolarsError::ComputeError(
                            format!(
                                "BigQuery Storage API read error: {}",
                                err.format_causal_chain()
                            )
                            .into(),
                        ),
                    ));
                },
                Ok(None) => {
                    // Stream finished
                    self.rx = None;
                    return None;
                },
                Err(_) => {
                    // Timeout elapsed, loop again to check signals
                    continue;
                },
            }
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
        let mut rx_guard = self.receiver.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Stream receiver lock is poisoned")
        })?;
        let rx = rx_guard
            .take()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Stream already consumed"))?;

        let fields: Vec<polars_arrow::datatypes::Field> =
            self.schema.iter().map(|(_, field)| field.clone()).collect();
        let dtype = polars_arrow::datatypes::ArrowDataType::Struct(fields);

        let iter = ReceiverIterator {
            rx: Some(rx),
            dtype: dtype.clone(),
        };
        let box_iter = Box::new(iter)
            as Box<
                dyn Iterator<
                    Item = pyo3_polars::export::polars_error::PolarsResult<
                        Box<dyn polars_arrow::array::Array>,
                    >,
                >,
            >;

        let field = polars_arrow::datatypes::Field::new("".into(), dtype, false);

        let stream = polars_arrow::ffi::export_iterator(box_iter, field);

        let capsule = pyo3::types::PyCapsule::new(py, stream, Some(c"arrow_array_stream".into()))?;
        Ok(capsule.into())
    }
}
