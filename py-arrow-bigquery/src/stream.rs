use polars_arrow::datatypes::ArrowSchemaRef;
use polars_arrow::record_batch::RecordBatch;
use pyo3::prelude::*;

enum StreamSource {
    Receiver(arrow_bigquery_lib::BigQueryRecordBatchReceiver),
    Batch(RecordBatch),
}

fn record_batch_to_struct_array(
    batch: RecordBatch,
    dtype: polars_arrow::datatypes::ArrowDataType,
) -> Box<dyn polars_arrow::array::Array> {
    let len = batch.len();
    let (_, arrays) = batch.into_schema_and_arrays();
    let struct_array = polars_arrow::array::StructArray::new(dtype, len, arrays, None);
    Box::new(struct_array) as Box<dyn polars_arrow::array::Array>
}

/// A Python-exposed class that implements the Arrow C Stream interface.
///
/// This class acts as a bridge between the Rust BigQuery reader and Python Polars,
/// allowing Polars to consume the data stream directly via the Arrow C Data Interface
/// (`__arrow_c_stream__`) without copying data.
#[pyclass]
pub struct ArrowStreamExporter {
    /// The schema of the Arrow stream.
    schema: ArrowSchemaRef,
    /// Tracks whether `schema` was already validated by `validate_ffi_schema` /
    /// `parse_arrow_schema`.
    schema_ffi_validated: bool,
    /// The underlying BigQuery record batch source, wrapped in a mutex.
    /// It is an `Option` because the stream can only be consumed once.
    source: std::sync::Mutex<Option<StreamSource>>,
    /// Thread-safe handle enabling `close()` to abort background Tokio/gRPC stream
    /// tasks even after `__arrow_c_stream__` hands the receiver to a `PyCapsule`.
    abort_handle: Option<arrow_bigquery_lib::BigQueryStreamAbortHandle>,
}

impl ArrowStreamExporter {
    #[cfg(feature = "testing")]
    pub(crate) fn new(
        schema: ArrowSchemaRef,
        receiver: arrow_bigquery_lib::BigQueryRecordBatchReceiver,
    ) -> Self {
        Self::new_with_validation(schema, receiver, false)
    }

    pub(crate) fn new_validated(
        schema: ArrowSchemaRef,
        receiver: arrow_bigquery_lib::BigQueryRecordBatchReceiver,
    ) -> Self {
        Self::new_with_validation(schema, receiver, true)
    }

    fn new_with_validation(
        schema: ArrowSchemaRef,
        receiver: arrow_bigquery_lib::BigQueryRecordBatchReceiver,
        schema_ffi_validated: bool,
    ) -> Self {
        let abort_handle = Some(receiver.abort_handle());
        Self {
            schema,
            schema_ffi_validated,
            source: std::sync::Mutex::new(Some(StreamSource::Receiver(receiver))),
            abort_handle,
        }
    }

    pub(crate) fn from_record_batch(batch: RecordBatch) -> Self {
        let schema = std::sync::Arc::new(batch.schema().clone());
        Self {
            schema,
            schema_ffi_validated: false,
            source: std::sync::Mutex::new(Some(StreamSource::Batch(batch))),
            abort_handle: None,
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
    /// Shared handle checked so external `ArrowStreamExporter::close()` triggers an
    /// explicit error instead of appearing as a clean EOF (which would truncate rows).
    abort_handle: arrow_bigquery_lib::BigQueryStreamAbortHandle,
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
            if self.abort_handle.is_aborted() {
                self.rx = None;
                return Some(Err(
                    pyo3_polars::export::polars_error::PolarsError::ComputeError(
                        "BigQuery Arrow stream was closed before completion".into(),
                    ),
                ));
            }

            // True zero-GIL fast path for cached non-main threads (e.g., Polars/Rayon
            // streaming engine threads). `py.check_signals()` is a no-op outside the
            // main thread, so avoid touching the GIL on every batch.
            let result = if crate::cached_is_main_python_thread() == Some(false) {
                Ok(rt.block_on(rx.recv()))
            } else {
                let step = Python::attach(|py| {
                    let is_main = match crate::is_main_python_thread(py) {
                        Ok(v) => v,
                        Err(py_err) => {
                            let is_interrupt =
                                py_err.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py);
                            let msg = if is_interrupt {
                                "Python interrupt".to_string()
                            } else {
                                format!("Python error while checking thread state: {py_err}")
                            };
                            py_err.restore(py);
                            return Err(msg);
                        },
                    };

                    if !is_main {
                        // First call on this non-main thread: cache is now populated so
                        // subsequent batches skip `Python::attach` entirely.
                        return Ok(Ok(py.detach(|| rt.block_on(rx.recv()))));
                    }

                    // We need to be able to stop if the Python side decides to, so
                    // occasionally check to see if there were any interrupts.
                    if let Err(py_err) = py.check_signals() {
                        py_err.restore(py);
                        return Err("Python interrupt".to_string());
                    }

                    Ok(py.detach(|| {
                        rt.block_on(async {
                            tokio::time::timeout(timeout_duration, rx.recv()).await
                        })
                    }))
                });

                match step {
                    Ok(res) => res,
                    Err(msg) => {
                        self.rx = None;
                        return Some(Err(
                            pyo3_polars::export::polars_error::PolarsError::ComputeError(
                                msg.into(),
                            ),
                        ));
                    },
                }
            };

            match result {
                Ok(Some(Ok(batch))) => {
                    if self.abort_handle.is_aborted() {
                        self.rx = None;
                        return Some(Err(
                            pyo3_polars::export::polars_error::PolarsError::ComputeError(
                                "BigQuery Arrow stream was closed before completion".into(),
                            ),
                        ));
                    }
                    return Some(Ok(record_batch_to_struct_array(batch, self.dtype.clone())));
                },
                Ok(Some(Err(err))) => {
                    // Drop the receiver immediately so sibling background streams are aborted now.
                    self.rx = None;
                    // polars-arrow's Arrow C Stream export callback converts errors with
                    // `CString::new(...).unwrap()`, which panics across `extern "C"` on interior NUL bytes.
                    let sanitized_err = err.format_causal_chain().replace('\0', "\\x00");
                    return Some(Err(
                        pyo3_polars::export::polars_error::PolarsError::ComputeError(
                            format!("BigQuery Storage API read error: {sanitized_err}").into(),
                        ),
                    ));
                },
                Ok(None) => {
                    // Stream finished (or aborted externally via `ArrowStreamExporter::close`).
                    self.rx = None;
                    if self.abort_handle.is_aborted() {
                        return Some(Err(
                            pyo3_polars::export::polars_error::PolarsError::ComputeError(
                                "BigQuery Arrow stream was closed before completion".into(),
                            ),
                        ));
                    }
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
    /// Explicitly drop the underlying receiver / batch (if not yet exported) and
    /// signal cancellation on background Tokio/gRPC stream tasks so they abort
    /// immediately instead of waiting for garbage collection—even after
    /// `__arrow_c_stream__` transfers ownership of the receiver to a `PyCapsule`.
    fn close(&self) {
        let dropped = {
            let mut source_guard = self.source.lock().unwrap_or_else(|e| e.into_inner());
            source_guard.take()
        };
        // Drop outside `self.source` lock so `BigQueryRecordBatchReceiver::drop` /
        // `abort()` never runs under the mutex.
        drop(dropped);
        if let Some(abort_handle) = &self.abort_handle {
            abort_handle.abort();
        }
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = requested_schema;
        if !self.schema_ffi_validated {
            if let Err(err) = arrow_bigquery_lib::validate_ffi_schema(&self.schema) {
                // Drain the receiver immediately so background Tokio/gRPC stream tasks abort
                // instead of running until Python garbage-collects this exporter.
                self.close();
                return Err(crate::error::to_py_err(py, err));
            }
        }
        let source = {
            let mut source_guard = self.source.lock().map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err("Stream receiver lock is poisoned")
            })?;
            source_guard
                .take()
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Stream already consumed"))?
        };

        let fields: Vec<polars_arrow::datatypes::Field> =
            self.schema.iter().map(|(_, field)| field.clone()).collect();
        let dtype = polars_arrow::datatypes::ArrowDataType::Struct(fields);

        let box_iter: Box<
            dyn Iterator<
                Item = pyo3_polars::export::polars_error::PolarsResult<
                    Box<dyn polars_arrow::array::Array>,
                >,
            >,
        > = match source {
            StreamSource::Receiver(rx) => {
                let abort_handle = self
                    .abort_handle
                    .clone()
                    .unwrap_or_else(|| rx.abort_handle());
                Box::new(ReceiverIterator {
                    rx: Some(rx),
                    abort_handle,
                    dtype: dtype.clone(),
                })
            },
            StreamSource::Batch(batch) => Box::new(std::iter::once(Ok(
                record_batch_to_struct_array(batch, dtype.clone()),
            ))),
        };

        let field = polars_arrow::datatypes::Field::new("".into(), dtype, false);

        let stream = polars_arrow::ffi::export_iterator(box_iter, field);

        let capsule = pyo3::types::PyCapsule::new(py, stream, Some(c"arrow_array_stream".into()))?;
        Ok(capsule.into())
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use std::sync::Arc;

    use polars_arrow::array::{Array, Int64Array};
    use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
    use polars_arrow::record_batch::RecordBatch;

    use super::*;

    #[test]
    fn test_receiver_iterator_worker_thread_skips_gil_after_caching() {
        Python::initialize();

        let field = Field::new("x".into(), ArrowDataType::Int64, false);
        let schema = ArrowSchema::from_iter([field.clone()]);
        let dtype = ArrowDataType::Struct(vec![field]);
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let batch1 = RecordBatch::try_new(
            1,
            Arc::new(schema.clone()),
            vec![Box::new(Int64Array::from_slice([1])) as Box<dyn Array>],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            1,
            Arc::new(schema),
            vec![Box::new(Int64Array::from_slice([2])) as Box<dyn Array>],
        )
        .unwrap();
        rt.block_on(async {
            tx.send(Ok(batch1)).await.unwrap();
            tx.send(Ok(batch2)).await.unwrap();
            drop(tx);
        });
        let receiver = arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![]);

        use std::sync::atomic::{AtomicBool, Ordering};

        let cache_ready = Arc::new(AtomicBool::new(false));
        let gil_held = Arc::new(AtomicBool::new(false));
        let batch2_done = Arc::new(AtomicBool::new(false));
        let cache_ready_w = Arc::clone(&cache_ready);
        let gil_held_w = Arc::clone(&gil_held);
        let batch2_done_w = Arc::clone(&batch2_done);

        let worker = std::thread::spawn(move || {
            let abort_handle = receiver.abort_handle();
            let mut iter = ReceiverIterator {
                rx: Some(receiver),
                abort_handle,
                dtype,
            };
            // First batch populates `IS_MAIN_THREAD_CACHE` on this worker thread.
            assert!(iter.next().unwrap().is_ok());
            assert_eq!(crate::cached_is_main_python_thread(), Some(false));
            cache_ready_w.store(true, Ordering::Release);
            // Briefly yield until the parent thread actually holds the Python GIL.
            while !gil_held_w.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            // Pulling the second batch and EOF must complete WITHOUT acquiring the GIL!
            assert!(iter.next().unwrap().is_ok());
            assert!(iter.next().is_none());
            batch2_done_w.store(true, Ordering::Release);
        });

        while !cache_ready.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        // Hold the GIL on the test thread briefly (typically < 1 ms) and verify the
        // worker thread drains `batch2` + EOF without deadlocking on the GIL.
        Python::attach(|_py| {
            gil_held.store(true, Ordering::Release);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !batch2_done.load(Ordering::Acquire) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "worker thread deadlocked trying to acquire GIL on cached non-main path"
                );
                std::thread::yield_now();
            }
        });
        worker.join().unwrap();
    }

    #[test]
    fn test_exporter_rejects_unvalidated_ffi_schema_and_aborts_background_task() {
        use std::sync::atomic::{AtomicBool, Ordering};

        use polars_arrow::datatypes::IntervalUnit;

        Python::initialize();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_task = Arc::clone(&dropped);
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let guard = DropGuard(dropped_task);
        let handle = rt.spawn(async move {
            let _keep_tx = tx;
            let _guard = guard;
            std::future::pending::<()>().await;
        });

        let schema = Arc::new(ArrowSchema::from_iter([Field::new(
            "dur".into(),
            ArrowDataType::Interval(IntervalUnit::MonthDayNano),
            true,
        )]));
        let receiver =
            arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);
        let exporter = ArrowStreamExporter::new(schema, receiver);

        Python::attach(|py| {
            let err = exporter.__arrow_c_stream__(py, None).unwrap_err();
            assert!(err.is_instance_of::<crate::error::UnsupportedTypeError>(py));
            assert!(err.is_instance_of::<crate::error::BigQueryError>(py));
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !dropped.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            dropped.load(Ordering::Acquire),
            "background task must be aborted when validate_ffi_schema fails in __arrow_c_stream__"
        );
    }

    #[test]
    fn test_receiver_iterator_sanitizes_interior_nul_bytes_in_errors() {
        Python::initialize();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        rt.block_on(async {
            tx.send(Err(arrow_bigquery_lib::BigQueryError::Protocol(
                "bad\0msg".into(),
            )))
            .await
            .unwrap();
        });
        let receiver = arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![]);
        let abort_handle = receiver.abort_handle();
        let dtype =
            ArrowDataType::Struct(vec![Field::new("x".into(), ArrowDataType::Int64, false)]);
        let mut iter = ReceiverIterator {
            rx: Some(receiver),
            abort_handle,
            dtype,
        };
        let err = iter.next().unwrap().unwrap_err();
        let msg = err.to_string();
        assert!(!msg.contains('\0'), "interior NUL was not sanitized: {msg}");
        assert!(
            msg.contains("bad\\x00msg"),
            "unexpected sanitized error message: {msg}"
        );
    }

    #[test]
    fn test_exporter_close_after_capsule_export_aborts_task_and_errors_iterator() {
        use std::sync::atomic::{AtomicBool, Ordering};

        Python::initialize();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_task = Arc::clone(&dropped);
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let guard = DropGuard(dropped_task);
        let handle = rt.spawn(async move {
            let _keep_tx = tx;
            let _guard = guard;
            std::future::pending::<()>().await;
        });

        let field = Field::new("x".into(), ArrowDataType::Int64, false);
        let schema = Arc::new(ArrowSchema::from_iter([field.clone()]));
        let dtype = ArrowDataType::Struct(vec![field]);
        let receiver =
            arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);
        let exporter = ArrowStreamExporter::new(schema, receiver);

        // Hold the exported PyCapsule alive simulating a traceback frame pinning it.
        let _capsule = Python::attach(|py| exporter.__arrow_c_stream__(py, None).unwrap());
        assert!(!dropped.load(Ordering::Acquire));

        // Closing the exporter after `__arrow_c_stream__` must still abort the task.
        exporter.close();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !dropped.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            dropped.load(Ordering::Acquire),
            "background task must be aborted when close() is called after __arrow_c_stream__"
        );

        // Also verify `ReceiverIterator::next` fails loudly instead of returning a
        // silent truncated EOF (`None`) when externally closed.
        let (_tx2, rx2) = tokio::sync::mpsc::channel(4);
        let receiver2 =
            arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx2, vec![]);
        let abort_handle = receiver2.abort_handle();
        let mut iter = ReceiverIterator {
            rx: Some(receiver2),
            abort_handle: abort_handle.clone(),
            dtype,
        };
        abort_handle.abort();
        let err = iter
            .next()
            .expect("closed stream must yield Err instead of silent EOF")
            .expect_err("expected ComputeError on closed stream");
        assert!(
            err.to_string().contains("closed before completion"),
            "unexpected error: {err}"
        );
        assert!(iter.next().is_none());
    }
}
