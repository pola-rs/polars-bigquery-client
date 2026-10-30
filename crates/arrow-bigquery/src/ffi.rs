use std::time::Duration;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};

use crate::BigQueryRecordBatchReceiver;

/// Arrow C Stream Interface (`ArrowArrayStream`) struct exported by `arrow-bigquery`.
///
/// This is a re-export of [`arrow_array::ffi_stream::FFI_ArrowArrayStream`], which has
/// `#[repr(C)]` layout matching the canonical Arrow C Stream ABI (`struct ArrowArrayStream`).
pub type ArrowArrayStream = FFI_ArrowArrayStream;

type PollWrapperFn = Box<dyn FnMut(&mut (dyn FnMut() + Send)) -> Result<(), String> + Send>;

const DEFAULT_INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Options controlling how an asynchronous [`BigQueryRecordBatchReceiver`] is bridged
/// into a synchronous [`ArrowArrayStream`] (`FFI_ArrowArrayStream`).
pub struct ArrowStreamOptions {
    runtime_handle: tokio::runtime::Handle,
    poll_interval: Duration,
    poll_wrapper: Option<PollWrapperFn>,
}

impl std::fmt::Debug for ArrowStreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowStreamOptions")
            .field("poll_interval", &self.poll_interval)
            .field("has_poll_wrapper", &self.poll_wrapper.is_some())
            .finish()
    }
}

impl ArrowStreamOptions {
    /// Creates stream export options using `runtime_handle` to drive asynchronous
    /// channel receives inside the C Stream `get_next` callback.
    ///
    /// By default, no interrupt-polling callback is registered, so `get_next` blocks
    /// directly until the next batch (or end-of-stream) arrives without periodic timer wakeups.
    pub fn new(runtime_handle: tokio::runtime::Handle) -> Self {
        Self {
            runtime_handle,
            poll_interval: DEFAULT_INTERRUPT_POLL_INTERVAL,
            poll_wrapper: None,
        }
    }

    /// Sets the maximum duration to block per poll step when an interrupt check or
    /// poll wrapper is configured (defaults to `100ms`).
    pub fn with_poll_interval(mut self, poll_interval: Duration) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    /// Registers a periodic interrupt-polling callback invoked once before each
    /// `poll_interval` wait while `get_next` is blocked waiting for a [`RecordBatch`].
    ///
    /// If `check` returns `Err(err)`, the underlying [`BigQueryRecordBatchReceiver`] is
    /// dropped immediately (aborting background gRPC stream tasks) and `get_next`
    /// fails with the error message.
    pub fn with_interrupt_check<F, E>(mut self, mut check: F) -> Self
    where
        F: FnMut() -> Result<(), E> + Send + 'static,
        E: std::fmt::Display,
    {
        self.poll_wrapper = Some(Box::new(move |wait_fn| {
            check().map_err(|e| e.to_string())?;
            wait_fn();
            Ok(())
        }));
        self
    }

    /// Registers a custom poll step wrapper around each `poll_interval` blocking wait.
    ///
    /// This is useful for language runtimes (such as Python / PyO3) that need to both
    /// check for pending signals (e.g., `Ctrl-C`) and release thread-local runtime locks
    /// (such as detaching from the Python GIL) around `wait_fn` while blocked on Tokio.
    ///
    /// Implementations of `wrapper` **must** invoke `wait_fn` before returning `Ok(())`.
    pub fn with_poll_wrapper<F, E>(mut self, mut wrapper: F) -> Self
    where
        F: FnMut(&mut (dyn FnMut() + Send)) -> Result<(), E> + Send + 'static,
        E: std::fmt::Display,
    {
        self.poll_wrapper = Some(Box::new(move |wait_fn| {
            wrapper(wait_fn).map_err(|e| e.to_string())
        }));
        self
    }
}

/// Sanitizes an error string for `FFI_ArrowArrayStream` (`CString::new` panics on interior NULs).
fn ffi_safe_arrow_error(msg: String) -> ArrowError {
    let clean = if msg.contains('\0') {
        msg.replace('\0', "\u{FFFD}")
    } else {
        msg
    };
    ArrowError::ExternalError(clean.into())
}

struct BigQueryRecordBatchStreamReader {
    schema: SchemaRef,
    /// Held in an `Option` so it can be dropped immediately upon fatal error,
    /// interrupt, or EOF to abort sibling background stream tasks without waiting
    /// for the C consumer to release the `ArrowArrayStream`.
    rx: Option<BigQueryRecordBatchReceiver>,
    options: ArrowStreamOptions,
}

impl Iterator for BigQueryRecordBatchStreamReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let rx = self.rx.as_mut()?;
            let runtime = &self.options.runtime_handle;
            let poll_interval = self.options.poll_interval;

            let recv_outcome = if let Some(poll_wrapper) = self.options.poll_wrapper.as_mut() {
                let mut timed_result = None;
                if let Err(interrupt_msg) = poll_wrapper(&mut || {
                    timed_result =
                        Some(runtime.block_on(async {
                            tokio::time::timeout(poll_interval, rx.recv()).await
                        }));
                }) {
                    self.rx = None;
                    return Some(Err(ffi_safe_arrow_error(interrupt_msg)));
                }

                match timed_result {
                    Some(Ok(item)) => item,
                    Some(Err(_elapsed)) => continue,
                    None => {
                        self.rx = None;
                        return Some(Err(ffi_safe_arrow_error(
                            "ArrowStreamOptions poll_wrapper returned Ok(()) without invoking wait callback"
                                .to_owned(),
                        )));
                    },
                }
            } else {
                runtime.block_on(rx.recv())
            };

            match recv_outcome {
                Some(Ok(batch)) => return Some(Ok(batch)),
                Some(Err(err)) => {
                    self.rx = None;
                    let msg = format!(
                        "BigQuery Storage API read error: {}",
                        err.format_causal_chain()
                    );
                    return Some(Err(ffi_safe_arrow_error(msg)));
                },
                None => {
                    self.rx = None;
                    return None;
                },
            }
        }
    }
}

impl RecordBatchReader for BigQueryRecordBatchStreamReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl BigQueryRecordBatchReceiver {
    /// Exports this [`BigQueryRecordBatchReceiver`] and its [`SchemaRef`] as an
    /// Arrow C Stream Interface [`ArrowArrayStream`] (`FFI_ArrowArrayStream`).
    pub fn into_arrow_array_stream(
        self,
        schema: SchemaRef,
        options: ArrowStreamOptions,
    ) -> ArrowArrayStream {
        let reader = BigQueryRecordBatchStreamReader {
            schema,
            rx: Some(self),
            options,
        };
        FFI_ArrowArrayStream::new(Box::new(reader))
    }
}

/// Exports a single [`RecordBatch`] (such as the result of [`crate::Query::dry_run`])
/// as an Arrow C Stream Interface [`ArrowArrayStream`] (`FFI_ArrowArrayStream`).
pub fn record_batch_into_arrow_array_stream(batch: RecordBatch) -> ArrowArrayStream {
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(std::iter::once(Ok(batch)), schema);
    FFI_ArrowArrayStream::new(Box::new(reader))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    use arrow_array::ffi_stream::RecordBatchStreamReader;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::BigQueryError;

    fn sample_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]))
    }

    fn sample_batch(schema: &SchemaRef, values: &[i64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![Arc::new(Int64Array::from(values.to_vec()))],
        )
        .unwrap()
    }

    #[test]
    fn test_record_batch_into_arrow_array_stream_roundtrip() {
        let schema = sample_schema();
        let batch = sample_batch(&schema, &[10, 20, 30]);

        let ffi_stream = record_batch_into_arrow_array_stream(batch.clone());
        let mut reader = RecordBatchStreamReader::try_new(ffi_stream).unwrap();
        assert_eq!(reader.schema(), schema);

        let out = reader.next().unwrap().unwrap();
        assert_eq!(out, batch);
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_receiver_into_arrow_array_stream_multi_batch_and_empty() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let schema = sample_schema();
        let b1 = sample_batch(&schema, &[1, 2]);
        let b2 = sample_batch(&schema, &[3, 4, 5]);

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let b1_clone = b1.clone();
        let b2_clone = b2.clone();
        let handle = rt.spawn(async move {
            tx.send(Ok(b1_clone)).await.unwrap();
            tx.send(Ok(b2_clone)).await.unwrap();
        });

        let receiver = BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);
        let ffi_stream = receiver.into_arrow_array_stream(
            Arc::clone(&schema),
            ArrowStreamOptions::new(rt.handle().clone()),
        );

        let mut reader = RecordBatchStreamReader::try_new(ffi_stream).unwrap();
        assert_eq!(reader.schema(), schema);
        assert_eq!(reader.next().unwrap().unwrap(), b1);
        assert_eq!(reader.next().unwrap().unwrap(), b2);
        assert!(reader.next().is_none());

        // Empty stream
        let (empty_tx, empty_rx) = tokio::sync::mpsc::channel(1);
        drop(empty_tx);
        let empty_receiver = BigQueryRecordBatchReceiver::new_for_testing(empty_rx, vec![]);
        let empty_stream = empty_receiver.into_arrow_array_stream(
            Arc::clone(&schema),
            ArrowStreamOptions::new(rt.handle().clone()),
        );
        let mut empty_reader = RecordBatchStreamReader::try_new(empty_stream).unwrap();
        assert_eq!(empty_reader.schema(), schema);
        assert!(empty_reader.next().is_none());
    }

    struct DropGuard(Arc<AtomicBool>);

    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_receiver_arrow_stream_error_aborts_tasks_immediately_and_handles_nul_bytes() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let dropped_flag = Arc::new(AtomicBool::new(false));
        let guard = DropGuard(Arc::clone(&dropped_flag));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        let sibling_handle = rt.spawn(async move {
            let _guard = guard;
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        });
        rt.block_on(started_rx).unwrap();

        tx.try_send(Err(BigQueryError::Protocol(
            "corrupted payload\0with nul".into(),
        )))
        .unwrap();

        let receiver = BigQueryRecordBatchReceiver::new_for_testing(rx, vec![sibling_handle]);
        let schema = sample_schema();
        let ffi_stream =
            receiver.into_arrow_array_stream(schema, ArrowStreamOptions::new(rt.handle().clone()));

        // Keep `reader` alive while checking `dropped_flag` to prove `self.rx = None`
        // aborted the sibling task immediately on error before stream release!
        let mut reader = RecordBatchStreamReader::try_new(ffi_stream).unwrap();
        let err = reader.next().unwrap().unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("BigQuery Storage API read error:"));
        assert!(err_msg.contains("corrupted payload\u{FFFD}with nul"));

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !dropped_flag.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            dropped_flag.load(Ordering::SeqCst),
            "sibling task should be aborted immediately on stream error before reader is dropped"
        );
    }

    #[test]
    fn test_receiver_arrow_stream_interrupt_check_cancels_blocking_read() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let dropped_flag = Arc::new(AtomicBool::new(false));
        let guard = DropGuard(Arc::clone(&dropped_flag));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        let handle = rt.spawn(async move {
            let _guard = guard;
            let _keep_tx = tx;
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        });
        rt.block_on(started_rx).unwrap();

        let poll_count = Arc::new(AtomicUsize::new(0));
        let poll_count_clone = Arc::clone(&poll_count);

        let options = ArrowStreamOptions::new(rt.handle().clone())
            .with_poll_interval(Duration::from_millis(20))
            .with_interrupt_check(move || {
                let n = poll_count_clone.fetch_add(1, Ordering::SeqCst);
                if n >= 2 {
                    Err("Python interrupt")
                } else {
                    Ok(())
                }
            });

        let receiver = BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);
        let ffi_stream = receiver.into_arrow_array_stream(sample_schema(), options);
        let mut reader = RecordBatchStreamReader::try_new(ffi_stream).unwrap();

        let err = reader.next().unwrap().unwrap_err();
        assert!(err.to_string().contains("Python interrupt"));
        assert!(poll_count.load(Ordering::SeqCst) >= 3);

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !dropped_flag.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            dropped_flag.load(Ordering::SeqCst),
            "background stream task should be aborted immediately upon interrupt"
        );
    }

    #[test]
    fn test_dropping_arrow_array_stream_early_aborts_tasks() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let dropped_flag = Arc::new(AtomicBool::new(false));
        let guard = DropGuard(Arc::clone(&dropped_flag));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        let handle = rt.spawn(async move {
            let _guard = guard;
            let _keep_tx = tx;
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        });
        rt.block_on(started_rx).unwrap();

        let receiver = BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);
        let ffi_stream = receiver.into_arrow_array_stream(
            sample_schema(),
            ArrowStreamOptions::new(rt.handle().clone()),
        );

        assert!(!dropped_flag.load(Ordering::SeqCst));
        drop(ffi_stream);

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !dropped_flag.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            dropped_flag.load(Ordering::SeqCst),
            "dropping FFI_ArrowArrayStream should invoke release and abort background tasks"
        );
    }
}
