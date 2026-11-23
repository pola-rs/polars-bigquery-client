use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field, IntervalUnit};
use polars_arrow::io::ipc::read::{
    read_stream_metadata, StreamMetadata, StreamReader, StreamState,
};
use polars_arrow::record_batch::RecordBatch;

use crate::BigQueryError;

/// Maximum recursive type-nesting depth inspected by `decode.rs` helpers.
///
/// BigQuery limits `STRUCT`/`RECORD` nesting to 15 levels, so 64 comfortably covers
/// every valid BigQuery schema (including nested containers and extension wrappers)
/// while guarding the Rust call stack against malformed IPC metadata.
const MAX_SCHEMA_NESTING_DEPTH: usize = 64;

/// Controls schema validation performed when decoding Arrow IPC schema metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaDecodeOptions {
    /// Reject Arrow data types that can be decoded in pure-Rust IPC but panic or abort
    /// during `polars-arrow` C Data Interface FFI export (`Interval(MonthDayNano)`,
    /// mismatched `Time32`/`Time64` units, empty `Union`, and interior NUL bytes in
    /// field names or timestamp timezones).
    ///
    /// Note: types that `polars-arrow`'s IPC decoder cannot read at all (`Decimal256`,
    /// `Interval(DayTime)`, `Interval(MonthDayMillis)`, and `Unknown`) are rejected
    /// unconditionally even when `validate_ffi_types` is `false`.
    pub validate_ffi_types: bool,
}

impl Default for SchemaDecodeOptions {
    fn default() -> Self {
        Self {
            validate_ffi_types: true,
        }
    }
}

fn check_nesting_depth(depth: usize, path: &str) -> Result<(), BigQueryError> {
    if depth > MAX_SCHEMA_NESTING_DEPTH {
        return Err(BigQueryError::Protocol(format!(
            "Arrow schema nesting depth for `{path}` exceeds maximum supported depth ({MAX_SCHEMA_NESTING_DEPTH})"
        )));
    }
    Ok(())
}

fn child_field_path(parent: &str, field: &Field, idx: usize) -> String {
    if field.name.is_empty() {
        format!("{parent}[{idx}]")
    } else {
        format!("{parent}.{}", field.name)
    }
}

fn format_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn format_sql_cast_hint(path: &str) -> String {
    if path.contains('.') || path.contains('[') {
        "Use a SQL query to select or CAST the nested field (e.g. `CAST(field AS STRING)` or `CAST(field AS FLOAT64)`).".to_string()
    } else {
        let escaped = path.replace('`', "\\`");
        format!(
            "Use a SQL query to CAST the column (e.g. `CAST(`{escaped}` AS STRING)` or `CAST(`{escaped}` AS FLOAT64)`)"
        )
    }
}

fn validate_field_name_for_ffi(name: &str, path: &str) -> Result<(), BigQueryError> {
    if name.as_bytes().contains(&0) {
        return Err(BigQueryError::Protocol(format!(
            "column `{path}` contains an interior NUL byte in its field name, which cannot be exported over Arrow C Data Interface FFI"
        )));
    }
    Ok(())
}

fn validate_arrow_field(
    field: &Field,
    path: &str,
    depth: usize,
    validate_ffi_types: bool,
) -> Result<(), BigQueryError> {
    if validate_ffi_types {
        validate_field_name_for_ffi(field.name.as_str(), path)?;
    }
    validate_supported_arrow_dtype(field.dtype(), path, depth, validate_ffi_types)
}

fn validate_supported_arrow_dtype(
    dtype: &ArrowDataType,
    path: &str,
    depth: usize,
    validate_ffi_types: bool,
) -> Result<(), BigQueryError> {
    check_nesting_depth(depth, path)?;
    match dtype {
        // Types that `polars-arrow`'s IPC `StreamReader` (`deserialize::read`) cannot decode
        // without panicking inside `with_match_primitive_type_full!` / `to_physical_type()`.
        // Rejected unconditionally even when `validate_ffi_types == false`.
        ArrowDataType::Decimal256(precision, scale) => {
            let hint = format_sql_cast_hint(path);
            Err(BigQueryError::UnsupportedType(format!(
                "column `{path}` uses Arrow Decimal256({precision}, {scale}) (BigQuery BIGNUMERIC), \
                 which cannot be decoded by polars-arrow. {hint}"
            )))
        },
        ArrowDataType::Interval(IntervalUnit::DayTime) => {
            Err(BigQueryError::UnsupportedType(format!(
                "column `{path}` uses Arrow Interval(DayTime), which cannot be decoded by polars-arrow"
            )))
        },
        ArrowDataType::Interval(IntervalUnit::MonthDayMillis) => {
            Err(BigQueryError::UnsupportedType(format!(
                "column `{path}` uses Arrow Interval(MonthDayMillis), which cannot be decoded or exported over Arrow C Data Interface FFI"
            )))
        },
        ArrowDataType::Unknown => Err(BigQueryError::UnsupportedType(format!(
            "column `{path}` uses ArrowDataType::Unknown, which is not supported by polars-arrow"
        ))),
        // Types decodable in pure-Rust IPC but known to panic/abort during Arrow C Data Interface
        // FFI export (`polars-arrow::ffi::export_iterator`).
        ArrowDataType::Interval(IntervalUnit::MonthDayNano) if validate_ffi_types => {
            Err(BigQueryError::UnsupportedType(format!(
                "column `{path}` uses Arrow Interval(MonthDayNano) (BigQuery INTERVAL), \
                 which is not supported by Polars Arrow FFI \
                 (see https://github.com/pola-rs/polars/issues/30920 and \
                 https://github.com/pola-rs/polars/issues/30921)"
            )))
        },
        ArrowDataType::Time32(unit)
            if validate_ffi_types
                && !matches!(
                    unit,
                    polars_arrow::datatypes::TimeUnit::Second
                        | polars_arrow::datatypes::TimeUnit::Millisecond
                ) =>
        {
            Err(BigQueryError::Protocol(format!(
                "column `{path}` uses invalid Arrow Time32({unit:?}) unit"
            )))
        },
        ArrowDataType::Time64(unit)
            if validate_ffi_types
                && !matches!(
                    unit,
                    polars_arrow::datatypes::TimeUnit::Microsecond
                        | polars_arrow::datatypes::TimeUnit::Nanosecond
                ) =>
        {
            Err(BigQueryError::Protocol(format!(
                "column `{path}` uses invalid Arrow Time64({unit:?}) unit"
            )))
        },
        ArrowDataType::Timestamp(_, Some(tz))
            if validate_ffi_types && tz.as_bytes().contains(&0) =>
        {
            Err(BigQueryError::Protocol(format!(
                "column `{path}` timestamp timezone contains an interior NUL byte, which cannot be exported over Arrow C Data Interface FFI"
            )))
        },
        ArrowDataType::List(inner)
        | ArrowDataType::LargeList(inner)
        | ArrowDataType::FixedSizeList(inner, _)
        | ArrowDataType::Map(inner, _) => {
            validate_arrow_field(inner, path, depth + 1, validate_ffi_types)
        },
        ArrowDataType::Struct(fields) => {
            for (idx, field) in fields.iter().enumerate() {
                let child_path = child_field_path(path, field, idx);
                validate_arrow_field(field, &child_path, depth + 1, validate_ffi_types)?;
            }
            Ok(())
        },
        ArrowDataType::Dictionary(_, values, _) => {
            validate_supported_arrow_dtype(values, path, depth + 1, validate_ffi_types)
        },
        ArrowDataType::Union(union) => {
            if validate_ffi_types
                && (union.fields.is_empty()
                    || union.ids.as_ref().is_some_and(|ids| ids.is_empty()))
            {
                return Err(BigQueryError::UnsupportedType(format!(
                    "column `{path}` uses an empty Arrow Union, which is not supported by Arrow C Data Interface FFI export"
                )));
            }
            for (idx, field) in union.fields.iter().enumerate() {
                let child_path = child_field_path(path, field, idx);
                validate_arrow_field(field, &child_path, depth + 1, validate_ffi_types)?;
            }
            Ok(())
        },
        ArrowDataType::Extension(ext) => {
            validate_supported_arrow_dtype(&ext.inner, path, depth + 1, validate_ffi_types)
        },
        _ => Ok(()),
    }
}

/// Validates that an [`ArrowSchema`] does not contain any types known to panic or abort
/// when decoded from Arrow IPC or exported over the `polars-arrow` Arrow C Data Interface FFI layer.
pub fn validate_ffi_schema(schema: &ArrowSchema) -> Result<(), BigQueryError> {
    for (_, field) in schema.iter() {
        validate_arrow_field(field, field.name.as_str(), 0, true)?;
    }
    Ok(())
}

/// Parses IPC-serialized Arrow schema bytes into [`StreamMetadata`] using [`SchemaDecodeOptions`].
pub(crate) fn parse_arrow_schema_with_options(
    schema_bytes: &[u8],
    options: SchemaDecodeOptions,
) -> Result<StreamMetadata, BigQueryError> {
    // Defense in depth: `polars-arrow`'s IPC schema reader uses `todo!()` for some newer
    // Arrow logical types (`RunEndEncoded`, `ListView`, `LargeListView`). Catch unwinds so
    // an unexpected IPC payload never crashes the Tokio worker or Python process.
    let metadata = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut cursor = Cursor::new(schema_bytes);
        read_stream_metadata(&mut cursor)
    })) {
        Ok(res) => res?,
        Err(payload) => {
            let msg = format_panic_payload(payload);
            return Err(BigQueryError::Protocol(format!(
                "Arrow IPC schema decoder panicked: {msg}"
            )));
        },
    };
    for (_, field) in metadata.schema.iter() {
        validate_arrow_field(field, field.name.as_str(), 0, options.validate_ffi_types)?;
    }
    Ok(metadata)
}

/// Convenience wrapper using default [`SchemaDecodeOptions`] for unit tests.
#[cfg(test)]
pub(crate) fn parse_arrow_schema(schema_bytes: &[u8]) -> Result<StreamMetadata, BigQueryError> {
    parse_arrow_schema_with_options(schema_bytes, SchemaDecodeOptions::default())
}

/// Validates that an optional serialized Arrow record batch payload is consistent
/// with `expected_row_count`, returning `Ok(None)` for empty/heartbeat payloads.
pub(crate) fn validate_non_empty_arrow_bytes(
    serialized_record_batch: Option<bytes::Bytes>,
    expected_row_count: i64,
) -> Result<Option<bytes::Bytes>, BigQueryError> {
    match serialized_record_batch.filter(|b| !b.is_empty()) {
        Some(bytes) => Ok(Some(bytes)),
        None if expected_row_count == 0 => Ok(None),
        None => Err(BigQueryError::Protocol(format!(
            "Row count mismatch: gRPC protobuf reported {expected_row_count} rows, but Arrow IPC decoded 0 rows"
        ))),
    }
}

/// Decodes an Arrow IPC record batch payload and verifies its total row count
/// matches `expected_row_count`.
pub(crate) fn decode_arrow_record_batches(
    serialized_record_batch: bytes::Bytes,
    expected_row_count: i64,
    metadata: StreamMetadata,
) -> Result<Vec<RecordBatch>, BigQueryError> {
    let cursor = Cursor::new(serialized_record_batch);
    let mut reader = StreamReader::new(cursor, metadata, None);
    let mut batches = Vec::with_capacity(1);
    let mut actual_rows: i64 = 0;

    for result in &mut reader {
        match result {
            Ok(StreamState::Some(batch)) => {
                actual_rows = actual_rows.checked_add(batch.len() as i64).ok_or_else(|| {
                    BigQueryError::Protocol("Decoded batch row count overflow".into())
                })?;
                batches.push(batch);
            },
            Ok(StreamState::Waiting) => break,
            Err(e) => return Err(BigQueryError::Arrow(e)),
        }
    }

    if actual_rows != expected_row_count {
        return Err(BigQueryError::Protocol(format!(
            "Row count mismatch: gRPC protobuf reported {} rows, but Arrow IPC decoded {} rows",
            expected_row_count, actual_rows
        )));
    }

    Ok(batches)
}

pub(crate) struct DecodeJob {
    pub(crate) serialized_record_batch: bytes::Bytes,
    pub(crate) expected_row_count: i64,
    pub(crate) metadata: Arc<StreamMetadata>,
    pub(crate) permit: tokio::sync::mpsc::OwnedPermit<Result<RecordBatch, BigQueryError>>,
    pub(crate) cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
    pub(crate) cancel_rx: tokio::sync::watch::Receiver<bool>,
}

impl DecodeJob {
    fn run(self) {
        let Self {
            serialized_record_batch,
            expected_row_count,
            metadata,
            permit,
            cancel_tx,
            cancel_rx,
        } = self;

        // If the stream task or receiver was cancelled while this job was queued,
        // skip decompression immediately.
        if *cancel_rx.borrow() {
            return;
        }

        let res = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decode_arrow_record_batches(
                serialized_record_batch,
                expected_row_count,
                Arc::unwrap_or_clone(metadata),
            )
        })) {
            Ok(inner) => inner,
            Err(payload) => {
                let msg = format_panic_payload(payload);
                Err(BigQueryError::Protocol(format!(
                    "Arrow IPC decoder panicked: {msg}"
                )))
            },
        };

        if *cancel_rx.borrow() {
            return;
        }

        match res {
            Ok(batches) => {
                let mut iter = batches.into_iter();
                if let Some(first) = iter.next() {
                    let tx = permit.send(Ok(first));
                    for batch in iter {
                        if *cancel_rx.borrow() || tx.is_closed() {
                            return;
                        }
                        if tx.blocking_send(Ok(batch)).is_err() {
                            return;
                        }
                    }
                }
            },
            Err(err) => {
                let _ = cancel_tx.send_replace(true);
                let _ = permit.send(Err(err));
            },
        }
    }
}

struct OrderedQueueState {
    queue: VecDeque<DecodeJob>,
    running: bool,
}

/// A FIFO decode queue that keeps all decode jobs for a `maintain_order = true`
/// read in a single worker queue without work-stealing across workers.
pub(crate) struct OrderedDecodeQueue {
    state: Mutex<OrderedQueueState>,
}

impl OrderedDecodeQueue {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(OrderedQueueState {
                queue: VecDeque::new(),
                running: false,
            }),
        })
    }

    fn push(self: &Arc<Self>, job: DecodeJob) {
        let should_spawn = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.queue.push_back(job);
            if !state.running {
                state.running = true;
                true
            } else {
                false
            }
        };

        if should_spawn {
            let queue = Arc::clone(self);
            rayon::spawn_fifo(move || {
                queue.drain();
            });
        }
    }

    fn drain_on_abort(state: &mut OrderedQueueState) {
        state.running = false;
        let orphaned = std::mem::take(&mut state.queue);
        let mut reported = false;
        for job in orphaned {
            let _ = job.cancel_tx.send_replace(true);
            if !reported {
                let _ = job.permit.send(Err(BigQueryError::Protocol(
                    "Arrow IPC ordered decode worker panicked unexpectedly".into(),
                )));
                reported = true;
            }
            // Remaining permits drop here, releasing channel capacity so producers cannot deadlock.
        }
    }

    fn drain(&self) {
        struct ResetRunningOnPanic<'a> {
            state: &'a Mutex<OrderedQueueState>,
            disarmed: bool,
        }
        impl Drop for ResetRunningOnPanic<'_> {
            fn drop(&mut self) {
                if !self.disarmed {
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    OrderedDecodeQueue::drain_on_abort(&mut state);
                }
            }
        }

        let mut guard = ResetRunningOnPanic {
            state: &self.state,
            disarmed: false,
        };
        loop {
            let next_job = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                match state.queue.pop_front() {
                    Some(job) => job,
                    None => {
                        state.running = false;
                        guard.disarmed = true;
                        return;
                    },
                }
            };
            next_job.run();
        }
    }
}

#[derive(Clone)]
pub(crate) enum DecodeQueue {
    /// Uses Rayon's global work-stealing pool when `maintain_order = false`.
    WorkStealing,
    /// Keeps all decode jobs in a single sequential queue when `maintain_order = true`.
    Ordered(Arc<OrderedDecodeQueue>),
}

impl DecodeQueue {
    pub(crate) fn new(maintain_order: bool) -> Self {
        if maintain_order {
            Self::Ordered(OrderedDecodeQueue::new())
        } else {
            Self::WorkStealing
        }
    }

    pub(crate) fn push(&self, job: DecodeJob) {
        match self {
            Self::WorkStealing => {
                rayon::spawn_fifo(move || job.run());
            },
            Self::Ordered(queue) => {
                queue.push(job);
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use polars_arrow::array::Int64Array;

    use super::*;
    use crate::bigquery_read_stream::tests::create_test_arrow_payload_with_range;

    #[tokio::test]
    async fn test_ordered_decode_queue_preserves_batch_order() {
        let queue = DecodeQueue::new(true);

        // Earlier batches are much larger than later ones so work-stealing would finish
        // the smaller batches first if they were not kept in the worker's ordered queue.
        let sizes = [50_000, 1, 25_000, 1, 10_000, 1, 5_000, 1];
        let mut schema_bytes = Vec::new();
        let mut payloads = Vec::with_capacity(sizes.len());
        let mut next_start = 0_i64;

        for &count in &sizes {
            let (s_bytes, batch_bytes) = create_test_arrow_payload_with_range(next_start, count);
            if schema_bytes.is_empty() {
                schema_bytes = s_bytes;
            }
            next_start += count as i64;
            payloads.push((bytes::Bytes::from(batch_bytes), count as i64));
        }

        let metadata = Arc::new(parse_arrow_schema(&schema_bytes).unwrap());
        let (tx, mut rx) = tokio::sync::mpsc::channel(sizes.len() + 2);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        for (serialized_record_batch, expected_row_count) in payloads {
            let permit = tx.clone().reserve_owned().await.unwrap();
            queue.push(DecodeJob {
                serialized_record_batch,
                expected_row_count,
                metadata: Arc::clone(&metadata),
                permit,
                cancel_tx: Arc::clone(&cancel_tx),
                cancel_rx: cancel_rx.clone(),
            });
        }
        drop(tx);

        let mut expected_next = 0_i64;
        let mut received_sizes = Vec::new();
        while let Some(res) = rx.recv().await {
            let batch = res.unwrap();
            received_sizes.push(batch.len());
            let col = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for &val in col.values().as_slice() {
                assert_eq!(val, expected_next);
                expected_next += 1;
            }
        }
        assert_eq!(received_sizes, sizes);
        assert_eq!(expected_next, next_start);
    }

    #[tokio::test]
    async fn test_work_stealing_decode_queue_decodes_all_batches() {
        let queue = DecodeQueue::new(false);

        let sizes = [20_000, 10, 15_000, 10, 10_000, 10];
        let mut schema_bytes = Vec::new();
        let mut payloads = Vec::with_capacity(sizes.len());
        let mut next_start = 0_i64;

        for &count in &sizes {
            let (s_bytes, batch_bytes) = create_test_arrow_payload_with_range(next_start, count);
            if schema_bytes.is_empty() {
                schema_bytes = s_bytes;
            }
            next_start += count as i64;
            payloads.push((bytes::Bytes::from(batch_bytes), count as i64));
        }

        let metadata = Arc::new(parse_arrow_schema(&schema_bytes).unwrap());
        let (tx, mut rx) = tokio::sync::mpsc::channel(sizes.len() + 2);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        for (serialized_record_batch, expected_row_count) in payloads {
            let permit = tx.clone().reserve_owned().await.unwrap();
            queue.push(DecodeJob {
                serialized_record_batch,
                expected_row_count,
                metadata: Arc::clone(&metadata),
                permit,
                cancel_tx: Arc::clone(&cancel_tx),
                cancel_rx: cancel_rx.clone(),
            });
        }
        drop(tx);

        let mut all_values = Vec::with_capacity(next_start as usize);
        while let Some(res) = rx.recv().await {
            let batch = res.unwrap();
            let col = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(col.values().as_slice());
        }
        all_values.sort_unstable();
        let expected: Vec<i64> = (0..next_start).collect();
        assert_eq!(all_values, expected);
    }

    #[tokio::test]
    async fn test_decode_job_error_signals_cancellation_and_pushes_err() {
        let queue = DecodeQueue::new(true);

        let (schema_bytes, valid_batch) = create_test_arrow_payload_with_range(0, 5);
        let metadata = Arc::new(parse_arrow_schema(&schema_bytes).unwrap());

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        // Enqueue a job with a mismatched expected_row_count followed by a valid job.
        let permit1 = tx.clone().reserve_owned().await.unwrap();
        queue.push(DecodeJob {
            serialized_record_batch: bytes::Bytes::from(valid_batch.clone()),
            expected_row_count: 999,
            metadata: Arc::clone(&metadata),
            permit: permit1,
            cancel_tx: Arc::clone(&cancel_tx),
            cancel_rx: cancel_rx.clone(),
        });

        let permit2 = tx.clone().reserve_owned().await.unwrap();
        queue.push(DecodeJob {
            serialized_record_batch: bytes::Bytes::from(valid_batch),
            expected_row_count: 5,
            metadata,
            permit: permit2,
            cancel_tx: Arc::clone(&cancel_tx),
            cancel_rx: cancel_rx.clone(),
        });
        drop(tx);

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, Err(BigQueryError::Protocol(_))));
        assert!(
            *cancel_tx.borrow(),
            "decode failure should signal cancellation"
        );
        // The second job should observe cancellation and drop its permit without sending.
        assert!(rx.recv().await.is_none());
    }

    fn serialize_arrow_schema(schema: &ArrowSchema) -> Vec<u8> {
        use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};

        let mut schema_bytes = Vec::new();
        {
            let mut writer =
                StreamWriter::new(&mut schema_bytes, WriteOptions { compression: None });
            writer.start(schema, None).unwrap();
        }
        schema_bytes
    }

    #[test]
    fn test_parse_arrow_schema_accepts_supported_types() {
        use polars_arrow::datatypes::{Field, TimeUnit};

        let schema = ArrowSchema::from_iter(vec![
            Field::new("id".into(), ArrowDataType::Int64, true),
            Field::new(
                "ts".into(),
                ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
            Field::new(
                "items".into(),
                ArrowDataType::List(Box::new(Field::new(
                    "item".into(),
                    ArrowDataType::Utf8,
                    true,
                ))),
                true,
            ),
            Field::new(
                "meta".into(),
                ArrowDataType::Struct(vec![Field::new("ok".into(), ArrowDataType::Boolean, true)]),
                true,
            ),
        ]);

        let bytes = serialize_arrow_schema(&schema);
        let metadata = parse_arrow_schema(&bytes).expect("supported schema should parse");
        assert_eq!(metadata.schema.len(), 4);
    }

    #[test]
    fn test_parse_arrow_schema_rejects_top_level_month_day_nano() {
        use polars_arrow::datatypes::Field;

        let schema = ArrowSchema::from_iter(vec![
            Field::new("id".into(), ArrowDataType::Int64, true),
            Field::new(
                "dur".into(),
                ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                true,
            ),
        ]);

        let bytes = serialize_arrow_schema(&schema);
        let err = parse_arrow_schema(&bytes).expect_err("MonthDayNano should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        let msg = err.to_string();
        assert!(msg.contains("dur"), "unexpected error: {msg}");
        assert!(
            msg.contains("https://github.com/pola-rs/polars/issues/30920"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("https://github.com/pola-rs/polars/issues/30921"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_parse_arrow_schema_rejects_nested_list_and_struct_month_day_nano() {
        use polars_arrow::datatypes::{ExtensionType, Field};

        // List-nested MonthDayNano
        let list_schema = ArrowSchema::from_iter(vec![Field::new(
            "dur_list".into(),
            ArrowDataType::List(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                true,
            ))),
            true,
        )]);
        let err = parse_arrow_schema(&serialize_arrow_schema(&list_schema))
            .expect_err("List<MonthDayNano> should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("dur_list"));

        // LargeList-nested MonthDayNano
        let large_list_schema = ArrowSchema::from_iter(vec![Field::new(
            "dur_large_list".into(),
            ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                true,
            ))),
            true,
        )]);
        let err = parse_arrow_schema(&serialize_arrow_schema(&large_list_schema))
            .expect_err("LargeList<MonthDayNano> should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("dur_large_list"));

        // FixedSizeList-nested MonthDayNano
        let fixed_list_schema = ArrowSchema::from_iter(vec![Field::new(
            "dur_fixed_list".into(),
            ArrowDataType::FixedSizeList(
                Box::new(Field::new(
                    "item".into(),
                    ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                    true,
                )),
                2,
            ),
            true,
        )]);
        let err = parse_arrow_schema(&serialize_arrow_schema(&fixed_list_schema))
            .expect_err("FixedSizeList<MonthDayNano> should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("dur_fixed_list"));

        // Struct-nested MonthDayNano
        let struct_schema = ArrowSchema::from_iter(vec![Field::new(
            "meta".into(),
            ArrowDataType::Struct(vec![
                Field::new("ok".into(), ArrowDataType::Int64, true),
                Field::new(
                    "bad".into(),
                    ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                    true,
                ),
            ]),
            true,
        )]);
        let err = parse_arrow_schema(&serialize_arrow_schema(&struct_schema))
            .expect_err("Struct<bad: MonthDayNano> should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("meta.bad"));

        // Map-nested MonthDayNano
        let map_schema = ArrowSchema::from_iter(vec![Field::new(
            "dur_map".into(),
            ArrowDataType::Map(
                Box::new(Field::new(
                    "entries".into(),
                    ArrowDataType::Struct(vec![
                        Field::new("key".into(), ArrowDataType::Utf8, false),
                        Field::new(
                            "value".into(),
                            ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                            true,
                        ),
                    ]),
                    false,
                )),
                false,
            ),
            true,
        )]);
        let err = parse_arrow_schema(&serialize_arrow_schema(&map_schema))
            .expect_err("Map<Utf8, MonthDayNano> should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("dur_map.value"));

        // Extension-nested MonthDayNano (checked directly on ArrowSchema)
        let ext_schema = ArrowSchema::from_iter(vec![Field::new(
            "dur_ext".into(),
            ArrowDataType::Extension(Box::new(ExtensionType {
                name: "custom:interval".into(),
                inner: ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                metadata: None,
            })),
            true,
        )]);
        let err = validate_ffi_schema(&ext_schema)
            .expect_err("Extension<MonthDayNano> should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("dur_ext"));
    }

    #[test]
    fn test_validate_dictionary_union_month_day_millis_unknown_and_depth() {
        use polars_arrow::datatypes::{IntegerType, UnionMode, UnionType};

        // Dictionary<Int32, Interval(MonthDayNano)> rejected
        let dict_bad = ArrowSchema::from_iter(vec![Field::new(
            "d_bad".into(),
            ArrowDataType::Dictionary(
                IntegerType::Int32,
                Box::new(ArrowDataType::Interval(IntervalUnit::MonthDayNano)),
                false,
            ),
            true,
        )]);
        assert!(matches!(
            validate_ffi_schema(&dict_bad),
            Err(BigQueryError::UnsupportedType(_))
        ));

        // Union containing Interval(MonthDayNano) rejected
        let union_bad = ArrowSchema::from_iter(vec![Field::new(
            "u_bad".into(),
            ArrowDataType::Union(Box::new(UnionType {
                fields: vec![Field::new(
                    "dur".into(),
                    ArrowDataType::Interval(IntervalUnit::MonthDayNano),
                    true,
                )],
                ids: Some(vec![0]),
                mode: UnionMode::Sparse,
            })),
            true,
        )]);
        let err = validate_ffi_schema(&union_bad).unwrap_err();
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("u_bad.dur"));

        // Interval(MonthDayMillis) and Unknown rejected
        let millis_schema = ArrowSchema::from_iter(vec![Field::new(
            "ms".into(),
            ArrowDataType::Interval(IntervalUnit::MonthDayMillis),
            true,
        )]);
        assert!(matches!(
            validate_ffi_schema(&millis_schema),
            Err(BigQueryError::UnsupportedType(_))
        ));
        let unknown_schema =
            ArrowSchema::from_iter(vec![Field::new("unk".into(), ArrowDataType::Unknown, true)]);
        assert!(matches!(
            validate_ffi_schema(&unknown_schema),
            Err(BigQueryError::UnsupportedType(_))
        ));

        // Excessive nesting depth is rejected cleanly with Protocol error instead of stack overflow
        let mut deep_dtype = ArrowDataType::Int64;
        for _ in 0..=MAX_SCHEMA_NESTING_DEPTH + 1 {
            deep_dtype = ArrowDataType::List(Box::new(Field::new("item".into(), deep_dtype, true)));
        }
        let deep_schema = ArrowSchema::from_iter(vec![Field::new("deep".into(), deep_dtype, true)]);
        let depth_err = validate_ffi_schema(&deep_schema).unwrap_err();
        assert!(matches!(depth_err, BigQueryError::Protocol(_)));

        // Disabling validate_ffi_types allows pure-Rust consumers to inspect MonthDayNano IPC
        let interval_schema = ArrowSchema::from_iter(vec![Field::new(
            "dur".into(),
            ArrowDataType::Interval(IntervalUnit::MonthDayNano),
            true,
        )]);
        let raw_bytes = serialize_arrow_schema(&interval_schema);
        let parsed_unchecked = parse_arrow_schema_with_options(
            &raw_bytes,
            SchemaDecodeOptions {
                validate_ffi_types: false,
            },
        )
        .unwrap();
        assert_eq!(
            parsed_unchecked.schema.get("dur").unwrap().dtype(),
            &ArrowDataType::Interval(IntervalUnit::MonthDayNano)
        );
    }

    #[test]
    fn test_validate_rejects_bignumeric_decimal256_daytime_bad_time_empty_union_and_nul() {
        use polars_arrow::datatypes::{TimeUnit, UnionMode, UnionType};

        // Decimal256 (BigQuery BIGNUMERIC) must be rejected unconditionally even when
        // `validate_ffi_types == false`, because `polars-arrow`'s IPC decoder panics on Int256.
        let dec256_schema = ArrowSchema::from_iter(vec![Field::new(
            "amount".into(),
            ArrowDataType::Decimal256(76, 38),
            true,
        )]);
        let dec256_bytes = serialize_arrow_schema(&dec256_schema);
        let err = parse_arrow_schema_with_options(
            &dec256_bytes,
            SchemaDecodeOptions {
                validate_ffi_types: false,
            },
        )
        .expect_err("Decimal256 should be rejected even with validate_ffi_types=false");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("BIGNUMERIC"));
        assert!(err.to_string().contains("amount"));

        // Nested Decimal256 in Struct and List rejected via validate_ffi_schema
        let nested_dec256 = ArrowSchema::from_iter(vec![Field::new(
            "meta".into(),
            ArrowDataType::Struct(vec![Field::new(
                "vals".into(),
                ArrowDataType::List(Box::new(Field::new(
                    "item".into(),
                    ArrowDataType::Decimal256(76, 38),
                    true,
                ))),
                true,
            )]),
            true,
        )]);
        let err = validate_ffi_schema(&nested_dec256).unwrap_err();
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("meta.vals"));

        // Interval(DayTime) rejected unconditionally even with validate_ffi_types=false
        let daytime_schema = ArrowSchema::from_iter(vec![Field::new(
            "dt_iv".into(),
            ArrowDataType::Interval(IntervalUnit::DayTime),
            true,
        )]);
        let daytime_err = parse_arrow_schema_with_options(
            &serialize_arrow_schema(&daytime_schema),
            SchemaDecodeOptions {
                validate_ffi_types: false,
            },
        )
        .unwrap_err();
        assert!(matches!(daytime_err, BigQueryError::UnsupportedType(_)));

        // Mismatched Time32/Time64 units rejected before FFI export (avoids unreachable!() panic)
        let bad_time32 = ArrowSchema::from_iter(vec![Field::new(
            "t32".into(),
            ArrowDataType::Time32(TimeUnit::Microsecond),
            true,
        )]);
        assert!(matches!(
            validate_ffi_schema(&bad_time32),
            Err(BigQueryError::Protocol(_))
        ));
        let bad_time64 = ArrowSchema::from_iter(vec![Field::new(
            "t64".into(),
            ArrowDataType::Time64(TimeUnit::Second),
            true,
        )]);
        assert!(matches!(
            validate_ffi_schema(&bad_time64),
            Err(BigQueryError::Protocol(_))
        ));

        // Empty Union rejected before FFI export (avoids ids.len() - 1 underflow)
        let empty_union = ArrowSchema::from_iter(vec![Field::new(
            "empty_u".into(),
            ArrowDataType::Union(Box::new(UnionType {
                fields: vec![],
                ids: None,
                mode: UnionMode::Sparse,
            })),
            true,
        )]);
        assert!(matches!(
            validate_ffi_schema(&empty_union),
            Err(BigQueryError::UnsupportedType(_))
        ));

        // Interior NUL bytes in field name or Timestamp timezone rejected before CString::new().unwrap()
        let nul_name = ArrowSchema::from_iter(vec![Field::new(
            "bad\0col".into(),
            ArrowDataType::Int64,
            true,
        )]);
        assert!(matches!(
            validate_ffi_schema(&nul_name),
            Err(BigQueryError::Protocol(_))
        ));
        let nul_tz = ArrowSchema::from_iter(vec![Field::new(
            "ts".into(),
            ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UT\0C".into())),
            true,
        )]);
        assert!(matches!(
            validate_ffi_schema(&nul_tz),
            Err(BigQueryError::Protocol(_))
        ));
    }

    #[tokio::test]
    async fn test_ordered_decode_queue_drain_on_abort_releases_permits_and_signals_error() {
        let queue = OrderedDecodeQueue::new();
        let (schema_bytes, valid_batch) = create_test_arrow_payload_with_range(0, 2);
        let metadata = Arc::new(parse_arrow_schema(&schema_bytes).unwrap());

        // Channel capacity = 2; fill all permits inside `queue` so producer would block
        // if `drain_on_abort` failed to drop orphaned permits.
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        let permit1 = tx.clone().reserve_owned().await.unwrap();
        let permit2 = tx.clone().reserve_owned().await.unwrap();
        {
            let mut state = queue.state.lock().unwrap();
            state.running = true;
            for permit in [permit1, permit2] {
                state.queue.push_back(DecodeJob {
                    serialized_record_batch: bytes::Bytes::from(valid_batch.clone()),
                    expected_row_count: 2,
                    metadata: Arc::clone(&metadata),
                    permit,
                    cancel_tx: Arc::clone(&cancel_tx),
                    cancel_rx: cancel_rx.clone(),
                });
            }
            OrderedDecodeQueue::drain_on_abort(&mut state);
            assert!(!state.running);
            assert!(state.queue.is_empty());
        }
        drop(tx);

        assert!(
            *cancel_tx.borrow(),
            "abort should signal cancellation to sibling readers"
        );
        let first = rx.recv().await.expect("expected fatal error");
        assert!(matches!(first, Err(BigQueryError::Protocol(_))));
        // Second orphaned permit was dropped cleanly; channel closes without hanging.
        assert!(rx.recv().await.is_none());
    }
}
