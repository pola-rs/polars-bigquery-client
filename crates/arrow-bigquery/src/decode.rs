use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field, IntervalUnit};
use polars_arrow::io::ipc::read::{
    read_stream_metadata, StreamMetadata, StreamReader, StreamState,
};
use polars_arrow::record_batch::RecordBatch;

use crate::BigQueryError;

const BQ_DATETIME_EXTENSION_NAME: &str = "google:sqlType:datetime";

/// Maximum recursive type-nesting depth inspected by `decode.rs` helpers.
///
/// BigQuery limits `STRUCT`/`RECORD` nesting to 15 levels, so 64 comfortably covers
/// every valid BigQuery schema (including nested containers and extension wrappers)
/// while guarding the Rust call stack against malformed IPC metadata.
const MAX_SCHEMA_NESTING_DEPTH: usize = 64;

fn check_nesting_depth(depth: usize, path: &str) -> Result<(), BigQueryError> {
    if depth > MAX_SCHEMA_NESTING_DEPTH {
        return Err(BigQueryError::Protocol(format!(
            "Arrow schema nesting depth for `{path}` exceeds maximum supported depth ({MAX_SCHEMA_NESTING_DEPTH})"
        )));
    }
    Ok(())
}

fn is_expected_datetime_storage(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Timestamp(polars_arrow::datatypes::TimeUnit::Microsecond, None) => true,
        ArrowDataType::List(inner)
        | ArrowDataType::LargeList(inner)
        | ArrowDataType::FixedSizeList(inner, _) => {
            matches!(
                inner.dtype(),
                ArrowDataType::Timestamp(polars_arrow::datatypes::TimeUnit::Microsecond, None)
            )
        },
        _ => false,
    }
}

fn child_field_path(parent: &str, field: &Field, idx: usize) -> String {
    if field.name.is_empty() {
        format!("{parent}[{idx}]")
    } else {
        format!("{parent}.{}", field.name)
    }
}

fn field_metadata_has_datetime_extension(field: &Field) -> bool {
    field.metadata.as_ref().is_some_and(|meta| {
        meta.get("ARROW:extension:name").map(|v| v.as_str()) == Some(BQ_DATETIME_EXTENSION_NAME)
    })
}

fn strip_datetime_extension_dtype(
    dtype: ArrowDataType,
    path: &str,
    depth: usize,
) -> Result<ArrowDataType, BigQueryError> {
    check_nesting_depth(depth, path)?;
    match dtype {
        ArrowDataType::Extension(mut ext) => {
            let inner = strip_datetime_extension_dtype(ext.inner, path, depth + 1)?;
            if ext.name.as_str() == BQ_DATETIME_EXTENSION_NAME
                && is_expected_datetime_storage(&inner)
            {
                Ok(inner)
            } else {
                ext.inner = inner;
                Ok(ArrowDataType::Extension(ext))
            }
        },
        ArrowDataType::List(inner) => Ok(ArrowDataType::List(Box::new(
            strip_datetime_extension_field(*inner, path, depth + 1)?,
        ))),
        ArrowDataType::LargeList(inner) => Ok(ArrowDataType::LargeList(Box::new(
            strip_datetime_extension_field(*inner, path, depth + 1)?,
        ))),
        ArrowDataType::FixedSizeList(inner, size) => Ok(ArrowDataType::FixedSizeList(
            Box::new(strip_datetime_extension_field(*inner, path, depth + 1)?),
            size,
        )),
        ArrowDataType::Map(inner, ordered) => Ok(ArrowDataType::Map(
            Box::new(strip_datetime_extension_field(*inner, path, depth + 1)?),
            ordered,
        )),
        ArrowDataType::Struct(fields) => {
            let mut stripped = Vec::with_capacity(fields.len());
            for (idx, field) in fields.into_iter().enumerate() {
                let child_path = child_field_path(path, &field, idx);
                stripped.push(strip_datetime_extension_field(
                    field,
                    &child_path,
                    depth + 1,
                )?);
            }
            Ok(ArrowDataType::Struct(stripped))
        },
        ArrowDataType::Dictionary(key, values, is_ordered) => Ok(ArrowDataType::Dictionary(
            key,
            Box::new(strip_datetime_extension_dtype(*values, path, depth + 1)?),
            is_ordered,
        )),
        ArrowDataType::Union(mut union) => {
            let mut stripped = Vec::with_capacity(union.fields.len());
            for (idx, field) in union.fields.into_iter().enumerate() {
                let child_path = child_field_path(path, &field, idx);
                stripped.push(strip_datetime_extension_field(
                    field,
                    &child_path,
                    depth + 1,
                )?);
            }
            union.fields = stripped;
            Ok(ArrowDataType::Union(union))
        },
        other => Ok(other),
    }
}

fn strip_datetime_extension_field(
    mut field: Field,
    path: &str,
    depth: usize,
) -> Result<Field, BigQueryError> {
    field.dtype = strip_datetime_extension_dtype(field.dtype, path, depth)?;
    if field_metadata_has_datetime_extension(&field) && is_expected_datetime_storage(&field.dtype) {
        if let Some(meta) = field.metadata.as_ref() {
            let mut updated = (**meta).clone();
            updated.remove("ARROW:extension:name");
            updated.remove("ARROW:extension:metadata");
            field.metadata = if updated.is_empty() {
                None
            } else {
                Some(Arc::new(updated))
            };
        }
    }
    Ok(field)
}

fn strip_datetime_extensions(schema: ArrowSchema) -> Result<ArrowSchema, BigQueryError> {
    let mut fields = Vec::with_capacity(schema.len());
    for (_, field) in schema.into_iter() {
        let name = field.name.to_string();
        fields.push(strip_datetime_extension_field(field, &name, 0)?);
    }
    Ok(ArrowSchema::from_iter(fields))
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

fn validate_arrow_field(
    field: &Field,
    path: &str,
    depth: usize,
    has_datetime_ext: &mut bool,
) -> Result<(), BigQueryError> {
    if field_metadata_has_datetime_extension(field) && is_expected_datetime_storage(field.dtype()) {
        *has_datetime_ext = true;
    }
    validate_supported_arrow_dtype(field.dtype(), path, depth, has_datetime_ext)
}

fn validate_supported_arrow_dtype(
    dtype: &ArrowDataType,
    path: &str,
    depth: usize,
    has_datetime_ext: &mut bool,
) -> Result<(), BigQueryError> {
    check_nesting_depth(depth, path)?;
    match dtype {
        // Types that `polars-arrow`'s IPC `StreamReader` (`deserialize::read`) cannot decode
        // without panicking inside `with_match_primitive_type_full!` / `to_physical_type()`.
        ArrowDataType::Decimal256(precision, scale) => {
            let hint = format_sql_cast_hint(path);
            Err(BigQueryError::UnsupportedType(format!(
                "column `{path}` uses Arrow Decimal256({precision}, {scale}) (BigQuery BIGNUMERIC), \
                 which cannot be decoded by polars-arrow. {hint}"
            )))
        },
        // Decodable in IPC but panics/aborts during `polars-arrow::ffi::export_iterator`.
        ArrowDataType::Interval(IntervalUnit::MonthDayNano) => {
            Err(BigQueryError::UnsupportedType(format!(
                "column `{path}` uses Arrow Interval(MonthDayNano) (BigQuery INTERVAL), \
                 which is not supported by Polars Arrow FFI \
                 (see https://github.com/pola-rs/polars/issues/30920 and \
                 https://github.com/pola-rs/polars/issues/30921)"
            )))
        },
        ArrowDataType::List(inner)
        | ArrowDataType::LargeList(inner)
        | ArrowDataType::FixedSizeList(inner, _)
        | ArrowDataType::Map(inner, _) => {
            validate_arrow_field(inner, path, depth + 1, has_datetime_ext)
        },
        ArrowDataType::Struct(fields) => {
            for (idx, field) in fields.iter().enumerate() {
                let child_path = child_field_path(path, field, idx);
                validate_arrow_field(field, &child_path, depth + 1, has_datetime_ext)?;
            }
            Ok(())
        },
        ArrowDataType::Dictionary(_, values, _) => {
            validate_supported_arrow_dtype(values, path, depth + 1, has_datetime_ext)
        },
        ArrowDataType::Union(union) => {
            for (idx, field) in union.fields.iter().enumerate() {
                let child_path = child_field_path(path, field, idx);
                validate_arrow_field(field, &child_path, depth + 1, has_datetime_ext)?;
            }
            Ok(())
        },
        ArrowDataType::Extension(ext) => {
            if ext.name.as_str() == BQ_DATETIME_EXTENSION_NAME
                && is_expected_datetime_storage(&ext.inner)
            {
                *has_datetime_ext = true;
            }
            validate_supported_arrow_dtype(&ext.inner, path, depth + 1, has_datetime_ext)
        },
        _ => Ok(()),
    }
}

/// Validates that an [`ArrowSchema`] does not contain any types known to panic or abort
/// when decoded from Arrow IPC or exported over the `polars-arrow` Arrow C Data Interface FFI layer.
pub fn validate_ffi_schema(schema: &ArrowSchema) -> Result<(), BigQueryError> {
    let mut ignored = false;
    for (_, field) in schema.iter() {
        validate_arrow_field(field, field.name.as_str(), 0, &mut ignored)?;
    }
    Ok(())
}

/// Parses IPC-serialized Arrow schema bytes into [`StreamMetadata`], validates
/// that all fields can be decoded and exported over Arrow C Data Interface FFI,
/// and strips `google:sqlType:datetime` extension metadata on naive microsecond
/// `Timestamp(Microsecond, None)` and `ARRAY<DATETIME>` fields.
pub(crate) fn parse_arrow_schema(schema_bytes: &[u8]) -> Result<StreamMetadata, BigQueryError> {
    // Defense in depth: `polars-arrow`'s IPC schema reader uses `todo!()` for some newer
    // Arrow logical types (`RunEndEncoded`, `ListView`, `LargeListView`). Catch unwinds so
    // an unexpected IPC payload never crashes the Tokio worker or Python process.
    let mut metadata = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
    let mut has_datetime_ext = false;
    for (_, field) in metadata.schema.iter() {
        validate_arrow_field(field, field.name.as_str(), 0, &mut has_datetime_ext)?;
    }
    if has_datetime_ext {
        metadata.schema = strip_datetime_extensions(metadata.schema)?;
    }
    Ok(metadata)
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
    fn test_validate_rejects_excessive_nesting_depth() {
        // Excessive nesting depth is rejected cleanly with Protocol error instead of stack overflow.
        let mut deep_dtype = ArrowDataType::Int64;
        for _ in 0..=MAX_SCHEMA_NESTING_DEPTH + 1 {
            deep_dtype = ArrowDataType::List(Box::new(Field::new("item".into(), deep_dtype, true)));
        }
        let deep_schema = ArrowSchema::from_iter(vec![Field::new("deep".into(), deep_dtype, true)]);
        let depth_err = validate_ffi_schema(&deep_schema).unwrap_err();
        assert!(matches!(depth_err, BigQueryError::Protocol(_)));
    }

    #[test]
    fn test_validate_rejects_bignumeric_decimal256() {
        // Decimal256 (BigQuery BIGNUMERIC) must be rejected because `polars-arrow`'s IPC decoder
        // panics on Int256.
        let dec256_schema = ArrowSchema::from_iter(vec![Field::new(
            "amount".into(),
            ArrowDataType::Decimal256(76, 38),
            true,
        )]);
        let dec256_bytes = serialize_arrow_schema(&dec256_schema);
        let err = parse_arrow_schema(&dec256_bytes)
            .expect_err("Decimal256 should be rejected during schema parse");
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
    }

    #[test]
    fn test_parse_arrow_schema_strips_datetime_extension_and_preserves_json_geography() {
        use polars_arrow::array::{Array, PrimitiveArray};
        use polars_arrow::datatypes::{ExtensionType, Metadata, TimeUnit};
        use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};

        let dt_storage = ArrowDataType::Timestamp(TimeUnit::Microsecond, None);
        let dt_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: dt_storage.clone(),
            metadata: None,
        }));
        let dt_arr_storage = ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            dt_storage.clone(),
            true,
        )));
        let dt_arr_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: dt_arr_storage.clone(),
            metadata: None,
        }));
        let json_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: "google:sqlType:json".into(),
            inner: ArrowDataType::Utf8,
            metadata: None,
        }));
        let geo_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: "google:sqlType:geography".into(),
            inner: ArrowDataType::Utf8,
            metadata: Some(r#"{"encoding": "WKT"}"#.into()),
        }));

        let mut dt_meta = Metadata::default();
        dt_meta.insert(
            "ARROW:extension:name".into(),
            BQ_DATETIME_EXTENSION_NAME.into(),
        );
        dt_meta.insert("ARROW:extension:metadata".into(), "{}".into());
        dt_meta.insert("custom_key".into(), "keep_me".into());
        let dt_field_with_meta =
            Field::new("dt_meta".into(), dt_ext.clone(), true).with_metadata(dt_meta);

        let schema = ArrowSchema::from_iter(vec![
            Field::new("dt".into(), dt_ext.clone(), true),
            dt_field_with_meta,
            Field::new("dt_arr".into(), dt_arr_ext, true),
            Field::new(
                "nested".into(),
                ArrowDataType::Struct(vec![
                    Field::new("dt".into(), dt_ext, true),
                    Field::new("json".into(), json_ext.clone(), true),
                ]),
                true,
            ),
            Field::new("json_col".into(), json_ext.clone(), true),
            Field::new("geo_col".into(), geo_ext.clone(), true),
        ]);

        let stripped = strip_datetime_extensions(schema).unwrap();
        assert_eq!(stripped.get("dt").unwrap().dtype(), &dt_storage);
        let dt_meta_field = stripped.get("dt_meta").unwrap();
        assert_eq!(dt_meta_field.dtype(), &dt_storage);
        assert_eq!(
            dt_meta_field
                .metadata
                .as_ref()
                .and_then(|m| m.get("custom_key"))
                .map(|v| v.as_str()),
            Some("keep_me")
        );
        assert!(dt_meta_field
            .metadata
            .as_ref()
            .and_then(|m| m.get("ARROW:extension:name"))
            .is_none());
        assert!(dt_meta_field
            .metadata
            .as_ref()
            .and_then(|m| m.get("ARROW:extension:metadata"))
            .is_none());
        assert_eq!(stripped.get("dt_arr").unwrap().dtype(), &dt_arr_storage);
        match stripped.get("nested").unwrap().dtype() {
            ArrowDataType::Struct(fields) => {
                assert_eq!(fields[0].dtype(), &dt_storage);
                assert_eq!(fields[1].dtype(), &json_ext);
            },
            other => panic!("expected Struct, got {other:?}"),
        }
        assert_eq!(stripped.get("json_col").unwrap().dtype(), &json_ext);
        assert_eq!(stripped.get("geo_col").unwrap().dtype(), &geo_ext);

        // Verify IPC round-trip decoding of an actual Timestamp(us, None) batch
        // whose serialized IPC schema contains the google:sqlType:datetime metadata.
        let mut wire_meta = Metadata::default();
        wire_meta.insert(
            "ARROW:extension:name".into(),
            BQ_DATETIME_EXTENSION_NAME.into(),
        );
        let wire_schema =
            ArrowSchema::from_iter(vec![
                Field::new("dt".into(), dt_storage.clone(), false).with_metadata(wire_meta)
            ]);
        let mut schema_bytes = Vec::new();
        {
            let mut writer =
                StreamWriter::new(&mut schema_bytes, WriteOptions { compression: None });
            writer.start(&wire_schema, None).unwrap();
        }
        let arr =
            PrimitiveArray::<i64>::from_vec(vec![1_704_110_400_000_000_i64]).to(dt_storage.clone());
        let batch = RecordBatch::try_new(
            1,
            Arc::new(wire_schema.clone()),
            vec![Box::new(arr) as Box<dyn Array>],
        )
        .unwrap();
        let mut full_bytes = Vec::new();
        {
            let mut writer = StreamWriter::new(&mut full_bytes, WriteOptions { compression: None });
            writer.start(&wire_schema, None).unwrap();
            writer.write(&batch, None).unwrap();
        }
        let batch_bytes = bytes::Bytes::from(full_bytes[schema_bytes.len()..].to_vec());

        let parsed = parse_arrow_schema(&schema_bytes).unwrap();
        let parsed_dt = parsed.schema.get("dt").unwrap();
        assert_eq!(parsed_dt.dtype(), &dt_storage);
        assert!(parsed_dt.metadata.is_none());

        let decoded = decode_arrow_record_batches(batch_bytes, 1, parsed).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].arrays()[0].dtype(), &dt_storage);
    }

    #[test]
    fn test_strip_datetime_extension_in_deeply_nested_structs_and_containers() {
        use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray, Utf8Array};
        use polars_arrow::datatypes::{ExtensionType, TimeUnit};
        use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};
        use polars_arrow::offset::OffsetsBuffer;

        let dt_storage = ArrowDataType::Timestamp(TimeUnit::Microsecond, None);
        let dt_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: dt_storage.clone(),
            metadata: None,
        }));
        let dt_list_storage = ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            dt_storage.clone(),
            true,
        )));
        let dt_list_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: dt_list_storage.clone(),
            metadata: None,
        }));
        let json_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: "google:sqlType:json".into(),
            inner: ArrowDataType::Utf8,
            metadata: None,
        }));

        // 1. 15-level nested Struct (BigQuery's maximum RECORD nesting depth) containing
        //    both DATETIME and ARRAY<DATETIME> plus a preserved JSON extension leaf.
        let mut wire_dtype = ArrowDataType::Struct(vec![
            Field::new("dt".into(), dt_ext.clone(), true),
            Field::new("dt_arr".into(), dt_list_ext.clone(), true),
            Field::new("j".into(), json_ext.clone(), true),
        ]);
        let mut expected_dtype = ArrowDataType::Struct(vec![
            Field::new("dt".into(), dt_storage.clone(), true),
            Field::new("dt_arr".into(), dt_list_storage.clone(), true),
            Field::new("j".into(), json_ext.clone(), true),
        ]);
        for level in (0..14).rev() {
            let field_name = format!("level_{level}");
            wire_dtype = ArrowDataType::Struct(vec![Field::new(
                field_name.clone().into(),
                wire_dtype,
                true,
            )]);
            expected_dtype =
                ArrowDataType::Struct(vec![Field::new(field_name.into(), expected_dtype, true)]);
        }

        // Also cover LargeList, FixedSizeList, and Map containing DATETIME subfields.
        let large_list_dt =
            ArrowDataType::LargeList(Box::new(Field::new("item".into(), dt_ext.clone(), true)));
        let fixed_list_dt = ArrowDataType::FixedSizeList(
            Box::new(Field::new("item".into(), dt_ext.clone(), true)),
            2,
        );
        let map_dt = ArrowDataType::Map(
            Box::new(Field::new(
                "entries".into(),
                ArrowDataType::Struct(vec![
                    Field::new("key".into(), ArrowDataType::Utf8, false),
                    Field::new("value".into(), dt_ext.clone(), true),
                ]),
                false,
            )),
            false,
        );

        let schema = ArrowSchema::from_iter(vec![
            Field::new("deep".into(), wire_dtype, true),
            Field::new("large_list".into(), large_list_dt, true),
            Field::new("fixed_list".into(), fixed_list_dt, true),
            Field::new("map_col".into(), map_dt, true),
        ]);
        let stripped = strip_datetime_extensions(schema).unwrap();
        assert_eq!(stripped.get("deep").unwrap().dtype(), &expected_dtype);
        assert_eq!(
            stripped.get("large_list").unwrap().dtype(),
            &ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                dt_storage.clone(),
                true
            )))
        );
        assert_eq!(
            stripped.get("fixed_list").unwrap().dtype(),
            &ArrowDataType::FixedSizeList(
                Box::new(Field::new("item".into(), dt_storage.clone(), true)),
                2
            )
        );
        assert_eq!(
            stripped.get("map_col").unwrap().dtype(),
            &ArrowDataType::Map(
                Box::new(Field::new(
                    "entries".into(),
                    ArrowDataType::Struct(vec![
                        Field::new("key".into(), ArrowDataType::Utf8, false),
                        Field::new("value".into(), dt_storage.clone(), true),
                    ]),
                    false,
                )),
                false,
            )
        );

        // 2. Full IPC StreamWriter -> parse_arrow_schema -> decode_arrow_record_batches
        //    round-trip on STRUCT<mid: STRUCT<items: ARRAY<STRUCT<dt: DATETIME, j: JSON>>>>
        //    where leaf fields carry Extension(google:sqlType:datetime) / Extension(google:sqlType:json).
        let leaf_struct_fields = vec![
            Field::new("dt".into(), dt_ext.clone(), false),
            Field::new("j".into(), json_ext.clone(), false),
        ];
        let leaf_struct_dtype = ArrowDataType::Struct(leaf_struct_fields);
        let items_list_dtype = ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            leaf_struct_dtype.clone(),
            false,
        )));
        let mid_struct_dtype = ArrowDataType::Struct(vec![Field::new(
            "items".into(),
            items_list_dtype.clone(),
            false,
        )]);
        let outer_struct_dtype = ArrowDataType::Struct(vec![Field::new(
            "mid".into(),
            mid_struct_dtype.clone(),
            false,
        )]);
        let wire_schema = ArrowSchema::from_iter(vec![Field::new(
            "outer".into(),
            outer_struct_dtype.clone(),
            false,
        )]);

        let dt_vals = vec![1_704_110_400_000_000_i64, 1_704_196_800_000_000_i64];
        let dt_arr = PrimitiveArray::<i64>::from_vec(dt_vals.clone()).to(dt_ext.clone());
        let j_arr = Utf8Array::<i32>::from_slice([r#"{"a":1}"#, r#"{"a":2}"#]);
        let j_ext_arr = Utf8Array::<i32>::try_new(
            json_ext.clone(),
            j_arr.offsets().clone(),
            j_arr.values().clone(),
            None,
        )
        .unwrap();
        let leaf_struct_arr = StructArray::new(
            leaf_struct_dtype,
            2,
            vec![
                Box::new(dt_arr) as Box<dyn Array>,
                Box::new(j_ext_arr) as Box<dyn Array>,
            ],
            None,
        );
        let items_offsets = OffsetsBuffer::<i32>::try_from(vec![0, 2]).unwrap();
        let items_list_arr = ListArray::<i32>::new(
            items_list_dtype,
            items_offsets,
            Box::new(leaf_struct_arr),
            None,
        );
        let mid_struct_arr = StructArray::new(
            mid_struct_dtype,
            1,
            vec![Box::new(items_list_arr) as Box<dyn Array>],
            None,
        );
        let outer_struct_arr = StructArray::new(
            outer_struct_dtype,
            1,
            vec![Box::new(mid_struct_arr) as Box<dyn Array>],
            None,
        );
        let batch = RecordBatch::try_new(
            1,
            Arc::new(wire_schema.clone()),
            vec![Box::new(outer_struct_arr) as Box<dyn Array>],
        )
        .unwrap();

        let mut schema_bytes = Vec::new();
        {
            let mut writer =
                StreamWriter::new(&mut schema_bytes, WriteOptions { compression: None });
            writer.start(&wire_schema, None).unwrap();
        }
        let mut full_bytes = Vec::new();
        {
            let mut writer = StreamWriter::new(&mut full_bytes, WriteOptions { compression: None });
            writer.start(&wire_schema, None).unwrap();
            writer.write(&batch, None).unwrap();
        }
        let batch_bytes = bytes::Bytes::from(full_bytes[schema_bytes.len()..].to_vec());

        // Confirm the raw serialized IPC metadata actually contains the datetime extension
        // on the nested leaf before parse_arrow_schema strips it.
        let raw_metadata = read_stream_metadata(&mut Cursor::new(&schema_bytes)).unwrap();
        let ArrowDataType::Struct(raw_mid) = raw_metadata.schema.get("outer").unwrap().dtype()
        else {
            panic!("expected outer Struct");
        };
        let ArrowDataType::Struct(raw_list) = raw_mid[0].dtype() else {
            panic!("expected mid Struct");
        };
        let ArrowDataType::List(raw_item) = raw_list[0].dtype() else {
            panic!("expected items List");
        };
        let ArrowDataType::Struct(raw_inner) = raw_item.dtype() else {
            panic!("expected inner Struct");
        };
        assert_eq!(raw_inner[0].dtype(), &dt_ext);
        assert_eq!(raw_inner[1].dtype(), &json_ext);

        let parsed = parse_arrow_schema(&schema_bytes).unwrap();
        let decoded = decode_arrow_record_batches(batch_bytes, 1, parsed.clone()).unwrap();
        assert_eq!(decoded.len(), 1);

        // Verify the deeply nested dt leaf has its extension stripped while j keeps JSON extension.
        let outer_arr = decoded[0].arrays()[0]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let mid_arr = outer_arr.values()[0]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let items_arr = mid_arr.values()[0]
            .as_any()
            .downcast_ref::<ListArray<i32>>()
            .unwrap();
        let leaf_arr = items_arr
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(leaf_arr.values()[0].dtype(), &dt_storage);
        assert_eq!(leaf_arr.values()[1].dtype(), &json_ext);
        let decoded_dt = leaf_arr.values()[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(decoded_dt.values().as_slice(), dt_vals.as_slice());

        // Inspect schema metadata directly on the parsed deepest Struct fields.
        let ArrowDataType::Struct(mid_fields) = parsed.schema.get("outer").unwrap().dtype() else {
            panic!("expected outer Struct");
        };
        let ArrowDataType::Struct(list_fields) = mid_fields[0].dtype() else {
            panic!("expected mid Struct");
        };
        let ArrowDataType::List(item_field) = list_fields[0].dtype() else {
            panic!("expected items List");
        };
        let ArrowDataType::Struct(inner_fields) = item_field.dtype() else {
            panic!("expected inner Struct");
        };
        assert_eq!(inner_fields[0].dtype(), &dt_storage);
        assert!(inner_fields[0].metadata.is_none());
        assert_eq!(inner_fields[1].dtype(), &json_ext);
    }

    #[test]
    fn test_array_datetime_ipc_roundtrip_strips_outer_extension() {
        use polars_arrow::array::{Array, ListArray, PrimitiveArray};
        use polars_arrow::datatypes::{ExtensionType, TimeUnit};
        use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};
        use polars_arrow::offset::OffsetsBuffer;

        let dt_storage = ArrowDataType::Timestamp(TimeUnit::Microsecond, None);
        let list_storage = ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            dt_storage.clone(),
            false,
        )));
        let dt_array_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: list_storage.clone(),
            metadata: None,
        }));
        let wire_schema = ArrowSchema::from_iter(vec![Field::new(
            "dt_arr".into(),
            dt_array_ext.clone(),
            false,
        )]);

        let dt_vals = vec![1_704_110_400_000_000_i64, 1_704_196_800_000_000_i64];
        let inner = PrimitiveArray::<i64>::from_vec(dt_vals.clone()).to(dt_storage.clone());
        let offsets = OffsetsBuffer::<i32>::try_from(vec![0, 2]).unwrap();
        let list_arr = ListArray::<i32>::new(dt_array_ext, offsets, Box::new(inner), None);
        let batch = RecordBatch::try_new(
            1,
            Arc::new(wire_schema.clone()),
            vec![Box::new(list_arr) as Box<dyn Array>],
        )
        .unwrap();

        let mut schema_bytes = Vec::new();
        {
            let mut writer =
                StreamWriter::new(&mut schema_bytes, WriteOptions { compression: None });
            writer.start(&wire_schema, None).unwrap();
        }
        let mut full_bytes = Vec::new();
        {
            let mut writer = StreamWriter::new(&mut full_bytes, WriteOptions { compression: None });
            writer.start(&wire_schema, None).unwrap();
            writer.write(&batch, None).unwrap();
        }
        let batch_bytes = bytes::Bytes::from(full_bytes[schema_bytes.len()..].to_vec());

        // Strips outer `google:sqlType:datetime` extension on ARRAY<DATETIME>.
        let parsed = parse_arrow_schema(&schema_bytes).unwrap();
        assert_eq!(parsed.schema.get("dt_arr").unwrap().dtype(), &list_storage);
        let decoded = decode_arrow_record_batches(batch_bytes, 1, parsed).unwrap();
        assert_eq!(decoded[0].arrays()[0].dtype(), &list_storage);
        let decoded_list = decoded[0].arrays()[0]
            .as_any()
            .downcast_ref::<ListArray<i32>>()
            .unwrap();
        let inner_arr = decoded_list
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(inner_arr.values().as_slice(), dt_vals.as_slice());
    }

    #[test]
    fn test_validate_and_strip_dictionary_union_and_depth() {
        use polars_arrow::datatypes::{ExtensionType, IntegerType, TimeUnit, UnionMode, UnionType};

        let dt_storage = ArrowDataType::Timestamp(TimeUnit::Microsecond, None);
        let dt_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: dt_storage.clone(),
            metadata: None,
        }));

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

        // Dictionary and Union wrapping datetime extension are stripped
        let container_dt = ArrowSchema::from_iter(vec![
            Field::new(
                "dict_dt".into(),
                ArrowDataType::Dictionary(IntegerType::Int32, Box::new(dt_ext.clone()), false),
                true,
            ),
            Field::new(
                "union_dt".into(),
                ArrowDataType::Union(Box::new(UnionType {
                    fields: vec![Field::new("dt".into(), dt_ext, true)],
                    ids: Some(vec![0]),
                    mode: UnionMode::Sparse,
                })),
                true,
            ),
        ]);
        let stripped = strip_datetime_extensions(container_dt).unwrap();
        assert_eq!(
            stripped.get("dict_dt").unwrap().dtype(),
            &ArrowDataType::Dictionary(IntegerType::Int32, Box::new(dt_storage.clone()), false)
        );
        match stripped.get("union_dt").unwrap().dtype() {
            ArrowDataType::Union(u) => assert_eq!(u.fields[0].dtype(), &dt_storage),
            other => panic!("expected Union, got {other:?}"),
        }

        // `strip_datetime_extensions` must fail closed (Protocol error), NOT skip silently,
        // when schema nesting depth exceeds MAX_SCHEMA_NESTING_DEPTH.
        let mut deep_dtype = ArrowDataType::Int64;
        for _ in 0..=MAX_SCHEMA_NESTING_DEPTH + 1 {
            deep_dtype = ArrowDataType::List(Box::new(Field::new("item".into(), deep_dtype, true)));
        }
        let deep_schema = ArrowSchema::from_iter(vec![Field::new("deep".into(), deep_dtype, true)]);
        assert!(matches!(
            strip_datetime_extensions(deep_schema),
            Err(BigQueryError::Protocol(_))
        ));
    }

    #[test]
    fn test_strip_datetime_preserves_unexpected_inner_dtype() {
        use polars_arrow::datatypes::{ExtensionType, Metadata, TimeUnit};

        // An unexpected `google:sqlType:datetime` extension wrapping Utf8 (not microsecond
        // naive Timestamp) must NOT be silently stripped to plain Utf8.
        let string_dt_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: ArrowDataType::Utf8,
            metadata: None,
        }));
        let tz_dt_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            metadata: None,
        }));
        let ns_dt_ext = ArrowDataType::Extension(Box::new(ExtensionType {
            name: BQ_DATETIME_EXTENSION_NAME.into(),
            inner: ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
            metadata: None,
        }));
        let mut meta = Metadata::default();
        meta.insert(
            "ARROW:extension:name".into(),
            BQ_DATETIME_EXTENSION_NAME.into(),
        );
        let schema = ArrowSchema::from_iter(vec![
            Field::new("unexpected_dt".into(), string_dt_ext.clone(), true)
                .with_metadata(meta.clone()),
            Field::new("tz_dt".into(), tz_dt_ext.clone(), true).with_metadata(meta.clone()),
            Field::new("ns_dt".into(), ns_dt_ext.clone(), true).with_metadata(meta),
        ]);
        let stripped = strip_datetime_extensions(schema).unwrap();
        let f = stripped.get("unexpected_dt").unwrap();
        assert_eq!(f.dtype(), &string_dt_ext);
        assert_eq!(
            f.metadata
                .as_ref()
                .and_then(|m| m.get("ARROW:extension:name"))
                .map(|v| v.as_str()),
            Some(BQ_DATETIME_EXTENSION_NAME)
        );
        assert_eq!(stripped.get("tz_dt").unwrap().dtype(), &tz_dt_ext);
        assert_eq!(stripped.get("ns_dt").unwrap().dtype(), &ns_dt_ext);
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
