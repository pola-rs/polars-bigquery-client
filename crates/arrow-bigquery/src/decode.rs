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

fn strip_datetime_extension_dtype(dtype: ArrowDataType) -> ArrowDataType {
    match dtype {
        ArrowDataType::Extension(mut ext) => {
            let inner = strip_datetime_extension_dtype(ext.inner);
            if ext.name.as_str() == BQ_DATETIME_EXTENSION_NAME {
                inner
            } else {
                ext.inner = inner;
                ArrowDataType::Extension(ext)
            }
        },
        ArrowDataType::List(inner) => {
            ArrowDataType::List(Box::new(strip_datetime_extension_field(*inner)))
        },
        ArrowDataType::LargeList(inner) => {
            ArrowDataType::LargeList(Box::new(strip_datetime_extension_field(*inner)))
        },
        ArrowDataType::FixedSizeList(inner, size) => {
            ArrowDataType::FixedSizeList(Box::new(strip_datetime_extension_field(*inner)), size)
        },
        ArrowDataType::Map(inner, ordered) => {
            ArrowDataType::Map(Box::new(strip_datetime_extension_field(*inner)), ordered)
        },
        ArrowDataType::Struct(fields) => ArrowDataType::Struct(
            fields
                .into_iter()
                .map(strip_datetime_extension_field)
                .collect(),
        ),
        other => other,
    }
}

fn strip_datetime_extension_field(mut field: Field) -> Field {
    field.dtype = strip_datetime_extension_dtype(field.dtype);
    if let Some(meta) = field.metadata.as_ref() {
        if meta.get("ARROW:extension:name").map(|v| v.as_str()) == Some(BQ_DATETIME_EXTENSION_NAME)
        {
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
    field
}

fn strip_datetime_extensions(schema: ArrowSchema) -> ArrowSchema {
    schema
        .into_iter()
        .map(|(_, field)| strip_datetime_extension_field(field))
        .collect()
}

fn validate_supported_arrow_dtype(dtype: &ArrowDataType, path: &str) -> Result<(), BigQueryError> {
    match dtype {
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
        | ArrowDataType::Map(inner, _) => validate_supported_arrow_dtype(inner.dtype(), path),
        ArrowDataType::Struct(fields) => {
            for (idx, field) in fields.iter().enumerate() {
                let child_path = if field.name.is_empty() {
                    format!("{path}[{idx}]")
                } else {
                    format!("{path}.{}", field.name)
                };
                validate_supported_arrow_dtype(field.dtype(), &child_path)?;
            }
            Ok(())
        },
        ArrowDataType::Extension(ext) => validate_supported_arrow_dtype(&ext.inner, path),
        _ => Ok(()),
    }
}

fn validate_supported_arrow_schema(schema: &ArrowSchema) -> Result<(), BigQueryError> {
    for (_, field) in schema.iter() {
        validate_supported_arrow_dtype(field.dtype(), field.name.as_str())?;
    }
    Ok(())
}

/// Parses IPC-serialized Arrow schema bytes into [`StreamMetadata`].
///
/// Rejects unsupported `Interval(MonthDayNano)` fields (BigQuery `INTERVAL`) and
/// strips the `google:sqlType:datetime` extension wrapper so naive microsecond
/// `DATETIME` columns expose native Arrow `Timestamp(Microsecond, None)`.
pub(crate) fn parse_arrow_schema(schema_bytes: &[u8]) -> Result<StreamMetadata, BigQueryError> {
    let mut cursor = Cursor::new(schema_bytes);
    let mut metadata = read_stream_metadata(&mut cursor)?;
    validate_supported_arrow_schema(&metadata.schema)?;
    metadata.schema = strip_datetime_extensions(metadata.schema);
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

        let res = decode_arrow_record_batches(
            serialized_record_batch,
            expected_row_count,
            Arc::unwrap_or_clone(metadata),
        );

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
            let mut state = self.state.lock().unwrap();
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

    fn drain(&self) {
        loop {
            let next_job = {
                let mut state = self.state.lock().unwrap();
                match state.queue.pop_front() {
                    Some(job) => job,
                    None => {
                        state.running = false;
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
        let err = validate_supported_arrow_schema(&ext_schema)
            .expect_err("Extension<MonthDayNano> should be rejected");
        assert!(matches!(err, BigQueryError::UnsupportedType(_)));
        assert!(err.to_string().contains("dur_ext"));
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

        let stripped = strip_datetime_extensions(schema);
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
}
