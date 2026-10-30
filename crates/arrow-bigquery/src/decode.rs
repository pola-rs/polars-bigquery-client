use std::collections::{HashMap, VecDeque};
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_buffer::Buffer;
use arrow_ipc::reader::{read_dictionary, read_record_batch, StreamReader};
use arrow_ipc::{root_as_message, MessageHeader};
use arrow_schema::{ArrowError, SchemaRef};

use crate::BigQueryError;

const IPC_CONTINUATION_MARKER: [u8; 4] = [0xFF; 4];

/// Parses IPC-serialized Arrow schema bytes into a [`SchemaRef`].
pub(crate) fn parse_arrow_schema(schema_bytes: &[u8]) -> Result<SchemaRef, BigQueryError> {
    let reader = StreamReader::try_new(Cursor::new(schema_bytes), None)?;
    Ok(reader.schema())
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
    schema: &SchemaRef,
) -> Result<Vec<RecordBatch>, BigQueryError> {
    let mut buf = Buffer::from(serialized_record_batch);
    let mut dictionaries_by_id: HashMap<i64, ArrayRef> = HashMap::new();
    let mut batches = Vec::with_capacity(1);
    let mut actual_rows: i64 = 0;

    while !buf.is_empty() {
        if buf.len() < 4 {
            return Err(
                ArrowError::ParseError("Unexpected EOF reading IPC message prefix".into()).into(),
            );
        }
        let mut len_bytes: [u8; 4] = buf[..4].try_into().unwrap();
        buf.advance(4);

        if len_bytes == IPC_CONTINUATION_MARKER {
            if buf.len() < 4 {
                return Err(ArrowError::ParseError(
                    "Unexpected EOF reading IPC metadata length".into(),
                )
                .into());
            }
            len_bytes = buf[..4].try_into().unwrap();
            buf.advance(4);
        }

        let raw_meta_len = i32::from_le_bytes(len_bytes);
        if raw_meta_len == 0 {
            break;
        }
        let meta_len = usize::try_from(raw_meta_len).map_err(|_| {
            ArrowError::ParseError(format!("Negative IPC metadata length: {raw_meta_len}"))
        })?;
        if buf.len() < meta_len {
            return Err(ArrowError::ParseError(
                "Unexpected EOF reading IPC message metadata".into(),
            )
            .into());
        }

        let meta_buf = buf.slice_with_length(0, meta_len);
        buf.advance(meta_len);
        let message = root_as_message(&meta_buf)
            .map_err(|e| ArrowError::ParseError(format!("Unable to get root as message: {e:?}")))?;

        let raw_body_len = message.bodyLength();
        let body_len = usize::try_from(raw_body_len).map_err(|_| {
            ArrowError::ParseError(format!("Negative IPC body length: {raw_body_len}"))
        })?;
        if buf.len() < body_len {
            return Err(
                ArrowError::ParseError("Unexpected EOF reading IPC message body".into()).into(),
            );
        }

        let body = buf.slice_with_length(0, body_len);
        buf.advance(body_len);

        match message.header_type() {
            MessageHeader::RecordBatch => {
                let ipc_batch = message.header_as_record_batch().ok_or_else(|| {
                    ArrowError::ParseError("Unable to read IPC message as record batch".into())
                })?;
                let batch = read_record_batch(
                    &body,
                    ipc_batch,
                    Arc::clone(schema),
                    &dictionaries_by_id,
                    None,
                    &message.version(),
                )?;
                let batch_rows = i64::try_from(batch.num_rows()).map_err(|_| {
                    BigQueryError::Protocol("Decoded batch row count overflow".into())
                })?;
                actual_rows = actual_rows.checked_add(batch_rows).ok_or_else(|| {
                    BigQueryError::Protocol("Decoded batch row count overflow".into())
                })?;
                batches.push(batch);
            },
            MessageHeader::DictionaryBatch => {
                let dict_batch = message.header_as_dictionary_batch().ok_or_else(|| {
                    ArrowError::ParseError("Unable to read IPC message as dictionary batch".into())
                })?;
                read_dictionary(
                    &body,
                    dict_batch,
                    schema,
                    &mut dictionaries_by_id,
                    &message.version(),
                )?;
            },
            MessageHeader::NONE => {},
            other => {
                return Err(ArrowError::IpcError(format!(
                    "Unexpected IPC message header type in record batch payload: {other:?}"
                ))
                .into());
            },
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
    pub(crate) schema: SchemaRef,
    pub(crate) permit: tokio::sync::mpsc::OwnedPermit<Result<RecordBatch, BigQueryError>>,
    pub(crate) cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
    pub(crate) cancel_rx: tokio::sync::watch::Receiver<bool>,
}

impl DecodeJob {
    fn run(self) {
        let Self {
            serialized_record_batch,
            expected_row_count,
            schema,
            permit,
            cancel_tx,
            cancel_rx,
        } = self;

        // If the stream task or receiver was cancelled while this job was queued,
        // skip decompression immediately.
        if *cancel_rx.borrow() {
            return;
        }

        let res = decode_arrow_record_batches(serialized_record_batch, expected_row_count, &schema);

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
    use arrow_array::Int64Array;
    use arrow_ipc::writer::{IpcWriteOptions, StreamWriter};
    use arrow_ipc::CompressionType;
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::bigquery_read_stream::tests::create_test_arrow_payload_with_range;

    #[test]
    fn test_decode_compressed_arrow_record_batches_lz4_and_zstd() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let values: Vec<i64> = (0..1_000).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(values.clone()))],
        )
        .unwrap();

        for compression in [CompressionType::LZ4_FRAME, CompressionType::ZSTD] {
            let write_opts = IpcWriteOptions::default()
                .try_with_compression(Some(compression))
                .unwrap();

            let mut schema_only = Vec::new();
            let _ =
                StreamWriter::try_new_with_options(&mut schema_only, &schema, write_opts.clone())
                    .unwrap();

            let mut full_stream = Vec::new();
            let mut writer =
                StreamWriter::try_new_with_options(&mut full_stream, &schema, write_opts).unwrap();
            writer.write(&batch).unwrap();
            let batch_bytes = full_stream[schema_only.len()..].to_vec();

            let parsed_schema = parse_arrow_schema(&schema_only).unwrap();
            let decoded =
                decode_arrow_record_batches(bytes::Bytes::from(batch_bytes), 1_000, &parsed_schema)
                    .unwrap();
            assert_eq!(decoded.len(), 1);
            assert_eq!(decoded[0].num_rows(), 1_000);
            let col = decoded[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(col.values(), &values[..]);
        }
    }

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

        let schema = parse_arrow_schema(&schema_bytes).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(sizes.len() + 2);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        for (serialized_record_batch, expected_row_count) in payloads {
            let permit = tx.clone().reserve_owned().await.unwrap();
            queue.push(DecodeJob {
                serialized_record_batch,
                expected_row_count,
                schema: Arc::clone(&schema),
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
            received_sizes.push(batch.num_rows());
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for &val in col.values() {
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

        let schema = parse_arrow_schema(&schema_bytes).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(sizes.len() + 2);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        for (serialized_record_batch, expected_row_count) in payloads {
            let permit = tx.clone().reserve_owned().await.unwrap();
            queue.push(DecodeJob {
                serialized_record_batch,
                expected_row_count,
                schema: Arc::clone(&schema),
                permit,
                cancel_tx: Arc::clone(&cancel_tx),
                cancel_rx: cancel_rx.clone(),
            });
        }
        drop(tx);

        let mut all_values = Vec::with_capacity(next_start as usize);
        while let Some(res) = rx.recv().await {
            let batch = res.unwrap();
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(col.values());
        }
        all_values.sort_unstable();
        let expected: Vec<i64> = (0..next_start).collect();
        assert_eq!(all_values, expected);
    }

    #[tokio::test]
    async fn test_decode_job_error_signals_cancellation_and_pushes_err() {
        let queue = DecodeQueue::new(true);

        let (schema_bytes, valid_batch) = create_test_arrow_payload_with_range(0, 5);
        let schema = parse_arrow_schema(&schema_bytes).unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        // Enqueue a job with a mismatched expected_row_count followed by a valid job.
        let permit1 = tx.clone().reserve_owned().await.unwrap();
        queue.push(DecodeJob {
            serialized_record_batch: bytes::Bytes::from(valid_batch.clone()),
            expected_row_count: 999,
            schema: Arc::clone(&schema),
            permit: permit1,
            cancel_tx: Arc::clone(&cancel_tx),
            cancel_rx: cancel_rx.clone(),
        });

        let permit2 = tx.clone().reserve_owned().await.unwrap();
        queue.push(DecodeJob {
            serialized_record_batch: bytes::Bytes::from(valid_batch),
            expected_row_count: 5,
            schema,
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
}
