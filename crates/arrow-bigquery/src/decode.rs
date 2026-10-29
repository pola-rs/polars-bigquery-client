use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use polars_arrow::io::ipc::read::{
    read_stream_metadata, StreamMetadata, StreamReader, StreamState,
};
use polars_arrow::record_batch::RecordBatch;

use crate::BigQueryError;

/// Parses IPC-serialized Arrow schema bytes into [`StreamMetadata`].
pub(crate) fn parse_arrow_schema(schema_bytes: &[u8]) -> Result<StreamMetadata, BigQueryError> {
    let mut cursor = Cursor::new(schema_bytes);
    Ok(read_stream_metadata(&mut cursor)?)
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
}
