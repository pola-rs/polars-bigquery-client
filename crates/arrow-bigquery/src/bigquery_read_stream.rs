//! ## bigquery_read_stream.rs
//!
//! This module reads from BQ Storage Read API stream and writes the results
//! as arrow record batches to a tokio mpsc queue as messages arrive.
//!
//! To robustly handle disruptions, this module uses the following state
//! machine, containing single API method retries and stream-level reconnection
//! logic.
//!
//! ```text
//! ┌─ read_stream_inner ─────────────────────────────────────────────────┐
//! │              ┌──►──┐                                                │
//! │   Receive valid message                                             │
//! │  including empty/waiting                                            │
//! │          ┌───┼─────▼────┐                                           │
//! │        consume_next_message ◄─────────────────────┐                 │
//! │  ┌───────┼   Running    │                         │                 │
//! │  │       └───────┬──────┘                     ReadRows              │
//! │  │    Recoverable│error                        success              │
//! │  │               │                         ┌──────┼───────┐         │
//! │  │       ┌───────▼──────┐ Exponential  connect_read_rows_stream     │
//! │  │       │ Backing off  ┼──Back─off───────►│(Re)Connecting│         │
//! │  ▼       └─────────────┬┘                  └───────┬──────┘         │
//! │ Unrecoverable error   Out of retries       ReadRows│error           │
//! │ or no more messages    │                 after exhausting retries   │
//! │  │                  ┌──▼───────────┐               │                │
//! │  └──────────────────►  Terminated──◄───────────────┘                │
//! │                     └──────────────┘                                │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use std::io::Cursor;
use std::sync::{Arc, LazyLock};

use google_cloud_bigquery::builder::read::ReadRows;
use google_cloud_bigquery::model::{read_rows_response, ReadRowsResponse};
use google_cloud_gax::streaming::ResponseStream;
use polars_arrow::datatypes::ArrowSchemaRef;
use polars_arrow::io::ipc::read::{
    read_stream_metadata, StreamMetadata, StreamReader, StreamState,
};
use polars_arrow::record_batch::RecordBatch;

use super::bigquery_read_retry::{self, StreamRetryConfig};
use crate::BigQueryError;

pub(crate) type SchemaSender = tokio::sync::oneshot::Sender<Result<ArrowSchemaRef, BigQueryError>>;

static DEFAULT_DECODE_POOL: LazyLock<Arc<rayon::ThreadPool>> = LazyLock::new(|| {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(|i| format!("arrow-bq-decode-{i}"))
        .build()
        .expect("failed to build Arrow IPC decode thread pool");
    Arc::new(pool)
});

/// Returns a handle to the shared bounded thread pool used for CPU-bound
/// Arrow IPC (LZ4/ZSTD) decompression.
pub(crate) fn default_decode_pool() -> Arc<rayon::ThreadPool> {
    Arc::clone(&DEFAULT_DECODE_POOL)
}

/// Parses IPC-serialized Arrow schema bytes into [`StreamMetadata`].
pub(crate) fn parse_arrow_schema(schema_bytes: &[u8]) -> Result<StreamMetadata, BigQueryError> {
    let mut cursor = Cursor::new(schema_bytes);
    Ok(read_stream_metadata(&mut cursor)?)
}

/// Decodes an Arrow IPC record batch payload and verifies its total row count
/// matches `expected_row_count`.
pub(crate) fn decode_arrow_record_batches(
    serialized_record_batch: bytes::Bytes,
    expected_row_count: i64,
    metadata: &StreamMetadata,
) -> Result<Vec<RecordBatch>, BigQueryError> {
    if serialized_record_batch.is_empty() {
        if expected_row_count != 0 {
            return Err(BigQueryError::Protocol(format!(
                "Row count mismatch: gRPC protobuf reported {} rows, but Arrow IPC decoded 0 rows",
                expected_row_count
            )));
        }
        return Ok(Vec::new());
    }

    let cursor = Cursor::new(serialized_record_batch);
    let mut reader = StreamReader::new(cursor, metadata.clone(), None);
    let mut batches = Vec::new();
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

/// Extracts the serialized Arrow record batch bytes and expected row count from
/// a [`ReadRowsResponse`].
fn extract_response_arrow_bytes(
    response: ReadRowsResponse,
) -> Result<Option<(bytes::Bytes, i64)>, BigQueryError> {
    let row_count = response.row_count;

    let serialized_record_batch = match response.rows {
        Some(read_rows_response::Rows::ArrowRecordBatch(value)) => value.serialized_record_batch,
        None => {
            if row_count != 0 {
                return Err(BigQueryError::Protocol(format!(
                    "Row count mismatch: gRPC protobuf reported {} rows, rows field didn't included any rows",
                    row_count
                )));
            }
            return Ok(None);
        },
        _ => {
            return Err(BigQueryError::Protocol(
                "Unexpectedly got some format other than arrow bytes".into(),
            ))
        },
    };

    if serialized_record_batch.is_empty() {
        if row_count != 0 {
            return Err(BigQueryError::Protocol(format!(
                "Row count mismatch: gRPC protobuf reported {} rows, but Arrow IPC decoded 0 rows",
                row_count
            )));
        }
        return Ok(None);
    }

    Ok(Some((serialized_record_batch, row_count)))
}

#[cfg(test)]
fn read_rows_response_to_record_batch(
    response: ReadRowsResponse,
    schema: &[u8],
) -> Result<Option<(RecordBatch, i64)>, BigQueryError> {
    let Some((serialized_record_batch, row_count)) = extract_response_arrow_bytes(response)? else {
        return Ok(None);
    };
    let metadata = parse_arrow_schema(schema)?;
    let mut batches = decode_arrow_record_batches(serialized_record_batch, row_count, &metadata)?;
    if batches.is_empty() {
        Ok(None)
    } else {
        let batch = batches.remove(0);
        let actual_rows = batch.len() as i64;
        Ok(Some((batch, actual_rows)))
    }
}

pub(crate) async fn decode_batches_on_pool(
    decode_pool: &rayon::ThreadPool,
    serialized_record_batch: bytes::Bytes,
    expected_row_count: i64,
    metadata: Arc<StreamMetadata>,
) -> Result<Vec<RecordBatch>, BigQueryError> {
    if expected_row_count == 0 && serialized_record_batch.is_empty() {
        return Ok(Vec::new());
    }

    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    decode_pool.spawn(move || {
        // If the stream task was cancelled while this job was queued in the pool,
        // skip decompression immediately.
        if done_tx.is_closed() {
            return;
        }
        let res =
            decode_arrow_record_batches(serialized_record_batch, expected_row_count, &metadata);
        let _ = done_tx.send(res);
    });

    done_rx.await.map_err(|_| {
        BigQueryError::Other("Arrow IPC decode worker terminated unexpectedly".into())
    })?
}

async fn decode_response_on_pool(
    decode_pool: &rayon::ThreadPool,
    response: ReadRowsResponse,
    metadata: Arc<StreamMetadata>,
) -> Result<Vec<RecordBatch>, BigQueryError> {
    let Some((serialized_record_batch, row_count)) = extract_response_arrow_bytes(response)? else {
        return Ok(Vec::new());
    };
    decode_batches_on_pool(decode_pool, serialized_record_batch, row_count, metadata).await
}

/// Represents the state of the stream reading state machine.
#[derive(Debug)]
enum ReadStreamState {
    /// Establishing or resuming the BigQuery read stream at `current_offset`.
    Connecting,
    /// Waiting for the next message in a BigQuery read stream.
    Running,
    /// Encountered a transient mid-stream disconnection; backing off before reconnecting.
    BackingOff(google_cloud_bigquery::Error),
    /// Stream completed cleanly, fatal error occurred, or consumer dropped the receiver.
    Terminated,
}

enum ConnectOutcome {
    Connected(ResponseStream<ReadRowsResponse>),
    Failed(google_cloud_bigquery::Error),
    Cancelled,
}

/// Layer 1: Connects to a BigQuery read stream at `offset`, retrying transient gRPC connection errors.
async fn connect_read_rows_stream(
    read_rows: &ReadRows,
    offset: i64,
    retry_config: &StreamRetryConfig,
    cancel_rx: &mut tokio::sync::watch::Receiver<bool>,
) -> ConnectOutcome {
    let mut connect_session = retry_config.make_connect_session();
    loop {
        if *cancel_rx.borrow_and_update() {
            return ConnectOutcome::Cancelled;
        }

        let send_fut = read_rows.clone().set_offset(offset).send();
        let res = tokio::select! {
            biased;
            _ = cancel_rx.changed() => return ConnectOutcome::Cancelled,
            res = send_fut => res,
        };

        match res {
            Ok(stream) => {
                connect_session.record_success();
                return ConnectOutcome::Connected(stream);
            },
            Err(err) => match connect_session.next_delay(err, cancel_rx).await {
                Ok(Some(())) => continue,
                Ok(None) => return ConnectOutcome::Cancelled,
                Err(err) => return ConnectOutcome::Failed(err),
            },
        }
    }
}

async fn send_fatal_error(
    err: BigQueryError,
    schema_tx: &mut Option<SchemaSender>,
    tx: &tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    cancel_tx: &tokio::sync::watch::Sender<bool>,
) {
    let _ = cancel_tx.send_replace(true);
    if let Some(schema_tx) = schema_tx.take() {
        let _ = schema_tx.send(Err(err));
    } else {
        let _ = tx.send(Err(err)).await;
    }
}

/// Layer 2: Consumes a single message from an established read stream and returns the next [`ReadStreamState`].
///
/// Processes the next message from an active read stream while in [`ReadStreamState::Running`].
#[allow(clippy::too_many_arguments)]
async fn consume_next_message(
    stream: &mut ResponseStream<ReadRowsResponse>,
    metadata: &mut Option<Arc<StreamMetadata>>,
    schema_tx: &mut Option<SchemaSender>,
    current_offset: &mut i64,
    tx: &tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    decode_pool: &rayon::ThreadPool,
    cancel_tx: &tokio::sync::watch::Sender<bool>,
    cancel_rx: &mut tokio::sync::watch::Receiver<bool>,
) -> ReadStreamState {
    if *cancel_rx.borrow_and_update() {
        return ReadStreamState::Terminated;
    }

    let next_item = tokio::select! {
        biased;
        _ = cancel_rx.changed() => return ReadStreamState::Terminated,
        item = stream.next() => item,
    };

    match next_item {
        Some(Ok(mut value)) => {
            let metadata_arc = match metadata {
                Some(ref m) => Arc::clone(m),
                None => {
                    let serialized_schema = match value.schema.take() {
                        Some(read_rows_response::Schema::ArrowSchema(arrow_schema)) => {
                            arrow_schema.serialized_schema
                        },
                        None if value.rows.is_none() && value.row_count == 0 => {
                            // Empty heartbeat response sent by BigQuery while waiting
                            // for a running query job to produce its schema and rows.
                            return ReadStreamState::Running;
                        },
                        _ => {
                            send_fatal_error(
                                BigQueryError::Protocol(
                                    "Unexpectedly got schema type other than arrow".into(),
                                ),
                                schema_tx,
                                tx,
                                cancel_tx,
                            )
                            .await;
                            return ReadStreamState::Terminated;
                        },
                    };
                    let parsed = match parse_arrow_schema(&serialized_schema) {
                        Ok(parsed) => Arc::new(parsed),
                        Err(err) => {
                            send_fatal_error(err, schema_tx, tx, cancel_tx).await;
                            return ReadStreamState::Terminated;
                        },
                    };
                    if let Some(sender) = schema_tx.take() {
                        let _ = sender.send(Ok(Arc::new(parsed.schema.clone())));
                    }
                    *metadata = Some(Arc::clone(&parsed));
                    parsed
                },
            };

            let decoded = tokio::select! {
                biased;
                _ = cancel_rx.changed() => return ReadStreamState::Terminated,
                res = decode_response_on_pool(decode_pool, value, metadata_arc) => res,
            };
            match decoded {
                Ok(batches) => {
                    for batch in batches {
                        let row_count = batch.len() as i64;
                        match current_offset.checked_add(row_count) {
                            Some(next_offset) => *current_offset = next_offset,
                            None => {
                                send_fatal_error(
                                    BigQueryError::Protocol(format!(
                                        "stream offset overflow: current_offset={current_offset}, row_count={row_count}"
                                    )),
                                    schema_tx,
                                    tx,
                                    cancel_tx,
                                )
                                .await;
                                return ReadStreamState::Terminated;
                            },
                        }
                        tokio::select! {
                            biased;
                            _ = cancel_rx.changed() => return ReadStreamState::Terminated,
                            send_res = tx.send(Ok(batch)) => {
                                if send_res.is_err() {
                                    // `tx.send` returns `Err` strictly when all `Receiver` handles (`rx`) have been
                                    // dropped or closed. Terminating this stream cleanly prevents orphan background tasks.
                                    return ReadStreamState::Terminated;
                                }
                            }
                        }
                    }
                    ReadStreamState::Running
                },
                Err(err) => {
                    send_fatal_error(err, schema_tx, tx, cancel_tx).await;
                    ReadStreamState::Terminated
                },
            }
        },
        None => {
            if schema_tx.is_some() {
                send_fatal_error(
                    BigQueryError::Protocol(
                        "ReadRows stream closed without sending an Arrow schema".into(),
                    ),
                    schema_tx,
                    tx,
                    cancel_tx,
                )
                .await;
            }
            ReadStreamState::Terminated
        },
        Some(Err(err)) => {
            if bigquery_read_retry::stream_reconnect_predicate(&err) {
                ReadStreamState::BackingOff(err)
            } else {
                send_fatal_error(BigQueryError::Grpc(err), schema_tx, tx, cancel_tx).await;
                ReadStreamState::Terminated
            }
        },
    }
}

/// Layer 3: Orchestrates stream connection, consumption, and mid-stream reconnections as an explicit state machine.
///
/// Uses a [`bigquery_read_retry::BackoffSession`] instantiated from `retry_config` to handle backoff delays
/// upon mid-stream gRPC disconnections.
///
/// ### A note regarding the transition from BackingOff to Connecting
///
/// If we are able to successfully call ReadRows but don't receive a successful
/// message, the exponential backoff session state is preserved from the
/// previous attempt. This ensures that we are able to make progress in the
/// case of retriable/reconnectable failures.
///
/// When we do receive a successful message, the backoff state **is reset**
/// (`backoff_session.reset()`). This ensures that long-lived data streams
/// experiencing rare, transient network interruptions separated by minutes or
/// hours always receive a full retry budget.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn read_stream_inner(
    read_rows: ReadRows,
    mut current_offset: i64,
    mut metadata: Option<Arc<StreamMetadata>>,
    mut schema_tx: Option<SchemaSender>,
    tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    retry_config: StreamRetryConfig,
    decode_pool: Arc<rayon::ThreadPool>,
    cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
    mut cancel_rx: tokio::sync::watch::Receiver<bool>,
) {
    let mut backoff_session = retry_config.make_reconnect_session();
    let mut state = ReadStreamState::Connecting;
    let mut stream: Option<ResponseStream<ReadRowsResponse>> = None;

    while !matches!(state, ReadStreamState::Terminated) {
        if *cancel_rx.borrow_and_update() {
            break;
        }

        state = match state {
            ReadStreamState::Connecting => {
                match connect_read_rows_stream(
                    &read_rows,
                    current_offset,
                    &retry_config,
                    &mut cancel_rx,
                )
                .await
                {
                    ConnectOutcome::Connected(inner_stream) => {
                        stream = Some(inner_stream);
                        ReadStreamState::Running
                    },
                    ConnectOutcome::Failed(err) => {
                        send_fatal_error(BigQueryError::Grpc(err), &mut schema_tx, &tx, &cancel_tx)
                            .await;
                        ReadStreamState::Terminated
                    },
                    ConnectOutcome::Cancelled => ReadStreamState::Terminated,
                }
            },
            ReadStreamState::Running => match stream {
                Some(ref mut inner_stream) => {
                    let prev_offset = current_offset;
                    let next_state = consume_next_message(
                        inner_stream,
                        &mut metadata,
                        &mut schema_tx,
                        &mut current_offset,
                        &tx,
                        &decode_pool,
                        &cancel_tx,
                        &mut cancel_rx,
                    )
                    .await;
                    // Reset backoff state after making data progress so
                    // future transient errors get a full retry budget.
                    if current_offset > prev_offset {
                        backoff_session.reset();
                    }
                    next_state
                },
                None => {
                    send_fatal_error(
                        BigQueryError::Other(
                            "Tried to read from BigQuery stream but not connected".into(),
                        ),
                        &mut schema_tx,
                        &tx,
                        &cancel_tx,
                    )
                    .await;
                    ReadStreamState::Terminated
                },
            },
            ReadStreamState::BackingOff(last_err) => {
                match backoff_session.next_delay(last_err, &mut cancel_rx).await {
                    Ok(Some(())) => ReadStreamState::Connecting,
                    Ok(None) => ReadStreamState::Terminated,
                    Err(err) => {
                        send_fatal_error(BigQueryError::Grpc(err), &mut schema_tx, &tx, &cancel_tx)
                            .await;
                        ReadStreamState::Terminated
                    },
                }
            },
            ReadStreamState::Terminated => break,
        };
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn read_stream(
    read_rows: ReadRows,
    offset: i64,
    metadata: Option<Arc<StreamMetadata>>,
    schema_tx: Option<SchemaSender>,
    tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    retry_config: StreamRetryConfig,
    decode_pool: Arc<rayon::ThreadPool>,
    cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
) {
    read_stream_inner(
        read_rows,
        offset,
        metadata,
        schema_tx,
        tx,
        retry_config,
        decode_pool,
        cancel_tx,
        cancel_rx,
    )
    .await;
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use google_cloud_bigquery::client::Read;
    use google_cloud_bigquery::model::{ArrowRecordBatch, AvroRows, ReadRowsRequest};
    use google_cloud_gax::backoff_policy::BackoffPolicy;
    use google_cloud_gax::error::rpc::{Code, Status};
    use google_cloud_gax::error::Error;
    use google_cloud_gax::options::RequestOptions;
    use google_cloud_gax::retry_policy::RetryPolicyExt;
    use google_cloud_gax::retry_state::RetryState;
    use google_cloud_gax::streaming::ResponseStream;

    use super::*;
    use crate::bigquery_read_retry::RetryableErrors;

    #[test]
    fn test_read_rows_response_empty() {
        let response = ReadRowsResponse::new();
        assert!(read_rows_response_to_record_batch(response, &[])
            .unwrap()
            .is_none());

        let response2 = ReadRowsResponse::new().set_arrow_record_batch(ArrowRecordBatch::new());
        assert!(read_rows_response_to_record_batch(response2, &[])
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_read_rows_response_protocol_error() {
        let response = ReadRowsResponse::new()
            .set_avro_rows(AvroRows::new())
            .set_row_count(5);
        let err = read_rows_response_to_record_batch(response, &[]).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[test]
    fn test_read_rows_response_arrow_error() {
        let response = ReadRowsResponse::new()
            .set_arrow_record_batch(
                ArrowRecordBatch::new().set_serialized_record_batch(vec![0x00, 0x01, 0x02, 0x03]),
            )
            .set_row_count(5);
        let err = read_rows_response_to_record_batch(response, &[0x00, 0x00]).unwrap_err();
        assert!(matches!(err, BigQueryError::Arrow(_)));
    }

    fn create_test_arrow_payload(num_rows: usize) -> (Vec<u8>, ReadRowsResponse) {
        use polars_arrow::array::Int32Array;
        use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
        use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};

        let field = Field::new("col1".into(), ArrowDataType::Int32, false);
        let schema = ArrowSchema::from_iter(vec![field]);

        let mut schema_bytes = Vec::new();
        {
            let mut writer =
                StreamWriter::new(&mut schema_bytes, WriteOptions { compression: None });
            writer.start(&schema, None).unwrap();
        }
        let schema_len = schema_bytes.len();

        let array = Int32Array::from_slice((0..num_rows as i32).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(
            num_rows,
            Arc::new(schema.clone()),
            vec![Box::new(array) as Box<dyn polars_arrow::array::Array>],
        )
        .unwrap();

        let mut full_stream_bytes = Vec::new();
        {
            let mut writer =
                StreamWriter::new(&mut full_stream_bytes, WriteOptions { compression: None });
            writer.start(&schema, None).unwrap();
            writer.write(&batch, None).unwrap();
        }

        let batch_bytes = full_stream_bytes[schema_len..].to_vec();

        let response = ReadRowsResponse::new()
            .set_arrow_record_batch(
                ArrowRecordBatch::new().set_serialized_record_batch(batch_bytes),
            )
            .set_row_count(num_rows as i64);

        (schema_bytes, response)
    }

    #[test]
    fn test_read_rows_response_success_and_mismatch() {
        let (schema_bytes, response) = create_test_arrow_payload(5);
        let (batch, rows) = read_rows_response_to_record_batch(response.clone(), &schema_bytes)
            .unwrap()
            .unwrap();
        assert_eq!(rows, 5);
        assert_eq!(batch.len(), 5);

        // Test row count mismatch
        let bad_response = response.set_row_count(10); // mismatch with 5 decoded rows
        let err = read_rows_response_to_record_batch(bad_response, &schema_bytes).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[test]
    fn test_read_rows_response_row_count_mismatch_when_reader_empty() {
        let (schema_bytes, _) = create_test_arrow_payload(0);
        // Replace empty serialized_record_batch with an Arrow IPC End-Of-Stream marker (8 bytes)
        // so serialized_record_batch is non-empty, but StreamReader::next() returns None.
        let response =
            ReadRowsResponse::new()
                .set_arrow_record_batch(ArrowRecordBatch::new().set_serialized_record_batch(vec![
                    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00,
                ]))
                .set_row_count(10);
        let err = read_rows_response_to_record_batch(response, &schema_bytes).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[derive(Debug, Default)]
    struct NoBackoff;

    impl BackoffPolicy for NoBackoff {
        fn on_failure(&self, _state: &RetryState) -> Duration {
            Duration::ZERO
        }
    }

    fn test_retry_config(max_attempts: u32) -> StreamRetryConfig {
        use google_cloud_gax::retry_throttler::{CircuitBreaker, RetryThrottlerArg};

        StreamRetryConfig::new(
            Some(Arc::new(RetryableErrors.with_attempt_limit(max_attempts))),
            Some(Arc::new(NoBackoff)),
            RetryThrottlerArg::from(CircuitBreaker::default()).into(),
        )
    }

    type MockStreamResult =
        google_cloud_bigquery::Result<Vec<google_cloud_bigquery::Result<ReadRowsResponse>>>;

    #[derive(Debug)]
    struct MockClient {
        requests: Arc<Mutex<Vec<ReadRowsRequest>>>,
        streams: Arc<Mutex<Vec<MockStreamResult>>>,
    }

    impl google_cloud_bigquery::stub::Read for MockClient {
        async fn read_rows(
            &self,
            req: ReadRowsRequest,
            _options: RequestOptions,
        ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
            self.requests.lock().unwrap().push(req);
            let next_stream = {
                let mut streams = self.streams.lock().unwrap();
                if !streams.is_empty() {
                    streams.remove(0)
                } else {
                    Ok(vec![])
                }
            };
            let messages = next_stream?;
            let (tx, rx) = tokio::sync::mpsc::channel(messages.len().max(1));
            for msg in messages {
                let _ = tx.send(msg).await;
            }
            Ok(ResponseStream::from(rx))
        }
    }

    #[tokio::test]
    async fn test_stream_reconnection_offset() {
        let (schema_bytes, resp1) = create_test_arrow_payload(3);
        let (_, resp2) = create_test_arrow_payload(2);
        let metadata = Arc::new(parse_arrow_schema(&schema_bytes).unwrap());

        let stream1 = Ok(vec![
            Ok(resp1),
            Err(Error::service(
                Status::default()
                    .set_code(Code::Unavailable)
                    .set_message("transient disconnection"),
            )),
        ]);
        let stream2 = Ok(vec![Ok(resp2)]);

        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![stream1, stream2])),
        };

        let client = Read::from_stub(mock_client);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        read_stream(
            read_rows,
            10,
            Some(metadata),
            None,
            tx,
            test_retry_config(3),
            default_decode_pool(),
            Arc::new(cancel_tx),
            cancel_rx,
        )
        .await;

        let batch1 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch1.len(), 3);
        let batch2 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch2.len(), 2);
        assert!(rx.recv().await.is_none());

        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].offset, 10);
        assert_eq!(reqs[1].offset, 13);
    }

    #[tokio::test]
    async fn test_connect_read_rows_stream_retries_transient_error() {
        let (schema_bytes, resp1) = create_test_arrow_payload(4);
        let metadata = Arc::new(parse_arrow_schema(&schema_bytes).unwrap());

        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![
                Err(Error::service(
                    Status::default()
                        .set_code(Code::Unavailable)
                        .set_message("initial connect unavailable"),
                )),
                Ok(vec![Ok(resp1)]),
            ])),
        };

        let client = Read::from_stub(mock_client);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        read_stream(
            read_rows,
            0,
            Some(metadata),
            None,
            tx,
            test_retry_config(3),
            default_decode_pool(),
            Arc::new(cancel_tx),
            cancel_rx,
        )
        .await;

        let batch = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch.len(), 4);
        assert!(rx.recv().await.is_none());

        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].offset, 0);
        assert_eq!(reqs[1].offset, 0);
    }

    #[tokio::test]
    async fn test_stream_unrecoverable_error_terminates_without_retry() {
        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![Ok(vec![Err(Error::service(
                Status::default()
                    .set_code(Code::PermissionDenied)
                    .set_message("permission denied"),
            ))])])),
        };

        let client = Read::from_stub(mock_client);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        read_stream(
            read_rows,
            0,
            None,
            None,
            tx,
            test_retry_config(3),
            default_decode_pool(),
            Arc::clone(&cancel_tx),
            cancel_rx,
        )
        .await;

        let result = rx.recv().await.unwrap();
        assert!(matches!(result, Err(BigQueryError::Grpc(_))));
        assert!(
            *cancel_tx.borrow(),
            "unrecoverable stream error should signal sibling cancellation"
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_stream_reconnection_exhaustion() {
        // Stream repeatedly disconnects immediately without making progress.
        // Verifies the state machine terminates and yields an Err after retries are exhausted.
        #[derive(Debug)]
        struct InfiniteDisconnectClient;

        impl google_cloud_bigquery::stub::Read for InfiniteDisconnectClient {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                let _ = tx
                    .send(Err(Error::service(
                        Status::default()
                            .set_code(Code::Unavailable)
                            .set_message("persistent disconnect"),
                    )))
                    .await;
                Ok(ResponseStream::from(rx))
            }
        }

        let client = Read::from_stub(InfiniteDisconnectClient);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        read_stream(
            read_rows,
            0,
            None,
            None,
            tx,
            test_retry_config(3),
            default_decode_pool(),
            Arc::clone(&cancel_tx),
            cancel_rx,
        )
        .await;

        let result = rx.recv().await.unwrap();
        assert!(matches!(result, Err(BigQueryError::Grpc(_))));
        assert!(
            *cancel_tx.borrow(),
            "fatal stream error should signal sibling cancellation"
        );
    }

    #[tokio::test]
    async fn test_stream_cancellation_terminates_blocked_sibling() {
        #[derive(Debug)]
        struct HangingClient {
            _hold_tx: Arc<
                Mutex<
                    Option<
                        tokio::sync::mpsc::Sender<google_cloud_bigquery::Result<ReadRowsResponse>>,
                    >,
                >,
            >,
        }

        impl google_cloud_bigquery::stub::Read for HangingClient {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                *self._hold_tx.lock().unwrap() = Some(tx);
                Ok(ResponseStream::from(rx))
            }
        }

        let hold_tx = Arc::new(Mutex::new(None));
        let client = Read::from_stub(HangingClient {
            _hold_tx: Arc::clone(&hold_tx),
        });
        let read_rows = client.read_rows().set_read_stream("hanging_stream");

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        let task = tokio::spawn(read_stream(
            read_rows,
            0,
            None,
            None,
            tx,
            test_retry_config(3),
            default_decode_pool(),
            Arc::clone(&cancel_tx),
            cancel_rx,
        ));

        // Signal cancellation as a sibling stream would upon fatal error.
        let _ = cancel_tx.send_replace(true);

        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("cancelled stream should exit promptly")
            .expect("task should not panic");
    }
}
