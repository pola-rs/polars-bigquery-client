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
use std::iter::Iterator;
use std::sync::Arc;

use gcloud_sdk::google::cloud::bigquery::storage::v1::big_query_read_client::BigQueryReadClient;
use gcloud_sdk::google::cloud::bigquery::storage::v1::{
    read_rows_response, ReadRowsRequest, ReadRowsResponse,
};
use gcloud_sdk::*;
use polars_arrow::io::ipc::read::{read_stream_metadata, StreamReader, StreamState};
use polars_arrow::record_batch::RecordBatch;
use tower::ServiceExt;

use super::bigquery_read_retry;
use crate::BigQueryError;

/// Convert a ReadRowsResponse into an Arrow record batch + stream offset.
///
/// Returns an error if we got an invalid message. Rather than try to continue
/// in an invalid state, this should terminate the reader.
fn read_rows_response_to_record_batch(
    response: ReadRowsResponse,
    schema: &[u8],
) -> Result<Option<(RecordBatch, i64)>, BigQueryError> {
    let row_count = response.row_count;

    let mut serialized_record_batch = match response.rows {
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

    let mut buffer = Vec::with_capacity(schema.len() + serialized_record_batch.len());
    buffer.extend_from_slice(schema);
    buffer.append(&mut serialized_record_batch);

    let mut cursor = Cursor::new(buffer);
    let metadata = read_stream_metadata(&mut cursor)?;
    let mut reader = StreamReader::new(cursor, metadata, None);

    match reader.next() {
        Some(Ok(StreamState::Some(batch))) => {
            let actual_rows = batch.len() as i64;
            if actual_rows != row_count {
                return Err(BigQueryError::Protocol(format!(
                    "Row count mismatch: gRPC protobuf reported {} rows, but Arrow IPC decoded {} rows",
                    row_count, actual_rows
                )));
            }
            Ok(Some((batch, actual_rows)))
        },
        Some(Ok(StreamState::Waiting)) | None => {
            if row_count != 0 {
                return Err(BigQueryError::Protocol(format!(
                    "Row count mismatch: gRPC protobuf reported {} rows, but Arrow IPC decoded 0 rows",
                    row_count
                )));
            }
            Ok(None)
        },
        Some(Err(e)) => Err(BigQueryError::Arrow(e)),
    }
}

/// Abstraction over a gRPC stream yielding [`ReadRowsResponse`] messages.
///
/// Decoupling the streaming loop from concrete [`gcloud_sdk::tonic::Streaming`] allows unit tests
/// to deterministically simulate mid-stream gRPC disconnections (`tonic::Status`) without requiring
/// network sockets or complex HTTP/2 server mocking.
///
/// `#[async_trait]` is used to guarantee `Send` bounds across Tokio tasks, matching the `tonic` ecosystem.
#[gcloud_sdk::tonic::async_trait]
pub trait ReadRowsStreamTrait: Send + Unpin {
    async fn next_message(&mut self)
        -> Result<Option<ReadRowsResponse>, gcloud_sdk::tonic::Status>;
}

#[gcloud_sdk::tonic::async_trait]
impl ReadRowsStreamTrait for gcloud_sdk::tonic::Streaming<ReadRowsResponse> {
    async fn next_message(
        &mut self,
    ) -> Result<Option<ReadRowsResponse>, gcloud_sdk::tonic::Status> {
        self.message().await
    }
}

/// Abstraction over the BigQuery Storage Read API client for establishing row streams.
///
/// Decoupling client creation from [`GoogleApiClient`] allows unit tests to inspect the [`ReadRowsRequest`]
/// parameters sent during stream reconnections (verifying that `offset` resumption is correct).
///
/// In production, `read_stream_inner` consumes this via static dispatch (`C: BigQueryReadClientTrait`),
/// ensuring zero dynamic dispatch (`dyn`) overhead.
#[gcloud_sdk::tonic::async_trait]
pub trait BigQueryReadClientTrait: Send + Sync {
    type Stream: ReadRowsStreamTrait;
    async fn read_rows_stream(
        &self,
        request: ReadRowsRequest,
    ) -> Result<Self::Stream, gcloud_sdk::tonic::Status>;
}

#[gcloud_sdk::tonic::async_trait]
impl<B> BigQueryReadClientTrait for GoogleApiClient<B, BigQueryReadClient<GoogleAuthMiddleware>>
where
    B: GoogleApiClientBuilder<BigQueryReadClient<GoogleAuthMiddleware>> + Send + Sync + 'static,
{
    type Stream = gcloud_sdk::tonic::Streaming<ReadRowsResponse>;
    async fn read_rows_stream(
        &self,
        request: ReadRowsRequest,
    ) -> Result<Self::Stream, gcloud_sdk::tonic::Status> {
        let resp = self.get().read_rows(request).await?;
        Ok(resp.into_inner())
    }
}

pub async fn read_stream<B>(
    read_client: Arc<GoogleApiClient<B, BigQueryReadClient<GoogleAuthMiddleware>>>,
    schema: Arc<Vec<u8>>,
    stream_name: String,
    tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
) where
    B: GoogleApiClientBuilder<BigQueryReadClient<GoogleAuthMiddleware>> + Send + Sync + 'static,
{
    read_stream_inner(
        read_client.as_ref(),
        schema,
        stream_name,
        tx,
        bigquery_read_retry::stream_reconnect_policy(),
    )
    .await;
}

/// Represents the state of the stream reading state machine.
#[derive(Debug)]
enum ReadStreamState {
    /// Establishing or resuming the BigQuery read stream at `current_offset`.
    Connecting,
    /// Waiting for the next message in a BigQuery read stream.
    Running,
    /// Encountered a transient mid-stream disconnection; backing off before reconnecting.
    BackingOff(gcloud_sdk::tonic::Status),
    /// Stream completed cleanly, fatal error occurred, or consumer dropped the receiver.
    Terminated,
}

/// Layer 1: Connects to a BigQuery read stream at `offset`, retrying transient gRPC connection errors.
async fn connect_read_rows_stream<C: BigQueryReadClientTrait>(
    read_client: &C,
    stream_name: &str,
    offset: i64,
) -> Result<C::Stream, gcloud_sdk::tonic::Status> {
    let read_rows_request = ReadRowsRequest {
        read_stream: stream_name.to_string(),
        offset,
    };

    let policy = bigquery_read_retry::read_rows_policy();
    let service =
        tower::service_fn(
            |req: ReadRowsRequest| async move { read_client.read_rows_stream(req).await },
        );
    tower::retry::Retry::new(policy, service)
        .oneshot(read_rows_request)
        .await
}

/// Layer 2: Consumes a single message from an established read stream and returns the next [`ReadStreamState`].
///
/// Processes the next message from an active read stream while in [`ReadStreamState::Running`].
async fn consume_next_message<S: ReadRowsStreamTrait>(
    stream: &mut S,
    schema: &[u8],
    current_offset: &mut i64,
    tx: &tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
) -> ReadStreamState {
    match stream.next_message().await {
        Ok(Some(value)) => match read_rows_response_to_record_batch(value, schema) {
            Ok(Some((batch, row_count))) => {
                *current_offset += row_count;
                if tx.send(Ok(batch)).await.is_err() {
                    // `tx.send` returns `Err` strictly when all `Receiver` handles (`rx`) have been
                    // dropped. This happens when either:
                    // 1) The consumer aborted reading early (e.g. stopped iteration or dropped receiver), or
                    // 2) Another concurrent stream sent an `Err(...)` over `tx`, prompting the consumer
                    //    to raise an exception and drop `rx`.
                    // In either case, the consumer closed the channel and cannot receive more batches,
                    // so terminating this stream cleanly prevents orphan background tasks.
                    ReadStreamState::Terminated
                } else {
                    ReadStreamState::Running
                }
            },
            Ok(None) => ReadStreamState::Running,
            Err(err) => {
                let _ = tx.send(Err(err)).await;
                ReadStreamState::Terminated
            },
        },
        Ok(None) => ReadStreamState::Terminated,
        Err(status) => {
            if bigquery_read_retry::reconnect_stream_predicate(&status) {
                ReadStreamState::BackingOff(status)
            } else {
                let _ = tx.send(Err(BigQueryError::Grpc(status))).await;
                ReadStreamState::Terminated
            }
        },
    }
}

/// Layer 3: Orchestrates stream connection, consumption, and mid-stream reconnections as an explicit state machine.
///
/// Uses a [`bigquery_read_retry::BackoffSession`] instantiated from `retry_policy` to handle backoff delays
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
/// (`backoff_session = retry_policy.make_session()`).  This ensures that
/// long-lived data streams experiencing rare, transient network interruptions
/// separated by minutes or hours always receive a full retry budget.
pub(crate) async fn read_stream_inner<C, P, R>(
    read_client: &C,
    schema: Arc<Vec<u8>>,
    stream_name: String,
    tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    retry_policy: bigquery_read_retry::RetryPolicy<P, R>,
) where
    C: BigQueryReadClientTrait,
    P: Fn(&gcloud_sdk::tonic::Status) -> bool + Clone,
    R: tower::util::rng::Rng + Clone,
{
    let mut current_offset = 0i64;
    let mut backoff_session = retry_policy.make_session();
    let mut state = ReadStreamState::Connecting;
    let mut stream: Option<C::Stream> = None;

    while !matches!(state, ReadStreamState::Terminated) {
        state = match state {
            ReadStreamState::Connecting => {
                let connection =
                    connect_read_rows_stream(read_client, &stream_name, current_offset).await;
                match connection {
                    Ok(inner_stream) => {
                        stream = Some(inner_stream);
                        ReadStreamState::Running
                    },
                    Err(status) => {
                        let _ = tx.send(Err(BigQueryError::Grpc(status))).await;
                        ReadStreamState::Terminated
                    },
                }
            },
            ReadStreamState::Running => {
                match stream {
                    Some(ref mut inner_stream) => {
                        let prev_offset = current_offset;
                        let next_state =
                            consume_next_message(inner_stream, &schema, &mut current_offset, &tx)
                                .await;
                        // Reset backoff state after making data progress so
                        // future transient errors get a full retry budget.
                        if current_offset > prev_offset {
                            backoff_session = retry_policy.make_session();
                        }
                        next_state
                    },
                    None => {
                        let _ = tx
                            .send(Err(BigQueryError::Other(
                                format!("Tried to read from {} but not connected", stream_name)
                                    .into(),
                            )))
                            .await;
                        ReadStreamState::Terminated
                    },
                }
            },
            ReadStreamState::BackingOff(last_status) => {
                if backoff_session.next_delay().await {
                    ReadStreamState::Connecting
                } else {
                    let _ = tx.send(Err(BigQueryError::Grpc(last_status))).await;
                    ReadStreamState::Terminated
                }
            },
            ReadStreamState::Terminated => break,
        };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use gcloud_sdk::google::cloud::bigquery::storage::v1::{
        read_rows_response, ArrowRecordBatch, AvroRows,
    };
    use gcloud_sdk::tonic::{Code, Status};

    use super::*;

    #[test]
    fn test_read_rows_response_empty() {
        let response = ReadRowsResponse {
            rows: None,
            row_count: 0,
            ..Default::default()
        };
        assert!(read_rows_response_to_record_batch(response, &[])
            .unwrap()
            .is_none());

        let response2 = ReadRowsResponse {
            rows: Some(read_rows_response::Rows::ArrowRecordBatch(
                ArrowRecordBatch {
                    serialized_record_batch: vec![],
                    ..Default::default()
                },
            )),
            row_count: 0,
            ..Default::default()
        };
        assert!(read_rows_response_to_record_batch(response2, &[])
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_read_rows_response_protocol_error() {
        let response = ReadRowsResponse {
            rows: Some(read_rows_response::Rows::AvroRows(AvroRows::default())),
            row_count: 5,
            ..Default::default()
        };
        let err = read_rows_response_to_record_batch(response, &[]).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[test]
    fn test_read_rows_response_arrow_error() {
        let response = ReadRowsResponse {
            rows: Some(read_rows_response::Rows::ArrowRecordBatch(
                ArrowRecordBatch {
                    serialized_record_batch: vec![0x00, 0x01, 0x02, 0x03],
                    ..Default::default()
                },
            )),
            row_count: 5,
            ..Default::default()
        };
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

        let response = ReadRowsResponse {
            rows: Some(read_rows_response::Rows::ArrowRecordBatch(
                ArrowRecordBatch {
                    serialized_record_batch: batch_bytes,
                    ..Default::default()
                },
            )),
            row_count: num_rows as i64,
            ..Default::default()
        };

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
        let mut bad_response = response;
        bad_response.row_count = 10; // mismatch with 5 decoded rows
        let err = read_rows_response_to_record_batch(bad_response, &schema_bytes).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    struct MockStream {
        messages: Vec<Result<Option<ReadRowsResponse>, Status>>,
    }

    #[gcloud_sdk::tonic::async_trait]
    impl ReadRowsStreamTrait for MockStream {
        async fn next_message(&mut self) -> Result<Option<ReadRowsResponse>, Status> {
            if !self.messages.is_empty() {
                self.messages.remove(0)
            } else {
                Ok(None)
            }
        }
    }

    struct MockClient {
        requests: Arc<Mutex<Vec<ReadRowsRequest>>>,
        streams: Arc<Mutex<Vec<Result<MockStream, Status>>>>,
    }

    #[gcloud_sdk::tonic::async_trait]
    impl BigQueryReadClientTrait for MockClient {
        type Stream = MockStream;
        async fn read_rows_stream(&self, request: ReadRowsRequest) -> Result<Self::Stream, Status> {
            self.requests.lock().unwrap().push(request);
            let mut streams = self.streams.lock().unwrap();
            if !streams.is_empty() {
                streams.remove(0)
            } else {
                Ok(MockStream { messages: vec![] })
            }
        }
    }

    fn test_policy(
    ) -> bigquery_read_retry::RetryPolicy<fn(&Status) -> bool, tower::util::rng::HasherRng> {
        bigquery_read_retry::RetryPolicy::new(
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(1),
            1.3,
            0.0,
            tower::util::rng::HasherRng::default(),
            bigquery_read_retry::reconnect_stream_predicate as fn(&Status) -> bool,
        )
        .with_max_times(3)
    }

    #[test]
    fn test_read_rows_response_row_count_mismatch_when_reader_empty() {
        let (schema_bytes, mut response) = create_test_arrow_payload(0);
        // Replace empty serialized_record_batch with an Arrow IPC End-Of-Stream marker (8 bytes)
        // so serialized_record_batch is non-empty, but StreamReader::next() returns None.
        if let Some(read_rows_response::Rows::ArrowRecordBatch(ref mut arrow_batch)) = response.rows
        {
            arrow_batch.serialized_record_batch =
                vec![0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
        }
        response.row_count = 10;
        let err = read_rows_response_to_record_batch(response, &schema_bytes).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[tokio::test]
    async fn test_stream_reconnection_offset() {
        let (schema_bytes, resp1) = create_test_arrow_payload(3);
        let (_, resp2) = create_test_arrow_payload(2);

        let stream1 = MockStream {
            messages: vec![
                Ok(Some(resp1)),
                Err(Status::new(Code::Unavailable, "transient disconnection")),
            ],
        };
        let stream2 = MockStream {
            messages: vec![Ok(Some(resp2)), Ok(None)],
        };

        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![Ok(stream1), Ok(stream2)])),
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        read_stream_inner(
            &mock_client,
            Arc::new(schema_bytes),
            "test_stream".into(),
            tx,
            test_policy(),
        )
        .await;

        let batch1 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch1.len(), 3);
        let batch2 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch2.len(), 2);
        assert!(rx.recv().await.is_none());

        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].offset, 0);
        assert_eq!(reqs[1].offset, 3);
    }

    #[tokio::test]
    async fn test_stream_reconnection_exhaustion() {
        // Stream repeatedly disconnects immediately without making progress.
        // Verifies the state machine terminates and yields an Err after retries are exhausted.
        struct InfiniteDisconnectClient;

        #[gcloud_sdk::tonic::async_trait]
        impl BigQueryReadClientTrait for InfiniteDisconnectClient {
            type Stream = MockStream;
            async fn read_rows_stream(
                &self,
                _request: ReadRowsRequest,
            ) -> Result<Self::Stream, Status> {
                Ok(MockStream {
                    messages: vec![Err(Status::new(Code::Unavailable, "persistent disconnect"))],
                })
            }
        }

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        read_stream_inner(
            &InfiniteDisconnectClient,
            Arc::new(vec![]),
            "test_stream".into(),
            tx,
            test_policy(),
        )
        .await;

        let result = rx.recv().await.unwrap();
        assert!(matches!(result, Err(BigQueryError::Grpc(_))));
    }
}
