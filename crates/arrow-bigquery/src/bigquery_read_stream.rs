//! ## bigquery_read_stream.rs
//!
//! This module reads from BQ Storage Read API stream and writes the results
//! as arrow record batches to a tokio mpsc queue as messages arrive.

use std::io::Cursor;
use std::iter::Iterator;

use google_cloud_bigquery::model::{read_rows_response, ReadRowsResponse};
use google_cloud_bigquery::read::Reader;
use polars_arrow::io::ipc::read::{read_stream_metadata, StreamReader, StreamState};
use polars_arrow::record_batch::RecordBatch;

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

    let mut cursor = Cursor::new(schema);
    let metadata = read_stream_metadata(&mut cursor)?;
    let cursor = Cursor::new(serialized_record_batch);
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

pub async fn read_stream(
    mut reader: Reader,
    schema: bytes::Bytes,
    tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
) {
    while let Some(res) = reader.next().await {
        match res {
            Ok(value) => match read_rows_response_to_record_batch(value, &schema) {
                Ok(Some((batch, _row_count))) => {
                    if tx.send(Ok(batch)).await.is_err() {
                        // `tx.send` returns `Err` strictly when all `Receiver` handles (`rx`) have been
                        // dropped. This happens when either:
                        // 1) The consumer aborted reading early (e.g. stopped iteration or dropped receiver), or
                        // 2) Another concurrent stream sent an `Err(...)` over `tx`, prompting the consumer
                        //    to raise an exception and drop `rx`.
                        // In either case, the consumer closed the channel and cannot receive more batches,
                        // so terminating this stream cleanly prevents orphan background tasks.
                        break;
                    }
                },
                Ok(None) => {},
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                    break;
                },
            },
            Err(err) => {
                let _ = tx.send(Err(BigQueryError::Grpc(err))).await;
                break;
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use google_cloud_bigquery::client::Read;
    use google_cloud_bigquery::model::{ArrowRecordBatch, AvroRows, ReadRowsRequest};
    use google_cloud_bigquery::read::retry_policy::RetryableErrors;
    use google_cloud_gax::backoff_policy::BackoffPolicy;
    use google_cloud_gax::error::rpc::{Code, Status};
    use google_cloud_gax::error::Error;
    use google_cloud_gax::options::RequestOptions;
    use google_cloud_gax::retry_policy::RetryPolicyExt;
    use google_cloud_gax::retry_state::RetryState;
    use google_cloud_gax::streaming::ResponseStream;

    use super::*;

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
        let response = ReadRowsResponse::new()
            .set_arrow_record_batch(
                ArrowRecordBatch::new()
                    .set_serialized_record_batch(vec![0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]),
            )
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

    #[derive(Debug)]
    struct MockClient {
        requests: Arc<Mutex<Vec<ReadRowsRequest>>>,
        streams: Arc<Mutex<Vec<Vec<google_cloud_bigquery::Result<ReadRowsResponse>>>>>,
    }

    impl google_cloud_bigquery::stub::Read for MockClient {
        async fn read_rows(
            &self,
            req: ReadRowsRequest,
            _options: RequestOptions,
        ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
            self.requests.lock().unwrap().push(req);
            let messages = {
                let mut streams = self.streams.lock().unwrap();
                if !streams.is_empty() {
                    streams.remove(0)
                } else {
                    vec![]
                }
            };
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

        let stream1 = vec![
            Ok(resp1),
            Err(Error::service(
                Status::default()
                    .set_code(Code::Unavailable)
                    .set_message("transient disconnection"),
            )),
        ];
        let stream2 = vec![Ok(resp2)];

        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![stream1, stream2])),
        };

        let client = Read::from_stub(mock_client);
        let reader = client
            .read_rows()
            .set_read_stream("test_stream")
            .into_reader()
            .with_backoff_policy(NoBackoff);

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        read_stream(reader, bytes::Bytes::from(schema_bytes), tx).await;

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
        // Verifies the reader terminates and yields an Err after retries are exhausted.
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
        let reader = client
            .read_rows()
            .set_read_stream("test_stream")
            .into_reader()
            .with_retry_policy(RetryableErrors.with_attempt_limit(3))
            .with_backoff_policy(NoBackoff);

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        read_stream(reader, bytes::Bytes::new(), tx).await;

        let result = rx.recv().await.unwrap();
        assert!(matches!(result, Err(BigQueryError::Grpc(_))));
    }
}
