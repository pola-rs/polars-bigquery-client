use std::io::Cursor;
use std::iter::Iterator;
use std::sync::Arc;

use backon::Retryable;
use gcloud_sdk::google::cloud::bigquery::storage::v1::big_query_read_client::BigQueryReadClient;
use gcloud_sdk::google::cloud::bigquery::storage::v1::{
    read_rows_response, ReadRowsRequest, ReadRowsResponse,
};
use gcloud_sdk::*;
use polars_arrow::io::ipc::read::{read_stream_metadata, StreamReader, StreamState};
use polars_arrow::record_batch::RecordBatch;

use super::bigquery_read_retry;
use crate::BigQueryError;

fn read_rows_response_to_record_batch(
    response: ReadRowsResponse,
    schema: &[u8],
) -> Result<Option<(RecordBatch, i64)>, BigQueryError> {
    let row_count = response.row_count;

    let mut buffer = Vec::new();
    buffer.extend_from_slice(schema);

    let mut serialized_record_batch = match response.rows {
        Some(read_rows_response::Rows::ArrowRecordBatch(value)) => value.serialized_record_batch,
        None => return Ok(None),
        _ => {
            return Err(BigQueryError::Protocol(
                "Unexpectedly got some format other than arrow bytes".into(),
            ))
        }
    };

    if serialized_record_batch.is_empty() || row_count == 0 {
        return Ok(None);
    }
    buffer.append(&mut serialized_record_batch);

    let mut cursor = Cursor::new(buffer);
    let metadata = match
        read_stream_metadata(&mut cursor) {
            Ok(metadata) => metadata,
            Err(e) => return Err(BigQueryError::Arrow(e))
        };
    let mut reader = StreamReader::new(cursor, metadata, None);

    match reader.next() {
        Some(Ok(StreamState::Some(batch))) => Ok(Some((batch, row_count))),
        Some(Ok(StreamState::Waiting)) | None => Ok(None),
        Some(Err(e)) => return Err(BigQueryError::Arrow(e))
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
    let mut current_offset = 0i64;

    loop {
        let read_rows_request = ReadRowsRequest {
            read_stream: stream_name.clone(),
            offset: current_offset,
        };

        let read_rows_response =
            (|| async { read_client.get().read_rows(read_rows_request.clone()).await })
                .retry(bigquery_read_retry::READ_ROWS_RETRY)
                .sleep(tokio::time::sleep)
                .when(bigquery_read_retry::read_rows_predicate)
                .await;

        let mut messages = match read_rows_response {
            Ok(messages) => messages.into_inner(),
            Err(status) => {
                let _ = tx.send(Err(BigQueryError::Grpc(status))).await;
                return;
            }
        };

        'messages: loop {
            match messages.message().await {
                Ok(Some(value)) => {
                    match read_rows_response_to_record_batch(
                        value,
                        &schema,
                    ) {
                        Ok(Some((batch, row_count))) => {
                            current_offset += row_count;
                            if tx.send(Ok(batch)).await.is_err() {
                                // Receiver dropped on consumer side, stop reading.
                                return;
                            }
                        }
                        Ok(None) => {} // Skip empty chunks / control payloads
                        Err(err) => {
                            let _ = tx.send(Err(err)).await;
                            return;
                        }
                    }
                }
                Ok(None) => {
                    // Stream finished cleanly.
                    return;
                }
                Err(status) => {
                    if bigquery_read_retry::reconnect_stream_predicate(&status) {
                        // Transient gRPC error mid-stream, break inner loop to retry read_rows at current_offset.
                        break 'messages;
                    } else {
                        let _ = tx.send(Err(BigQueryError::Grpc(status))).await;
                        return;
                    }
                }
            }
        }
    }
}
