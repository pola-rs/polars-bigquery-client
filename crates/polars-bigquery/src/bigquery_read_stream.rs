use std::io::Cursor;
use std::iter::Iterator;
use std::sync::Arc;

use gcloud_sdk::google::cloud::bigquery::storage::v1::big_query_read_client::BigQueryReadClient;
use gcloud_sdk::google::cloud::bigquery::storage::v1::{
    read_rows_response, ReadRowsRequest,
    ReadRowsResponse,
};
use gcloud_sdk::*;
use polars_arrow::io::ipc::read::{read_stream_metadata, StreamReader, StreamState};
use polars_arrow::record_batch::RecordBatch;

fn read_rows_response_to_record_batch(response: ReadRowsResponse, schema: &[u8]) -> RecordBatch {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(schema);
    // TODO: Bubble up if we unexpectedly get a record batch with no rows.
    // TODO: This might not actually be unexpected? What happens when there's a
    // super selective row filter?
    let mut serialized_record_batch = match response.rows.unwrap() {
        read_rows_response::Rows::ArrowRecordBatch(value) => value.serialized_record_batch,
        _ => panic!("unexpectedly got some format other than arrow bytes"),
    };
    buffer.append(&mut serialized_record_batch);

    let mut cursor = Cursor::new(buffer);
    let metadata = read_stream_metadata(&mut cursor).unwrap();
    let mut reader = StreamReader::new(cursor, metadata, None);

    // TODO: maybe double-check that there are no recordbatches after this?
    // There should only be one if the API returned the expected results.
    match reader.next().unwrap().unwrap() {
        StreamState::Some(batch) => batch,
        _ => panic!("expected a batch"),
    }
}

pub async fn read_stream<B>(
    read_client: Arc<GoogleApiClient<B, BigQueryReadClient<GoogleAuthMiddleware>>>,
    schema: Arc<Vec<u8>>,
    stream_name: String,
    tx: tokio::sync::mpsc::Sender<RecordBatch>,
) where
    B: GoogleApiClientBuilder<BigQueryReadClient<GoogleAuthMiddleware>> + Send + Sync + 'static,
{
    let read_rows_request = ReadRowsRequest {
        read_stream: stream_name.clone(),
        offset: 0,
    };

    let messages = read_client
        .get()
        .read_rows(read_rows_request)
        .await
        .unwrap();
    let mut messages = messages.into_inner();

    'messages: loop {
        // TODO: if there's an error, call read_rows with the most recent
        // offset to resume.
        let message = messages.message().await.unwrap();
        match message {
            Some(value) => {
                let batch = read_rows_response_to_record_batch(value, &schema);
                if tx.send(batch).await.is_err() {
                    // Receiver was dropped, stop reading.
                    break 'messages;
                }
            },
            None => {
                break 'messages;
            },
        }
    }
}
