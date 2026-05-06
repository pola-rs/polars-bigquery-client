use std::io::Cursor;
use std::iter::Iterator;
use std::sync::Arc;

use gcloud_sdk::google::cloud::bigquery::storage::v1::big_query_read_client::BigQueryReadClient;
use gcloud_sdk::google::cloud::bigquery::storage::v1::{
    read_rows_response, read_session, CreateReadSessionRequest, DataFormat, ReadRowsRequest,
    ReadRowsResponse, ReadSession,
};
use gcloud_sdk::google::shopping::merchant::quota;
use gcloud_sdk::*;
use polars::prelude::*;
use polars_io::ipc::IpcStreamReader;
use polars_io::SerReader;

fn read_rows_response_to_record_batch(response: ReadRowsResponse, schema: &Vec<u8>) -> DataFrame {
    let mut buffer = Vec::new();
    buffer.append(&mut schema.clone());
    // TODO: Bubble up if we unexpectedly get a record batch with no rows.
    // TODO: This might not actually be unexpected? What happens when there's a
    // super selective row filter?
    let mut serialized_record_batch = match response.rows.unwrap() {
        read_rows_response::Rows::ArrowRecordBatch(value) => value.serialized_record_batch,
        _ => panic!("unexpectedly got some format other than arrow bytes"),
    };
    buffer.append(&mut serialized_record_batch);

    let cursor = Cursor::new(buffer);
    let reader = IpcStreamReader::new(cursor);

    // TODO: maybe double-check that there are no recordbatches after this?
    // There should only be one if the API returned the expected results.
    reader.finish().unwrap()
}

async fn read_stream(
    read_client: Arc<GoogleApi<BigQueryReadClient<GoogleAuthMiddleware>>>,
    schema: Arc<Vec<u8>>,
    stream_name: String,
    tx: Arc<tokio::sync::mpsc::Sender<DataFrame>>,
) {
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
                // If there's an error here, that means the receiver dropped, so
                // we should just exit rather than try to restart the stream at
                // offset.
                tx.send(read_rows_response_to_record_batch(value, &schema))
                    .await
                    .unwrap();
            },
            None => {
                break 'messages;
            },
        }
    }
}

#[derive(Debug, Clone)]
struct InvalidTableId;

impl std::fmt::Display for InvalidTableId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "invalid table id")
    }
}

impl std::error::Error for InvalidTableId {}

fn table_id_to_table_path(table_id: &str) -> Result<String, Box<dyn std::error::Error>> {
    let re = regex::Regex::new(r"(?<project>.+)\.(?<dataset>[^.]+)\.(?<table>[^.]+)")?;
    let caps = re.captures(table_id).ok_or(InvalidTableId)?;
    Ok(format!(
        "projects/{}/datasets/{}/tables/{}",
        &caps["project"], &caps["dataset"], &caps["table"]
    ))
}

static INIT_CRYPTO: std::sync::Once = std::sync::Once::new();

pub async fn read_bigquery_async(
    table_id: &str,
    quota_project_id: &str,
    is_ordered: bool,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    // TODO(tswast): Set the user-agent header. See
    // https://github.com/abdolence/gcloud-sdk-rs/issues/226 for some attempts
    // so far.
    let read_client: GoogleApi<BigQueryReadClient<GoogleAuthMiddleware>> =
        GoogleApi::from_function(
            // Maximum row size in BigQuery is 100 MB, so this should allow for
            // the largest possible row plus some overhead.
            |inner| BigQueryReadClient::new(inner).max_decoding_message_size(128 * 1024 * 1024),
            "https://bigquerystorage.googleapis.com",
            None,
        )
        .await?;

    let read_session = ReadSession {
        data_format: DataFormat::Arrow as i32,
        table: table_id_to_table_path(table_id)?,
        ..Default::default()
    };

    let request = CreateReadSessionRequest {
        parent: format!("projects/{quota_project_id}"),
        // If you are reading from a query results table where order matters,
        // limit this to a single stream.
        max_stream_count: if is_ordered {
            1
        } else {
            match std::thread::available_parallelism() {
                Ok(value) => value.get() as i32,
                Err(_) => 1,
            }
        },
        read_session: Some(read_session),
        ..Default::default()
    };

    let read_session = read_client
        .get()
        .create_read_session(request)
        .await?
        .into_inner();
    let schema = match read_session.schema.unwrap() {
        read_session::Schema::ArrowSchema(value) => value.serialized_schema,
        _ => panic!("unexpectedly got schema type other than arrow"),
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel(1024); // Create an MPSC channel
    let shared_tx = Arc::new(tx);
    let shared_client = Arc::new(read_client);
    let shared_schema = Arc::new(schema);
    let mut handles = Vec::new();

    for stream in read_session.streams {
        let stream_name = stream.name;
        let handle = tokio::task::spawn(read_stream(
            shared_client.clone(),
            shared_schema.clone(),
            stream_name,
            shared_tx.clone(),
        ));
        handles.push(handle);
    }

    // Don't need the sender here anymore. Drop it so that the receiver can know
    // when all the other uses of sender have finished.
    std::mem::drop(shared_tx);

    let mut batches = Vec::new();

    loop {
        match rx.recv().await {
            Some(value) => batches.push(value),
            None => break,
        }
    }

    // Wait for all spawned threads to complete
    for handle in handles {
        handle.await?;
    }

    let concat_args = UnionArgs {
        parallel: true,
        rechunk: true,
        to_supertypes: false,
        diagonal: false,
        from_partitioned_ds: false,
        maintain_order: false,
        strict: false,
    };
    let combined = polars::prelude::concat(
        batches
            .into_iter()
            .map(|df| df.lazy())
            .collect::<Vec<LazyFrame>>(),
        concat_args,
    );
    let collected = combined?.collect();
    match collected {
        Ok(value) => Ok(value),
        Err(err) => Err(Box::new(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_id_to_table_path_success() -> Result<(), Box<dyn std::error::Error>> {
        let result = table_id_to_table_path("my-project.my_dataset.my_table")?;
        assert_eq!(
            result,
            "projects/my-project/datasets/my_dataset/tables/my_table"
        );
        Ok(())
    }

    #[test]
    fn table_id_to_table_path_success_legacy_project() -> Result<(), Box<dyn std::error::Error>> {
        let result = table_id_to_table_path("google.com:my-project.my_dataset.my_table")?;
        assert_eq!(
            result,
            "projects/google.com:my-project/datasets/my_dataset/tables/my_table"
        );
        Ok(())
    }
}
