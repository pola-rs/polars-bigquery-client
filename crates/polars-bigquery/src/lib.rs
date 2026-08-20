mod bigquery_read_retry;
mod bigquery_read_stream;
pub mod client_builder;
mod error;

pub use client_builder::*;
pub use error::BigQueryError;

use std::io::Cursor;
use std::sync::Arc;

use gcloud_sdk::google::cloud::bigquery::storage::v1::big_query_read_client::BigQueryReadClient;
use gcloud_sdk::google::cloud::bigquery::storage::v1::{
    read_session, CreateReadSessionRequest, DataFormat, ReadSession,
};
use gcloud_sdk::{GoogleApiClient, GoogleApiClientBuilder, GoogleAuthMiddleware};
use polars_arrow::datatypes::ArrowSchemaRef;
use polars_arrow::io::ipc::read::read_stream_metadata;
use polars_arrow::record_batch::RecordBatch;
use tower::ServiceExt;

/// A receiver that yields [`RecordBatch`]es read from BigQuery.
///
/// It manages the background tasks reading from the BigQuery Storage API streams
/// and provides a stream-like interface to receive the data.
pub struct BigQueryRecordBatchReceiver {
    /// The channel receiver for receiving [`RecordBatch`]es produced by the background tasks.
    rx: tokio::sync::mpsc::Receiver<Result<RecordBatch, BigQueryError>>,
    /// Join handles for the background tasks reading from the BigQuery streams.
    ///
    /// These handles are kept so that the background tasks can be aborted when
    /// the receiver is dropped, preventing resource leaks from orphan background tasks.
    _handles: Vec<tokio::task::JoinHandle<()>>,
}

impl BigQueryRecordBatchReceiver {
    pub async fn recv(&mut self) -> Option<Result<RecordBatch, BigQueryError>> {
        self.rx.recv().await
    }

    /// Creates a placeholder receiver for testing purposes.
    pub fn new_for_testing(
        rx: tokio::sync::mpsc::Receiver<Result<RecordBatch, BigQueryError>>,
        handles: Vec<tokio::task::JoinHandle<()>>,
    ) -> Self {
        Self {
            rx,
            _handles: handles,
        }
    }
}

impl Drop for BigQueryRecordBatchReceiver {
    fn drop(&mut self) {
        for handle in &self._handles {
            handle.abort();
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

impl From<InvalidTableId> for BigQueryError {
    fn from(e: InvalidTableId) -> Self {
        Self::Other(Box::new(e))
    }
}

impl From<regex::Error> for BigQueryError {
    fn from(e: regex::Error) -> Self {
        Self::Other(Box::new(e))
    }
}

fn table_id_to_table_path(table_id: &str) -> Result<String, BigQueryError> {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"(?<project>.+)\.(?<dataset>[^.]+)\.(?<table>[^.]+)")
            .expect("valid regex pattern")
    });
    let caps = re.captures(table_id).ok_or(InvalidTableId)?;
    Ok(format!(
        "projects/{}/datasets/{}/tables/{}",
        &caps["project"], &caps["dataset"], &caps["table"]
    ))
}

pub async fn read_bigquery_with_client<B>(
    read_client: GoogleApiClient<B, BigQueryReadClient<GoogleAuthMiddleware>>,
    table_id: &str,
    quota_project_id: &str,
    maintain_order: bool,
) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError>
where
    B: GoogleApiClientBuilder<BigQueryReadClient<GoogleAuthMiddleware>> + Send + Sync + 'static,
{
    let read_session = ReadSession {
        data_format: DataFormat::Arrow as i32,
        table: table_id_to_table_path(table_id)?,
        ..Default::default()
    };

    let request = CreateReadSessionRequest {
        parent: format!("projects/{quota_project_id}"),
        // If you are reading from a query results table where order matters,
        // limit this to a single stream.
        max_stream_count: if maintain_order {
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

    let policy = bigquery_read_retry::create_read_session_policy();
    let service = tower::service_fn(|req: CreateReadSessionRequest| {
        let mut client = read_client.get();
        async move { client.create_read_session(req).await }
    });
    let read_session = tower::retry::Retry::new(policy, service)
        .oneshot(request)
        .await?
        .into_inner();
    let schema = match read_session.schema {
        Some(read_session::Schema::ArrowSchema(value)) => value.serialized_schema,
        _ => {
            return Err(BigQueryError::Protocol(
                "Unexpectedly got schema type other than arrow".into(),
            ))
        },
    };

    let mut schema_cursor = Cursor::new(schema.clone());
    let metadata = read_stream_metadata(&mut schema_cursor)?;
    let schema_ref = Arc::new(metadata.schema);

    let channel_size = match std::thread::available_parallelism() {
        Ok(value) => value.get() * 2,
        Err(_) => 2,
    };
    let (tx, rx) = tokio::sync::mpsc::channel(channel_size);
    let shared_client = Arc::new(read_client);
    let shared_schema = Arc::new(schema);
    let mut handles = Vec::new();

    for stream in read_session.streams {
        let stream_name = stream.name;
        let handle = tokio::task::spawn(bigquery_read_stream::read_stream(
            shared_client.clone(),
            shared_schema.clone(),
            stream_name,
            tx.clone(),
        ));
        handles.push(handle);
    }

    Ok((
        schema_ref,
        BigQueryRecordBatchReceiver {
            rx,
            _handles: handles,
        },
    ))
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
