use std::io::Cursor;
use std::iter::Iterator;
use std::sync::Arc;

use gcloud_sdk::google::cloud::bigquery::storage::v1::big_query_read_client::BigQueryReadClient;
use gcloud_sdk::google::cloud::bigquery::storage::v1::{
    read_rows_response, read_session, CreateReadSessionRequest, DataFormat, ReadRowsRequest,
    ReadRowsResponse, ReadSession,
};
use gcloud_sdk::tonic::async_trait;
use gcloud_sdk::*;
use hyper::header::{HeaderValue, USER_AGENT};
use hyper::HeaderMap;
use polars_arrow::datatypes::ArrowSchemaRef;
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

#[derive(Clone)]
pub struct BigQueryReadClientBuilder {
    max_decoding_message_size: usize,
}

#[async_trait]
impl GoogleApiClientBuilder<BigQueryReadClient<GoogleAuthMiddleware>>
    for BigQueryReadClientBuilder
{
    fn create_client(
        &self,
        channel: GoogleAuthMiddleware,
    ) -> BigQueryReadClient<GoogleAuthMiddleware> {
        BigQueryReadClient::new(channel).max_decoding_message_size(self.max_decoding_message_size)
    }
}

pub struct PolarsBigQueryClientBuilder {
    max_decoding_message_size: usize,
    token_source_type: TokenSourceType,
    scopes: Vec<String>,
    user_agent: Option<String>,
}

impl PolarsBigQueryClientBuilder {
    pub fn new() -> Self {
        Self {
            max_decoding_message_size: 128 * 1024 * 1024, // 128MB default
            token_source_type: TokenSourceType::Default,
            scopes: vec!["https://www.googleapis.com/auth/cloud-platform".to_string()],
            user_agent: None,
        }
    }

    pub fn with_max_decoding_message_size(mut self, size: usize) -> Self {
        self.max_decoding_message_size = size;
        self
    }

    pub fn with_token_source(mut self, token_source: TokenSourceType) -> Self {
        self.token_source_type = token_source;
        self
    }

    pub fn with_scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = scopes;
        self
    }

    pub fn with_user_agent(mut self, extension: String) -> Self {
        self.user_agent = Some(extension);
        self
    }

    pub async fn build(
        self,
    ) -> Result<
        GoogleApiClient<BigQueryReadClientBuilder, BigQueryReadClient<GoogleAuthMiddleware>>,
        Box<dyn std::error::Error>,
    > {
        init_crypto();
        let builder = BigQueryReadClientBuilder {
            max_decoding_message_size: self.max_decoding_message_size,
        };

        // Construct User-Agent header
        let default_user_agent = format!("polars-bigquery/{}", env!("CARGO_PKG_VERSION"));
        let user_agent = match self.user_agent {
            Some(ext) => format!("{} {}", default_user_agent, ext),
            None => default_user_agent,
        };

        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);

        let client = GoogleApiClient::with_token_source_and_headers(
            builder,
            "https://bigquerystorage.googleapis.com",
            None,
            self.token_source_type,
            self.scopes,
            headers,
        )
        .await?;

        Ok(client)
    }
}

/// A receiver that yields [`RecordBatch`]es read from BigQuery.
///
/// It manages the background tasks reading from the BigQuery Storage API streams
/// and provides a stream-like interface to receive the data.
pub struct BigQueryRecordBatchReceiver {
    /// The channel receiver for receiving [`RecordBatch`]es produced by the background tasks.
    rx: tokio::sync::mpsc::Receiver<RecordBatch>,
    /// Join handles for the background tasks reading from the BigQuery streams.
    ///
    /// These handles are kept so that the background tasks can be aborted when
    /// the receiver is dropped, preventing resource leaks from orphan background tasks.
    _handles: Vec<tokio::task::JoinHandle<()>>,
}

impl BigQueryRecordBatchReceiver {
    pub async fn recv(&mut self) -> Option<RecordBatch> {
        self.rx.recv().await
    }

    /// Creates a placeholder receiver for testing purposes.
    pub fn new_for_testing(
        rx: tokio::sync::mpsc::Receiver<RecordBatch>,
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

async fn read_stream<B>(
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

pub fn init_crypto() {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });
}

pub async fn read_bigquery_with_client<B>(
    read_client: GoogleApiClient<B, BigQueryReadClient<GoogleAuthMiddleware>>,
    table_id: &str,
    quota_project_id: &str,
    maintain_order: bool,
) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), Box<dyn std::error::Error>>
where
    B: GoogleApiClientBuilder<BigQueryReadClient<GoogleAuthMiddleware>> + Send + Sync + 'static,
{
    init_crypto();

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

    let read_session = read_client
        .get()
        .create_read_session(request)
        .await?
        .into_inner();
    let schema = match read_session.schema.unwrap() {
        read_session::Schema::ArrowSchema(value) => value.serialized_schema,
        _ => panic!("unexpectedly got schema type other than arrow"),
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
        let handle = tokio::task::spawn(read_stream(
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
