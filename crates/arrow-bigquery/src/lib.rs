mod bigquery_read_retry;
mod bigquery_read_stream;
pub mod client_builder;
mod error;

use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub use client_builder::*;
pub use error::BigQueryError;
use google_cloud_bigquery::model::{
    arrow_serialization_options, read_session, ArrowSerializationOptions, CreateReadSessionRequest,
    DataFormat, ReadSession,
};
use google_cloud_gax::backoff_policy::BackoffPolicy;
use google_cloud_gax::options::RequestOptionsBuilder;
use google_cloud_gax::retry_policy::RetryPolicy;
use google_cloud_gax::retry_throttler::SharedRetryThrottler;
use google_cloud_wkt::Timestamp;
use polars_arrow::datatypes::ArrowSchemaRef;
use polars_arrow::io::ipc::read::read_stream_metadata;
use polars_arrow::record_batch::RecordBatch;
use winnow::combinator::{eof, opt, separated, terminated};
use winnow::token::take_while;
use winnow::Parser;

const DEFAULT_COMPRESSION: arrow_serialization_options::CompressionCodec =
    arrow_serialization_options::CompressionCodec::Lz4Frame;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BigQueryTableId {
    pub project_id: String,
    pub dataset_id: String,
    pub table_id: String,
}

impl BigQueryTableId {
    pub fn new(
        project_id: impl Into<String>,
        dataset_id: impl Into<String>,
        table_id: impl Into<String>,
    ) -> Self {
        Self {
            project_id: project_id.into(),
            dataset_id: dataset_id.into(),
            table_id: table_id.into(),
        }
    }

    pub fn to_table_path(&self) -> String {
        format!(
            "projects/{}/datasets/{}/tables/{}",
            self.project_id, self.dataset_id, self.table_id
        )
    }
}

impl std::fmt::Display for BigQueryTableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}",
            self.project_id, self.dataset_id, self.table_id
        )
    }
}

fn parse_table_id<'s>(input: &mut &'s str) -> winnow::Result<BigQueryTableId> {
    // In the past, organizations could prefix their project IDs with a domain
    // name. Such projects still exist, especially at Google.
    let domain: Option<&'s str> =
        opt(terminated(take_while(1.., |c| c != ':'), ':')).parse_next(input)?;

    // Project ID cannot contain '.' or ':'
    let project: &'s str = take_while(1.., |c| c != '.' && c != ':').parse_next(input)?;

    // Separator between project and dataset
    let _ = '.'.parse_next(input)?;

    // Match dataset or catalog + namespace.
    // Namespace could be arbitrarily deeply nested in Iceberg/BigLake.
    // Separated into at least 2 non-empty parts: inner parts (dataset) and table.
    let mut parts: Vec<&'s str> =
        separated(2.., take_while(1.., |c| c != '.'), '.').parse_next(input)?;

    let _ = eof.parse_next(input)?;

    let table_id = parts
        .pop()
        .expect("parts is guaranteed to have at least 2 elements")
        .to_string();
    let dataset_id = parts.join(".");
    let project_id = match domain {
        Some(d) => format!("{d}:{project}"),
        None => project.to_string(),
    };

    Ok(BigQueryTableId {
        project_id,
        dataset_id,
        table_id,
    })
}

impl std::str::FromStr for BigQueryTableId {
    type Err = BigQueryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.contains('`') || s.contains('/') {
            return Err(InvalidTableId.into());
        }

        let mut input = s;
        parse_table_id(&mut input).map_err(|_| InvalidTableId.into())
    }
}

impl TryFrom<&str> for BigQueryTableId {
    type Error = BigQueryError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        s.parse()
    }
}

#[derive(Default)]
pub struct ReadOptions {
    pub maintain_order: bool,
    pub snapshot_time: Option<chrono::DateTime<chrono::Utc>>,
    pub selected_fields: Vec<String>,
    pub row_restriction: String,
    pub arrow_buffer_compression: Option<arrow_serialization_options::CompressionCodec>,
    pub sample_percentage: Option<f64>,
    pub max_stream_count: Option<i32>,
}

impl ReadOptions {
    fn build_request(self, table_path: String, quota_project_id: &str) -> CreateReadSessionRequest {
        let arrow_options = ArrowSerializationOptions::new()
            .set_buffer_compression(self.arrow_buffer_compression.unwrap_or(DEFAULT_COMPRESSION));
        let table_modifiers = read_session::TableModifiers::new().set_or_clear_snapshot_time(
            self.snapshot_time.map(|snapshot_time| {
                Timestamp::clamp(
                    snapshot_time.timestamp(),
                    snapshot_time.timestamp_subsec_nanos() as i32,
                )
            }),
        );
        let read_options = read_session::TableReadOptions::new()
            .set_arrow_serialization_options(arrow_options)
            .set_selected_fields(self.selected_fields)
            .set_row_restriction(self.row_restriction)
            .set_or_clear_sample_percentage(self.sample_percentage);
        let read_session = ReadSession::new()
            .set_data_format(DataFormat::Arrow)
            .set_table(table_path)
            .set_table_modifiers(table_modifiers)
            .set_read_options(read_options);
        let max_stream_count = if self.maintain_order {
            // If you are reading from a query results table where order matters,
            // limit this to a single stream.
            1
        } else {
            self.max_stream_count
                .unwrap_or_else(|| match std::thread::available_parallelism() {
                    Ok(value) => value.get() as i32,
                    Err(_) => 1,
                })
        };

        CreateReadSessionRequest::new()
            .set_parent(format!("projects/{}", quota_project_id))
            .set_max_stream_count(max_stream_count)
            .set_read_session(read_session)
    }
}

/// A receiver that yields [`RecordBatch`]es read from BigQuery.
///
/// It manages the background tasks reading from the BigQuery Storage API streams
/// and provides a stream-like interface to receive the data.
pub struct BigQueryRecordBatchReceiver {
    /// The channel receiver for receiving [`RecordBatch`]es produced by the background tasks.
    rx: tokio::sync::mpsc::Receiver<Result<RecordBatch, BigQueryError>>,
    /// Cooperative cancellation sender shared across all sibling stream tasks.
    cancel_tx: Option<Arc<tokio::sync::watch::Sender<bool>>>,
    /// Join handles for the background tasks reading from the BigQuery streams.
    ///
    /// These handles are kept so that the background tasks can be aborted when
    /// the receiver is dropped or encounters a fatal error, preventing resource leaks
    /// from orphan background tasks.
    _handles: Vec<tokio::task::JoinHandle<()>>,
}

impl BigQueryRecordBatchReceiver {
    pub async fn recv(&mut self) -> Option<Result<RecordBatch, BigQueryError>> {
        let item = self.rx.recv().await;
        if matches!(item, None | Some(Err(_))) {
            self.abort();
        }
        item
    }

    /// Immediately cancels and aborts all background stream tasks owned by this receiver.
    pub fn abort(&mut self) {
        if let Some(cancel_tx) = self.cancel_tx.take() {
            let _ = cancel_tx.send_replace(true);
        }
        self.rx.close();
        for handle in self._handles.drain(..) {
            handle.abort();
        }
    }

    /// Creates a placeholder receiver for testing purposes.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn new_for_testing(
        rx: tokio::sync::mpsc::Receiver<Result<RecordBatch, BigQueryError>>,
        handles: Vec<tokio::task::JoinHandle<()>>,
    ) -> Self {
        Self {
            rx,
            cancel_tx: None,
            _handles: handles,
        }
    }
}

impl Drop for BigQueryRecordBatchReceiver {
    fn drop(&mut self) {
        self.abort();
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

pub type BigQueryClient = google_cloud_bigquery::client::Read;

/// A BigQuery client for reading tables using the Storage Read API.
///
/// Keeps the gRPC channel pool open across multiple read operations.
#[derive(Clone)]
pub struct Client {
    clients: Arc<[BigQueryClient]>,
    next_client: Arc<AtomicUsize>,
    quota_project_id: String,
    user_agent: Option<String>,
    retry_policy: Option<Arc<dyn RetryPolicy>>,
    backoff_policy: Option<Arc<dyn BackoffPolicy>>,
    retry_throttler: SharedRetryThrottler,
    decode_pool: Arc<rayon::ThreadPool>,
}

impl Client {
    pub fn builder() -> ServiceConfigBuilder {
        ServiceConfigBuilder::new()
    }

    pub fn new(client: BigQueryClient, quota_project_id: String) -> Self {
        Self::from_parts(
            vec![client],
            quota_project_id,
            None,
            None,
            None,
            client_builder::default_retry_throttler(),
        )
    }

    pub fn from_arc(client: Arc<BigQueryClient>, quota_project_id: String) -> Self {
        Self::new((*client).clone(), quota_project_id)
    }

    pub(crate) fn from_parts(
        clients: Vec<BigQueryClient>,
        quota_project_id: String,
        user_agent: Option<String>,
        retry_policy: Option<Arc<dyn RetryPolicy>>,
        backoff_policy: Option<Arc<dyn BackoffPolicy>>,
        retry_throttler: SharedRetryThrottler,
    ) -> Self {
        debug_assert!(!clients.is_empty(), "Client requires at least one BigQueryClient");
        Self {
            clients: Arc::from(clients.into_boxed_slice()),
            next_client: Arc::new(AtomicUsize::new(0)),
            quota_project_id,
            user_agent,
            retry_policy,
            backoff_policy,
            retry_throttler,
            decode_pool: bigquery_read_stream::default_decode_pool(),
        }
    }

    pub fn quota_project_id(&self) -> &str {
        &self.quota_project_id
    }

    pub fn grpc_subchannel_count(&self) -> usize {
        self.clients.len()
    }

    pub async fn from_builder(builder: ServiceConfigBuilder) -> Result<Self, BigQueryError> {
        builder.build().await
    }

    fn select_client(&self) -> &BigQueryClient {
        let idx = self.next_client.fetch_add(1, Ordering::Relaxed);
        &self.clients[idx % self.clients.len()]
    }

    fn apply_request_options<B: RequestOptionsBuilder>(&self, builder: B) -> B {
        let builder = builder.with_quota_project(&self.quota_project_id);
        match &self.user_agent {
            Some(user_agent) => builder.with_user_agent(user_agent),
            None => builder,
        }
    }

    fn stream_retry_config(&self) -> bigquery_read_retry::StreamRetryConfig {
        // Always attach the Client-wide shared retry throttler so all stream readers
        // share a unified adaptive retry budget instead of isolated per-stream defaults.
        bigquery_read_retry::StreamRetryConfig::new(
            self.retry_policy.clone(),
            self.backoff_policy.clone(),
            self.retry_throttler.clone(),
        )
    }

    pub async fn read_table(
        &self,
        table: &BigQueryTableId,
        options: ReadOptions,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let request = options.build_request(table.to_table_path(), &self.quota_project_id);
        let create_session = self.apply_request_options(
            self.select_client()
                .create_read_session()
                .with_request(request)
                .with_idempotency(true)
                .with_retry_policy(
                    self.retry_policy
                        .clone()
                        .unwrap_or_else(bigquery_read_retry::create_read_session_retry_policy),
                )
                .with_backoff_policy(
                    self.backoff_policy
                        .clone()
                        .unwrap_or_else(bigquery_read_retry::default_backoff_policy),
                )
                .with_retry_throttler(self.retry_throttler.clone()),
        );
        let read_session = create_session.send().await?;
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
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        let retry_config = self.stream_retry_config();
        let mut handles = Vec::with_capacity(read_session.streams.len());

        for (idx, stream) in read_session.streams.into_iter().enumerate() {
            let client = &self.clients[idx % self.clients.len()];
            let read_rows =
                self.apply_request_options(client.read_rows().set_read_stream(stream.name));
            let handle = tokio::task::spawn(bigquery_read_stream::read_stream(
                read_rows,
                schema.clone(),
                tx.clone(),
                retry_config.clone(),
                Arc::clone(&self.decode_pool),
                Arc::clone(&cancel_tx),
                cancel_rx.clone(),
            ));
            handles.push(handle);
        }

        Ok((
            schema_ref,
            BigQueryRecordBatchReceiver {
                rx,
                cancel_tx: Some(cancel_tx),
                _handles: handles,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use super::*;

    #[test]
    fn table_id_to_table_path_success() -> Result<(), Box<dyn std::error::Error>> {
        let id: BigQueryTableId = "my-project.my_dataset.my_table"
            .parse()
            .expect("valid table id");
        let result = id.to_table_path();
        assert_eq!(
            result,
            "projects/my-project/datasets/my_dataset/tables/my_table"
        );
        Ok(())
    }

    #[test]
    fn table_id_to_table_path_success_legacy_project() -> Result<(), Box<dyn std::error::Error>> {
        let id: BigQueryTableId = "google.com:my-project.my_dataset.my_table"
            .parse()
            .expect("valid table id");
        let result = id.to_table_path();
        assert_eq!(
            result,
            "projects/google.com:my-project/datasets/my_dataset/tables/my_table"
        );
        Ok(())
    }

    #[test]
    fn test_from_str_valid_string() {
        let id: BigQueryTableId = "proj.ds.tab".parse().expect("valid table id");
        assert_eq!(
            id,
            BigQueryTableId {
                project_id: "proj".to_string(),
                dataset_id: "ds".to_string(),
                table_id: "tab".to_string(),
            }
        );
    }

    #[test]
    fn test_from_str_with_colon() {
        let id: BigQueryTableId = "google.com:project.ds.tab"
            .parse()
            .expect("valid table id with legacy domain");
        assert_eq!(
            id,
            BigQueryTableId {
                project_id: "google.com:project".to_string(),
                dataset_id: "ds".to_string(),
                table_id: "tab".to_string(),
            }
        );
    }

    #[test]
    fn test_from_str_multipart() {
        let id: BigQueryTableId = "too.many.parts.here"
            .parse()
            .expect("valid multipart table id");
        assert_eq!(
            id,
            BigQueryTableId {
                project_id: "too".to_string(),
                dataset_id: "many.parts".to_string(),
                table_id: "here".to_string(),
            }
        );
    }

    #[test]
    fn test_from_str_invalid_format() {
        assert!("just_a_string".parse::<BigQueryTableId>().is_err());
        assert!("proj.tab".parse::<BigQueryTableId>().is_err());
    }

    #[test]
    fn test_from_str_backtick() {
        assert!("`proj.ds.tab`".parse::<BigQueryTableId>().is_err());
        assert!("proj.`ds`.tab".parse::<BigQueryTableId>().is_err());
    }

    #[test]
    fn test_from_str_consecutive_dots() {
        assert!("proj..tab".parse::<BigQueryTableId>().is_err());
        assert!("proj.a..b.tab".parse::<BigQueryTableId>().is_err());
    }

    #[test]
    fn test_from_str_empty_string() {
        assert!("".parse::<BigQueryTableId>().is_err());
    }

    #[test]
    fn test_from_str_trailing_and_leading_dots() {
        assert!(".proj.ds.tab".parse::<BigQueryTableId>().is_err());
        assert!("proj.ds.tab.".parse::<BigQueryTableId>().is_err());
        assert!("proj.ds.".parse::<BigQueryTableId>().is_err());
        assert!(".ds.tab".parse::<BigQueryTableId>().is_err());
        assert!(".".parse::<BigQueryTableId>().is_err());
        assert!("..".parse::<BigQueryTableId>().is_err());
        assert!("...".parse::<BigQueryTableId>().is_err());
    }

    #[test]
    fn test_from_str_invalid_legacy_domain() {
        assert!(":proj.ds.tab".parse::<BigQueryTableId>().is_err());
        assert!("google.com:".parse::<BigQueryTableId>().is_err());
        assert!("google.com:.ds.tab".parse::<BigQueryTableId>().is_err());
        assert!("google.com:proj.tab".parse::<BigQueryTableId>().is_err());
        assert!("google.com:proj.ds.tab."
            .parse::<BigQueryTableId>()
            .is_err());
        assert!("google.com:proj:extra.ds.tab"
            .parse::<BigQueryTableId>()
            .is_err());
    }

    #[test]
    fn test_read_options_defaults() {
        let options = ReadOptions::default();
        let request = options.build_request(
            "projects/test-project/datasets/test_dataset/tables/test_table".to_string(),
            "quota-project-123",
        );
        let available_parallelism = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(1).expect("hardcoded"))
            .get() as i32;

        assert_eq!(request.parent, "projects/quota-project-123");
        assert_eq!(request.max_stream_count, available_parallelism);

        let session = request
            .read_session
            .expect("read_session should be present");
        assert_eq!(session.data_format, DataFormat::Arrow);
        assert_eq!(
            session.table,
            "projects/test-project/datasets/test_dataset/tables/test_table"
        );

        let table_modifiers = session
            .table_modifiers
            .expect("table_modifiers should be present");
        assert_eq!(table_modifiers.snapshot_time, None);

        let read_options = session
            .read_options
            .expect("read_options should be present");
        assert!(read_options.selected_fields.is_empty());
        assert_eq!(read_options.row_restriction, "");
        assert_eq!(read_options.sample_percentage, None);

        match read_options.output_format_serialization_options {
            Some(
                read_session::table_read_options::OutputFormatSerializationOptions::ArrowSerializationOptions(
                    arrow_opts,
                ),
            ) => {
                assert_eq!(
                    arrow_opts.buffer_compression,
                    arrow_serialization_options::CompressionCodec::Lz4Frame
                );
            },
            other => panic!("expected ArrowSerializationOptions, got {:?}", other),
        }
    }

    #[test]
    fn test_read_options_all_fields_plumbed() {
        let snapshot_dt =
            chrono::DateTime::from_timestamp(1_700_000_000, 500_000_000).expect("valid timestamp");
        let expected_timestamp = Timestamp::clamp(
            snapshot_dt.timestamp(),
            snapshot_dt.timestamp_subsec_nanos() as i32,
        );

        let options = ReadOptions {
            maintain_order: false,
            snapshot_time: Some(snapshot_dt),
            selected_fields: vec!["col1".to_string(), "col2".to_string()],
            row_restriction: "col1 > 100".to_string(),
            arrow_buffer_compression: Some(arrow_serialization_options::CompressionCodec::Zstd),
            sample_percentage: Some(42.5),
            max_stream_count: Some(12),
        };

        let request =
            options.build_request("projects/p/datasets/d/tables/t".to_string(), "custom-quota");

        assert_eq!(request.parent, "projects/custom-quota");
        assert_eq!(request.max_stream_count, 12);

        let session = request
            .read_session
            .expect("read_session should be present");
        assert_eq!(session.data_format, DataFormat::Arrow);
        assert_eq!(session.table, "projects/p/datasets/d/tables/t");

        let table_modifiers = session
            .table_modifiers
            .expect("table_modifiers should be present");
        assert_eq!(table_modifiers.snapshot_time, Some(expected_timestamp));

        let read_options = session
            .read_options
            .expect("read_options should be present");
        assert_eq!(read_options.selected_fields, vec!["col1", "col2"]);
        assert_eq!(read_options.row_restriction, "col1 > 100");
        assert_eq!(read_options.sample_percentage, Some(42.5));

        match read_options.output_format_serialization_options {
            Some(
                read_session::table_read_options::OutputFormatSerializationOptions::ArrowSerializationOptions(
                    arrow_opts,
                ),
            ) => {
                assert_eq!(
                    arrow_opts.buffer_compression,
                    arrow_serialization_options::CompressionCodec::Zstd
                );
            },
            other => panic!("expected ArrowSerializationOptions, got {:?}", other),
        }
    }

    #[test]
    fn test_read_options_maintain_order_controls_max_stream_count() {
        let options_ordered = ReadOptions {
            maintain_order: true,
            ..Default::default()
        };
        let request_ordered = options_ordered.build_request("table".to_string(), "quota");
        assert_eq!(request_ordered.max_stream_count, 1);

        let options_unordered = ReadOptions {
            maintain_order: false,
            max_stream_count: Some(16),
            ..Default::default()
        };
        let request_unordered = options_unordered.build_request("table".to_string(), "quota");
        assert_eq!(request_unordered.max_stream_count, 16);
    }

    #[test]
    fn test_read_options_compression_codecs() {
        let codecs = [
            (
                None,
                arrow_serialization_options::CompressionCodec::Lz4Frame,
            ),
            (
                Some(arrow_serialization_options::CompressionCodec::Lz4Frame),
                arrow_serialization_options::CompressionCodec::Lz4Frame,
            ),
            (
                Some(arrow_serialization_options::CompressionCodec::Zstd),
                arrow_serialization_options::CompressionCodec::Zstd,
            ),
            (
                Some(arrow_serialization_options::CompressionCodec::CompressionUnspecified),
                arrow_serialization_options::CompressionCodec::CompressionUnspecified,
            ),
        ];

        for (codec, expected_val) in codecs {
            let options = ReadOptions {
                arrow_buffer_compression: codec,
                ..Default::default()
            };
            let request = options.build_request("table".to_string(), "quota");
            let session = request
                .read_session
                .expect("read_session should be present");
            let read_options = session
                .read_options
                .expect("read_options should be present");
            match read_options.output_format_serialization_options {
                Some(
                    read_session::table_read_options::OutputFormatSerializationOptions::ArrowSerializationOptions(
                        arrow_opts,
                    ),
                ) => {
                    assert_eq!(arrow_opts.buffer_compression, expected_val);
                },
                other => panic!("expected ArrowSerializationOptions, got {:?}", other),
            }
        }
    }

    fn empty_arrow_schema_bytes() -> Vec<u8> {
        use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
        use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};

        let schema =
            ArrowSchema::from_iter(vec![Field::new("col1".into(), ArrowDataType::Int32, false)]);
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::new(&mut bytes, WriteOptions { compression: None });
        writer.start(&schema, None).unwrap();
        bytes
    }

    #[tokio::test]
    async fn test_client_read_table_applies_options_and_forwards_retry_config() {
        use std::sync::Mutex;
        use std::time::Duration;

        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowSchema as BqArrowSchema, ReadRowsRequest, ReadRowsResponse, ReadStream,
        };
        use google_cloud_gax::error::rpc::{Code, Status};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::retry_policy::RetryPolicyExt;
        use google_cloud_gax::retry_state::RetryState;
        use google_cloud_gax::streaming::ResponseStream;

        use crate::bigquery_read_retry::RetryableErrors;

        #[derive(Debug, Default)]
        struct NoBackoff;
        impl BackoffPolicy for NoBackoff {
            fn on_failure(&self, _state: &RetryState) -> Duration {
                Duration::ZERO
            }
        }

        #[derive(Debug, Default)]
        struct RecordedOptions {
            create_quota_project: Option<String>,
            create_user_agent: Option<String>,
            read_rows_calls: usize,
            read_rows_quota_project: Option<String>,
            read_rows_user_agent: Option<String>,
        }

        #[derive(Debug)]
        struct MockReadStub {
            recorded: Arc<Mutex<RecordedOptions>>,
        }

        impl google_cloud_bigquery::stub::Read for MockReadStub {
            async fn create_read_session(
                &self,
                _req: CreateReadSessionRequest,
                options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                {
                    let mut rec = self.recorded.lock().unwrap();
                    rec.create_quota_project = options.quota_project().clone();
                    rec.create_user_agent = options.user_agent().clone();
                }
                let session = ReadSession::new()
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(empty_arrow_schema_bytes()),
                    )
                    .set_streams(vec![ReadStream::new().set_name("streams/s1")]);
                Ok(Response::from(session))
            }

            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let mut rec = self.recorded.lock().unwrap();
                rec.read_rows_calls += 1;
                rec.read_rows_quota_project = options.quota_project().clone();
                rec.read_rows_user_agent = options.user_agent().clone();
                Err(GaxError::service(
                    Status::default()
                        .set_code(Code::Unavailable)
                        .set_message("transient failure"),
                ))
            }
        }

        let recorded = Arc::new(Mutex::new(RecordedOptions::default()));
        let stub = MockReadStub {
            recorded: recorded.clone(),
        };
        // Configure attempt_limit(1) so stream reader performs 0 retries (proving Client's retry_policy
        // is forwarded to the stream retry config rather than discarded in favor of defaults).
        let client = Client::from_parts(
            vec![Read::from_stub(stub)],
            "test-quota-proj".to_string(),
            Some("custom-ua/2.0".to_string()),
            Some(Arc::new(RetryableErrors.with_attempt_limit(1))),
            Some(Arc::new(NoBackoff)),
            client_builder::default_retry_throttler(),
        );

        let table: BigQueryTableId = "proj.ds.tbl".parse().unwrap();
        let (_, mut rx) = client
            .read_table(&table, ReadOptions::default())
            .await
            .unwrap();

        let err = rx.recv().await.unwrap().unwrap_err();
        assert!(matches!(err, BigQueryError::Grpc(_)));

        let rec = recorded.lock().unwrap();
        assert_eq!(rec.create_quota_project.as_deref(), Some("test-quota-proj"));
        assert_eq!(rec.create_user_agent.as_deref(), Some("custom-ua/2.0"));
        assert_eq!(
            rec.read_rows_quota_project.as_deref(),
            Some("test-quota-proj")
        );
        assert_eq!(rec.read_rows_user_agent.as_deref(), Some("custom-ua/2.0"));
        assert_eq!(
            rec.read_rows_calls, 1,
            "Stream reader should respect Client's attempt_limit(1) instead of discarding it"
        );
    }

    #[tokio::test]
    async fn test_client_read_table_preserves_causal_error_chain() {
        use std::error::Error as _;

        use google_cloud_auth::errors::CredentialsError;
        use google_cloud_bigquery::client::Read;
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct FailingAuthStub;

        impl google_cloud_bigquery::stub::Read for FailingAuthStub {
            async fn create_read_session(
                &self,
                _req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let io_err = std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "socket permission denied",
                );
                let cred_err = CredentialsError::new(false, "custom auth provider failed", io_err);
                Err(GaxError::authentication(cred_err))
            }
        }

        let client = Client::new(Read::from_stub(FailingAuthStub), "quota-proj".to_string());
        let table: BigQueryTableId = "proj.ds.tbl".parse().unwrap();
        let err = match client.read_table(&table, ReadOptions::default()).await {
            Err(e) => e,
            Ok(_) => panic!("expected read_table to fail"),
        };

        let s1 = err.source().expect("GaxError");
        let s2 = s1.source().expect("CredentialsError");
        let s3 = s2.source().expect("io::Error");
        assert!(s3.to_string().contains("socket permission denied"));

        let chain = err.format_causal_chain();
        assert!(chain.contains("custom auth provider failed"));
        assert!(chain.contains("caused by: socket permission denied"));
    }

    #[test]
    fn test_from_str_slash_rejection() {
        assert!("proj/inject.ds.tbl".parse::<BigQueryTableId>().is_err());
    }

    #[tokio::test]
    async fn test_client_shares_retry_throttler_and_distributes_subchannels() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Mutex;
        use std::time::Duration;

        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowSchema as BqArrowSchema, ReadRowsRequest, ReadRowsResponse, ReadStream,
        };
        use google_cloud_gax::error::rpc::{Code, Status};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::retry_policy::RetryPolicyExt;
        use google_cloud_gax::retry_state::RetryState;
        use google_cloud_gax::retry_throttler::RetryThrottler;
        use google_cloud_gax::streaming::ResponseStream;

        use crate::bigquery_read_retry::RetryableErrors;

        #[derive(Debug, Default)]
        struct NoBackoff;
        impl BackoffPolicy for NoBackoff {
            fn on_failure(&self, _state: &RetryState) -> Duration {
                Duration::ZERO
            }
        }

        #[derive(Debug)]
        struct CountingThrottler {
            throttle_checks: Arc<AtomicUsize>,
        }
        impl RetryThrottler for CountingThrottler {
            fn throttle_retry_attempt(&self) -> bool {
                self.throttle_checks.fetch_add(1, Ordering::SeqCst);
                true // throttle retries immediately
            }
            fn on_retry_failure(&mut self, _flow: &google_cloud_gax::retry_result::RetryResult) {}
            fn on_success(&mut self) {}
        }

        #[derive(Debug)]
        struct SubchannelStub {
            subchannel_idx: usize,
            streams_seen: Arc<Mutex<Vec<std::collections::HashSet<String>>>>,
            started_tx: tokio::sync::watch::Sender<usize>,
            started_rx: tokio::sync::watch::Receiver<usize>,
        }

        impl google_cloud_bigquery::stub::Read for SubchannelStub {
            async fn create_read_session(
                &self,
                _req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let streams: Vec<ReadStream> = (0..5)
                    .map(|i| ReadStream::new().set_name(format!("streams/s{i}")))
                    .collect();
                let session = ReadSession::new()
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(empty_arrow_schema_bytes()),
                    )
                    .set_streams(streams);
                Ok(Response::from(session))
            }

            async fn read_rows(
                &self,
                req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let total_distinct = {
                    let mut seen = self.streams_seen.lock().unwrap();
                    seen[self.subchannel_idx].insert(req.read_stream.clone());
                    seen.iter().map(|s| s.len()).sum::<usize>()
                };
                let _ = self.started_tx.send_replace(total_distinct);

                if req.read_stream == "streams/s4" {
                    let mut rx_wait = self.started_rx.clone();
                    while *rx_wait.borrow_and_update() < 5 {
                        rx_wait.changed().await.unwrap();
                    }
                    let (tx, rx) = tokio::sync::mpsc::channel(1);
                    let _ = tx
                        .send(Err(GaxError::service(
                            Status::default()
                                .set_code(Code::Unavailable)
                                .set_message("unavailable"),
                        )))
                        .await;
                    Ok(ResponseStream::from(rx))
                } else {
                    let (_tx, rx) = tokio::sync::mpsc::channel(1);
                    Ok(ResponseStream::from(rx))
                }
            }
        }

        let streams_seen = Arc::new(Mutex::new(vec![
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
        ]));
        let (started_tx, started_rx) = tokio::sync::watch::channel(0usize);
        let stubs: Vec<Read> = (0..3)
            .map(|i| {
                Read::from_stub(SubchannelStub {
                    subchannel_idx: i,
                    streams_seen: Arc::clone(&streams_seen),
                    started_tx: started_tx.clone(),
                    started_rx: started_rx.clone(),
                })
            })
            .collect();

        let throttle_checks = Arc::new(AtomicUsize::new(0));
        let shared_throttler: SharedRetryThrottler = Arc::new(Mutex::new(CountingThrottler {
            throttle_checks: Arc::clone(&throttle_checks),
        }));

        let client = Client::from_parts(
            stubs,
            "quota-proj".to_string(),
            None,
            Some(Arc::new(RetryableErrors.with_attempt_limit(5))),
            Some(Arc::new(NoBackoff)),
            shared_throttler,
        );

        let table = BigQueryTableId::new("proj", "ds", "tbl");
        let (_, mut rx) = client
            .read_table(&table, ReadOptions::default())
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, Err(BigQueryError::Grpc(_))));

        let counts: Vec<usize> = streams_seen
            .lock()
            .unwrap()
            .iter()
            .map(|s| s.len())
            .collect();
        // 5 streams distributed round-robin across 3 subchannels: [2, 2, 1].
        assert_eq!(counts, vec![2, 2, 1]);
        assert!(
            throttle_checks.load(Ordering::SeqCst) >= 1,
            "shared retry throttler should be consulted by stream readers"
        );
    }
}
