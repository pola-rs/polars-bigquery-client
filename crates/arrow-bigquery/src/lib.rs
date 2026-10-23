mod bigquery_read_retry;
mod bigquery_read_stream;
pub mod client_builder;
mod decode;
mod error;
pub mod query;

use std::sync::Arc;

pub use client_builder::*;
pub use error::{BigQueryError, JobError, JobFailedError};
pub use google_cloud_bigquery::model::arrow_serialization_options::CompressionCodec;
use google_cloud_bigquery::model::{
    arrow_serialization_options, read_session, ArrowSerializationOptions, CreateReadSessionRequest,
    DataFormat, ReadSession,
};
use google_cloud_gax::options::RequestOptionsBuilder;
use google_cloud_gax::retry_throttler::SharedRetryThrottler;
use google_cloud_wkt::Timestamp;
use polars_arrow::datatypes::ArrowSchemaRef;
use polars_arrow::record_batch::RecordBatch;
use winnow::combinator::{eof, opt, separated, terminated};
use winnow::token::take_while;
use winnow::Parser;

pub(crate) const DEFAULT_COMPRESSION: arrow_serialization_options::CompressionCodec =
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

#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    pub offset: i64,
    /// Optional IPC-serialized Arrow schema bytes.
    ///
    /// If `None`, the schema is extracted from the first `ReadRowsResponse`
    /// received from the stream.
    pub schema: Option<bytes::Bytes>,
    pub arrow_buffer_compression: Option<arrow_serialization_options::CompressionCodec>,
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
    handles: Vec<tokio::task::JoinHandle<()>>,
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
        for handle in self.handles.drain(..) {
            handle.abort();
        }
    }

    pub(crate) fn new(
        rx: tokio::sync::mpsc::Receiver<Result<RecordBatch, BigQueryError>>,
        cancel_tx: Option<Arc<tokio::sync::watch::Sender<bool>>>,
        handles: Vec<tokio::task::JoinHandle<()>>,
    ) -> Self {
        Self {
            rx,
            cancel_tx,
            handles,
        }
    }

    /// Creates a placeholder receiver for testing purposes.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn new_for_testing(
        rx: tokio::sync::mpsc::Receiver<Result<RecordBatch, BigQueryError>>,
        handles: Vec<tokio::task::JoinHandle<()>>,
    ) -> Self {
        Self::new(rx, None, handles)
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

pub type BigQueryReadClient = google_cloud_bigquery::client::Read;
pub type BigQueryJobService = google_cloud_bigquery_v2::client::JobService;

/// A BigQuery client for reading tables using the Storage Read API.
///
/// Keeps the gRPC channel open across multiple read operations.
#[derive(Clone, Debug)]
pub struct Client {
    pub(crate) bq_client: BigQueryJobService,
    read_client: BigQueryReadClient,
    pub(crate) quota_project_id: Arc<str>,
    user_agent: Option<Arc<str>>,
    retry_throttler: SharedRetryThrottler,
    retry_config: bigquery_read_retry::StreamRetryConfig,
    pub(crate) decode_pool: Arc<rayon::ThreadPool>,
}

pub(crate) struct FallbackReadSession {
    pub(crate) project_id: String,
    pub(crate) job_id: String,
    pub(crate) location: String,
    pub(crate) read_options: ReadOptions,
}

impl Client {
    pub fn builder() -> ServiceConfigBuilder {
        ServiceConfigBuilder::new()
    }

    pub(crate) fn from_parts(
        bq_client: BigQueryJobService,
        read_client: BigQueryReadClient,
        quota_project_id: String,
        user_agent: Option<String>,
        retry_throttler: SharedRetryThrottler,
    ) -> Self {
        // Always attach the Client-wide shared retry throttler so all stream readers
        // share a unified adaptive retry budget instead of isolated per-stream defaults.
        let retry_config =
            bigquery_read_retry::StreamRetryConfig::new(None, None, retry_throttler.clone());
        Self {
            bq_client,
            read_client,
            quota_project_id: Arc::from(quota_project_id),
            user_agent: user_agent.map(Arc::from),
            retry_throttler,
            retry_config,
            decode_pool: decode::default_decode_pool(),
        }
    }

    /// Creates a client for testing purposes.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn new_for_testing(
        bq_client: BigQueryJobService,
        read_client: BigQueryReadClient,
        quota_project_id: String,
    ) -> Self {
        Self::from_parts(
            bq_client,
            read_client,
            quota_project_id,
            None,
            client_builder::default_retry_throttler(),
        )
    }

    pub fn quota_project_id(&self) -> &str {
        &self.quota_project_id
    }

    pub fn query(&self, sql: impl Into<String>) -> query::Query<'_> {
        query::Query::new(self, sql)
    }

    pub async fn from_builder(builder: ServiceConfigBuilder) -> Result<Self, BigQueryError> {
        builder.build().await
    }

    pub(crate) fn apply_request_options<B: RequestOptionsBuilder>(&self, builder: B) -> B {
        let builder = builder.with_quota_project(&*self.quota_project_id);
        match &self.user_agent {
            Some(user_agent) => builder.with_user_agent(&**user_agent),
            None => builder,
        }
    }

    pub(crate) fn stream_retry_config(&self) -> bigquery_read_retry::StreamRetryConfig {
        self.retry_config.clone()
    }

    pub(crate) fn channel_size(&self) -> usize {
        self.decode_pool.current_num_threads().max(1) * 2
    }

    pub(crate) fn prepare_read_rows(
        &self,
        stream_name: String,
        offset: i64,
        arrow_buffer_compression: Option<arrow_serialization_options::CompressionCodec>,
    ) -> google_cloud_bigquery::builder::read::ReadRows {
        let mut builder = self
            .read_client
            .read_rows()
            .set_read_stream(stream_name)
            .set_offset(offset);
        if let Some(codec) = arrow_buffer_compression {
            builder = builder.set_arrow_serialization_options(
                ArrowSerializationOptions::new().set_buffer_compression(codec),
            );
        }
        self.apply_request_options(builder)
    }

    pub(crate) fn spawn_query_stream(
        &self,
        first_page_bytes: bytes::Bytes,
        page_row_count: i64,
        read_rows: Option<google_cloud_bigquery::builder::read::ReadRows>,
        metadata: Arc<polars_arrow::io::ipc::read::StreamMetadata>,
        maintain_order: bool,
    ) -> BigQueryRecordBatchReceiver {
        let (tx, rx) = tokio::sync::mpsc::channel(self.channel_size());
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        let handle = tokio::task::spawn(bigquery_read_stream::read_query_stream(
            bigquery_read_stream::ReadQueryStreamOptions {
                first_page_bytes,
                page_row_count,
                read_rows,
                metadata,
                tx,
                retry_config: self.stream_retry_config(),
                decode_pool: Arc::clone(&self.decode_pool),
                maintain_order,
                cancel_tx: Arc::clone(&cancel_tx),
                cancel_rx,
            },
        ));
        BigQueryRecordBatchReceiver::new(rx, Some(cancel_tx), vec![handle])
    }

    pub(crate) async fn create_table_read_session(
        &self,
        table: &BigQueryTableId,
        options: ReadOptions,
    ) -> Result<(bytes::Bytes, Vec<String>), BigQueryError> {
        let request = options.build_request(table.to_table_path(), &self.quota_project_id);
        let create_session = self.apply_request_options(
            self.read_client
                .create_read_session()
                .with_request(request)
                .with_idempotency(true)
                .with_retry_policy(bigquery_read_retry::create_read_session_retry_policy())
                .with_backoff_policy(bigquery_read_retry::default_backoff_policy())
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

        let streams = read_session
            .streams
            .into_iter()
            .map(|stream| stream.name)
            .collect();
        Ok((schema, streams))
    }

    pub async fn read_table(
        &self,
        table: &BigQueryTableId,
        options: ReadOptions,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let maintain_order = options.maintain_order;
        let (schema, streams) = self.create_table_read_session(table, options).await?;
        self.read_streams(
            streams,
            StreamOptions {
                offset: 0,
                schema: Some(schema),
                arrow_buffer_compression: None,
            },
            maintain_order,
        )
        .await
    }

    pub async fn read_stream(
        &self,
        stream: impl Into<String>,
        options: StreamOptions,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        self.read_streams(vec![stream.into()], options, true).await
    }

    pub(crate) async fn read_stream_with_first_page_check(
        &self,
        stream: String,
        options: StreamOptions,
        maintain_order: bool,
        first_page_check: Option<(bigquery_read_stream::FirstPageCheck, FallbackReadSession)>,
    ) -> Result<(Option<ArrowSchemaRef>, BigQueryRecordBatchReceiver), BigQueryError> {
        self.read_streams_inner(vec![stream], options, maintain_order, first_page_check)
            .await
    }

    async fn read_streams(
        &self,
        streams: Vec<String>,
        options: StreamOptions,
        maintain_order: bool,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let (schema_ref, receiver) = self
            .read_streams_inner(streams, options, maintain_order, None)
            .await?;
        let schema_ref = schema_ref.ok_or_else(|| {
            BigQueryError::Protocol("ReadRows stream closed without sending an Arrow schema".into())
        })?;
        Ok((schema_ref, receiver))
    }

    async fn switch_to_read_session(
        &self,
        fallback: FallbackReadSession,
        tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
        cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
        cancel_rx: tokio::sync::watch::Receiver<bool>,
    ) -> Result<(), BigQueryError> {
        let maintain_order = fallback.read_options.maintain_order;
        let destination_table = query::get_destination_table(
            self,
            &fallback.project_id,
            &fallback.job_id,
            &fallback.location,
        )
        .await?;
        let (schema, streams) = self
            .create_table_read_session(&destination_table, fallback.read_options)
            .await?;
        let metadata = Arc::new(decode::parse_arrow_schema(&schema)?);
        let retry_config = self.stream_retry_config();
        let mut join_set = tokio::task::JoinSet::new();

        for stream_name in streams {
            let read_rows = self.prepare_read_rows(stream_name, 0, None);
            join_set.spawn(bigquery_read_stream::read_stream(
                bigquery_read_stream::ReadStreamOptions {
                    read_rows,
                    current_offset: 0,
                    metadata: Some(Arc::clone(&metadata)),
                    schema_tx: None,
                    first_page_check: None,
                    tx: tx.clone(),
                    retry_config: retry_config.clone(),
                    decode_pool: Arc::clone(&self.decode_pool),
                    maintain_order,
                    cancel_tx: Arc::clone(&cancel_tx),
                    cancel_rx: cancel_rx.clone(),
                },
            ));
        }
        drop(tx);
        while join_set.join_next().await.is_some() {}
        Ok(())
    }

    async fn read_streams_inner(
        &self,
        streams: Vec<String>,
        options: StreamOptions,
        maintain_order: bool,
        mut first_page_check: Option<(bigquery_read_stream::FirstPageCheck, FallbackReadSession)>,
    ) -> Result<(Option<ArrowSchemaRef>, BigQueryRecordBatchReceiver), BigQueryError> {
        let (metadata, schema_ref, mut schema_tx, schema_rx) = match &options.schema {
            Some(schema) => {
                let metadata = Arc::new(decode::parse_arrow_schema(schema)?);
                let schema_ref = Arc::new(metadata.schema.clone());
                (Some(metadata), Some(schema_ref), None, None)
            },
            None => {
                if streams.is_empty() {
                    return Err(BigQueryError::Protocol(
                        "Cannot determine schema when there are no streams and no schema was provided"
                            .into(),
                    ));
                }
                let (tx, rx) = tokio::sync::oneshot::channel();
                (None, None, Some(tx), Some(rx))
            },
        };

        let channel_size = self.channel_size();
        let (tx, rx) = tokio::sync::mpsc::channel(channel_size);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        let retry_config = self.stream_retry_config();
        let mut handles = Vec::with_capacity(streams.len());

        for stream_name in streams {
            let read_rows = self.prepare_read_rows(
                stream_name,
                options.offset,
                options.arrow_buffer_compression.clone(),
            );
            let (check, fallback) = match first_page_check.take() {
                Some((check, fallback)) => (Some(check), Some((self.clone(), fallback))),
                None => (None, None),
            };
            let metadata = metadata.clone();
            let schema_tx = schema_tx.take();
            let tx = tx.clone();
            let retry_config = retry_config.clone();
            let decode_pool = Arc::clone(&self.decode_pool);
            let cancel_tx = Arc::clone(&cancel_tx);
            let cancel_rx = cancel_rx.clone();
            let offset = options.offset;
            let handle = tokio::task::spawn(async move {
                let reason =
                    bigquery_read_stream::read_stream(bigquery_read_stream::ReadStreamOptions {
                        read_rows,
                        current_offset: offset,
                        metadata,
                        schema_tx,
                        first_page_check: check,
                        tx: tx.clone(),
                        retry_config,
                        decode_pool,
                        maintain_order,
                        cancel_tx: Arc::clone(&cancel_tx),
                        cancel_rx: cancel_rx.clone(),
                    })
                    .await;

                if reason == bigquery_read_stream::TerminationReason::SwitchToReadSession {
                    if let Some((client, fallback)) = fallback {
                        let tx_err = tx.clone();
                        if let Err(err) = client
                            .switch_to_read_session(fallback, tx, Arc::clone(&cancel_tx), cancel_rx)
                            .await
                        {
                            let _ = cancel_tx.send_replace(true);
                            let _ = tx_err.send(Err(err)).await;
                        }
                    }
                }
            });
            handles.push(handle);
        }

        let receiver = BigQueryRecordBatchReceiver::new(rx, Some(cancel_tx), handles);

        let schema_ref = match schema_ref {
            Some(schema_ref) => Some(schema_ref),
            None => {
                let schema_rx = schema_rx.expect("schema_rx must be Some when schema_ref is None");
                schema_rx.await.map_err(|_| {
                    BigQueryError::Other(
                        "ReadRows stream task terminated before sending schema".into(),
                    )
                })??
            },
        };

        Ok((schema_ref, receiver))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::num::NonZero;

    use super::*;
    use crate::bigquery_read_stream::tests::{create_test_arrow_payload, CannedReadRowsStub};

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

    pub(crate) async fn test_bq_client() -> BigQueryJobService {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        BigQueryJobService::builder()
            .with_credentials(cred)
            .build()
            .await
            .expect("should build test BigQueryJobService")
    }

    pub(crate) async fn test_read_client() -> BigQueryReadClient {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        BigQueryReadClient::builder()
            .with_credentials(cred)
            .build()
            .await
            .expect("should build test BigQueryReadClient")
    }

    #[tokio::test]
    async fn test_client_read_table_applies_options() {
        use std::sync::Mutex;

        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowSchema as BqArrowSchema, ReadRowsRequest, ReadRowsResponse, ReadStream,
        };
        use google_cloud_gax::error::rpc::{Code, Status};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::streaming::ResponseStream;

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
                        BqArrowSchema::new().set_serialized_schema(create_test_arrow_payload(0).0),
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
                        .set_code(Code::PermissionDenied)
                        .set_message("permission denied"),
                ))
            }
        }

        let recorded = Arc::new(Mutex::new(RecordedOptions::default()));
        let stub = MockReadStub {
            recorded: recorded.clone(),
        };
        let client = Client::from_parts(
            test_bq_client().await,
            Read::from_stub(stub),
            "test-quota-proj".to_string(),
            Some("custom-ua/2.0".to_string()),
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
        assert_eq!(rec.read_rows_calls, 1);
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

        let client = Client::new_for_testing(
            test_bq_client().await,
            Read::from_stub(FailingAuthStub),
            "quota-proj".to_string(),
        );
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
    async fn test_client_shares_retry_throttler() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowSchema as BqArrowSchema, ReadRowsRequest, ReadRowsResponse, ReadStream,
        };
        use google_cloud_gax::error::rpc::{Code, Status};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::retry_throttler::RetryThrottler;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug)]
        struct CountingThrottler {
            failure_notifications: Arc<AtomicUsize>,
        }
        impl RetryThrottler for CountingThrottler {
            fn throttle_retry_attempt(&self) -> bool {
                true
            }
            fn on_retry_failure(&mut self, _flow: &google_cloud_gax::retry_result::RetryResult) {
                self.failure_notifications.fetch_add(1, Ordering::SeqCst);
            }
            fn on_success(&mut self) {}
        }

        #[derive(Debug)]
        struct FailingStreamStub;

        impl google_cloud_bigquery::stub::Read for FailingStreamStub {
            async fn create_read_session(
                &self,
                _req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let session = ReadSession::new()
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(create_test_arrow_payload(0).0),
                    )
                    .set_streams(vec![ReadStream::new().set_name("streams/s0")]);
                Ok(Response::from(session))
            }

            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                Err(GaxError::service(
                    Status::default()
                        .set_code(Code::PermissionDenied)
                        .set_message("permission denied"),
                ))
            }
        }

        let failure_notifications = Arc::new(AtomicUsize::new(0));
        let shared_throttler: SharedRetryThrottler = Arc::new(Mutex::new(CountingThrottler {
            failure_notifications: Arc::clone(&failure_notifications),
        }));

        let client = Client::from_parts(
            test_bq_client().await,
            Read::from_stub(FailingStreamStub),
            "quota-proj".to_string(),
            None,
            shared_throttler,
        );

        let table = BigQueryTableId::new("proj", "ds", "tbl");
        let (_, mut rx) = client
            .read_table(&table, ReadOptions::default())
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, Err(BigQueryError::Grpc(_))));

        assert_eq!(
            failure_notifications.load(Ordering::SeqCst),
            1,
            "shared retry throttler should be notified by stream readers"
        );
    }

    #[tokio::test]
    async fn test_client_read_stream_extracts_schema_from_first_response() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{ArrowSchema as BqArrowSchema, ReadRowsResponse};

        let heartbeat = ReadRowsResponse::new();
        let (schema_bytes, resp1) = create_test_arrow_payload(3);
        let (_, resp2) = create_test_arrow_payload(2);
        let resp1 =
            resp1.set_arrow_schema(BqArrowSchema::new().set_serialized_schema(schema_bytes));

        let (stub, recorded) = CannedReadRowsStub::new(vec![heartbeat, resp1, resp2]);
        let client = Client::from_parts(
            test_bq_client().await,
            Read::from_stub(stub),
            "test-quota-proj".to_string(),
            Some("custom-ua/2.0".to_string()),
            client_builder::default_retry_throttler(),
        );

        let stream_name = "projects/p/locations/us/sessions/s/streams/0";
        let (schema, mut rx) = client
            .read_stream(stream_name, StreamOptions::default())
            .await
            .unwrap();

        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);

        let batch1 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch1.len(), 3);
        let batch2 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch2.len(), 2);
        assert!(rx.recv().await.is_none());

        let rec = recorded.lock().unwrap();
        assert_eq!(rec.stream_name.as_deref(), Some(stream_name));
        assert_eq!(rec.offset, Some(0));
        assert_eq!(rec.arrow_buffer_compression, None);
        assert_eq!(rec.quota_project.as_deref(), Some("test-quota-proj"));
        assert_eq!(rec.user_agent.as_deref(), Some("custom-ua/2.0"));
    }

    #[tokio::test]
    async fn test_client_read_stream_with_provided_schema() {
        use google_cloud_bigquery::client::Read;

        let (schema_bytes, resp1) = create_test_arrow_payload(4);
        let (stub, recorded) = CannedReadRowsStub::new(vec![resp1]);
        let client = Client::new_for_testing(
            test_bq_client().await,
            Read::from_stub(stub),
            "test-quota-proj".to_string(),
        );

        let (schema, mut rx) = client
            .read_stream(
                "streams/s1",
                StreamOptions {
                    offset: 42,
                    schema: Some(bytes::Bytes::from(schema_bytes)),
                    arrow_buffer_compression: Some(
                        arrow_serialization_options::CompressionCodec::Zstd,
                    ),
                },
            )
            .await
            .unwrap();

        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);

        let batch = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch.len(), 4);
        assert!(rx.recv().await.is_none());
        let rec = recorded.lock().unwrap();
        assert_eq!(rec.offset, Some(42));
        assert_eq!(
            rec.arrow_buffer_compression,
            Some(arrow_serialization_options::CompressionCodec::Zstd)
        );
    }

    #[tokio::test]
    async fn test_client_read_stream_errors() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{ReadRowsRequest, ReadRowsResponse};
        use google_cloud_gax::error::rpc::{Code, Status};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug)]
        struct MissingSchemaStub;

        impl google_cloud_bigquery::stub::Read for MissingSchemaStub {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                let _ = tx.send(Ok(ReadRowsResponse::new())).await;
                Ok(ResponseStream::from(rx))
            }
        }

        let client = Client::new_for_testing(
            test_bq_client().await,
            Read::from_stub(MissingSchemaStub),
            "test-quota-proj".to_string(),
        );
        let err = match client
            .read_stream("streams/s1", StreamOptions::default())
            .await
        {
            Err(e) => e,
            Ok(_) => panic!("expected Protocol error when stream closes without schema"),
        };
        assert!(matches!(err, BigQueryError::Protocol(_)));

        #[derive(Debug)]
        struct EmptyStreamStub;

        impl google_cloud_bigquery::stub::Read for EmptyStreamStub {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let (_tx, rx) = tokio::sync::mpsc::channel(1);
                Ok(ResponseStream::from(rx))
            }
        }

        let client = Client::new_for_testing(
            test_bq_client().await,
            Read::from_stub(EmptyStreamStub),
            "test-quota-proj".to_string(),
        );
        let err = match client
            .read_stream("streams/s1", StreamOptions::default())
            .await
        {
            Err(e) => e,
            Ok(_) => panic!("expected Protocol error when stream closes without sending schema"),
        };
        assert!(matches!(err, BigQueryError::Protocol(_)));

        #[derive(Debug)]
        struct FailingReadRowsStub;

        impl google_cloud_bigquery::stub::Read for FailingReadRowsStub {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                Err(GaxError::service(
                    Status::default()
                        .set_code(Code::PermissionDenied)
                        .set_message("permission denied"),
                ))
            }
        }

        let client = Client::new_for_testing(
            test_bq_client().await,
            Read::from_stub(FailingReadRowsStub),
            "test-quota-proj".to_string(),
        );

        let err = match client
            .read_stream("streams/s1", StreamOptions::default())
            .await
        {
            Err(e) => e,
            Ok(_) => panic!("expected Grpc error when ReadRows fails before schema is known"),
        };
        assert!(matches!(err, BigQueryError::Grpc(_)));

        let err = match client
            .read_stream(
                "streams/s1",
                StreamOptions {
                    offset: 0,
                    schema: Some(bytes::Bytes::new()),
                    arrow_buffer_compression: None,
                },
            )
            .await
        {
            Err(e) => e,
            Ok(_) => panic!("expected Arrow error when schema bytes are empty/invalid"),
        };
        assert!(matches!(err, BigQueryError::Arrow(_)));

        let (schema_bytes, _) = create_test_arrow_payload(1);
        let (_schema, mut rx) = client
            .read_stream(
                "streams/s1",
                StreamOptions {
                    offset: 0,
                    schema: Some(bytes::Bytes::from(schema_bytes)),
                    arrow_buffer_compression: None,
                },
            )
            .await
            .unwrap();
        let err = rx
            .recv()
            .await
            .expect("expected error item")
            .expect_err("expected Grpc error when ReadRows fails");
        assert!(matches!(err, BigQueryError::Grpc(_)));
    }
}
