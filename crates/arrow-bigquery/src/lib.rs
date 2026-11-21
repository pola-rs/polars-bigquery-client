mod bigquery_read_retry;
mod bigquery_read_stream;
pub mod client_builder;
mod decode;
mod error;
mod job_cancel;
mod job_service_retry;
pub mod query;

use std::sync::Arc;

pub use client_builder::*;
pub use decode::{validate_ffi_schema, SchemaDecodeOptions};
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
    project_id: String,
    dataset_id: String,
    table_id: String,
}

fn is_valid_project_label(label: &str) -> bool {
    !label.is_empty()
        && label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn validate_project_id(project_id: &str) -> Result<(), InvalidTableId> {
    if project_id.is_empty() {
        return Err(InvalidTableId("project_id must not be empty".into()));
    }

    match project_id.split_once(':') {
        Some((domain, project)) => {
            if domain.is_empty() || !domain.split('.').all(is_valid_project_label) {
                return Err(InvalidTableId(
                    "project_id contains invalid domain prefix".into(),
                ));
            }
            if !is_valid_project_label(project) {
                return Err(InvalidTableId(
                    "project_id contains invalid characters".into(),
                ));
            }
        },
        None => {
            if !is_valid_project_label(project_id) {
                return Err(InvalidTableId(
                    "project_id contains invalid characters".into(),
                ));
            }
        },
    }

    Ok(())
}

fn is_valid_dataset_char(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_alphanumeric() || matches!(c, '_' | '-')
    } else {
        !c.is_control() && !c.is_whitespace()
    }
}

fn validate_dataset_id(dataset_id: &str) -> Result<(), InvalidTableId> {
    if dataset_id.is_empty() {
        return Err(InvalidTableId("dataset_id must not be empty".into()));
    }
    if !dataset_id
        .split('.')
        .all(|part| !part.is_empty() && part.chars().all(is_valid_dataset_char))
    {
        return Err(InvalidTableId(
            "dataset_id contains invalid segment or characters".into(),
        ));
    }
    Ok(())
}

fn is_unicode_space_separator(c: char) -> bool {
    matches!(
        c,
        ' ' | '\u{00A0}' | '\u{1680}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    ) || ('\u{2000}'..='\u{200A}').contains(&c)
}

fn is_valid_table_base_char(c: char) -> bool {
    if c.is_ascii() {
        // BigQuery table names allow Unicode categories L, M, N, Pc ('_'), Pd ('-'), and Zs (' ').
        c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ' ')
    } else {
        // Allow non-ASCII Unicode letters, marks, numbers, connectors, dashes, and Zs spaces,
        // while rejecting C1 control chars and Unicode line/paragraph separators (Zl, Zp).
        !c.is_control() && !matches!(c, '\u{2028}' | '\u{2029}')
    }
}

fn is_valid_decorator_value(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn validate_table_id(table_id: &str) -> Result<(), InvalidTableId> {
    if table_id.is_empty() {
        return Err(InvalidTableId("table_id must not be empty".into()));
    }

    let (base, suffix) = match table_id.find(['$', '@']) {
        Some(idx) => (&table_id[..idx], Some(&table_id[idx..])),
        None => (table_id, None),
    };

    if base.is_empty() || base.chars().all(is_unicode_space_separator) {
        return Err(InvalidTableId(
            "table_id must contain a non-empty table name".into(),
        ));
    }
    if !base.chars().all(is_valid_table_base_char) {
        return Err(InvalidTableId(
            "table_id contains invalid characters".into(),
        ));
    }

    if let Some(mut rest) = suffix {
        let mut saw_dollar = false;
        let mut saw_at = false;
        while !rest.is_empty() {
            match rest.as_bytes()[0] {
                b'$' => {
                    if saw_dollar {
                        return Err(InvalidTableId(
                            "table_id contains duplicate partition decorator".into(),
                        ));
                    }
                    saw_dollar = true;
                },
                b'@' => {
                    if saw_at {
                        return Err(InvalidTableId(
                            "table_id contains duplicate snapshot decorator".into(),
                        ));
                    }
                    saw_at = true;
                },
                _ => unreachable!("suffix always begins at '$' or '@'"),
            }
            let after_sep = &rest[1..];
            let next_sep = after_sep.find(['$', '@']).unwrap_or(after_sep.len());
            let value = &after_sep[..next_sep];
            if !is_valid_decorator_value(value) {
                return Err(InvalidTableId(
                    "table_id contains invalid decorator segment".into(),
                ));
            }
            rest = &after_sep[next_sep..];
        }
    }

    Ok(())
}

impl BigQueryTableId {
    pub fn new(
        project_id: impl Into<String>,
        dataset_id: impl Into<String>,
        table_id: impl Into<String>,
    ) -> Result<Self, BigQueryError> {
        let project_id = project_id.into();
        let dataset_id = dataset_id.into();
        let table_id = table_id.into();

        validate_project_id(&project_id)?;
        validate_dataset_id(&dataset_id)?;
        validate_table_id(&table_id)?;

        Ok(Self {
            project_id,
            dataset_id,
            table_id,
        })
    }

    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    pub fn dataset_id(&self) -> &str {
        &self.dataset_id
    }

    pub fn table_id(&self) -> &str {
        &self.table_id
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

fn parse_table_id_parts<'s>(input: &mut &'s str) -> winnow::Result<(String, String, String)> {
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

    Ok((project_id, dataset_id, table_id))
}

impl std::str::FromStr for BigQueryTableId {
    type Err = BigQueryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut input = s;
        let (project_id, dataset_id, table_id) = parse_table_id_parts(&mut input)
            .map_err(|_| InvalidTableId("expected format 'project.dataset.table'".into()))?;
        Self::new(project_id, dataset_id, table_id)
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
    pub maintain_order: bool,
    pub offset: i64,
    /// Optional IPC-serialized Arrow schema bytes.
    ///
    /// If `None`, the schema is extracted from the first `ReadRowsResponse`
    /// received from the stream.
    pub schema: Option<bytes::Bytes>,
    pub arrow_buffer_compression: Option<arrow_serialization_options::CompressionCodec>,
}

pub(crate) fn apply_request_options_parts<B: RequestOptionsBuilder>(
    builder: B,
    quota_project_id: &str,
    user_agent: Option<&str>,
) -> B {
    let builder = builder.with_quota_project(quota_project_id);
    match user_agent {
        Some(user_agent) => builder.with_user_agent(user_agent),
        None => builder,
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
    handles: Vec<tokio::task::JoinHandle<()>>,
    /// Best-effort `jobs.cancel` slot triggered if dropped/aborted before clean EOF.
    job_cancel: job_cancel::JobCancelSlot,
}

impl BigQueryRecordBatchReceiver {
    pub async fn recv(&mut self) -> Option<Result<RecordBatch, BigQueryError>> {
        let item = self.rx.recv().await;
        match &item {
            None | Some(Err(BigQueryError::JobFailed(_))) => {
                // Clean EOF (or a defensive guard if a stream path emits a confirmed
                // server-side `JobFailed`, e.g. from cutover `DestinationTableRead::resolve`):
                // disarm cancellation to skip issuing a redundant `jobs.cancel`.
                self.job_cancel.disarm();
                self.abort();
            },
            Some(Err(_)) => {
                // Mid-stream transport, protocol, or decode error: keep `job_cancel` armed
                // so `abort()` triggers best-effort `jobs.cancel` if the job is still running.
                self.abort();
            },
            Some(Ok(_)) => {},
        }
        item
    }

    /// Immediately cancels and aborts all background stream tasks owned by this receiver.
    ///
    /// If attached to an incomplete BigQuery query job that has not yet reached EOF,
    /// spawns a best-effort detached `jobs.cancel` RPC to release server-side resources.
    pub fn abort(&mut self) {
        if let Some(cancel_tx) = self.cancel_tx.take() {
            let _ = cancel_tx.send_replace(true);
        }
        self.rx.close();
        for handle in self.handles.drain(..) {
            handle.abort();
        }
        self.job_cancel.trigger();
    }

    pub(crate) fn set_job_cancel(&mut self, handle: job_cancel::JobCancelHandle) {
        self.job_cancel.arm(handle);
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
            job_cancel: job_cancel::JobCancelSlot::Inactive,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct InvalidTableId(String);

impl std::fmt::Display for InvalidTableId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid table ID: {}", self.0)
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
    decode_options: SchemaDecodeOptions,
    cancel_pool: job_cancel::JobCancelPool,
}

fn channel_size() -> usize {
    rayon::current_num_threads().max(1) * 2
}

impl Client {
    pub fn builder() -> ServiceConfigBuilder {
        ServiceConfigBuilder::new()
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn from_parts(
        bq_client: BigQueryJobService,
        read_client: BigQueryReadClient,
        quota_project_id: String,
        user_agent: Option<String>,
        retry_throttler: SharedRetryThrottler,
    ) -> Self {
        Self::from_parts_with_options(
            bq_client,
            read_client,
            quota_project_id,
            user_agent,
            retry_throttler,
            SchemaDecodeOptions::default(),
        )
    }

    pub(crate) fn from_parts_with_options(
        bq_client: BigQueryJobService,
        read_client: BigQueryReadClient,
        quota_project_id: String,
        user_agent: Option<String>,
        retry_throttler: SharedRetryThrottler,
        decode_options: SchemaDecodeOptions,
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
            decode_options,
            cancel_pool: job_cancel::JobCancelPool::new(),
        }
    }

    pub(crate) fn schema_decode_options(&self) -> SchemaDecodeOptions {
        self.decode_options
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
        apply_request_options_parts(builder, &self.quota_project_id, self.user_agent.as_deref())
    }

    pub(crate) fn new_job_cancel_handle(
        &self,
        project_id: impl Into<String>,
        job_id: impl Into<String>,
        location: impl Into<String>,
        runtime: tokio::runtime::Handle,
        token: job_cancel::JobCancelToken,
    ) -> job_cancel::JobCancelHandle {
        job_cancel::JobCancelHandle::new(
            self.bq_client.clone(),
            Arc::clone(&self.quota_project_id),
            self.user_agent.clone(),
            project_id,
            job_id,
            location,
            runtime,
            self.cancel_pool.clone(),
            token,
        )
    }

    #[cfg(test)]
    pub(crate) fn cancel_pool(&self) -> &job_cancel::JobCancelPool {
        &self.cancel_pool
    }

    #[cfg(test)]
    pub(crate) fn read_retry_throttler(&self) -> &SharedRetryThrottler {
        &self.retry_throttler
    }

    pub(crate) fn stream_retry_config(&self) -> bigquery_read_retry::StreamRetryConfig {
        self.retry_config.clone()
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
        let (tx, rx) = tokio::sync::mpsc::channel(channel_size());
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
                maintain_order,
                cancel_tx: Arc::clone(&cancel_tx),
                cancel_rx,
                schema_decode_options: self.schema_decode_options(),
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

    /// Opens a Storage Read API `ReadSession` for `table`, parses its Arrow IPC
    /// schema, and prepares [`bigquery_read_stream::ReadStreamOptions`] for each
    /// session stream wired to `(tx, cancel_tx, cancel_rx)`.
    ///
    /// Shared by [`Self::read_table`] (upfront table and destination-table reads)
    /// and [`Self::switch_to_read_session`] (mid-flight cutover from a query job's
    /// `streams/_default` into an already-open batch channel).
    async fn prepare_table_read_session(
        &self,
        table: &BigQueryTableId,
        options: ReadOptions,
        tx: &tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
        cancel_tx: &Arc<tokio::sync::watch::Sender<bool>>,
        cancel_rx: &tokio::sync::watch::Receiver<bool>,
    ) -> Result<(ArrowSchemaRef, Vec<bigquery_read_stream::ReadStreamOptions>), BigQueryError> {
        let maintain_order = options.maintain_order;
        let (schema, streams) = self.create_table_read_session(table, options).await?;
        let decode_options = self.schema_decode_options();
        let metadata = Arc::new(decode::parse_arrow_schema_with_options(
            &schema,
            decode_options,
        )?);
        let schema_ref = Arc::new(metadata.schema.clone());
        let retry_config = self.stream_retry_config();
        let stream_options = streams
            .into_iter()
            .map(|stream_name| bigquery_read_stream::ReadStreamOptions {
                read_rows: self.prepare_read_rows(stream_name, 0, None),
                current_offset: 0,
                metadata: Some(Arc::clone(&metadata)),
                schema_tx: None,
                first_page_check: None,
                tx: tx.clone(),
                retry_config: retry_config.clone(),
                maintain_order,
                cancel_tx: Arc::clone(cancel_tx),
                cancel_rx: cancel_rx.clone(),
                schema_decode_options: decode_options,
            })
            .collect();
        Ok((schema_ref, stream_options))
    }

    /// Reads a BigQuery table using a Storage Read API `ReadSession`, spawning a
    /// background reader task per session stream.
    pub async fn read_table(
        &self,
        table: &BigQueryTableId,
        options: ReadOptions,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let (tx, rx) = tokio::sync::mpsc::channel(channel_size());
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        let (schema_ref, stream_options) = self
            .prepare_table_read_session(table, options, &tx, &cancel_tx, &cancel_rx)
            .await?;
        let handles = stream_options
            .into_iter()
            .map(|opts| {
                tokio::task::spawn(async move {
                    let _ = bigquery_read_stream::read_stream(opts).await;
                })
            })
            .collect();
        Ok((
            schema_ref,
            BigQueryRecordBatchReceiver::new(rx, Some(cancel_tx), handles),
        ))
    }

    /// Reads a single BigQuery Storage Read API stream using the provided
    /// [`StreamOptions`].
    pub async fn read_stream(
        &self,
        stream: impl Into<String>,
        options: StreamOptions,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let (schema_ref, receiver) = self
            .read_stream_with_first_page_check(stream.into(), options, None)
            .await?;
        let schema_ref = schema_ref.ok_or_else(|| {
            BigQueryError::Protocol("ReadRows stream closed without sending an Arrow schema".into())
        })?;
        Ok((schema_ref, receiver))
    }

    /// Cuts over mid-flight from a query job's single `streams/_default` stream
    /// to a parallel multi-stream `ReadSession` on the job's destination table,
    /// streaming batches into the already-open `(tx, cancel_tx, cancel_rx)`
    /// channel owned by the caller's [`BigQueryRecordBatchReceiver`].
    ///
    /// Resolves the destination table via [`query::DestinationTableRead::resolve`]
    /// and prepares the session streams via [`Self::prepare_table_read_session`]
    /// (matching [`query::DestinationTableRead::read`] / [`Self::read_table`]),
    /// running the stream tasks inside a [`tokio::task::JoinSet`] so aborting the
    /// outer receiver task also aborts all child stream tasks.
    async fn switch_to_read_session(
        &self,
        destination_table_read: query::DestinationTableRead,
        tx: &tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
        cancel_tx: &Arc<tokio::sync::watch::Sender<bool>>,
        cancel_rx: &tokio::sync::watch::Receiver<bool>,
    ) -> Result<(), BigQueryError> {
        let (destination_table, options) = destination_table_read.resolve(self).await?;
        let (_schema_ref, stream_options) = self
            .prepare_table_read_session(&destination_table, options, tx, cancel_tx, cancel_rx)
            .await?;
        let mut join_set = tokio::task::JoinSet::new();
        for opts in stream_options {
            join_set.spawn(bigquery_read_stream::read_stream(opts));
        }
        while join_set.join_next().await.is_some() {}
        Ok(())
    }

    /// Reads a single BigQuery Storage Read API stream, optionally extracting
    /// the Arrow schema from the first non-heartbeat `ReadRowsResponse` and
    /// optionally cutting over to a multi-stream destination-table
    /// `ReadSession` via [`Self::switch_to_read_session`] if `first_page_check`
    /// triggers.
    pub(crate) async fn read_stream_with_first_page_check(
        &self,
        stream: String,
        options: StreamOptions,
        first_page_check: Option<(
            bigquery_read_stream::FirstPageCheck,
            query::DestinationTableRead,
        )>,
    ) -> Result<(Option<ArrowSchemaRef>, BigQueryRecordBatchReceiver), BigQueryError> {
        let decode_options = self.schema_decode_options();
        let (metadata, schema_ref, schema_tx, schema_rx) = match &options.schema {
            Some(schema) => {
                let metadata = Arc::new(decode::parse_arrow_schema_with_options(
                    schema,
                    decode_options,
                )?);
                let schema_ref = Arc::new(metadata.schema.clone());
                (Some(metadata), Some(schema_ref), None, None)
            },
            None => {
                let (tx, rx) = tokio::sync::oneshot::channel();
                (None, None, Some(tx), Some(rx))
            },
        };

        let (tx, rx) = tokio::sync::mpsc::channel(channel_size());
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        let retry_config = self.stream_retry_config();
        let maintain_order = options.maintain_order;
        let offset = options.offset;
        let read_rows = self.prepare_read_rows(stream, offset, options.arrow_buffer_compression);
        let (check, fallback) = match first_page_check {
            Some((check, destination_table_read)) => {
                (Some(check), Some((self.clone(), destination_table_read)))
            },
            None => (None, None),
        };
        let task_cancel_tx = Arc::clone(&cancel_tx);
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
                    maintain_order,
                    cancel_tx: Arc::clone(&task_cancel_tx),
                    cancel_rx: cancel_rx.clone(),
                    schema_decode_options: decode_options,
                })
                .await;

            if reason == bigquery_read_stream::TerminationReason::SwitchToReadSession {
                if let Some((client, destination_table_read)) = fallback {
                    if let Err(err) = client
                        .switch_to_read_session(
                            destination_table_read,
                            &tx,
                            &task_cancel_tx,
                            &cancel_rx,
                        )
                        .await
                    {
                        let _ = task_cancel_tx.send_replace(true);
                        let _ = tx.send(Err(err)).await;
                    }
                }
            }
        });

        let receiver = BigQueryRecordBatchReceiver::new(rx, Some(cancel_tx), vec![handle]);

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
        assert!("google..com:proj.ds.tab"
            .parse::<BigQueryTableId>()
            .is_err());
        assert!(".google.com:proj.ds.tab"
            .parse::<BigQueryTableId>()
            .is_err());
        assert!("google.com.:proj.ds.tab"
            .parse::<BigQueryTableId>()
            .is_err());
        assert!("..:proj.ds.tab".parse::<BigQueryTableId>().is_err());
        assert!("google.com:proj.ds:extra.tab"
            .parse::<BigQueryTableId>()
            .is_err());
        assert!("google.com:proj.ds.tab:extra"
            .parse::<BigQueryTableId>()
            .is_err());
    }

    #[test]
    fn test_table_id_new_and_getters_valid_cases() {
        let valid_cases = [
            ("proj", "ds", "tab"),
            ("p", "d", "t"),
            ("my-project-123", "my_dataset", "my_table"),
            ("my_project", "_temp_dataset", "anon_table_123"),
            ("123456789", "ds-obj", "tab-obj"),
            ("google.com:my-project", "my_dataset", "my_table"),
            ("sub.example.com:proj-1", "cat.ns1.ns2", "tab-1"),
            ("proj", "region-us.INFORMATION_SCHEMA", "JOBS_BY_PROJECT"),
            ("proj", "ds", "my_table$20200101"),
            ("proj", "ds", "my_table$__NULL__"),
            ("proj", "ds", "my_table@1700000000000"),
            ("proj", "ds", "my_table@-3600000"),
            ("proj", "ds", "my_table@-3600000-"),
            ("proj", "ds", "my_table@1000-2000"),
            ("proj", "ds", "table 01"),
            ("proj", "ds", "my  table"),
            ("proj", "ds", " table 01 "),
            ("proj", "ds", "table\u{00A0}01"),
            ("proj", "ds", "テーブル\u{3000}01"),
            ("proj", "ds", "table–01"),
            ("proj", "ds", "table＿01"),
            ("proj", "ds", "table 01$20200101"),
            ("proj", "ds", "table 01@1700000000000"),
            ("proj", "ds", "my_table$20200101@1700000000000"),
            ("google.com:my-project", "ds", "table 01"),
            ("proj", "ds", "étudiant-01"),
            ("proj", "ds", "00_お客様"),
            ("proj", "ds", "ग्राहक"),
            ("proj", "ds", "e\u{0301}tudiant"),
            ("proj", "数据_集", "étudiant-01$20240101"),
        ];

        for (project, dataset, table) in valid_cases {
            let id = BigQueryTableId::new(project, dataset, table)
                .unwrap_or_else(|e| panic!("expected ({project}, {dataset}, {table}) valid: {e}"));
            assert_eq!(id.project_id(), project);
            assert_eq!(id.dataset_id(), dataset);
            assert_eq!(id.table_id(), table);
            assert_eq!(
                id.to_table_path(),
                format!("projects/{project}/datasets/{dataset}/tables/{table}")
            );

            // Round-trip through Display and FromStr
            let formatted = id.to_string();
            let parsed: BigQueryTableId = formatted
                .parse()
                .unwrap_or_else(|e| panic!("round-trip parse failed for {formatted:?}: {e}"));
            assert_eq!(parsed, id);
        }
    }

    #[test]
    fn test_table_id_new_and_from_str_reject_invalid_segments() {
        let invalid_projects = [
            "",
            ".",
            "..",
            "proj.extra",
            ":proj",
            "google.com:",
            "google..com:proj",
            ".google.com:proj",
            "google.com.:proj",
            "..:proj",
            ".:proj",
            "google.com:proj:extra",
            "google.com:proj.extra",
            "proj/../../evil",
            "proj/evil",
            "proj\\evil",
            "proj?fields=all",
            "proj#frag",
            "proj%2f",
            "proj%2e%2e",
            "`proj`",
            "proj ",
            " proj",
            "proj name",
            "proj\t",
            "proj\n",
            "proj\r\n",
            "proj\x00",
            "proj\x1f",
            "proj\x7f",
            "proj\u{0085}",
            "proj\u{00a0}",
            "proj\u{3000}",
            "proj;evil=1",
            "proj&x=1",
            "proj=1",
            "proj$1",
            "proj@1",
            "café",
        ];
        for bad in invalid_projects {
            assert!(
                BigQueryTableId::new(bad, "ds", "tab").is_err(),
                "expected project_id {bad:?} to be rejected by new()"
            );
            if !matches!(bad, "proj.extra" | "google.com:proj.extra") {
                // Note: concatenating ".ds.tab" onto "proj.extra" or "google.com:proj.extra"
                // produces a valid multi-part dataset string (dataset "extra.ds").
                assert!(
                    format!("{bad}.ds.tab").parse::<BigQueryTableId>().is_err(),
                    "expected project_id {bad:?} to be rejected by from_str()"
                );
            }
        }

        let invalid_datasets = [
            "",
            ".",
            "..",
            ".ds",
            "ds.",
            "a..b",
            "ds:extra",
            "ds/../../evil",
            "ds/evil",
            "ds\\evil",
            "ds?fields=all",
            "ds#frag",
            "ds%2f",
            "`ds`",
            "ds ",
            " ds",
            "ds name",
            "ds\t",
            "ds\n",
            "ds\r\n",
            "ds\x00",
            "ds\x1f",
            "ds\x7f",
            "ds\u{0085}",
            "ds\u{009f}",
            "ds\u{00a0}",
            "ds\u{2028}",
            "ds\u{2029}",
            "ds\u{3000}",
            "ds;evil=1",
            "ds&x=1",
            "ds=1",
            "ds$1",
            "ds@1",
            "ds'quote",
            "ds\"quote",
        ];
        for bad in invalid_datasets {
            assert!(
                BigQueryTableId::new("proj", bad, "tab").is_err(),
                "expected dataset_id {bad:?} to be rejected by new()"
            );
            assert!(
                format!("proj.{bad}.tab")
                    .parse::<BigQueryTableId>()
                    .is_err(),
                "expected dataset_id {bad:?} to be rejected by from_str()"
            );
        }

        let invalid_tables = [
            "",
            " ",
            "   ",
            "\u{00a0}",
            "\u{3000}",
            "   $20200101",
            ".",
            "..",
            "a.b",
            "tab:delete",
            "tab/../../evil",
            "tab/evil",
            "tab\\evil",
            "tab?fields=all#",
            "tab?fields=etag#frag",
            "tab#frag",
            "tab%2f",
            "tab%2e%2e",
            "`tab`",
            "tab\t",
            "tab\n",
            "tab\r\n",
            "tab\x0b",
            "tab\x0c",
            "tab\x00",
            "tab\x1f",
            "tab\x7f",
            "tab\u{0085}",
            "tab\u{009f}",
            "tab\u{2028}",
            "tab\u{2029}",
            "tab HTTP/1.1",
            "tab;evil=1",
            "tab&fields=all",
            "tab=1",
            "tab'quote",
            "tab\"quote",
            "tab<script>",
            "$",
            "@",
            "$20200101",
            "@1700000000000",
            "tab$",
            "tab@",
            "tab$$20200101",
            "tab$2020$2021",
            "tab@@1234",
            "tab@1@2",
            "tab$@1234",
            "tab@$1234",
            "tab$2020 01",
            "tab@123 456",
            "tab$2020?x=1",
        ];
        for bad in invalid_tables {
            assert!(
                BigQueryTableId::new("proj", "ds", bad).is_err(),
                "expected table_id {bad:?} to be rejected by new()"
            );
            if bad != "a.b" {
                // Note: "proj.ds.a.b" parses as multi-part dataset "ds.a" and table "b".
                assert!(
                    format!("proj.ds.{bad}").parse::<BigQueryTableId>().is_err(),
                    "expected table_id {bad:?} to be rejected by from_str()"
                );
            }
        }
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

        let table = BigQueryTableId::new("proj", "ds", "tbl").unwrap();
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

        let default_options = StreamOptions::default();
        assert!(!default_options.maintain_order);

        let stream_name = "projects/p/locations/us/sessions/s/streams/0";
        let (schema, mut rx) = client
            .read_stream(stream_name, default_options)
            .await
            .unwrap();

        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);

        let batch1 = rx.recv().await.unwrap().unwrap();
        let batch2 = rx.recv().await.unwrap().unwrap();
        assert!(rx.recv().await.is_none());
        let mut batch_lens = [batch1.len(), batch2.len()];
        batch_lens.sort_unstable();
        assert_eq!(batch_lens, [2, 3]);

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
        let (_, resp2) = create_test_arrow_payload(1);
        let (stub, recorded) = CannedReadRowsStub::new(vec![resp1, resp2]);
        let client = Client::new_for_testing(
            test_bq_client().await,
            Read::from_stub(stub),
            "test-quota-proj".to_string(),
        );

        let (schema, mut rx) = client
            .read_stream(
                "streams/s1",
                StreamOptions {
                    maintain_order: true,
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

        let batch1 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch1.len(), 4);
        let batch2 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch2.len(), 1);
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
                    schema: Some(bytes::Bytes::new()),
                    ..Default::default()
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
                    schema: Some(bytes::Bytes::from(schema_bytes)),
                    ..Default::default()
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
