use std::sync::Arc;

use google_cloud_bigquery_v2::model::query_request::{JobCreationMode, QueryResultsFormat};
use google_cloud_bigquery_v2::model::{
    arrow_serialization_options, query_response, ArrowSerializationOptions, QueryRequest,
    QueryResponse,
};
use google_cloud_gax::options::RequestOptionsBuilder;
use polars_arrow::array::{Array, BooleanArray, Int64Array, Utf8Array};
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, ArrowSchemaRef, Field, TimeUnit};
use polars_arrow::record_batch::RecordBatch;

use crate::error::{BigQueryError, JobError, JobFailedError};
use crate::{BigQueryRecordBatchReceiver, StreamOptions};

/// Minimum estimated remaining bytes in the single job stream before switching
/// to a multi-stream `ReadSession` on the destination table when `maintain_order`
/// is `false`.
const MULTI_STREAM_THRESHOLD_BYTES: f64 = (250 * 1024 * 1024) as f64;

fn bq_to_gapic_compression_codec(
    codec: crate::CompressionCodec,
) -> arrow_serialization_options::CompressionCodec {
    use arrow_serialization_options::CompressionCodec as V2;

    use crate::CompressionCodec as V1;
    match codec {
        V1::CompressionUnspecified => V2::CompressionUnspecified,
        V1::Lz4Frame => V2::Lz4Frame,
        V1::Zstd => V2::Zstd,
        other => {
            if let Some(val) = other.value() {
                V2::from(val)
            } else if let Some(name) = other.name() {
                V2::from(name)
            } else {
                V2::CompressionUnspecified
            }
        },
    }
}

#[derive(Debug)]
pub struct Query<'a> {
    client: &'a crate::Client,
    sql: String,
    arrow_buffer_compression: Option<crate::CompressionCodec>,
    maintain_order: bool,
    use_query_cache: Option<bool>,
}

impl<'a> Query<'a> {
    pub(crate) fn new(client: &'a crate::Client, sql: impl Into<String>) -> Self {
        Self {
            client,
            sql: sql.into(),
            arrow_buffer_compression: None,
            maintain_order: true,
            use_query_cache: None,
        }
    }

    pub fn with_arrow_buffer_compression(
        mut self,
        codec: impl Into<Option<crate::CompressionCodec>>,
    ) -> Self {
        self.arrow_buffer_compression = codec.into();
        self
    }

    pub fn with_maintain_order(mut self, maintain_order: bool) -> Self {
        self.maintain_order = maintain_order;
        self
    }

    pub fn with_use_query_cache(mut self, use_query_cache: impl Into<Option<bool>>) -> Self {
        self.use_query_cache = use_query_cache.into();
        self
    }

    fn arrow_buffer_compression(&self) -> crate::CompressionCodec {
        self.arrow_buffer_compression
            .clone()
            .unwrap_or(crate::DEFAULT_COMPRESSION)
    }

    fn build_read_request(&self) -> QueryRequest {
        let arrow_options = ArrowSerializationOptions::new().set_buffer_compression(
            bq_to_gapic_compression_codec(self.arrow_buffer_compression()),
        );
        QueryRequest::new()
            .set_job_creation_mode(JobCreationMode::JobCreationOptional)
            .set_query_results_format(QueryResultsFormat::Arrow)
            .set_arrow_serialization_options(arrow_options)
            .set_or_clear_use_query_cache(self.use_query_cache)
    }

    fn prepare_query(
        &self,
        query_request: QueryRequest,
    ) -> google_cloud_bigquery_v2::builder::job_service::Query {
        self.client.apply_request_options(
            self.client
                .bq_client
                .query()
                .set_project_id(&*self.client.quota_project_id)
                .set_query_request(query_request.set_query(&self.sql).set_use_legacy_sql(false)),
        )
    }

    async fn read_destination_table(
        &self,
        response: &QueryResponse,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let (project_id, job_id, location) = extract_required_job_reference(response)?;
        let destination_table =
            get_destination_table(self.client, project_id, job_id, location).await?;
        self.client
            .read_table(
                &destination_table,
                crate::ReadOptions {
                    maintain_order: false,
                    arrow_buffer_compression: Some(self.arrow_buffer_compression()),
                    ..Default::default()
                },
            )
            .await
    }

    async fn read_job_stream_or_table(
        &self,
        response: &QueryResponse,
        stream_name: String,
        offset: i64,
        serialized_schema: Option<bytes::Bytes>,
    ) -> Result<(Option<ArrowSchemaRef>, BigQueryRecordBatchReceiver), BigQueryError> {
        let first_page_check = if !self.maintain_order {
            let (project_id, job_id, location) = extract_required_job_reference(response)?;
            Some((
                crate::bigquery_read_stream::FirstPageCheck {
                    threshold_bytes: MULTI_STREAM_THRESHOLD_BYTES,
                    fallback_total_rows: response.total_rows,
                },
                crate::FallbackReadSession {
                    project_id: project_id.to_string(),
                    job_id: job_id.to_string(),
                    location: location.to_string(),
                    read_options: crate::ReadOptions {
                        maintain_order: false,
                        arrow_buffer_compression: Some(self.arrow_buffer_compression()),
                        ..Default::default()
                    },
                },
            ))
        } else {
            None
        };
        self.client
            .read_stream_with_first_page_check(
                stream_name,
                StreamOptions {
                    offset,
                    schema: serialized_schema,
                    arrow_buffer_compression: Some(self.arrow_buffer_compression()),
                },
                first_page_check,
            )
            .await
    }

    pub async fn dry_run(&self) -> Result<RecordBatch, BigQueryError> {
        let response = self
            .prepare_query(QueryRequest::new().set_dry_run(true))
            .with_idempotency(true)
            .send()
            .await
            .map_err(map_query_error)?;
        // Keep column names/types in sync with pandas-gbq:
        // https://github.com/googleapis/google-cloud-python/blob/main/packages/pandas-gbq/pandas_gbq/dry_runs.py#L4
        query_response_to_record_batch(&response)
    }

    pub async fn read(
        &self,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let mut response = self
            .prepare_query(self.build_read_request())
            .send()
            .await
            .map_err(map_query_error)?;

        if response.job_complete == Some(false) {
            let stream_name = default_job_stream_path(&response)?;
            let res = self
                .read_job_stream_or_table(&response, stream_name, 0, None)
                .await;
            return match res {
                Ok((Some(schema_ref), receiver)) => Ok((schema_ref, receiver)),
                Ok((None, _receiver)) => self.resolve_empty_job_stream(&response).await,
                Err(err) => Err(self.resolve_job_error(&response, err).await),
            };
        }

        let serialized_schema = match response.results_schema.take() {
            Some(query_response::ResultsSchema::ArrowSchema(value))
                if !value.serialized_schema.is_empty() =>
            {
                Some(value.serialized_schema)
            },
            None | Some(query_response::ResultsSchema::ArrowSchema(_)) => None,
            Some(_) => {
                return Err(BigQueryError::Protocol(
                    "Unexpectedly got schema type other than arrow".into(),
                ))
            },
        };

        let page_row_count = u64::try_from(response.page_row_count).map_err(|_| {
            BigQueryError::Protocol(format!(
                "Negative page_row_count in QueryResponse: {}",
                response.page_row_count
            ))
        })?;

        let raw_first_page = match response.results.take() {
            Some(query_response::Results::ArrowRecordBatch(value)) => {
                Some(value.serialized_record_batch)
            },
            None => None,
            Some(_) => {
                return Err(BigQueryError::Protocol(
                    "Unexpectedly got some format other than arrow bytes".into(),
                ));
            },
        };
        let first_page_bytes = crate::bigquery_read_stream::validate_non_empty_arrow_bytes(
            raw_first_page,
            response.page_row_count,
        )?;

        if !self.maintain_order
            && crate::bigquery_read_stream::estimate_remaining_bytes(
                first_page_bytes.as_ref(),
                page_row_count,
                response.total_rows,
            ) >= MULTI_STREAM_THRESHOLD_BYTES
        {
            return self.read_destination_table(&response).await;
        }

        let has_remaining_rows = response
            .total_rows
            .is_some_and(|total_rows| total_rows > page_row_count)
            || !response.page_token.is_empty();

        let stream_name = if has_remaining_rows {
            Some(default_job_stream_path(&response)?)
        } else {
            None
        };

        let Some(serialized_schema) = serialized_schema else {
            if first_page_bytes.is_none()
                && stream_name.is_none()
                && response.total_rows.is_none()
                && response.statement_type != "SELECT"
            {
                return Ok((Arc::new(ArrowSchema::default()), empty_receiver()));
            }
            return Err(BigQueryError::Protocol(
                "QueryResponse is missing Arrow schema".into(),
            ));
        };

        match (first_page_bytes, stream_name) {
            (None, None) => {
                let metadata = crate::bigquery_read_stream::parse_arrow_schema(&serialized_schema)?;
                Ok((Arc::new(metadata.schema), empty_receiver()))
            },
            (None, Some(stream_name)) => {
                let (schema_ref, receiver) = self
                    .read_job_stream_or_table(
                        &response,
                        stream_name,
                        response.page_row_count,
                        Some(serialized_schema),
                    )
                    .await?;
                let schema_ref =
                    schema_ref.expect("schema_ref must be Some when serialized_schema is Some");
                Ok((schema_ref, receiver))
            },
            (Some(first_page_bytes), stream_name) => {
                let metadata = crate::bigquery_read_stream::parse_arrow_schema(&serialized_schema)?;
                let read_rows = stream_name.map(|name| {
                    self.client.prepare_read_rows(
                        name,
                        response.page_row_count,
                        Some(self.arrow_buffer_compression()),
                    )
                });
                let schema_ref = Arc::new(metadata.schema.clone());
                let receiver = self.client.spawn_query_stream(
                    first_page_bytes,
                    response.page_row_count,
                    read_rows,
                    Arc::new(metadata),
                );
                Ok((schema_ref, receiver))
            },
        }
    }

    async fn resolve_empty_job_stream(
        &self,
        response: &QueryResponse,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let (project_id, job_id, location) = extract_required_job_reference(response)?;
        let job = fetch_job(self.client, project_id, job_id, Some(location)).await?;
        check_job_status_error(job.status, project_id, Some(location), job_id)?;

        let query_stats = job.statistics.as_ref().and_then(|s| s.query.as_ref());
        let statement_type = query_stats
            .and_then(|q| non_empty_str(&q.statement_type))
            .or_else(|| non_empty_str(&response.statement_type));
        let is_ddl = query_stats
            .and_then(|q| non_empty_str(&q.ddl_operation_performed))
            .is_some();

        if is_ddl || statement_type.is_some_and(|st| st != "SELECT") {
            return Ok((Arc::new(ArrowSchema::default()), empty_receiver()));
        }

        let Some(destination_table) = extract_destination_table(job.configuration)? else {
            if statement_type == Some("SELECT") {
                return Err(BigQueryError::Protocol(
                    "Job response is missing configuration.query.destination_table".into(),
                ));
            }
            return Ok((Arc::new(ArrowSchema::default()), empty_receiver()));
        };

        let (serialized_schema, _streams) = self
            .client
            .create_table_read_session(
                &destination_table,
                crate::ReadOptions {
                    maintain_order: true,
                    arrow_buffer_compression: Some(self.arrow_buffer_compression()),
                    max_stream_count: Some(1),
                    ..Default::default()
                },
            )
            .await?;
        let metadata = crate::bigquery_read_stream::parse_arrow_schema(&serialized_schema)?;
        Ok((Arc::new(metadata.schema), empty_receiver()))
    }

    async fn resolve_job_error(
        &self,
        response: &QueryResponse,
        original_err: BigQueryError,
    ) -> BigQueryError {
        let (Some(project_id), Some(job_id), location) = extract_job_reference(response) else {
            return original_err;
        };

        if let Ok(job) = fetch_job(self.client, project_id, job_id, location).await {
            if let Err(job_err) = check_job_status_error(job.status, project_id, location, job_id) {
                return job_err;
            }
        }

        original_err
    }
}

fn empty_receiver() -> BigQueryRecordBatchReceiver {
    let (_tx, rx) = tokio::sync::mpsc::channel(1);
    BigQueryRecordBatchReceiver::new(rx, None, Vec::new())
}

fn non_empty_str(s: &str) -> Option<&str> {
    (!s.is_empty()).then_some(s)
}

async fn fetch_job(
    client: &crate::Client,
    project_id: &str,
    job_id: &str,
    location: Option<&str>,
) -> Result<google_cloud_bigquery_v2::model::Job, BigQueryError> {
    let mut get_job = client
        .bq_client
        .get_job()
        .set_project_id(project_id)
        .set_job_id(job_id);
    if let Some(location) = location {
        get_job = get_job.set_location(location);
    }
    Ok(client
        .apply_request_options(get_job)
        .with_idempotency(true)
        .send()
        .await?)
}

fn check_job_status_error(
    status: Option<google_cloud_bigquery_v2::model::JobStatus>,
    project_id: &str,
    location: Option<&str>,
    job_id: &str,
) -> Result<(), BigQueryError> {
    if let Some(status) = status {
        if let Some(error_result) = status.error_result {
            return Err(BigQueryError::JobFailed(Box::new(JobFailedError {
                project_id: Some(project_id.to_string()),
                location: location.map(str::to_string),
                job_id: Some(job_id.to_string()),
                error: JobError::from(error_result),
                errors: status.errors.into_iter().map(JobError::from).collect(),
            })));
        }
    }
    Ok(())
}

fn extract_destination_table(
    configuration: Option<google_cloud_bigquery_v2::model::JobConfiguration>,
) -> Result<Option<crate::BigQueryTableId>, BigQueryError> {
    let Some(destination_table) = configuration
        .and_then(|c| c.query)
        .and_then(|q| q.destination_table)
    else {
        return Ok(None);
    };

    if destination_table.project_id.is_empty()
        || destination_table.dataset_id.is_empty()
        || destination_table.table_id.is_empty()
    {
        return Err(BigQueryError::Protocol(
            "Job destination_table is missing project_id, dataset_id, or table_id".into(),
        ));
    }

    Ok(Some(crate::BigQueryTableId::new(
        destination_table.project_id,
        destination_table.dataset_id,
        destination_table.table_id,
    )))
}

pub(crate) async fn get_destination_table(
    client: &crate::Client,
    project_id: &str,
    job_id: &str,
    location: &str,
) -> Result<crate::BigQueryTableId, BigQueryError> {
    let job = fetch_job(client, project_id, job_id, Some(location)).await?;
    extract_destination_table(job.configuration)?.ok_or_else(|| {
        BigQueryError::Protocol(
            "Job response is missing configuration.query.destination_table".into(),
        )
    })
}

fn job_failed_from_gax_status(status: &google_cloud_gax::error::rpc::Status) -> JobFailedError {
    use google_cloud_gax::error::rpc::StatusDetails;

    let debug_info = status.details.iter().find_map(|d| match d {
        StatusDetails::DebugInfo(info) if !info.detail.is_empty() => Some(info.detail.clone()),
        _ => None,
    });

    let mut errors: Vec<JobError> = status
        .details
        .iter()
        .filter_map(|d| match d {
            StatusDetails::ErrorInfo(info) => {
                let message = info
                    .metadata
                    .get("message")
                    .cloned()
                    .unwrap_or_else(|| status.message.clone());
                let location = info
                    .metadata
                    .get("location")
                    .filter(|s| !s.is_empty())
                    .cloned();
                let item_debug_info = info
                    .metadata
                    .get("debugInfo")
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .or_else(|| debug_info.clone());
                Some(JobError {
                    reason: (!info.reason.is_empty()).then(|| info.reason.clone()),
                    location,
                    message,
                    debug_info: item_debug_info,
                })
            },
            _ => None,
        })
        .collect();

    let error = match errors.first() {
        Some(first) => JobError {
            reason: first.reason.clone(),
            location: first.location.clone(),
            message: if !status.message.is_empty() {
                status.message.clone()
            } else {
                first.message.clone()
            },
            debug_info: debug_info.or_else(|| first.debug_info.clone()),
        },
        None => JobError {
            reason: None,
            location: None,
            message: status.message.clone(),
            debug_info,
        },
    };

    if errors.is_empty() {
        errors.push(error.clone());
    }

    JobFailedError {
        project_id: None,
        location: None,
        job_id: None,
        error,
        errors,
    }
}

fn map_query_error(err: google_cloud_bigquery_v2::Error) -> BigQueryError {
    if let Some(status) = err.status() {
        return BigQueryError::JobFailed(Box::new(job_failed_from_gax_status(status)));
    }
    BigQueryError::from(err)
}

fn extract_job_reference(response: &QueryResponse) -> (Option<&str>, Option<&str>, Option<&str>) {
    let job_ref = response.job_reference.as_ref();
    let project_id = job_ref.and_then(|jr| non_empty_str(&jr.project_id));
    let job_id = job_ref.and_then(|jr| non_empty_str(&jr.job_id));
    let location = job_ref
        .and_then(|jr| jr.location.as_deref())
        .and_then(non_empty_str)
        .or_else(|| non_empty_str(&response.location));
    (project_id, job_id, location)
}

fn extract_required_job_reference(
    response: &QueryResponse,
) -> Result<(&str, &str, &str), BigQueryError> {
    let (project_id, job_id, location) = extract_job_reference(response);
    let project_id = project_id.ok_or_else(|| {
        BigQueryError::Protocol(
            "QueryResponse has remaining rows, but job_reference.project_id is missing".into(),
        )
    })?;
    let location = location.ok_or_else(|| {
        BigQueryError::Protocol(
            "QueryResponse has remaining rows, but job location is missing".into(),
        )
    })?;
    let job_id = job_id.ok_or_else(|| {
        BigQueryError::Protocol(
            "QueryResponse has remaining rows, but job_reference.job_id is missing".into(),
        )
    })?;
    Ok((project_id, job_id, location))
}

fn default_job_stream_path(response: &QueryResponse) -> Result<String, BigQueryError> {
    let (project_id, job_id, location) = extract_required_job_reference(response)?;
    Ok(format!(
        "projects/{project_id}/locations/{location}/jobs/{job_id}/streams/_default"
    ))
}

fn query_response_to_record_batch(response: &QueryResponse) -> Result<RecordBatch, BigQueryError> {
    // 1. bigquerySchema
    let bigquery_schema = response
        .schema
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|e| BigQueryError::Other(Box::new(e)))?;
    let bigquery_schema_arr = Utf8Array::<i32>::from([bigquery_schema.as_deref()]).boxed();

    // 2. jobReference fields: projectId, jobId, location
    let (project_id, job_id, location) = extract_job_reference(response);

    let project_id_arr = Utf8Array::<i32>::from([project_id]).boxed();
    let job_id_arr = Utf8Array::<i32>::from([job_id]).boxed();
    let location_arr = Utf8Array::<i32>::from([location]).boxed();

    // 3. query response statistics fields: totalBytesProcessed, cacheHit, statementType
    let total_bytes_processed = response.total_bytes_processed.unwrap_or(0);
    let total_bytes_processed_arr = Int64Array::from_slice([total_bytes_processed]).boxed();

    let cache_hit_arr = BooleanArray::from([response.cache_hit]).boxed();

    let statement_type = non_empty_str(&response.statement_type);
    let statement_type_arr = Utf8Array::<i32>::from([statement_type]).boxed();

    // 4. creationTime (ms, UTC)
    let creation_time_dtype = ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()));
    let creation_time = response.creation_time.filter(|&t| t > 0);
    let creation_time_arr = Int64Array::from([creation_time])
        .to(creation_time_dtype.clone())
        .boxed();

    let arrow_schema = ArrowSchema::from_iter([
        Field::new("bigquerySchema".into(), ArrowDataType::Utf8, true),
        Field::new("projectId".into(), ArrowDataType::Utf8, true),
        Field::new("jobId".into(), ArrowDataType::Utf8, true),
        Field::new("location".into(), ArrowDataType::Utf8, true),
        Field::new("totalBytesProcessed".into(), ArrowDataType::Int64, false),
        Field::new("cacheHit".into(), ArrowDataType::Boolean, true),
        Field::new("statementType".into(), ArrowDataType::Utf8, true),
        Field::new("creationTime".into(), creation_time_dtype, true),
    ]);

    let columns: Vec<Box<dyn Array>> = vec![
        bigquery_schema_arr,
        project_id_arr,
        job_id_arr,
        location_arr,
        total_bytes_processed_arr,
        cache_hit_arr,
        statement_type_arr,
        creation_time_arr,
    ];

    Ok(RecordBatch::try_new(1, Arc::new(arrow_schema), columns)?)
}

#[cfg(test)]
mod tests {
    use google_cloud_bigquery_v2::model::{JobReference, TableFieldSchema, TableSchema};

    use super::*;
    use crate::bigquery_read_stream::tests::{
        create_test_arrow_payload_with_range, CannedReadRowsStub,
    };
    use crate::tests::{test_bq_client, test_read_client};

    fn utf8_val(batch: &RecordBatch, col_idx: usize) -> Option<&str> {
        let arr = batch.arrays()[col_idx]
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .expect("expected Utf8Array<i32>");
        arr.iter().next().unwrap()
    }

    fn bool_val(batch: &RecordBatch, col_idx: usize) -> Option<bool> {
        let arr = batch.arrays()[col_idx]
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("expected BooleanArray");
        arr.iter().next().unwrap()
    }

    fn int64_val(batch: &RecordBatch, col_idx: usize) -> Option<i64> {
        let arr = batch.arrays()[col_idx]
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("expected Int64Array");
        arr.iter().next().unwrap().copied()
    }

    #[test]
    fn test_query_response_to_record_batch_all_fields() {
        let total_bytes_processed = 15i64;
        let schema = TableSchema::new().set_fields([TableFieldSchema::new()
            .set_name("col1")
            .set_type("INTEGER")
            .set_mode("NULLABLE")]);
        let expected_schema_json = serde_json::to_string(&schema).unwrap();

        let response = QueryResponse::new()
            .set_schema(schema)
            .set_job_reference(
                JobReference::new()
                    .set_project_id("project-id")
                    .set_job_id("job-id")
                    .set_location("US"),
            )
            .set_total_bytes_processed(total_bytes_processed)
            .set_cache_hit(true)
            .set_statement_type("SELECT")
            .set_creation_time(1_767_037_135_155i64);

        let batch = query_response_to_record_batch(&response).unwrap();
        assert_eq!(batch.len(), 1);

        let col_names: Vec<&str> = batch.schema().iter_names().map(|s| s.as_str()).collect();
        assert_eq!(
            col_names,
            vec![
                "bigquerySchema",
                "projectId",
                "jobId",
                "location",
                "totalBytesProcessed",
                "cacheHit",
                "statementType",
                "creationTime",
            ]
        );

        assert_eq!(utf8_val(&batch, 0), Some(expected_schema_json.as_str()));
        assert_eq!(utf8_val(&batch, 1), Some("project-id"));
        assert_eq!(utf8_val(&batch, 2), Some("job-id"));
        assert_eq!(utf8_val(&batch, 3), Some("US"));
        assert_eq!(int64_val(&batch, 4), Some(15));
        assert_eq!(bool_val(&batch, 5), Some(true));
        assert_eq!(utf8_val(&batch, 6), Some("SELECT"));
        assert_eq!(int64_val(&batch, 7), Some(1_767_037_135_155));
        assert_eq!(
            batch.arrays()[7].dtype(),
            &ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()))
        );
    }

    #[test]
    fn test_query_response_to_record_batch_missing_bytes_use_zero() {
        let response = QueryResponse::new()
            .set_job_reference(
                JobReference::new()
                    .set_project_id("project-id")
                    .set_job_id("job-id")
                    .set_location("US"),
            )
            .set_cache_hit(true)
            .set_statement_type("SELECT");

        let batch = query_response_to_record_batch(&response).unwrap();
        assert_eq!(batch.len(), 1);

        assert_eq!(utf8_val(&batch, 0), None);
        assert_eq!(utf8_val(&batch, 1), Some("project-id"));
        assert_eq!(utf8_val(&batch, 2), Some("job-id"));
        assert_eq!(utf8_val(&batch, 3), Some("US"));
        assert_eq!(int64_val(&batch, 4), Some(0));
        assert_eq!(bool_val(&batch, 5), Some(true));
        assert_eq!(utf8_val(&batch, 6), Some("SELECT"));
        assert_eq!(int64_val(&batch, 7), None);
    }

    #[tokio::test]
    async fn test_dry_run_with_mock_server() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);
            assert!(
                req_str.contains("/projects/test-quota-proj/queries"),
                "unexpected request path: {req_str}"
            );
            assert!(
                req_str.contains("\"dryRun\":true"),
                "expected dryRun: true in request body: {req_str}"
            );

            let response_query = QueryResponse::new()
                .set_job_reference(
                    JobReference::new()
                        .set_project_id("test-quota-proj")
                        .set_location("US"),
                )
                .set_creation_time(1_767_037_135_155i64)
                .set_total_bytes_processed(0i64)
                .set_cache_hit(false)
                .set_statement_type("SELECT")
                .set_schema(
                    TableSchema::new()
                        .set_fields([TableFieldSchema::new().set_name("one").set_type("INT64")]),
                );
            let body = serde_json::to_string(&response_query).unwrap();
            let http_resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(http_resp.as_bytes()).await.unwrap();
        });

        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let bq_client = crate::BigQueryJobService::builder()
            .with_credentials(cred.clone())
            .with_endpoint(format!("http://{addr}"))
            .build()
            .await
            .unwrap();
        let read_client = crate::BigQueryReadClient::builder()
            .with_credentials(cred)
            .build()
            .await
            .unwrap();
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let batch = client.query("SELECT 1 AS one").dry_run().await.unwrap();
        server_task.await.unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(
            utf8_val(&batch, 0),
            Some(r#"{"fields":[{"name":"one","type":"INT64"}]}"#)
        );
        assert_eq!(utf8_val(&batch, 1), Some("test-quota-proj"));
        assert_eq!(utf8_val(&batch, 2), None);
        assert_eq!(utf8_val(&batch, 3), Some("US"));
        assert_eq!(int64_val(&batch, 4), Some(0));
        assert_eq!(bool_val(&batch, 5), Some(false));
        assert_eq!(utf8_val(&batch, 6), Some("SELECT"));
        assert_eq!(int64_val(&batch, 7), Some(1_767_037_135_155));
    }

    #[tokio::test]
    async fn test_read_with_mock_server() {
        use google_cloud_bigquery_v2::model::{
            ArrowRecordBatch as BqArrowRecordBatch, ArrowSchema as BqArrowSchema,
        };
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let (schema_bytes, batch_bytes) = create_test_arrow_payload_with_range(0, 3);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);
            assert!(
                req_str.contains("/projects/test-quota-proj/queries"),
                "unexpected request path: {req_str}"
            );
            assert!(
                req_str.contains("\"jobCreationMode\":2")
                    || req_str.contains("\"jobCreationMode\":\"JOB_CREATION_OPTIONAL\""),
                "expected jobCreationMode set to JobCreationOptional in request body: {req_str}"
            );
            assert!(
                req_str.contains("\"queryResultsFormat\":2")
                    || req_str.contains("\"queryResultsFormat\":\"ARROW\""),
                "expected queryResultsFormat set to Arrow in request body: {req_str}"
            );
            assert!(
                req_str.contains("\"bufferCompression\":1")
                    || req_str.contains("\"bufferCompression\":\"LZ4_FRAME\""),
                "expected default Lz4Frame bufferCompression in request body: {req_str}"
            );

            let response_query = QueryResponse::new()
                .set_job_reference(
                    JobReference::new()
                        .set_project_id("test-quota-proj")
                        .set_location("US"),
                )
                .set_arrow_schema(BqArrowSchema::new().set_serialized_schema(schema_bytes))
                .set_arrow_record_batch(
                    BqArrowRecordBatch::new().set_serialized_record_batch(batch_bytes),
                )
                .set_page_row_count(3);
            let body = serde_json::to_string(&response_query).unwrap();
            let http_resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(http_resp.as_bytes()).await.unwrap();
        });

        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let bq_client = crate::BigQueryJobService::builder()
            .with_credentials(cred)
            .with_endpoint(format!("http://{addr}"))
            .build()
            .await
            .unwrap();
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let (schema, mut receiver) = client.query("SELECT col1 FROM tbl").read().await.unwrap();
        server_task.await.unwrap();

        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);

        let batch = receiver.recv().await.unwrap().unwrap();
        assert_eq!(batch.len(), 3);
        let arr = batch.arrays()[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.values().as_slice(), &[0, 1, 2]);
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_read_empty_results() {
        use google_cloud_bigquery_v2::model::{ArrowSchema as BqArrowSchema, PostQueryRequest};
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct EmptyQueryStub {
            schema_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for EmptyQueryStub {
            async fn query(
                &self,
                req: PostQueryRequest,
                options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                assert_eq!(req.project_id, "test-quota-proj");
                assert_eq!(options.quota_project().as_deref(), Some("test-quota-proj"));
                let q = req.query_request.expect("query_request should be set");
                assert_eq!(q.query, "SELECT col1 FROM tbl WHERE FALSE");
                assert_eq!(q.use_legacy_sql, Some(false));
                assert_eq!(q.job_creation_mode, JobCreationMode::JobCreationOptional);
                assert_eq!(q.query_results_format, QueryResultsFormat::Arrow);
                let arrow_opts = q
                    .arrow_serialization_options()
                    .expect("arrow_serialization_options should be set");
                assert_eq!(
                    arrow_opts.buffer_compression,
                    arrow_serialization_options::CompressionCodec::Lz4Frame
                );

                let resp = QueryResponse::new().set_arrow_schema(
                    BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                );
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, _) = create_test_arrow_payload_with_range(0, 0);
        let bq_client = crate::BigQueryJobService::from_stub(EmptyQueryStub { schema_bytes });
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let (schema, mut receiver) = client
            .query("SELECT col1 FROM tbl WHERE FALSE")
            .read()
            .await
            .unwrap();
        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_read_http_error_with_mock_server() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = socket.read(&mut buf).await.unwrap();

            let body = r#"{"error":{"code":400,"message":"Syntax error: Unexpected end of script at [1:7]","errors":[{"message":"Syntax error: Unexpected end of script at [1:7]","domain":"global","reason":"invalidQuery","location":"q","locationType":"parameter"}],"status":"INVALID_ARGUMENT"}}"#;
            let http_resp = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(http_resp.as_bytes()).await.unwrap();
        });

        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let bq_client = crate::BigQueryJobService::builder()
            .with_credentials(cred)
            .with_endpoint(format!("http://{addr}"))
            .build()
            .await
            .unwrap();
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let err = match client.query("SELECT").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected JobFailed error for HTTP 400"),
        };
        server_task.await.unwrap();
        match &err {
            BigQueryError::JobFailed(job_failed) => {
                assert_eq!(job_failed.project_id, None);
                assert_eq!(job_failed.location, None);
                assert_eq!(job_failed.job_id, None);
                assert_eq!(job_failed.error.reason.as_deref(), Some("invalidQuery"));
                assert_eq!(job_failed.error.location.as_deref(), Some("q"));
                assert_eq!(
                    job_failed.error.message,
                    "Syntax error: Unexpected end of script at [1:7]"
                );
            },
            other => panic!("expected BigQueryError::JobFailed, got {other:?}"),
        }
        assert_eq!(
            err.to_string(),
            "BigQuery job failed (invalidQuery at q): Syntax error: Unexpected end of script at [1:7]"
        );
    }

    #[tokio::test]
    async fn test_read_with_warnings_in_errors_succeeds() {
        use google_cloud_bigquery_v2::model::{
            ArrowRecordBatch as BqArrowRecordBatch, ArrowSchema as BqArrowSchema, ErrorProto,
            PostQueryRequest,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct WarningQueryStub {
            schema_bytes: Vec<u8>,
            batch_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for WarningQueryStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_errors([ErrorProto::new().set_message("non-fatal warning")])
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_arrow_record_batch(
                        BqArrowRecordBatch::new()
                            .set_serialized_record_batch(self.batch_bytes.clone()),
                    )
                    .set_page_row_count(2);
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, batch_bytes) = create_test_arrow_payload_with_range(0, 2);
        let bq_client = crate::BigQueryJobService::from_stub(WarningQueryStub {
            schema_bytes,
            batch_bytes,
        });
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let (schema, mut receiver) = client.query("SELECT col1 FROM tbl").read().await.unwrap();
        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);
        let batch = receiver.recv().await.unwrap().unwrap();
        assert_eq!(batch.len(), 2);
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_read_missing_arrow_schema_error() {
        use google_cloud_bigquery_v2::model::PostQueryRequest;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct MissingSchemaStub;

        impl google_cloud_bigquery_v2::stub::JobService for MissingSchemaStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                Ok(Response::from(
                    QueryResponse::new().set_statement_type("SELECT"),
                ))
            }
        }

        let bq_client = crate::BigQueryJobService::from_stub(MissingSchemaStub);
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let err = match client.query("SELECT 1").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected Protocol error"),
        };
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[tokio::test]
    async fn test_read_corrupt_arrow_record_batch_error() {
        use google_cloud_bigquery_v2::model::{
            ArrowRecordBatch as BqArrowRecordBatch, ArrowSchema as BqArrowSchema, PostQueryRequest,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct CorruptBatchStub {
            schema_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for CorruptBatchStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_arrow_record_batch(
                        BqArrowRecordBatch::new()
                            .set_serialized_record_batch(vec![0x00, 0x01, 0x02, 0x03]),
                    );
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, _) = create_test_arrow_payload_with_range(0, 0);
        let bq_client = crate::BigQueryJobService::from_stub(CorruptBatchStub { schema_bytes });
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let (_, mut receiver) = client.query("SELECT 1").read().await.unwrap();
        let err = receiver.recv().await.unwrap().unwrap_err();
        assert!(matches!(err, BigQueryError::Arrow(_)));
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_query_compression_codecs() {
        use google_cloud_bigquery::model::arrow_serialization_options::CompressionCodec as V1Codec;
        use google_cloud_bigquery_v2::model::arrow_serialization_options::CompressionCodec as V2Codec;

        let client = crate::Client::new_for_testing(
            test_bq_client().await,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let default_req = client.query("SELECT 1").build_read_request();
        assert_eq!(
            default_req.job_creation_mode,
            JobCreationMode::JobCreationOptional
        );
        assert_eq!(
            default_req
                .arrow_serialization_options()
                .unwrap()
                .buffer_compression,
            V2Codec::Lz4Frame
        );

        let v1_codecs = [
            (None, V2Codec::Lz4Frame),
            (Some(V1Codec::Lz4Frame), V2Codec::Lz4Frame),
            (Some(V1Codec::Zstd), V2Codec::Zstd),
            (
                Some(V1Codec::CompressionUnspecified),
                V2Codec::CompressionUnspecified,
            ),
        ];

        for (codec, expected) in v1_codecs {
            let req = client
                .query("SELECT 1")
                .with_arrow_buffer_compression(codec)
                .build_read_request();
            assert_eq!(
                req.arrow_serialization_options()
                    .unwrap()
                    .buffer_compression,
                expected
            );
        }

        let direct_v1_req = client
            .query("SELECT 1")
            .with_arrow_buffer_compression(V1Codec::Zstd)
            .build_read_request();
        assert_eq!(
            direct_v1_req
                .arrow_serialization_options()
                .unwrap()
                .buffer_compression,
            V2Codec::Zstd
        );
        assert_eq!(default_req.use_query_cache, None);

        let no_cache_req = client
            .query("SELECT 1")
            .with_use_query_cache(false)
            .build_read_request();
        assert_eq!(no_cache_req.use_query_cache, Some(false));

        let cache_req = client
            .query("SELECT 1")
            .with_use_query_cache(Some(true))
            .build_read_request();
        assert_eq!(cache_req.use_query_cache, Some(true));
    }

    #[tokio::test]
    async fn test_read_opens_stream_for_remaining_rows() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowRecordBatch as StorageArrowRecordBatch, ReadRowsResponse,
        };
        use google_cloud_bigquery_v2::model::{
            ArrowRecordBatch as BqArrowRecordBatch, ArrowSchema as BqArrowSchema, PostQueryRequest,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct MultiPageQueryStub {
            schema_bytes: Vec<u8>,
            first_page_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for MultiPageQueryStub {
            async fn query(
                &self,
                req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let q = req.query_request.expect("query_request should be set");
                let arrow_opts = q
                    .arrow_serialization_options()
                    .expect("arrow_serialization_options should be set");
                assert_eq!(
                    arrow_opts.buffer_compression,
                    arrow_serialization_options::CompressionCodec::Zstd
                );

                let resp = QueryResponse::new()
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("EU")
                            .set_job_id("job-123"),
                    )
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_arrow_record_batch(
                        BqArrowRecordBatch::new()
                            .set_serialized_record_batch(self.first_page_bytes.clone()),
                    )
                    .set_page_row_count(3)
                    .set_total_rows(8_u64);
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, first_page_bytes) = create_test_arrow_payload_with_range(0, 3);
        let (_, stream_batch1_bytes) = create_test_arrow_payload_with_range(3, 2);
        let (_, stream_batch2_bytes) = create_test_arrow_payload_with_range(5, 3);

        let resp1 = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(stream_batch1_bytes),
            )
            .set_row_count(2);
        let resp2 = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(stream_batch2_bytes),
            )
            .set_row_count(3);

        let bq_client = crate::BigQueryJobService::from_stub(MultiPageQueryStub {
            schema_bytes,
            first_page_bytes,
        });
        let (read_stub, recorded) = CannedReadRowsStub::new(vec![resp1, resp2]);
        let read_client = Read::from_stub(read_stub);
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let (schema, mut receiver) = client
            .query("SELECT col1 FROM tbl")
            .with_arrow_buffer_compression(crate::CompressionCodec::Zstd)
            .read()
            .await
            .unwrap();

        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);

        let mut all_values = Vec::new();
        while let Some(batch) = receiver.recv().await {
            let batch = batch.unwrap();
            let arr = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(arr.values().as_slice());
        }
        assert_eq!(all_values, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        let rec = recorded.lock().unwrap();
        assert_eq!(
            rec.stream_name.as_deref(),
            Some("projects/job-proj/locations/EU/jobs/job-123/streams/_default")
        );
        assert_eq!(rec.offset, Some(3));
        assert_eq!(
            rec.arrow_buffer_compression,
            Some(crate::CompressionCodec::Zstd)
        );
    }

    #[tokio::test]
    async fn test_read_does_not_open_stream_when_no_remaining_rows() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{ReadRowsRequest, ReadRowsResponse};
        use google_cloud_bigquery_v2::model::{
            ArrowRecordBatch as BqArrowRecordBatch, ArrowSchema as BqArrowSchema, PostQueryRequest,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug)]
        struct SinglePageQueryStub {
            schema_bytes: Vec<u8>,
            first_page_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for SinglePageQueryStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("job-123"),
                    )
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_arrow_record_batch(
                        BqArrowRecordBatch::new()
                            .set_serialized_record_batch(self.first_page_bytes.clone()),
                    )
                    .set_page_row_count(3)
                    .set_total_rows(3_u64);
                Ok(Response::from(resp))
            }
        }

        #[derive(Debug)]
        struct PanicOnReadRowsStub;

        impl google_cloud_bigquery::stub::Read for PanicOnReadRowsStub {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                panic!("read_rows should not be called when total_rows == page_row_count");
            }
        }

        let (schema_bytes, first_page_bytes) = create_test_arrow_payload_with_range(0, 3);
        let bq_client = crate::BigQueryJobService::from_stub(SinglePageQueryStub {
            schema_bytes,
            first_page_bytes,
        });
        let client = crate::Client::new_for_testing(
            bq_client,
            Read::from_stub(PanicOnReadRowsStub),
            "test-quota-proj".to_string(),
        );

        let (_, mut receiver) = client.query("SELECT col1 FROM tbl").read().await.unwrap();
        let batch = receiver.recv().await.unwrap().unwrap();
        assert_eq!(batch.len(), 3);
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_read_remaining_rows_missing_job_reference_error() {
        use google_cloud_bigquery_v2::model::{ArrowSchema as BqArrowSchema, PostQueryRequest};
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct MissingJobRefStub {
            schema_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for MissingJobRefStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_page_row_count(0)
                    .set_total_rows(5_u64);
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, _) = create_test_arrow_payload_with_range(0, 0);
        let bq_client = crate::BigQueryJobService::from_stub(MissingJobRefStub { schema_bytes });
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let err = match client.query("SELECT col1 FROM tbl").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected Protocol error when job_reference is missing"),
        };
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[tokio::test]
    async fn test_read_job_incomplete_streams_from_read_rows() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowRecordBatch as StorageArrowRecordBatch, ArrowSchema as StorageArrowSchema,
            ReadRowsResponse,
        };
        use google_cloud_bigquery_v2::model::PostQueryRequest;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct IncompleteJobQueryStub;

        impl google_cloud_bigquery_v2::stub::JobService for IncompleteJobQueryStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_complete(false)
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("running-job-456"),
                    );
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, batch1_bytes) = create_test_arrow_payload_with_range(0, 3);
        let (_, batch2_bytes) = create_test_arrow_payload_with_range(3, 2);

        // Simulate a server heartbeat (empty ReadRowsResponse) while query is still executing,
        // followed by the first response with schema + rows, and then a second response.
        let heartbeat = ReadRowsResponse::new();
        let resp1 = ReadRowsResponse::new()
            .set_arrow_schema(StorageArrowSchema::new().set_serialized_schema(schema_bytes))
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(batch1_bytes),
            )
            .set_row_count(3);
        let resp2 = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(batch2_bytes),
            )
            .set_row_count(2);

        let bq_client = crate::BigQueryJobService::from_stub(IncompleteJobQueryStub);
        let (read_stub, recorded) = CannedReadRowsStub::new(vec![heartbeat, resp1, resp2]);
        let read_client = Read::from_stub(read_stub);
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let (schema, mut receiver) = client.query("SELECT col1 FROM tbl").read().await.unwrap();
        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);

        let mut all_values = Vec::new();
        while let Some(batch) = receiver.recv().await {
            let batch = batch.unwrap();
            let arr = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(arr.values().as_slice());
        }
        assert_eq!(all_values, vec![0, 1, 2, 3, 4]);

        let rec = recorded.lock().unwrap();
        assert_eq!(
            rec.stream_name.as_deref(),
            Some("projects/job-proj/locations/US/jobs/running-job-456/streams/_default")
        );
        assert_eq!(rec.offset, Some(0));
        assert_eq!(
            rec.arrow_buffer_compression,
            Some(crate::CompressionCodec::Lz4Frame)
        );
    }

    #[tokio::test]
    async fn test_read_empty_first_page_with_remaining_rows_streams_from_read_rows() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowRecordBatch as StorageArrowRecordBatch, ReadRowsResponse,
        };
        use google_cloud_bigquery_v2::model::{ArrowSchema as BqArrowSchema, PostQueryRequest};
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct EmptyFirstPageQueryStub {
            schema_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for EmptyFirstPageQueryStub {
            async fn query(
                &self,
                req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let q = req.query_request.expect("query_request should be set");
                let arrow_opts = q
                    .arrow_serialization_options()
                    .expect("arrow_serialization_options should be set");
                assert_eq!(
                    arrow_opts.buffer_compression,
                    arrow_serialization_options::CompressionCodec::CompressionUnspecified
                );

                let resp = QueryResponse::new()
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("job-empty-first-page"),
                    )
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_page_row_count(0)
                    .set_total_rows(3_u64);
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, stream_batch_bytes) = create_test_arrow_payload_with_range(0, 3);
        let resp = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(stream_batch_bytes),
            )
            .set_row_count(3);

        let bq_client =
            crate::BigQueryJobService::from_stub(EmptyFirstPageQueryStub { schema_bytes });
        let (read_stub, recorded) = CannedReadRowsStub::new(vec![resp]);
        let read_client = Read::from_stub(read_stub);
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let (_, mut receiver) = client
            .query("SELECT col1 FROM tbl")
            .with_arrow_buffer_compression(crate::CompressionCodec::CompressionUnspecified)
            .read()
            .await
            .unwrap();

        let batch = receiver.recv().await.unwrap().unwrap();
        assert_eq!(batch.len(), 3);
        assert!(receiver.recv().await.is_none());

        let rec = recorded.lock().unwrap();
        assert_eq!(
            rec.stream_name.as_deref(),
            Some("projects/job-proj/locations/US/jobs/job-empty-first-page/streams/_default")
        );
        assert_eq!(rec.offset, Some(0));
        assert_eq!(
            rec.arrow_buffer_compression,
            Some(crate::CompressionCodec::CompressionUnspecified)
        );
    }

    #[tokio::test]
    async fn test_read_first_page_row_count_mismatch_error() {
        use google_cloud_bigquery_v2::model::{
            ArrowRecordBatch as BqArrowRecordBatch, ArrowSchema as BqArrowSchema, PostQueryRequest,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct MismatchQueryStub {
            schema_bytes: Vec<u8>,
            batch_bytes: Option<Vec<u8>>,
            page_row_count: i64,
        }

        impl google_cloud_bigquery_v2::stub::JobService for MismatchQueryStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let mut resp = QueryResponse::new()
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_page_row_count(self.page_row_count);
                if let Some(batch_bytes) = &self.batch_bytes {
                    resp = resp.set_arrow_record_batch(
                        BqArrowRecordBatch::new().set_serialized_record_batch(batch_bytes.clone()),
                    );
                }
                Ok(Response::from(resp))
            }
        }

        let (schema_bytes, batch_bytes) = create_test_arrow_payload_with_range(0, 2);

        // Case 1: No results batch, but page_row_count > 0 -> immediate Protocol error
        let bq_client = crate::BigQueryJobService::from_stub(MismatchQueryStub {
            schema_bytes: schema_bytes.clone(),
            batch_bytes: None,
            page_row_count: 2,
        });
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );
        let err = match client.query("SELECT col1 FROM tbl").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected Protocol error when results is empty and page_row_count > 0"),
        };
        assert!(matches!(err, BigQueryError::Protocol(_)));

        // Case 2: 2 rows in batch, but page_row_count == 5 -> Protocol error sent on channel
        let bq_client = crate::BigQueryJobService::from_stub(MismatchQueryStub {
            schema_bytes,
            batch_bytes: Some(batch_bytes),
            page_row_count: 5,
        });
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );
        let (_, mut receiver) = client.query("SELECT col1 FROM tbl").read().await.unwrap();
        let err = receiver.recv().await.unwrap().unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
        assert!(receiver.recv().await.is_none());
    }

    #[test]
    fn test_estimate_remaining_bytes() {
        use crate::bigquery_read_stream::estimate_remaining_bytes;

        let page = bytes::Bytes::from(vec![0u8; 1024]);
        assert_eq!(estimate_remaining_bytes(None, 10, Some(100)), 0.0);
        assert_eq!(estimate_remaining_bytes(Some(&page), 0, Some(100)), 0.0);
        assert_eq!(estimate_remaining_bytes(Some(&page), 10, None), 0.0);
        assert_eq!(estimate_remaining_bytes(Some(&page), 10, Some(10)), 0.0);
        assert_eq!(estimate_remaining_bytes(Some(&page), 10, Some(5)), 0.0);
        // 1024 bytes for 10 rows, 90 remaining rows => 9 * 1024 = 9216 bytes
        assert_eq!(estimate_remaining_bytes(Some(&page), 10, Some(100)), 9216.0);
    }

    #[tokio::test]
    async fn test_read_maintain_order_false_switches_to_read_table_above_threshold() {
        use std::sync::Mutex;

        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            read_session, ArrowRecordBatch as StorageArrowRecordBatch,
            ArrowSchema as StorageArrowSchema, CreateReadSessionRequest, ReadRowsRequest,
            ReadRowsResponse, ReadSession, ReadStream,
        };
        use google_cloud_bigquery_v2::model::{
            ArrowRecordBatch as BqArrowRecordBatch, ArrowSchema as BqArrowSchema, GetJobRequest,
            Job, JobConfiguration, JobConfigurationQuery, PostQueryRequest, TableReference,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug, Default)]
        struct RecordedCalls {
            get_job_calls: usize,
            get_job_project_id: Option<String>,
            get_job_id: Option<String>,
            get_job_location: Option<String>,
            create_session_table: Option<String>,
            create_session_max_streams: Option<i32>,
            create_session_compression: Option<crate::CompressionCodec>,
            read_rows_streams: Vec<String>,
        }

        #[derive(Debug)]
        struct LargeQueryJobStub {
            schema_bytes: Vec<u8>,
            first_page_bytes: Vec<u8>,
            recorded: Arc<Mutex<RecordedCalls>>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for LargeQueryJobStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                // 2 rows in first_page_bytes (~144 bytes). Set total_rows high enough
                // so estimated remaining bytes > 250 MB.
                let resp = QueryResponse::new()
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("large-job-1"),
                    )
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_arrow_record_batch(
                        BqArrowRecordBatch::new()
                            .set_serialized_record_batch(self.first_page_bytes.clone()),
                    )
                    .set_page_row_count(2)
                    .set_total_rows(10_000_000_u64);
                Ok(Response::from(resp))
            }

            async fn get_job(
                &self,
                req: GetJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<Job>> {
                let mut rec = self.recorded.lock().unwrap();
                rec.get_job_calls += 1;
                rec.get_job_project_id = Some(req.project_id);
                rec.get_job_id = Some(req.job_id);
                rec.get_job_location = Some(req.location);

                let job = Job::new().set_configuration(
                    JobConfiguration::new().set_query(
                        JobConfigurationQuery::new().set_destination_table(
                            TableReference::new()
                                .set_project_id("dest-proj")
                                .set_dataset_id("_temp_dataset")
                                .set_table_id("anon_table_123"),
                        ),
                    ),
                );
                Ok(Response::from(job))
            }
        }

        #[derive(Debug)]
        struct MultiStreamReadStub {
            schema_bytes: Vec<u8>,
            stream1_bytes: Vec<u8>,
            stream2_bytes: Vec<u8>,
            recorded: Arc<Mutex<RecordedCalls>>,
        }

        impl google_cloud_bigquery::stub::Read for MultiStreamReadStub {
            async fn create_read_session(
                &self,
                req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let mut rec = self.recorded.lock().unwrap();
                rec.create_session_max_streams = Some(req.max_stream_count);
                if let Some(rs) = req.read_session {
                    rec.create_session_table = Some(rs.table);
                    if let Some(ro) = rs.read_options {
                        if let Some(
                            read_session::table_read_options::OutputFormatSerializationOptions::ArrowSerializationOptions(
                                arrow_opts,
                            ),
                        ) = ro.output_format_serialization_options
                        {
                            rec.create_session_compression = Some(arrow_opts.buffer_compression);
                        }
                    }
                }

                let session = ReadSession::new()
                    .set_arrow_schema(
                        StorageArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_streams([
                        ReadStream::new().set_name("streams/s1"),
                        ReadStream::new().set_name("streams/s2"),
                    ]);
                Ok(Response::from(session))
            }

            async fn read_rows(
                &self,
                req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                self.recorded
                    .lock()
                    .unwrap()
                    .read_rows_streams
                    .push(req.read_stream.clone());
                let batch_bytes = match req.read_stream.as_str() {
                    "streams/s1" => self.stream1_bytes.clone(),
                    "streams/s2" => self.stream2_bytes.clone(),
                    other => panic!("unexpected stream: {other}"),
                };
                let resp = ReadRowsResponse::new()
                    .set_arrow_record_batch(
                        StorageArrowRecordBatch::new().set_serialized_record_batch(batch_bytes),
                    )
                    .set_row_count(2);
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                tx.try_send(Ok(resp)).unwrap();
                drop(tx);
                Ok(ResponseStream::from(rx))
            }
        }

        // First page has values [100, 101] which must be discarded when switching to read_table.
        let (schema_bytes, first_page_bytes) = create_test_arrow_payload_with_range(100, 2);
        let (_, stream1_bytes) = create_test_arrow_payload_with_range(0, 2);
        let (_, stream2_bytes) = create_test_arrow_payload_with_range(2, 2);
        let recorded = Arc::new(Mutex::new(RecordedCalls::default()));

        let bq_client = crate::BigQueryJobService::from_stub(LargeQueryJobStub {
            schema_bytes: schema_bytes.clone(),
            first_page_bytes,
            recorded: Arc::clone(&recorded),
        });
        let read_client = Read::from_stub(MultiStreamReadStub {
            schema_bytes,
            stream1_bytes,
            stream2_bytes,
            recorded: Arc::clone(&recorded),
        });
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let (_, mut receiver) = client
            .query("SELECT col1 FROM big_tbl")
            .with_maintain_order(false)
            .with_arrow_buffer_compression(crate::CompressionCodec::Zstd)
            .read()
            .await
            .unwrap();

        let mut all_values = Vec::new();
        while let Some(batch) = receiver.recv().await {
            let batch = batch.unwrap();
            let arr = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(arr.values().as_slice());
        }
        all_values.sort_unstable();
        // First page [100, 101] was discarded; only [0, 1, 2, 3] from destination table streams are returned.
        assert_eq!(all_values, vec![0, 1, 2, 3]);

        let rec = recorded.lock().unwrap();
        assert_eq!(rec.get_job_calls, 1);
        assert_eq!(rec.get_job_project_id.as_deref(), Some("job-proj"));
        assert_eq!(rec.get_job_id.as_deref(), Some("large-job-1"));
        assert_eq!(rec.get_job_location.as_deref(), Some("US"));
        assert_eq!(
            rec.create_session_table.as_deref(),
            Some("projects/dest-proj/datasets/_temp_dataset/tables/anon_table_123")
        );
        assert_eq!(
            rec.create_session_compression,
            Some(crate::CompressionCodec::Zstd)
        );
        let mut streams = rec.read_rows_streams.clone();
        streams.sort();
        assert_eq!(streams, vec!["streams/s1", "streams/s2"]);
    }

    #[tokio::test]
    async fn test_read_maintain_order_false_empty_first_page_switches_to_read_table_above_threshold(
    ) {
        use std::sync::Mutex;

        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            read_session, ArrowRecordBatch as StorageArrowRecordBatch,
            ArrowSchema as StorageArrowSchema, CreateReadSessionRequest, ReadRowsRequest,
            ReadRowsResponse, ReadSession, ReadStream,
        };
        use google_cloud_bigquery_v2::model::{
            ArrowSchema as BqArrowSchema, GetJobRequest, Job, JobConfiguration,
            JobConfigurationQuery, PostQueryRequest, TableReference,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug, Default)]
        struct RecordedCalls {
            get_job_calls: usize,
            create_session_table: Option<String>,
            create_session_compression: Option<crate::CompressionCodec>,
            read_rows_streams: Vec<String>,
            job_stream_compression: Option<crate::CompressionCodec>,
        }

        #[derive(Debug)]
        struct EmptyFirstPageLargeJobStub {
            schema_bytes: Vec<u8>,
            recorded: Arc<Mutex<RecordedCalls>>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for EmptyFirstPageLargeJobStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                // Real BigQuery multi-page Arrow QueryResponse: page_row_count=0, total_rows=0, non-empty page_token.
                let resp = QueryResponse::new()
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("large-job-2"),
                    )
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_page_row_count(0)
                    .set_total_rows(0_u64)
                    .set_page_token("page-token-xyz");
                Ok(Response::from(resp))
            }

            async fn get_job(
                &self,
                _req: GetJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<Job>> {
                self.recorded.lock().unwrap().get_job_calls += 1;
                let job = Job::new().set_configuration(
                    JobConfiguration::new().set_query(
                        JobConfigurationQuery::new().set_destination_table(
                            TableReference::new()
                                .set_project_id("dest-proj")
                                .set_dataset_id("_temp_dataset")
                                .set_table_id("anon_table_456"),
                        ),
                    ),
                );
                Ok(Response::from(job))
            }
        }

        #[derive(Debug)]
        struct JobStreamThenTableReadStub {
            schema_bytes: Vec<u8>,
            job_first_page_bytes: Vec<u8>,
            table_stream_bytes: Vec<u8>,
            recorded: Arc<Mutex<RecordedCalls>>,
        }

        impl google_cloud_bigquery::stub::Read for JobStreamThenTableReadStub {
            async fn create_read_session(
                &self,
                req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let mut rec = self.recorded.lock().unwrap();
                if let Some(rs) = req.read_session {
                    rec.create_session_table = Some(rs.table);
                    if let Some(ro) = rs.read_options {
                        if let Some(
                            read_session::table_read_options::OutputFormatSerializationOptions::ArrowSerializationOptions(
                                arrow_opts,
                            ),
                        ) = ro.output_format_serialization_options
                        {
                            rec.create_session_compression = Some(arrow_opts.buffer_compression);
                        }
                    }
                }
                let session = ReadSession::new()
                    .set_arrow_schema(
                        StorageArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_streams([ReadStream::new().set_name("streams/table_s1")]);
                Ok(Response::from(session))
            }

            async fn read_rows(
                &self,
                req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let mut rec = self.recorded.lock().unwrap();
                rec.read_rows_streams.push(req.read_stream.clone());
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                if req.read_stream.ends_with("/streams/_default") {
                    rec.job_stream_compression = req
                        .arrow_serialization_options()
                        .map(|o| o.buffer_compression.clone());
                    let resp = ReadRowsResponse::new()
                        .set_arrow_record_batch(
                            StorageArrowRecordBatch::new()
                                .set_serialized_record_batch(self.job_first_page_bytes.clone()),
                        )
                        .set_row_count(2)
                        .set_total_estimated_row_count(10_000_000);
                    tx.try_send(Ok(resp)).unwrap();
                } else {
                    let resp = ReadRowsResponse::new()
                        .set_arrow_record_batch(
                            StorageArrowRecordBatch::new()
                                .set_serialized_record_batch(self.table_stream_bytes.clone()),
                        )
                        .set_row_count(3);
                    tx.try_send(Ok(resp)).unwrap();
                }
                drop(tx);
                Ok(ResponseStream::from(rx))
            }
        }

        let (schema_bytes, job_first_page_bytes) = create_test_arrow_payload_with_range(900, 2);
        let (_, table_stream_bytes) = create_test_arrow_payload_with_range(10, 3);
        let recorded = Arc::new(Mutex::new(RecordedCalls::default()));

        let bq_client = crate::BigQueryJobService::from_stub(EmptyFirstPageLargeJobStub {
            schema_bytes: schema_bytes.clone(),
            recorded: Arc::clone(&recorded),
        });
        let read_client = Read::from_stub(JobStreamThenTableReadStub {
            schema_bytes,
            job_first_page_bytes,
            table_stream_bytes,
            recorded: Arc::clone(&recorded),
        });
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let (_, mut receiver) = client
            .query("SELECT col1 FROM big_tbl")
            .with_maintain_order(false)
            .with_arrow_buffer_compression(crate::CompressionCodec::Zstd)
            .read()
            .await
            .unwrap();

        let mut all_values = Vec::new();
        while let Some(batch) = receiver.recv().await {
            let batch = batch.unwrap();
            let arr = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(arr.values().as_slice());
        }
        // Job stream first page [900, 901] was discarded; only [10, 11, 12] from destination table are returned.
        assert_eq!(all_values, vec![10, 11, 12]);

        let rec = recorded.lock().unwrap();
        assert_eq!(rec.get_job_calls, 1);
        assert_eq!(
            rec.job_stream_compression,
            Some(crate::CompressionCodec::Zstd)
        );
        assert_eq!(
            rec.create_session_table.as_deref(),
            Some("projects/dest-proj/datasets/_temp_dataset/tables/anon_table_456")
        );
        assert_eq!(
            rec.create_session_compression,
            Some(crate::CompressionCodec::Zstd)
        );
        assert_eq!(
            rec.read_rows_streams,
            vec![
                "projects/job-proj/locations/US/jobs/large-job-2/streams/_default",
                "streams/table_s1"
            ]
        );
    }

    #[tokio::test]
    async fn test_read_maintain_order_below_threshold_or_default_true_keeps_single_stream() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowRecordBatch as StorageArrowRecordBatch, ReadRowsResponse,
        };
        use google_cloud_bigquery_v2::model::{
            ArrowSchema as BqArrowSchema, GetJobRequest, Job, PostQueryRequest,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct PanicOnGetJobStub {
            schema_bytes: Vec<u8>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for PanicOnGetJobStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("job-single-stream"),
                    )
                    .set_arrow_schema(
                        BqArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_page_row_count(0)
                    .set_total_rows(0_u64)
                    .set_page_token("token-1");
                Ok(Response::from(resp))
            }

            async fn get_job(
                &self,
                _req: GetJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<Job>> {
                panic!(
                    "get_job should not be called when maintain_order is true or below threshold"
                );
            }
        }

        let (schema_bytes, batch1_bytes) = create_test_arrow_payload_with_range(0, 2);
        let (_, batch2_bytes) = create_test_arrow_payload_with_range(2, 3);

        // Case 1: maintain_order = false, but total_estimated_row_count is 5 (< 250 MB threshold).
        // Must keep batch1 and continue reading batch2 from the same _default stream.
        let resp1 = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(batch1_bytes.clone()),
            )
            .set_row_count(2)
            .set_total_estimated_row_count(5);
        let resp2 = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(batch2_bytes.clone()),
            )
            .set_row_count(3);

        let bq_client = crate::BigQueryJobService::from_stub(PanicOnGetJobStub {
            schema_bytes: schema_bytes.clone(),
        });
        let (read_stub, recorded) = CannedReadRowsStub::new(vec![resp1, resp2]);
        let client = crate::Client::new_for_testing(
            bq_client,
            Read::from_stub(read_stub),
            "test-quota-proj".to_string(),
        );

        let (_, mut receiver) = client
            .query("SELECT col1 FROM small_tbl")
            .with_maintain_order(false)
            .read()
            .await
            .unwrap();

        let mut all_values = Vec::new();
        while let Some(batch) = receiver.recv().await {
            let batch = batch.unwrap();
            let arr = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(arr.values().as_slice());
        }
        assert_eq!(all_values, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            recorded.lock().unwrap().arrow_buffer_compression,
            Some(crate::CompressionCodec::Lz4Frame)
        );

        // Case 2: default maintain_order = true, even with total_estimated_row_count above threshold.
        let large_resp1 = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(batch1_bytes),
            )
            .set_row_count(2)
            .set_total_estimated_row_count(10_000_000);
        let large_resp2 = ReadRowsResponse::new()
            .set_arrow_record_batch(
                StorageArrowRecordBatch::new().set_serialized_record_batch(batch2_bytes),
            )
            .set_row_count(3);

        let bq_client = crate::BigQueryJobService::from_stub(PanicOnGetJobStub { schema_bytes });
        let (read_stub, _) = CannedReadRowsStub::new(vec![large_resp1, large_resp2]);
        let client = crate::Client::new_for_testing(
            bq_client,
            Read::from_stub(read_stub),
            "test-quota-proj".to_string(),
        );

        let (_, mut receiver) = client
            .query("SELECT col1 FROM big_tbl")
            .read()
            .await
            .unwrap();
        let mut all_values = Vec::new();
        while let Some(batch) = receiver.recv().await {
            let batch = batch.unwrap();
            let arr = batch.arrays()[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            all_values.extend_from_slice(arr.values().as_slice());
        }
        assert_eq!(all_values, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn test_read_job_incomplete_failure_surfaces_job_failed_error() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{ReadRowsRequest, ReadRowsResponse};
        use google_cloud_bigquery_v2::model::{
            ErrorProto, GetJobRequest, Job, JobStatus, PostQueryRequest,
        };
        use google_cloud_gax::error::rpc::{Code, Status};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug)]
        struct FailedJobStub {
            get_job_result: std::sync::Mutex<Option<google_cloud_bigquery_v2::Result<Job>>>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for FailedJobStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_complete(false)
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("failed-job-789"),
                    );
                Ok(Response::from(resp))
            }

            async fn get_job(
                &self,
                req: GetJobRequest,
                options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<Job>> {
                assert_eq!(req.project_id, "job-proj");
                assert_eq!(req.job_id, "failed-job-789");
                assert_eq!(req.location, "US");
                assert_eq!(options.quota_project().as_deref(), Some("test-quota-proj"));
                assert_eq!(options.idempotent(), Some(true));

                let res = self
                    .get_job_result
                    .lock()
                    .unwrap()
                    .take()
                    .expect("get_job should be called once");
                res.map(Response::from)
            }
        }

        #[derive(Debug)]
        struct InvalidArgReadRowsStub;

        impl google_cloud_bigquery::stub::Read for InvalidArgReadRowsStub {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                let status = Status::default()
                    .set_code(Code::InvalidArgument)
                    .set_message("Job failed-job-789 failed with error: Resources exceeded");
                let _ = tx.send(Err(GaxError::service(status))).await;
                Ok(ResponseStream::from(rx))
            }
        }

        // Case 1: get_job succeeds with status.error_result -> returns BigQueryError::JobFailed(job_failed)
        let expected_error_result = ErrorProto::new()
            .set_reason("resourcesExceeded")
            .set_message("Resources exceeded during query execution: Out of memory.");
        let expected_status = JobStatus::new()
            .set_state("DONE")
            .set_error_result(expected_error_result.clone())
            .set_errors([expected_error_result]);
        let bq_client = crate::BigQueryJobService::from_stub(FailedJobStub {
            get_job_result: std::sync::Mutex::new(Some(Ok(Job::new().set_status(expected_status)))),
        });
        let read_client = Read::from_stub(InvalidArgReadRowsStub);
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let err = match client.query("SELECT * FROM huge_tbl").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected JobFailed error"),
        };
        let expected_job_error = JobError {
            reason: Some("resourcesExceeded".into()),
            location: None,
            message: "Resources exceeded during query execution: Out of memory.".into(),
            debug_info: None,
        };
        match &err {
            BigQueryError::JobFailed(job_failed) => {
                assert_eq!(
                    job_failed.as_ref(),
                    &JobFailedError {
                        project_id: Some("job-proj".into()),
                        location: Some("US".into()),
                        job_id: Some("failed-job-789".into()),
                        error: expected_job_error.clone(),
                        errors: vec![expected_job_error],
                    }
                );
            },
            other => panic!("expected BigQueryError::JobFailed, got {other:?}"),
        }
        assert_eq!(
            err.format_causal_chain(),
            "BigQuery job US:failed-job-789 failed (resourcesExceeded): Resources exceeded during query execution: Out of memory."
        );

        // Case 2: get_job fails -> returns original ReadRows Grpc error, not get_job error
        let get_job_err = GaxError::service(
            Status::default()
                .set_code(Code::PermissionDenied)
                .set_message("get_job permission denied"),
        );
        let bq_client = crate::BigQueryJobService::from_stub(FailedJobStub {
            get_job_result: std::sync::Mutex::new(Some(Err(get_job_err))),
        });
        let read_client = Read::from_stub(InvalidArgReadRowsStub);
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let err = match client.query("SELECT * FROM huge_tbl").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected original Grpc error when get_job fails"),
        };
        assert!(matches!(err, BigQueryError::Grpc(_)));
        assert!(err
            .to_string()
            .contains("Job failed-job-789 failed with error: Resources exceeded"));
        assert!(!err.to_string().contains("get_job permission denied"));

        // Case 3: get_job succeeds but job status has no error_result -> returns original ReadRows Grpc error
        let bq_client = crate::BigQueryJobService::from_stub(FailedJobStub {
            get_job_result: std::sync::Mutex::new(Some(Ok(
                Job::new().set_status(JobStatus::new().set_state("DONE"))
            ))),
        });
        let read_client = Read::from_stub(InvalidArgReadRowsStub);
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let err = match client.query("SELECT * FROM huge_tbl").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected original Grpc error when job has no error_result"),
        };
        assert!(matches!(err, BigQueryError::Grpc(_)));
    }

    #[test]
    fn test_map_query_error_debug_info_and_non_service() {
        use google_cloud_gax::error::rpc::{Code, Status, StatusDetails};
        use google_cloud_gax::error::Error as GaxError;

        // Service status with DebugInfo in details (no ErrorInfo) -> JobFailed
        let detail: StatusDetails = serde_json::from_value(serde_json::json!({
            "@type": "type.googleapis.com/google.rpc.DebugInfo",
            "detail": "RESOURCES_EXCEEDED: Out of memory"
        }))
        .expect("valid StatusDetails JSON");
        let status = Status::default()
            .set_code(Code::InvalidArgument)
            .set_message("Resources exceeded during query execution: Out of memory.")
            .set_details([detail]);
        let err = map_query_error(GaxError::service(status));
        match &err {
            BigQueryError::JobFailed(job_failed) => {
                assert_eq!(job_failed.project_id, None);
                assert_eq!(job_failed.location, None);
                assert_eq!(job_failed.job_id, None);
                assert_eq!(
                    job_failed.error.message,
                    "Resources exceeded during query execution: Out of memory."
                );
                assert_eq!(
                    job_failed.error.debug_info.as_deref(),
                    Some("RESOURCES_EXCEEDED: Out of memory")
                );
                assert_eq!(job_failed.errors.len(), 1);
            },
            other => panic!("expected BigQueryError::JobFailed, got {other:?}"),
        }
        assert_eq!(
            err.to_string(),
            "BigQuery job failed: Resources exceeded during query execution: Out of memory."
        );

        // Non-service transport error -> remains BigQueryError::Grpc
        let io_err = GaxError::io("connection reset");
        let mapped = map_query_error(io_err);
        assert!(matches!(mapped, BigQueryError::Grpc(_)));
    }

    #[tokio::test]
    async fn test_read_short_ddl_query_no_schema() {
        use google_cloud_bigquery_v2::model::PostQueryRequest;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct ShortDdlStub;

        impl google_cloud_bigquery_v2::stub::JobService for ShortDdlStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_complete(true)
                    .set_statement_type("CREATE_SCHEMA")
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("ddl-job-1"),
                    );
                Ok(Response::from(resp))
            }
        }

        let bq_client = crate::BigQueryJobService::from_stub(ShortDdlStub);
        let client = crate::Client::new_for_testing(
            bq_client,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );

        let (schema, mut receiver) = client
            .query("CREATE SCHEMA IF NOT EXISTS my_dataset")
            .read()
            .await
            .unwrap();
        assert_eq!(schema.len(), 0);
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_read_long_ddl_query_no_schema() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            CreateReadSessionRequest, ReadRowsRequest, ReadRowsResponse, ReadSession,
        };
        use google_cloud_bigquery_v2::model::{
            GetJobRequest, Job, JobConfiguration, JobConfigurationQuery, JobStatistics,
            JobStatistics2, JobStatus, PostQueryRequest, TableReference,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug)]
        struct LongDdlJobStub {
            get_job_calls: Arc<std::sync::Mutex<usize>>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for LongDdlJobStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_complete(false)
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("create-model-job-1"),
                    );
                Ok(Response::from(resp))
            }

            async fn get_job(
                &self,
                req: GetJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<Job>> {
                *self.get_job_calls.lock().unwrap() += 1;
                assert_eq!(req.project_id, "job-proj");
                assert_eq!(req.job_id, "create-model-job-1");
                assert_eq!(req.location, "US");

                // Real BigQuery CREATE_MODEL jobs populate destination_table with the model
                // reference (which fails if passed to create_read_session) and set
                // statement_type = "CREATE_MODEL", ddl_operation_performed = "CREATE".
                let job = Job::new()
                    .set_status(JobStatus::new().set_state("DONE"))
                    .set_statistics(
                        JobStatistics::new().set_query(
                            JobStatistics2::new()
                                .set_statement_type("CREATE_MODEL")
                                .set_ddl_operation_performed("CREATE"),
                        ),
                    )
                    .set_configuration(
                        JobConfiguration::new().set_query(
                            JobConfigurationQuery::new().set_destination_table(
                                TableReference::new()
                                    .set_project_id("job-proj")
                                    .set_dataset_id("my_ds")
                                    .set_table_id("my_model"),
                            ),
                        ),
                    );
                Ok(Response::from(job))
            }
        }

        #[derive(Debug)]
        struct HeartbeatThenCloseReadStub;

        impl google_cloud_bigquery::stub::Read for HeartbeatThenCloseReadStub {
            async fn create_read_session(
                &self,
                _req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                panic!("create_read_session must not be called for DDL queries");
            }

            async fn read_rows(
                &self,
                req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                assert_eq!(
                    req.read_stream,
                    "projects/job-proj/locations/US/jobs/create-model-job-1/streams/_default"
                );
                let (tx, rx) = tokio::sync::mpsc::channel(2);
                // Simulate heartbeat while model is training, followed by clean EOF.
                tx.try_send(Ok(ReadRowsResponse::new())).unwrap();
                drop(tx);
                Ok(ResponseStream::from(rx))
            }
        }

        let get_job_calls = Arc::new(std::sync::Mutex::new(0));
        let bq_client = crate::BigQueryJobService::from_stub(LongDdlJobStub {
            get_job_calls: Arc::clone(&get_job_calls),
        });
        let read_client = Read::from_stub(HeartbeatThenCloseReadStub);
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let (schema, mut receiver) = client
            .query("CREATE OR REPLACE MODEL my_ds.my_model OPTIONS(model_type='linear_reg') AS SELECT 1 AS x, 2 AS label")
            .read()
            .await
            .unwrap();
        assert_eq!(schema.len(), 0);
        assert!(receiver.recv().await.is_none());
        assert_eq!(*get_job_calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_read_long_empty_select_fetches_schema_from_destination_table() {
        use std::sync::Mutex;

        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{
            ArrowSchema as StorageArrowSchema, CreateReadSessionRequest, ReadRowsRequest,
            ReadRowsResponse, ReadSession, ReadStream,
        };
        use google_cloud_bigquery_v2::model::{
            GetJobRequest, Job, JobConfiguration, JobConfigurationQuery, JobStatistics,
            JobStatistics2, JobStatus, PostQueryRequest, TableReference,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;
        use google_cloud_gax::streaming::ResponseStream;

        #[derive(Debug, Default)]
        struct Recorded {
            get_job_calls: usize,
            create_session_table: Option<String>,
            read_rows_streams: Vec<String>,
        }

        #[derive(Debug)]
        struct LongEmptySelectJobStub {
            recorded: Arc<Mutex<Recorded>>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for LongEmptySelectJobStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_complete(false)
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("long-empty-select-1"),
                    );
                Ok(Response::from(resp))
            }

            async fn get_job(
                &self,
                _req: GetJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<Job>> {
                self.recorded.lock().unwrap().get_job_calls += 1;
                let job = Job::new()
                    .set_status(JobStatus::new().set_state("DONE"))
                    .set_statistics(
                        JobStatistics::new()
                            .set_query(JobStatistics2::new().set_statement_type("SELECT")),
                    )
                    .set_configuration(
                        JobConfiguration::new().set_query(
                            JobConfigurationQuery::new().set_destination_table(
                                TableReference::new()
                                    .set_project_id("dest-proj")
                                    .set_dataset_id("_anon_ds")
                                    .set_table_id("anon_empty_tbl"),
                            ),
                        ),
                    );
                Ok(Response::from(job))
            }
        }

        #[derive(Debug)]
        struct LongEmptySelectReadStub {
            schema_bytes: Vec<u8>,
            recorded: Arc<Mutex<Recorded>>,
        }

        impl google_cloud_bigquery::stub::Read for LongEmptySelectReadStub {
            async fn create_read_session(
                &self,
                req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let mut rec = self.recorded.lock().unwrap();
                rec.create_session_table = req.read_session.map(|rs| rs.table);
                let session = ReadSession::new()
                    .set_arrow_schema(
                        StorageArrowSchema::new().set_serialized_schema(self.schema_bytes.clone()),
                    )
                    .set_streams([ReadStream::new().set_name("streams/unused_table_stream")]);
                Ok(Response::from(session))
            }

            async fn read_rows(
                &self,
                req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                self.recorded
                    .lock()
                    .unwrap()
                    .read_rows_streams
                    .push(req.read_stream.clone());
                assert!(
                    req.read_stream.ends_with("/streams/_default"),
                    "should only call read_rows on the _default job stream, not the empty destination table stream"
                );
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                tx.try_send(Ok(ReadRowsResponse::new())).unwrap();
                drop(tx);
                Ok(ResponseStream::from(rx))
            }
        }

        let (schema_bytes, _) = create_test_arrow_payload_with_range(0, 0);
        let recorded = Arc::new(Mutex::new(Recorded::default()));
        let bq_client = crate::BigQueryJobService::from_stub(LongEmptySelectJobStub {
            recorded: Arc::clone(&recorded),
        });
        let read_client = Read::from_stub(LongEmptySelectReadStub {
            schema_bytes,
            recorded: Arc::clone(&recorded),
        });
        let client =
            crate::Client::new_for_testing(bq_client, read_client, "test-quota-proj".to_string());

        let (schema, mut receiver) = client
            .query("SELECT col1 FROM tbl WHERE FALSE")
            .read()
            .await
            .unwrap();
        let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
        assert_eq!(col_names, vec!["col1"]);
        assert!(receiver.recv().await.is_none());

        let rec = recorded.lock().unwrap();
        assert_eq!(rec.get_job_calls, 1);
        assert_eq!(
            rec.create_session_table.as_deref(),
            Some("projects/dest-proj/datasets/_anon_ds/tables/anon_empty_tbl")
        );
        assert_eq!(
            rec.read_rows_streams,
            vec!["projects/job-proj/locations/US/jobs/long-empty-select-1/streams/_default"]
        );
    }

    #[tokio::test]
    async fn test_read_long_empty_select_missing_destination_table_error() {
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery_v2::model::{
            ErrorProto, GetJobRequest, Job, JobStatistics, JobStatistics2, JobStatus,
            PostQueryRequest,
        };
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct ConfigurableEmptyJobStub {
            job: std::sync::Mutex<Option<Job>>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for ConfigurableEmptyJobStub {
            async fn query(
                &self,
                _req: PostQueryRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<QueryResponse>> {
                let resp = QueryResponse::new()
                    .set_job_complete(false)
                    .set_job_reference(
                        JobReference::new()
                            .set_project_id("job-proj")
                            .set_location("US")
                            .set_job_id("empty-stream-job"),
                    );
                Ok(Response::from(resp))
            }

            async fn get_job(
                &self,
                _req: GetJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<Job>> {
                let job = self.job.lock().unwrap().take().unwrap();
                Ok(Response::from(job))
            }
        }

        // Case 1: SELECT job with no destination_table -> Protocol error
        let select_without_dest = Job::new()
            .set_status(JobStatus::new().set_state("DONE"))
            .set_statistics(
                JobStatistics::new().set_query(JobStatistics2::new().set_statement_type("SELECT")),
            );
        let bq_client = crate::BigQueryJobService::from_stub(ConfigurableEmptyJobStub {
            job: std::sync::Mutex::new(Some(select_without_dest)),
        });
        let (read_stub, _) = CannedReadRowsStub::new(vec![]);
        let client = crate::Client::new_for_testing(
            bq_client,
            Read::from_stub(read_stub),
            "test-quota-proj".to_string(),
        );
        let err = match client.query("SELECT 1 WHERE FALSE").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected Protocol error when SELECT job has no destination_table"),
        };
        assert!(matches!(err, BigQueryError::Protocol(_)));

        // Case 2: Failed job where ReadRows closed cleanly at EOF -> JobFailed error
        let failed_job = Job::new().set_status(
            JobStatus::new()
                .set_state("DONE")
                .set_error_result(ErrorProto::new().set_reason("invalid").set_message("bad")),
        );
        let bq_client = crate::BigQueryJobService::from_stub(ConfigurableEmptyJobStub {
            job: std::sync::Mutex::new(Some(failed_job)),
        });
        let (read_stub, _) = CannedReadRowsStub::new(vec![]);
        let client = crate::Client::new_for_testing(
            bq_client,
            Read::from_stub(read_stub),
            "test-quota-proj".to_string(),
        );
        let err = match client.query("CREATE MODEL ...").read().await {
            Err(e) => e,
            Ok(_) => panic!("expected JobFailed error when job failed"),
        };
        assert!(matches!(err, BigQueryError::JobFailed(_)));
    }
}
