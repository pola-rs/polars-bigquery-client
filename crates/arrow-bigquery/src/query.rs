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

use crate::error::BigQueryError;
use crate::{BigQueryRecordBatchReceiver, StreamOptions};

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
    arrow_buffer_compression: Option<arrow_serialization_options::CompressionCodec>,
    use_query_cache: Option<bool>,
}

impl<'a> Query<'a> {
    pub(crate) fn new(client: &'a crate::Client, sql: impl Into<String>) -> Self {
        Self {
            client,
            sql: sql.into(),
            arrow_buffer_compression: None,
            use_query_cache: None,
        }
    }

    pub fn with_arrow_buffer_compression(
        mut self,
        codec: impl Into<Option<crate::CompressionCodec>>,
    ) -> Self {
        self.arrow_buffer_compression = codec.into().map(bq_to_gapic_compression_codec);
        self
    }

    pub fn with_use_query_cache(mut self, use_query_cache: impl Into<Option<bool>>) -> Self {
        self.use_query_cache = use_query_cache.into();
        self
    }

    fn build_read_request(&self) -> QueryRequest {
        let arrow_options = ArrowSerializationOptions::new().set_buffer_compression(
            self.arrow_buffer_compression
                .clone()
                .unwrap_or_else(|| bq_to_gapic_compression_codec(crate::DEFAULT_COMPRESSION)),
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

    pub async fn dry_run(&self) -> Result<RecordBatch, BigQueryError> {
        let response = self
            .prepare_query(QueryRequest::new().set_dry_run(true))
            .with_idempotency(true)
            .send()
            .await?;
        // Keep column names/types in sync with pandas-gbq:
        // https://github.com/googleapis/google-cloud-python/blob/main/packages/pandas-gbq/pandas_gbq/dry_runs.py#L4
        query_response_to_record_batch(&response)
    }

    pub async fn read(
        &self,
    ) -> Result<(ArrowSchemaRef, BigQueryRecordBatchReceiver), BigQueryError> {
        let mut response = self.prepare_query(self.build_read_request()).send().await?;

        if response.job_complete == Some(false) {
            let stream_name = default_job_stream_path(&response)?;
            return self
                .client
                .read_stream(
                    stream_name,
                    StreamOptions {
                        offset: 0,
                        schema: None,
                    },
                )
                .await;
        }

        let serialized_schema = match response.results_schema.take() {
            Some(query_response::ResultsSchema::ArrowSchema(value)) => value.serialized_schema,
            _ => {
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

        let has_remaining_rows = response
            .total_rows
            .is_some_and(|total_rows| total_rows > page_row_count)
            || !response.page_token.is_empty();

        let stream_name = if has_remaining_rows {
            Some(default_job_stream_path(&response)?)
        } else {
            None
        };

        let raw_first_page = match response.results {
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

        match (first_page_bytes, stream_name) {
            (None, None) => {
                let metadata = crate::bigquery_read_stream::parse_arrow_schema(&serialized_schema)?;
                let (_tx, rx) = tokio::sync::mpsc::channel(1);
                Ok((
                    Arc::new(metadata.schema),
                    BigQueryRecordBatchReceiver::new(rx, None, Vec::new()),
                ))
            },
            (None, Some(stream_name)) => {
                self.client
                    .read_stream(
                        stream_name,
                        StreamOptions {
                            offset: response.page_row_count,
                            schema: Some(serialized_schema),
                        },
                    )
                    .await
            },
            (Some(first_page_bytes), stream_name) => {
                let metadata = crate::bigquery_read_stream::parse_arrow_schema(&serialized_schema)?;
                let read_rows = stream_name
                    .map(|name| self.client.prepare_read_rows(name, response.page_row_count));
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
}

fn non_empty_str(s: &str) -> Option<&str> {
    (!s.is_empty()).then_some(s)
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

fn default_job_stream_path(response: &QueryResponse) -> Result<String, BigQueryError> {
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
            Ok(_) => panic!("expected Grpc error for HTTP 400"),
        };
        server_task.await.unwrap();
        assert!(matches!(err, BigQueryError::Grpc(_)));
        assert!(err
            .to_string()
            .contains("Syntax error: Unexpected end of script"));
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
                Ok(Response::from(QueryResponse::new()))
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
}
