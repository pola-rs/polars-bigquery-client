use std::sync::Arc;

use google_cloud_bigquery::model_ext::QueryMetadata;
use google_cloud_bigquery_v2::model::TableReference;
use polars_arrow::array::{Array, BooleanArray, Int64Array, ListArray, StructArray, Utf8Array};
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field, TimeUnit};
use polars_arrow::offset::Offsets;
use polars_arrow::record_batch::RecordBatch;

use crate::error::BigQueryError;

#[derive(Debug)]
pub struct Query {
    bq_client: crate::BigQueryClient,
    sql: String,
    quota_project_id: String,
}

impl Query {
    pub(crate) fn new(
        bq_client: crate::BigQueryClient,
        sql: impl Into<String>,
        quota_project_id: impl Into<String>,
    ) -> Self {
        Self {
            bq_client,
            sql: sql.into(),
            quota_project_id: quota_project_id.into(),
        }
    }

    fn prepare_query(&self) -> google_cloud_bigquery::builder::bigquery::Query {
        // TODO: populate user-agent too, after
        // https://github.com/googleapis/google-cloud-rust/pull/7193
        // is released.
        self.bq_client
            .query(&self.sql)
            .with_project_id(&self.quota_project_id)
    }

    pub async fn dry_run(&self) -> Result<RecordBatch, BigQueryError> {
        let query = self.prepare_query().set_dry_run(true).send().await?;
        // Keep schema in sync with pandas-gbq:
        // https://github.com/googleapis/google-cloud-python/blob/main/packages/pandas-gbq/pandas_gbq/dry_runs.py#L4
        query_metadata_to_record_batch(query.metadata())
    }
}

fn non_empty_str(s: &str) -> Option<&str> {
    (!s.is_empty()).then_some(s)
}

fn table_ref_struct_dtype() -> ArrowDataType {
    ArrowDataType::Struct(vec![
        Field::new("projectId".into(), ArrowDataType::Utf8, true),
        Field::new("datasetId".into(), ArrowDataType::Utf8, true),
        Field::new("tableId".into(), ArrowDataType::Utf8, true),
    ])
}

fn table_references_to_struct_array(tables: &[TableReference]) -> StructArray {
    let project_ids =
        Utf8Array::<i32>::from_iter(tables.iter().map(|t| non_empty_str(&t.project_id))).boxed();
    let dataset_ids =
        Utf8Array::<i32>::from_iter(tables.iter().map(|t| non_empty_str(&t.dataset_id))).boxed();
    let table_ids =
        Utf8Array::<i32>::from_iter(tables.iter().map(|t| non_empty_str(&t.table_id))).boxed();

    StructArray::new(
        table_ref_struct_dtype(),
        tables.len(),
        vec![project_ids, dataset_ids, table_ids],
        None,
    )
}

fn query_metadata_to_record_batch(metadata: &QueryMetadata) -> Result<RecordBatch, BigQueryError> {
    let query_config = metadata
        .configuration
        .as_ref()
        .and_then(|c| c.query.as_ref());
    let query_stats = metadata.statistics.as_ref().and_then(|s| s.query.as_ref());

    // 1. bigquerySchema
    let schema = metadata
        .schema
        .as_ref()
        .or_else(|| query_stats.and_then(|q| q.schema.as_ref()));
    let bigquery_schema = schema
        .map(serde_json::to_string)
        .transpose()
        .map_err(|e| BigQueryError::Other(Box::new(e)))?;
    let bigquery_schema_arr = Utf8Array::<i32>::from([bigquery_schema.as_deref()]).boxed();

    // 2. jobReference fields: projectId, jobId, location
    let job_ref = metadata.job_reference.as_ref();
    let project_id = job_ref.and_then(|jr| non_empty_str(&jr.project_id));
    let job_id = job_ref.and_then(|jr| non_empty_str(&jr.job_id));
    let location = job_ref
        .and_then(|jr| jr.location.as_deref())
        .and_then(non_empty_str)
        .or_else(|| non_empty_str(&metadata.location));

    let project_id_arr = Utf8Array::<i32>::from([project_id]).boxed();
    let job_id_arr = Utf8Array::<i32>::from([job_id]).boxed();
    let location_arr = Utf8Array::<i32>::from([location]).boxed();

    // 3. configuration fields: jobType, dispatchedSql, destinationTable, useLegacySql
    let job_type = metadata
        .configuration
        .as_ref()
        .and_then(|c| non_empty_str(&c.job_type));
    let dispatched_sql = query_config.and_then(|q| non_empty_str(&q.query));
    let destination_table_dtype = table_ref_struct_dtype();
    let destination_table_arr = match query_config.and_then(|q| q.destination_table.as_ref()) {
        Some(table_ref) => table_references_to_struct_array(std::slice::from_ref(table_ref)).boxed(),
        None => StructArray::new_null(destination_table_dtype.clone(), 1).boxed(),
    };
    let use_legacy_sql = query_config.and_then(|q| q.use_legacy_sql);

    let job_type_arr = Utf8Array::<i32>::from([job_type]).boxed();
    let dispatched_sql_arr = Utf8Array::<i32>::from([dispatched_sql]).boxed();
    let use_legacy_sql_arr = BooleanArray::from([use_legacy_sql]).boxed();

    // 4. statistics.query fields: referencedTables, totalBytesProcessed, cacheHit, statementType
    let referenced_tables_dtype = ListArray::<i32>::default_datatype(table_ref_struct_dtype());
    let referenced_tables_arr = match query_stats.map(|q| q.referenced_tables.as_slice()) {
        Some(tables) if !tables.is_empty() => {
            let values = table_references_to_struct_array(tables).boxed();
            let offsets = Offsets::<i32>::try_from_iter([tables.len()])?.into();
            ListArray::<i32>::try_new(referenced_tables_dtype.clone(), offsets, values, None)?
                .boxed()
        },
        _ => ListArray::<i32>::new_null(referenced_tables_dtype.clone(), 1).boxed(),
    };

    let total_bytes_processed = metadata
        .total_bytes_processed
        .or_else(|| query_stats.and_then(|q| q.total_bytes_processed))
        .or_else(|| {
            metadata
                .statistics
                .as_ref()
                .and_then(|s| s.total_bytes_processed)
        })
        .unwrap_or(0);
    let total_bytes_processed_arr = Int64Array::from_slice([total_bytes_processed]).boxed();

    let cache_hit = metadata
        .cache_hit
        .or_else(|| query_stats.and_then(|q| q.cache_hit));
    let cache_hit_arr = BooleanArray::from([cache_hit]).boxed();

    let statement_type = non_empty_str(&metadata.statement_type)
        .or_else(|| query_stats.and_then(|q| non_empty_str(&q.statement_type)));
    let statement_type_arr = Utf8Array::<i32>::from([statement_type]).boxed();

    // 5. statistics.creationTime (ms, UTC)
    let creation_time_dtype = ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()));
    let creation_time = metadata.creation_time.or_else(|| {
        metadata
            .statistics
            .as_ref()
            .and_then(|s| (s.creation_time > 0).then_some(s.creation_time))
    });
    let creation_time_arr = Int64Array::from([creation_time])
        .to(creation_time_dtype.clone())
        .boxed();

    let arrow_schema = ArrowSchema::from_iter([
        Field::new("bigquerySchema".into(), ArrowDataType::Utf8, true),
        Field::new("projectId".into(), ArrowDataType::Utf8, true),
        Field::new("jobId".into(), ArrowDataType::Utf8, true),
        Field::new("location".into(), ArrowDataType::Utf8, true),
        Field::new("jobType".into(), ArrowDataType::Utf8, true),
        Field::new("dispatchedSql".into(), ArrowDataType::Utf8, true),
        Field::new("destinationTable".into(), destination_table_dtype, true),
        Field::new("useLegacySql".into(), ArrowDataType::Boolean, true),
        Field::new("referencedTables".into(), referenced_tables_dtype, true),
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
        job_type_arr,
        dispatched_sql_arr,
        destination_table_arr,
        use_legacy_sql_arr,
        referenced_tables_arr,
        total_bytes_processed_arr,
        cache_hit_arr,
        statement_type_arr,
        creation_time_arr,
    ];

    Ok(RecordBatch::try_new(1, Arc::new(arrow_schema), columns)?)
}

#[cfg(test)]
mod tests {
    use google_cloud_bigquery_v2::model::{
        Job, JobConfiguration, JobConfigurationQuery, JobReference, JobStatistics, JobStatistics2,
        TableFieldSchema, TableSchema,
    };

    use super::*;

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
    fn test_query_metadata_to_record_batch_all_fields() {
        let total_bytes_processed = 15i64;
        let schema = TableSchema::new().set_fields([TableFieldSchema::new()
            .set_name("col1")
            .set_type("INTEGER")
            .set_mode("NULLABLE")]);
        let expected_schema_json = serde_json::to_string(&schema).unwrap();

        let metadata = QueryMetadata::new()
            .set_schema(schema.clone())
            .set_job_reference(
                JobReference::new()
                    .set_project_id("project-id")
                    .set_job_id("job-id")
                    .set_location("US"),
            )
            .set_configuration(
                JobConfiguration::new()
                    .set_job_type("QUERY")
                    .set_query(
                        JobConfigurationQuery::new()
                            .set_query("SELECT * FROM `test_table`")
                            .set_destination_table(
                                TableReference::new()
                                    .set_project_id("project-id")
                                    .set_dataset_id("dataset-id")
                                    .set_table_id("table-id"),
                            )
                            .set_use_legacy_sql(false),
                    ),
            )
            .set_statistics(
                JobStatistics::new()
                    .set_creation_time(1_767_037_135_155i64)
                    .set_total_bytes_processed(total_bytes_processed)
                    .set_query(
                        JobStatistics2::new()
                            .set_referenced_tables([TableReference::new()
                                .set_project_id("ref-proj")
                                .set_dataset_id("ref-ds")
                                .set_table_id("test_table")])
                            .set_total_bytes_processed(total_bytes_processed)
                            .set_cache_hit(true)
                            .set_statement_type("SELECT")
                            .set_schema(schema),
                    ),
            )
            .set_total_bytes_processed(total_bytes_processed)
            .set_cache_hit(true)
            .set_statement_type("SELECT")
            .set_creation_time(1_767_037_135_155i64);

        let batch = query_metadata_to_record_batch(&metadata).unwrap();
        assert_eq!(batch.len(), 1);

        let col_names: Vec<&str> = batch
            .schema()
            .iter_names()
            .map(|s| s.as_str())
            .collect();
        assert_eq!(
            col_names,
            vec![
                "bigquerySchema",
                "projectId",
                "jobId",
                "location",
                "jobType",
                "dispatchedSql",
                "destinationTable",
                "useLegacySql",
                "referencedTables",
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
        assert_eq!(utf8_val(&batch, 4), Some("QUERY"));
        assert_eq!(utf8_val(&batch, 5), Some("SELECT * FROM `test_table`"));

        let dest_struct = batch.arrays()[6]
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("expected StructArray for destinationTable");
        assert!(dest_struct.validity().is_none());
        let dest_proj = dest_struct.values()[0]
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .unwrap();
        let dest_ds = dest_struct.values()[1]
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .unwrap();
        let dest_tbl = dest_struct.values()[2]
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .unwrap();
        assert_eq!(dest_proj.value(0), "project-id");
        assert_eq!(dest_ds.value(0), "dataset-id");
        assert_eq!(dest_tbl.value(0), "table-id");

        assert_eq!(bool_val(&batch, 7), Some(false));

        let ref_tables_list = batch.arrays()[8]
            .as_any()
            .downcast_ref::<ListArray<i32>>()
            .expect("expected ListArray<i32> for referencedTables");
        assert!(ref_tables_list.validity().is_none());
        let first_list_elem = ref_tables_list.value(0);
        let ref_struct = first_list_elem
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(ref_struct.len(), 1);
        assert_eq!(
            ref_struct.values()[0]
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap()
                .value(0),
            "ref-proj"
        );
        assert_eq!(
            ref_struct.values()[1]
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap()
                .value(0),
            "ref-ds"
        );
        assert_eq!(
            ref_struct.values()[2]
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap()
                .value(0),
            "test_table"
        );

        assert_eq!(int64_val(&batch, 9), Some(15));
        assert_eq!(bool_val(&batch, 10), Some(true));
        assert_eq!(utf8_val(&batch, 11), Some("SELECT"));
        assert_eq!(int64_val(&batch, 12), Some(1_767_037_135_155));
        assert_eq!(
            batch.arrays()[12].dtype(),
            &ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()))
        );
    }

    #[test]
    fn test_query_metadata_to_record_batch_missing_bytes_use_zero() {
        let metadata = QueryMetadata::new()
            .set_job_reference(
                JobReference::new()
                    .set_project_id("project-id")
                    .set_job_id("job-id")
                    .set_location("US"),
            )
            .set_configuration(
                JobConfiguration::new()
                    .set_job_type("QUERY")
                    .set_query(
                        JobConfigurationQuery::new()
                            .set_query("SELECT * FROM `test_table`")
                            .set_use_legacy_sql(false),
                    ),
            )
            .set_statistics(
                JobStatistics::new().set_query(
                    JobStatistics2::new()
                        .set_cache_hit(true)
                        .set_statement_type("SELECT"),
                ),
            );

        let batch = query_metadata_to_record_batch(&metadata).unwrap();
        assert_eq!(batch.len(), 1);

        assert_eq!(utf8_val(&batch, 0), None);
        assert_eq!(utf8_val(&batch, 1), Some("project-id"));
        assert_eq!(utf8_val(&batch, 2), Some("job-id"));
        assert_eq!(utf8_val(&batch, 3), Some("US"));
        assert_eq!(utf8_val(&batch, 4), Some("QUERY"));
        assert_eq!(utf8_val(&batch, 5), Some("SELECT * FROM `test_table`"));

        let dest_struct = batch.arrays()[6]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(
            dest_struct.validity().map(|v| v.get_bit(0)),
            Some(false),
            "destinationTable should be null when absent"
        );

        assert_eq!(bool_val(&batch, 7), Some(false));

        let ref_tables_list = batch.arrays()[8]
            .as_any()
            .downcast_ref::<ListArray<i32>>()
            .unwrap();
        assert_eq!(
            ref_tables_list.validity().map(|v| v.get_bit(0)),
            Some(false),
            "referencedTables should be null when empty/absent"
        );

        assert_eq!(int64_val(&batch, 9), Some(0));
        assert_eq!(bool_val(&batch, 10), Some(true));
        assert_eq!(utf8_val(&batch, 11), Some("SELECT"));
        assert_eq!(int64_val(&batch, 12), None);
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
                req_str.contains("/projects/test-quota-proj/jobs"),
                "unexpected request path: {req_str}"
            );
            assert!(
                req_str.contains("\"dryRun\":true"),
                "expected dryRun: true in request body: {req_str}"
            );

            let response_job = Job::new()
                .set_job_reference(
                    JobReference::new()
                        .set_project_id("test-quota-proj")
                        .set_location("US"),
                )
                .set_configuration(
                    JobConfiguration::new()
                        .set_job_type("QUERY")
                        .set_dry_run(true)
                        .set_query(
                            JobConfigurationQuery::new()
                                .set_query("SELECT 1 AS one")
                                .set_destination_table(
                                    TableReference::new()
                                        .set_project_id("test-quota-proj")
                                        .set_dataset_id("_anon_ds")
                                        .set_table_id("anon_tbl"),
                                )
                                .set_use_legacy_sql(false),
                        ),
                )
                .set_statistics(
                    JobStatistics::new()
                        .set_creation_time(1_767_037_135_155i64)
                        .set_total_bytes_processed(0i64)
                        .set_query(
                            JobStatistics2::new()
                                .set_total_bytes_processed(0i64)
                                .set_cache_hit(false)
                                .set_statement_type("SELECT")
                                .set_schema(TableSchema::new().set_fields([
                                    TableFieldSchema::new().set_name("one").set_type("INT64"),
                                ])),
                        ),
                );
            let body = serde_json::to_string(&response_job).unwrap();
            let http_resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(http_resp.as_bytes()).await.unwrap();
        });

        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let bq_client = crate::BigQueryClient::builder()
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
        let client = crate::Client::new_for_testing(
            bq_client,
            read_client,
            "test-quota-proj".to_string(),
        );

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
        assert_eq!(utf8_val(&batch, 4), Some("QUERY"));
        assert_eq!(utf8_val(&batch, 5), Some("SELECT 1 AS one"));
        assert_eq!(bool_val(&batch, 7), Some(false));
        assert_eq!(int64_val(&batch, 9), Some(0));
        assert_eq!(bool_val(&batch, 10), Some(false));
        assert_eq!(utf8_val(&batch, 11), Some("SELECT"));
        assert_eq!(int64_val(&batch, 12), Some(1_767_037_135_155));
    }
}

