use std::env;

use arrow_bigquery::*;
use polars_arrow::array::{Int64Array, Utf8Array};

#[tokio::test(flavor = "multi_thread")]
async fn test_read_small_public_table() {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    let client = Client::from_builder(
        ServiceConfigBuilder::new()
            .with_user_agent(Some("integration-test/1.0".to_string()))
            .with_quota_project_id(Some(quota_project_id)),
    )
    .await
    .expect("should build client");

    let table = BigQueryTableId {
        project_id: "bigquery-public-data".to_string(),
        dataset_id: "usa_names".to_string(),
        table_id: "usa_1910_2013".to_string(),
    };

    let (_, mut receiver) = client
        .read_table(
            &table,
            ReadOptions {
                maintain_order: false,
                ..Default::default()
            },
        )
        .await
        .expect("public table read should work with default credentials");

    let mut total_rows = 0;
    while let Some(batch) = receiver.recv().await {
        total_rows += batch.expect("should receive valid record batch").len();
    }
    assert!(total_rows > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dry_run_query() {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    let client = Client::from_builder(
        ServiceConfigBuilder::new()
            .with_user_agent(Some("integration-test/1.0".to_string()))
            .with_quota_project_id(Some(quota_project_id.clone())),
    )
    .await
    .expect("should build client");

    let sql =
        "SELECT name, number FROM `bigquery-public-data.usa_names.usa_1910_2013` WHERE state = 'WA'";
    let batch = client
        .query(sql)
        .dry_run()
        .await
        .expect("dry_run should succeed");

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
            "totalBytesProcessed",
            "cacheHit",
            "statementType",
            "creationTime",
        ]
    );

    let bq_schema = batch.arrays()[0]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap()
        .value(0);
    assert!(bq_schema.contains("name"));
    assert!(bq_schema.contains("number"));

    let total_bytes_processed = batch.arrays()[4]
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert!(total_bytes_processed > 0);

    let statement_type = batch.arrays()[6]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap()
        .value(0);
    assert_eq!(statement_type, "SELECT");

    let creation_time = batch.arrays()[7]
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert!(creation_time > 0);
}
