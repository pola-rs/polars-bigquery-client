use std::env;

use arrow_bigquery::*;
use polars_arrow::array::{BooleanArray, Int64Array, ListArray, StructArray, Utf8Array};

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

    let bq_schema = batch.arrays()[0]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap()
        .value(0);
    assert!(bq_schema.contains("name"));
    assert!(bq_schema.contains("number"));

    let project_id = batch.arrays()[1]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap()
        .value(0);
    assert_eq!(project_id, quota_project_id);

    let job_type = batch.arrays()[4]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap()
        .value(0);
    assert_eq!(job_type, "QUERY");

    let dispatched_sql = batch.arrays()[5]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap()
        .value(0);
    assert_eq!(dispatched_sql, sql);

    let use_legacy_sql = batch.arrays()[7]
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .value(0);
    assert!(!use_legacy_sql);

    let ref_tables_list = batch.arrays()[8]
        .as_any()
        .downcast_ref::<ListArray<i32>>()
        .unwrap();
    let ref_tables_elem = ref_tables_list.value(0);
    let ref_struct = ref_tables_elem
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
        "bigquery-public-data"
    );
    assert_eq!(
        ref_struct.values()[1]
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .unwrap()
            .value(0),
        "usa_names"
    );
    assert_eq!(
        ref_struct.values()[2]
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .unwrap()
            .value(0),
        "usa_1910_2013"
    );

    let total_bytes_processed = batch.arrays()[9]
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert!(total_bytes_processed > 0);

    let statement_type = batch.arrays()[11]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap()
        .value(0);
    assert_eq!(statement_type, "SELECT");

    let creation_time = batch.arrays()[12]
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert!(creation_time > 0);
}
