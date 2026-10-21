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

#[tokio::test(flavor = "multi_thread")]
async fn test_read_query() {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    let client = Client::from_builder(
        ServiceConfigBuilder::new()
            .with_user_agent(Some("integration-test/1.0".to_string()))
            .with_quota_project_id(Some(quota_project_id)),
    )
    .await
    .expect("should build client");

    let sql = "SELECT 1 AS one, 'hello' AS greeting";
    let (schema, mut receiver) = client
        .query(sql)
        .read()
        .await
        .expect("query read should succeed");

    let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
    assert_eq!(col_names, vec!["one", "greeting"]);

    let mut total_rows = 0;
    while let Some(batch) = receiver.recv().await {
        let batch = batch.expect("should receive valid record batch");
        total_rows += batch.len();
    }
    assert_eq!(total_rows, 1);

    let empty_sql = "SELECT 1 AS one, 'hello' AS greeting LIMIT 0";
    let (empty_schema, mut empty_receiver) = client
        .query(empty_sql)
        .with_arrow_buffer_compression(CompressionCodec::Zstd)
        .read()
        .await
        .expect("empty query read should succeed");
    let empty_col_names: Vec<&str> = empty_schema.iter_names().map(|s| s.as_str()).collect();
    assert_eq!(empty_col_names, vec!["one", "greeting"]);
    assert!(empty_receiver.recv().await.is_none());

    let multipage_sql =
        "SELECT * FROM `bigquery-public-data.sec_quarterly_financials.submission` LIMIT 250000";
    let (_, mut multipage_receiver) = client
        .query(multipage_sql)
        .with_arrow_buffer_compression(CompressionCodec::Zstd)
        .read()
        .await
        .expect("multipage query read should succeed");

    let mut multipage_rows = 0;
    while let Some(batch) = multipage_receiver.recv().await {
        let batch = batch.expect("should receive valid record batch");
        multipage_rows += batch.len();
    }
    assert_eq!(multipage_rows, 250_000);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_long_running_query() {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    let client = Client::from_builder(
        ServiceConfigBuilder::new()
            .with_user_agent(Some("integration-test/1.0".to_string()))
            .with_quota_project_id(Some(quota_project_id)),
    )
    .await
    .expect("should build client");

    let sql = r#"
        WITH
          base AS (
            SELECT
              a * 1000 + b AS id,
              FARM_FINGERPRINT(CAST(a * 1000 + b AS STRING)) AS fp,
              TO_HEX(SHA256(CAST(a * 1000 + b AS STRING))) AS digest
            FROM
              UNNEST(GENERATE_ARRAY(0, 999)) AS a
            CROSS JOIN
              UNNEST(GENERATE_ARRAY(0, 999)) AS b
          ),
          stage1 AS (
            SELECT
              b1.id AS id1,
              b2.id AS id2,
              FARM_FINGERPRINT(CONCAT(b1.digest, b2.digest)) AS combined_fp,
              TO_HEX(SHA256(CONCAT(b1.digest, b2.digest))) AS combined_digest
            FROM
              base AS b1
            INNER JOIN
              base AS b2
            ON
              MOD(ABS(b1.fp), 200000) = MOD(ABS(b2.fp), 200000)
              AND b1.id != b2.id
          ),
          stage2 AS (
            SELECT
              s1.id1,
              COUNT(*) AS match_count,
              MAX(s1.combined_digest) AS max_digest
            FROM
              stage1 AS s1
            INNER JOIN
              base AS b3
            ON
              MOD(ABS(s1.combined_fp), 500000) = b3.id
            GROUP BY
              s1.id1
          )
        SELECT
          id1,
          match_count,
          max_digest
        FROM
          stage2
        ORDER BY
          match_count DESC,
          id1 ASC
        LIMIT 10000
    "#;

    let (schema, mut receiver) = client
        .query(sql)
        .with_use_query_cache(false)
        .read()
        .await
        .expect("long running query read should succeed");

    let col_names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
    assert_eq!(col_names, vec!["id1", "match_count", "max_digest"]);

    let mut total_rows = 0;
    while let Some(batch) = receiver.recv().await {
        let batch = batch.expect("should receive valid record batch");
        total_rows += batch.len();
    }
    assert_eq!(total_rows, 10_000);
}
