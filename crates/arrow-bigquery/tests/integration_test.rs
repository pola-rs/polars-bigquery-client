use std::env;

use arrow_array::{Int64Array, StringArray, TimestampMillisecondArray};
use arrow_bigquery::*;

async fn setup_client() -> Client {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    Client::from_builder(
        ServiceConfigBuilder::new()
            .with_user_agent(Some("integration-test/1.0".to_string()))
            .with_quota_project_id(Some(quota_project_id)),
    )
    .await
    .expect("should build client")
}

async fn consume_receiver(mut receiver: BigQueryRecordBatchReceiver) -> usize {
    let mut total_rows = 0;
    while let Some(batch) = receiver.recv().await {
        total_rows += batch.expect("should receive valid record batch").num_rows();
    }
    total_rows
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_small_public_table() {
    let client = setup_client().await;

    let table = BigQueryTableId {
        project_id: "bigquery-public-data".to_string(),
        dataset_id: "usa_names".to_string(),
        table_id: "usa_1910_2013".to_string(),
    };

    let (_, receiver) = client
        .read_table(
            &table,
            ReadOptions {
                maintain_order: false,
                ..Default::default()
            },
        )
        .await
        .expect("public table read should work with default credentials");

    assert!(consume_receiver(receiver).await > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dry_run_query() {
    let client = setup_client().await;

    let sql =
        "SELECT name, number FROM `bigquery-public-data.usa_names.usa_1910_2013` WHERE state = 'WA'";
    let batch = client
        .query(sql)
        .dry_run()
        .await
        .expect("dry_run should succeed");

    assert_eq!(batch.num_rows(), 1);
    let schema = batch.schema();
    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
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

    let bq_schema = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0);
    assert!(bq_schema.contains("name"));
    assert!(bq_schema.contains("number"));

    let total_bytes_processed = batch
        .column(4)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert!(total_bytes_processed > 0);

    let statement_type = batch
        .column(6)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0);
    assert_eq!(statement_type, "SELECT");

    let creation_time = batch
        .column(7)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap()
        .value(0);
    assert!(creation_time > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_query() {
    let client = setup_client().await;

    let sql = "SELECT 1 AS one, 'hello' AS greeting";
    let (schema, receiver) = client
        .query(sql)
        .read()
        .await
        .expect("query read should succeed");

    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(col_names, vec!["one", "greeting"]);
    assert_eq!(consume_receiver(receiver).await, 1);

    let empty_sql = "SELECT 1 AS one, 'hello' AS greeting LIMIT 0";
    let (empty_schema, mut empty_receiver) = client
        .query(empty_sql)
        .with_arrow_buffer_compression(CompressionCodec::Zstd)
        .read()
        .await
        .expect("empty query read should succeed");
    let empty_col_names: Vec<&str> = empty_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(empty_col_names, vec!["one", "greeting"]);
    assert!(empty_receiver.recv().await.is_none());

    let multipage_sql =
        "SELECT * FROM `bigquery-public-data.sec_quarterly_financials.submission` LIMIT 250000";
    let (_, multipage_receiver) = client
        .query(multipage_sql)
        .with_arrow_buffer_compression(CompressionCodec::Zstd)
        .read()
        .await
        .expect("multipage query read should succeed");

    assert_eq!(consume_receiver(multipage_receiver).await, 250_000);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_long_running_query() {
    let client = setup_client().await;

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

    let (schema, receiver) = client
        .query(sql)
        // Set a shorter initial wait so that we exercise the ReadRows code path
        // to wait for this query to finish.
        .with_initial_wait(chrono::Duration::seconds(5))
        .with_use_query_cache(false)
        .read()
        .await
        .expect("long running query read should succeed");

    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(col_names, vec!["id1", "match_count", "max_digest"]);
    assert_eq!(consume_receiver(receiver).await, 10_000);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_long_running_query_oom_fails_with_job_failed() {
    let client = setup_client().await;

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
          ROW_NUMBER() OVER (ORDER BY REPEAT(max_digest, 2048) DESC, id1 ASC) AS rn
        FROM
          stage2
        ORDER BY
          rn ASC
        LIMIT 10000
    "#;

    let err = match client
        .query(sql)
        // Set a shorter initial wait so that we exercise the ReadRows code path
        // to wait for this query to finish.
        .with_initial_wait(chrono::Duration::seconds(5))
        .with_use_query_cache(false)
        .read()
        .await
    {
        Err(err) => err,
        Ok(_) => panic!("expected long-running OOM query to fail"),
    };

    match &err {
        BigQueryError::JobFailed(job_failed) => {
            assert!(job_failed.job_id.is_some());
            assert_eq!(
                job_failed.error.reason.as_deref(),
                Some("resourcesExceeded")
            );
            assert!(job_failed.error.message.contains("Resources exceeded"));
        },
        other => panic!("expected BigQueryError::JobFailed, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_initial_query_oom_failure() {
    let client = setup_client().await;

    let sql = r#"
        WITH
          base AS (
            SELECT
              a * 1000 + b AS id,
              TO_HEX(SHA256(CAST(a * 1000 + b AS STRING))) AS digest
            FROM
              UNNEST(GENERATE_ARRAY(0, 999)) AS a
            CROSS JOIN
              UNNEST(GENERATE_ARRAY(0, 999)) AS b
          )
        SELECT
          id,
          ROW_NUMBER() OVER (ORDER BY REPEAT(digest, 2048) DESC, id ASC) AS rn
        FROM
          base
        ORDER BY
          rn ASC
        LIMIT 10000
    "#;

    let err = match client.query(sql).with_use_query_cache(false).read().await {
        Err(err) => err,
        Ok(_) => panic!("expected initial query OOM to fail"),
    };

    match &err {
        BigQueryError::JobFailed(job_failed) => {
            assert!(
                job_failed.error.message.contains("Resources exceeded"),
                "unexpected error message: {}",
                job_failed.error.message
            );
        },
        other => panic!("expected BigQueryError::JobFailed, got {other:?}"),
    }
    assert!(err.to_string().starts_with("BigQuery job failed"));
    assert!(err.to_string().contains("Resources exceeded"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_long_running_empty_query() {
    let client = setup_client().await;

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
        WHERE
          match_count < 0
    "#;

    let (schema, receiver) = client
        .query(sql)
        // Set a shorter initial wait so that we exercise the ReadRows code path
        // to wait for this query to finish.
        .with_initial_wait(chrono::Duration::seconds(5))
        .with_use_query_cache(false)
        .read()
        .await
        .expect("long running empty query read should succeed");

    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(col_names, vec!["id1", "match_count", "max_digest"]);
    assert_eq!(consume_receiver(receiver).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_ddl_queries() {
    let client = setup_client().await;
    let project_id = client.quota_project_id().to_string();
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dataset_id = format!("_arrow_bq_ddl_test_{suffix}");

    // 1. Short-running DDL query with no schema (CREATE SCHEMA)
    let create_schema_sql = format!(
        "CREATE SCHEMA `{project_id}.{dataset_id}` OPTIONS(default_table_expiration_days=1)"
    );
    let (schema, receiver) = client
        .query(create_schema_sql)
        .read()
        .await
        .expect("CREATE SCHEMA query().read() should succeed");
    assert_eq!(schema.fields().len(), 0);
    assert_eq!(consume_receiver(receiver).await, 0);

    let test_result = async {
        // 2. Short-running DDL query (CREATE TABLE)
        let create_table_sql = format!(
            "CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.test_tbl` (x INT64, y STRING)"
        );
        let (_, receiver) = client
            .query(create_table_sql)
            .read()
            .await
            .expect("CREATE TABLE query().read() should succeed");
        assert_eq!(consume_receiver(receiver).await, 0);

        // 3. Long-running DDL query (CREATE MODEL)
        let create_model_sql = format!(
            "CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.test_model` \
             OPTIONS(model_type='linear_reg', max_iterations=1) AS \
             SELECT CAST(x AS FLOAT64) AS f1, CAST(x * 2 + 1 AS FLOAT64) AS label \
             FROM UNNEST(GENERATE_ARRAY(1, 100)) AS x"
        );
        let (model_schema, model_receiver) = client
            .query(create_model_sql)
            .read()
            .await
            .expect("CREATE MODEL query().read() should succeed");
        assert_eq!(model_schema.fields().len(), 0);
        assert_eq!(consume_receiver(model_receiver).await, 0);
    }
    .await;

    // 4. Always clean up the temporary dataset (DROP SCHEMA ... CASCADE)
    let drop_schema_sql = format!("DROP SCHEMA IF EXISTS `{project_id}.{dataset_id}` CASCADE");
    let (drop_schema, drop_receiver) = client
        .query(drop_schema_sql)
        .read()
        .await
        .expect("DROP SCHEMA query().read() should succeed");
    assert_eq!(drop_schema.fields().len(), 0);
    assert_eq!(consume_receiver(drop_receiver).await, 0);

    let () = test_result;
}
