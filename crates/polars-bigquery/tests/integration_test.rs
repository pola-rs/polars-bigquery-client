use std::env;

use polars_bigquery::*;

#[tokio::test(flavor = "multi_thread")]
async fn test_read_small_public_table() {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    let client = PolarsBigQueryClientBuilder::new()
        .with_token_source(gcloud_sdk::TokenSourceType::Default)
        .with_user_agent("integration-test/1.0".to_string())
        .build()
        .await
        .expect("should build client");

    let result = read_bigquery_with_client(
        client,
        "bigquery-public-data.usa_names.usa_1910_2013",
        &quota_project_id,
        false,
    )
    .await;

    result.expect("public table read should work with default credentials");
}
