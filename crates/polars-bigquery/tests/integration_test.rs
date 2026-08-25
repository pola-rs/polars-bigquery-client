use std::env;

use polars_bigquery::*;

#[tokio::test(flavor = "multi_thread")]
async fn test_read_small_public_table() {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    let client = ServiceConfigBuilder::new()
        .with_cred(gcloud_sdk::TokenSourceType::Default)
        .with_user_agent(Some("integration-test/1.0".to_string()))
        .build()
        .await
        .expect("should build client");

    let (_, mut receiver) = read_bigquery_with_client(
        client,
        "bigquery-public-data.usa_names.usa_1910_2013",
        &quota_project_id,
        false,
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
async fn test_client_reuse_channel() {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    let client = Client::from_builder(
        ServiceConfigBuilder::new()
            .with_cred(gcloud_sdk::TokenSourceType::Default)
            .with_user_agent(Some("integration-test/1.0".to_string())),
    )
    .await
    .expect("should build client");

    // Read first table using client
    let (_, mut receiver1) = client
        .read_table(
            "bigquery-public-data.usa_names.usa_1910_2013",
            &quota_project_id,
            false,
        )
        .await
        .expect("first read should succeed");

    let mut total_rows1 = 0;
    while let Some(batch) = receiver1.recv().await {
        total_rows1 += batch.expect("should receive valid record batch").len();
    }
    assert!(total_rows1 > 0);

    // Read second table using the same client (reusing the open channel)
    let (_, mut receiver2) = client
        .read_table(
            "bigquery-public-data.usa_names.usa_1910_2013",
            &quota_project_id,
            false,
        )
        .await
        .expect("second read reusing channel should succeed");

    let mut total_rows2 = 0;
    while let Some(batch) = receiver2.recv().await {
        total_rows2 += batch.expect("should receive valid record batch").len();
    }
    assert!(total_rows2 > 0);
}
