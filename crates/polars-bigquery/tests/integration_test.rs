use polars_bigquery::*;

#[tokio::test(flavor = "multi_thread")]
async fn test_read_small_public_table() {
    let result = read_bigquery_async(
        "bigquery-public-data.usa_names.usa_1910_2013",
        "some-project",
        false,
        gcloud_sdk::TokenSourceType::Default,
    )
    .await;
    assert!(result.is_ok());
}
