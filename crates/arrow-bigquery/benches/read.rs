use std::env;
use std::sync::Arc;

use arrow_bigquery::{
    BigQueryReadClientBuilder, BigQueryTableId, Client, ReadOptions, ServiceConfigBuilder,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use tokio::runtime::Runtime;

async fn setup_client() -> Client {
    let quota_project_id = env::var("GOOGLE_CLOUD_PROJECT")
        .expect("must set GOOGLE_CLOUD_PROJECT to run integration tests");

    Client::from_builder(
        ServiceConfigBuilder::new()
            .with_cred(gcloud_sdk::TokenSourceType::Default)
            .with_user_agent(Some("integration-test/1.0".to_string()))
            .with_quota_project_id(Some(quota_project_id)),
    )
    .await
    .expect("should build client")
}

async fn read_sec_quarterly_financials_table(client: &Arc<Client>, table_id: &str) {
    let table = BigQueryTableId {
        project_id: "bigquery-public-data".to_owned(),
        dataset_id: "sec_quarterly_financials".to_owned(),
        table_id: table_id.to_owned(),
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

async fn read_numbers_table(client: &Arc<Client>, proportion: f64) {
    let table = BigQueryTableId {
        project_id: "bigquery-public-data".to_owned(),
        dataset_id: "sec_quarterly_financials".to_owned(),
        table_id: "numbers".to_owned(),
    };
    let (_, mut receiver) = client
                        .read_table(
                            &table,
                            ReadOptions {
                                maintain_order: false,
                                row_restriction: format!(
                                    r#"
                                    TIMESTAMP_TRUNC(_PARTITIONTIME, DAY)
                                        BETWEEN TIMESTAMP("2019-01-01") AND TIMESTAMP("2019-12-31")
                                    AND ((FARM_FINGERPRINT(submission_number) / POW(2.0, 64)) + 0.5) < {proportion}
                                    "#,
                                    proportion = proportion),
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

fn read_table(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let client = Arc::new(rt.block_on(async { setup_client().await }));

    let mut group = c.benchmark_group("read_table");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    group.bench_function("sec_quarterly_financials_sic_codes", |b| {
        let client = &client.clone();
        b.to_async(&rt)
            .iter(|| async move { read_sec_quarterly_financials_table(client, "sic_codes").await });
    });

    group.bench_function("sec_quarterly_financials_submission", |b| {
        let client = &client.clone();
        b.to_async(&rt)
            .iter(|| async move { read_sec_quarterly_financials_table(client, "submission").await });
    });

    // The portion of the "numbers" table we are reading is about 5 GB total, so
    // these reads vary from about 500 MB to 5 GB.
    let proportions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    for proportion in proportions {
        group.bench_with_input(
            BenchmarkId::new("sec_quarterly_financials_numbers", proportion),
            &proportion,
            |b, &proportion| {
                let client = &client.clone();
                b.to_async(&rt)
                    .iter(|| async move { read_numbers_table(client, proportion).await });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, read_table);
criterion_main!(benches);
