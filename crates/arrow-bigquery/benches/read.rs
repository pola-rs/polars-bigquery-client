use std::env;
use std::sync::{Arc, Once};
use std::time::Duration;

use arrow_bigquery::{
    BigQueryRecordBatchReceiver, BigQueryTableId, Client, ReadOptions, ServiceConfigBuilder,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use tokio::runtime::Runtime;

// The portion of the "numbers" table we are reading is about 5 GB total, so
// these reads vary from about 500 MB to 5 GB.
const PROPORTIONS: [f64; 3] = [0.1, 0.5, 1.0];

static INIT: Once = Once::new();

fn init_console() {
    INIT.call_once(|| {
        // Initializes the console subscriber once for the whole test binary
        console_subscriber::init();
    });
}

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

async fn consume_receiver(mut receiver: BigQueryRecordBatchReceiver) {
    let mut total_rows = 0;
    while let Some(batch) = receiver.recv().await {
        total_rows += batch.expect("should receive valid record batch").len();
    }
    assert!(total_rows > 0);
}

async fn read_sec_quarterly_financials_table(client: &Arc<Client>, table_id: &str) {
    let table = BigQueryTableId {
        project_id: "bigquery-public-data".to_owned(),
        dataset_id: "sec_quarterly_financials".to_owned(),
        table_id: table_id.to_owned(),
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

    consume_receiver(receiver).await;
}

async fn read_numbers_table(client: &Arc<Client>, proportion: f64) {
    let table = BigQueryTableId {
        project_id: "bigquery-public-data".to_owned(),
        dataset_id: "sec_quarterly_financials".to_owned(),
        table_id: "numbers".to_owned(),
    };
    let (_, receiver) = client
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

    consume_receiver(receiver).await;
}

fn read_table(c: &mut Criterion) {
    init_console();
    let rt = Runtime::new().unwrap();
    let client = Arc::new(rt.block_on(async { setup_client().await }));

    let mut group = c.benchmark_group("read_table");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    group.measurement_time(Duration::from_secs_f64(15.0));
    group.bench_function("sec_quarterly_financials_sic_codes", |b| {
        let client = &client.clone();
        b.to_async(&rt)
            .iter(|| async move { read_sec_quarterly_financials_table(client, "sic_codes").await });
    });

    group.measurement_time(Duration::from_secs_f64(15.0));
    group.bench_function("sec_quarterly_financials_submission", |b| {
        let client = &client.clone();
        b.to_async(&rt).iter(|| async move {
            read_sec_quarterly_financials_table(client, "submission").await
        });
    });

    for (proportion, target_secs) in PROPORTIONS.into_iter().zip([15.0, 500.0, 750.0]) {
        group.measurement_time(Duration::from_secs_f64(target_secs));
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

async fn execute_query(client: &Arc<Client>, sql: &str, maintain_order: bool) {
    let (_, receiver) = client
        .query(sql)
        .with_maintain_order(maintain_order)
        .read()
        .await
        .expect("query read should work with default credentials");

    consume_receiver(receiver).await;
}

fn format_sec_quarterly_financials_query(table_id: &str) -> String {
    format!("SELECT * FROM `bigquery-public-data.sec_quarterly_financials.{table_id}`")
}

fn format_numbers_query(proportion: f64) -> String {
    format!(
        r#"
        SELECT *
        FROM `bigquery-public-data.sec_quarterly_financials.numbers`
        WHERE
            TIMESTAMP_TRUNC(_PARTITIONTIME, DAY)
                BETWEEN TIMESTAMP("2019-01-01") AND TIMESTAMP("2019-12-31")
            AND ((FARM_FINGERPRINT(submission_number) / POW(2.0, 64)) + 0.5) < {proportion}
        "#,
    )
}

fn read_query(c: &mut Criterion) {
    init_console();
    let rt = Runtime::new().unwrap();
    let client = Arc::new(rt.block_on(async { setup_client().await }));

    let mut group = c.benchmark_group("read_query");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    group.measurement_time(Duration::from_secs_f64(10.0));
    group.bench_function("select_1", |b| {
        let client = &client;
        b.to_async(&rt)
            .iter(|| async move { execute_query(client, "SELECT 1", true).await });
    });

    let sic_codes_sql = format_sec_quarterly_financials_query("sic_codes");
    group.measurement_time(Duration::from_secs_f64(10.0));
    group.bench_function("sec_quarterly_financials_sic_codes", |b| {
        let client = &client;
        let sql = sic_codes_sql.as_str();
        b.to_async(&rt)
            .iter(|| async move { execute_query(client, sql, true).await });
    });

    let submission_sql = format_sec_quarterly_financials_query("submission");
    group.measurement_time(Duration::from_secs_f64(100.0));
    group.bench_function("sec_quarterly_financials_submission", |b| {
        let client = &client;
        let sql = submission_sql.as_str();
        b.to_async(&rt)
            .iter(|| async move { execute_query(client, sql, true).await });
    });

    for (proportion, target_secs) in PROPORTIONS.into_iter().zip([150.0, 500.0, 800.0]) {
        let sql = format_numbers_query(proportion);
        group.measurement_time(Duration::from_secs_f64(target_secs));
        group.bench_with_input(
            BenchmarkId::new("sec_quarterly_financials_numbers", proportion),
            &sql,
            |b, sql| {
                let client = &client;
                b.to_async(&rt)
                    .iter(|| async move { execute_query(client, sql, false).await });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, read_table, read_query);
criterion_main!(benches);
