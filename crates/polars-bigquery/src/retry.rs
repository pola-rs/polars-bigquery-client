use core::time::Duration;

use backon::ExponentialBuilder;
use gcloud_sdk::tonic;

/// Retry configuration for create_read_session.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L148-L162
const CREATE_READ_SESSION_RETRY: ExponentialBuilder = ExponentialBuilder::new()
    .with_min_delay(Duration::from_millis(100))
    .with_max_delay(Duration::from_secs(60))
    .with_factor(1.3)
    .with_total_delay(Some(Duration::from_secs(600)))
    .with_jitter();

/// When to retry create_read_session requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L154-L157
pub fn create_read_session_predicate(err: &tonic::Status) -> bool {
    match err.code() {
        tonic::Code::DeadlineExceeded => true,
        tonic::Code::Unavailable => true,
        _ => false,
    }
}

/// Retry configuration for read_rows.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L163-L176
const READ_ROWS_RETRY: ExponentialBuilder = ExponentialBuilder::new()
    .with_min_delay(Duration::from_millis(100))
    .with_max_delay(Duration::from_secs(60))
    .with_factor(1.3)
    .with_total_delay(Some(Duration::from_hours(24)))
    .with_jitter();

/// When to retry read_rows requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L169-L171
pub fn read_rows_predicate(err: &tonic::Status) -> bool {
    match err.code() {
        tonic::Code::Unavailable => true,
        _ => false,
    }
}
