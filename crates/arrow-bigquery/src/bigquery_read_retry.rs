//! Retry and backoff policies for the BigQuery Storage Read API.
//!
//! Adapts the BigQuery Storage Read API retry parameters and reconnection session
//! tracking to [`google_cloud_gax`] policies (`RetryPolicy`, `BackoffPolicy`,
//! `SharedRetryThrottler`).

use std::sync::Arc;
use std::time::Duration;

use google_cloud_gax::backoff_policy::BackoffPolicy;
use google_cloud_gax::error::rpc::Code;
use google_cloud_gax::error::Error;
use google_cloud_gax::exponential_backoff::ExponentialBackoffBuilder;
use google_cloud_gax::retry_policy::{RetryPolicy, RetryPolicyExt};
use google_cloud_gax::retry_result::RetryResult;
use google_cloud_gax::retry_state::RetryState;
use google_cloud_gax::retry_throttler::SharedRetryThrottler;

const INITIAL_DELAY: Duration = Duration::from_millis(100);
const MAXIMUM_DELAY: Duration = Duration::from_secs(60);
const SCALING_FACTOR: f64 = 1.3;
const CREATE_READ_SESSION_TIMEOUT: Duration = Duration::from_secs(600);
const STREAM_RECONNECT_MAX_ATTEMPTS: u32 = 10;

/// Follows the RPC retry strategy recommended for the BigQuery Storage Read API.
///
/// Retries transient network/transport errors, HTTP/2 stream resets, rate-limit
/// errors (`ResourceExhausted`), and transient server statuses (`DeadlineExceeded`,
/// `Aborted`, `ResourceExhausted`, `Unavailable`, `Internal`) when the operation
/// is idempotent.
///
/// This policy must be decorated with [`RetryPolicyExt::with_time_limit`] or
/// [`RetryPolicyExt::with_attempt_limit`] to bound the retry loop.
#[derive(Clone, Debug, Default)]
pub(crate) struct RetryableErrors;

impl RetryPolicy for RetryableErrors {
    fn on_error(&self, state: &RetryState, error: Error) -> RetryResult {
        if is_retryable_error(&error, state.idempotent) {
            RetryResult::Continue(error)
        } else {
            RetryResult::Permanent(error)
        }
    }
}

pub(crate) fn is_retryable_error(error: &Error, idempotent: bool) -> bool {
    if error.is_transient_and_before_rpc() {
        return true;
    }
    if !idempotent {
        return false;
    }
    if error.is_io() || error.is_timeout() || error.is_connect() {
        return true;
    }
    if error.is_transport() && error.http_status_code().is_none() {
        return true;
    }
    if let Some(429 | 500 | 502 | 503 | 504) = error.http_status_code() {
        return true;
    }
    if let Some(status) = error.status() {
        return matches!(
            status.code,
            Code::DeadlineExceeded
                | Code::Aborted
                | Code::ResourceExhausted
                // Unavailable includes common transport-level failures such as
                // Hyper/h2 connection errors and ConnectionReset.
                | Code::Unavailable
                // Internal includes common transport-level failures such as
                // broken pipe / RST_STREAM.
                | Code::Internal
        );
    }
    false
}

/// When to retry `create_read_session` requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L154-L157
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn create_read_session_predicate(err: &Error) -> bool {
    is_retryable_error(err, true)
}

/// When to retry `read_rows` requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L169-L171
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn read_rows_predicate(err: &Error) -> bool {
    is_retryable_error(err, true)
}

/// When to reconnect/resume an active `read_rows` stream after encountering a gRPC error mid-read.
///
/// While currently identical to [`read_rows_predicate`], having a dedicated predicate allows fine-tuning
/// reconnection behavior separately from initial request establishment.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn stream_reconnect_predicate(err: &Error) -> bool {
    is_retryable_error(err, true)
}

/// Default exponential backoff policy for BigQuery Storage Read API operations.
///
/// - `initial_delay` (100ms): Initial sleep duration before first retry.
/// - `maximum_delay` (60s): Upper bound cap on any single exponential backoff sleep.
/// - `scaling` (1.3): Multiplier scaling factor for exponential backoff (matching Python BigQuery Storage client standard).
pub(crate) fn default_backoff_policy() -> Arc<dyn BackoffPolicy> {
    Arc::new(
        ExponentialBackoffBuilder::new()
            .with_initial_delay(INITIAL_DELAY)
            .with_maximum_delay(MAXIMUM_DELAY)
            .with_scaling(SCALING_FACTOR)
            .build()
            .expect("hardcoded value guaranteed to be valid"),
    )
}

/// Retry policy for `create_read_session` requests.
///
/// - `maximum_duration` (600s): Maximum cumulative duration across retries before giving up.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L148-L162
pub(crate) fn create_read_session_retry_policy() -> Arc<dyn RetryPolicy> {
    Arc::new(RetryableErrors.with_time_limit(CREATE_READ_SESSION_TIMEOUT))
}

/// Retry policy for mid-stream `read_rows` reconnections.
///
/// - `maximum_attempts` (10): Limits total consecutive failed reconnection attempts when no data progress is made.
pub(crate) fn stream_reconnect_retry_policy() -> Arc<dyn RetryPolicy> {
    Arc::new(RetryableErrors.with_attempt_limit(STREAM_RECONNECT_MAX_ATTEMPTS))
}

/// Retry and backoff configuration shared across stream tasks.
#[derive(Clone, Debug)]
pub(crate) struct StreamRetryConfig {
    pub(crate) retry_policy: Arc<dyn RetryPolicy>,
    pub(crate) backoff_policy: Arc<dyn BackoffPolicy>,
    pub(crate) retry_throttler: SharedRetryThrottler,
}

impl StreamRetryConfig {
    pub(crate) fn new(
        retry_policy: Option<Arc<dyn RetryPolicy>>,
        backoff_policy: Option<Arc<dyn BackoffPolicy>>,
        retry_throttler: SharedRetryThrottler,
    ) -> Self {
        Self {
            retry_policy: retry_policy.unwrap_or_else(stream_reconnect_retry_policy),
            backoff_policy: backoff_policy.unwrap_or_else(default_backoff_policy),
            retry_throttler,
        }
    }
}

#[cfg(test)]
mod tests {
    use google_cloud_gax::error::rpc::Status;

    use super::*;

    fn status_err(code: Code) -> Error {
        Error::service(Status::default().set_code(code).set_message("test status"))
    }

    #[test]
    fn test_reconnect_stream_predicate_retryable_codes() {
        let retryable_codes = [
            Code::DeadlineExceeded,
            Code::Unavailable,
            Code::Aborted,
            Code::Internal,
            Code::ResourceExhausted,
        ];

        for code in retryable_codes {
            let err = status_err(code);
            assert!(
                stream_reconnect_predicate(&err),
                "Expected stream_reconnect_predicate to be true for code {:?}",
                code
            );
        }
    }

    #[test]
    fn test_reconnect_stream_predicate_non_retryable_codes() {
        let non_retryable_codes = [
            Code::Ok,
            Code::Cancelled,
            Code::Unknown,
            Code::InvalidArgument,
            Code::NotFound,
            Code::AlreadyExists,
            Code::PermissionDenied,
            Code::FailedPrecondition,
            Code::OutOfRange,
            Code::Unimplemented,
            Code::DataLoss,
            Code::Unauthenticated,
        ];

        for code in non_retryable_codes {
            let err = status_err(code);
            assert!(
                !stream_reconnect_predicate(&err),
                "Expected stream_reconnect_predicate to be false for code {:?}",
                code
            );
        }
    }

    #[test]
    fn test_read_rows_predicate_matches_reconnect() {
        let codes = [
            Code::Unavailable,
            Code::Aborted,
            Code::NotFound,
            Code::InvalidArgument,
        ];
        for code in codes {
            let err = status_err(code);
            assert_eq!(read_rows_predicate(&err), stream_reconnect_predicate(&err));
            assert_eq!(
                create_read_session_predicate(&err),
                stream_reconnect_predicate(&err)
            );
        }
    }
}
