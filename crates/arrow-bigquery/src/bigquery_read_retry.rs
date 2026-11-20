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
const JOB_SERVICE_MAX_ATTEMPTS: u32 = 5;
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

/// Retry policy for BigQuery v2 `JobService` (`jobs.query`, `jobs.get`, `jobs.cancel`) requests.
///
/// Retries idempotent transient network/transport errors, HTTP 429/500/502/503/504,
/// and transient RPC status codes (`ResourceExhausted`, `Unavailable`, `Internal`,
/// `DeadlineExceeded`, `Aborted`) up to a maximum of 5 total attempts and a 600s
/// cumulative time budget. Capping attempts avoids repeating long-running `jobs.query`
/// waits indefinitely across transient timeouts.
pub(crate) fn job_service_retry_policy() -> Arc<dyn RetryPolicy> {
    Arc::new(
        RetryableErrors
            .with_time_limit(CREATE_READ_SESSION_TIMEOUT)
            .with_attempt_limit(JOB_SERVICE_MAX_ATTEMPTS),
    )
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

    fn transient_error() -> Error {
        Error::service(Status::default().set_code(Code::Unavailable))
    }

    fn permanent_error() -> Error {
        Error::service(Status::default().set_code(Code::PermissionDenied))
    }

    #[test]
    fn test_job_service_retry_policy_attempt_and_time_limits() {
        let policy = job_service_retry_policy();
        let state = RetryState::new(true);

        let remaining = policy
            .remaining_time(&state)
            .expect("job_service_retry_policy must have a time limit");
        assert!(remaining <= CREATE_READ_SESSION_TIMEOUT);
        assert!(remaining > Duration::from_secs(590));

        // Permanent errors terminate immediately without retrying.
        assert!(policy
            .on_error(&state.clone().set_attempt_count(1_u32), permanent_error())
            .is_permanent());

        // Non-idempotent requests are not retried for service errors.
        assert!(policy
            .on_error(
                &RetryState::new(false).set_attempt_count(1_u32),
                transient_error(),
            )
            .is_permanent());

        // Transient idempotent errors retry up to JOB_SERVICE_MAX_ATTEMPTS total attempts.
        for attempt in 1..JOB_SERVICE_MAX_ATTEMPTS {
            assert!(
                policy
                    .on_error(&state.clone().set_attempt_count(attempt), transient_error())
                    .is_continue(),
                "expected attempt {attempt} to continue"
            );
        }
        assert!(policy
            .on_error(
                &state.clone().set_attempt_count(JOB_SERVICE_MAX_ATTEMPTS),
                transient_error(),
            )
            .is_exhausted());

        // Also exhausts once the cumulative time limit has expired.
        let expired_state = state.set_start(
            std::time::Instant::now()
                .checked_sub(CREATE_READ_SESSION_TIMEOUT + Duration::from_secs(1))
                .expect("instant subtraction should succeed"),
        );
        assert!(policy
            .on_error(&expired_state.set_attempt_count(1_u32), transient_error())
            .is_exhausted());
    }
}
