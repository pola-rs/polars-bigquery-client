//! Retry and backoff policies for BigQuery Storage Read API (gRPC) and
//! BigQuery v2 `JobService` (REST) requests.
//!
//! Adapts both transports' retry parameters and reconnection session tracking
//! to [`google_cloud_gax`] policies (`RetryPolicy`, `BackoffPolicy`,
//! `SharedRetryThrottler`).

use std::sync::Arc;
use std::time::Duration;

use google_cloud_gax::backoff_policy::BackoffPolicy;
use google_cloud_gax::error::rpc::{Code, StatusDetails};
use google_cloud_gax::error::Error;
use google_cloud_gax::exponential_backoff::ExponentialBackoffBuilder;
use google_cloud_gax::retry_policy::{RetryPolicy, RetryPolicyExt};
use google_cloud_gax::retry_result::RetryResult;
use google_cloud_gax::retry_state::RetryState;
use google_cloud_gax::retry_throttler::SharedRetryThrottler;
use google_cloud_gax::throttle_result::ThrottleResult;

// Storage Read API (gRPC) retry/backoff constants.
const INITIAL_DELAY: Duration = Duration::from_millis(100);
const MAXIMUM_DELAY: Duration = Duration::from_secs(60);
const SCALING_FACTOR: f64 = 1.3;
const CREATE_READ_SESSION_TIMEOUT: Duration = Duration::from_secs(600);
const STREAM_RECONNECT_MAX_ATTEMPTS: u32 = 10;

// BigQuery v2 JobService (REST) retry/backoff constants.
const JOB_SERVICE_INITIAL_DELAY: Duration = Duration::from_secs(1);
const JOB_SERVICE_MAXIMUM_DELAY: Duration = Duration::from_secs(60);
const JOB_SERVICE_SCALING_FACTOR: f64 = 2.0;
pub(crate) const JOB_SERVICE_RETRY_TIMEOUT: Duration = Duration::from_secs(600);
const JOB_SERVICE_MAX_ATTEMPTS: u32 = 5;

// Best-effort `jobs.cancel` retry/backoff constants (bounded inside `JOB_CANCEL_TIMEOUT`).
const JOB_CANCEL_INITIAL_DELAY: Duration = Duration::from_millis(100);
const JOB_CANCEL_MAXIMUM_DELAY: Duration = Duration::from_secs(2);
const JOB_CANCEL_SCALING_FACTOR: f64 = 2.0;
pub(crate) const JOB_CANCEL_MAX_ATTEMPTS: u32 = 3;

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

/// Retry policy for BigQuery v2 `JobService` REST requests (`jobs.query`, `jobs.get`,
/// and `jobs.cancel`).
///
/// Extends [`RetryableErrors`] with BigQuery REST-specific transient errors:
/// - HTTP `408 Request Timeout`
/// - HTTP `403` with BigQuery `ErrorInfo.reason` of `"rateLimitExceeded"`,
///   `"userRateLimitExceeded"`, `"backendError"`, or `"internalError"`
///
/// Fails fast when throttled (`on_throttle` returns [`ThrottleResult::Exhausted`])
/// so an open circuit breaker stops retry storms immediately instead of sleeping
/// across the full `JOB_SERVICE_RETRY_TIMEOUT` budget.
#[derive(Clone, Debug, Default)]
pub(crate) struct JobServiceRetryableErrors;

impl RetryPolicy for JobServiceRetryableErrors {
    fn on_error(&self, state: &RetryState, error: Error) -> RetryResult {
        if is_job_service_retryable_error(&error, state.idempotent) {
            RetryResult::Continue(error)
        } else {
            RetryResult::Permanent(error)
        }
    }

    fn on_throttle(&self, _state: &RetryState, error: Error) -> ThrottleResult {
        ThrottleResult::Exhausted(error)
    }
}

fn has_retryable_bigquery_rest_reason(error: &Error) -> bool {
    let Some(status) = error.status() else {
        return false;
    };
    status.details.iter().any(|detail| {
        matches!(
            detail,
            StatusDetails::ErrorInfo(info)
                if matches!(
                    info.reason.as_str(),
                    "rateLimitExceeded"
                        | "userRateLimitExceeded"
                        | "backendError"
                        | "internalError"
                )
        )
    })
}

/// Returns `true` when a BigQuery v2 REST error carries a transient HTTP status
/// code (`408`, `429`, `500`, `502`, `503`, `504`) or a legacy HTTP `403`
/// rate-limit / backend error reason (`rateLimitExceeded`,
/// `userRateLimitExceeded`, `backendError`, `internalError`).
pub(crate) fn is_transient_job_service_http_error(error: &Error) -> bool {
    match error.http_status_code() {
        Some(408 | 429 | 500 | 502 | 503 | 504) => true,
        Some(403) => has_retryable_bigquery_rest_reason(error),
        _ => false,
    }
}

pub(crate) fn is_job_service_retryable_error(error: &Error, idempotent: bool) -> bool {
    if error.is_transient_and_before_rpc() {
        return true;
    }
    if !idempotent {
        return false;
    }
    if is_transient_job_service_http_error(error) {
        return true;
    }
    is_retryable_error(error, true)
}

/// Exponential backoff policy for BigQuery v2 `JobService` REST requests (`jobs.query`, `jobs.get`).
///
/// Uses full jitter over `1s * 2.0^(attempt-1)` capped at `60s` (`1s, 2s, 4s, 8s`
/// across the default 5-attempt cap, ~15s max backoff), avoiding burning all retries
/// inside sub-second bursts like the Storage Read API's `100ms / 1.3x` curve.
pub(crate) fn job_service_backoff_policy() -> Arc<dyn BackoffPolicy> {
    Arc::new(
        ExponentialBackoffBuilder::new()
            .with_initial_delay(JOB_SERVICE_INITIAL_DELAY)
            .with_maximum_delay(JOB_SERVICE_MAXIMUM_DELAY)
            .with_scaling(JOB_SERVICE_SCALING_FACTOR)
            .build()
            .expect("hardcoded value guaranteed to be valid"),
    )
}

/// Fast exponential backoff policy for detached best-effort `jobs.cancel` retries.
///
/// Uses full jitter over `100ms * 2.0^(attempt-1)` capped at `2s`, keeping retries
/// inside the short `JOB_CANCEL_TIMEOUT` (10s) window.
pub(crate) fn job_service_cancel_backoff_policy() -> Arc<dyn BackoffPolicy> {
    Arc::new(
        ExponentialBackoffBuilder::new()
            .with_initial_delay(JOB_CANCEL_INITIAL_DELAY)
            .with_maximum_delay(JOB_CANCEL_MAXIMUM_DELAY)
            .with_scaling(JOB_CANCEL_SCALING_FACTOR)
            .build()
            .expect("hardcoded value guaranteed to be valid"),
    )
}

/// Retry policy for BigQuery v2 `JobService` (`jobs.query`, `jobs.get`) requests.
///
/// Retries idempotent transient network/transport errors, HTTP 408/429/500/502/503/504,
/// legacy HTTP 403 `rateLimitExceeded`/`userRateLimitExceeded`/`backendError`/`internalError`,
/// and transient RPC status codes (`ResourceExhausted`, `Unavailable`, `Internal`,
/// `DeadlineExceeded`, `Aborted`) up to a maximum of 5 total attempts and a `600s`
/// cumulative time budget. Capping attempts avoids repeating long-running `jobs.query`
/// waits indefinitely across transient timeouts.
///
/// For detached `jobs.cancel` cleanup requests, use [`job_service_cancel_retry_policy`].
pub(crate) fn job_service_retry_policy() -> Arc<dyn RetryPolicy> {
    job_service_retry_policy_with_time_limit(JOB_SERVICE_RETRY_TIMEOUT)
}

/// Variant of [`job_service_retry_policy`] with a caller-specified total time budget
/// (used when a query sets `initial_wait` near or beyond [`JOB_SERVICE_RETRY_TIMEOUT`]).
pub(crate) fn job_service_retry_policy_with_time_limit(
    time_limit: Duration,
) -> Arc<dyn RetryPolicy> {
    Arc::new(
        JobServiceRetryableErrors
            .with_time_limit(time_limit)
            .with_attempt_limit(JOB_SERVICE_MAX_ATTEMPTS),
    )
}

/// Retry policy for detached best-effort `jobs.cancel` requests.
///
/// Bounded to [`JOB_CANCEL_MAX_ATTEMPTS`] (3 total attempts) within `time_limit` so
/// a single stale TCP reset or transient `503` does not silently abandon a running
/// BigQuery query job, without inheriting `JobService`'s 600s retry budget.
pub(crate) fn job_service_cancel_retry_policy(time_limit: Duration) -> Arc<dyn RetryPolicy> {
    Arc::new(
        JobServiceRetryableErrors
            .with_time_limit(time_limit)
            .with_attempt_limit(JOB_CANCEL_MAX_ATTEMPTS),
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

    fn http_403_with_reason(reason: &str) -> Error {
        let body = serde_json::json!({
            "error": {
                "code": 403,
                "message": "err",
                "errors": [{"reason": reason, "domain": "global", "message": "err"}],
                "status": "PERMISSION_DENIED"
            }
        });
        let status = Status::try_from(&bytes::Bytes::from(body.to_string()))
            .expect("status json should parse");
        Error::service_with_http_metadata(status, Some(403), None)
    }

    #[tokio::test(start_paused = true)]
    async fn test_job_service_retry_policy_attempt_and_time_limits() {
        let policy = job_service_retry_policy();
        let start = tokio::time::Instant::now().into_std();
        let state = RetryState::new(true).set_start(start);

        let remaining = policy
            .remaining_time(&state)
            .expect("job_service_retry_policy must have a time limit");
        assert!(remaining <= JOB_SERVICE_RETRY_TIMEOUT);
        assert!(remaining > Duration::from_secs(590));

        // Permanent errors terminate immediately without retrying.
        assert!(policy
            .on_error(&state.clone().set_attempt_count(1_u32), permanent_error())
            .is_permanent());

        // Non-idempotent requests are not retried for service errors.
        assert!(policy
            .on_error(
                &RetryState::new(false)
                    .set_start(start)
                    .set_attempt_count(1_u32),
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

        // Throttled retries fail fast instead of spinning until the time limit expires.
        assert!(matches!(
            policy.on_throttle(&state.clone().set_attempt_count(1_u32), transient_error()),
            ThrottleResult::Exhausted(_)
        ));

        // Advance Tokio's paused clock forward past the budget to verify time exhaustion
        // without subtracting from the OS monotonic boot epoch (safe on fresh Windows VMs).
        tokio::time::advance(JOB_SERVICE_RETRY_TIMEOUT + Duration::from_secs(1)).await;
        assert!(policy
            .on_error(&state.clone().set_attempt_count(1_u32), transient_error())
            .is_exhausted());
    }

    #[test]
    fn test_job_service_retry_classification_and_backoff() {
        let policy = job_service_retry_policy();
        let state = RetryState::new(true).set_attempt_count(1_u32);

        // HTTP 408 Request Timeout is retryable.
        let timeout_408 = Error::http(408, Default::default(), bytes::Bytes::new());
        assert!(policy.on_error(&state, timeout_408).is_continue());

        // Legacy BigQuery REST HTTP 403 rate-limit / backend reasons are retryable.
        for reason in [
            "rateLimitExceeded",
            "userRateLimitExceeded",
            "backendError",
            "internalError",
        ] {
            assert!(
                policy
                    .on_error(&state, http_403_with_reason(reason))
                    .is_continue(),
                "expected 403 {reason} to be retryable"
            );
        }

        // Genuine 403 permission errors are permanent.
        assert!(policy
            .on_error(&state, http_403_with_reason("accessDenied"))
            .is_permanent());
        assert!(policy
            .on_error(&state, http_403_with_reason("billingNotEnabled"))
            .is_permanent());

        // Verify backoff bounds for JobService (1s * 2^(attempt-1)).
        let backoff = job_service_backoff_policy();
        let d1 = backoff.on_failure(&RetryState::new(true).set_attempt_count(1_u32));
        let d4 = backoff.on_failure(&RetryState::new(true).set_attempt_count(4_u32));
        assert!(d1 <= Duration::from_secs(1));
        assert!(d4 <= Duration::from_secs(8));

        // Cancel retry policy is capped at JOB_CANCEL_MAX_ATTEMPTS (3).
        let cancel_policy = job_service_cancel_retry_policy(Duration::from_secs(10));
        for attempt in 1..JOB_CANCEL_MAX_ATTEMPTS {
            assert!(cancel_policy
                .on_error(
                    &RetryState::new(true).set_attempt_count(attempt),
                    transient_error()
                )
                .is_continue());
        }
        assert!(cancel_policy
            .on_error(
                &RetryState::new(true).set_attempt_count(JOB_CANCEL_MAX_ATTEMPTS),
                transient_error(),
            )
            .is_exhausted());
    }
}
