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
use google_cloud_gax::throttle_result::ThrottleResult;

const INITIAL_DELAY: Duration = Duration::from_millis(100);
const MAXIMUM_DELAY: Duration = Duration::from_secs(60);
const SCALING_FACTOR: f64 = 1.3;
const CREATE_READ_SESSION_TIMEOUT: Duration = Duration::from_secs(600);
const READ_ROWS_TIMEOUT: Duration = Duration::from_secs(900);
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

/// Retry policy for initial `read_rows` stream establishment requests.
///
/// - `maximum_duration` (900s): Maximum cumulative duration across retries before giving up.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L163-L176
pub(crate) fn read_rows_retry_policy() -> Arc<dyn RetryPolicy> {
    Arc::new(RetryableErrors.with_time_limit(READ_ROWS_TIMEOUT))
}

/// Retry policy for mid-stream `read_rows` reconnections.
///
/// - `maximum_attempts` (10): Limits total consecutive failed reconnection attempts when no data progress is made.
///
/// Important! If data progress is made (`current_offset > prev_offset`), the session
/// must be reset to grant a fresh 10-attempt allowance.
pub(crate) fn stream_reconnect_retry_policy() -> Arc<dyn RetryPolicy> {
    Arc::new(RetryableErrors.with_attempt_limit(STREAM_RECONNECT_MAX_ATTEMPTS))
}

/// Retry and backoff configuration shared across stream tasks.
#[derive(Clone, Debug)]
pub(crate) struct StreamRetryConfig {
    connect_retry_policy: Arc<dyn RetryPolicy>,
    reconnect_retry_policy: Arc<dyn RetryPolicy>,
    backoff_policy: Arc<dyn BackoffPolicy>,
    retry_throttler: SharedRetryThrottler,
}

impl StreamRetryConfig {
    pub(crate) fn new(
        retry_policy: Option<Arc<dyn RetryPolicy>>,
        backoff_policy: Option<Arc<dyn BackoffPolicy>>,
        retry_throttler: SharedRetryThrottler,
    ) -> Self {
        let connect_retry_policy = retry_policy.clone().unwrap_or_else(read_rows_retry_policy);
        let reconnect_retry_policy = retry_policy.unwrap_or_else(stream_reconnect_retry_policy);
        let backoff_policy = backoff_policy.unwrap_or_else(default_backoff_policy);
        Self {
            connect_retry_policy,
            reconnect_retry_policy,
            backoff_policy,
            retry_throttler,
        }
    }

    /// Creates a new [`BackoffSession`] for initial `read_rows` stream establishment (Layer 1).
    pub(crate) fn make_connect_session(&self) -> BackoffSession {
        BackoffSession::new(
            Arc::clone(&self.connect_retry_policy),
            Arc::clone(&self.backoff_policy),
            self.retry_throttler.clone(),
        )
    }

    /// Creates a new [`BackoffSession`] for mid-stream reconnections (Layer 3).
    pub(crate) fn make_reconnect_session(&self) -> BackoffSession {
        BackoffSession::new(
            Arc::clone(&self.reconnect_retry_policy),
            Arc::clone(&self.backoff_policy),
            self.retry_throttler.clone(),
        )
    }
}

/// Tracks active retry and backoff state (attempt count, start time, and throttler interaction)
/// for a single connection or stream reconnection session.
#[derive(Clone, Debug)]
pub(crate) struct BackoffSession {
    retry_policy: Arc<dyn RetryPolicy>,
    backoff_policy: Arc<dyn BackoffPolicy>,
    retry_throttler: SharedRetryThrottler,
    retry_state: RetryState,
}

impl BackoffSession {
    pub(crate) fn new(
        retry_policy: Arc<dyn RetryPolicy>,
        backoff_policy: Arc<dyn BackoffPolicy>,
        retry_throttler: SharedRetryThrottler,
    ) -> Self {
        Self {
            retry_policy,
            backoff_policy,
            retry_throttler,
            retry_state: RetryState::new(true),
        }
    }

    /// Records a successful RPC or stream message with the shared retry throttler.
    pub(crate) fn record_success(&self) {
        self.retry_throttler
            .lock()
            .expect("retry throttler lock is poisoned")
            .on_success();
    }

    /// Resets the retry attempt count and start time after data progress is made,
    /// granting future transient errors a fresh retry budget.
    pub(crate) fn reset(&mut self) {
        self.record_success();
        self.retry_state = RetryState::new(true);
    }

    /// Evaluates `err` against the retry policy, sleeps for the computed backoff delay
    /// (unless cancelled via `cancel_rx`), and consults the shared retry throttler.
    ///
    /// Returns:
    /// - `Ok(Some(()))` if the caller should proceed with the next retry attempt.
    /// - `Ok(None)` if cancelled via `cancel_rx` while backing off.
    /// - `Err(Error)` if the error is non-retryable or the retry policy / throttler is exhausted.
    pub(crate) async fn next_delay(
        &mut self,
        err: Error,
        cancel_rx: &mut tokio::sync::watch::Receiver<bool>,
    ) -> Result<Option<()>, Error> {
        if *cancel_rx.borrow_and_update() {
            return Ok(None);
        }

        self.retry_state.attempt_count += 1;
        let flow = self.retry_policy.on_error(&self.retry_state, err);
        self.retry_throttler
            .lock()
            .expect("retry throttler lock is poisoned")
            .on_retry_failure(&flow);

        let mut prev_err = match flow {
            RetryResult::Permanent(e) | RetryResult::Exhausted(e) => return Err(e),
            RetryResult::Continue(e) => e,
        };

        let mut delay = self.backoff_policy.on_failure(&self.retry_state);

        loop {
            if self
                .retry_policy
                .remaining_time(&self.retry_state)
                .is_some_and(|remaining| remaining <= delay)
            {
                return Err(Error::exhausted(prev_err));
            }

            tokio::select! {
                biased;
                _ = cancel_rx.changed() => return Ok(None),
                _ = tokio::time::sleep(delay) => {},
            }

            if *cancel_rx.borrow_and_update() {
                return Ok(None);
            }

            let throttled = self
                .retry_throttler
                .lock()
                .expect("retry throttler lock is poisoned")
                .throttle_retry_attempt();
            if !throttled {
                return Ok(Some(()));
            }

            self.retry_state.attempt_count += 1;
            prev_err = match self.retry_policy.on_throttle(&self.retry_state, prev_err) {
                ThrottleResult::Exhausted(e) => return Err(e),
                ThrottleResult::Continue(e) => e,
            };
            delay = self.backoff_policy.on_failure(&self.retry_state);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use google_cloud_gax::error::rpc::Status;
    use google_cloud_gax::retry_throttler::RetryThrottler;

    use super::*;
    use crate::client_builder::default_retry_throttler;

    #[derive(Debug, Default)]
    struct NoBackoff;

    impl BackoffPolicy for NoBackoff {
        fn on_failure(&self, _state: &RetryState) -> Duration {
            Duration::ZERO
        }
    }

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

    #[tokio::test]
    async fn test_backoff_session_attempt_limit_and_reset() {
        use google_cloud_gax::retry_throttler::{CircuitBreaker, RetryThrottlerArg};

        let mut session = BackoffSession::new(
            Arc::new(RetryableErrors.with_attempt_limit(3)),
            Arc::new(NoBackoff),
            RetryThrottlerArg::from(CircuitBreaker::default()).into(),
        );
        let (_cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);

        // Attempt 1 -> Continue
        assert!(session
            .next_delay(status_err(Code::Unavailable), &mut cancel_rx)
            .await
            .unwrap()
            .is_some());
        // Attempt 2 -> Continue
        assert!(session
            .next_delay(status_err(Code::Unavailable), &mut cancel_rx)
            .await
            .unwrap()
            .is_some());
        // Attempt 3 -> Exhausted
        assert!(session
            .next_delay(status_err(Code::Unavailable), &mut cancel_rx)
            .await
            .is_err());

        // Resetting after data progress restores the full retry budget.
        session.reset();
        assert!(session
            .next_delay(status_err(Code::Unavailable), &mut cancel_rx)
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn test_backoff_session_throttling_exhausts_attempt_limit() {
        #[derive(Debug)]
        struct AlwaysThrottle {
            checks: Arc<AtomicUsize>,
        }
        impl RetryThrottler for AlwaysThrottle {
            fn throttle_retry_attempt(&self) -> bool {
                self.checks.fetch_add(1, Ordering::SeqCst);
                true
            }
            fn on_retry_failure(&mut self, _flow: &RetryResult) {}
            fn on_success(&mut self) {}
        }

        let checks = Arc::new(AtomicUsize::new(0));
        let throttler: SharedRetryThrottler = Arc::new(Mutex::new(AlwaysThrottle {
            checks: Arc::clone(&checks),
        }));

        let mut session = BackoffSession::new(
            Arc::new(RetryableErrors.with_attempt_limit(3)),
            Arc::new(NoBackoff),
            throttler,
        );
        let (_cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);

        let res = session
            .next_delay(status_err(Code::Unavailable), &mut cancel_rx)
            .await;
        assert!(res.is_err());
        assert_eq!(checks.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_backoff_session_cancellation() {
        let mut session = BackoffSession::new(
            Arc::new(RetryableErrors.with_attempt_limit(3)),
            Arc::new(
                ExponentialBackoffBuilder::new()
                    .with_initial_delay(Duration::from_secs(60))
                    .with_maximum_delay(Duration::from_secs(60))
                    .build()
                    .unwrap(),
            ),
            default_retry_throttler(),
        );
        let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
        let _ = cancel_tx.send_replace(true);

        let res = session
            .next_delay(status_err(Code::Unavailable), &mut cancel_rx)
            .await
            .unwrap();
        assert!(res.is_none());
    }
}
