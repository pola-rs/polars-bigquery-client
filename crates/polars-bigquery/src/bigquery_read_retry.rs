//! Retry policies and backoff session management built on Tower ([`tower::retry`]).
//!
//! This module separates static **policy configuration** from active **runtime session state**:
//!
//! - [`RetryPolicy`]: An immutable blueprint holding retry configuration (min/max delays, custom multiplier `factor`,
//!   jitter, max attempt limits, total duration timeouts, and error predicates). It implements [`tower::retry::Policy`]
//!   for use with Tower middleware services ([`tower::retry::Retry`]), such as initial RPC calls.
//! - [`BackoffSession`]: Tracks the mutable runtime state for a single operation attempt (current iteration count,
//!   start time, multiplier calculation, and sleep generation).
//!
//! ### Why Custom Multiplier Factor?
//! Tower's built-in [`tower::retry::backoff::ExponentialBackoff`] hardcodes a binary multiplier of `2.0` ($2^n$).
//! The BigQuery Storage API Python reference client specifies a multiplier factor of `1.3` (100ms -> 130ms -> 169ms ...),
//! which ramps up more gently. `RetryPolicy` and `BackoffSession` support configurable multiplier factors while remaining
//! 100% compatible with Tower's [`tower::retry::Policy`] trait.

use core::time::Duration;
use std::fmt::Debug;
use std::time::Instant;

use gcloud_sdk::tonic;
use tower::retry::Policy;
use tower::util::rng::{HasherRng, Rng};

/// Tracks active backoff state (attempt count, start time, and sleep generator) for a single operation session.
///
/// Created via [`RetryPolicy::make_session`]. Supports configurable exponential growth factor ($min \times factor^{attempts}$).
#[derive(Debug, Clone)]
pub struct BackoffSession<R = HasherRng> {
    min_delay: Duration,
    max_delay: Duration,
    factor: f64,
    jitter: f64,
    rng: R,
    attempts: u32,
    max_times: Option<u32>,
    start_time: Option<Instant>,
    max_total_delay: Option<Duration>,
}

impl<R: Rng> BackoffSession<R> {
    /// Calculate base delay before jitter: min_delay * (factor ^ attempts), capped at max_delay.
    fn base_delay(&self) -> Duration {
        let mult = self.factor.powi(self.attempts as i32);
        let secs = self.min_delay.as_secs_f64() * mult;
        Duration::try_from_secs_f64(secs)
            .unwrap_or(self.max_delay)
            .min(self.max_delay)
    }

    /// Calculate random uniform jitter added to base delay.
    fn jitter_delay(&mut self, base: Duration) -> Duration {
        if self.jitter == 0.0 {
            Duration::ZERO
        } else {
            let jitter_factor = self.rng.next_f64() * self.jitter;
            let jitter_secs = base.as_secs_f64() * jitter_factor;
            let remaining = self.max_delay.saturating_sub(base);
            Duration::from_secs_f64(jitter_secs).min(remaining)
        }
    }

    /// Calculate next backoff duration (base + jitter).
    pub fn compute_next_delay(&mut self) -> Duration {
        let base = self.base_delay();
        let jitter = self.jitter_delay(base);
        base + jitter
    }

    /// Delay for the next backoff attempt if retries are not exhausted.
    /// Returns `true` if it slept and can retry, `false` if retries were exhausted.
    pub async fn next_delay(&mut self) -> bool {
        if let Some(max_times) = self.max_times {
            if self.attempts >= max_times {
                return false;
            }
        }
        if let Some(max_total_delay) = self.max_total_delay {
            let start = *self.start_time.get_or_insert_with(Instant::now);
            if start.elapsed() >= max_total_delay {
                return false;
            }
        }
        let delay = self.compute_next_delay();
        self.attempts += 1;
        tokio::time::sleep(delay).await;
        true
    }
}

/// Immutable configuration blueprint for exponential backoff retries, implementing [`tower::retry::Policy`].
///
/// `RetryPolicy` stores static configuration parameters (min/max delays, custom multiplier factor, jitter, attempt/delay limits, and predicate).
/// When cloned by `tower::retry::Retry` for a request session, it lazily creates a [`BackoffSession`] to track
/// attempt counts and elapsed time across retries of that request.
#[derive(Debug)]
pub struct RetryPolicy<P, R = HasherRng> {
    min_delay: Duration,
    max_delay: Duration,
    factor: f64,
    jitter: f64,
    rng: R,
    predicate: P,
    max_times: Option<u32>,
    max_total_delay: Option<Duration>,
    session: Option<BackoffSession<R>>,
}

impl<P, R> RetryPolicy<P, R>
where
    R: Rng + Clone,
{
    pub fn new(
        min_delay: Duration,
        max_delay: Duration,
        factor: f64,
        jitter: f64,
        rng: R,
        predicate: P,
    ) -> Self {
        assert!(min_delay <= max_delay, "min_delay must be <= max_delay");
        assert!(max_delay > Duration::ZERO, "max_delay must be > 0");
        assert!(factor >= 1.0, "factor must be >= 1.0");
        assert!(jitter >= 0.0, "jitter must be >= 0.0");

        Self {
            min_delay,
            max_delay,
            factor,
            jitter,
            rng,
            predicate,
            max_times: None,
            max_total_delay: None,
            session: None,
        }
    }

    pub fn with_max_times(mut self, max_times: u32) -> Self {
        self.max_times = Some(max_times);
        self
    }

    pub fn with_total_delay(mut self, total_delay: Duration) -> Self {
        self.max_total_delay = Some(total_delay);
        self
    }

    pub fn make_session(&self) -> BackoffSession<R> {
        BackoffSession {
            min_delay: self.min_delay,
            max_delay: self.max_delay,
            factor: self.factor,
            jitter: self.jitter,
            rng: self.rng.clone(),
            attempts: 0,
            max_times: self.max_times,
            start_time: None,
            max_total_delay: self.max_total_delay,
        }
    }
}

impl<P: Clone, R: Clone> Clone for RetryPolicy<P, R> {
    fn clone(&self) -> Self {
        Self {
            min_delay: self.min_delay,
            max_delay: self.max_delay,
            factor: self.factor,
            jitter: self.jitter,
            rng: self.rng.clone(),
            predicate: self.predicate.clone(),
            max_times: self.max_times,
            max_total_delay: self.max_total_delay,
            session: None,
        }
    }
}

impl<Req, Res, E, P, R> Policy<Req, Res, E> for RetryPolicy<P, R>
where
    Req: Clone,
    P: Fn(&E) -> bool,
    R: Rng + Clone,
{
    type Future = tokio::time::Sleep;

    fn retry(&mut self, _req: &mut Req, result: &mut Result<Res, E>) -> Option<Self::Future> {
        match result {
            Ok(_) => {
                self.session = None;
                None
            },
            Err(err) => {
                if !(self.predicate)(err) {
                    return None;
                }
                if self.session.is_none() {
                    self.session = Some(self.make_session());
                }
                let session = self.session.as_mut().unwrap();
                if let Some(max_times) = session.max_times {
                    if session.attempts >= max_times {
                        return None;
                    }
                }
                if let Some(max_total_delay) = session.max_total_delay {
                    let start = *session.start_time.get_or_insert_with(Instant::now);
                    if start.elapsed() >= max_total_delay {
                        return None;
                    }
                }
                let delay = session.compute_next_delay();
                session.attempts += 1;
                Some(tokio::time::sleep(delay))
            },
        }
    }

    fn clone_request(&mut self, req: &Req) -> Option<Req> {
        Some(req.clone())
    }
}

fn is_retryable_status(err: &tonic::Status) -> bool {
    matches!(
        err.code(),
        tonic::Code::DeadlineExceeded
            | tonic::Code::Aborted
            | tonic::Code::ResourceExhausted
            // Unavailable includes common transport-level failures such as,
            // - Hyper/h2 connection errors, as seen in
            //   https://github.com/grpc/grpc-rust/pull/629
            // - Connection errors, like ConnectionReset, as seen in
            //   https://github.com/grpc/grpc-rust/blob/230544e31ef9b513e493c273d4076e843478e934/tonic/src/status.rs#L757-L761
            | tonic::Code::Unavailable
            // Internal includes common transport-level failures such as,
            // - broken pipe, as seen in
            //   https://github.com/grpc/grpc-rust/blob/230544e31ef9b513e493c273d4076e843478e934/tonic/src/status.rs#L753-L756
            | tonic::Code::Internal
    )
}

/// When to retry create_read_session requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L154-L157
pub fn create_read_session_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

/// Retry configuration policy for create_read_session.
///
/// - `min_delay` (100ms): Initial sleep duration before first retry.
/// - `max_delay` (60s): Upper bound cap on any single exponential backoff sleep.
/// - `factor` (1.3): Multiplier scaling factor for exponential backoff (matching Python BigQuery Storage client standard).
/// - `with_total_delay` (600s): Maximum cumulative duration across retries before giving up.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L148-L162
pub fn create_read_session_policy() -> RetryPolicy<fn(&tonic::Status) -> bool, HasherRng> {
    RetryPolicy::new(
        Duration::from_millis(100),
        Duration::from_secs(60),
        1.3,
        0.2,
        HasherRng::default(),
        create_read_session_predicate as fn(&tonic::Status) -> bool,
    )
    .with_total_delay(Duration::from_secs(600))
}

/// When to retry read_rows requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L169-L171
pub fn read_rows_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

/// Retry configuration policy for initial read_rows requests.
///
/// - `min_delay` (100ms): Initial sleep duration before first retry.
/// - `max_delay` (60s): Upper bound cap on any single exponential backoff sleep.
/// - `factor` (1.3): Multiplier scaling factor for exponential backoff (matching Python BigQuery Storage client standard).
/// - `with_total_delay` (900s): Maximum cumulative duration across retries before giving up.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L163-L176
pub fn read_rows_policy() -> RetryPolicy<fn(&tonic::Status) -> bool, HasherRng> {
    RetryPolicy::new(
        Duration::from_millis(100),
        Duration::from_secs(60),
        1.3,
        0.2,
        HasherRng::default(),
        read_rows_predicate as fn(&tonic::Status) -> bool,
    )
    .with_total_delay(Duration::from_secs(900))
}

/// When to reconnect/resume an active read_rows stream after encountering a gRPC error mid-read.
///
/// While currently identical to [`read_rows_predicate`], having a dedicated predicate allows fine-tuning
/// reconnection behavior separately from initial request establishment.
pub fn reconnect_stream_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

/// Retry configuration policy for mid-stream read_rows reconnections.
///
/// Important! If data progress is made (`made_progress == true`), the session
/// must be reset reset to grant a fresh 10-attempt allowance.
///
/// ### Parameters
/// - `min_delay` (100ms): Starts with a short initial backoff to quickly recover from brief network blips.
/// - `max_delay` (60s): Caps the backoff delay so exponential growth (100ms -> 130ms -> 169ms ...) does not produce excessively long single delays.
/// - `factor` (1.3): Multiplier scaling factor for exponential backoff (matching Python BigQuery Storage client standard).
/// - `with_max_times` (10): Limits total consecutive failed reconnection attempts when no data progress is made.
pub fn stream_reconnect_policy() -> RetryPolicy<fn(&tonic::Status) -> bool, HasherRng> {
    RetryPolicy::new(
        Duration::from_millis(100),
        Duration::from_secs(60),
        1.3,
        0.2,
        HasherRng::default(),
        reconnect_stream_predicate as fn(&tonic::Status) -> bool,
    )
    .with_max_times(10)
}

#[cfg(test)]
mod tests {
    use gcloud_sdk::tonic::{Code, Status};

    use super::*;

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
            let status = Status::new(code, "transient stream error");
            assert!(
                reconnect_stream_predicate(&status),
                "Expected reconnect_stream_predicate to be true for code {:?}",
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
            let status = Status::new(code, "fatal stream error");
            assert!(
                !reconnect_stream_predicate(&status),
                "Expected reconnect_stream_predicate to be false for code {:?}",
                code
            );
        }
    }

    #[test]
    fn test_read_rows_predicate_matches_reconnect() {
        // Ensure initial request predicate and stream reconnection predicate currently match on behavior
        let codes = [
            Code::Unavailable,
            Code::Aborted,
            Code::NotFound,
            Code::InvalidArgument,
        ];
        for code in codes {
            let status = Status::new(code, "test");
            assert_eq!(
                read_rows_predicate(&status),
                reconnect_stream_predicate(&status)
            );
        }
    }

    #[tokio::test]
    async fn test_retry_policy_max_times() {
        let policy = RetryPolicy::new(
            Duration::from_millis(1),
            Duration::from_millis(1),
            1.3,
            0.0,
            HasherRng::default(),
            |_: &Status| true,
        )
        .with_max_times(2);

        let mut session = policy.make_session();
        assert!(session.next_delay().await);
        assert!(session.next_delay().await);
        assert!(!session.next_delay().await);
    }

    #[test]
    fn test_backoff_factor_growth() {
        let policy = RetryPolicy::new(
            Duration::from_millis(100),
            Duration::from_secs(60),
            1.3,
            0.0, // no jitter for deterministic check
            HasherRng::default(),
            |_: &Status| true,
        );
        let mut session = policy.make_session();
        assert_eq!(session.compute_next_delay(), Duration::from_millis(100)); // 100 * 1.3^0
        session.attempts += 1;
        assert_eq!(session.compute_next_delay(), Duration::from_millis(130)); // 100 * 1.3^1
        session.attempts += 1;
        assert_eq!(session.compute_next_delay(), Duration::from_millis(169)); // 100 * 1.3^2
    }

    #[test]
    fn test_backoff_overflow_safety() {
        let policy = RetryPolicy::new(
            Duration::from_millis(100),
            Duration::from_secs(60),
            1.3,
            0.0,
            HasherRng::default(),
            |_: &Status| true,
        );
        let mut session = policy.make_session();
        session.attempts = 1000; // 1.3^1000 is infinity in f64
        assert_eq!(session.compute_next_delay(), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_retry_policy_session_reset_on_ok() {
        let mut policy = RetryPolicy::new(
            Duration::from_millis(1),
            Duration::from_secs(60),
            1.3,
            0.0,
            HasherRng::default(),
            |_: &Status| true,
        );

        let mut req = ();
        let mut err_res: Result<(), Status> = Err(Status::unavailable("transient"));
        assert!(policy.retry(&mut req, &mut err_res).is_some());
        assert!(policy.session.is_some());

        let mut ok_res: Result<(), Status> = Ok(());
        assert!(policy.retry(&mut req, &mut ok_res).is_none());
        assert!(policy.session.is_none());
    }
}
