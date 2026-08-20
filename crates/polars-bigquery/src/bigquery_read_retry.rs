//! Retry policies and backoff session management built on Tower ([`tower::retry`]).
//!
//! This module separates static **policy configuration** from active **runtime session state**:
//!
//! - [`RetryParameters`]: An immutable struct holding retry configuration (min/max delays, custom multiplier `factor`,
//!   jitter, max attempt limits, and total duration timeouts).
//! - [`RetryParametersBuilder`]: Builder for validating and constructing [`RetryParameters`].
//! - [`RetryPolicy`]: Combines [`RetryParameters`] with an error predicate and RNG source. It implements [`tower::retry::Policy`]
//!   for use with Tower middleware services ([`tower::retry::Retry`]), such as initial RPC calls.
//! - [`BackoffSession`]: Tracks the mutable runtime state for a single operation attempt (current iteration count,
//!   start time, multiplier calculation, and sleep generation), initialized from [`RetryParameters`].
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

/// Common immutable configuration parameters for exponential backoff retries.
///
/// Holds static timing parameters and limits (min/max delays, custom multiplier factor,
/// jitter ratio, attempt count limits, and total duration timeouts) shared across
/// [`RetryPolicy`] and [`BackoffSession`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryParameters {
    pub min_delay: Duration,
    pub max_delay: Duration,
    pub factor: f64,
    pub jitter: f64,
    pub max_times: Option<u32>,
    pub max_total_delay: Option<Duration>,
}

impl RetryParameters {
    /// Creates a new [`RetryParametersBuilder`] to configure and validate retry parameters.
    pub fn builder() -> RetryParametersBuilder {
        RetryParametersBuilder::new()
    }
}

/// Builder for constructing and validating [`RetryParameters`].
#[derive(Debug, Clone, Default)]
pub struct RetryParametersBuilder {
    min_delay: Option<Duration>,
    max_delay: Option<Duration>,
    factor: Option<f64>,
    jitter: Option<f64>,
    max_times: Option<u32>,
    max_total_delay: Option<Duration>,
}

impl RetryParametersBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn check_delay_bounds(&self) -> Result<(), String> {
        let Some(min_delay) = self.min_delay else {
            return Ok(());
        };
        let Some(max_delay) = self.max_delay else {
            return Ok(());
        };
        if min_delay > max_delay {
            return Err("min_delay must be <= max_delay".to_owned());
        }
        Ok(())
    }

    pub fn with_min_delay(mut self, min_delay: Duration) -> Result<Self, String> {
        self.min_delay = Some(min_delay);
        self.check_delay_bounds()?;
        Ok(self)
    }

    pub fn with_max_delay(mut self, max_delay: Duration) -> Result<Self, String> {
        if max_delay <= Duration::ZERO {
            return Err("max_delay must be > 0".to_owned());
        }
        self.max_delay = Some(max_delay);
        self.check_delay_bounds()?;
        Ok(self)
    }

    pub fn with_factor(mut self, factor: f64) -> Self {
        self.factor = Some(factor);
        self
    }

    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = Some(jitter);
        self
    }

    pub fn with_max_times(mut self, max_times: u32) -> Self {
        self.max_times = Some(max_times);
        self
    }

    pub fn with_max_total_delay(mut self, max_total_delay: Duration) -> Self {
        self.max_total_delay = Some(max_total_delay);
        self
    }

    pub fn build(self) -> Result<RetryParameters, String> {
        let min_delay = self.min_delay.ok_or("min_delay is required")?;
        let max_delay = self.max_delay.ok_or("max_delay is required")?;
        let factor = self.factor.ok_or("factor is required")?;
        let jitter = self.jitter.ok_or("jitter is required")?;
        if max_delay <= Duration::ZERO {
            return Err("max_delay must be > 0".to_owned());
        }
        if min_delay > max_delay {
            return Err("min_delay must be <= max_delay".to_owned());
        }
        Ok(RetryParameters {
            min_delay,
            max_delay,
            factor,
            jitter,
            max_times: self.max_times,
            max_total_delay: self.max_total_delay,
        })
    }
}

/// Tracks active backoff state (attempt count, start time, and sleep generator) for a single operation session.
///
/// Created via [`RetryPolicy::make_session`]. Supports configurable exponential growth factor ($min \times factor^{attempts}$).
#[derive(Debug, Clone)]
pub struct BackoffSession<R = HasherRng> {
    parameters: RetryParameters,
    rng: R,
    attempts: u32,
    start_time: Option<Instant>,
}

impl<R: Rng> BackoffSession<R> {
    /// Calculate base delay before jitter: min_delay * (factor ^ attempts), capped at max_delay.
    fn base_delay(&self) -> Duration {
        let mult = self.parameters.factor.powi(self.attempts as i32);
        let secs = self.parameters.min_delay.as_secs_f64() * mult;
        Duration::try_from_secs_f64(secs)
            .unwrap_or(self.parameters.max_delay)
            .min(self.parameters.max_delay)
    }

    /// Calculate random uniform jitter added to base delay.
    fn jitter_delay(&mut self, base: Duration) -> Duration {
        if self.parameters.jitter == 0.0 {
            Duration::ZERO
        } else {
            let jitter_factor = self.rng.next_f64() * self.parameters.jitter;
            let jitter_secs = base.as_secs_f64() * jitter_factor;
            let remaining = self.parameters.max_delay.saturating_sub(base);
            Duration::from_secs_f64(jitter_secs).min(remaining)
        }
    }

    /// Calculate next backoff duration (base + jitter).
    fn compute_next_delay(&mut self) -> Duration {
        let base = self.base_delay();
        let jitter = self.jitter_delay(base);
        base + jitter
    }

    /// Calculate and return the next delay if retries are not exhausted by limits.
    /// Increments `attempts` and updates session start time if applicable.
    fn advance_delay(&mut self) -> Option<Duration> {
        if let Some(max_times) = self.parameters.max_times {
            if self.attempts >= max_times {
                return None;
            }
        }
        if let Some(max_total_delay) = self.parameters.max_total_delay {
            let start = *self.start_time.get_or_insert_with(Instant::now);
            if start.elapsed() >= max_total_delay {
                return None;
            }
        }
        let delay = self.compute_next_delay();
        self.attempts += 1;
        Some(delay)
    }

    /// Delay for the next backoff attempt if retries are not exhausted.
    /// Returns `true` if it slept and can retry, `false` if retries were exhausted.
    pub async fn next_delay(&mut self) -> bool {
        if let Some(delay) = self.advance_delay() {
            tokio::time::sleep(delay).await;
            true
        } else {
            false
        }
    }
}

/// Immutable configuration blueprint for exponential backoff retries, implementing [`tower::retry::Policy`].
///
/// `RetryPolicy` stores static configuration [`RetryParameters`], an RNG source, and an error predicate.
/// When cloned by `tower::retry::Retry` for a request session, it lazily creates a [`BackoffSession`] to track
/// attempt counts and elapsed time across retries of that request.
#[derive(Debug)]
pub struct RetryPolicy<P, R = HasherRng> {
    pub parameters: RetryParameters,
    pub rng: R,
    pub predicate: P,
    pub session: Option<BackoffSession<R>>,
}

impl<P, R> RetryPolicy<P, R> {
    /// Creates a new [`RetryPolicy`] with the specified parameters, error predicate, and RNG.
    pub fn new(parameters: RetryParameters, predicate: P, rng: R) -> Self {
        Self {
            parameters,
            rng,
            predicate,
            session: None,
        }
    }
}

impl<P, R> RetryPolicy<P, R>
where
    R: Rng + Clone,
{
    pub fn make_session(&self) -> BackoffSession<R> {
        BackoffSession {
            parameters: self.parameters,
            rng: self.rng.clone(),
            attempts: 0,
            start_time: None,
        }
    }
}

impl<P: Clone, R: Clone> Clone for RetryPolicy<P, R> {
    fn clone(&self) -> Self {
        Self {
            parameters: self.parameters,
            rng: self.rng.clone(),
            predicate: self.predicate.clone(),
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
                let delay = session.advance_delay()?;
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

/// Retry parameters for create_read_session requests.
///
/// - `min_delay` (100ms): Initial sleep duration before first retry.
/// - `max_delay` (60s): Upper bound cap on any single exponential backoff sleep.
/// - `factor` (1.3): Multiplier scaling factor for exponential backoff (matching Python BigQuery Storage client standard).
/// - `max_total_delay` (600s): Maximum cumulative duration across retries before giving up.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L148-L162
pub fn create_read_session_parameters() -> RetryParameters {
    RetryParameters::builder()
        .with_min_delay(Duration::from_millis(100))
        .expect("hardcoded value guaranteed to be valid")
        .with_max_delay(Duration::from_secs(60))
        .expect("hardcoded value guaranteed to be valid")
        .with_factor(1.3)
        .with_jitter(0.2)
        .with_max_total_delay(Duration::from_secs(600))
        .build()
        .expect("hardcoded value guaranteed to be valid")
}

/// Retry configuration policy for create_read_session.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L148-L162
pub fn create_read_session_policy() -> RetryPolicy<fn(&tonic::Status) -> bool, HasherRng> {
    RetryPolicy::new(
        create_read_session_parameters(),
        create_read_session_predicate as fn(&tonic::Status) -> bool,
        HasherRng::default(),
    )
}

/// When to retry read_rows requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L169-L171
pub fn read_rows_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

/// Retry parameters for initial read_rows requests.
///
/// - `min_delay` (100ms): Initial sleep duration before first retry.
/// - `max_delay` (60s): Upper bound cap on any single exponential backoff sleep.
/// - `factor` (1.3): Multiplier scaling factor for exponential backoff (matching Python BigQuery Storage client standard).
/// - `max_total_delay` (900s): Maximum cumulative duration across retries before giving up.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L163-L176
pub fn read_rows_parameters() -> RetryParameters {
    RetryParameters::builder()
        .with_min_delay(Duration::from_millis(100))
        .expect("hardcoded value guaranteed to be valid")
        .with_max_delay(Duration::from_secs(60))
        .expect("hardcoded value guaranteed to be valid")
        .with_factor(1.3)
        .with_jitter(0.2)
        .with_max_total_delay(Duration::from_secs(900))
        .build()
        .expect("hardcoded value guaranteed to be valid")
}

/// Retry configuration policy for initial read_rows requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L163-L176
pub fn read_rows_policy() -> RetryPolicy<fn(&tonic::Status) -> bool, HasherRng> {
    RetryPolicy::new(
        read_rows_parameters(),
        read_rows_predicate as fn(&tonic::Status) -> bool,
        HasherRng::default(),
    )
}

/// When to reconnect/resume an active read_rows stream after encountering a gRPC error mid-read.
///
/// While currently identical to [`read_rows_predicate`], having a dedicated predicate allows fine-tuning
/// reconnection behavior separately from initial request establishment.
pub fn reconnect_stream_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

/// Retry parameters for mid-stream read_rows reconnections.
///
/// - `min_delay` (100ms): Starts with a short initial backoff to quickly recover from brief network blips.
/// - `max_delay` (60s): Caps the backoff delay so exponential growth (100ms -> 130ms -> 169ms ...) does not produce excessively long single delays.
/// - `factor` (1.3): Multiplier scaling factor for exponential backoff (matching Python BigQuery Storage client standard).
/// - `max_times` (10): Limits total consecutive failed reconnection attempts when no data progress is made.
pub fn stream_reconnect_parameters() -> RetryParameters {
    RetryParameters::builder()
        .with_min_delay(Duration::from_millis(100))
        .expect("hardcoded value guaranteed to be valid")
        .with_max_delay(Duration::from_secs(60))
        .expect("hardcoded value guaranteed to be valid")
        .with_factor(1.3)
        .with_jitter(0.2)
        .with_max_times(10)
        .build()
        .expect("hardcoded value guaranteed to be valid")
}

/// Retry configuration policy for mid-stream read_rows reconnections.
///
/// Important! If data progress is made (`current_offset > prev_offset`), the session
/// must be reset to grant a fresh 10-attempt allowance.
pub fn stream_reconnect_policy() -> RetryPolicy<fn(&tonic::Status) -> bool, HasherRng> {
    RetryPolicy::new(
        stream_reconnect_parameters(),
        reconnect_stream_predicate as fn(&tonic::Status) -> bool,
        HasherRng::default(),
    )
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
        let params = RetryParameters::builder()
            .with_min_delay(Duration::from_millis(1))
            .expect("hardcoded value guaranteed to be valid")
            .with_max_delay(Duration::from_millis(1))
            .expect("hardcoded value guaranteed to be valid")
            .with_factor(1.3)
            .with_jitter(0.0)
            .with_max_times(2)
            .build()
            .expect("hardcoded value guaranteed to be valid");
        let policy = RetryPolicy::new(params, |_: &Status| true, HasherRng::default());

        let mut session = policy.make_session();
        assert!(session.next_delay().await);
        assert!(session.next_delay().await);
        assert!(!session.next_delay().await);
    }

    #[test]
    fn test_backoff_factor_growth() {
        let params = RetryParameters::builder()
            .with_min_delay(Duration::from_millis(100))
            .expect("hardcoded value guaranteed to be valid")
            .with_max_delay(Duration::from_secs(60))
            .expect("hardcoded value guaranteed to be valid")
            .with_factor(1.3)
            .with_jitter(0.0) // no jitter for deterministic check
            .build()
            .expect("hardcoded value guaranteed to be valid");
        let policy = RetryPolicy::new(params, |_: &Status| true, HasherRng::default());
        let mut session = policy.make_session();
        assert_eq!(session.compute_next_delay(), Duration::from_millis(100)); // 100 * 1.3^0
        session.attempts += 1;
        assert_eq!(session.compute_next_delay(), Duration::from_millis(130)); // 100 * 1.3^1
        session.attempts += 1;
        assert_eq!(session.compute_next_delay(), Duration::from_millis(169)); // 100 * 1.3^2
    }

    #[test]
    fn test_backoff_overflow_safety() {
        let params = RetryParameters::builder()
            .with_min_delay(Duration::from_millis(100))
            .expect("hardcoded value guaranteed to be valid")
            .with_max_delay(Duration::from_secs(60))
            .expect("hardcoded value guaranteed to be valid")
            .with_factor(1.3)
            .with_jitter(0.0)
            .build()
            .expect("hardcoded value guaranteed to be valid");
        let policy = RetryPolicy::new(params, |_: &Status| true, HasherRng::default());
        let mut session = policy.make_session();
        session.attempts = 1000; // 1.3^1000 is infinity in f64
        assert_eq!(session.compute_next_delay(), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_retry_policy_session_reset_on_ok() {
        let params = RetryParameters::builder()
            .with_min_delay(Duration::from_millis(1))
            .expect("hardcoded value guaranteed to be valid")
            .with_max_delay(Duration::from_secs(60))
            .expect("hardcoded value guaranteed to be valid")
            .with_factor(1.3)
            .with_jitter(0.0)
            .build()
            .expect("hardcoded value guaranteed to be valid");
        let mut policy = RetryPolicy::new(params, |_: &Status| true, HasherRng::default());

        let mut req = ();
        let mut err_res: Result<(), Status> = Err(Status::unavailable("transient"));
        assert!(policy.retry(&mut req, &mut err_res).is_some());
        assert!(policy.session.is_some());

        let mut ok_res: Result<(), Status> = Ok(());
        assert!(policy.retry(&mut req, &mut ok_res).is_none());
        assert!(policy.session.is_none());
    }

    #[test]
    fn test_retry_parameters_builder_validation() {
        // Zero or negative max_delay should fail
        assert!(RetryParameters::builder()
            .with_max_delay(Duration::ZERO)
            .is_err());

        // min_delay > max_delay should fail at setter
        assert!(RetryParameters::builder()
            .with_max_delay(Duration::from_secs(5))
            .unwrap()
            .with_min_delay(Duration::from_secs(10))
            .is_err());

        // Missing required fields should fail on build
        assert!(RetryParameters::builder()
            .with_min_delay(Duration::from_millis(100))
            .unwrap()
            .build()
            .is_err());

        // Valid configuration succeeds
        let params = RetryParameters::builder()
            .with_min_delay(Duration::from_millis(100))
            .unwrap()
            .with_max_delay(Duration::from_secs(60))
            .unwrap()
            .with_factor(1.3)
            .with_jitter(0.2)
            .with_max_times(5)
            .with_max_total_delay(Duration::from_secs(300))
            .build();
        assert!(params.is_ok());
        let params = params.unwrap();
        assert_eq!(params.min_delay, Duration::from_millis(100));
        assert_eq!(params.max_delay, Duration::from_secs(60));
        assert_eq!(params.factor, 1.3);
        assert_eq!(params.jitter, 0.2);
        assert_eq!(params.max_times, Some(5));
        assert_eq!(params.max_total_delay, Some(Duration::from_secs(300)));
    }
}


