//! Retry, backoff, and circuit-breaker policies for BigQuery v2 `JobService` (REST)
//! requests (`jobs.query`, `jobs.get`, and `jobs.cancel`).
//!
//! Separated from [`crate::bigquery_read_retry`] (gRPC Storage Read API) so HTTP
//! status/reason semantics and `request_id` idempotency bounds remain strictly
//! isolated from gRPC `ReadRows` retry rules.

use std::sync::Arc;
use std::time::{Duration, Instant};

use google_cloud_gax::backoff_policy::BackoffPolicy;
use google_cloud_gax::error::rpc::{Code, StatusDetails};
use google_cloud_gax::error::Error;
use google_cloud_gax::exponential_backoff::{ExponentialBackoff, ExponentialBackoffBuilder};
use google_cloud_gax::retry_policy::{RetryPolicy, RetryPolicyExt};
use google_cloud_gax::retry_result::RetryResult;
use google_cloud_gax::retry_state::RetryState;
use google_cloud_gax::retry_throttler::{RetryThrottler, RetryThrottlerArg, SharedRetryThrottler};
use google_cloud_gax::throttle_result::ThrottleResult;

// BigQuery v2 `JobService` (REST) retry/backoff constants.
const JOB_SERVICE_INITIAL_DELAY: Duration = Duration::from_secs(1);
const JOB_SERVICE_MAXIMUM_DELAY: Duration = Duration::from_secs(60);
const JOB_SERVICE_SCALING_FACTOR: f64 = 2.0;
const JOB_SERVICE_MAX_ATTEMPTS: u32 = 5;

/// Cumulative retry budget (`600 s` = 10 min) for `JobService` metadata requests
/// (`jobs.get`, dry-run `jobs.query`) and maximum retry-initiation window for
/// `jobs.query` executions.
///
/// Kept strictly below BigQuery's 15-minute (`900 s`) `request_id` idempotency
/// lifetime (even accounting for the worst-case final backoff of `<= 60 s`) so a
/// mutating DML/DDL query is never retried after its `request_id` may have expired.
pub(crate) const JOB_SERVICE_RETRY_TIMEOUT: Duration = Duration::from_secs(600);

/// Conservative safety ceiling on BigQuery `QueryRequest.request_id` idempotency
/// lifetime (`14 min` vs BigQuery's documented `15 min` window).
const REQUEST_ID_SAFE_WINDOW: Duration = Duration::from_secs(14 * 60);

const _: () = assert!(
    JOB_SERVICE_RETRY_TIMEOUT.as_secs() + JOB_SERVICE_MAXIMUM_DELAY.as_secs()
        <= REQUEST_ID_SAFE_WINDOW.as_secs(),
    "JOB_SERVICE_RETRY_TIMEOUT + max backoff must stay within BigQuery's 15-minute request_id TTL"
);

// Best-effort `jobs.cancel` retry/backoff constants (bounded inside `JOB_CANCEL_TIMEOUT`).
const JOB_CANCEL_INITIAL_DELAY: Duration = Duration::from_millis(100);
const JOB_CANCEL_MAXIMUM_DELAY: Duration = Duration::from_secs(2);
const JOB_CANCEL_SCALING_FACTOR: f64 = 2.0;
pub(crate) const JOB_CANCEL_MAX_ATTEMPTS: u32 = 3;

// Token-bucket circuit breaker defaults for `JobService` and detached `jobs.cancel`.
const CIRCUIT_BREAKER_MAX_TOKENS: u64 = 1000;
const CIRCUIT_BREAKER_MIN_TOKENS: u64 = 250;
const CIRCUIT_BREAKER_ERROR_COST: u64 = 10;
/// Passive refill cadence (`1 token / 500 ms` = `2 tokens / s`).
///
/// Opens after 75 rapid transient attempt failures (`1000 -> 250`). Even if an
/// outage drains the bucket completely to `0`, a quiet/low-QPS client passively
/// recovers above `min_tokens` (`251`) in ~126 s instead of staying latched open
/// for 251 consecutive successful queries, while any sustained failure rate
/// faster than `1 failure / 5 s` continues draining tokens and keeps it open.
const CIRCUIT_BREAKER_REFILL_INTERVAL: Duration = Duration::from_millis(500);

/// Returns `true` for BigQuery REST `ErrorInfo.reason` values representing
/// transient rate limits or backend failures (supports both legacy camelCase
/// `error.errors[].reason` and AIP-193 `UPPER_SNAKE_CASE` values).
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
                        | "RATE_LIMIT_EXCEEDED"
                        | "userRateLimitExceeded"
                        | "USER_RATE_LIMIT_EXCEEDED"
                        | "jobRateLimitExceeded"
                        | "JOB_RATE_LIMIT_EXCEEDED"
                        | "backendError"
                        | "BACKEND_ERROR"
                        | "internalError"
                        | "INTERNAL_ERROR"
                )
        )
    })
}

fn is_transient_rpc_code(code: Code) -> bool {
    matches!(
        code,
        Code::DeadlineExceeded
            | Code::Aborted
            | Code::ResourceExhausted
            | Code::Unavailable
            | Code::Internal
    )
}

/// Returns `true` when a BigQuery v2 `JobService` service/HTTP error represents
/// a transient rate limit, timeout, or backend outage rather than a permanent
/// SQL/job failure or permission error.
///
/// Shared as the single source of truth between [`JobServiceRetryableErrors`],
/// [`QueryReadRetryPolicy`], and `query::map_query_error` so retries and final
/// error variant mapping never disagree.
///
/// Unlike gRPC error checks, this **does not** fall through to `status.code` when
/// an explicit HTTP status code is present: e.g., an HTTP `403` whose parsed
/// status happens to carry `Code::ResourceExhausted` or `Code::Aborted` is still
/// strictly gated by the BigQuery REST `ErrorInfo.reason` allowlist.
pub(crate) fn is_transient_job_service_status_error(error: &Error) -> bool {
    match error.http_status_code() {
        Some(408 | 429 | 500 | 502 | 503 | 504) => true,
        Some(403) => has_retryable_bigquery_rest_reason(error),
        Some(_) => false,
        None => error
            .status()
            .is_some_and(|status| is_transient_rpc_code(status.code)),
    }
}

/// Determines whether a `JobService` error is retryable.
///
/// When `retry_timeouts` is `false` (used for `jobs.query` execute calls),
/// client-side `Error::timeout()` is **not** retried so slow uncached `SELECT`
/// queries are neither duplicated nor double-billed after a full `timeoutMs` wait.
pub(crate) fn is_job_service_retryable_error(
    error: &Error,
    idempotent: bool,
    retry_timeouts: bool,
) -> bool {
    if error.is_transient_and_before_rpc() {
        return true;
    }
    if !idempotent {
        return false;
    }
    if error.is_timeout() {
        return retry_timeouts;
    }
    if error.is_io() || error.is_connect() {
        return true;
    }
    if error.is_transport() && error.http_status_code().is_none() {
        return true;
    }
    is_transient_job_service_status_error(error)
}

/// Retry policy for fast, idempotent BigQuery v2 `JobService` REST requests
/// (`jobs.get`, dry-run `jobs.query`, and `jobs.cancel`).
///
/// Fails fast when throttled (`on_throttle` returns [`ThrottleResult::Exhausted`])
/// so an open circuit breaker avoids spinning across the full retry budget.
#[derive(Clone, Debug, Default)]
pub(crate) struct JobServiceRetryableErrors;

impl RetryPolicy for JobServiceRetryableErrors {
    fn on_error(&self, state: &RetryState, error: Error) -> RetryResult {
        if is_job_service_retryable_error(&error, state.idempotent, true) {
            RetryResult::Continue(error)
        } else {
            RetryResult::Permanent(error)
        }
    }

    fn on_throttle(&self, _state: &RetryState, error: Error) -> ThrottleResult {
        ThrottleResult::Exhausted(error)
    }
}

/// Retry policy tailored for BigQuery v2 `jobs.query` execute requests.
///
/// Enforces three invariants that standard `with_time_limit` violates for
/// long-polling `jobs.query` RPCs:
/// 1. **No client-side timeout retries:** returns [`RetryResult::Exhausted`] on
///    `error.is_timeout()` so an expired `timeoutMs + buffer` wait stops without
///    spawning duplicate uncached `SELECT` queries (while still charging the
///    circuit breaker instead of crediting it as a success).
/// 2. **`request_id` TTL bound:** never schedules a retry once elapsed time
///    reaches `max_retry_window` (default `600 s`), ensuring mutating DML/DDL
///    queries cannot be retried past BigQuery's 15-minute `request_id` window
///    regardless of how large `initial_wait` (`timeoutMs`) is set.
/// 3. **No premature HTTP timeout clamping:** returns `None` from
///    `remaining_time()` so `google-cloud-gax` does not clamp a retry's
///    `reqwest` timeout below `timeoutMs + QUERY_ATTEMPT_TIMEOUT_BUFFER` while
///    re-sending a static JSON `timeoutMs` field. Per-attempt HTTP waits remain
///    bounded by `.with_attempt_timeout(attempt_timeout)`.
#[derive(Clone, Debug)]
struct QueryReadRetryPolicy {
    max_retry_window: Duration,
    max_attempts: u32,
}

impl RetryPolicy for QueryReadRetryPolicy {
    fn on_error(&self, state: &RetryState, error: Error) -> RetryResult {
        if state.idempotent && error.is_timeout() {
            // Do not re-execute slow queries on client timeout, and treat it as
            // Exhausted (not Permanent) so the retry throttler accounts for the outage.
            return RetryResult::Exhausted(error);
        }
        if !is_job_service_retryable_error(&error, state.idempotent, false) {
            return RetryResult::Permanent(error);
        }
        if state.attempt_count >= self.max_attempts {
            return RetryResult::Exhausted(error);
        }
        let now = tokio::time::Instant::now().into_std();
        if now.saturating_duration_since(state.start) >= self.max_retry_window {
            return RetryResult::Exhausted(error);
        }
        RetryResult::Continue(error)
    }

    fn on_throttle(&self, _state: &RetryState, error: Error) -> ThrottleResult {
        ThrottleResult::Exhausted(error)
    }

    fn remaining_time(&self, _state: &RetryState) -> Option<Duration> {
        // Leave per-attempt deadline strictly to `RequestOptions::attempt_timeout`.
        None
    }
}

/// Equal-jitter exponential backoff decorator drawing from `[cap / 2, cap]`.
///
/// Unlike `google_cloud_gax`'s bare `ExponentialBackoff` (full jitter `[0, cap]`),
/// equal jitter enforces a non-zero lower bound so rate-limited (`429` / `403
/// rateLimitExceeded`) requests do not retry immediately after ~0 ms when a
/// `Retry-After` header cannot be observed through `BackoffPolicy`.
#[derive(Debug)]
struct EqualJitterBackoff {
    inner: ExponentialBackoff,
    initial_delay: Duration,
    maximum_delay: Duration,
    scaling: f64,
}

impl EqualJitterBackoff {
    fn new(initial_delay: Duration, maximum_delay: Duration, scaling: f64) -> Self {
        let inner = ExponentialBackoffBuilder::new()
            .with_initial_delay(initial_delay)
            .with_maximum_delay(maximum_delay)
            .with_scaling(scaling)
            .build()
            .expect("hardcoded backoff parameters are guaranteed to be valid");
        Self {
            inner,
            initial_delay,
            maximum_delay,
            scaling,
        }
    }

    fn delay_cap(&self, attempt_count: u32) -> Duration {
        let exp = attempt_count.min(i32::MAX as u32).saturating_sub(1) as i32;
        let factor = self.scaling.powi(exp);
        if factor >= self.maximum_delay.div_duration_f64(self.initial_delay) {
            self.maximum_delay
        } else {
            self.initial_delay.mul_f64(factor)
        }
    }
}

impl BackoffPolicy for EqualJitterBackoff {
    fn on_failure(&self, state: &RetryState) -> Duration {
        let cap = self.delay_cap(state.attempt_count);
        let half = cap / 2;
        // `ExponentialBackoff::on_failure` draws uniformly from `[0, cap]`, so
        // `cap / 2 + sample / 2` distributes uniformly across `[cap / 2, cap]`.
        let jitter = self.inner.on_failure(state) / 2;
        half.saturating_add(jitter).min(cap)
    }
}

/// Token-bucket circuit breaker with passive time-based refill.
///
/// Mirrors `google_cloud_gax::retry_throttler::CircuitBreaker` token accounting
/// (`max_tokens = 1000`, `min_tokens = 250`, `error_cost = 10`, `success = +1`),
/// adding a monotonic passive refill so low-QPS clients recover automatically
/// after a quiet period following a backend outage instead of latching retries
/// disabled for hundreds of subsequent requests.
#[derive(Clone, Debug)]
pub(crate) struct RefillingCircuitBreaker {
    max_tokens: u64,
    min_tokens: u64,
    cur_tokens: u64,
    error_cost: u64,
    refill_interval: Duration,
    last_refill: Instant,
}

impl RefillingCircuitBreaker {
    fn new(max_tokens: u64, min_tokens: u64, error_cost: u64, refill_interval: Duration) -> Self {
        Self {
            max_tokens,
            min_tokens: min_tokens.min(max_tokens),
            cur_tokens: max_tokens,
            error_cost,
            refill_interval,
            last_refill: tokio::time::Instant::now().into_std(),
        }
    }

    fn effective_tokens(&self, now: Instant) -> (u64, u64) {
        let interval_nanos = self.refill_interval.as_nanos();
        if interval_nanos == 0 || self.cur_tokens >= self.max_tokens {
            return (self.cur_tokens, 0);
        }
        let elapsed_nanos = now.saturating_duration_since(self.last_refill).as_nanos();
        let ticks = (elapsed_nanos / interval_nanos).min(u64::MAX as u128) as u64;
        let tokens = self.cur_tokens.saturating_add(ticks).min(self.max_tokens);
        (tokens, ticks)
    }

    fn sync_refill(&mut self, now: Instant) {
        if self.cur_tokens >= self.max_tokens {
            self.last_refill = now;
            return;
        }
        let (tokens, ticks) = self.effective_tokens(now);
        if tokens >= self.max_tokens {
            self.cur_tokens = self.max_tokens;
            self.last_refill = now;
        } else if ticks > 0 {
            self.cur_tokens = tokens;
            let advance = self
                .refill_interval
                .saturating_mul(u32::try_from(ticks).unwrap_or(u32::MAX));
            self.last_refill = self
                .last_refill
                .checked_add(advance)
                .filter(|&t| t <= now)
                .unwrap_or(now);
        }
    }
}

impl RetryThrottler for RefillingCircuitBreaker {
    fn throttle_retry_attempt(&self) -> bool {
        let now = tokio::time::Instant::now().into_std();
        let (tokens, _) = self.effective_tokens(now);
        tokens <= self.min_tokens
    }

    fn on_retry_failure(&mut self, flow: &RetryResult) {
        let now = tokio::time::Instant::now().into_std();
        self.sync_refill(now);
        match flow {
            RetryResult::Continue(_) | RetryResult::Exhausted(_) => {
                self.cur_tokens = self.cur_tokens.saturating_sub(self.error_cost);
            },
            RetryResult::Permanent(_) => {
                self.cur_tokens = self.cur_tokens.saturating_add(1).min(self.max_tokens);
            },
        }
    }

    fn on_success(&mut self) {
        let now = tokio::time::Instant::now().into_std();
        self.sync_refill(now);
        self.cur_tokens = self.cur_tokens.saturating_add(1).min(self.max_tokens);
    }
}

/// Shared time-refilling circuit breaker for a BigQuery v2 `JobService` client
/// (or a client's detached `jobs.cancel` pool).
pub(crate) fn job_service_retry_throttler() -> SharedRetryThrottler {
    RetryThrottlerArg::from(RefillingCircuitBreaker::new(
        CIRCUIT_BREAKER_MAX_TOKENS,
        CIRCUIT_BREAKER_MIN_TOKENS,
        CIRCUIT_BREAKER_ERROR_COST,
        CIRCUIT_BREAKER_REFILL_INTERVAL,
    ))
    .into()
}

/// Equal-jitter exponential backoff policy for BigQuery v2 `JobService` REST
/// requests (`jobs.query`, `jobs.get`): `[cap/2, cap]` for `1s * 2^(attempt-1)`
/// capped at `60s`.
pub(crate) fn job_service_backoff_policy() -> Arc<dyn BackoffPolicy> {
    Arc::new(EqualJitterBackoff::new(
        JOB_SERVICE_INITIAL_DELAY,
        JOB_SERVICE_MAXIMUM_DELAY,
        JOB_SERVICE_SCALING_FACTOR,
    ))
}

/// Equal-jitter exponential backoff policy for detached best-effort `jobs.cancel`
/// retries: `[cap/2, cap]` for `100ms * 2^(attempt-1)` capped at `2s`.
pub(crate) fn job_service_cancel_backoff_policy() -> Arc<dyn BackoffPolicy> {
    Arc::new(EqualJitterBackoff::new(
        JOB_CANCEL_INITIAL_DELAY,
        JOB_CANCEL_MAXIMUM_DELAY,
        JOB_CANCEL_SCALING_FACTOR,
    ))
}

/// Default retry policy attached to the BigQuery v2 `JobService` client
/// (used for `jobs.get` and `jobs.query` dry runs).
pub(crate) fn job_service_retry_policy() -> Arc<dyn RetryPolicy> {
    Arc::new(
        JobServiceRetryableErrors
            .with_time_limit(JOB_SERVICE_RETRY_TIMEOUT)
            .with_attempt_limit(JOB_SERVICE_MAX_ATTEMPTS),
    )
}

/// Retry policy for `jobs.query` execution calls (`Query::read`).
///
/// Caps retry initiation to [`JOB_SERVICE_RETRY_TIMEOUT`] (`600 s`, strictly
/// inside BigQuery's 15-minute `request_id` TTL), rejects retrying client-side
/// timeouts, and preserves the caller's full per-attempt timeout without
/// truncating below `timeoutMs + buffer`.
pub(crate) fn query_read_retry_policy() -> Arc<dyn RetryPolicy> {
    Arc::new(QueryReadRetryPolicy {
        max_retry_window: JOB_SERVICE_RETRY_TIMEOUT,
        max_attempts: JOB_SERVICE_MAX_ATTEMPTS,
    })
}

/// Retry policy for detached best-effort `jobs.cancel` requests, bounded to
/// [`JOB_CANCEL_MAX_ATTEMPTS`] (3) within the remaining `time_limit`.
pub(crate) fn job_service_cancel_retry_policy(time_limit: Duration) -> Arc<dyn RetryPolicy> {
    Arc::new(
        JobServiceRetryableErrors
            .with_time_limit(time_limit)
            .with_attempt_limit(JOB_CANCEL_MAX_ATTEMPTS),
    )
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

    fn http_error_with_status_and_reason(http_code: u16, status_name: &str, reason: &str) -> Error {
        let body = serde_json::json!({
            "error": {
                "code": http_code,
                "message": "err",
                "errors": [{"reason": reason, "domain": "global", "message": "err"}],
                "status": status_name
            }
        });
        let status = Status::try_from(&bytes::Bytes::from(body.to_string()))
            .expect("status json should parse");
        Error::service_with_http_metadata(status, Some(http_code), None)
    }

    fn http_403_with_reason(reason: &str) -> Error {
        http_error_with_status_and_reason(403, "PERMISSION_DENIED", reason)
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

        // Timeouts ARE retried on fast metadata RPCs (`jobs.get`, dry runs).
        assert!(policy
            .on_error(
                &state.clone().set_attempt_count(1_u32),
                Error::timeout("simulated timeout"),
            )
            .is_continue());

        // Throttled retries fail fast instead of spinning until the time limit expires.
        assert!(matches!(
            policy.on_throttle(&state.clone().set_attempt_count(1_u32), transient_error()),
            ThrottleResult::Exhausted(_)
        ));

        // Advance Tokio's paused clock past the budget (safe on fresh Windows VMs).
        tokio::time::advance(JOB_SERVICE_RETRY_TIMEOUT + Duration::from_secs(1)).await;
        assert!(policy
            .on_error(&state.clone().set_attempt_count(1_u32), transient_error())
            .is_exhausted());
    }

    #[tokio::test(start_paused = true)]
    async fn test_query_read_retry_policy_bounds_ttl_and_skips_timeout_retries() {
        let policy = query_read_retry_policy();
        let start = tokio::time::Instant::now().into_std();
        let state = RetryState::new(true).set_start(start);

        // Must return `None` so `google-cloud-gax` does not clamp `attempt_timeout`
        // below `timeoutMs + buffer` on retries or long `initial_wait` calls.
        assert_eq!(policy.remaining_time(&state), None);

        // Client-side timeout after waiting out `timeoutMs` must NOT retry, and must
        // return `Exhausted` (not `Permanent`) so the circuit breaker records the failure.
        let timeout_flow = policy.on_error(
            &state.clone().set_attempt_count(1_u32),
            Error::timeout("client waited full timeoutMs"),
        );
        assert!(timeout_flow.is_exhausted());

        // Fast transient 503 retries up to JOB_SERVICE_MAX_ATTEMPTS.
        for attempt in 1..JOB_SERVICE_MAX_ATTEMPTS {
            let err_503 = Error::http(503, Default::default(), bytes::Bytes::new());
            assert!(policy
                .on_error(&state.clone().set_attempt_count(attempt), err_503)
                .is_continue());
        }
        let err_503 = Error::http(503, Default::default(), bytes::Bytes::new());
        assert!(policy
            .on_error(
                &state.clone().set_attempt_count(JOB_SERVICE_MAX_ATTEMPTS),
                err_503
            )
            .is_exhausted());

        // Throttled retries fail fast.
        assert!(matches!(
            policy.on_throttle(&state.clone().set_attempt_count(1_u32), transient_error()),
            ThrottleResult::Exhausted(_)
        ));

        // Never retry past `JOB_SERVICE_RETRY_TIMEOUT` (safely inside 15-min request_id TTL).
        tokio::time::advance(JOB_SERVICE_RETRY_TIMEOUT).await;
        assert!(policy
            .on_error(
                &state.clone().set_attempt_count(1_u32),
                Error::http(503, Default::default(), bytes::Bytes::new()),
            )
            .is_exhausted());
    }

    #[test]
    fn test_job_service_retry_classification_and_backoff() {
        let policy = job_service_retry_policy();
        let state = RetryState::new(true).set_attempt_count(1_u32);

        // HTTP 408 Request Timeout is retryable.
        let timeout_408 = Error::http(408, Default::default(), bytes::Bytes::new());
        assert!(policy.on_error(&state, timeout_408).is_continue());

        // Legacy camelCase AND AIP-193 UPPER_SNAKE_CASE 403 rate-limit/backend reasons retry.
        for reason in [
            "rateLimitExceeded",
            "RATE_LIMIT_EXCEEDED",
            "userRateLimitExceeded",
            "USER_RATE_LIMIT_EXCEEDED",
            "jobRateLimitExceeded",
            "JOB_RATE_LIMIT_EXCEEDED",
            "backendError",
            "BACKEND_ERROR",
            "internalError",
            "INTERNAL_ERROR",
        ] {
            assert!(
                policy
                    .on_error(&state, http_403_with_reason(reason))
                    .is_continue(),
                "expected 403 {reason} to be retryable"
            );
        }

        // Genuine 403 permission/quota errors are permanent EVEN WHEN `status` maps
        // to a transient RPC code such as `RESOURCE_EXHAUSTED` or `ABORTED`.
        assert!(policy
            .on_error(&state, http_403_with_reason("accessDenied"))
            .is_permanent());
        assert!(policy
            .on_error(&state, http_403_with_reason("billingNotEnabled"))
            .is_permanent());
        assert!(policy
            .on_error(
                &state,
                http_error_with_status_and_reason(403, "RESOURCE_EXHAUSTED", "quotaExceeded"),
            )
            .is_permanent());
        assert!(policy
            .on_error(
                &state,
                http_error_with_status_and_reason(403, "ABORTED", "billingTierLimitExceeded"),
            )
            .is_permanent());
        assert!(policy
            .on_error(
                &state,
                http_error_with_status_and_reason(409, "ABORTED", "duplicate"),
            )
            .is_permanent());
        assert!(policy
            .on_error(
                &state,
                http_error_with_status_and_reason(400, "INTERNAL", "invalidQuery"),
            )
            .is_permanent());

        // Equal-jitter backoff stays strictly inside `[cap / 2, cap]`.
        let backoff = job_service_backoff_policy();
        for _ in 0..20 {
            let d1 = backoff.on_failure(&RetryState::new(true).set_attempt_count(1_u32));
            assert!(d1 >= Duration::from_millis(500) && d1 <= Duration::from_secs(1));
            let d4 = backoff.on_failure(&RetryState::new(true).set_attempt_count(4_u32));
            assert!(d4 >= Duration::from_secs(4) && d4 <= Duration::from_secs(8));
            let d_hi = backoff.on_failure(&RetryState::new(true).set_attempt_count(20_u32));
            assert!(d_hi >= Duration::from_secs(30) && d_hi <= Duration::from_secs(60));
        }

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

        // Cancel backoff uses equal jitter `[cap / 2, cap]` bounded inside 2s.
        let cancel_backoff = job_service_cancel_backoff_policy();
        let cd1 = cancel_backoff.on_failure(&RetryState::new(true).set_attempt_count(1_u32));
        assert!(cd1 >= Duration::from_millis(50) && cd1 <= Duration::from_millis(100));
    }

    #[tokio::test(start_paused = true)]
    async fn test_refilling_circuit_breaker_opens_on_burst_and_recovers_over_time() {
        let mut breaker = RefillingCircuitBreaker::new(
            CIRCUIT_BREAKER_MAX_TOKENS,
            CIRCUIT_BREAKER_MIN_TOKENS,
            CIRCUIT_BREAKER_ERROR_COST,
            CIRCUIT_BREAKER_REFILL_INTERVAL,
        );

        // Fresh client accepts transient retries immediately.
        assert!(!breaker.throttle_retry_attempt());

        // 74 failures: 1000 -> 260 (> 250 min_tokens), still open for retries.
        for _ in 0..74 {
            breaker.on_retry_failure(&RetryResult::Continue(transient_error()));
        }
        assert!(!breaker.throttle_retry_attempt());

        // 75th failure reaches 250 (<= min_tokens): circuit breaker opens.
        breaker.on_retry_failure(&RetryResult::Continue(transient_error()));
        assert!(breaker.throttle_retry_attempt());

        // Continue draining to 0 during prolonged outage.
        for _ in 0..30 {
            breaker.on_retry_failure(&RetryResult::Exhausted(transient_error()));
        }
        assert_eq!(breaker.cur_tokens, 0);
        assert!(breaker.throttle_retry_attempt());

        // Wait 125 s (`250` tokens refilled at 1/500ms -> still `<= 250` throttled).
        tokio::time::advance(Duration::from_secs(125)).await;
        assert!(breaker.throttle_retry_attempt());

        // 500 ms later (`251` tokens > 250): breaker passively closes without requiring
        // 251 consecutive successful requests.
        tokio::time::advance(Duration::from_millis(500)).await;
        assert!(!breaker.throttle_retry_attempt());

        // Advancing a huge duration clamps cleanly at `max_tokens` (no overflow).
        tokio::time::advance(Duration::from_secs(3600)).await;
        breaker.on_success();
        assert_eq!(breaker.cur_tokens, CIRCUIT_BREAKER_MAX_TOKENS);
    }

    #[tokio::test(start_paused = true)]
    async fn test_refilling_circuit_breaker_preserves_fractional_ticks() {
        let mut breaker = RefillingCircuitBreaker::new(100, 50, 10, Duration::from_millis(500));
        breaker.on_retry_failure(&RetryResult::Continue(transient_error()));
        assert_eq!(breaker.cur_tokens, 90);

        // Two sub-interval failures separated by 300 ms each (600 ms total > 500 ms)
        // must yield +1 passive refill token rather than discarding fractional time.
        tokio::time::advance(Duration::from_millis(300)).await;
        breaker.on_retry_failure(&RetryResult::Permanent(permanent_error())); // +1 -> 91
        tokio::time::advance(Duration::from_millis(300)).await;
        breaker.on_success(); // +1 passive refill (at 600ms >= 500ms) + 1 success -> 93
        assert_eq!(breaker.cur_tokens, 93);
    }
}
