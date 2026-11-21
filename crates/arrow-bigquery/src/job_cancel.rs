//! Bounded, best-effort server-side cancellation (`jobs.cancel`) for incomplete
//! BigQuery query jobs.
//!
//! Uses an explicit atomic state machine ([`JobCancelState`]) so:
//! - at most one `jobs.cancel` HTTP flow is scheduled per incomplete job,
//! - confirming the job is `DONE` (`jobs.get`, clean EOF, or `BigQueryError::JobFailed`)
//!   permanently disarms cancellation—even *after* a detached cancel task has been
//!   enqueued and is waiting on the concurrency semaphore, preventing redundant RPCs,
//! - pending queue depth ([`MAX_PENDING_JOB_CANCELS`]) is mathematically bounded to
//!   fit inside [`JOB_CANCEL_TIMEOUT`] (`10 s`), avoiding dead wave build-ups.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use google_cloud_gax::options::RequestOptionsBuilder;
use google_cloud_gax::retry_throttler::SharedRetryThrottler;

use crate::{apply_request_options_parts, job_service_retry, BigQueryJobService};

/// Overall timeout budget applied to detached best-effort `jobs.cancel` tasks
/// (covers semaphore wait, token refresh, and up to `JOB_CANCEL_MAX_ATTEMPTS` tries).
pub(crate) const JOB_CANCEL_TIMEOUT: Duration = Duration::from_secs(10);

/// Per-attempt HTTP timeout applied to each `jobs.cancel` try inside [`JOB_CANCEL_TIMEOUT`].
///
/// Sized at `3 s` so all `JOB_CANCEL_MAX_ATTEMPTS` (3) tries plus their short
/// equal-jitter backoff sleeps (`<= 0.3 s`) fit inside [`JOB_CANCEL_TIMEOUT`] (`10 s`).
pub(crate) const JOB_CANCEL_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(3);

/// Minimum remaining time budget inside [`JOB_CANCEL_TIMEOUT`] required to launch
/// a queued `jobs.cancel` HTTP request after waiting for a concurrency permit.
///
/// Prevents firing doomed TLS/HTTP connections with a few milliseconds remaining.
pub(crate) const MIN_CANCEL_ATTEMPT_TIMEOUT: Duration = Duration::from_millis(500);

/// Maximum number of concurrent in-flight detached `jobs.cancel` HTTP calls per [`crate::Client`].
pub(crate) const MAX_CONCURRENT_JOB_CANCELS: usize = 16;

/// Number of full [`JOB_CANCEL_ATTEMPT_TIMEOUT`] waves that can execute within
/// [`JOB_CANCEL_TIMEOUT`] (`10s / 3s = 3`).
const MAX_QUEUED_CANCEL_WAVES: usize =
    (JOB_CANCEL_TIMEOUT.as_secs() / JOB_CANCEL_ATTEMPT_TIMEOUT.as_secs()) as usize;

/// Maximum number of detached `jobs.cancel` tasks (queued + in-flight) buffered per [`crate::Client`].
///
/// Aligned with [`JOB_CANCEL_TIMEOUT`] / [`JOB_CANCEL_ATTEMPT_TIMEOUT`] (`16 * 3 = 48`):
/// 1 active wave plus 2 queued waves fit cleanly within the `10 s` deadline even when
/// requests consume their full `3 s` attempt timeout. Excess cancels during bulk drop
/// storms are shed synchronously with a `tracing::warn!`.
pub(crate) const MAX_PENDING_JOB_CANCELS: usize =
    MAX_CONCURRENT_JOB_CANCELS * MAX_QUEUED_CANCEL_WAVES;

const _: () = assert!(
    JOB_CANCEL_ATTEMPT_TIMEOUT.as_millis() * (job_service_retry::JOB_CANCEL_MAX_ATTEMPTS as u128)
        <= JOB_CANCEL_TIMEOUT.as_millis(),
    "all cancel attempts must fit inside JOB_CANCEL_TIMEOUT"
);
const _: () = assert!(
    MAX_QUEUED_CANCEL_WAVES >= 1,
    "at least one wave of job cancels must fit inside JOB_CANCEL_TIMEOUT"
);
const _: () = assert!(
    MIN_CANCEL_ATTEMPT_TIMEOUT.as_millis() <= JOB_CANCEL_ATTEMPT_TIMEOUT.as_millis(),
    "MIN_CANCEL_ATTEMPT_TIMEOUT cannot exceed JOB_CANCEL_ATTEMPT_TIMEOUT"
);

/// Returns `true` when BigQuery reports the job has reached a terminal state
/// (`status.error_result` is set or `status.state` is `"DONE"`).
pub(crate) fn is_job_done(job: &google_cloud_bigquery_v2::model::Job) -> bool {
    job.status.as_ref().is_some_and(|status| {
        status.error_result.is_some() || status.state.eq_ignore_ascii_case("DONE")
    })
}

/// Lifecycle state of a single cancellable BigQuery query job.
///
/// ```text
///                 disarm() / job confirmed DONE
///   ┌───────────────────────────────────────────────┐
///   │                                               ▼
/// Armed ──(trigger / enqueue)──► Queued ──────► Disarmed
///                                  │               ▲
///                                  │ disarm() while│
///                                  │ waiting for   │
///                                  │ permit ───────┘
///                                  ▼
///                               Sending
/// ```
///
/// Transitions are strictly monotonic so each job can issue at most one
/// `jobs.cancel` sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum JobCancelState {
    /// Job has not been confirmed complete; dropping or aborting the owning handle
    /// will schedule a detached `jobs.cancel` task.
    Armed = 0,
    /// A detached `jobs.cancel` task has been enqueued and is awaiting a concurrency
    /// permit before starting the HTTP RPC. Can still transition to [`Self::Disarmed`].
    Queued = 1,
    /// Detached task acquired its concurrency permit and began issuing the `jobs.cancel` RPC.
    Sending = 2,
    /// Job is confirmed complete (clean EOF, `JobFailed`, or `jobs.get` `DONE`).
    /// Cancellation is permanently suppressed.
    Disarmed = 3,
}

impl JobCancelState {
    fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Armed,
            1 => Self::Queued,
            2 => Self::Sending,
            3 => Self::Disarmed,
            other => unreachable!("invalid JobCancelState discriminant: {other}"),
        }
    }
}

/// Result of attempting to transition [`JobCancelState::Armed`] -> [`JobCancelState::Queued`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueueTransition {
    Queued,
    AlreadyDisarmed,
    AlreadyTriggered(JobCancelState),
}

/// Result of attempting to transition [`JobCancelState::Queued`] -> [`JobCancelState::Sending`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendTransition {
    Proceed,
    Skipped(JobCancelState),
}

/// Result of scheduling or executing a detached `jobs.cancel` task.
#[derive(Debug)]
enum CancelTaskOutcome {
    DisarmedBeforeSend,
    ConcurrencyLimiterClosed,
    InsufficientBudget { remaining: Duration },
    Completed,
    Failed(google_cloud_bigquery_v2::Error),
    TimedOut,
}

/// Lock-free, cloneable handle to a query job's [`JobCancelState`] state machine.
///
/// Shared between a [`JobCancelHandle`] (owned by [`JobCancelGuard`] /
/// [`crate::BigQueryRecordBatchReceiver`]) and asynchronous `jobs.get` inspection
/// points (`DestinationTableRead::resolve`, `resolve_empty_job_stream`,
/// `resolve_job_error`).
#[derive(Clone, Debug)]
pub(crate) struct JobCancelToken {
    state: Arc<AtomicU8>,
}

impl JobCancelToken {
    pub(crate) fn armed() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(JobCancelState::Armed as u8)),
        }
    }

    #[cfg(test)]
    pub(crate) fn disarmed() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(JobCancelState::Disarmed as u8)),
        }
    }

    pub(crate) fn state(&self) -> JobCancelState {
        JobCancelState::from_raw(self.state.load(Ordering::Acquire))
    }

    pub(crate) fn is_disarmed(&self) -> bool {
        self.state() == JobCancelState::Disarmed
    }

    /// Permanently suppresses cancellation when the job is still [`JobCancelState::Armed`]
    /// or [`JobCancelState::Queued`] (before the HTTP call starts).
    ///
    /// Returns the terminal observed state after the transition attempt.
    pub(crate) fn disarm(&self) -> JobCancelState {
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            match JobCancelState::from_raw(current) {
                JobCancelState::Armed | JobCancelState::Queued => {
                    match self.state.compare_exchange_weak(
                        current,
                        JobCancelState::Disarmed as u8,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => return JobCancelState::Disarmed,
                        Err(actual) => current = actual,
                    }
                },
                terminal @ (JobCancelState::Sending | JobCancelState::Disarmed) => {
                    return terminal;
                },
            }
        }
    }

    /// Disarms cancellation if `job` reports a terminal state (`DONE` or `error_result`).
    /// Returns `true` when the job was confirmed done.
    pub(crate) fn disarm_if_job_done(&self, job: &google_cloud_bigquery_v2::model::Job) -> bool {
        if is_job_done(job) {
            self.disarm();
            true
        } else {
            false
        }
    }

    fn try_queue(&self) -> QueueTransition {
        match self.state.compare_exchange(
            JobCancelState::Armed as u8,
            JobCancelState::Queued as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => QueueTransition::Queued,
            Err(actual) => match JobCancelState::from_raw(actual) {
                JobCancelState::Disarmed => QueueTransition::AlreadyDisarmed,
                other => QueueTransition::AlreadyTriggered(other),
            },
        }
    }

    fn try_begin_send(&self) -> SendTransition {
        match self.state.compare_exchange(
            JobCancelState::Queued as u8,
            JobCancelState::Sending as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => SendTransition::Proceed,
            Err(actual) => SendTransition::Skipped(JobCancelState::from_raw(actual)),
        }
    }
}

/// Shared bounded concurrency/queue pool and isolated circuit breaker used by a
/// [`crate::Client`] (and all clones) for detached best-effort `jobs.cancel` calls.
#[derive(Clone, Debug)]
pub(crate) struct JobCancelPool {
    inflight_limiter: Arc<tokio::sync::Semaphore>,
    queue_limiter: Arc<tokio::sync::Semaphore>,
    retry_throttler: SharedRetryThrottler,
}

impl JobCancelPool {
    pub(crate) fn new() -> Self {
        Self {
            inflight_limiter: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_JOB_CANCELS)),
            queue_limiter: Arc::new(tokio::sync::Semaphore::new(MAX_PENDING_JOB_CANCELS)),
            retry_throttler: job_service_retry::job_service_retry_throttler(),
        }
    }

    #[cfg(test)]
    pub(crate) fn retry_throttler(&self) -> &SharedRetryThrottler {
        &self.retry_throttler
    }

    #[cfg(test)]
    pub(crate) fn queue_limiter(&self) -> &Arc<tokio::sync::Semaphore> {
        &self.queue_limiter
    }

    #[cfg(test)]
    pub(crate) fn inflight_limiter(&self) -> &Arc<tokio::sync::Semaphore> {
        &self.inflight_limiter
    }
}

/// Best-effort server-side cancellation handle for an incomplete `jobs.query` job.
///
/// Holds only the REST `JobService` transport pieces needed for `jobs.cancel`
/// (never pinning the Storage Read API gRPC channel after a client closes).
/// Dropping a bare [`JobCancelHandle`] without invoking [`Self::trigger`] is a
/// no-op; ownership triggers are mediated by [`JobCancelGuard`] and [`JobCancelSlot`].
pub(crate) struct JobCancelHandle {
    bq_client: BigQueryJobService,
    quota_project_id: Arc<str>,
    user_agent: Option<Arc<str>>,
    project_id: String,
    job_id: String,
    location: String,
    runtime: tokio::runtime::Handle,
    pool: JobCancelPool,
    token: JobCancelToken,
}

impl JobCancelHandle {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        bq_client: BigQueryJobService,
        quota_project_id: Arc<str>,
        user_agent: Option<Arc<str>>,
        project_id: impl Into<String>,
        job_id: impl Into<String>,
        location: impl Into<String>,
        runtime: tokio::runtime::Handle,
        pool: JobCancelPool,
        token: JobCancelToken,
    ) -> Self {
        Self {
            bq_client,
            quota_project_id,
            user_agent,
            project_id: project_id.into(),
            job_id: job_id.into(),
            location: location.into(),
            runtime,
            pool,
            token,
        }
    }

    pub(crate) fn disarm(self) {
        self.token.disarm();
    }

    #[cfg(test)]
    pub(crate) fn retry_throttler(&self) -> &SharedRetryThrottler {
        self.pool.retry_throttler()
    }

    /// Schedules a bounded detached `jobs.cancel` task unless the paired
    /// [`JobCancelToken`] is already disarmed or triggered, or the pending
    /// cancel queue is full.
    pub(crate) fn trigger(self) {
        if self.token.is_disarmed() {
            return;
        }
        let Ok(queue_permit) = Arc::clone(&self.pool.queue_limiter).try_acquire_owned() else {
            tracing::warn!(
                project_id = %self.project_id,
                job_id = %self.job_id,
                location = %self.location,
                max_pending = MAX_PENDING_JOB_CANCELS,
                "dropping best-effort BigQuery jobs.cancel: pending cancel queue is full"
            );
            return;
        };
        match self.token.try_queue() {
            QueueTransition::Queued => {},
            QueueTransition::AlreadyDisarmed | QueueTransition::AlreadyTriggered(_) => return,
        }

        // Capture the wall-clock deadline when enqueuing so executor queue delay is
        // counted inside the overall `JOB_CANCEL_TIMEOUT` budget.
        let deadline = tokio::time::Instant::now() + JOB_CANCEL_TIMEOUT;
        let Self {
            bq_client,
            quota_project_id,
            user_agent,
            project_id,
            job_id,
            location,
            runtime,
            pool,
            token,
        } = self;

        runtime.spawn(async move {
            let _queue_permit = queue_permit;
            let outcome = match tokio::time::timeout_at(
                deadline,
                execute_cancel(
                    &bq_client,
                    &quota_project_id,
                    user_agent.as_deref(),
                    &project_id,
                    &job_id,
                    &location,
                    &pool,
                    &token,
                    deadline,
                ),
            )
            .await
            {
                Ok(outcome) => outcome,
                Err(_elapsed) => CancelTaskOutcome::TimedOut,
            };

            match outcome {
                CancelTaskOutcome::Completed => {},
                CancelTaskOutcome::DisarmedBeforeSend => {
                    tracing::debug!(
                        project_id = %project_id,
                        job_id = %job_id,
                        location = %location,
                        "skipping queued BigQuery jobs.cancel: job already finished"
                    );
                },
                CancelTaskOutcome::ConcurrencyLimiterClosed => {},
                CancelTaskOutcome::InsufficientBudget { remaining } => {
                    tracing::warn!(
                        project_id = %project_id,
                        job_id = %job_id,
                        location = %location,
                        remaining_ms = remaining.as_millis() as u64,
                        min_attempt_timeout_ms = MIN_CANCEL_ATTEMPT_TIMEOUT.as_millis() as u64,
                        "dropping best-effort BigQuery jobs.cancel: insufficient remaining budget after queue wait"
                    );
                },
                CancelTaskOutcome::Failed(err) => {
                    tracing::warn!(
                        project_id = %project_id,
                        job_id = %job_id,
                        location = %location,
                        error = %err,
                        "best-effort BigQuery jobs.cancel failed"
                    );
                },
                CancelTaskOutcome::TimedOut => {
                    tracing::warn!(
                        project_id = %project_id,
                        job_id = %job_id,
                        location = %location,
                        timeout_secs = JOB_CANCEL_TIMEOUT.as_secs(),
                        "best-effort BigQuery jobs.cancel timed out"
                    );
                },
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_cancel(
    bq_client: &BigQueryJobService,
    quota_project_id: &str,
    user_agent: Option<&str>,
    project_id: &str,
    job_id: &str,
    location: &str,
    pool: &JobCancelPool,
    token: &JobCancelToken,
    deadline: tokio::time::Instant,
) -> CancelTaskOutcome {
    if token.is_disarmed() {
        return CancelTaskOutcome::DisarmedBeforeSend;
    }

    let Ok(_inflight_permit) = Arc::clone(&pool.inflight_limiter).acquire_owned().await else {
        return CancelTaskOutcome::ConcurrencyLimiterClosed;
    };

    // Re-check both cancellation state and remaining budget AFTER waiting for a
    // concurrency permit so a concurrently completed job (or starved task) never
    // emits a spurious or guaranteed-to-timeout HTTP call.
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    if remaining < MIN_CANCEL_ATTEMPT_TIMEOUT {
        return CancelTaskOutcome::InsufficientBudget { remaining };
    }

    match token.try_begin_send() {
        SendTransition::Proceed => {},
        SendTransition::Skipped(_) => return CancelTaskOutcome::DisarmedBeforeSend,
    }

    let cancel_req = apply_request_options_parts(
        bq_client
            .cancel_job()
            .set_project_id(project_id)
            .set_job_id(job_id)
            .set_location(location)
            .with_idempotency(true),
        quota_project_id,
        user_agent,
    )
    .with_attempt_timeout(JOB_CANCEL_ATTEMPT_TIMEOUT.min(remaining))
    .with_retry_policy(job_service_retry::job_service_cancel_retry_policy(
        remaining,
    ))
    .with_backoff_policy(job_service_retry::job_service_cancel_backoff_policy())
    .with_retry_throttler(Arc::clone(&pool.retry_throttler));

    match cancel_req.send().await {
        Ok(_) => CancelTaskOutcome::Completed,
        Err(err) => CancelTaskOutcome::Failed(err),
    }
}

/// Inner state enum for [`JobCancelGuard`].
enum GuardState {
    /// Actively tracks an incomplete job; dropping triggers cancellation unless disarmed.
    Armed(JobCancelHandle),
    /// Inactive: no runtime handle was available, explicitly disarmed, or transferred.
    Inactive,
}

/// Explicit RAII guard guarding an incomplete query job while awaiting the
/// initial `streams/_default` schema/first-page response.
///
/// Dropping an armed guard invokes [`JobCancelHandle::trigger`].
pub(crate) struct JobCancelGuard {
    state: GuardState,
}

impl JobCancelGuard {
    pub(crate) fn arm(handle: JobCancelHandle) -> Self {
        Self {
            state: GuardState::Armed(handle),
        }
    }

    pub(crate) fn inactive() -> Self {
        Self {
            state: GuardState::Inactive,
        }
    }

    /// Permanently disarms the guard so dropping it becomes a no-op.
    pub(crate) fn disarm(&mut self) {
        if let GuardState::Armed(handle) = std::mem::replace(&mut self.state, GuardState::Inactive)
        {
            handle.disarm();
        }
    }

    /// Transfers ownership of the armed handle (if any) out of the guard.
    pub(crate) fn take(&mut self) -> Option<JobCancelHandle> {
        match std::mem::replace(&mut self.state, GuardState::Inactive) {
            GuardState::Armed(handle) => Some(handle),
            GuardState::Inactive => None,
        }
    }
}

impl Drop for JobCancelGuard {
    fn drop(&mut self) {
        if let GuardState::Armed(handle) = std::mem::replace(&mut self.state, GuardState::Inactive)
        {
            handle.trigger();
        }
    }
}

/// Compact ownership slot stored on [`crate::BigQueryRecordBatchReceiver`].
///
/// `Box<JobCancelHandle>` avoids bloating non-query (`read_table` / `read_stream`)
/// receivers when inactive.
pub(crate) enum JobCancelSlot {
    /// No server-side cancellation handle is attached (or already disarmed/triggered).
    Inactive,
    /// Armed with a best-effort cancel handle for an incomplete query job.
    Armed(Box<JobCancelHandle>),
}

impl JobCancelSlot {
    pub(crate) fn arm(&mut self, handle: JobCancelHandle) {
        *self = Self::Armed(Box::new(handle));
    }

    pub(crate) fn disarm(&mut self) {
        if let Self::Armed(handle) = std::mem::replace(self, Self::Inactive) {
            handle.disarm();
        }
    }

    pub(crate) fn trigger(&mut self) {
        if let Self::Armed(handle) = std::mem::replace(self, Self::Inactive) {
            handle.trigger();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use google_cloud_bigquery_v2::model::{
        CancelJobRequest, ErrorProto, Job, JobCancelResponse, JobStatus,
    };
    use google_cloud_gax::options::RequestOptions;
    use google_cloud_gax::response::Response;

    use super::*;
    use crate::tests::{test_bq_client, test_read_client};

    #[test]
    fn test_job_cancel_token_state_machine_transitions() {
        // Happy path: Armed -> Queued -> Sending (disarm after Sending is a no-op).
        let tok = JobCancelToken::armed();
        assert_eq!(tok.state(), JobCancelState::Armed);
        assert_eq!(tok.try_queue(), QueueTransition::Queued);
        assert_eq!(tok.state(), JobCancelState::Queued);
        assert_eq!(
            tok.try_queue(),
            QueueTransition::AlreadyTriggered(JobCancelState::Queued)
        );
        assert_eq!(tok.try_begin_send(), SendTransition::Proceed);
        assert_eq!(tok.state(), JobCancelState::Sending);
        assert_eq!(tok.disarm(), JobCancelState::Sending);
        assert_eq!(
            tok.try_begin_send(),
            SendTransition::Skipped(JobCancelState::Sending)
        );

        // Disarm before trigger: Armed -> Disarmed.
        let tok_early = JobCancelToken::armed();
        assert_eq!(tok_early.disarm(), JobCancelState::Disarmed);
        assert!(tok_early.is_disarmed());
        assert_eq!(tok_early.try_queue(), QueueTransition::AlreadyDisarmed);

        // Concurrent disarm while queued: Armed -> Queued -> Disarmed blocks send.
        let tok_queued = JobCancelToken::armed();
        assert_eq!(tok_queued.try_queue(), QueueTransition::Queued);
        assert_eq!(tok_queued.disarm(), JobCancelState::Disarmed);
        assert_eq!(
            tok_queued.try_begin_send(),
            SendTransition::Skipped(JobCancelState::Disarmed)
        );
    }

    #[test]
    fn test_disarm_if_job_done() {
        let running = Job::new().set_status(JobStatus::new().set_state("RUNNING"));
        let done = Job::new().set_status(JobStatus::new().set_state("done"));
        let errored = Job::new().set_status(
            JobStatus::new()
                .set_state("RUNNING")
                .set_error_result(ErrorProto::new().set_message("boom")),
        );

        let tok = JobCancelToken::armed();
        assert!(!tok.disarm_if_job_done(&running));
        assert_eq!(tok.state(), JobCancelState::Armed);

        assert!(tok.disarm_if_job_done(&done));
        assert_eq!(tok.state(), JobCancelState::Disarmed);

        let tok_err = JobCancelToken::armed();
        assert!(tok_err.disarm_if_job_done(&errored));
        assert_eq!(tok_err.state(), JobCancelState::Disarmed);
    }

    #[tokio::test]
    async fn test_job_cancel_shares_throttler_and_bounds_pending_queue() {
        let client = crate::Client::new_for_testing(
            test_bq_client().await,
            test_read_client().await,
            "test-quota-proj".to_string(),
        );
        let h1 = client.new_job_cancel_handle(
            "p",
            "j1",
            "US",
            tokio::runtime::Handle::current(),
            JobCancelToken::armed(),
        );
        let h2 = client.new_job_cancel_handle(
            "p",
            "j2",
            "US",
            tokio::runtime::Handle::current(),
            JobCancelToken::armed(),
        );
        assert!(
            Arc::ptr_eq(h1.retry_throttler(), h2.retry_throttler()),
            "detached jobs.cancel tasks must share one circuit breaker across a Client"
        );
        assert!(
            !Arc::ptr_eq(h1.retry_throttler(), client.read_retry_throttler()),
            "cancel circuit breaker must be isolated from Storage Read throttler"
        );

        // Drain all queue permits to verify `trigger()` sheds excess cancels synchronously
        // without spawning unbounded background tasks or panicking.
        let permits = client
            .cancel_pool()
            .queue_limiter()
            .try_acquire_many(MAX_PENDING_JOB_CANCELS as u32)
            .expect("all queue permits should be available");
        assert_eq!(client.cancel_pool().queue_limiter().available_permits(), 0);
        h1.trigger();
        // Dropping `h2` cleanly without calling `trigger()` sends nothing.
        drop(h2);
        drop(permits);
        assert_eq!(
            client.cancel_pool().queue_limiter().available_permits(),
            MAX_PENDING_JOB_CANCELS
        );
        // Queue bound equals aligned wave count: 16 * (10 / 3) = 48.
        assert_eq!(MAX_PENDING_JOB_CANCELS, 48);
        assert!(!client
            .cancel_pool()
            .retry_throttler()
            .lock()
            .unwrap()
            .throttle_retry_attempt());
    }

    #[tokio::test]
    async fn test_queued_cancel_skips_rpc_when_disarmed_before_permit_acquired() {
        #[derive(Debug)]
        struct CountingCancelStub {
            calls: Arc<AtomicUsize>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for CountingCancelStub {
            async fn cancel_job(
                &self,
                _req: CancelJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<JobCancelResponse>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(Response::from(JobCancelResponse::new()))
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let client = crate::Client::new_for_testing(
            crate::BigQueryJobService::from_stub(CountingCancelStub {
                calls: Arc::clone(&calls),
            }),
            test_read_client().await,
            "p".to_string(),
        );

        // Hold all inflight permits so the triggered cancel task blocks in `Queued`.
        let inflight_permits = client
            .cancel_pool()
            .inflight_limiter()
            .try_acquire_many(MAX_CONCURRENT_JOB_CANCELS as u32)
            .expect("all inflight permits should be available");

        let token = JobCancelToken::armed();
        let handle = client.new_job_cancel_handle(
            "p",
            "job-race",
            "US",
            tokio::runtime::Handle::current(),
            token.clone(),
        );
        handle.trigger();
        assert_eq!(token.state(), JobCancelState::Queued);

        // Let the spawned task park awaiting `inflight_limiter`.
        tokio::task::yield_now().await;

        // Concurrent `jobs.get` confirms job is DONE while cancel task is still queued.
        let done_job = Job::new().set_status(JobStatus::new().set_state("DONE"));
        assert!(token.disarm_if_job_done(&done_job));
        assert_eq!(token.state(), JobCancelState::Disarmed);

        // Release permits and allow the task to drain.
        drop(inflight_permits);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "queued cancel task must skip RPC when disarmed while waiting on semaphore"
        );
        assert_eq!(
            client.cancel_pool().queue_limiter().available_permits(),
            MAX_PENDING_JOB_CANCELS
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_queued_cancel_drops_when_remaining_budget_below_minimum() {
        #[derive(Debug)]
        struct CountingCancelStub {
            calls: Arc<AtomicUsize>,
        }

        impl google_cloud_bigquery_v2::stub::JobService for CountingCancelStub {
            async fn cancel_job(
                &self,
                _req: CancelJobRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery_v2::Result<Response<JobCancelResponse>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(Response::from(JobCancelResponse::new()))
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let client = crate::Client::new_for_testing(
            crate::BigQueryJobService::from_stub(CountingCancelStub {
                calls: Arc::clone(&calls),
            }),
            test_read_client().await,
            "p".to_string(),
        );

        let inflight_permits = client
            .cancel_pool()
            .inflight_limiter()
            .try_acquire_many(MAX_CONCURRENT_JOB_CANCELS as u32)
            .expect("all inflight permits should be available");

        let token = JobCancelToken::armed();
        let handle = client.new_job_cancel_handle(
            "p",
            "job-starved",
            "US",
            tokio::runtime::Handle::current(),
            token.clone(),
        );
        handle.trigger();
        tokio::task::yield_now().await;

        // Advance paused clock until less than `MIN_CANCEL_ATTEMPT_TIMEOUT` remains.
        tokio::time::advance(JOB_CANCEL_TIMEOUT - Duration::from_millis(100)).await;
        drop(inflight_permits);
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "starved cancel task with < MIN_CANCEL_ATTEMPT_TIMEOUT left must not launch HTTP call"
        );
        assert_eq!(token.state(), JobCancelState::Queued);
        assert_eq!(
            client.cancel_pool().queue_limiter().available_permits(),
            MAX_PENDING_JOB_CANCELS
        );
    }
}
