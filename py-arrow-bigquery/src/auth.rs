use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use chrono::Utc;
use google_cloud_auth::credentials::{CacheableResource, CredentialsProvider, EntityTag};
use google_cloud_auth::errors::CredentialsError;
use pyo3::exceptions::{
    PyAttributeError, PyException, PyImportError, PyKeyError, PyNotImplementedError, PySyntaxError,
    PyTypeError, PyValueError,
};
use pyo3::prelude::*;

/// Duration before actual token expiration at which we trigger a non-blocking
/// background refresh while continuing to serve the cached token (Stale-While-Revalidate).
///
/// Set to 120 seconds so it sits well inside `google-auth`'s 225-second refresh threshold,
/// guaranteeing that Python's `google-auth` issues a fresh token when invoked.
const SOFT_EXPIRY_BUFFER: Duration = Duration::from_secs(120);

/// Duration before actual token expiration at which the cached token is no longer
/// served and callers must wait for a synchronous refresh.
const HARD_EXPIRY_BUFFER: Duration = Duration::from_secs(60);

/// Default upper bound on how long a single Python credentials refresh call may take
/// before timing out and releasing the refresh lock.
const DEFAULT_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Initial cooldown before retrying a failed background soft-expiry refresh.
const DEFAULT_SOFT_REFRESH_BASE_COOLDOWN: Duration = Duration::from_secs(1);

/// Maximum cooldown between failed background soft-expiry refresh attempts.
const MAX_SOFT_REFRESH_COOLDOWN: Duration = Duration::from_secs(30);

/// Classifies whether an exception raised by the Python credentials callable is likely
/// transient (e.g., network/metadata server/HTTP 5xx error) or permanent (programming/type/config bug).
fn is_transient_python_error(py: Python<'_>, err: &PyErr) -> bool {
    // Non-Exception BaseExceptions (KeyboardInterrupt, SystemExit, GeneratorExit) must never retry.
    if !err.is_instance_of::<PyException>(py) {
        return false;
    }

    // Programming, type, and structural configuration errors will never succeed on retry.
    if err.is_instance_of::<PyTypeError>(py)
        || err.is_instance_of::<PyValueError>(py)
        || err.is_instance_of::<PyKeyError>(py)
        || err.is_instance_of::<PyAttributeError>(py)
        || err.is_instance_of::<PyImportError>(py)
        || err.is_instance_of::<PyNotImplementedError>(py)
        || err.is_instance_of::<PySyntaxError>(py)
    {
        return false;
    }

    // Transport, runtime, OS, and custom google.auth.exceptions.* errors are treated as transient.
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Freshness {
    /// Token is well within its validity window (`now < soft_expiry` or non-expiring).
    Fresh,
    /// Token is still valid for requests (`soft_expiry <= now < hard_expiry`),
    /// but a background refresh should be triggered.
    SoftExpired,
    /// Token is expired or too close to expiring (`now >= hard_expiry`); must refresh before use.
    HardExpired,
}

#[derive(Clone, Copy)]
struct TokenDeadlines {
    soft_expiry: Instant,
    hard_expiry: Instant,
}

impl TokenDeadlines {
    fn from_utc_expiry(
        expiry: chrono::DateTime<Utc>,
        now_utc: chrono::DateTime<Utc>,
        now_instant: Instant,
    ) -> Self {
        let remaining = (expiry - now_utc).to_std().unwrap_or(Duration::ZERO);
        Self {
            soft_expiry: now_instant + remaining.saturating_sub(SOFT_EXPIRY_BUFFER),
            hard_expiry: now_instant + remaining.saturating_sub(HARD_EXPIRY_BUFFER),
        }
    }

    fn freshness(&self, now: Instant) -> Freshness {
        if now < self.soft_expiry {
            Freshness::Fresh
        } else if now < self.hard_expiry {
            Freshness::SoftExpired
        } else {
            Freshness::HardExpired
        }
    }
}

#[derive(Clone)]
struct CachedToken {
    headers: http::HeaderMap,
    deadlines: Option<TokenDeadlines>,
    entity_tag: EntityTag,
}

impl CachedToken {
    fn freshness(&self, now: Instant) -> Freshness {
        match &self.deadlines {
            Some(deadlines) => deadlines.freshness(now),
            None => Freshness::Fresh,
        }
    }

    fn to_cacheable_resource(
        &self,
        extensions: &http::Extensions,
    ) -> CacheableResource<http::HeaderMap> {
        match extensions.get::<EntityTag>() {
            Some(tag) if self.entity_tag == *tag => CacheableResource::NotModified,
            _ => CacheableResource::New {
                entity_tag: self.entity_tag.clone(),
                data: self.headers.clone(),
            },
        }
    }
}

#[derive(Default)]
struct CacheState {
    token: Option<CachedToken>,
    soft_refresh_failures: u32,
    next_soft_refresh_at: Option<Instant>,
}

impl CacheState {
    fn can_attempt_soft_refresh(&self, now: Instant) -> bool {
        self.next_soft_refresh_at.is_none_or(|t| now >= t)
    }

    fn record_soft_refresh_failure(&mut self, now: Instant, base_cooldown: Duration) -> Duration {
        let shift = self.soft_refresh_failures.min(5);
        self.soft_refresh_failures = self.soft_refresh_failures.saturating_add(1);
        let cooldown = base_cooldown
            .saturating_mul(1u32 << shift)
            .min(MAX_SOFT_REFRESH_COOLDOWN);
        self.next_soft_refresh_at = Some(now + cooldown);
        cooldown
    }
}

struct PythonTokenSourceInner {
    /// The Python callable (e.g., a function or method) that returns a tuple of
    /// `(token_data, expiration_timestamp_float_or_none)`.
    provider: Py<PyAny>,
    /// Concurrent read-optimized cache for the retrieved token headers and soft-refresh backoff state.
    cache: RwLock<CacheState>,
    /// Async mutex serializing token refresh calls so concurrent callers coalesce
    /// onto a single Python invocation instead of triggering a thundering herd.
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
    /// Maximum duration a single Python refresh invocation may run before timing out.
    refresh_timeout: Duration,
    /// Base cooldown applied after a failed soft-expiry background refresh.
    soft_refresh_base_cooldown: Duration,
}

impl PythonTokenSourceInner {
    fn fetch_from_python(&self) -> Result<CachedToken, CredentialsError> {
        Python::attach(|py| -> Result<CachedToken, CredentialsError> {
            let provider = self.provider.bind(py);
            let result = provider.call0().map_err(|err| {
                let is_transient = is_transient_python_error(py, &err);
                CredentialsError::new(
                    is_transient,
                    format!("Python credentials provider failed: {err}"),
                    err,
                )
            })?;

            // result is (token_data, expiration)
            let tuple = result.cast::<pyo3::types::PyTuple>().map_err(|_| {
                CredentialsError::from_msg(
                    false,
                    "Python credentials provider must return a 2-tuple (token_data, expiration)",
                )
            })?;

            let token_data = tuple.get_item(0).map_err(|err| {
                CredentialsError::new(
                    false,
                    "Python credentials provider tuple missing token_data at index 0",
                    err,
                )
            })?;

            let expiration = tuple.get_item(1).map_err(|err| {
                CredentialsError::new(
                    false,
                    "Python credentials provider tuple missing expiration at index 1",
                    err,
                )
            })?;

            let bearer_token: String = token_data
                .get_item("bearer_token")
                .map_err(|err| {
                    CredentialsError::new(false, "token_data missing 'bearer_token' key", err)
                })?
                .cast::<pyo3::types::PyString>()
                .map_err(|_| CredentialsError::from_msg(false, "'bearer_token' must be a string"))?
                .to_str()
                .map_err(|err| {
                    CredentialsError::new(false, "'bearer_token' is not valid UTF-8", err)
                })?
                .to_string();

<<<<<<< HEAD
=======
            if bearer_token.is_empty() {
                return Err(CredentialsError::from_msg(
                    false,
                    "invalid bearer token header value",
                ));
            }

>>>>>>> upstream/main
            // expiration is a float/int (timestamp) or None
            let deadlines = if expiration.is_none() {
                None
            } else {
                let expiry_f: f64 = expiration.extract().map_err(|err| {
                    CredentialsError::new(
                        false,
                        "expiration must be a float/int timestamp or None",
                        err,
                    )
                })?;

                if !expiry_f.is_finite() || expiry_f < 0.0 {
                    return Err(CredentialsError::from_msg(
                        false,
                        "expiration timestamp is out of range",
                    ));
                }

                let secs = expiry_f.trunc() as i64;
                let nsecs = (expiry_f.fract() * 1_000_000_000.0) as u32;
                let dt = chrono::DateTime::from_timestamp(secs, nsecs).ok_or_else(|| {
                    CredentialsError::from_msg(false, "expiration timestamp is out of range")
                })?;
                Some(TokenDeadlines::from_utc_expiry(
                    dt,
                    Utc::now(),
                    Instant::now(),
                ))
            };

            let mut header_value =
                http::header::HeaderValue::from_str(&format!("Bearer {bearer_token}")).map_err(
                    |err| CredentialsError::new(false, "invalid bearer token header value", err),
                )?;
            header_value.set_sensitive(true);

            let mut headers = http::HeaderMap::new();
            headers.insert(http::header::AUTHORIZATION, header_value);

            Ok(CachedToken {
                headers,
                deadlines,
                entity_tag: EntityTag::new(),
            })
        })
    }

    /// Executes a token refresh inside a detached task that owns `guard`.
    ///
    /// Cancellation & timeout guarantees:
    /// - If the caller's `headers()` future is cancelled/dropped mid-refresh, the detached task
    ///   continues holding `guard` until the Python call finishes (or times out) and writes the
    ///   freshly minted token into `self.cache`, preventing both concurrent Python invocations
    ///   and lost token updates.
    /// - If the Python callable hangs longer than `self.refresh_timeout`, the timeout branch sets
<<<<<<< HEAD
    ///   `cancelled = true` (preventing a late completion from overwriting newer tokens), releases
=======
    ///   `timed_out = true` (preventing a late completion from overwriting newer tokens), releases
>>>>>>> upstream/main
    ///   `guard` so subsequent callers are not deadlocked, and returns a transient `CredentialsError`.
    async fn refresh_token_with_guard(
        self: &Arc<Self>,
        guard: tokio::sync::OwnedMutexGuard<()>,
    ) -> Result<CacheableResource<http::HeaderMap>, CredentialsError> {
        let inner = Arc::clone(self);
        let refresh_timeout = self.refresh_timeout;

        let detached = tokio::spawn(async move {
            let _guard = guard;
            let timed_out = Arc::new(AtomicBool::new(false));
            let timed_out_for_worker = Arc::clone(&timed_out);
            let inner_for_worker = Arc::clone(&inner);

            let blocking = tokio::task::spawn_blocking(move || {
                let cached = inner_for_worker.fetch_from_python()?;
                if timed_out_for_worker.load(Ordering::Acquire) {
                    return Err(CredentialsError::from_msg(
                        true,
                        "Python credentials provider refresh completed after timeout",
                    ));
                }
                let response = CacheableResource::New {
                    entity_tag: cached.entity_tag.clone(),
                    data: cached.headers.clone(),
                };
                let mut cache = inner_for_worker.cache.write().map_err(|_| {
                    CredentialsError::from_msg(false, "token cache lock is poisoned")
                })?;
                if !timed_out_for_worker.load(Ordering::Acquire) {
                    cache.token = Some(cached);
                    cache.soft_refresh_failures = 0;
                    cache.next_soft_refresh_at = None;
                }
                Ok(response)
            });

            match tokio::time::timeout(refresh_timeout, blocking).await {
                Ok(join_res) => join_res.map_err(|err| {
                    CredentialsError::new(true, "Python token refresh task failed", err)
                })?,
                Err(_) => {
                    timed_out.store(true, Ordering::Release);
                    Err(CredentialsError::from_msg(
                        true,
                        format!(
                            "Python credentials provider timed out after {}ms",
                            refresh_timeout.as_millis()
                        ),
                    ))
                },
            }
        });

        detached.await.map_err(|err| {
            CredentialsError::new(true, "Python token refresh coordinator task failed", err)
        })?
    }
}

/// A token source that delegates authentication to a Python callable.
///
/// This struct implements the [`CredentialsProvider`] trait, allowing the Rust
/// Google Cloud SDK to retrieve OAuth2 tokens by calling back into Python code
/// (e.g., using `google-auth`).
///
/// Concurrency and performance characteristics:
/// - **Happy path**: Uses a shared [`RwLock`] read lock and monotonic [`Instant`] expiry
///   comparisons, avoiding mutex contention across concurrent BigQuery streams, clock-skew
///   issues, and Python GIL acquisition.
/// - **Stale-While-Revalidate**: When a token enters [`SOFT_EXPIRY_BUFFER`] before [`HARD_EXPIRY_BUFFER`],
///   the cached token is returned immediately while a single background task refreshes the cache,
///   with exponential backoff on failure to prevent background task storms.
/// - **Refresh coalescing & cancellation safety**: Concurrent callers on a cold or hard-expired
///   cache serialize on an async [`tokio::sync::Mutex`] with double-checked locking, run the refresh
///   in a cancellation-safe detached task with a hard timeout, and execute Python on Tokio's
///   blocking pool (`spawn_blocking`) so async I/O workers never park on the GIL.
pub(crate) struct PythonTokenSource {
    inner: Arc<PythonTokenSourceInner>,
}

impl PythonTokenSource {
    pub(crate) fn new(provider: Py<PyAny>) -> Self {
        Self::new_with_config(
            provider,
            DEFAULT_REFRESH_TIMEOUT,
            DEFAULT_SOFT_REFRESH_BASE_COOLDOWN,
        )
    }

    pub(crate) fn new_with_config(
        provider: Py<PyAny>,
        refresh_timeout: Duration,
        soft_refresh_base_cooldown: Duration,
    ) -> Self {
        Self {
            inner: Arc::new(PythonTokenSourceInner {
                provider,
                cache: RwLock::new(CacheState::default()),
                refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
                refresh_timeout,
                soft_refresh_base_cooldown,
            }),
        }
    }
}

impl std::fmt::Debug for PythonTokenSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonTokenSource")
            .field("provider", &"<python_callable>")
            .finish()
    }
}

impl CredentialsProvider for PythonTokenSource {
    async fn headers(
        &self,
        extensions: http::Extensions,
    ) -> Result<CacheableResource<http::HeaderMap>, CredentialsError> {
        let now = Instant::now();

        // 1. Happy path: shared read lock on the cache + monotonic Instant check.
        let (freshness, soft_resource, can_attempt_soft_refresh) = {
            let cache = self
                .inner
                .cache
                .read()
                .map_err(|_| CredentialsError::from_msg(false, "token cache lock is poisoned"))?;
            match cache.token.as_ref() {
                Some(cached) => match cached.freshness(now) {
                    Freshness::Fresh => return Ok(cached.to_cacheable_resource(&extensions)),
                    Freshness::SoftExpired => (
                        Freshness::SoftExpired,
                        Some(cached.to_cacheable_resource(&extensions)),
                        cache.can_attempt_soft_refresh(now),
                    ),
                    Freshness::HardExpired => (Freshness::HardExpired, None, false),
                },
                None => (Freshness::HardExpired, None, false),
            }
        };

        // 2. Stale-While-Revalidate: return the still-valid cached token immediately
        //    and spawn at most one non-blocking background refresh if not in failure cooldown.
        if let (Freshness::SoftExpired, Some(resource)) = (freshness, soft_resource) {
            if can_attempt_soft_refresh {
                if let Ok(guard) = Arc::clone(&self.inner.refresh_lock).try_lock_owned() {
                    let inner = Arc::clone(&self.inner);
                    tokio::spawn(async move {
                        let should_refresh = inner
                            .cache
                            .read()
                            .ok()
                            .map(|c| {
                                let now = Instant::now();
                                c.can_attempt_soft_refresh(now)
                                    && c.token
                                        .as_ref()
                                        .is_some_and(|t| t.freshness(now) != Freshness::Fresh)
                            })
                            .unwrap_or(false);
                        if should_refresh {
                            if let Err(err) = inner.refresh_token_with_guard(guard).await {
                                let cooldown = inner
                                    .cache
                                    .write()
                                    .ok()
                                    .map(|mut c| {
                                        c.record_soft_refresh_failure(
                                            Instant::now(),
                                            inner.soft_refresh_base_cooldown,
                                        )
                                    })
                                    .unwrap_or(inner.soft_refresh_base_cooldown);
                                tracing::warn!(
                                    error = %err,
                                    backoff_ms = cooldown.as_millis() as u64,
                                    "Background soft-expiry token refresh failed; continuing to serve cached token until hard expiry"
                                );
                            }
                        }
                    });
                }
            }
            return Ok(resource);
        }

        // 3. Cold or hard-expired path: coalesce concurrent refreshes via async Mutex.
        let refresh_guard = Arc::clone(&self.inner.refresh_lock).lock_owned().await;

        // Double-checked locking: check if another task refreshed the token while we waited.
        {
            let cache = self
                .inner
                .cache
                .read()
                .map_err(|_| CredentialsError::from_msg(false, "token cache lock is poisoned"))?;
            if let Some(cached) = cache.token.as_ref() {
                if cached.freshness(Instant::now()) != Freshness::HardExpired {
                    return Ok(cached.to_cacheable_resource(&extensions));
                }
            }
        }

        self.inner.refresh_token_with_guard(refresh_guard).await
    }

    async fn universe_domain(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use pyo3::types::PyDict;

    use super::*;

    #[test]
    fn test_python_token_source_caches_and_not_modified() {
        Python::initialize();
        let (provider, counter) = Python::attach(|py| {
            let locals = PyDict::new(py);
            let future_ts = (Utc::now() + chrono::Duration::seconds(3600)).timestamp() as f64;
            locals.set_item("future_ts", future_ts).unwrap();
            py.run(
                c"
class Counter:
    def __init__(self):
        self.count = 0
    def __call__(self):
        self.count += 1
        return ({'bearer_token': f'secret-token-{self.count}'}, future_ts)
counter = Counter()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let c = locals.get_item("counter").unwrap().unwrap().unbind();
            let c_clone = c.clone_ref(py);
            (c, c_clone)
        });

        let source = PythonTokenSource::new(provider);

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            assert_eq!(source.universe_domain().await, None);

            let res1 = source.headers(http::Extensions::new()).await.unwrap();
            let (tag1, headers1) = match res1 {
                CacheableResource::New { entity_tag, data } => (entity_tag, data),
                CacheableResource::NotModified => panic!("expected New"),
            };
            let auth_val = headers1.get(http::header::AUTHORIZATION).unwrap();
            assert_eq!(auth_val.to_str().unwrap(), "Bearer secret-token-1");
            assert!(auth_val.is_sensitive());

            // Debug output must not leak token
            let debug_str = format!("{source:?}");
            assert!(!debug_str.contains("secret-token-1"));

            // Second call without EntityTag should return cached headers without calling Python again
            let res2 = source.headers(http::Extensions::new()).await.unwrap();
            let (tag2, headers2) = match res2 {
                CacheableResource::New { entity_tag, data } => (entity_tag, data),
                CacheableResource::NotModified => panic!("expected New"),
            };
            assert_eq!(tag1, tag2);
            assert_eq!(
                headers2
                    .get(http::header::AUTHORIZATION)
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "Bearer secret-token-1"
            );

            // Call with matching EntityTag should return NotModified
            let mut ext = http::Extensions::new();
            ext.insert(tag1);
            let res3 = source.headers(ext).await.unwrap();
            assert!(matches!(res3, CacheableResource::NotModified));
        });

        let count: i32 = Python::attach(|py| {
            counter
                .bind(py)
                .getattr("count")
                .unwrap()
                .extract()
                .unwrap()
        });
        assert_eq!(count, 1);
    }

    #[test]
    fn test_python_token_source_refreshes_when_expiring_soon() {
        Python::initialize();
        let provider = Python::attach(|py| {
            let locals = PyDict::new(py);
            // First token expires in 30s (within the 60s hard expiry window), second in 3600s
            let expiring_soon = (Utc::now() + chrono::Duration::seconds(30)).timestamp() as f64;
            let future_ts = (Utc::now() + chrono::Duration::seconds(3600)).timestamp() as f64;
            locals.set_item("expiring_soon", expiring_soon).unwrap();
            locals.set_item("future_ts", future_ts).unwrap();
            py.run(
                c"
class ExpiringProvider:
    def __init__(self):
        self.count = 0
    def __call__(self):
        self.count += 1
        ts = expiring_soon if self.count == 1 else future_ts
        return ({'bearer_token': f'token-{self.count}'}, ts)
provider = ExpiringProvider()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            locals.get_item("provider").unwrap().unwrap().unbind()
        });

        let source = PythonTokenSource::new(provider);

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            let res1 = source.headers(http::Extensions::new()).await.unwrap();
            let tag1 = match res1 {
                CacheableResource::New { entity_tag, data } => {
                    assert_eq!(
                        data.get(http::header::AUTHORIZATION)
                            .unwrap()
                            .to_str()
                            .unwrap(),
                        "Bearer token-1"
                    );
                    entity_tag
                },
                CacheableResource::NotModified => panic!("expected New"),
            };

            // Since token-1 expires in <60s (hard-expired), the next call refreshes synchronously
            let res2 = source.headers(http::Extensions::new()).await.unwrap();
            match res2 {
                CacheableResource::New { entity_tag, data } => {
                    assert_ne!(entity_tag, tag1);
                    assert_eq!(
                        data.get(http::header::AUTHORIZATION)
                            .unwrap()
                            .to_str()
                            .unwrap(),
                        "Bearer token-2"
                    );
                },
                CacheableResource::NotModified => panic!("expected New"),
            }
        });
    }

    #[test]
    fn test_python_token_source_stale_while_revalidate_and_failure_cooldown() {
        Python::initialize();
        let (provider, counter) = Python::attach(|py| {
            let locals = PyDict::new(py);
            // First token expires in 90s: between HARD_EXPIRY_BUFFER (60s) and SOFT_EXPIRY_BUFFER (120s).
            let soft_expiring = (Utc::now() + chrono::Duration::seconds(90)).timestamp() as f64;
            let future_ts = (Utc::now() + chrono::Duration::seconds(3600)).timestamp() as f64;
            locals.set_item("soft_expiring", soft_expiring).unwrap();
            locals.set_item("future_ts", future_ts).unwrap();
            py.run(
                c"
class SoftExpiringProvider:
    def __init__(self):
        self.count = 0
        self.fail_next = False
    def __call__(self):
        self.count += 1
        if self.fail_next:
            raise RuntimeError('transient metadata blip')
        ts = soft_expiring if self.count == 1 else future_ts
        return ({'bearer_token': f'token-{self.count}'}, ts)
provider = SoftExpiringProvider()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let p = locals.get_item("provider").unwrap().unwrap().unbind();
            let p_clone = p.clone_ref(py);
            (p, p_clone)
        });

<<<<<<< HEAD
        // Configure a 80ms base cooldown so we can test both cooldown throttling and recovery
=======
        // Configure an 80ms base cooldown so we can test both cooldown throttling and recovery
>>>>>>> upstream/main
        let source = PythonTokenSource::new_with_config(
            provider,
            Duration::from_secs(5),
            Duration::from_millis(80),
        );

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            // 1. Cold start fetches token-1 (90s TTL -> immediately in soft-expiry window)
            let res1 = source.headers(http::Extensions::new()).await.unwrap();
            let tag1 = match res1 {
                CacheableResource::New { entity_tag, data } => {
                    assert_eq!(
                        data.get(http::header::AUTHORIZATION)
                            .unwrap()
                            .to_str()
                            .unwrap(),
                        "Bearer token-1"
                    );
                    entity_tag
                },
                CacheableResource::NotModified => panic!("expected New"),
            };

            // Inject a transient failure for background refreshes
            Python::attach(|py| {
                counter.bind(py).setattr("fail_next", true).unwrap();
            });

            // 2. Call in soft-expiry window: returns token-1 immediately and triggers 1 background refresh
            let mut ext = http::Extensions::new();
            ext.insert(tag1.clone());
            let res2 = source.headers(ext).await.unwrap();
            assert!(matches!(res2, CacheableResource::NotModified));

            // Wait for the failing background refresh to finish and record its 80ms cooldown
            tokio::time::sleep(Duration::from_millis(25)).await;

            // Multiple calls during the cooldown window must NOT trigger additional Python calls!
            for _ in 0..5 {
                let _ = source.headers(http::Extensions::new()).await.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(15)).await;

            let count_during_cooldown: i32 = Python::attach(|py| {
                counter
                    .bind(py)
                    .getattr("count")
                    .unwrap()
                    .extract()
                    .unwrap()
            });
            assert_eq!(
                count_during_cooldown, 2,
                "soft-expiry cooldown must suppress background retry storms during cooldown window"
            );

            // Clear the failure and wait for the 80ms cooldown to expire
            Python::attach(|py| {
                counter.bind(py).setattr("fail_next", false).unwrap();
            });
            tokio::time::sleep(Duration::from_millis(60)).await;

            // 3. Next call after cooldown expires triggers a background refresh that succeeds (token-3)
            let _ = source.headers(http::Extensions::new()).await.unwrap();

            for _ in 0..40 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                let res = source.headers(http::Extensions::new()).await.unwrap();
                if let CacheableResource::New { data, .. } = res {
                    if data
                        .get(http::header::AUTHORIZATION)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        == "Bearer token-3"
                    {
                        return;
                    }
                }
            }
            panic!("expected background refresh after cooldown to update cache to token-3");
        });
    }

    #[test]
    fn test_python_token_source_coalesces_concurrent_refreshes() {
        Python::initialize();
        let (provider, counter) = Python::attach(|py| {
            let locals = PyDict::new(py);
            let future_ts = (Utc::now() + chrono::Duration::seconds(3600)).timestamp() as f64;
            locals.set_item("future_ts", future_ts).unwrap();
            py.run(
                c"
import time
class SlowProvider:
    def __init__(self):
        self.count = 0
    def __call__(self):
        time.sleep(0.05)
        self.count += 1
        return ({'bearer_token': f'coalesced-token-{self.count}'}, future_ts)
provider = SlowProvider()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let p = locals.get_item("provider").unwrap().unwrap().unbind();
            let p_clone = p.clone_ref(py);
            (p, p_clone)
        });

        let source = Arc::new(PythonTokenSource::new(provider));

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        // Detach the main thread from the GIL while Tokio's blocking pool runs SlowProvider
        Python::attach(|py| {
            py.detach(|| {
                rt.block_on(async {
                    let mut handles = Vec::new();
                    for _ in 0..16 {
                        let src = Arc::clone(&source);
                        handles.push(tokio::spawn(async move {
                            src.headers(http::Extensions::new()).await.unwrap()
                        }));
                    }

                    for h in handles {
                        let res = h.await.unwrap();
                        match res {
                            CacheableResource::New { data, .. } => {
                                assert_eq!(
                                    data.get(http::header::AUTHORIZATION)
                                        .unwrap()
                                        .to_str()
                                        .unwrap(),
                                    "Bearer coalesced-token-1"
                                );
                            },
                            CacheableResource::NotModified => panic!("expected New"),
                        }
                    }
                });
            });
        });

        let count: i32 = Python::attach(|py| {
            counter
                .bind(py)
                .getattr("count")
                .unwrap()
                .extract()
                .unwrap()
        });
        assert_eq!(
            count, 1,
            "16 concurrent requests on cold cache must coalesce into 1 Python call"
        );
    }

    #[test]
    fn test_python_token_source_cancellation_safety_preserves_inflight_refresh() {
        Python::initialize();
        let (provider, counter) = Python::attach(|py| {
            let locals = PyDict::new(py);
            let future_ts = (Utc::now() + chrono::Duration::seconds(3600)).timestamp() as f64;
            locals.set_item("future_ts", future_ts).unwrap();
            py.run(
                c"
import time
class SlowProvider:
    def __init__(self):
        self.count = 0
    def __call__(self):
        time.sleep(0.06)
        self.count += 1
        return ({'bearer_token': f'saved-token-{self.count}'}, future_ts)
provider = SlowProvider()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let p = locals.get_item("provider").unwrap().unwrap().unbind();
            let p_clone = p.clone_ref(py);
            (p, p_clone)
        });

        let source = Arc::new(PythonTokenSource::new(provider));
        let rt = pyo3_async_runtimes::tokio::get_runtime();

        Python::attach(|py| {
            py.detach(|| {
                rt.block_on(async {
                    // Caller 1 starts refreshing and is cancelled after 15ms (mid-Python execution)
                    let src1 = Arc::clone(&source);
                    let _ = tokio::time::timeout(
                        Duration::from_millis(15),
                        src1.headers(http::Extensions::new()),
                    )
                    .await;

                    // Caller 2 arrives while Caller 1's detached refresh is still in flight;
                    // it must coalesce onto the same Python invocation and receive saved-token-1!
                    let res2 = source.headers(http::Extensions::new()).await.unwrap();
                    match res2 {
                        CacheableResource::New { data, .. } => {
                            assert_eq!(
                                data.get(http::header::AUTHORIZATION)
                                    .unwrap()
                                    .to_str()
                                    .unwrap(),
                                "Bearer saved-token-1"
                            );
                        },
                        CacheableResource::NotModified => panic!("expected New"),
                    }
                });
            });
        });

        let count: i32 = Python::attach(|py| {
            counter
                .bind(py)
                .getattr("count")
                .unwrap()
                .extract()
                .unwrap()
        });
        assert_eq!(
            count, 1,
            "cancelling caller 1 must not lose the in-flight token or trigger a duplicate Python call"
        );
    }

    #[test]
    fn test_python_token_source_refresh_timeout_releases_lock() {
        Python::initialize();
        let provider = Python::attach(|py| {
            let locals = PyDict::new(py);
            let future_ts = (Utc::now() + chrono::Duration::seconds(3600)).timestamp() as f64;
            locals.set_item("future_ts", future_ts).unwrap();
            py.run(
                c"
import time
class HungThenFastProvider:
    def __init__(self):
        self.count = 0
    def __call__(self):
        self.count += 1
        if self.count == 1:
            time.sleep(0.25)
            return ({'bearer_token': 'stale-timed-out-token'}, future_ts)
        return ({'bearer_token': 'fresh-recovered-token'}, future_ts)
provider = HungThenFastProvider()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            locals.get_item("provider").unwrap().unwrap().unbind()
        });

        let source = PythonTokenSource::new_with_config(
            provider,
            Duration::from_millis(50),
            Duration::from_secs(1),
        );
        let rt = pyo3_async_runtimes::tokio::get_runtime();

        Python::attach(|py| {
            py.detach(|| {
                rt.block_on(async {
                    // First call times out after 50ms and marks the error transient
                    let err = source.headers(http::Extensions::new()).await.unwrap_err();
                    assert!(err.is_transient(), "timeout error must be transient");
                    assert!(err.to_string().contains("timed out"));

                    // Second call immediately acquires the released lock and succeeds
                    let res = source.headers(http::Extensions::new()).await.unwrap();
                    match res {
                        CacheableResource::New { data, .. } => {
                            assert_eq!(
                                data.get(http::header::AUTHORIZATION)
                                    .unwrap()
                                    .to_str()
                                    .unwrap(),
                                "Bearer fresh-recovered-token"
                            );
                        },
                        CacheableResource::NotModified => panic!("expected New"),
                    }

                    // Wait for the hung first call to finish in the background and verify it
                    // did NOT overwrite fresh-recovered-token.
                    tokio::time::sleep(Duration::from_millis(230)).await;
                    let res_after = source.headers(http::Extensions::new()).await.unwrap();
                    if let CacheableResource::New { data, .. } = res_after {
                        assert_eq!(
                            data.get(http::header::AUTHORIZATION)
                                .unwrap()
                                .to_str()
                                .unwrap(),
                            "Bearer fresh-recovered-token"
                        );
                    }
                });
            });
        });
    }

    #[test]
    fn test_python_token_source_none_expiration() {
        Python::initialize();
        let (provider, counter) = Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
class NonExpiringProvider:
    def __init__(self):
        self.count = 0
    def __call__(self):
        self.count += 1
        return ({'bearer_token': 'forever-token'}, None)
provider = NonExpiringProvider()
",
                None,
                Some(&locals),
            )
            .unwrap();
            let p = locals.get_item("provider").unwrap().unwrap().unbind();
            let p_clone = p.clone_ref(py);
            (p, p_clone)
        });

        let source = PythonTokenSource::new(provider);

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            let res1 = source.headers(http::Extensions::new()).await.unwrap();
            match res1 {
                CacheableResource::New { data, .. } => {
                    assert_eq!(
                        data.get(http::header::AUTHORIZATION)
                            .unwrap()
                            .to_str()
                            .unwrap(),
                        "Bearer forever-token"
                    );
                },
                CacheableResource::NotModified => panic!("expected New"),
            }

            let _ = source.headers(http::Extensions::new()).await.unwrap();
        });

        let count: i32 = Python::attach(|py| {
            counter
                .bind(py)
                .getattr("count")
                .unwrap()
                .extract()
                .unwrap()
        });
        assert_eq!(count, 1);
    }

    #[test]
    fn test_python_token_source_errors_classification_and_causal_source() {
        Python::initialize();
        let cases = [
            // Runtime / transport errors from Python callable -> transient = true
            (
                "def p(): raise RuntimeError('metadata 503 unavailable')",
                "metadata 503 unavailable",
                true,
                true,
            ),
            (
                "def p(): raise ConnectionError('connection reset')",
                "connection reset",
                true,
                true,
            ),
            // Programming / type / value errors from Python callable -> transient = false
            (
                "def p(): raise ValueError('invalid credentials config')",
                "invalid credentials config",
                false,
                true,
            ),
            (
                "def p(): raise TypeError('bad callable signature')",
                "bad callable signature",
                false,
                true,
            ),
            // Malformed return values -> transient = false
            ("def p(): return 'not-a-tuple'", "2-tuple", false, false),
            (
                "def p(): return ({}, 1700000000.0)",
                "bearer_token",
                false,
                true,
            ),
<<<<<<< HEAD
=======
            (
                "def p(): return ({'bearer_token': ''}, 1700000000.0)",
                "invalid bearer token",
                false,
                false,
            ),
            (
                "def p(): return ({'bearer_token': 'bad\\ntoken'}, 1700000000.0)",
                "invalid bearer token",
                false,
                true,
            ),
            (
                "def p(): return ({'bearer_token': 'tok'}, -1.0)",
                "out of range",
                false,
                false,
            ),
>>>>>>> upstream/main
        ];

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        for (code, expected_msg, expected_transient, expect_source) in cases {
            let provider = Python::attach(|py| {
                let locals = PyDict::new(py);
                let c_code = std::ffi::CString::new(code).unwrap();
                py.run(&c_code, None, Some(&locals)).unwrap();
                locals.get_item("p").unwrap().unwrap().unbind()
            });
            let source = PythonTokenSource::new(provider);
            rt.block_on(async {
                let err = source.headers(http::Extensions::new()).await.unwrap_err();
                assert_eq!(
                    err.is_transient(),
                    expected_transient,
                    "unexpected is_transient() for `{code}`"
                );
                assert!(
                    err.to_string().contains(expected_msg),
                    "expected '{}' in '{}'",
                    expected_msg,
                    err
                );
                if expect_source {
                    assert!(
                        err.source().is_some(),
                        "expected CredentialsError to preserve underlying PyErr as source"
                    );
                }
            });
        }
    }
}
