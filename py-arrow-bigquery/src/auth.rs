use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use chrono::Utc;
use google_cloud_auth::credentials::{CacheableResource, CredentialsProvider, EntityTag};
use google_cloud_auth::errors::CredentialsError;
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

struct PythonTokenSourceInner {
    /// The Python callable (e.g., a function or method) that returns a tuple of
    /// `(token_data, expiration_timestamp_float_or_none)`.
    provider: Py<PyAny>,
    /// Concurrent read-optimized cache for the retrieved token headers.
    cache: RwLock<Option<CachedToken>>,
    /// Async mutex serializing token refresh calls so concurrent callers coalesce
    /// onto a single Python invocation instead of triggering a thundering herd.
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
}

impl PythonTokenSourceInner {
    fn fetch_from_python(&self) -> Result<CachedToken, CredentialsError> {
        Python::attach(|py| -> Result<CachedToken, CredentialsError> {
            let provider = self.provider.bind(py);
            let result = provider.call0().map_err(|err| {
                CredentialsError::new(
                    false,
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

    async fn refresh_token(
        self: &Arc<Self>,
    ) -> Result<CacheableResource<http::HeaderMap>, CredentialsError> {
        let inner = Arc::clone(self);
        let cached = tokio::task::spawn_blocking(move || inner.fetch_from_python())
            .await
            .map_err(|err| {
                CredentialsError::new(false, "Python token refresh task failed", err)
            })??;

        let response = CacheableResource::New {
            entity_tag: cached.entity_tag.clone(),
            data: cached.headers.clone(),
        };

        {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| CredentialsError::from_msg(false, "token cache lock is poisoned"))?;
            *cache = Some(cached);
        }

        Ok(response)
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
///   the cached token is returned immediately while a single background task refreshes the cache.
/// - **Refresh coalescing**: Concurrent callers on a cold or hard-expired cache serialize on an
///   async [`tokio::sync::Mutex`] with double-checked locking and execute the Python call on
///   Tokio's blocking pool (`spawn_blocking`) so async I/O workers never park on the GIL.
pub(crate) struct PythonTokenSource {
    inner: Arc<PythonTokenSourceInner>,
}

impl PythonTokenSource {
    pub(crate) fn new(provider: Py<PyAny>) -> Self {
        Self {
            inner: Arc::new(PythonTokenSourceInner {
                provider,
                cache: RwLock::new(None),
                refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
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
        let (freshness, soft_resource) = {
            let cache =
                self.inner.cache.read().map_err(|_| {
                    CredentialsError::from_msg(false, "token cache lock is poisoned")
                })?;
            match cache.as_ref() {
                Some(cached) => match cached.freshness(now) {
                    Freshness::Fresh => return Ok(cached.to_cacheable_resource(&extensions)),
                    Freshness::SoftExpired => (
                        Freshness::SoftExpired,
                        Some(cached.to_cacheable_resource(&extensions)),
                    ),
                    Freshness::HardExpired => (Freshness::HardExpired, None),
                },
                None => (Freshness::HardExpired, None),
            }
        };

        // 2. Stale-While-Revalidate: return the still-valid cached token immediately
        //    and spawn at most one non-blocking background refresh.
        if let (Freshness::SoftExpired, Some(resource)) = (freshness, soft_resource) {
            if let Ok(guard) = Arc::clone(&self.inner.refresh_lock).try_lock_owned() {
                let inner = Arc::clone(&self.inner);
                tokio::spawn(async move {
                    let _guard = guard;
                    let needs_refresh = inner
                        .cache
                        .read()
                        .ok()
                        .and_then(|c| {
                            c.as_ref()
                                .map(|t| t.freshness(Instant::now()) != Freshness::Fresh)
                        })
                        .unwrap_or(false);
                    if needs_refresh {
                        // Ignore errors during background soft-expiry refresh so the still-valid
                        // cached token continues serving requests until hard_expiry.
                        let _ = inner.refresh_token().await;
                    }
                });
            }
            return Ok(resource);
        }

        // 3. Cold or hard-expired path: coalesce concurrent refreshes via async Mutex.
        let _refresh_guard = self.inner.refresh_lock.lock().await;

        // Double-checked locking: check if another task refreshed the token while we waited.
        {
            let cache =
                self.inner.cache.read().map_err(|_| {
                    CredentialsError::from_msg(false, "token cache lock is poisoned")
                })?;
            if let Some(cached) = cache.as_ref() {
                if cached.freshness(Instant::now()) != Freshness::HardExpired {
                    return Ok(cached.to_cacheable_resource(&extensions));
                }
            }
        }

        self.inner.refresh_token().await
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
    fn test_python_token_source_stale_while_revalidate_soft_expiry() {
        Python::initialize();
        let (provider, counter) = Python::attach(|py| {
            let locals = PyDict::new(py);
            // First token expires in 90s: between HARD_EXPIRY_BUFFER (60s) and SOFT_EXPIRY_BUFFER (120s).
            // Second call fails transiently, proving soft-expiry keeps serving token-1.
            // Third call succeeds with a 3600s token.
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
            self.fail_next = False
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

        let source = PythonTokenSource::new(provider);

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

            // Inject a transient failure for the first background refresh
            Python::attach(|py| {
                counter.bind(py).setattr("fail_next", true).unwrap();
            });

            // 2. Call in soft-expiry window: returns token-1 immediately (NotModified when tag matches!)
            //    and triggers background refresh (which fails transiently without evicting token-1).
            let mut ext = http::Extensions::new();
            ext.insert(tag1.clone());
            let res2 = source.headers(ext).await.unwrap();
            assert!(matches!(res2, CacheableResource::NotModified));

            // Wait briefly for the failing background refresh to finish
            tokio::time::sleep(Duration::from_millis(50)).await;

            // 3. Token-1 is still valid in cache; next call returns token-1 and triggers a
            //    second background refresh which succeeds and installs token-3 (3600s TTL).
            let res3 = source.headers(http::Extensions::new()).await.unwrap();
            match res3 {
                CacheableResource::New { data, .. } => {
                    assert_eq!(
                        data.get(http::header::AUTHORIZATION)
                            .unwrap()
                            .to_str()
                            .unwrap(),
                        "Bearer token-1"
                    );
                },
                CacheableResource::NotModified => panic!("expected New"),
            }

            // Wait for the successful background refresh to swap in token-3
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
            panic!("expected background refresh to update cache to token-3");
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
    fn test_python_token_source_errors_and_causal_source() {
        Python::initialize();
        let cases = [
            (
                "def p(): raise RuntimeError('auth failed')",
                "auth failed",
                true,
            ),
            ("def p(): return 'not-a-tuple'", "2-tuple", false),
            ("def p(): return ({}, 1700000000.0)", "bearer_token", true),
        ];

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        for (code, expected_msg, expect_source) in cases {
            let provider = Python::attach(|py| {
                let locals = PyDict::new(py);
                let c_code = std::ffi::CString::new(code).unwrap();
                py.run(&c_code, None, Some(&locals)).unwrap();
                locals.get_item("p").unwrap().unwrap().unbind()
            });
            let source = PythonTokenSource::new(provider);
            rt.block_on(async {
                let err = source.headers(http::Extensions::new()).await.unwrap_err();
                assert!(!err.is_transient());
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
