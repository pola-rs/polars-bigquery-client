mod auth;
mod error;
mod stream;
mod table_id;
#[cfg(feature = "testing")]
mod testing;

use std::cell::Cell;
use std::sync::RwLock;

use google_cloud_auth::credentials::Credentials;
use google_cloud_bigquery::model::arrow_serialization_options::CompressionCodec;
use pyo3::exceptions::{
    PyAttributeError, PyImportError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDateTime, PyInt};
pub use stream::ArrowStreamExporter;
pub use table_id::BigQueryTableId;

thread_local! {
    /// Per-OS-thread cache of `(pid, is_main_python_thread)`.
    /// Keying by `std::process::id()` automatically invalidates entries after `os.fork()`.
    static IS_MAIN_THREAD_CACHE: Cell<Option<(u32, bool)>> = const { Cell::new(None) };
}

/// Fast-path lookup of the cached main-thread status without acquiring the GIL.
pub(crate) fn cached_is_main_python_thread() -> Option<bool> {
    let pid = std::process::id();
    IS_MAIN_THREAD_CACHE.with(|cache| match cache.get() {
        Some((cached_pid, is_main)) if cached_pid == pid => Some(is_main),
        _ => None,
    })
}

/// Returns `true` when called from Python's main thread.
///
/// Outside the main thread `py.check_signals()` is a no-op, so we avoid
/// unnecessary 10 Hz GIL acquisitions during blocking waits on worker threads.
/// Caches the result per OS thread keyed by PID (invalidating across `fork`).
/// Propagates pending exceptions such as `KeyboardInterrupt` instead of silently
/// swallowing them; falls back to `true` only if `threading` lookup fails with
/// `ImportError`/`AttributeError` (e.g., during interpreter teardown).
pub(crate) fn is_main_python_thread(py: Python<'_>) -> PyResult<bool> {
    let pid = std::process::id();
    if let Some(is_main) = cached_is_main_python_thread() {
        return Ok(is_main);
    }

    let lookup = (|| -> PyResult<bool> {
        // Compare thread identifiers instead of calling `current_thread()` so native
        // worker threads (e.g., Rayon/Polars pool threads) do not instantiate and register
        // a CPython `threading._DummyThread` entry inside `threading._active`.
        let threading = py.import("threading")?;
        let current_ident = threading.call_method0("get_ident")?;
        let main_ident = threading.call_method0("main_thread")?.getattr("ident")?;
        if main_ident.is_none() {
            return Ok(true);
        }
        current_ident.eq(&main_ident)
    })();

    match lookup {
        Ok(is_main) => {
            IS_MAIN_THREAD_CACHE.with(|cache| cache.set(Some((pid, is_main))));
            Ok(is_main)
        },
        Err(err)
            if err.is_instance_of::<PyImportError>(py)
                || err.is_instance_of::<PyAttributeError>(py) =>
        {
            Ok(true)
        },
        Err(err) => Err(err),
    }
}

/// Bounded grace wait for an aborted Tokio task to execute its `Drop` hooks
/// (e.g., `CancelOnDrop` firing best-effort `jobs.cancel`) before returning
/// `KeyboardInterrupt` to Python.
const ABORT_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(100);

/// RAII guard that aborts the spawned Tokio task if dropped before completion
/// (for example on panic or unexpected early return).
struct AbortOnDrop(Option<tokio::task::AbortHandle>);

impl AbortOnDrop {
    fn disarm(&mut self) {
        self.0 = None;
    }

    fn abort_now(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.abort_now();
    }
}

/// Drives an async Rust future on the shared Tokio runtime while periodically
/// polling `py.check_signals()` on the Python main thread so `Ctrl+C`
/// (`KeyboardInterrupt`) interrupts long-running waits (such as BigQuery
/// `jobs.query` initial waits, `CreateReadSession` retries, or client setup)
/// promptly instead of blocking uninterruptibly.
///
/// On non-main Python threads (where `py.check_signals()` is a no-op), awaits
/// the future directly without periodic GIL re-acquisition.
fn block_on_interruptible<F, T>(
    py: Python<'_>,
    fut: F,
) -> PyResult<Result<T, arrow_bigquery_lib::BigQueryError>>
where
    F: std::future::Future<Output = Result<T, arrow_bigquery_lib::BigQueryError>> + Send + 'static,
    T: Send + 'static,
{
    // Fail fast on already-pending signals BEFORE spawning any network RPC.
    py.check_signals()?;

    let rt = pyo3_async_runtimes::tokio::get_runtime();
    if !is_main_python_thread(py)? {
        // Spawn onto Tokio on worker threads too so task panics convert consistently
        // into `PyRuntimeError` rather than unwinding as `PanicException`.
        let join_res = py.detach(|| rt.block_on(async move { rt.spawn(fut).await }));
        return join_res.map_err(|join_err| {
            PyRuntimeError::new_err(format!("BigQuery task failed unexpectedly: {join_err}"))
        });
    }

    let handle = rt.spawn(fut);
    let mut abort_guard = AbortOnDrop(Some(handle.abort_handle()));
    let poll_interval = std::time::Duration::from_millis(100);
    tokio::pin!(handle);

    loop {
        if let Err(py_err) = py.check_signals() {
            abort_guard.abort_now();
            // Briefly yield for the cancelled task to unwind and execute `Drop`
            // cleanup (e.g., spawning detached `jobs.cancel` from `CancelOnDrop`)
            // before returning `KeyboardInterrupt` to Python.
            let _ = py.detach(|| {
                rt.block_on(async { tokio::time::timeout(ABORT_DRAIN_TIMEOUT, &mut handle).await })
            });
            return Err(py_err);
        }

        let step = py.detach(|| {
            rt.block_on(async { tokio::time::timeout(poll_interval, &mut handle).await })
        });
        match step {
            Ok(Ok(result)) => {
                abort_guard.disarm();
                return Ok(result);
            },
            Ok(Err(join_err)) => {
                abort_guard.disarm();
                return Err(PyRuntimeError::new_err(format!(
                    "BigQuery task failed unexpectedly: {join_err}"
                )));
            },
            Err(_) => continue,
        }
    }
}

fn extract_maximum_bytes_billed(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<i64>> {
    let Some(obj) = value else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    if obj.is_instance_of::<PyBool>() || !obj.is_instance_of::<PyInt>() {
        let type_name = obj.get_type().name()?;
        return Err(PyTypeError::new_err(format!(
            "maximum_bytes_billed must be an int or None, got {type_name}"
        )));
    }
    match obj.extract::<i64>() {
        Ok(limit) if limit < 0 => Err(PyValueError::new_err(format!(
            "maximum_bytes_billed must be non-negative, got {limit}"
        ))),
        Ok(limit) => Ok(Some(limit)),
        Err(err) if err.is_instance_of::<PyOverflowError>(py) => {
            // Avoid calling `obj.__str__()` on arbitrarily huge Python ints (e.g., `10**5000`),
            // which hits CPython 3.11+'s `sys.get_int_max_str_digits()` limit.
            if obj.lt(0)? {
                Err(PyValueError::new_err(
                    "maximum_bytes_billed must be non-negative",
                ))
            } else {
                Err(PyValueError::new_err(format!(
                    "maximum_bytes_billed must be <= {}",
                    i64::MAX
                )))
            }
        },
        Err(err) => Err(err),
    }
}

/// A Python-exposed client that keeps the BigQuery Storage Read API gRPC channel
/// open across multiple table read operations, caching the OAuth2 token in Rust.
#[pyclass(name = "Client")]
pub struct Client {
    client: RwLock<Option<arrow_bigquery_lib::Client>>,
}

impl Client {
    fn clone_inner(&self) -> PyResult<arrow_bigquery_lib::Client> {
        let guard = self
            .client
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.as_ref().cloned().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot use a closed arrow_bigquery.Client instance.")
        })
    }
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (*, quota_project_id, credentials_provider=None, user_agent=None, bq_endpoint=None, bqstorage_endpoint=None))]
    pub fn new(
        py: Python<'_>,
        quota_project_id: String,
        credentials_provider: Option<Py<PyAny>>,
        user_agent: Option<String>,
        bq_endpoint: Option<String>,
        bqstorage_endpoint: Option<String>,
    ) -> PyResult<Self> {
        let cred = match credentials_provider {
            Some(provider) if !provider.is_none(py) => {
                let token_source = auth::PythonTokenSource::new(provider);
                Some(Credentials::from(token_source))
            },
            _ => None,
        };

        let result = block_on_interruptible(py, async move {
            let mut builder = arrow_bigquery_lib::Client::builder()
                .with_user_agent(user_agent)
                .with_quota_project_id(Some(quota_project_id));
            if let Some(cred) = cred {
                builder = builder.with_cred(cred);
            }
            if let Some(endpoint) = bq_endpoint {
                builder = builder.with_bq_endpoint(endpoint);
            }
            if let Some(endpoint) = bqstorage_endpoint {
                builder = builder.with_bqstorage_endpoint(endpoint);
            }

            arrow_bigquery_lib::Client::from_builder(builder).await
        })?;

        let client = result.map_err(|err| error::to_py_err(py, err))?;
        Ok(Self {
            client: RwLock::new(Some(client)),
        })
    }

    /// Release the underlying native Rust client, cached connections, and Python
    /// credentials provider reference. Idempotent.
    pub fn close(&self, _py: Python<'_>) {
        let dropped = {
            let mut guard = self
                .client
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.take()
        };
        drop(dropped);
    }

    /// Whether this client has been closed via [`close()`](Client::close).
    #[getter]
    pub fn closed(&self) -> bool {
        self.client
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_none()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        if slf.closed() {
            return Err(PyRuntimeError::new_err(
                "Cannot use a closed arrow_bigquery.Client instance.",
            ));
        }
        Ok(slf)
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) {
        self.close(py);
    }

    /// Reads a BigQuery table and returns an [`ArrowStreamExporter`] which can be
    /// consumed by Polars in Python.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        table,
        *,
        arrow_buffer_compression="lz4frame",
        maintain_order=false,
        max_stream_count=None,
        row_restriction="",
        sample_percentage=None,
        selected_fields=None,
        snapshot_time=None
    ))]
    pub fn read_table(
        &self,
        py: Python<'_>,
        table: &BigQueryTableId,
        arrow_buffer_compression: &str,
        maintain_order: bool,
        max_stream_count: Option<i32>,
        row_restriction: &str,
        sample_percentage: Option<f64>,
        selected_fields: Option<Vec<String>>,
        snapshot_time: Option<Py<PyDateTime>>,
    ) -> PyResult<ArrowStreamExporter> {
        let client = self.clone_inner()?;
        let read_options = parse_read_options(
            py,
            arrow_buffer_compression,
            maintain_order,
            max_stream_count,
            row_restriction,
            sample_percentage,
            selected_fields,
            snapshot_time,
        )?;
        let table = table.inner.clone();
        let result =
            block_on_interruptible(
                py,
                async move { client.read_table(&table, read_options).await },
            )?;

        let (schema, receiver) = result.map_err(|err| error::to_py_err(py, err))?;
        Ok(ArrowStreamExporter::new_validated(schema, receiver))
    }

    /// Executes a BigQuery SQL query (or dry run) and returns an [`ArrowStreamExporter`]
    /// which can be consumed by Polars in Python.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        query,
        *,
        arrow_buffer_compression="lz4frame",
        dry_run=false,
        maintain_order=true,
        use_query_cache=None,
        maximum_bytes_billed=None
    ))]
    pub fn read_query(
        &self,
        py: Python<'_>,
        query: &str,
        arrow_buffer_compression: &str,
        dry_run: bool,
        maintain_order: bool,
        use_query_cache: Option<bool>,
        maximum_bytes_billed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<ArrowStreamExporter> {
        let client = self.clone_inner()?;
        let arrow_buffer_compression = parse_compression_codec(arrow_buffer_compression)?;
        let maximum_bytes_billed = extract_maximum_bytes_billed(py, maximum_bytes_billed)?;
        let query = query.to_owned();
        if dry_run {
            let result = block_on_interruptible(py, async move {
                client
                    .query(query)
                    .with_arrow_buffer_compression(arrow_buffer_compression)
                    .with_maintain_order(maintain_order)
                    .with_use_query_cache(use_query_cache)
                    .with_maximum_bytes_billed(maximum_bytes_billed)
                    .dry_run()
                    .await
            })?;
            let batch = result.map_err(|err| error::to_py_err(py, err))?;
            Ok(ArrowStreamExporter::from_record_batch(batch))
        } else {
            let result = block_on_interruptible(py, async move {
                client
                    .query(query)
                    .with_arrow_buffer_compression(arrow_buffer_compression)
                    .with_maintain_order(maintain_order)
                    .with_use_query_cache(use_query_cache)
                    .with_maximum_bytes_billed(maximum_bytes_billed)
                    .read()
                    .await
            })?;
            let (schema, receiver) = result.map_err(|err| error::to_py_err(py, err))?;
            Ok(ArrowStreamExporter::new_validated(schema, receiver))
        }
    }
}

fn parse_compression_codec(arrow_buffer_compression: &str) -> PyResult<CompressionCodec> {
    match arrow_buffer_compression {
        "unspecified" => Ok(CompressionCodec::CompressionUnspecified),
        "lz4frame" => Ok(CompressionCodec::Lz4Frame),
        "zstd" => Ok(CompressionCodec::Zstd),
        _ => Err(PyValueError::new_err(format!(
            "got unexpected compression codec {arrow_buffer_compression}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_read_options(
    py: Python<'_>,
    arrow_buffer_compression: &str,
    maintain_order: bool,
    max_stream_count: Option<i32>,
    row_restriction: &str,
    sample_percentage: Option<f64>,
    selected_fields: Option<Vec<String>>,
    snapshot_time: Option<Py<PyDateTime>>,
) -> PyResult<arrow_bigquery_lib::ReadOptions> {
    let snapshot_time: Option<chrono::DateTime<chrono::Utc>> = match snapshot_time {
        Some(dt) => Some(
            dt.extract::<chrono::DateTime<chrono::Utc>>(py)
                .map_err(|_| PyValueError::new_err("failed to extract snapshot_time"))?,
        ),
        None => None,
    };
    let arrow_buffer_compression = Some(parse_compression_codec(arrow_buffer_compression)?);
    Ok(arrow_bigquery_lib::ReadOptions {
        arrow_buffer_compression,
        maintain_order,
        max_stream_count,
        row_restriction: row_restriction.to_owned(),
        sample_percentage,
        selected_fields: selected_fields.unwrap_or_default(),
        snapshot_time,
    })
}

#[pymodule]
#[pyo3(name = "_native")]
fn polars_bigquery(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("BigQueryError", m.py().get_type::<error::BigQueryError>())?;
    m.add(
        "UnsupportedTypeError",
        m.py().get_type::<error::UnsupportedTypeError>(),
    )?;
    m.add_class::<Client>()?;
    m.add_class::<BigQueryTableId>()?;
    #[cfg(feature = "testing")]
    {
        m.add_wrapped(wrap_pyfunction!(testing::_create_test_exporter))?;
        m.add_wrapped(wrap_pyfunction!(
            testing::_test_create_exporter_with_drop_flag
        ))?;
        m.add_class::<testing::DropFlag>()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;

    #[test]
    fn test_parse_read_options_defaults() {
        Python::initialize();
        Python::attach(|py| {
            let options =
                parse_read_options(py, "lz4frame", false, None, "", None, None, None).unwrap();
            assert!(!options.maintain_order);
            assert_eq!(options.snapshot_time, None);
            assert!(options.selected_fields.is_empty());
            assert_eq!(options.row_restriction, "");
            assert_eq!(
                options.arrow_buffer_compression,
                Some(CompressionCodec::Lz4Frame)
            );
            assert_eq!(options.sample_percentage, None);
        });
    }

    #[test]
    fn test_parse_read_options_all_fields() {
        Python::initialize();
        Python::attach(|py| {
            let datetime_mod = py.import("datetime").unwrap();
            let timezone = datetime_mod
                .getattr("timezone")
                .unwrap()
                .getattr("utc")
                .unwrap();
            let dt = datetime_mod
                .getattr("datetime")
                .unwrap()
                .call1((2023, 11, 14, 22, 13, 20, 500_000, timezone))
                .unwrap();
            let dt_py: Py<PyDateTime> = dt.extract().unwrap();
            let expected_dt = chrono::DateTime::from_timestamp(1_700_000_000, 500_000_000).unwrap();

            let options = parse_read_options(
                py,
                "zstd",
                true,
                Some(16),
                "col1 > 100",
                Some(42.5),
                Some(vec!["col1".to_string(), "col2".to_string()]),
                Some(dt_py),
            )
            .unwrap();

            assert!(options.maintain_order);
            assert_eq!(options.max_stream_count, Some(16));
            assert_eq!(options.snapshot_time, Some(expected_dt));
            assert_eq!(options.selected_fields, vec!["col1", "col2"]);
            assert_eq!(options.row_restriction, "col1 > 100");
            assert_eq!(
                options.arrow_buffer_compression,
                Some(CompressionCodec::Zstd)
            );
            assert_eq!(options.sample_percentage, Some(42.5));
        });
    }

    #[test]
    fn test_parse_read_options_compression_codecs() {
        Python::initialize();

        let valid_codecs = [
            ("unspecified", CompressionCodec::CompressionUnspecified),
            ("lz4frame", CompressionCodec::Lz4Frame),
            ("zstd", CompressionCodec::Zstd),
        ];

        Python::attach(|py| {
            for (codec_str, expected) in valid_codecs {
                let options =
                    parse_read_options(py, codec_str, false, None, "", None, None, None).unwrap();
                assert_eq!(options.arrow_buffer_compression, Some(expected));
            }

            let err =
                match parse_read_options(py, "invalid_codec", false, None, "", None, None, None) {
                    Err(e) => e,
                    Ok(_) => panic!("expected Err for invalid compression codec"),
                };
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("got unexpected compression codec invalid_codec"));
        });
    }

    #[test]
    fn test_parse_read_options_snapshot_time_naive_error() {
        Python::initialize();
        Python::attach(|py| {
            let datetime_mod = py.import("datetime").unwrap();
            let dt = datetime_mod
                .getattr("datetime")
                .unwrap()
                .call1((2023, 11, 14, 22, 13, 20))
                .unwrap();
            let dt_py: Py<PyDateTime> = dt.extract().unwrap();

            let err = match parse_read_options(
                py,
                "lz4frame",
                false,
                None,
                "",
                None,
                None,
                Some(dt_py),
            ) {
                Err(e) => e,
                Ok(_) => panic!("expected Err for naive datetime"),
            };
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("failed to extract snapshot_time"));
        });
    }

    #[test]
    fn test_client_read_table_invalid_compression() {
        Python::initialize();
        Python::attach(|py| {
            let client =
                Client::new(py, "test-project".to_string(), None, None, None, None).unwrap();
            let table =
                BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string()).unwrap();
            let err = match client.read_table(
                py,
                &table,
                "invalid_codec",
                false,
                None,
                "",
                None,
                None,
                None,
            ) {
                Err(e) => e,
                Ok(_) => panic!("expected Err for invalid compression codec"),
            };
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("got unexpected compression codec invalid_codec"));
        });
    }

    #[test]
    fn test_client_read_table_naive_snapshot_time() {
        Python::initialize();
        Python::attach(|py| {
            let datetime_mod = py.import("datetime").unwrap();
            let dt = datetime_mod
                .getattr("datetime")
                .unwrap()
                .call1((2023, 11, 14, 22, 13, 20))
                .unwrap();
            let dt_py: Py<PyDateTime> = dt.extract().unwrap();

            let client =
                Client::new(py, "test-project".to_string(), None, None, None, None).unwrap();
            let table =
                BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string()).unwrap();
            let err = match client.read_table(
                py,
                &table,
                "lz4frame",
                false,
                None,
                "",
                None,
                None,
                Some(dt_py),
            ) {
                Err(e) => e,
                Ok(_) => panic!("expected Err for naive datetime"),
            };
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("failed to extract snapshot_time"));
        });
    }

    #[test]
    fn test_client_new_with_credentials_provider() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"def p(): return ({'bearer_token': 'tok'}, None)",
                None,
                Some(&locals),
            )
            .unwrap();
            let provider = locals.get_item("p").unwrap().unwrap().unbind();

            let client = Client::new(
                py,
                "test-project".to_string(),
                Some(provider),
                Some("test-agent/1.0".to_string()),
                Some("https://custom.bq.endpoint.com".to_string()),
                Some("https://custom.bqstorage.endpoint.com".to_string()),
            );
            assert!(client.is_ok());
        });
    }

    #[test]
    fn test_client_read_table_preserves_python_cause_chain() {
        use google_cloud_auth::errors::CredentialsError;
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{CreateReadSessionRequest, ReadSession};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct FailingStub;

        impl google_cloud_bigquery::stub::Read for FailingStub {
            async fn create_read_session(
                &self,
                _req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let py_err = Python::attach(|_py| {
                    PyValueError::new_err("underlying python token generator broke")
                });
                let cred_err =
                    CredentialsError::new(false, "Python credentials provider failed", py_err);
                Err(GaxError::authentication(cred_err))
            }
        }

        Python::initialize();
        Python::attach(|py| {
            let rt = pyo3_async_runtimes::tokio::get_runtime();
            let bq_client = rt.block_on(async {
                arrow_bigquery_lib::BigQueryJobService::builder()
                    .with_credentials(
                        google_cloud_auth::credentials::anonymous::Builder::new().build(),
                    )
                    .build()
                    .await
                    .unwrap()
            });
            let inner_client = arrow_bigquery_lib::Client::new_for_testing(
                bq_client,
                Read::from_stub(FailingStub),
                "test-project".to_string(),
            );
            let client = Client {
                client: RwLock::new(Some(inner_client)),
            };
            let table =
                BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string()).unwrap();

            let err = match client
                .read_table(py, &table, "lz4frame", false, None, "", None, None, None)
            {
                Err(e) => e,
                Ok(_) => panic!("expected read_table to fail"),
            };

            assert!(err.is_instance_of::<error::BigQueryError>(py));
            let msg = err.to_string();
            assert!(msg.contains("Python credentials provider failed"));
            assert!(msg.contains("caused by: ValueError: underlying python token generator broke"));

            // Verify Python __cause__ is directly the original ValueError
            let cause = err.cause(py).expect("expected direct Python __cause__");
            assert!(cause.is_instance_of::<PyValueError>(py));
            assert!(cause
                .to_string()
                .contains("underlying python token generator broke"));
            assert!(cause.cause(py).is_none());
        });
    }

    #[test]
    fn test_client_read_query_invalid_compression() {
        Python::initialize();
        Python::attach(|py| {
            let client =
                Client::new(py, "test-project".to_string(), None, None, None, None).unwrap();
            let err = match client.read_query(
                py,
                "SELECT 1",
                "invalid_codec",
                false,
                false,
                None,
                None,
            ) {
                Err(e) => e,
                Ok(_) => panic!("expected Err for invalid compression codec"),
            };
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("got unexpected compression codec invalid_codec"));
        });
    }

    #[test]
    fn test_client_read_query_maximum_bytes_billed_validation() {
        Python::initialize();
        Python::attach(|py| {
            let client =
                Client::new(py, "test-project".to_string(), None, None, None, None).unwrap();

            // Negative value raises ValueError.
            let neg = (-5_i64).into_pyobject(py).unwrap().into_any();
            let err = client
                .read_query(py, "SELECT 1", "lz4frame", false, true, None, Some(&neg))
                .err()
                .expect("expected Err for negative maximum_bytes_billed");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("maximum_bytes_billed must be non-negative"));

            // Boolean value raises TypeError (must not coerce True -> 1).
            let py_true = pyo3::types::PyBool::new(py, true).to_owned().into_any();
            let err = client
                .read_query(
                    py,
                    "SELECT 1",
                    "lz4frame",
                    false,
                    true,
                    None,
                    Some(&py_true),
                )
                .err()
                .expect("expected Err for bool maximum_bytes_billed");
            assert!(err.is_instance_of::<PyTypeError>(py));

            // Value above i64::MAX raises ValueError (not OverflowError).
            let huge = (i64::MAX as i128 + 1).into_pyobject(py).unwrap().into_any();
            let err = client
                .read_query(py, "SELECT 1", "lz4frame", false, true, None, Some(&huge))
                .err()
                .expect("expected Err for overflowing maximum_bytes_billed");
            assert!(err.is_instance_of::<PyValueError>(py));

            // Negative value below i64::MIN also raises ValueError without hitting
            // CPython's integer-to-string digit limit.
            let builtins = py.import("builtins").unwrap();
            let pow = builtins.getattr("pow").unwrap();
            let giant_pos = pow.call1((10_i64, 5000_i64)).unwrap();
            let err = client
                .read_query(
                    py,
                    "SELECT 1",
                    "lz4frame",
                    false,
                    true,
                    None,
                    Some(&giant_pos),
                )
                .err()
                .expect("expected Err for giant positive maximum_bytes_billed");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("maximum_bytes_billed must be <="));

            let giant_neg = giant_pos.neg().unwrap();
            let err = client
                .read_query(
                    py,
                    "SELECT 1",
                    "lz4frame",
                    false,
                    true,
                    None,
                    Some(&giant_neg),
                )
                .err()
                .expect("expected Err for giant negative maximum_bytes_billed");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("maximum_bytes_billed must be non-negative"));

            // close() rejects subsequent reads with RuntimeError and is idempotent.
            assert!(!client.closed());
            client.close(py);
            assert!(client.closed());
            client.close(py);
            let err = client
                .read_query(py, "SELECT 1", "lz4frame", false, true, None, None)
                .err()
                .expect("expected RuntimeError on closed client");
            assert!(err.is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn test_is_main_python_thread_does_not_create_dummy_thread_and_caches_worker() {
        Python::initialize();
        let handle = std::thread::spawn(|| {
            let (is_main, cached, in_active) = Python::attach(|py| {
                let threading = py.import("threading").unwrap();
                let ident = threading.call_method0("get_ident").unwrap();
                let is_main = is_main_python_thread(py).unwrap();
                let active = threading.getattr("_active").unwrap();
                let in_active: bool = active.contains(&ident).unwrap();
                (is_main, cached_is_main_python_thread(), in_active)
            });
            assert!(
                !is_main,
                "spawned std::thread must not be reported as main thread"
            );
            assert_eq!(cached, Some(false));
            assert!(
                !in_active,
                "is_main_python_thread must not register a _DummyThread in threading._active"
            );
        });
        handle.join().unwrap();
    }
}
