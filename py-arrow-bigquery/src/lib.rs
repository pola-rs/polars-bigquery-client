use std::sync::{Mutex, Once};

use chrono::Utc;
use google_cloud_auth::credentials::{
    CacheableResource, Credentials, CredentialsProvider, EntityTag,
};
use google_cloud_auth::errors::CredentialsError;
use google_cloud_bigquery::model::arrow_serialization_options::CompressionCodec;
use polars_arrow::datatypes::ArrowSchemaRef;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::*;

static INIT_CRYPTO: Once = Once::new();

struct CachedToken {
    headers: http::HeaderMap,
    expiry: Option<chrono::DateTime<Utc>>,
    entity_tag: EntityTag,
}

/// A token source that delegates authentication to a Python callable.
///
/// This struct implements the [`CredentialsProvider`] trait, allowing the Rust
/// Google Cloud SDK to retrieve OAuth2 tokens by calling back into Python code
/// (e.g., using `google-auth`). It includes a thread-safe cache to avoid
/// the overhead of calling into Python on every request if the token is still valid.
struct PythonTokenSource {
    /// The Python callable (e.g., a function or method) that returns a tuple of
    /// `(token_data, expiration_timestamp_float_or_none)`.
    provider: Py<PyAny>,
    /// A thread-safe cache for the retrieved token headers.
    cache: Mutex<Option<CachedToken>>,
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
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.as_ref() {
                let is_valid = match cached.expiry {
                    Some(expiry) => expiry > Utc::now() + chrono::Duration::seconds(60),
                    None => true,
                };
                if is_valid {
                    return match extensions.get::<EntityTag>() {
                        Some(tag) if cached.entity_tag == *tag => {
                            Ok(CacheableResource::NotModified)
                        },
                        _ => Ok(CacheableResource::New {
                            entity_tag: cached.entity_tag.clone(),
                            data: cached.headers.clone(),
                        }),
                    };
                }
            }
        }

        let cached = Python::attach(|py| -> Result<CachedToken, CredentialsError> {
            let provider = self.provider.bind(py);
            let result = provider.call0().map_err(|err| {
                CredentialsError::from_msg(
                    false,
                    format!("Python credentials provider failed: {err}"),
                )
            })?;

            // result is (token_data, expiration)
            let tuple = result.cast::<pyo3::types::PyTuple>().map_err(|_| {
                CredentialsError::from_msg(
                    false,
                    "Python credentials provider must return a 2-tuple (token_data, expiration)",
                )
            })?;

            let token_data = tuple.get_item(0).map_err(|_| {
                CredentialsError::from_msg(
                    false,
                    "Python credentials provider tuple missing token_data at index 0",
                )
            })?;

            let expiration = tuple.get_item(1).map_err(|_| {
                CredentialsError::from_msg(
                    false,
                    "Python credentials provider tuple missing expiration at index 1",
                )
            })?;

            let bearer_token: String = token_data
                .get_item("bearer_token")
                .map_err(|_| {
                    CredentialsError::from_msg(false, "token_data missing 'bearer_token' key")
                })?
                .cast::<pyo3::types::PyString>()
                .map_err(|_| CredentialsError::from_msg(false, "'bearer_token' must be a string"))?
                .to_str()
                .map_err(|_| {
                    CredentialsError::from_msg(false, "'bearer_token' is not valid UTF-8")
                })?
                .to_string();

            // expiration is a float/int (timestamp) or None
            let expiry = if expiration.is_none() {
                None
            } else {
                let expiry_f: f64 = expiration.extract().map_err(|_| {
                    CredentialsError::from_msg(
                        false,
                        "expiration must be a float/int timestamp or None",
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
                Some(dt)
            };

            let mut header_value =
                http::header::HeaderValue::from_str(&format!("Bearer {bearer_token}"))
                    .map_err(|err| CredentialsError::from_source(false, err))?;
            header_value.set_sensitive(true);

            let mut headers = http::HeaderMap::new();
            headers.insert(http::header::AUTHORIZATION, header_value);

            Ok(CachedToken {
                headers,
                expiry,
                entity_tag: EntityTag::new(),
            })
        })?;

        let response = CacheableResource::New {
            entity_tag: cached.entity_tag.clone(),
            data: cached.headers.clone(),
        };

        {
            let mut cache = self.cache.lock().unwrap();
            *cache = Some(cached);
        }

        Ok(response)
    }

    async fn universe_domain(&self) -> Option<String> {
        None
    }
}

/// A Python-exposed class that implements the Arrow C Stream interface.
///
/// This class acts as a bridge between the Rust BigQuery reader and Python Polars,
/// allowing Polars to consume the data stream directly via the Arrow C Data Interface
/// (`__arrow_c_stream__`) without copying data.
#[pyclass]
pub struct ArrowStreamExporter {
    /// The schema of the Arrow stream.
    schema: ArrowSchemaRef,
    /// The underlying BigQuery record batch receiver, wrapped in a mutex.
    /// It is an `Option` because the stream can only be consumed once.
    receiver: std::sync::Mutex<Option<arrow_bigquery_lib::BigQueryRecordBatchReceiver>>,
}

/// An iterator that adapts the asynchronous [`BigQueryRecordBatchReceiver`] into
/// a synchronous iterator yielding Arrow arrays.
///
/// This is used internally by [`ArrowStreamExporter`] to feed the Arrow C Stream.
/// Each iteration blocks on the Tokio runtime to receive the next batch.
struct ReceiverIterator {
    /// The receiver yielding record batches from the BigQuery Storage Read API.
    rx: arrow_bigquery_lib::BigQueryRecordBatchReceiver,
    /// The Arrow datatype (specifically a `Struct` type) matching the schema of the batches.
    dtype: polars_arrow::datatypes::ArrowDataType,
}

impl Iterator for ReceiverIterator {
    type Item =
        pyo3_polars::export::polars_error::PolarsResult<Box<dyn polars_arrow::array::Array>>;

    fn next(&mut self) -> Option<Self::Item> {
        let rt = pyo3_async_runtimes::tokio::get_runtime();

        loop {
            // We need to be able to stop if the Python side decides to, so
            // occasionally check to see if there were any interrupts.
            if let Err(py_err) = Python::attach(|py| py.check_signals()) {
                Python::attach(|py| py_err.restore(py));
                return Some(Err(
                    pyo3_polars::export::polars_error::PolarsError::ComputeError(
                        "Python interrupt".into(),
                    ),
                ));
            }

            let timeout_duration = std::time::Duration::from_millis(100);
            let result = Python::attach(|py| {
                py.detach(|| {
                    rt.block_on(async {
                        tokio::time::timeout(timeout_duration, self.rx.recv()).await
                    })
                })
            });

            match result {
                Ok(Some(Ok(batch))) => {
                    let len = batch.len();
                    let (_, arrays) = batch.into_schema_and_arrays();
                    let struct_array = polars_arrow::array::StructArray::new(
                        self.dtype.clone(),
                        len,
                        arrays,
                        None,
                    );
                    return Some(Ok(
                        Box::new(struct_array) as Box<dyn polars_arrow::array::Array>
                    ));
                },
                Ok(Some(Err(err))) => {
                    // Stream failed: bubble up exception immediately to prevent silent truncation.
                    return Some(Err(
                        pyo3_polars::export::polars_error::PolarsError::ComputeError(
                            format!("BigQuery Storage API read error: {}", err).into(),
                        ),
                    ));
                },
                Ok(None) => {
                    // Stream finished
                    return None;
                },
                Err(_) => {
                    // Timeout elapsed, loop again to check signals
                    continue;
                },
            }
        }
    }
}

#[pymethods]
impl ArrowStreamExporter {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = requested_schema;
        let mut rx_guard = self.receiver.lock().unwrap();
        let rx = rx_guard
            .take()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Stream already consumed"))?;

        let fields: Vec<polars_arrow::datatypes::Field> =
            self.schema.iter().map(|(_, field)| field.clone()).collect();
        let dtype = polars_arrow::datatypes::ArrowDataType::Struct(fields);

        let iter = ReceiverIterator {
            rx,
            dtype: dtype.clone(),
        };
        let box_iter = Box::new(iter)
            as Box<
                dyn Iterator<
                    Item = pyo3_polars::export::polars_error::PolarsResult<
                        Box<dyn polars_arrow::array::Array>,
                    >,
                >,
            >;

        let field = polars_arrow::datatypes::Field::new("".into(), dtype, false);

        let stream = polars_arrow::ffi::export_iterator(box_iter, field);

        let capsule = pyo3::types::PyCapsule::new(py, stream, Some(c"arrow_array_stream".into()))?;
        Ok(capsule.into())
    }
}

/// A Python-exposed representation of a BigQuery table identifier.
#[pyclass(name = "BigQueryTableId")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BigQueryTableId {
    pub inner: arrow_bigquery_lib::BigQueryTableId,
}

#[pymethods]
impl BigQueryTableId {
    #[new]
    #[pyo3(signature = (project_id, dataset_id, table_id))]
    pub fn new(project_id: String, dataset_id: String, table_id: String) -> Self {
        Self {
            inner: arrow_bigquery_lib::BigQueryTableId {
                project_id,
                dataset_id,
                table_id,
            },
        }
    }

    #[staticmethod]
    pub fn from_string(s: &str) -> PyResult<Self> {
        let inner: arrow_bigquery_lib::BigQueryTableId = s.parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid table ID: {s}"))
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> PyResult<Self> {
        Self::from_string(s)
    }

    /// Parse a table ID from a string, a BigQueryTableId instance, or an object
    /// with table reference attributes (`project`/`project_id`, `dataset_id`, `table_id`).
    #[staticmethod]
    pub fn parse(table_id: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(existing) = table_id.cast::<BigQueryTableId>() {
            return Ok(existing.borrow().clone());
        }

        if let Ok(s) = table_id.extract::<&str>() {
            return Self::from_string(s);
        }

        let project = table_id
            .getattr("project")
            .ok()
            .and_then(|p| p.extract::<String>().ok())
            .or_else(|| {
                table_id
                    .getattr("project_id")
                    .ok()
                    .and_then(|p| p.extract::<String>().ok())
            });

        let dataset = table_id
            .getattr("dataset_id")
            .ok()
            .and_then(|d| d.extract::<String>().ok());

        let table = table_id
            .getattr("table_id")
            .ok()
            .and_then(|t| t.extract::<String>().ok());

        if let (Some(project), Some(dataset), Some(table)) = (project, dataset, table) {
            return Ok(Self::new(project, dataset, table));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Expected table_id to be a string or BigQuery table object, got {}",
            table_id.get_type()
        )))
    }

    #[getter]
    pub fn project_id(&self) -> &str {
        &self.inner.project_id
    }

    #[setter]
    pub fn set_project_id(&mut self, value: String) {
        self.inner.project_id = value;
    }

    #[getter]
    pub fn dataset_id(&self) -> &str {
        &self.inner.dataset_id
    }

    #[setter]
    pub fn set_dataset_id(&mut self, value: String) {
        self.inner.dataset_id = value;
    }

    #[getter]
    pub fn table_id(&self) -> &str {
        &self.inner.table_id
    }

    #[setter]
    pub fn set_table_id(&mut self, value: String) {
        self.inner.table_id = value;
    }

    #[getter]
    pub fn project(&self) -> &str {
        &self.inner.project_id
    }

    fn __repr__(&self) -> String {
        format!(
            "BigQueryTableId(project_id='{}', dataset_id='{}', table_id='{}')",
            self.inner.project_id, self.inner.dataset_id, self.inner.table_id
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{}.{}.{}",
            self.inner.project_id, self.inner.dataset_id, self.inner.table_id
        )
    }

    fn __richcmp__(&self, other: &Self, op: pyo3::pyclass::CompareOp) -> bool {
        match op {
            pyo3::pyclass::CompareOp::Eq => self.inner == other.inner,
            pyo3::pyclass::CompareOp::Ne => self.inner != other.inner,
            _ => false,
        }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

impl From<BigQueryTableId> for arrow_bigquery_lib::BigQueryTableId {
    fn from(id: BigQueryTableId) -> Self {
        id.inner
    }
}

impl From<&BigQueryTableId> for arrow_bigquery_lib::BigQueryTableId {
    fn from(id: &BigQueryTableId) -> Self {
        id.inner.clone()
    }
}

impl From<arrow_bigquery_lib::BigQueryTableId> for BigQueryTableId {
    fn from(inner: arrow_bigquery_lib::BigQueryTableId) -> Self {
        Self { inner }
    }
}

impl std::str::FromStr for BigQueryTableId {
    type Err = arrow_bigquery_lib::BigQueryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let inner = s.parse()?;
        Ok(Self { inner })
    }
}

/// A Python-exposed client that keeps the BigQuery Storage Read API gRPC channel
/// open across multiple table read operations, caching the OAuth2 token in Rust.
#[pyclass(name = "Client")]
pub struct Client {
    client: arrow_bigquery_lib::Client,
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (*, quota_project_id, credentials_provider=None, user_agent=None))]
    pub fn new(
        quota_project_id: String,
        credentials_provider: Option<Py<PyAny>>,
        user_agent: Option<String>,
    ) -> PyResult<Self> {
        INIT_CRYPTO.call_once(|| {
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
            // ignore if another crate already set the default provider.
        });

        let cred = match credentials_provider {
            Some(provider) => {
                let is_none = Python::attach(|py| provider.is_none(py));
                if is_none {
                    None
                } else {
                    let token_source = PythonTokenSource {
                        provider,
                        cache: Mutex::new(None),
                    };
                    Some(Credentials::from(token_source))
                }
            },
            None => None,
        };

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let client = Python::attach(|py| {
            py.detach(|| {
                rt.block_on(async {
                    use arrow_bigquery_lib::BigQueryReadClientBuilder;

                    let mut builder = arrow_bigquery_lib::ServiceConfigBuilder::new()
                        .with_user_agent(user_agent)
                        .with_quota_project_id(Some(quota_project_id));
                    if let Some(cred) = cred {
                        builder = builder.with_cred(cred);
                    }

                    arrow_bigquery_lib::Client::from_builder(builder)
                        .await
                        .map_err(|err| err.to_string())
                })
            })
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(Self { client })
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
        table: &BigQueryTableId,
        arrow_buffer_compression: &str,
        maintain_order: bool,
        max_stream_count: Option<i32>,
        row_restriction: &str,
        sample_percentage: Option<f64>,
        selected_fields: Option<Vec<String>>,
        snapshot_time: Option<Py<PyDateTime>>,
    ) -> PyResult<ArrowStreamExporter> {
        let read_options = parse_read_options(
            arrow_buffer_compression,
            maintain_order,
            max_stream_count,
            row_restriction,
            sample_percentage,
            selected_fields,
            snapshot_time,
        )?;
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let client = self.client.clone();
        let table = table.inner.clone();
        let result = Python::attach(|py| {
            py.detach(|| {
                rt.block_on(async move {
                    client
                        .read_table(&table, read_options)
                        .await
                        .map_err(|err| err.to_string())
                })
            })
        });

        match result {
            Ok((schema, receiver)) => Ok(ArrowStreamExporter {
                schema,
                receiver: std::sync::Mutex::new(Some(receiver)),
            }),
            Err(err) => Err(pyo3::exceptions::PyRuntimeError::new_err(err)),
        }
    }
}

fn parse_read_options(
    arrow_buffer_compression: &str,
    maintain_order: bool,
    max_stream_count: Option<i32>,
    row_restriction: &str,
    sample_percentage: Option<f64>,
    selected_fields: Option<Vec<String>>,
    snapshot_time: Option<Py<PyDateTime>>,
) -> PyResult<arrow_bigquery_lib::ReadOptions> {
    let snapshot_time: Option<chrono::DateTime<chrono::Utc>> = match snapshot_time {
        Some(dt) => Some(Python::attach(|py| {
            dt.extract::<chrono::DateTime<chrono::Utc>>(py)
                .map_err(|_| PyValueError::new_err("failed to extract snapshot_time"))
        })?),
        None => None,
    };
    let arrow_buffer_compression = Some(match arrow_buffer_compression {
        "unspecified" => CompressionCodec::CompressionUnspecified,
        "lz4frame" => CompressionCodec::Lz4Frame,
        "zstd" => CompressionCodec::Zstd,
        _ => Err(PyValueError::new_err(format!(
            "got unexpected compression codec {arrow_buffer_compression}"
        )))?,
    });
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

#[pyfunction]
pub fn _create_test_exporter() -> ArrowStreamExporter {
    // Force initialization of the Tokio runtime inside a #[pyfunction] context
    let rt = pyo3_async_runtimes::tokio::get_runtime();
    rt.block_on(async {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    });

    let (tx, rx) = tokio::sync::mpsc::channel(10);
    // Leak the sender so the channel never closes, causing rx.recv() to block indefinitely.
    Box::leak(Box::new(tx));

    let field = polars_arrow::datatypes::Field::new(
        "placeholder".into(),
        polars_arrow::datatypes::ArrowDataType::Int32,
        true,
    );
    let schema = polars_arrow::datatypes::ArrowSchema::from_iter(vec![field]);
    let schema_ref = std::sync::Arc::new(schema);
    let receiver = arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, Vec::new());

    ArrowStreamExporter {
        schema: schema_ref,
        receiver: std::sync::Mutex::new(Some(receiver)),
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DropFlag {
    value: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[pymethods]
impl DropFlag {
    fn is_set(&self) -> bool {
        self.value.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[pyfunction]
pub fn _test_create_exporter_with_drop_flag() -> (ArrowStreamExporter, DropFlag) {
    let rt = pyo3_async_runtimes::tokio::get_runtime();

    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag_clone = flag.clone();

    // Spawn a placeholder task that runs forever until aborted, and sets the flag when dropped
    let handle = rt.spawn(async move {
        struct SetOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _cleanup = SetOnDrop(flag_clone);

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    });

    let (tx, rx) = tokio::sync::mpsc::channel(10);
    Box::leak(Box::new(tx));

    let field = polars_arrow::datatypes::Field::new(
        "placeholder".into(),
        polars_arrow::datatypes::ArrowDataType::Int32,
        true,
    );
    let schema = polars_arrow::datatypes::ArrowSchema::from_iter(vec![field]);
    let schema_ref = std::sync::Arc::new(schema);

    let receiver =
        arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);

    let exporter = ArrowStreamExporter {
        schema: schema_ref,
        receiver: std::sync::Mutex::new(Some(receiver)),
    };

    let drop_flag = DropFlag { value: flag };

    (exporter, drop_flag)
}

#[pymodule]
#[pyo3(name = "_native")]
fn polars_bigquery(m: &Bound<PyModule>) -> PyResult<()> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    m.add_class::<Client>().unwrap();
    m.add_class::<BigQueryTableId>().unwrap();
    m.add_wrapped(wrap_pyfunction!(_create_test_exporter))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(_test_create_exporter_with_drop_flag))
        .unwrap();
    m.add_class::<DropFlag>().unwrap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_read_options_defaults() {
        Python::initialize();
        let options = parse_read_options("lz4frame", false, None, "", None, None, None).unwrap();
        assert!(!options.maintain_order);
        assert_eq!(options.snapshot_time, None);
        assert!(options.selected_fields.is_empty());
        assert_eq!(options.row_restriction, "");
        assert_eq!(
            options.arrow_buffer_compression,
            Some(CompressionCodec::Lz4Frame)
        );
        assert_eq!(options.sample_percentage, None);
    }

    #[test]
    fn test_parse_read_options_all_fields() {
        Python::initialize();
        let (dt_py, expected_dt) = Python::attach(|py| {
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
            let py_dt: Py<PyDateTime> = dt.extract().unwrap();
            let expected = chrono::DateTime::from_timestamp(1_700_000_000, 500_000_000).unwrap();
            (py_dt, expected)
        });

        let options = parse_read_options(
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
    }

    #[test]
    fn test_parse_read_options_compression_codecs() {
        Python::initialize();

        let valid_codecs = [
            ("unspecified", CompressionCodec::CompressionUnspecified),
            ("lz4frame", CompressionCodec::Lz4Frame),
            ("zstd", CompressionCodec::Zstd),
        ];

        for (codec_str, expected) in valid_codecs {
            let options = parse_read_options(codec_str, false, None, "", None, None, None).unwrap();
            assert_eq!(options.arrow_buffer_compression, Some(expected));
        }

        let err = match parse_read_options("invalid_codec", false, None, "", None, None, None) {
            Err(e) => e,
            Ok(_) => panic!("expected Err for invalid compression codec"),
        };
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("got unexpected compression codec invalid_codec"));
        });
    }

    #[test]
    fn test_parse_read_options_snapshot_time_naive_error() {
        Python::initialize();
        let dt_py = Python::attach(|py| {
            let datetime_mod = py.import("datetime").unwrap();
            let dt = datetime_mod
                .getattr("datetime")
                .unwrap()
                .call1((2023, 11, 14, 22, 13, 20))
                .unwrap();
            let py_dt: Py<PyDateTime> = dt.extract().unwrap();
            py_dt
        });

        let err = match parse_read_options("lz4frame", false, None, "", None, None, Some(dt_py)) {
            Err(e) => e,
            Ok(_) => panic!("expected Err for naive datetime"),
        };
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("failed to extract snapshot_time"));
        });
    }

    #[test]
    fn test_client_read_table_invalid_compression() {
        Python::initialize();
        let client = Client::new("test-project".to_string(), None, None).unwrap();
        let table = BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string());
        let err =
            match client.read_table(&table, "invalid_codec", false, None, "", None, None, None) {
                Err(e) => e,
                Ok(_) => panic!("expected Err for invalid compression codec"),
            };
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("got unexpected compression codec invalid_codec"));
        });
    }

    #[test]
    fn test_client_read_table_naive_snapshot_time() {
        Python::initialize();
        let dt_py = Python::attach(|py| {
            let datetime_mod = py.import("datetime").unwrap();
            let dt = datetime_mod
                .getattr("datetime")
                .unwrap()
                .call1((2023, 11, 14, 22, 13, 20))
                .unwrap();
            let py_dt: Py<PyDateTime> = dt.extract().unwrap();
            py_dt
        });

        let client = Client::new("test-project".to_string(), None, None).unwrap();
        let table = BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string());
        let err =
            match client.read_table(&table, "lz4frame", false, None, "", None, None, Some(dt_py)) {
                Err(e) => e,
                Ok(_) => panic!("expected Err for naive datetime"),
            };
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("failed to extract snapshot_time"));
        });
    }

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

        let source = PythonTokenSource {
            provider,
            cache: Mutex::new(None),
        };

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
            // First token expires in 30s (within the 60s refresh window), second in 3600s
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

        let source = PythonTokenSource {
            provider,
            cache: Mutex::new(None),
        };

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

            // Since token-1 expires in <60s, the next call should refresh and produce token-2
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

        let source = PythonTokenSource {
            provider,
            cache: Mutex::new(None),
        };

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
    fn test_python_token_source_errors() {
        Python::initialize();
        let cases = [
            ("def p(): raise RuntimeError('auth failed')", "auth failed"),
            ("def p(): return 'not-a-tuple'", "2-tuple"),
            ("def p(): return ({}, 1700000000.0)", "bearer_token"),
        ];

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        for (code, expected_msg) in cases {
            let provider = Python::attach(|py| {
                let locals = PyDict::new(py);
                let c_code = std::ffi::CString::new(code).unwrap();
                py.run(&c_code, None, Some(&locals)).unwrap();
                locals.get_item("p").unwrap().unwrap().unbind()
            });
            let source = PythonTokenSource {
                provider,
                cache: Mutex::new(None),
            };
            rt.block_on(async {
                let err = source.headers(http::Extensions::new()).await.unwrap_err();
                assert!(!err.is_transient());
                assert!(
                    err.to_string().contains(expected_msg),
                    "expected '{}' in '{}'",
                    expected_msg,
                    err
                );
            });
        }
    }

    #[test]
    fn test_client_new_with_credentials_provider() {
        Python::initialize();
        let provider = Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"def p(): return ({'bearer_token': 'tok'}, None)",
                None,
                Some(&locals),
            )
            .unwrap();
            locals.get_item("p").unwrap().unwrap().unbind()
        });

        let client = Client::new(
            "test-project".to_string(),
            Some(provider),
            Some("test-agent/1.0".to_string()),
        );
        assert!(client.is_ok());
    }
}
