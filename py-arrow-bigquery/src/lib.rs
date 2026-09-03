use std::sync::{Mutex, Once};

use async_trait::async_trait;
use chrono::Utc;
use gcloud_sdk::google::cloud::bigquery::storage::v1::arrow_serialization_options::CompressionCodec;
use polars_arrow::datatypes::ArrowSchemaRef;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::*;

static INIT_CRYPTO: Once = Once::new();

/// A token source that delegates authentication to a Python callable.
///
/// This struct implements the [`gcloud_sdk::Source`] trait, allowing the Rust
/// Google Cloud SDK to retrieve OAuth2 tokens by calling back into Python code
/// (e.g., using `google-auth`). It includes a thread-safe cache to avoid
/// the overhead of calling into Python on every request if the token is still valid.
struct PythonTokenSource {
    /// The Python callable (e.g., a function or method) that returns a tuple of
    /// `(token_bytes, expiration_timestamp_float)`.
    provider: Py<PyAny>,
    /// A thread-safe cache for the retrieved token.
    cache: Mutex<Option<gcloud_sdk::Token>>,
}

#[async_trait]
impl gcloud_sdk::Source for PythonTokenSource {
    async fn token(&self) -> Result<gcloud_sdk::Token, gcloud_sdk::error::Error> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some(token) = cache.as_ref() {
                if token.expiry > Utc::now() + chrono::Duration::seconds(60) {
                    return Ok(token.clone());
                }
            }
        }

        let token = Python::attach(
            |py| -> Result<gcloud_sdk::Token, gcloud_sdk::error::Error> {
                let provider = self.provider.bind(py);
                let result = provider.call0().map_err(|_| {
                    gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                })?;

                // result is (token_data, expiration)
                let tuple = result.cast::<pyo3::types::PyTuple>().map_err(|_| {
                    gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                })?;

                let token_data = tuple.get_item(0).map_err(|_| {
                    gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                })?;

                let expiration = tuple.get_item(1).map_err(|_| {
                    gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                })?;

                let bearer_token: String = token_data
                    .get_item("bearer_token")
                    .map_err(|_| {
                        gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                    })?
                    .cast::<pyo3::types::PyString>()
                    .map_err(|_| {
                        gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                    })?
                    .to_str()
                    .map_err(|_| {
                        gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                    })?
                    .to_string();

                // expiration is a float (timestamp)
                let expiry_f: f64 = expiration.extract().map_err(|_| {
                    gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                })?;

                let expiry = chrono::DateTime::from_timestamp(
                    expiry_f as i64,
                    ((expiry_f % 1.0) * 1_000_000_000.0) as u32,
                )
                .ok_or_else(|| {
                    gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
                })?;

                Ok(gcloud_sdk::Token {
                    token: bearer_token.into(),
                    token_type: "Bearer".to_string(),
                    expiry,
                })
            },
        )?;

        {
            let mut cache = self.cache.lock().unwrap();
            *cache = Some(token.clone());
        }
        Ok(token)
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

        let token_source_type = match credentials_provider {
            Some(provider) => {
                let is_none = Python::attach(|py| provider.is_none(py));
                if is_none {
                    gcloud_sdk::TokenSourceType::Default
                } else {
                    let token_source = PythonTokenSource {
                        provider,
                        cache: Mutex::new(None),
                    };
                    gcloud_sdk::TokenSourceType::ExternalSource(Box::new(token_source))
                }
            },
            None => gcloud_sdk::TokenSourceType::Default,
        };

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let client = rt.block_on(async {
            use arrow_bigquery_lib::BigQueryReadClientBuilder;

            let builder = arrow_bigquery_lib::ServiceConfigBuilder::new()
                .with_cred(token_source_type)
                .with_user_agent(user_agent)
                .with_quota_project_id(Some(quota_project_id));

            arrow_bigquery_lib::Client::from_builder(builder)
                .await
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
        })?;

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
        table: &str,
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
        let result = rt.block_on(async move {
            client
                .read_table(table, read_options)
                .await
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
        });

        match result {
            Ok((schema, receiver)) => Ok(ArrowStreamExporter {
                schema,
                receiver: std::sync::Mutex::new(Some(receiver)),
            }),
            Err(err) => Err(err),
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
        ..Default::default()
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

        assert_eq!(options.maintain_order, true);
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
        let err = match client.read_table(
            "projects/p/datasets/d/tables/t",
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
        let err = match client.read_table(
            "projects/p/datasets/d/tables/t",
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
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("failed to extract snapshot_time"));
        });
    }
}
