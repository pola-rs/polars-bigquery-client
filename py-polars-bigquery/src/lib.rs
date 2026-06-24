use std::sync::{Mutex, Once};

use async_trait::async_trait;
use chrono::Utc;
use polars_arrow::datatypes::ArrowSchemaRef;
use pyo3::prelude::*;
use pyo3::pyfunction;

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
    receiver: std::sync::Mutex<Option<polars_bigquery_lib::BigQueryRecordBatchReceiver>>,
}

/// An iterator that adapts the asynchronous [`BigQueryRecordBatchReceiver`] into
/// a synchronous iterator yielding Arrow arrays.
///
/// This is used internally by [`ArrowStreamExporter`] to feed the Arrow C Stream.
/// Each iteration blocks on the Tokio runtime to receive the next batch.
struct ReceiverIterator {
    /// The receiver yielding record batches from the BigQuery Storage Read API.
    rx: polars_bigquery_lib::BigQueryRecordBatchReceiver,
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
                Ok(Some(batch)) => {
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

/// Reads a BigQuery table and returns an [`ArrowStreamExporter`] which can be
/// consumed by Polars in Python.
///
/// This function initializes the connection, sets up the BigQuery Storage Read API session,
/// spawns background tasks to read the streams, and returns the stream exporter.
///
/// # Arguments
/// * `table` - The BigQuery table ID in the format `project.dataset.table`.
/// * `quota_project_id` - The billing/quota project ID.
/// * `maintain_order` - If true, restricts the read session to a single stream to preserve row order.
/// * `credentials_provider` - A Python callable that returns Google OAuth2 credentials.
/// * `user_agent` - An optional user agent extension to append to the client header.
#[pyfunction]
pub fn read_bigquery(
    table: &str,
    quota_project_id: &str,
    maintain_order: bool,
    credentials_provider: Py<PyAny>,
    user_agent: Option<String>,
) -> pyo3::PyResult<ArrowStreamExporter> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    let token_source = PythonTokenSource {
        provider: credentials_provider,
        cache: Mutex::new(None),
    };
    let token_source_type = gcloud_sdk::TokenSourceType::ExternalSource(Box::new(token_source));

    let rt = pyo3_async_runtimes::tokio::get_runtime();

    let result = rt.block_on(async {
        let mut builder = polars_bigquery_lib::PolarsBigQueryClientBuilder::new()
            .with_token_source(token_source_type)
            .with_max_decoding_message_size(128 * 1024 * 1024);

        if let Some(ua) = user_agent {
            builder = builder.with_user_agent(ua);
        }

        let client = builder
            .build()
            .await
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

        polars_bigquery_lib::read_bigquery_with_client(
            client,
            table,
            quota_project_id,
            maintain_order,
        )
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
    let receiver =
        polars_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, Vec::new());

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
        polars_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);

    let exporter = ArrowStreamExporter {
        schema: schema_ref,
        receiver: std::sync::Mutex::new(Some(receiver)),
    };

    let drop_flag = DropFlag { value: flag };

    (exporter, drop_flag)
}

#[pymodule]
fn polars_bigquery(m: &Bound<PyModule>) -> PyResult<()> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    m.add_wrapped(wrap_pyfunction!(read_bigquery)).unwrap();
    m.add_wrapped(wrap_pyfunction!(_create_test_exporter))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(_test_create_exporter_with_drop_flag))
        .unwrap();
    m.add_class::<DropFlag>().unwrap();

    Ok(())
}
