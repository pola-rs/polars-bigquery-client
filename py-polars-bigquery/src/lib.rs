use std::sync::{Mutex, Once};

use async_trait::async_trait;
use chrono::Utc;
use polars_arrow::datatypes::ArrowSchemaRef;
use pyo3::prelude::*;
use pyo3::pyfunction;

static INIT_CRYPTO: Once = Once::new();

struct PythonTokenSource {
    provider: Py<PyAny>,
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

#[pyclass]
pub struct ArrowStreamExporter {
    schema: ArrowSchemaRef,
    receiver: std::sync::Mutex<Option<polars_bigquery_lib::BigQueryRecordBatchReceiver>>,
}

struct ReceiverIterator {
    rx: polars_bigquery_lib::BigQueryRecordBatchReceiver,
    dtype: polars_arrow::datatypes::ArrowDataType,
}

impl Iterator for ReceiverIterator {
    type Item = pyo3_polars::export::polars_error::PolarsResult<Box<dyn polars_arrow::array::Array>>;

    fn next(&mut self) -> Option<Self::Item> {
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let batch = rt.block_on(self.rx.recv())?;

        let len = batch.len();
        let (_, arrays) = batch.into_schema_and_arrays();
        let struct_array =
            polars_arrow::array::StructArray::new(self.dtype.clone(), len, arrays, None);
        Some(Ok(Box::new(struct_array) as Box<dyn polars_arrow::array::Array>))
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

        let iter = ReceiverIterator { rx, dtype: dtype.clone() };
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

        polars_bigquery_lib::read_bigquery_with_client(client, table, quota_project_id, maintain_order)
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

#[pymodule]
fn polars_bigquery(m: &Bound<PyModule>) -> PyResult<()> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    m.add_wrapped(wrap_pyfunction!(read_bigquery)).unwrap();

    Ok(())
}
