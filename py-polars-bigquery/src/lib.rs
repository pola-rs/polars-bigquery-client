use std::sync::Once;

use async_trait::async_trait;
use polars_bigquery_lib::read_bigquery_async;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3_polars::PyDataFrame;

static INIT_CRYPTO: Once = Once::new();

struct PythonTokenSource {
    provider: Py<PyAny>,
}

#[async_trait]
impl gcloud_sdk::Source for PythonTokenSource {
    async fn token(&self) -> Result<gcloud_sdk::Token, gcloud_sdk::error::Error> {
        Python::with_gil(|py| {
            let provider = self.provider.bind(py);
            let result = provider.call0().map_err(|_| {
                gcloud_sdk::error::Error::from(gcloud_sdk::error::ErrorKind::TokenSource)
            })?;

            // result is (token_data, expiration)
            let tuple = result.downcast::<pyo3::types::PyTuple>().map_err(|_| {
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
                .downcast::<pyo3::types::PyString>()
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
        })
    }
}

#[pyfunction]
pub fn read_bigquery(
    table: &str,
    quota_project_id: &str,
    is_ordered: bool,
    credentials_provider: Py<PyAny>,
) -> pyo3::PyResult<PyDataFrame> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    let token_source = PythonTokenSource {
        provider: credentials_provider,
    };
    let token_source_type = gcloud_sdk::TokenSourceType::ExternalSource(Box::new(token_source));

    let rt = pyo3_async_runtimes::tokio::get_runtime();
    let result = rt.block_on(read_bigquery_async(
        table,
        quota_project_id,
        is_ordered,
        token_source_type,
    ));
    match result {
        Ok(value) => Ok(pyo3_polars::PyDataFrame(value)),
        Err(err) => Err(pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            err.to_string(),
        )),
    }
}

#[pymodule]
fn polars_bigquery(m: &Bound<PyModule>) -> PyResult<()> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    m.add_wrapped(wrap_pyfunction!(read_bigquery)).unwrap();

    Ok(())
}
