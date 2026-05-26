use std::sync::Mutex;
use std::sync::Once;

use async_trait::async_trait;
use chrono::Utc;

use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3_polars::PyDataFrame;

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

        let token = Python::attach(|py| -> Result<gcloud_sdk::Token, gcloud_sdk::error::Error> {
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
        })?;

        {
            let mut cache = self.cache.lock().unwrap();
            *cache = Some(token.clone());
        }
        Ok(token)
    }
}

#[pyfunction]
pub fn read_bigquery(
    table: &str,
    quota_project_id: &str,
    is_ordered: bool,
    credentials_provider: Py<PyAny>,
    user_agent: Option<String>,
) -> pyo3::PyResult<PyDataFrame> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
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

        if let Some(ext) = user_agent {
            builder = builder.with_user_agent(ext);
        }

        let client = builder
            .build()
            .await
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

        polars_bigquery_lib::read_bigquery_with_client(
            client,
            table,
            quota_project_id,
            is_ordered,
        )
        .await
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
    });

    match result {
        Ok(value) => Ok(pyo3_polars::PyDataFrame(value)),
        Err(err) => Err(err),
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
