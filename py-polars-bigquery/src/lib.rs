use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3_polars::PyDataFrame;
use std::sync::Once;

use polars_bigquery_lib::read_bigquery_async;

static INIT_CRYPTO: Once = Once::new();

#[pyfunction]
pub fn read_bigquery(table: &str, quota_project_id: &str) -> pyo3::PyResult<PyDataFrame> {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });

    let rt = pyo3_async_runtimes::tokio::get_runtime();
    let result = rt.block_on(read_bigquery_async(table, quota_project_id));
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
