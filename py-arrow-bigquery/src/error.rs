use std::error::Error as StdError;

use arrow_bigquery_lib::BigQueryError as LibBigQueryError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    arrow_bigquery.exceptions,
    BigQueryError,
    PyRuntimeError,
    "Error raised by the BigQuery API."
);

/// Converts a [`LibBigQueryError`] into a [`PyErr`] while preserving the full causal chain:
/// - Configuration/validation errors ([`LibBigQueryError::InvalidConfig`]) raise [`PyValueError`].
/// - All other BigQuery errors raise [`BigQueryError`] (which subclasses [`PyRuntimeError`]),
///   with the top-level message formatted via [`LibBigQueryError::format_causal_chain`].
/// - If any underlying [`StdError::source`] is an original [`PyErr`] (such as an exception raised
///   by a Python credentials provider), it is attached via Python's `__cause__` (`raise ... from ...`).
pub(crate) fn to_py_err(py: Python<'_>, err: LibBigQueryError) -> PyErr {
    if matches!(err, LibBigQueryError::InvalidConfig(_)) {
        return PyValueError::new_err(err.format_causal_chain());
    }

    let mut current = err.source();
    let mut py_cause: Option<PyErr> = None;
    while let Some(src) = current {
        if let Some(orig_py_err) = src.downcast_ref::<PyErr>() {
            py_cause = Some(orig_py_err.clone_ref(py));
            break;
        }
        current = src.source();
    }

    let top_err = BigQueryError::new_err(err.format_causal_chain());
    if let Some(inner_cause) = py_cause {
        top_err.set_cause(py, Some(inner_cause));
    }
    top_err
}

#[cfg(test)]
mod tests {
    use google_cloud_auth::errors::CredentialsError;
    use google_cloud_bigquery::Error as GaxError;
    use google_cloud_gax::client_builder::Error as ClientBuilderError;

    use super::*;

    #[test]
    fn test_to_py_err_preserves_rust_causal_chain_without_duplicate_py_causes() {
        Python::initialize();
        let io_err = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing /etc/gcp/credentials.json",
        );
        let cred_err = CredentialsError::from_source(false, io_err);
        let builder_err = ClientBuilderError::cred(cred_err);
        let bq_err = LibBigQueryError::from(builder_err);

        Python::attach(|py| {
            let py_err = to_py_err(py, bq_err);
            assert!(py_err.is_instance_of::<BigQueryError>(py));
            assert!(py_err.is_instance_of::<PyRuntimeError>(py));
            let top_msg = py_err.to_string();
            assert!(top_msg.contains("could not create default credentials"));
            assert!(top_msg.contains("cannot create auth headers"));
            assert!(top_msg.contains("missing /etc/gcp/credentials.json"));

            // Pure-Rust error chains should not synthesize duplicate BigQueryError __cause__ layers
            assert!(py_err.cause(py).is_none());
        });
    }

    #[test]
    fn test_to_py_err_invalid_config_maps_to_value_error() {
        Python::initialize();
        Python::attach(|py| {
            let bq_err = LibBigQueryError::InvalidConfig("quota_project_id is required".into());
            let py_err = to_py_err(py, bq_err);
            assert!(py_err.is_instance_of::<PyValueError>(py));
            assert!(py_err.to_string().contains("quota_project_id is required"));
        });
    }

    #[test]
    fn test_to_py_err_recovers_original_python_exception_in_cause_chain() {
        Python::initialize();
        Python::attach(|py| {
            let orig_py_err = PyValueError::new_err("custom python token failure");
            let cred_err =
                CredentialsError::new(false, "Python credentials provider failed", orig_py_err);
            let gax_err = GaxError::authentication(cred_err);
            let bq_err = LibBigQueryError::from(gax_err);

            let py_err = to_py_err(py, bq_err);
            assert!(py_err.is_instance_of::<BigQueryError>(py));
            assert!(py_err.to_string().contains("custom python token failure"));

            // The original PyValueError should be attached directly as __cause__
            let cause = py_err.cause(py).expect("expected direct Python __cause__");
            assert!(cause.is_instance_of::<PyValueError>(py));
            assert!(cause.to_string().contains("custom python token failure"));
            assert!(cause.cause(py).is_none());
        });
    }
}
