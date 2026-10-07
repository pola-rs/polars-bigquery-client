use std::error::Error as StdError;

use arrow_bigquery_lib::BigQueryError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Converts a [`BigQueryError`] into a [`PyErr`] while preserving the full causal chain:
/// - The top-level [`PyRuntimeError`] message includes [`BigQueryError::format_causal_chain`].
/// - Every underlying [`StdError::source`] is attached via Python's `__cause__` (`raise ... from ...`),
///   recovering any original [`PyErr`] instances (such as exceptions raised by a Python credentials provider).
pub(crate) fn to_py_err(py: Python<'_>, err: BigQueryError) -> PyErr {
    let mut sources: Vec<&(dyn StdError + 'static)> = Vec::new();
    let mut current = err.source();
    while let Some(src) = current {
        sources.push(src);
        current = src.source();
    }

    let mut cause: Option<PyErr> = None;
    for src in sources.into_iter().rev() {
        if let Some(orig_py_err) = src.downcast_ref::<PyErr>() {
            let cloned = orig_py_err.clone_ref(py);
            if let Some(prev_cause) = cause.take() {
                cloned.set_cause(py, Some(prev_cause));
            }
            cause = Some(cloned);
        } else {
            let layer_err = PyRuntimeError::new_err(src.to_string());
            if let Some(prev_cause) = cause.take() {
                layer_err.set_cause(py, Some(prev_cause));
            }
            cause = Some(layer_err);
        }
    }

    let top_err = PyRuntimeError::new_err(err.format_causal_chain());
    if let Some(inner_cause) = cause {
        top_err.set_cause(py, Some(inner_cause));
    }
    top_err
}

#[cfg(test)]
mod tests {
    use google_cloud_auth::errors::CredentialsError;
    use google_cloud_bigquery::Error as GaxError;
    use google_cloud_gax::client_builder::Error as ClientBuilderError;
    use pyo3::exceptions::PyValueError;

    use super::*;

    #[test]
    fn test_to_py_err_preserves_rust_causal_chain_and_py_cause() {
        Python::initialize();
        let io_err = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing /etc/gcp/credentials.json",
        );
        let cred_err = CredentialsError::from_source(false, io_err);
        let builder_err = ClientBuilderError::cred(cred_err);
        let bq_err = BigQueryError::from(builder_err);

        Python::attach(|py| {
            let py_err = to_py_err(py, bq_err);
            let top_msg = py_err.to_string();
            assert!(top_msg.contains("could not create default credentials"));
            assert!(top_msg.contains("missing /etc/gcp/credentials.json"));

            // Walk Python __cause__ chain
            let cause1 = py_err.cause(py).expect("expected first __cause__");
            assert!(cause1
                .to_string()
                .contains("could not create default credentials"));

            let cause2 = cause1.cause(py).expect("expected second __cause__");
            assert!(cause2.to_string().contains("cannot create auth headers"));

            let cause3 = cause2.cause(py).expect("expected root __cause__");
            assert!(cause3
                .to_string()
                .contains("missing /etc/gcp/credentials.json"));
            assert!(cause3.cause(py).is_none());
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
            let bq_err = BigQueryError::from(gax_err);

            let py_err = to_py_err(py, bq_err);
            assert!(py_err.to_string().contains("custom python token failure"));

            // Walk down to the root __cause__ and verify it is the exact PyValueError
            let mut curr = py_err.cause(py);
            let mut root = None;
            while let Some(c) = curr {
                let next = c.cause(py);
                root = Some(c);
                curr = next;
            }
            let root_err = root.expect("expected root Python cause");
            assert!(root_err.is_instance_of::<PyValueError>(py));
            assert!(root_err.to_string().contains("custom python token failure"));
        });
    }
}
