use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Helper that extracts an optional string attribute from a Python object:
/// - Returns `Ok(None)` only when the attribute does not exist (`AttributeError`) or is `None`.
/// - Propagates any exception raised inside a property getter (e.g., `RuntimeError`, `KeyboardInterrupt`)
///   instead of silently swallowing it.
fn get_optional_string_attr(obj: &Bound<'_, PyAny>, attr: &str) -> PyResult<Option<String>> {
    let py = obj.py();
    match obj.getattr(attr) {
        Ok(val) if val.is_none() => Ok(None),
        Ok(val) => val.extract::<String>().map(Some),
        Err(err) if err.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(err) => Err(err),
    }
}

/// A Python-exposed representation of an immutable BigQuery table identifier.
#[pyclass(name = "BigQueryTableId", frozen, eq, hash)]
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
            inner: arrow_bigquery_lib::BigQueryTableId::new(project_id, dataset_id, table_id),
        }
    }

    #[staticmethod]
    pub fn from_string(s: &str) -> PyResult<Self> {
        let inner: arrow_bigquery_lib::BigQueryTableId = s
            .parse()
            .map_err(|_| PyValueError::new_err(format!("Invalid table ID: {s}")))?;
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
            return Ok(existing.get().clone());
        }

        if let Ok(s) = table_id.extract::<&str>() {
            return Self::from_string(s);
        }

        let project = match get_optional_string_attr(table_id, "project")? {
            Some(p) => Some(p),
            None => get_optional_string_attr(table_id, "project_id")?,
        };
        let dataset = get_optional_string_attr(table_id, "dataset_id")?;
        let table = get_optional_string_attr(table_id, "table_id")?;

        if let (Some(project), Some(dataset), Some(table)) = (project, dataset, table) {
            return Ok(Self::new(project, dataset, table));
        }

        Err(PyTypeError::new_err(format!(
            "Expected table_id to be a string or BigQuery table object, got {}",
            table_id.get_type()
        )))
    }

    #[getter]
    pub fn project_id(&self) -> &str {
        &self.inner.project_id
    }

    #[getter]
    pub fn dataset_id(&self) -> &str {
        &self.inner.dataset_id
    }

    #[getter]
    pub fn table_id(&self) -> &str {
        &self.inner.table_id
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

#[cfg(test)]
mod tests {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::PyDict;

    use super::*;

    #[test]
    fn test_bigquery_table_id_parse_propagates_property_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
class BrokenTable:
    @property
    def project(self):
        raise RuntimeError('custom property failure')
obj = BrokenTable()
",
                None,
                Some(&locals),
            )
            .unwrap();
            let obj = locals.get_item("obj").unwrap().unwrap();
            let err = BigQueryTableId::parse(&obj).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("custom property failure"));
        });
    }
}
