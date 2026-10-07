use pyo3::prelude::*;

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
