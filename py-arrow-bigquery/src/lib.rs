mod auth;
mod error;
mod stream;
mod table_id;
#[cfg(feature = "testing")]
mod testing;

use google_cloud_auth::credentials::Credentials;
use google_cloud_bigquery::model::arrow_serialization_options::CompressionCodec;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDateTime;
pub use stream::ArrowStreamExporter;
pub use table_id::BigQueryTableId;

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
        py: Python<'_>,
        quota_project_id: String,
        credentials_provider: Option<Py<PyAny>>,
        user_agent: Option<String>,
    ) -> PyResult<Self> {
        let cred = match credentials_provider {
            Some(provider) if !provider.is_none(py) => {
                let token_source = auth::PythonTokenSource::new(provider);
                Some(Credentials::from(token_source))
            },
            _ => None,
        };

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let result = py.detach(|| {
            rt.block_on(async {
                let mut builder = arrow_bigquery_lib::Client::builder()
                    .with_user_agent(user_agent)
                    .with_quota_project_id(Some(quota_project_id));
                if let Some(cred) = cred {
                    builder = builder.with_cred(cred);
                }

                arrow_bigquery_lib::Client::from_builder(builder).await
            })
        });

        let client = result.map_err(|err| error::to_py_err(py, err))?;
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
        py: Python<'_>,
        table: &BigQueryTableId,
        arrow_buffer_compression: &str,
        maintain_order: bool,
        max_stream_count: Option<i32>,
        row_restriction: &str,
        sample_percentage: Option<f64>,
        selected_fields: Option<Vec<String>>,
        snapshot_time: Option<Py<PyDateTime>>,
    ) -> PyResult<ArrowStreamExporter> {
        let read_options = parse_read_options(
            py,
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
        let table = table.inner.clone();
        let result =
            py.detach(|| rt.block_on(async move { client.read_table(&table, read_options).await }));

        let (schema, receiver) = result.map_err(|err| error::to_py_err(py, err))?;
        Ok(ArrowStreamExporter::new(schema, receiver))
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_read_options(
    py: Python<'_>,
    arrow_buffer_compression: &str,
    maintain_order: bool,
    max_stream_count: Option<i32>,
    row_restriction: &str,
    sample_percentage: Option<f64>,
    selected_fields: Option<Vec<String>>,
    snapshot_time: Option<Py<PyDateTime>>,
) -> PyResult<arrow_bigquery_lib::ReadOptions> {
    let snapshot_time: Option<chrono::DateTime<chrono::Utc>> = match snapshot_time {
        Some(dt) => Some(
            dt.extract::<chrono::DateTime<chrono::Utc>>(py)
                .map_err(|_| PyValueError::new_err("failed to extract snapshot_time"))?,
        ),
        None => None,
    };
    let arrow_buffer_compression = Some(match arrow_buffer_compression {
        "unspecified" => CompressionCodec::CompressionUnspecified,
        "lz4frame" => CompressionCodec::Lz4Frame,
        "zstd" => CompressionCodec::Zstd,
        _ => {
            return Err(PyValueError::new_err(format!(
                "got unexpected compression codec {arrow_buffer_compression}"
            )))
        },
    });
    Ok(arrow_bigquery_lib::ReadOptions {
        arrow_buffer_compression,
        maintain_order,
        max_stream_count,
        row_restriction: row_restriction.to_owned(),
        sample_percentage,
        selected_fields: selected_fields.unwrap_or_default(),
        snapshot_time,
    })
}

#[pymodule]
#[pyo3(name = "_native")]
fn polars_bigquery(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("BigQueryError", m.py().get_type::<error::BigQueryError>())?;
    m.add_class::<Client>()?;
    m.add_class::<BigQueryTableId>()?;
    #[cfg(feature = "testing")]
    {
        m.add_wrapped(wrap_pyfunction!(testing::_create_test_exporter))?;
        m.add_wrapped(wrap_pyfunction!(
            testing::_test_create_exporter_with_drop_flag
        ))?;
        m.add_class::<testing::DropFlag>()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;

    #[test]
    fn test_parse_read_options_defaults() {
        Python::initialize();
        Python::attach(|py| {
            let options =
                parse_read_options(py, "lz4frame", false, None, "", None, None, None).unwrap();
            assert!(!options.maintain_order);
            assert_eq!(options.snapshot_time, None);
            assert!(options.selected_fields.is_empty());
            assert_eq!(options.row_restriction, "");
            assert_eq!(
                options.arrow_buffer_compression,
                Some(CompressionCodec::Lz4Frame)
            );
            assert_eq!(options.sample_percentage, None);
        });
    }

    #[test]
    fn test_parse_read_options_all_fields() {
        Python::initialize();
        Python::attach(|py| {
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
            let dt_py: Py<PyDateTime> = dt.extract().unwrap();
            let expected_dt = chrono::DateTime::from_timestamp(1_700_000_000, 500_000_000).unwrap();

            let options = parse_read_options(
                py,
                "zstd",
                true,
                Some(16),
                "col1 > 100",
                Some(42.5),
                Some(vec!["col1".to_string(), "col2".to_string()]),
                Some(dt_py),
            )
            .unwrap();

            assert!(options.maintain_order);
            assert_eq!(options.max_stream_count, Some(16));
            assert_eq!(options.snapshot_time, Some(expected_dt));
            assert_eq!(options.selected_fields, vec!["col1", "col2"]);
            assert_eq!(options.row_restriction, "col1 > 100");
            assert_eq!(
                options.arrow_buffer_compression,
                Some(CompressionCodec::Zstd)
            );
            assert_eq!(options.sample_percentage, Some(42.5));
        });
    }

    #[test]
    fn test_parse_read_options_compression_codecs() {
        Python::initialize();

        let valid_codecs = [
            ("unspecified", CompressionCodec::CompressionUnspecified),
            ("lz4frame", CompressionCodec::Lz4Frame),
            ("zstd", CompressionCodec::Zstd),
        ];

        Python::attach(|py| {
            for (codec_str, expected) in valid_codecs {
                let options =
                    parse_read_options(py, codec_str, false, None, "", None, None, None).unwrap();
                assert_eq!(options.arrow_buffer_compression, Some(expected));
            }

            let err =
                match parse_read_options(py, "invalid_codec", false, None, "", None, None, None) {
                    Err(e) => e,
                    Ok(_) => panic!("expected Err for invalid compression codec"),
                };
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("got unexpected compression codec invalid_codec"));
        });
    }

    #[test]
    fn test_parse_read_options_snapshot_time_naive_error() {
        Python::initialize();
        Python::attach(|py| {
            let datetime_mod = py.import("datetime").unwrap();
            let dt = datetime_mod
                .getattr("datetime")
                .unwrap()
                .call1((2023, 11, 14, 22, 13, 20))
                .unwrap();
            let dt_py: Py<PyDateTime> = dt.extract().unwrap();

            let err = match parse_read_options(
                py,
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
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("failed to extract snapshot_time"));
        });
    }

    #[test]
    fn test_client_read_table_invalid_compression() {
        Python::initialize();
        Python::attach(|py| {
            let client = Client::new(py, "test-project".to_string(), None, None).unwrap();
            let table = BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string());
            let err = match client.read_table(
                py,
                &table,
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
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err
                .to_string()
                .contains("got unexpected compression codec invalid_codec"));
        });
    }

    #[test]
    fn test_client_read_table_naive_snapshot_time() {
        Python::initialize();
        Python::attach(|py| {
            let datetime_mod = py.import("datetime").unwrap();
            let dt = datetime_mod
                .getattr("datetime")
                .unwrap()
                .call1((2023, 11, 14, 22, 13, 20))
                .unwrap();
            let dt_py: Py<PyDateTime> = dt.extract().unwrap();

            let client = Client::new(py, "test-project".to_string(), None, None).unwrap();
            let table = BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string());
            let err = match client.read_table(
                py,
                &table,
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
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("failed to extract snapshot_time"));
        });
    }

    #[test]
    fn test_client_new_with_credentials_provider() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"def p(): return ({'bearer_token': 'tok'}, None)",
                None,
                Some(&locals),
            )
            .unwrap();
            let provider = locals.get_item("p").unwrap().unwrap().unbind();

            let client = Client::new(
                py,
                "test-project".to_string(),
                Some(provider),
                Some("test-agent/1.0".to_string()),
            );
            assert!(client.is_ok());
        });
    }

    #[test]
    fn test_client_read_table_preserves_python_cause_chain() {
        use google_cloud_auth::errors::CredentialsError;
        use google_cloud_bigquery::client::Read;
        use google_cloud_bigquery::model::{CreateReadSessionRequest, ReadSession};
        use google_cloud_gax::error::Error as GaxError;
        use google_cloud_gax::options::RequestOptions;
        use google_cloud_gax::response::Response;

        #[derive(Debug)]
        struct FailingStub;

        impl google_cloud_bigquery::stub::Read for FailingStub {
            async fn create_read_session(
                &self,
                _req: CreateReadSessionRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<Response<ReadSession>> {
                let py_err = Python::attach(|_py| {
                    PyValueError::new_err("underlying python token generator broke")
                });
                let cred_err =
                    CredentialsError::new(false, "Python credentials provider failed", py_err);
                Err(GaxError::authentication(cred_err))
            }
        }

        Python::initialize();
        Python::attach(|py| {
            let inner_client = arrow_bigquery_lib::Client::new(
                Read::from_stub(FailingStub),
                "test-project".to_string(),
            );
            let client = Client {
                client: inner_client,
            };
            let table = BigQueryTableId::new("p".to_string(), "d".to_string(), "t".to_string());

            let err = match client
                .read_table(py, &table, "lz4frame", false, None, "", None, None, None)
            {
                Err(e) => e,
                Ok(_) => panic!("expected read_table to fail"),
            };

            assert!(err.is_instance_of::<error::BigQueryError>(py));
            let msg = err.to_string();
            assert!(msg.contains("Python credentials provider failed"));
            assert!(msg.contains("caused by: ValueError: underlying python token generator broke"));

            // Verify Python __cause__ chain reaches the original ValueError
            let mut curr = err.cause(py);
            let mut root = None;
            while let Some(c) = curr {
                let next = c.cause(py);
                root = Some(c);
                curr = next;
            }
            let root_err = root.expect("expected root Python __cause__");
            assert!(root_err.is_instance_of::<PyValueError>(py));
            assert!(root_err
                .to_string()
                .contains("underlying python token generator broke"));
        });
    }
}
