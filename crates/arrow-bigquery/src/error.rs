#[derive(Debug)]
pub enum BigQueryError {
    Grpc(google_cloud_bigquery::Error),
    ClientBuilder(google_cloud_gax::client_builder::Error),
    Arrow(polars_error::PolarsError),
    Protocol(String),
    InvalidConfig(String),
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl BigQueryError {
    /// Formats this error along with its entire `std::error::Error::source()` causal chain,
    /// avoiding duplicate segments when an outer error's `Display` already embeds its direct child.
    pub fn format_causal_chain(&self) -> String {
        use std::error::Error as _;
        use std::fmt::Write as _;

        let mut full = self.to_string();
        let mut prev_msg = full.clone();
        let mut current = self.source();

        while let Some(src) = current {
            let msg = src.to_string();
            if !msg.is_empty() && !prev_msg.contains(&msg) {
                let _ = write!(full, ": caused by: {msg}");
            }
            prev_msg = msg;
            current = src.source();
        }

        full
    }
}

impl std::fmt::Display for BigQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Grpc(s) => write!(f, "BigQuery Storage API error: {}", s),
            Self::ClientBuilder(e) => write!(f, "BigQuery client builder error: {}", e),
            Self::Arrow(e) => write!(f, "Arrow decoding error: {}", e),
            Self::Protocol(msg) => write!(f, "BigQuery protocol error: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "BigQuery configuration error: {}", msg),
            Self::Other(e) => write!(f, "BigQuery client error: {}", e),
        }
    }
}

impl std::error::Error for BigQueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Grpc(e) => Some(e),
            Self::ClientBuilder(e) => Some(e),
            Self::Arrow(e) => Some(e),
            Self::Protocol(_) | Self::InvalidConfig(_) => None,
            Self::Other(e) => Some(e.as_ref()),
        }
    }
}

impl From<google_cloud_bigquery::Error> for BigQueryError {
    fn from(s: google_cloud_bigquery::Error) -> Self {
        Self::Grpc(s)
    }
}

impl From<google_cloud_gax::client_builder::Error> for BigQueryError {
    fn from(e: google_cloud_gax::client_builder::Error) -> Self {
        Self::ClientBuilder(e)
    }
}

impl From<polars_error::PolarsError> for BigQueryError {
    fn from(e: polars_error::PolarsError) -> Self {
        Self::Arrow(e)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use google_cloud_auth::errors::CredentialsError;
    use google_cloud_gax::client_builder::Error as ClientBuilderError;
    use google_cloud_gax::error::Error as GaxError;

    use super::*;

    #[test]
    fn test_bigquery_error_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<BigQueryError>();
    }

    #[test]
    fn test_bigquery_error_client_builder_causal_chain() {
        let io_err = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing /tmp/application_default_credentials.json",
        );
        let cred_err = CredentialsError::from_source(false, io_err);
        let builder_err = ClientBuilderError::cred(cred_err);
        let err = BigQueryError::from(builder_err);

        // Verify std::error::Error::source() walks all 3 underlying causes
        let src1 = err.source().expect("should have ClientBuilderError source");
        assert!(src1
            .to_string()
            .contains("could not create default credentials"));

        let src2 = src1.source().expect("should have CredentialsError source");
        assert!(src2.to_string().contains("cannot create auth headers"));

        let src3 = src2.source().expect("should have io::Error root cause");
        assert!(src3
            .to_string()
            .contains("missing /tmp/application_default_credentials.json"));
        assert!(src3.source().is_none());

        // Verify format_causal_chain() includes every cause without repeating duplicate layers
        let formatted = err.format_causal_chain();
        assert!(formatted
            .contains("BigQuery client builder error: could not create default credentials"));
        assert!(formatted.contains("caused by: cannot create auth headers"));
        assert!(formatted.contains("caused by: missing /tmp/application_default_credentials.json"));
        assert_eq!(
            formatted
                .matches("could not create default credentials")
                .count(),
            1
        );
    }

    #[test]
    fn test_bigquery_error_grpc_exhausted_auth_causal_chain() {
        let root_io = std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "metadata server reset connection",
        );
        let cred_err = CredentialsError::new(true, "token refresh failed", root_io);
        let auth_err = GaxError::authentication(cred_err);
        let exhausted_err = GaxError::exhausted(auth_err);
        let err = BigQueryError::from(exhausted_err);

        // Walk the std::error::Error::source() chain
        let s1 = err.source().expect("Exhausted GaxError");
        let s2 = s1.source().expect("Authentication GaxError");
        let s3 = s2.source().expect("CredentialsError");
        let s4 = s3.source().expect("std::io::Error");
        assert!(s4.to_string().contains("metadata server reset connection"));

        let formatted = err.format_causal_chain();
        assert!(formatted.contains("BigQuery Storage API error:"));
        assert!(formatted.contains("token refresh failed"));
        assert!(formatted.contains("caused by: metadata server reset connection"));
    }

    #[test]
    fn test_bigquery_error_invalid_config_and_protocol() {
        let err = BigQueryError::InvalidConfig("quota_project_id is required".into());
        assert!(err.source().is_none());
        assert_eq!(
            err.format_causal_chain(),
            "BigQuery configuration error: quota_project_id is required"
        );
    }
}
