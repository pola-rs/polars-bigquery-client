use google_cloud_bigquery_v2::model::ErrorProto;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobError {
    pub reason: Option<String>,
    pub location: Option<String>,
    pub message: String,
    pub debug_info: Option<String>,
}

impl From<ErrorProto> for JobError {
    fn from(proto: ErrorProto) -> Self {
        Self {
            reason: (!proto.reason.is_empty()).then_some(proto.reason),
            location: (!proto.location.is_empty()).then_some(proto.location),
            message: proto.message,
            debug_info: (!proto.debug_info.is_empty()).then_some(proto.debug_info),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobFailedError {
    pub project_id: Option<String>,
    pub location: Option<String>,
    pub job_id: Option<String>,
    pub error: JobError,
    pub errors: Vec<JobError>,
}

#[derive(Debug)]
pub enum BigQueryError {
    Grpc(google_cloud_bigquery::Error),
    ClientBuilder(google_cloud_gax::client_builder::Error),
    Arrow(polars_error::PolarsError),
    JobFailed(Box<JobFailedError>),
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

fn format_job_error(
    f: &mut std::fmt::Formatter,
    err: &JobError,
    is_primary: bool,
) -> std::fmt::Result {
    let reason = err.reason.as_deref().map(str::trim).unwrap_or("");
    let location = err.location.as_deref().map(str::trim).unwrap_or("");
    let message = err.message.trim_end();
    let has_meta = !reason.is_empty() || !location.is_empty();

    if is_primary && has_meta {
        write!(f, " ")?;
    }

    match (!reason.is_empty(), !location.is_empty()) {
        (true, true) => write!(f, "({reason} at {location})"),
        (true, false) => write!(f, "({reason})"),
        (false, true) => write!(f, "(at {location})"),
        (false, false) => Ok(()),
    }?;

    if !message.is_empty() {
        if is_primary || has_meta {
            write!(f, ": {message}")?;
        } else {
            write!(f, "{message}")?;
        }
    }
    Ok(())
}

fn is_same_job_error(a: &JobError, b: &JobError) -> bool {
    a.reason.as_deref().map(str::trim) == b.reason.as_deref().map(str::trim)
        && a.location.as_deref().map(str::trim) == b.location.as_deref().map(str::trim)
        && a.message.trim() == b.message.trim()
}

impl std::fmt::Display for JobFailedError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let location = self.location.as_deref().map(str::trim).unwrap_or("");
        let job_id = self.job_id.as_deref().map(str::trim).unwrap_or("");
        match (!location.is_empty(), !job_id.is_empty()) {
            (true, true) => write!(f, "BigQuery job {location}:{job_id} failed")?,
            (false, true) => write!(f, "BigQuery job {job_id} failed")?,
            _ => write!(f, "BigQuery job failed")?,
        }
        format_job_error(f, &self.error, true)?;
        for extra in &self.errors {
            if !is_same_job_error(&self.error, extra) {
                write!(f, "; ")?;
                format_job_error(f, extra, false)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for BigQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Grpc(s) => write!(f, "BigQuery Storage API error: {}", s),
            Self::ClientBuilder(e) => write!(f, "BigQuery client builder error: {}", e),
            Self::Arrow(e) => write!(f, "Arrow decoding error: {}", e),
            Self::JobFailed(err) => std::fmt::Display::fmt(err, f),
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
            Self::JobFailed(_) | Self::Protocol(_) | Self::InvalidConfig(_) => None,
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

    #[test]
    fn test_bigquery_error_job_failed_display_and_causal_chain() {
        let primary = JobError::from(
            ErrorProto::new()
                .set_reason("resourcesExceeded")
                .set_message("Resources exceeded during query execution: Out of memory.\n"),
        );
        let duplicate = primary.clone();
        let job_failed = JobFailedError {
            project_id: Some("proj".into()),
            location: Some("US".into()),
            job_id: Some("job-1".into()),
            error: primary,
            errors: vec![duplicate],
        };
        let err = BigQueryError::JobFailed(Box::new(job_failed));

        assert!(err.source().is_none());
        assert_eq!(
            err.format_causal_chain(),
            "BigQuery job US:job-1 failed (resourcesExceeded): Resources exceeded during query execution: Out of memory."
        );

        let multi_job_failed = JobFailedError {
            project_id: None,
            location: None,
            job_id: None,
            error: JobError {
                reason: Some("invalid".into()),
                location: None,
                message: "Too many errors encountered.".into(),
                debug_info: None,
            },
            errors: vec![
                JobError {
                    reason: Some("invalid".into()),
                    location: None,
                    message: "Too many errors encountered.".into(),
                    debug_info: None,
                },
                JobError {
                    reason: Some("invalid".into()),
                    location: Some("row 42".into()),
                    message: "Bad int64 value".into(),
                    debug_info: None,
                },
            ],
        };
        let multi_err = BigQueryError::JobFailed(Box::new(multi_job_failed));
        assert_eq!(
            multi_err.to_string(),
            "BigQuery job failed (invalid): Too many errors encountered.; (invalid at row 42): Bad int64 value"
        );
    }
}
