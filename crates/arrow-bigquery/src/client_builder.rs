use google_cloud_auth::credentials::Credentials;
use google_cloud_bigquery::client::Read;
use google_cloud_gax::retry_throttler::{
    AdaptiveThrottler, RetryThrottlerArg, SharedRetryThrottler,
};

use crate::decode::SchemaDecodeOptions;
use crate::{BigQueryError, BigQueryJobService, Client};

static INIT_CRYPTO: std::sync::Once = std::sync::Once::new();

fn init_crypto() {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });
}

pub(crate) fn default_retry_throttler() -> SharedRetryThrottler {
    RetryThrottlerArg::from(AdaptiveThrottler::default()).into()
}

#[derive(Default)]
pub struct ServiceConfigBuilder {
    cred: Option<Credentials>,
    bq_endpoint: Option<String>,
    bqstorage_endpoint: Option<String>,
    user_agent: Option<String>,
    quota_project_id: Option<String>,
    schema_decode_options: SchemaDecodeOptions,
}

impl ServiceConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cred(mut self, cred: Credentials) -> Self {
        self.cred = Some(cred);
        self
    }

    pub fn with_bq_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.bq_endpoint = Some(endpoint.into());
        self
    }

    pub fn with_bqstorage_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.bqstorage_endpoint = Some(endpoint.into());
        self
    }

    pub fn with_user_agent(mut self, user_agent: Option<String>) -> Self {
        self.user_agent = user_agent;
        self
    }

    pub fn with_quota_project_id(mut self, quota_project_id: Option<String>) -> Self {
        self.quota_project_id = quota_project_id;
        self
    }

    /// Configures whether BigQuery `google:sqlType:datetime` extension metadata
    /// is preserved on decoded Arrow schemas instead of being stripped to native
    /// `Timestamp(Microsecond, None)`.
    ///
    /// Defaults to `false` (stripped for Polars Arrow compatibility). Pure-Rust
    /// or non-Polars consumers that wish to preserve the original extension type
    /// may enable this option.
    pub fn with_preserve_datetime_extension(mut self, preserve: bool) -> Self {
        self.schema_decode_options.strip_datetime_extension = !preserve;
        self
    }

    /// Configures whether decoded Arrow schemas reject types unsupported by the
    /// `polars-arrow` C Data Interface FFI layer (`Interval(MonthDayNano)`,
    /// `Interval(MonthDayMillis)`, `Unknown`).
    ///
    /// Defaults to `true`. Pure-Rust consumers that decode `RecordBatch` values
    /// directly without exporting over the Arrow C Data Interface may disable this.
    pub fn with_validate_ffi_types(mut self, validate: bool) -> Self {
        self.schema_decode_options.validate_ffi_types = validate;
        self
    }

    pub async fn build(self) -> Result<Client, BigQueryError> {
        let quota_project_id = self
            .quota_project_id
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| BigQueryError::InvalidConfig("quota_project_id is required".into()))?;

        init_crypto();

        let retry_throttler = default_retry_throttler();

        let mut bq_builder = BigQueryJobService::builder();
        let mut read_builder = Read::builder()
            .with_retry_policy(crate::bigquery_read_retry::create_read_session_retry_policy())
            .with_backoff_policy(crate::bigquery_read_retry::default_backoff_policy())
            .with_retry_throttler(retry_throttler.clone());

        if let Some(cred) = self.cred {
            bq_builder = bq_builder.with_credentials(cred.clone());
            read_builder = read_builder.with_credentials(cred);
        }

        if let Some(endpoint) = self.bq_endpoint {
            bq_builder = bq_builder.with_endpoint(endpoint);
        }

        if let Some(endpoint) = self.bqstorage_endpoint {
            read_builder = read_builder.with_endpoint(endpoint);
        }

        let bq_client = bq_builder.build().await?;
        let read_client = read_builder.build().await?;

        Ok(Client::from_parts_with_options(
            bq_client,
            read_client,
            quota_project_id,
            self.user_agent,
            retry_throttler,
            self.schema_decode_options,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_config_builder_defaults() {
        let builder = ServiceConfigBuilder::new();
        assert!(builder.cred.is_none());
        assert!(builder.bq_endpoint.is_none());
        assert!(builder.bqstorage_endpoint.is_none());
        assert!(builder.user_agent.is_none());
        assert!(builder.quota_project_id.is_none());
        assert!(builder.schema_decode_options.strip_datetime_extension);
        assert!(builder.schema_decode_options.validate_ffi_types);
    }

    #[test]
    fn test_service_config_builder_custom() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let builder = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_bq_endpoint("https://custom.bq.endpoint.com")
            .with_bqstorage_endpoint("https://custom.bqstorage.endpoint.com")
            .with_user_agent(Some("custom-agent/1.0".to_string()))
            .with_quota_project_id(Some("custom-project".to_string()))
            .with_preserve_datetime_extension(true)
            .with_validate_ffi_types(false);

        assert!(builder.cred.is_some());
        assert_eq!(
            builder.bq_endpoint.as_deref(),
            Some("https://custom.bq.endpoint.com")
        );
        assert_eq!(
            builder.bqstorage_endpoint.as_deref(),
            Some("https://custom.bqstorage.endpoint.com")
        );
        assert_eq!(builder.user_agent.as_deref(), Some("custom-agent/1.0"));
        assert_eq!(builder.quota_project_id.as_deref(), Some("custom-project"));
        assert!(!builder.schema_decode_options.strip_datetime_extension);
        assert!(!builder.schema_decode_options.validate_ffi_types);
    }

    #[tokio::test]
    async fn test_service_config_builder_build_preserves_decode_options() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let client = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_quota_project_id(Some("custom-project".to_string()))
            .with_preserve_datetime_extension(true)
            .with_validate_ffi_types(false)
            .build()
            .await
            .expect("should build client");

        assert_eq!(
            client.schema_decode_options(),
            SchemaDecodeOptions {
                strip_datetime_extension: false,
                validate_ffi_types: false,
            }
        );
    }

    #[tokio::test]
    async fn test_service_config_builder_build_success() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let client = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_bq_endpoint("https://custom.bq.endpoint.com")
            .with_bqstorage_endpoint("https://custom.bqstorage.endpoint.com")
            .with_user_agent(Some("custom-agent/1.0".to_string()))
            .with_quota_project_id(Some("custom-project".to_string()))
            .build()
            .await
            .expect("should build client");

        assert_eq!(client.quota_project_id(), "custom-project");
    }

    #[tokio::test]
    async fn test_service_config_builder_missing_or_empty_quota_project_id() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let result = ServiceConfigBuilder::new()
            .with_cred(cred.clone())
            .build()
            .await;
        match result {
            Err(BigQueryError::InvalidConfig(msg)) => {
                assert!(msg.contains("quota_project_id is required"));
            },
            other => panic!("expected InvalidConfig error, got {:?}", other.err()),
        }

        let empty_result = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_quota_project_id(Some("   ".to_string()))
            .build()
            .await;
        assert!(matches!(empty_result, Err(BigQueryError::InvalidConfig(_))));
    }
}
