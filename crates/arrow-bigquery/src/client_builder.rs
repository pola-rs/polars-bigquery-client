use google_cloud_auth::credentials::Credentials;
use google_cloud_bigquery::client::Read;
use google_cloud_gax::retry_throttler::{AdaptiveThrottler, RetryThrottlerArg, SharedRetryThrottler};

use crate::{BigQueryError, Client};

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
    endpoint: Option<String>,
    user_agent: Option<String>,
    quota_project_id: Option<String>,
}

impl ServiceConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cred(mut self, cred: Credentials) -> Self {
        self.cred = Some(cred);
        self
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
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

    pub async fn build(self) -> Result<Client, BigQueryError> {
        let quota_project_id = self
            .quota_project_id
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| BigQueryError::InvalidConfig("quota_project_id is required".into()))?;

        init_crypto();

        let retry_throttler = default_retry_throttler();

        let mut builder = Read::builder()
            .with_retry_policy(crate::bigquery_read_retry::create_read_session_retry_policy())
            .with_backoff_policy(crate::bigquery_read_retry::default_backoff_policy())
            .with_retry_throttler(retry_throttler.clone());

        if let Some(cred) = self.cred {
            builder = builder.with_credentials(cred);
        }

        if let Some(endpoint) = self.endpoint {
            builder = builder.with_endpoint(endpoint);
        }

        let client = builder.build().await?;

        Ok(Client::from_parts(
            client,
            quota_project_id,
            self.user_agent,
            retry_throttler,
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
        assert!(builder.endpoint.is_none());
        assert!(builder.user_agent.is_none());
        assert!(builder.quota_project_id.is_none());
    }

    #[test]
    fn test_service_config_builder_custom() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let builder = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_endpoint("https://custom.endpoint.com")
            .with_user_agent(Some("custom-agent/1.0".to_string()))
            .with_quota_project_id(Some("custom-project".to_string()));

        assert!(builder.cred.is_some());
        assert_eq!(builder.endpoint.as_deref(), Some("https://custom.endpoint.com"));
        assert_eq!(builder.user_agent.as_deref(), Some("custom-agent/1.0"));
        assert_eq!(builder.quota_project_id.as_deref(), Some("custom-project"));
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
