use std::sync::Arc;

use google_cloud_auth::credentials::Credentials;
use google_cloud_bigquery::client::Read;
use google_cloud_gax::backoff_policy::{BackoffPolicy, BackoffPolicyArg};
use google_cloud_gax::retry_policy::{RetryPolicy, RetryPolicyArg};
use google_cloud_gax::retry_throttler::{RetryThrottlerArg, SharedRetryThrottler};

use crate::{BigQueryError, Client};

static INIT_CRYPTO: std::sync::Once = std::sync::Once::new();

fn init_crypto() {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });
}

#[derive(Default)]
pub struct ServiceConfigBuilder {
    cred: Option<Credentials>,
    endpoint: Option<String>,
    user_agent: Option<String>,
    quota_project_id: Option<String>,
    retry_policy: Option<Arc<dyn RetryPolicy>>,
    backoff_policy: Option<Arc<dyn BackoffPolicy>>,
    retry_throttler: Option<SharedRetryThrottler>,
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

    pub fn with_retry_policy<V: Into<RetryPolicyArg>>(mut self, retry_policy: V) -> Self {
        self.retry_policy = Some(retry_policy.into().into());
        self
    }

    pub fn with_backoff_policy<V: Into<BackoffPolicyArg>>(mut self, backoff_policy: V) -> Self {
        self.backoff_policy = Some(backoff_policy.into().into());
        self
    }

    pub fn with_retry_throttler<V: Into<RetryThrottlerArg>>(mut self, retry_throttler: V) -> Self {
        self.retry_throttler = Some(retry_throttler.into().into());
        self
    }

    pub fn quota_project_id(&self) -> Option<&str> {
        self.quota_project_id.as_deref()
    }

    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    pub fn user_agent(&self) -> Option<&str> {
        self.user_agent.as_deref()
    }

    pub async fn build(self) -> Result<Client, BigQueryError> {
        let quota_project_id = self
            .quota_project_id
            .ok_or_else(|| BigQueryError::InvalidConfig("quota_project_id is required".into()))?;

        init_crypto();

        let mut builder = Read::builder();

        if let Some(endpoint) = self.endpoint {
            builder = builder.with_endpoint(endpoint);
        }

        if let Some(cred) = self.cred {
            builder = builder.with_credentials(cred);
        }

        if let Some(ref retry_policy) = self.retry_policy {
            builder = builder.with_retry_policy(retry_policy.clone());
        }

        if let Some(ref backoff_policy) = self.backoff_policy {
            builder = builder.with_backoff_policy(backoff_policy.clone());
        }

        if let Some(ref retry_throttler) = self.retry_throttler {
            builder = builder.with_retry_throttler(retry_throttler.clone());
        }

        let client = builder.build().await?;

        Ok(Client::from_parts(
            client,
            quota_project_id,
            self.user_agent,
            self.retry_policy,
            self.backoff_policy,
            self.retry_throttler,
        ))
    }
}

#[cfg(test)]
mod tests {
    use google_cloud_bigquery::read::retry_policy::RetryableErrors;
    use google_cloud_gax::exponential_backoff::ExponentialBackoff;
    use google_cloud_gax::retry_policy::RetryPolicyExt;
    use google_cloud_gax::retry_throttler::AdaptiveThrottler;

    use super::*;

    #[test]
    fn test_service_config_builder_defaults() {
        let builder = ServiceConfigBuilder::new();
        assert!(builder.endpoint.is_none());
        assert!(builder.cred.is_none());
        assert!(builder.user_agent.is_none());
        assert!(builder.quota_project_id.is_none());
        assert!(builder.retry_policy.is_none());
        assert!(builder.backoff_policy.is_none());
        assert!(builder.retry_throttler.is_none());
    }

    #[test]
    fn test_service_config_builder_custom() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let builder = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_endpoint("https://custom.endpoint.com")
            .with_user_agent(Some("custom-agent/1.0".to_string()))
            .with_quota_project_id(Some("custom-project".to_string()))
            .with_retry_policy(RetryableErrors.with_attempt_limit(3))
            .with_backoff_policy(ExponentialBackoff::default())
            .with_retry_throttler(AdaptiveThrottler::default());

        assert_eq!(builder.endpoint(), Some("https://custom.endpoint.com"));
        assert!(builder.cred.is_some());
        assert_eq!(builder.user_agent(), Some("custom-agent/1.0"));
        assert_eq!(builder.quota_project_id(), Some("custom-project"));
        assert!(builder.retry_policy.is_some());
        assert!(builder.backoff_policy.is_some());
        assert!(builder.retry_throttler.is_some());
    }

    #[tokio::test]
    async fn test_service_config_builder_missing_quota_project_id() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let result = ServiceConfigBuilder::new().with_cred(cred).build().await;
        match result {
            Err(BigQueryError::InvalidConfig(msg)) => {
                assert!(msg.contains("quota_project_id is required"));
            },
            other => panic!("expected InvalidConfig error, got {:?}", other.err()),
        }
    }
}
