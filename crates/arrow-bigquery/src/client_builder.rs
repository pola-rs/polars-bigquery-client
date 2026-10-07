use std::sync::Arc;

use google_cloud_auth::credentials::Credentials;
use google_cloud_bigquery::client::Read;
use google_cloud_gax::backoff_policy::{BackoffPolicy, BackoffPolicyArg};
use google_cloud_gax::retry_policy::{RetryPolicy, RetryPolicyArg};
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

pub(crate) fn default_grpc_subchannel_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(1)
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
    grpc_subchannel_count: Option<usize>,
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

    /// Sets the number of independent gRPC channels (HTTP/2 connections) pooled by the client.
    ///
    /// Defaults to `available_parallelism().clamp(1, 8)`. Values of `0` are clamped to `1`.
    pub fn with_grpc_subchannel_count(mut self, count: usize) -> Self {
        self.grpc_subchannel_count = Some(count.max(1));
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

    pub fn grpc_subchannel_count(&self) -> Option<usize> {
        self.grpc_subchannel_count
    }

    pub async fn build(self) -> Result<Client, BigQueryError> {
        let quota_project_id = self
            .quota_project_id
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| BigQueryError::InvalidConfig("quota_project_id is required".into()))?;

        init_crypto();

        let subchannel_count = self
            .grpc_subchannel_count
            .unwrap_or_else(default_grpc_subchannel_count)
            .max(1);

        // Resolve credentials and the shared retry throttler once so all pooled gRPC channels
        // and stream readers share a single token cache and adaptive retry budget.
        let cred = match self.cred {
            Some(cred) => cred,
            None => google_cloud_auth::credentials::Builder::default()
                .build()
                .map_err(google_cloud_gax::client_builder::Error::cred)?,
        };

        let retry_throttler = self
            .retry_throttler
            .unwrap_or_else(default_retry_throttler);

        let mut clients = Vec::with_capacity(subchannel_count);
        for _ in 0..subchannel_count {
            let mut builder = Read::builder()
                .with_credentials(cred.clone())
                .with_retry_throttler(retry_throttler.clone());

            if let Some(ref endpoint) = self.endpoint {
                builder = builder.with_endpoint(endpoint.clone());
            }

            if let Some(ref retry_policy) = self.retry_policy {
                builder = builder.with_retry_policy(retry_policy.clone());
            }

            if let Some(ref backoff_policy) = self.backoff_policy {
                builder = builder.with_backoff_policy(backoff_policy.clone());
            }

            clients.push(builder.build().await?);
        }

        Ok(Client::from_parts(
            clients,
            quota_project_id,
            self.user_agent,
            self.retry_policy,
            self.backoff_policy,
            retry_throttler,
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
        assert!(builder.grpc_subchannel_count.is_none());
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
            .with_retry_throttler(AdaptiveThrottler::default())
            .with_grpc_subchannel_count(4);

        assert_eq!(builder.endpoint(), Some("https://custom.endpoint.com"));
        assert!(builder.cred.is_some());
        assert_eq!(builder.user_agent(), Some("custom-agent/1.0"));
        assert_eq!(builder.quota_project_id(), Some("custom-project"));
        assert!(builder.retry_policy.is_some());
        assert!(builder.backoff_policy.is_some());
        assert!(builder.retry_throttler.is_some());
        assert_eq!(builder.grpc_subchannel_count(), Some(4));

        let zero_clamped = ServiceConfigBuilder::new().with_grpc_subchannel_count(0);
        assert_eq!(zero_clamped.grpc_subchannel_count(), Some(1));
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

    #[tokio::test]
    async fn test_service_config_builder_builds_subchannel_pool() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let client = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_quota_project_id(Some("test-project".to_string()))
            .with_grpc_subchannel_count(3)
            .build()
            .await
            .expect("client should build");

        assert_eq!(client.grpc_subchannel_count(), 3);
    }
}
