use std::time::Duration;

use google_cloud_auth::credentials::Credentials;
use google_cloud_bigquery::client::Read;
use google_cloud_bigquery::read::retry_policy::RetryableErrors;
use google_cloud_gax::exponential_backoff::ExponentialBackoffBuilder;
use google_cloud_gax::retry_policy::RetryPolicyExt;

static INIT_CRYPTO: std::sync::Once = std::sync::Once::new();

fn init_crypto() {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });
}

const DEFAULT_BQSTORAGE_ENDPOINT: &str = "https://bigquerystorage.googleapis.com";

pub struct ServiceConfigBuilder {
    cred: Option<Credentials>,
    endpoint: String,
    pub(crate) user_agent: Option<String>,
    pub(crate) quota_project_id: Option<String>,
}

impl ServiceConfigBuilder {
    pub fn with_cred(mut self, cred: Credentials) -> Self {
        self.cred = Some(cred);
        self
    }

    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
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

    pub fn quota_project_id(&self) -> Option<&str> {
        self.quota_project_id.as_deref()
    }
}

pub trait BigQueryReadClientBuilder {
    fn new() -> Self;
    fn build(
        self,
    ) -> impl std::future::Future<Output = Result<Read, Box<dyn std::error::Error>>> + Send;
}

impl BigQueryReadClientBuilder for ServiceConfigBuilder {
    fn new() -> Self {
        ServiceConfigBuilder {
            cred: None,
            endpoint: DEFAULT_BQSTORAGE_ENDPOINT.to_owned(),
            user_agent: None,
            quota_project_id: None,
        }
    }

    async fn build(self) -> Result<Read, Box<dyn std::error::Error>> {
        init_crypto();

        let retry_policy = RetryableErrors.with_time_limit(Duration::from_secs(600));
        let backoff_policy = ExponentialBackoffBuilder::new()
            .with_initial_delay(Duration::from_millis(100))
            .with_maximum_delay(Duration::from_secs(60))
            .with_scaling(1.3)
            .build()
            .expect("hardcoded value guaranteed to be valid");

        let mut builder = Read::builder()
            .with_endpoint(self.endpoint)
            .with_retry_policy(retry_policy)
            .with_backoff_policy(backoff_policy);

        if let Some(cred) = self.cred {
            builder = builder.with_credentials(cred);
        }

        let client = builder.build().await?;

        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_config_builder_defaults() {
        let builder = ServiceConfigBuilder::new();
        assert_eq!(builder.endpoint, DEFAULT_BQSTORAGE_ENDPOINT);
        assert!(builder.cred.is_none());
        assert!(builder.user_agent.is_none());
        assert!(builder.quota_project_id.is_none());
    }

    #[test]
    fn test_service_config_builder_custom() {
        let cred = google_cloud_auth::credentials::anonymous::Builder::new().build();
        let builder = ServiceConfigBuilder::new()
            .with_cred(cred)
            .with_endpoint("https://custom.endpoint.com".to_string())
            .with_user_agent(Some("custom-agent/1.0".to_string()))
            .with_quota_project_id(Some("custom-project".to_string()));

        assert_eq!(builder.endpoint, "https://custom.endpoint.com");
        assert!(builder.cred.is_some());
        assert_eq!(builder.user_agent, Some("custom-agent/1.0".to_string()));
        assert_eq!(builder.quota_project_id, Some("custom-project".to_string()));
    }
}
