// Client configuration, common to all Google services, inspired by
// https://github.com/googleapis/google-cloud-rust/blob/b1fab5ff85e2f7d139fb1b1c608cebb4ac91c5ab/src/gax/src/client_builder.rs#L539-L565
// and
// https://github.com/googleapis/google-cloud-rust/blob/c2febbabe8f6db5f271aae4dc468b8d95198ce46/src/gax/src/options.rs#L35-L56

use gcloud_sdk::google::cloud::bigquery::storage::v1::big_query_read_client::BigQueryReadClient;
use gcloud_sdk::tonic::async_trait;
use gcloud_sdk::{GoogleApiClient, GoogleApiClientBuilder, GoogleAuthMiddleware, TokenSourceType};
use hyper::header::{HeaderValue, USER_AGENT};
use hyper::HeaderMap;

static INIT_CRYPTO: std::sync::Once = std::sync::Once::new();

fn init_crypto() {
    INIT_CRYPTO.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        // ignore if another crate already set the default provider.
    });
}

const DEFAULT_BQSTORAGE_ENDPOINT: &str = "https://bigquerystorage.googleapis.com";
const DEFAULT_GCP_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
pub struct ServiceConfigBuilder {
    cred: TokenSourceType,
    cred_scopes: Vec<String>,
    endpoint: String,
    user_agent: Option<String>,
}

impl ServiceConfigBuilder {
    pub fn with_cred(mut self, cred: TokenSourceType) -> Self {
        self.cred = cred;
        self
    }

    pub fn with_cred_scopes(mut self, scopes: Vec<String>) -> Self {
        self.cred_scopes = scopes;
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
}

#[async_trait]
pub trait BigQueryReadClientBuilder {
    fn new() -> Self;
    async fn build(
        self,
    ) -> Result<
        GoogleApiClient<BQStorageGoogleApiClientBuilder, BigQueryReadClient<GoogleAuthMiddleware>>,
        Box<dyn std::error::Error>,
    >;
}

#[async_trait]
impl BigQueryReadClientBuilder for ServiceConfigBuilder {
    fn new() -> Self {
        ServiceConfigBuilder {
            cred: TokenSourceType::Default,
            cred_scopes: vec![DEFAULT_GCP_SCOPE.to_owned()],
            endpoint: DEFAULT_BQSTORAGE_ENDPOINT.to_owned(),
            user_agent: None,
        }
    }

    async fn build(
        self,
    ) -> Result<
        GoogleApiClient<BQStorageGoogleApiClientBuilder, BigQueryReadClient<GoogleAuthMiddleware>>,
        Box<dyn std::error::Error>,
    > {
        init_crypto();
        let builder = BQStorageGoogleApiClientBuilder {};

        let mut headers = HeaderMap::new();
        if let Some(user_agent) = self.user_agent {
            headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);
        }

        let client = GoogleApiClient::with_token_source_and_headers(
            builder,
            self.endpoint,
            None, // cloud_resource_prefix
            self.cred,
            self.cred_scopes,
            headers,
        )
        .await?;

        Ok(client)
    }
}

#[derive(Clone, Debug)]
pub struct BQStorageGoogleApiClientBuilder;

#[async_trait]
impl GoogleApiClientBuilder<BigQueryReadClient<GoogleAuthMiddleware>>
    for BQStorageGoogleApiClientBuilder
{
    fn create_client(
        &self,
        channel: GoogleAuthMiddleware,
    ) -> BigQueryReadClient<GoogleAuthMiddleware> {
        BigQueryReadClient::new(channel).max_decoding_message_size(
            128 * 1024 * 1024, // 128MB, as recommended by the service team
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_config_builder_defaults() {
        let builder = ServiceConfigBuilder::new();
        assert_eq!(builder.endpoint, DEFAULT_BQSTORAGE_ENDPOINT);
        assert_eq!(
            builder.cred_scopes,
            vec!["https://www.googleapis.com/auth/cloud-platform"]
        );
        assert!(builder.user_agent.is_none());
    }

    #[test]
    fn test_service_config_builder_custom() {
        let builder = ServiceConfigBuilder::new()
            .with_endpoint("https://custom.endpoint.com".to_string())
            .with_cred_scopes(vec!["scope1".to_string(), "scope2".to_string()])
            .with_user_agent(Some("custom-agent/1.0".to_string()));

        assert_eq!(builder.endpoint, "https://custom.endpoint.com");
        assert_eq!(builder.cred_scopes, vec!["scope1", "scope2"]);
        assert_eq!(builder.user_agent, Some("custom-agent/1.0".to_string()));
    }
}
