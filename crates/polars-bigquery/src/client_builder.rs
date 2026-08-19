// Client configuration, common to all Google services, inspired by
// https://github.com/googleapis/google-cloud-rust/blob/b1fab5ff85e2f7d139fb1b1c608cebb4ac91c5ab/src/gax/src/client_builder.rs#L539-L565
// and
// https://github.com/googleapis/google-cloud-rust/blob/c2febbabe8f6db5f271aae4dc468b8d95198ce46/src/gax/src/options.rs#L35-L56

struct ServiceConfigBuilder {
    cred: gcloud_sdk::TokenSourceType,
    cred_scopes: Vec<String>,
    endpoint: String,
    user_agent: Option<String>,
}

impl ServiceConfigBuilder {
    pub fn with_cred(mut self, cred: gcloud_sdk::TokenSourceType) -> Self {
        self.cred = Some(cred);
        self
    }

    pub fn with_cred_scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = scopes;
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

trait BigQueryReadServiceConfigBuilder {
    pub fn new() -> ServiceConfigBuilder;
}

impl BigQueryReadClientBuilder {
    pub fn new() -> ServiceConfigBuilder {
        ServiceConfigBuilder {
            cred: gcloud_sdk::TokenSourceType::Default,
            cred_scopes: vec!["https://www.googleapis.com/auth/cloud-platform".to_owned()],
            endpoint: "https://bigquerystorage.googleapis.com".to_owned(),
            user_agent: None,
        }
    }
}

