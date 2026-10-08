use gcloud_sdk::tonic;

#[derive(Debug)]
pub enum BigQueryError {
    Grpc(tonic::Status),
    Arrow(polars_error::PolarsError),
    Protocol(String),
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for BigQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Grpc(s) => write!(f, "gRPC transport error: {}", s),
            Self::Arrow(e) => write!(f, "Arrow decoding error: {}", e),
            Self::Protocol(msg) => write!(f, "BigQuery protocol error: {}", msg),
            Self::Other(e) => write!(f, "BigQuery client error: {}", e),
        }
    }
}

impl std::error::Error for BigQueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Grpc(e) => Some(e),
            Self::Arrow(e) => Some(e),
            Self::Protocol(_) => None,
            Self::Other(e) => Some(e.as_ref()),
        }
    }
}

impl From<tonic::Status> for BigQueryError {
    fn from(s: tonic::Status) -> Self {
        Self::Grpc(s)
    }
}

impl From<polars_error::PolarsError> for BigQueryError {
    fn from(e: polars_error::PolarsError) -> Self {
        Self::Arrow(e)
    }
}
