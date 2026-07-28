use core::time::Duration;

use backon::ExponentialBuilder;
use gcloud_sdk::tonic;

/// Retry configuration for create_read_session.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L148-L162
pub const CREATE_READ_SESSION_RETRY: ExponentialBuilder = ExponentialBuilder::new()
    .with_min_delay(Duration::from_millis(100))
    .with_max_delay(Duration::from_secs(60))
    .with_factor(1.3)
    .with_total_delay(Some(Duration::from_secs(600)))
    .with_jitter();

fn is_retryable_status(err: &tonic::Status) -> bool {
    matches!(
        err.code(),
        tonic::Code::DeadlineExceeded
            | tonic::Code::Aborted
            | tonic::Code::ResourceExhausted
            // Unavailable includes common transport-level failures such as,
            // - Hyper/h2 connection errors, as seen in
            //   https://github.com/grpc/grpc-rust/pull/629
            // - Connection errors, like ConnectionReset, as seen in
            //   https://github.com/grpc/grpc-rust/blob/230544e31ef9b513e493c273d4076e843478e934/tonic/src/status.rs#L757-L761
            | tonic::Code::Unavailable
            // Internal includes common transport-level failures such as,
            // - broken pipe, as seen in
            //   https://github.com/grpc/grpc-rust/blob/230544e31ef9b513e493c273d4076e843478e934/tonic/src/status.rs#L753-L756
            | tonic::Code::Internal
    )
}

/// When to retry create_read_session requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L154-L157
pub fn create_read_session_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

/// Retry configuration for read_rows.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L163-L176
pub const READ_ROWS_RETRY: ExponentialBuilder = ExponentialBuilder::new()
    .with_min_delay(Duration::from_millis(100))
    .with_max_delay(Duration::from_secs(60))
    .with_factor(1.3)
    .with_total_delay(Some(Duration::from_secs(900)))
    .with_jitter();

/// Retry configuration for mid-stream read_rows reconnections.
pub const STREAM_RECONNECT_RETRY: ExponentialBuilder = ExponentialBuilder::new()
    .with_min_delay(Duration::from_millis(100))
    .with_max_delay(Duration::from_secs(60))
    .with_factor(1.3)
    .with_max_times(10)
    .with_jitter();


/// When to retry read_rows requests.
///
/// Inspired by the Python configuration at
/// https://github.com/googleapis/google-cloud-python/blob/c43caeee34e7c0878766d2806f69016c319697e2/packages/google-cloud-bigquery-storage/google/cloud/bigquery_storage_v1/services/big_query_read/transports/base.py#L169-L171
pub fn read_rows_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

/// When to reconnect/resume an active read_rows stream after encountering a gRPC error mid-read.
///
/// While currently identical to [`read_rows_predicate`], having a dedicated predicate allows fine-tuning
/// reconnection behavior separately from initial request establishment.
pub fn reconnect_stream_predicate(err: &tonic::Status) -> bool {
    is_retryable_status(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gcloud_sdk::tonic::{Code, Status};

    #[test]
    fn test_reconnect_stream_predicate_retryable_codes() {
        let retryable_codes = [
            Code::DeadlineExceeded,
            Code::Unavailable,
            Code::Aborted,
            Code::Internal,
            Code::ResourceExhausted,
        ];

        for code in retryable_codes {
            let status = Status::new(code, "transient stream error");
            assert!(
                reconnect_stream_predicate(&status),
                "Expected reconnect_stream_predicate to be true for code {:?}",
                code
            );
        }
    }

    #[test]
    fn test_reconnect_stream_predicate_non_retryable_codes() {
        let non_retryable_codes = [
            Code::Ok,
            Code::Cancelled,
            Code::Unknown,
            Code::InvalidArgument,
            Code::NotFound,
            Code::AlreadyExists,
            Code::PermissionDenied,
            Code::FailedPrecondition,
            Code::OutOfRange,
            Code::Unimplemented,
            Code::DataLoss,
            Code::Unauthenticated,
        ];

        for code in non_retryable_codes {
            let status = Status::new(code, "fatal stream error");
            assert!(
                !reconnect_stream_predicate(&status),
                "Expected reconnect_stream_predicate to be false for code {:?}",
                code
            );
        }
    }

    #[test]
    fn test_read_rows_predicate_matches_reconnect() {
        // Ensure initial request predicate and stream reconnection predicate currently match on behavior
        let codes = [Code::Unavailable, Code::Aborted, Code::NotFound, Code::InvalidArgument];
        for code in codes {
            let status = Status::new(code, "test");
            assert_eq!(
                read_rows_predicate(&status),
                reconnect_stream_predicate(&status)
            );
        }
    }
}
