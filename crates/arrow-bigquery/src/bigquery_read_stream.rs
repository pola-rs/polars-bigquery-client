//! ## bigquery_read_stream.rs
//!
//! This module reads from a BigQuery Storage Read API stream and writes the
//! results as Arrow record batches to a tokio mpsc queue as messages arrive.
//!
//! To robustly handle disruptions, this module uses the following state
//! machine for connection and mid-stream reconnection logic:
//!
//! ```text
//!                Start here
//!                    │
//!              ┌─────▼──────┐      Retryable       ┌─────────┐
//!        ┌─────┼ Connecting ┼─────────────────────►│ Backoff │
//!        │     └─────┬──────┘◄─────────────────────┼         │
//!        │           │          Deadline elapsed   └────▲────┘
//!   Exhausted        │                                  │
//!    Retries         │                             Retryable
//!       or           │                               Error
//! Unrecoverable      │                  (retains retry_state if Some, or
//!      Error   ┌─────▼──────┐            starts fresh RetryState if None)
//!        │     │  Reading   ┼───────────────────────────┘
//!        │     └─────┬──────┘ (clears retry_state
//!        │           │         on first message)
//!        │           ▼
//!        │      Unrecoverable
//!        │      Error or EOF
//!        │           │            ┌────────────┐
//!        │           └───────────►│            │
//!        └───────────────────────►│ Terminated │
//!                                 │            │
//!                                 └────────────┘
//! ```

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use google_cloud_bigquery::builder::read::ReadRows;
use google_cloud_bigquery::model::{read_rows_response, ReadRowsResponse};
use google_cloud_gax::retry_result::RetryResult;
use google_cloud_gax::retry_state::RetryState;
use google_cloud_gax::streaming::ResponseStream;
use google_cloud_gax::throttle_result::ThrottleResult;
use tokio::time::Instant;

use super::bigquery_read_retry::StreamRetryConfig;
use crate::decode::{parse_arrow_schema, validate_non_empty_arrow_bytes, DecodeJob, DecodeQueue};
use crate::BigQueryError;

pub(crate) type SchemaSender =
    tokio::sync::oneshot::Sender<Result<Option<SchemaRef>, BigQueryError>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminationReason {
    /// Stream completed cleanly, fatal error occurred, or consumer dropped the receiver.
    Finished,
    /// Cancelled after the first response to read in parallel from a fresh read session.
    SwitchToReadSession,
}

pub(crate) struct FirstPageCheck {
    pub(crate) threshold_bytes: f64,
    pub(crate) fallback_total_rows: Option<u64>,
}

/// Estimates the remaining serialized Arrow bytes in the single job stream based on
/// the first page's byte size, the first page's row count, and the total row count.
pub(crate) fn estimate_remaining_bytes(
    first_page_bytes: &bytes::Bytes,
    page_row_count: u64,
    total_rows: Option<u64>,
) -> f64 {
    let Some(total_rows) = total_rows else {
        return 0.0;
    };
    if page_row_count == 0 || total_rows <= page_row_count {
        return 0.0;
    }
    let remaining_rows = (total_rows - page_row_count) as f64;
    (first_page_bytes.len() as f64) * remaining_rows / (page_row_count as f64)
}

/// Extracts the serialized Arrow record batch bytes and expected row count from
/// a [`ReadRowsResponse`].
fn extract_response_arrow_bytes(
    response: ReadRowsResponse,
) -> Result<Option<(bytes::Bytes, i64)>, BigQueryError> {
    let row_count = response.row_count;
    let raw_bytes = match response.rows {
        Some(read_rows_response::Rows::ArrowRecordBatch(value)) => {
            Some(value.serialized_record_batch)
        },
        None => None,
        Some(_) => {
            return Err(BigQueryError::Protocol(
                "Unexpectedly got some format other than arrow bytes".into(),
            ))
        },
    };

    Ok(validate_non_empty_arrow_bytes(raw_bytes, row_count)?.map(|bytes| (bytes, row_count)))
}

#[cfg(test)]
fn read_rows_response_to_record_batch(
    response: ReadRowsResponse,
    schema: &[u8],
) -> Result<Option<(RecordBatch, i64)>, BigQueryError> {
    let Some((serialized_record_batch, row_count)) = extract_response_arrow_bytes(response)? else {
        return Ok(None);
    };
    let parsed_schema = parse_arrow_schema(schema)?;
    let mut batches = crate::decode::decode_arrow_record_batches(
        serialized_record_batch,
        row_count,
        &parsed_schema,
    )?;
    if batches.is_empty() {
        Ok(None)
    } else {
        let batch = batches.remove(0);
        let actual_rows = batch.num_rows() as i64;
        Ok(Some((batch, actual_rows)))
    }
}

/// State machine for reconnection logic. If reading fails at some point,
/// attempt to reconnect with the current row offset.
///
/// All mutable runtime state for stream reconnection (such as the current row
/// offset, active stream, and retry progress) is stored in [`ReadStreamState`],
/// keeping state transitions explicit and self-contained inside [`ReadStream::step`].
#[derive(Debug)]
pub(crate) enum ReadStreamState {
    /// Connect or reconnect to the BigQuery read stream.
    Connecting {
        offset: i64,
        retry_state: RetryState,
    },
    /// Waiting until `deadline` before reconnecting to the BigQuery read stream.
    Backoff {
        offset: i64,
        retry_state: RetryState,
        deadline: Instant,
    },
    /// Reading messages from an open BigQuery read stream.
    ///
    /// `retry_state` is `Some` while waiting for the first message after connecting
    /// (so failures before any data progress retain the retry count from `Connecting`),
    /// and is cleared to `None` once the first message is successfully received.
    Reading {
        offset: i64,
        retry_state: Option<RetryState>,
        stream: ResponseStream<ReadRowsResponse>,
    },
    /// Stream completed cleanly, fatal error occurred, consumer dropped the receiver,
    /// or cancelled after the first response to read in parallel from a fresh read session.
    Terminated(TerminationReason),
}

async fn enqueue_decode_batch(
    decode_queue: &DecodeQueue,
    serialized_record_batch: bytes::Bytes,
    expected_row_count: i64,
    schema: SchemaRef,
    tx: &tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    cancel_tx: &Arc<tokio::sync::watch::Sender<bool>>,
    cancel_rx: &mut tokio::sync::watch::Receiver<bool>,
) -> bool {
    let permit = tokio::select! {
        biased;
        _ = cancel_rx.changed() => return false,
        res = tx.clone().reserve_owned() => match res {
            Ok(permit) => permit,
            Err(_) => {
                // `reserve_owned` returns `Err` strictly when all `Receiver` handles (`rx`) have been
                // dropped or closed. Terminating this stream cleanly prevents orphan background tasks.
                return false;
            }
        }
    };

    decode_queue.push(DecodeJob {
        serialized_record_batch,
        expected_row_count,
        schema,
        permit,
        cancel_tx: Arc::clone(cancel_tx),
        cancel_rx: cancel_rx.clone(),
    });

    true
}

pub(crate) struct ReadStreamOptions {
    pub(crate) read_rows: ReadRows,
    pub(crate) current_offset: i64,
    pub(crate) schema: Option<SchemaRef>,
    pub(crate) schema_tx: Option<SchemaSender>,
    pub(crate) first_page_check: Option<FirstPageCheck>,
    pub(crate) tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    pub(crate) retry_config: StreamRetryConfig,
    pub(crate) maintain_order: bool,
    pub(crate) cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
    pub(crate) cancel_rx: tokio::sync::watch::Receiver<bool>,
}

pub(crate) struct ReadQueryStreamOptions {
    pub(crate) first_page_bytes: bytes::Bytes,
    pub(crate) page_row_count: i64,
    pub(crate) read_rows: Option<ReadRows>,
    pub(crate) schema: SchemaRef,
    pub(crate) tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    pub(crate) retry_config: StreamRetryConfig,
    pub(crate) maintain_order: bool,
    pub(crate) cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
    pub(crate) cancel_rx: tokio::sync::watch::Receiver<bool>,
}

pub(crate) struct ReadStream {
    read_rows: ReadRows,
    schema: Option<SchemaRef>,
    schema_tx: Option<SchemaSender>,
    first_page_check: Option<FirstPageCheck>,
    tx: tokio::sync::mpsc::Sender<Result<RecordBatch, BigQueryError>>,
    retry_config: StreamRetryConfig,
    decode_queue: DecodeQueue,
    cancel_tx: Arc<tokio::sync::watch::Sender<bool>>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    /// Encapsulates mutable connection and retry state for the stream reader.
    pub(crate) state: ReadStreamState,
}

impl ReadStream {
    fn new(options: ReadStreamOptions, decode_queue: DecodeQueue) -> Self {
        let ReadStreamOptions {
            read_rows,
            current_offset,
            schema,
            schema_tx,
            first_page_check,
            tx,
            retry_config,
            maintain_order: _,
            cancel_tx,
            cancel_rx,
        } = options;
        Self {
            read_rows,
            schema,
            schema_tx,
            first_page_check,
            tx,
            retry_config,
            decode_queue,
            cancel_tx,
            cancel_rx,
            state: ReadStreamState::Connecting {
                offset: current_offset,
                retry_state: RetryState::new(true),
            },
        }
    }

    fn record_success(&self) {
        self.retry_config
            .retry_throttler
            .lock()
            .expect("retry throttler lock is poisoned")
            .on_success();
    }

    async fn send_fatal_error(&mut self, err: BigQueryError) {
        let _ = self.cancel_tx.send_replace(true);
        if let Some(schema_tx) = self.schema_tx.take() {
            let _ = schema_tx.send(Err(err));
        } else {
            let _ = self.tx.send(Err(err)).await;
        }
    }

    async fn handle_error(
        &mut self,
        offset: i64,
        mut retry_state: RetryState,
        err: google_cloud_bigquery::Error,
    ) -> ReadStreamState {
        retry_state.attempt_count += 1;
        let flow = self.retry_config.retry_policy.on_error(&retry_state, err);
        self.retry_config
            .retry_throttler
            .lock()
            .expect("retry throttler lock is poisoned")
            .on_retry_failure(&flow);
        let prev_err = match flow {
            RetryResult::Permanent(e) | RetryResult::Exhausted(e) => {
                self.send_fatal_error(BigQueryError::Grpc(e)).await;
                return ReadStreamState::Terminated(TerminationReason::Finished);
            },
            RetryResult::Continue(e) => e,
        };

        let throttled = self
            .retry_config
            .retry_throttler
            .lock()
            .expect("retry throttler lock is poisoned")
            .throttle_retry_attempt();
        let prev_err = if throttled {
            retry_state.attempt_count += 1;
            match self
                .retry_config
                .retry_policy
                .on_throttle(&retry_state, prev_err)
            {
                ThrottleResult::Exhausted(e) => {
                    self.send_fatal_error(BigQueryError::Grpc(e)).await;
                    return ReadStreamState::Terminated(TerminationReason::Finished);
                },
                ThrottleResult::Continue(e) => e,
            }
        } else {
            prev_err
        };

        let delay = self.retry_config.backoff_policy.on_failure(&retry_state);
        if self
            .retry_config
            .retry_policy
            .remaining_time(&retry_state)
            .is_some_and(|remaining| remaining <= delay)
        {
            self.send_fatal_error(BigQueryError::Grpc(
                google_cloud_bigquery::Error::exhausted(prev_err),
            ))
            .await;
            return ReadStreamState::Terminated(TerminationReason::Finished);
        }
        ReadStreamState::Backoff {
            offset,
            retry_state,
            deadline: Instant::now() + delay,
        }
    }

    /// Advances the state machine by a single transition.
    pub(crate) async fn step(&mut self) {
        if !matches!(self.state, ReadStreamState::Terminated(_))
            && *self.cancel_rx.borrow_and_update()
        {
            self.state = ReadStreamState::Terminated(TerminationReason::Finished);
            return;
        }

        match &mut self.state {
            ReadStreamState::Connecting {
                offset,
                retry_state,
            } => {
                let offset = *offset;
                let retry_state = retry_state.clone();
                let req = self.read_rows.clone().set_offset(offset);
                let res = tokio::select! {
                    biased;
                    _ = self.cancel_rx.changed() => {
                        self.state = ReadStreamState::Terminated(TerminationReason::Finished);
                        return;
                    }
                    res = req.send() => res,
                };
                match res {
                    Ok(stream) => {
                        self.state = ReadStreamState::Reading {
                            offset,
                            retry_state: Some(retry_state),
                            stream,
                        };
                    },
                    Err(err) => {
                        self.state = self.handle_error(offset, retry_state, err).await;
                    },
                }
            },
            ReadStreamState::Backoff {
                offset,
                retry_state,
                deadline,
            } => {
                let offset = *offset;
                let retry_state = retry_state.clone();
                let deadline = *deadline;
                tokio::select! {
                    biased;
                    _ = self.cancel_rx.changed() => {
                        self.state = ReadStreamState::Terminated(TerminationReason::Finished);
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        self.state = ReadStreamState::Connecting {
                            offset,
                            retry_state,
                        };
                    }
                }
            },
            ReadStreamState::Reading {
                offset,
                retry_state,
                stream,
            } => {
                let next_item = tokio::select! {
                    biased;
                    _ = self.cancel_rx.changed() => {
                        self.state = ReadStreamState::Terminated(TerminationReason::Finished);
                        return;
                    }
                    item = stream.next() => item,
                };
                match next_item {
                    Some(Ok(mut value)) => {
                        let schema_ref = match self.schema {
                            Some(ref s) => Arc::clone(s),
                            None => {
                                let serialized_schema = match value.schema.take() {
                                    Some(read_rows_response::Schema::ArrowSchema(arrow_schema)) => {
                                        arrow_schema.serialized_schema
                                    },
                                    None if value.rows.is_none() && value.row_count == 0 => {
                                        // Empty heartbeat response sent by BigQuery while waiting
                                        // for a running query job to produce its schema and rows.
                                        *retry_state = None;
                                        self.record_success();
                                        return;
                                    },
                                    _ => {
                                        self.send_fatal_error(BigQueryError::Protocol(
                                            "Unexpectedly got schema type other than arrow".into(),
                                        ))
                                        .await;
                                        self.state = ReadStreamState::Terminated(
                                            TerminationReason::Finished,
                                        );
                                        return;
                                    },
                                };
                                let parsed = match parse_arrow_schema(&serialized_schema) {
                                    Ok(parsed) => parsed,
                                    Err(err) => {
                                        self.send_fatal_error(err).await;
                                        self.state = ReadStreamState::Terminated(
                                            TerminationReason::Finished,
                                        );
                                        return;
                                    },
                                };
                                if let Some(sender) = self.schema_tx.take() {
                                    let _ = sender.send(Ok(Some(Arc::clone(&parsed))));
                                }
                                self.schema = Some(Arc::clone(&parsed));
                                parsed
                            },
                        };

                        let total_estimated_row_count = value.total_estimated_row_count;
                        let (serialized_record_batch, row_count) =
                            match extract_response_arrow_bytes(value) {
                                Ok(Some(pair)) => pair,
                                Ok(None) => {
                                    *retry_state = None;
                                    self.record_success();
                                    return;
                                },
                                Err(err) => {
                                    self.send_fatal_error(err).await;
                                    self.state =
                                        ReadStreamState::Terminated(TerminationReason::Finished);
                                    return;
                                },
                            };

                        if let Some(check) = self.first_page_check.take() {
                            let page_row_count = match u64::try_from(row_count) {
                                Ok(n) => n,
                                Err(_) => {
                                    self.send_fatal_error(BigQueryError::Protocol(format!(
                                        "Negative row_count in ReadRowsResponse: {row_count}"
                                    )))
                                    .await;
                                    self.state =
                                        ReadStreamState::Terminated(TerminationReason::Finished);
                                    return;
                                },
                            };
                            let total_rows = total_estimated_row_count
                                .and_then(|n| u64::try_from(n).ok())
                                .filter(|&n| n > 0)
                                .or_else(|| check.fallback_total_rows.filter(|&n| n > 0));

                            if estimate_remaining_bytes(
                                &serialized_record_batch,
                                page_row_count,
                                total_rows,
                            ) >= check.threshold_bytes
                            {
                                self.state = ReadStreamState::Terminated(
                                    TerminationReason::SwitchToReadSession,
                                );
                                return;
                            }
                        }

                        let next_offset = match offset.checked_add(row_count) {
                            Some(next_offset) => next_offset,
                            None => {
                                let current_offset = *offset;
                                self.send_fatal_error(BigQueryError::Protocol(format!(
                                    "stream offset overflow: current_offset={current_offset}, row_count={row_count}"
                                )))
                                .await;
                                self.state =
                                    ReadStreamState::Terminated(TerminationReason::Finished);
                                return;
                            },
                        };
                        *offset = next_offset;
                        // Data progress was made: clear `retry_state` so any future
                        // mid-stream failure starts a fresh `RetryState::new(true)`.
                        *retry_state = None;
                        self.record_success();

                        if !enqueue_decode_batch(
                            &self.decode_queue,
                            serialized_record_batch,
                            row_count,
                            schema_ref,
                            &self.tx,
                            &self.cancel_tx,
                            &mut self.cancel_rx,
                        )
                        .await
                        {
                            self.state = ReadStreamState::Terminated(TerminationReason::Finished);
                        }
                    },
                    Some(Err(err)) => {
                        let offset = *offset;
                        let retry_state =
                            retry_state.take().unwrap_or_else(|| RetryState::new(true));
                        self.state = self.handle_error(offset, retry_state, err).await;
                    },
                    None => {
                        self.record_success();
                        if let Some(sender) = self.schema_tx.take() {
                            // TODO(b/573768143): If the Read API is updated to always send a schema before
                            // closing, this can go beack to being an error condition. Currently, we might
                            // encounter this case for long-running queries.
                            let _ = sender.send(Ok(None));
                        }
                        self.state = ReadStreamState::Terminated(TerminationReason::Finished);
                    },
                }
            },
            ReadStreamState::Terminated(_) => {},
        }
    }

    async fn run(&mut self) -> TerminationReason {
        loop {
            if let ReadStreamState::Terminated(reason) = self.state {
                return reason;
            }
            self.step().await;
        }
    }
}

/// Orchestrates stream connection, consumption, and mid-stream reconnections via [`ReadStream::step`].
pub(crate) async fn read_stream(options: ReadStreamOptions) -> TerminationReason {
    let decode_queue = DecodeQueue::new(options.maintain_order);
    let mut reader = ReadStream::new(options, decode_queue);
    reader.run().await
}

/// Enqueues the first page of query results for decoding and streams any remaining rows
/// from `read_rows`, overlapping first-page CPU decompression with the gRPC
/// `ReadRows` stream connection when `read_rows` is present.
pub(crate) async fn read_query_stream(options: ReadQueryStreamOptions) {
    let ReadQueryStreamOptions {
        first_page_bytes,
        page_row_count,
        read_rows,
        schema,
        tx,
        retry_config,
        maintain_order,
        cancel_tx,
        mut cancel_rx,
    } = options;
    let decode_queue = DecodeQueue::new(maintain_order);

    if !enqueue_decode_batch(
        &decode_queue,
        first_page_bytes,
        page_row_count,
        Arc::clone(&schema),
        &tx,
        &cancel_tx,
        &mut cancel_rx,
    )
    .await
    {
        return;
    }

    if let Some(read_rows) = read_rows {
        let mut reader = ReadStream::new(
            ReadStreamOptions {
                read_rows,
                current_offset: page_row_count,
                schema: Some(schema),
                schema_tx: None,
                first_page_check: None,
                tx,
                retry_config,
                maintain_order,
                cancel_tx,
                cancel_rx,
            },
            decode_queue,
        );
        let _ = reader.run().await;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use google_cloud_bigquery::client::Read;
    use google_cloud_bigquery::model::{ArrowRecordBatch, AvroRows, ReadRowsRequest};
    use google_cloud_gax::backoff_policy::BackoffPolicy;
    use google_cloud_gax::error::rpc::{Code, Status};
    use google_cloud_gax::error::Error;
    use google_cloud_gax::options::RequestOptions;
    use google_cloud_gax::retry_policy::RetryPolicyExt;
    use google_cloud_gax::retry_state::RetryState;
    use google_cloud_gax::streaming::ResponseStream;

    use super::*;
    use crate::bigquery_read_retry::RetryableErrors;

    #[derive(Debug, Default)]
    pub(crate) struct RecordedReadRows {
        pub(crate) stream_name: Option<String>,
        pub(crate) offset: Option<i64>,
        pub(crate) arrow_buffer_compression: Option<crate::CompressionCodec>,
        pub(crate) quota_project: Option<String>,
        pub(crate) user_agent: Option<String>,
    }

    #[derive(Debug)]
    pub(crate) struct CannedReadRowsStub {
        pub(crate) recorded: Arc<Mutex<RecordedReadRows>>,
        pub(crate) responses: Arc<Mutex<Vec<ReadRowsResponse>>>,
    }

    impl CannedReadRowsStub {
        pub(crate) fn new(
            responses: Vec<ReadRowsResponse>,
        ) -> (Self, Arc<Mutex<RecordedReadRows>>) {
            let recorded = Arc::new(Mutex::new(RecordedReadRows::default()));
            let stub = Self {
                recorded: Arc::clone(&recorded),
                responses: Arc::new(Mutex::new(responses)),
            };
            (stub, recorded)
        }
    }

    impl google_cloud_bigquery::stub::Read for CannedReadRowsStub {
        async fn read_rows(
            &self,
            req: ReadRowsRequest,
            options: RequestOptions,
        ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
            {
                let mut rec = self.recorded.lock().unwrap();
                rec.arrow_buffer_compression = req
                    .arrow_serialization_options()
                    .map(|opts| opts.buffer_compression.clone());
                rec.stream_name = Some(req.read_stream);
                rec.offset = Some(req.offset);
                rec.quota_project = options.quota_project().clone();
                rec.user_agent = options.user_agent().clone();
            }
            let responses = std::mem::take(&mut *self.responses.lock().unwrap());
            let (tx, rx) = tokio::sync::mpsc::channel(responses.len().max(1));
            for resp in responses {
                let _ = tx.send(Ok(resp)).await;
            }
            Ok(ResponseStream::from(rx))
        }
    }

    pub(crate) fn create_test_arrow_payload_with_range(
        start: i64,
        count: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        use arrow_array::{ArrayRef, Int64Array};
        use arrow_ipc::writer::StreamWriter;
        use arrow_schema::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![Field::new(
            "col1",
            DataType::Int64,
            false,
        )]));

        let mut schema_bytes = Vec::new();
        {
            let _writer = StreamWriter::try_new(&mut schema_bytes, &schema).unwrap();
        }
        let schema_len = schema_bytes.len();

        let array: ArrayRef = Arc::new(Int64Array::from(
            (start..(start + count as i64)).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap();

        let mut full_stream_bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut full_stream_bytes, &schema).unwrap();
            writer.write(&batch).unwrap();
        }

        let batch_bytes = full_stream_bytes[schema_len..].to_vec();
        (schema_bytes, batch_bytes)
    }

    pub(crate) fn create_test_arrow_payload(num_rows: usize) -> (Vec<u8>, ReadRowsResponse) {
        let (schema_bytes, batch_bytes) = create_test_arrow_payload_with_range(0, num_rows);
        let response = ReadRowsResponse::new()
            .set_arrow_record_batch(
                ArrowRecordBatch::new().set_serialized_record_batch(batch_bytes),
            )
            .set_row_count(num_rows as i64);

        (schema_bytes, response)
    }

    #[test]
    fn test_read_rows_response_empty() {
        let response = ReadRowsResponse::new();
        assert!(read_rows_response_to_record_batch(response, &[])
            .unwrap()
            .is_none());

        let response2 = ReadRowsResponse::new().set_arrow_record_batch(ArrowRecordBatch::new());
        assert!(read_rows_response_to_record_batch(response2, &[])
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_read_rows_response_protocol_error() {
        let response = ReadRowsResponse::new()
            .set_avro_rows(AvroRows::new())
            .set_row_count(5);
        let err = read_rows_response_to_record_batch(response, &[]).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[test]
    fn test_read_rows_response_arrow_error() {
        let response = ReadRowsResponse::new()
            .set_arrow_record_batch(
                ArrowRecordBatch::new().set_serialized_record_batch(vec![0x00, 0x01, 0x02, 0x03]),
            )
            .set_row_count(5);
        let err = read_rows_response_to_record_batch(response, &[0x00, 0x00]).unwrap_err();
        assert!(matches!(err, BigQueryError::Arrow(_)));
    }

    #[test]
    fn test_read_rows_response_success_and_mismatch() {
        let (schema_bytes, response) = create_test_arrow_payload(5);
        let (batch, rows) = read_rows_response_to_record_batch(response.clone(), &schema_bytes)
            .unwrap()
            .unwrap();
        assert_eq!(rows, 5);
        assert_eq!(batch.num_rows(), 5);

        // Test row count mismatch
        let bad_response = response.set_row_count(10); // mismatch with 5 decoded rows
        let err = read_rows_response_to_record_batch(bad_response, &schema_bytes).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[test]
    fn test_read_rows_response_row_count_mismatch_when_reader_empty() {
        let (schema_bytes, _) = create_test_arrow_payload(0);
        // Replace empty serialized_record_batch with an Arrow IPC End-Of-Stream marker (8 bytes)
        // so serialized_record_batch is non-empty, but StreamReader::next() returns None.
        let response =
            ReadRowsResponse::new()
                .set_arrow_record_batch(ArrowRecordBatch::new().set_serialized_record_batch(vec![
                    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00,
                ]))
                .set_row_count(10);
        let err = read_rows_response_to_record_batch(response, &schema_bytes).unwrap_err();
        assert!(matches!(err, BigQueryError::Protocol(_)));
    }

    #[derive(Debug, Default)]
    struct NoBackoff;

    impl BackoffPolicy for NoBackoff {
        fn on_failure(&self, _state: &RetryState) -> Duration {
            Duration::ZERO
        }
    }

    fn test_retry_config(max_attempts: u32) -> StreamRetryConfig {
        use google_cloud_gax::retry_throttler::{CircuitBreaker, RetryThrottlerArg};

        StreamRetryConfig::new(
            Some(Arc::new(RetryableErrors.with_attempt_limit(max_attempts))),
            Some(Arc::new(NoBackoff)),
            RetryThrottlerArg::from(CircuitBreaker::default()).into(),
        )
    }

    type MockStreamResult =
        google_cloud_bigquery::Result<Vec<google_cloud_bigquery::Result<ReadRowsResponse>>>;

    #[derive(Debug)]
    struct MockClient {
        requests: Arc<Mutex<Vec<ReadRowsRequest>>>,
        streams: Arc<Mutex<Vec<MockStreamResult>>>,
    }

    impl google_cloud_bigquery::stub::Read for MockClient {
        async fn read_rows(
            &self,
            req: ReadRowsRequest,
            _options: RequestOptions,
        ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
            self.requests.lock().unwrap().push(req);
            let next_stream = {
                let mut streams = self.streams.lock().unwrap();
                if !streams.is_empty() {
                    streams.remove(0)
                } else {
                    Ok(vec![])
                }
            };
            let messages = next_stream?;
            let (tx, rx) = tokio::sync::mpsc::channel(messages.len().max(1));
            for msg in messages {
                let _ = tx.send(msg).await;
            }
            Ok(ResponseStream::from(rx))
        }
    }

    #[tokio::test]
    async fn test_stream_reconnection_offset() {
        let (schema_bytes, resp1) = create_test_arrow_payload(3);
        let (_, resp2) = create_test_arrow_payload(2);
        let schema = parse_arrow_schema(&schema_bytes).unwrap();

        let stream1 = Ok(vec![
            Ok(resp1),
            Err(Error::service(
                Status::default()
                    .set_code(Code::Unavailable)
                    .set_message("transient disconnection"),
            )),
        ]);
        let stream2 = Ok(vec![Ok(resp2)]);

        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![stream1, stream2])),
        };

        let client = Read::from_stub(mock_client);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        read_stream(ReadStreamOptions {
            read_rows,
            current_offset: 10,
            schema: Some(schema),
            schema_tx: None,
            first_page_check: None,
            tx,
            retry_config: test_retry_config(3),
            maintain_order: true,
            cancel_tx: Arc::new(cancel_tx),
            cancel_rx,
        })
        .await;

        let batch1 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch1.num_rows(), 3);
        let batch2 = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch2.num_rows(), 2);
        assert!(rx.recv().await.is_none());

        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].offset, 10);
        assert_eq!(reqs[1].offset, 13);
    }

    #[tokio::test]
    async fn test_connect_read_rows_stream_retries_transient_error() {
        let (schema_bytes, resp1) = create_test_arrow_payload(4);
        let schema = parse_arrow_schema(&schema_bytes).unwrap();

        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![
                Err(Error::service(
                    Status::default()
                        .set_code(Code::Unavailable)
                        .set_message("initial connect unavailable"),
                )),
                Ok(vec![Ok(resp1)]),
            ])),
        };

        let client = Read::from_stub(mock_client);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        read_stream(ReadStreamOptions {
            read_rows,
            current_offset: 0,
            schema: Some(schema),
            schema_tx: None,
            first_page_check: None,
            tx,
            retry_config: test_retry_config(3),
            maintain_order: false,
            cancel_tx: Arc::new(cancel_tx),
            cancel_rx,
        })
        .await;

        let batch = rx.recv().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 4);
        assert!(rx.recv().await.is_none());

        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].offset, 0);
        assert_eq!(reqs[1].offset, 0);
    }

    #[tokio::test]
    async fn test_stream_unrecoverable_error_terminates_without_retry() {
        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(vec![Ok(vec![Err(Error::service(
                Status::default()
                    .set_code(Code::PermissionDenied)
                    .set_message("permission denied"),
            ))])])),
        };

        let client = Read::from_stub(mock_client);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        read_stream(ReadStreamOptions {
            read_rows,
            current_offset: 0,
            schema: None,
            schema_tx: None,
            first_page_check: None,
            tx,
            retry_config: test_retry_config(3),
            maintain_order: false,
            cancel_tx: Arc::clone(&cancel_tx),
            cancel_rx,
        })
        .await;

        let result = rx.recv().await.unwrap();
        assert!(matches!(result, Err(BigQueryError::Grpc(_))));
        assert!(
            *cancel_tx.borrow(),
            "unrecoverable stream error should signal sibling cancellation"
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_stream_reconnection_exhaustion() {
        // Stream repeatedly disconnects immediately without making progress.
        // Verifies the state machine terminates and yields an Err after retries are exhausted.
        #[derive(Debug)]
        struct InfiniteDisconnectClient;

        impl google_cloud_bigquery::stub::Read for InfiniteDisconnectClient {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                let _ = tx
                    .send(Err(Error::service(
                        Status::default()
                            .set_code(Code::Unavailable)
                            .set_message("persistent disconnect"),
                    )))
                    .await;
                Ok(ResponseStream::from(rx))
            }
        }

        let client = Read::from_stub(InfiniteDisconnectClient);
        let read_rows = client.read_rows().set_read_stream("test_stream");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);
        read_stream(ReadStreamOptions {
            read_rows,
            current_offset: 0,
            schema: None,
            schema_tx: None,
            first_page_check: None,
            tx,
            retry_config: test_retry_config(3),
            maintain_order: false,
            cancel_tx: Arc::clone(&cancel_tx),
            cancel_rx,
        })
        .await;

        let result = rx.recv().await.unwrap();
        assert!(matches!(result, Err(BigQueryError::Grpc(_))));
        assert!(
            *cancel_tx.borrow(),
            "fatal stream error should signal sibling cancellation"
        );
    }

    #[tokio::test]
    async fn test_stream_cancellation_terminates_blocked_sibling() {
        #[derive(Debug)]
        struct HangingClient {
            _hold_tx: Arc<
                Mutex<
                    Option<
                        tokio::sync::mpsc::Sender<google_cloud_bigquery::Result<ReadRowsResponse>>,
                    >,
                >,
            >,
        }

        impl google_cloud_bigquery::stub::Read for HangingClient {
            async fn read_rows(
                &self,
                _req: ReadRowsRequest,
                _options: RequestOptions,
            ) -> google_cloud_bigquery::Result<ResponseStream<ReadRowsResponse>> {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                *self._hold_tx.lock().unwrap() = Some(tx);
                Ok(ResponseStream::from(rx))
            }
        }

        let hold_tx = Arc::new(Mutex::new(None));
        let client = Read::from_stub(HangingClient {
            _hold_tx: Arc::clone(&hold_tx),
        });
        let read_rows = client.read_rows().set_read_stream("hanging_stream");

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let cancel_tx = Arc::new(cancel_tx);

        let task = tokio::spawn(read_stream(ReadStreamOptions {
            read_rows,
            current_offset: 0,
            schema: None,
            schema_tx: None,
            first_page_check: None,
            tx,
            retry_config: test_retry_config(3),
            maintain_order: false,
            cancel_tx: Arc::clone(&cancel_tx),
            cancel_rx,
        }));

        // Signal cancellation as a sibling stream would upon fatal error.
        let _ = cancel_tx.send_replace(true);

        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("cancelled stream should exit promptly")
            .expect("task should not panic");
    }

    fn transient_error() -> Error {
        Error::service(
            Status::default()
                .set_code(Code::Unavailable)
                .set_message("transient failure"),
        )
    }

    fn permanent_error() -> Error {
        Error::service(
            Status::default()
                .set_code(Code::PermissionDenied)
                .set_message("permission denied"),
        )
    }

    type TestReadStream = (
        ReadStream,
        Arc<Mutex<Vec<ReadRowsRequest>>>,
        tokio::sync::mpsc::Receiver<Result<RecordBatch, BigQueryError>>,
    );

    fn make_test_read_stream(
        streams: Vec<MockStreamResult>,
        schema: Option<SchemaRef>,
        retry_config: StreamRetryConfig,
    ) -> TestReadStream {
        let requests = Arc::new(Mutex::new(vec![]));
        let mock_client = MockClient {
            requests: Arc::clone(&requests),
            streams: Arc::new(Mutex::new(streams)),
        };
        let client = Read::from_stub(mock_client);
        let read_rows = client.read_rows().set_read_stream("streams/1");
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let decode_queue = DecodeQueue::new(false);
        let reader = ReadStream::new(
            ReadStreamOptions {
                read_rows,
                current_offset: 0,
                schema,
                schema_tx: None,
                first_page_check: None,
                tx,
                retry_config,
                maintain_order: false,
                cancel_tx: Arc::new(cancel_tx),
                cancel_rx,
            },
            decode_queue,
        );
        (reader, requests, rx)
    }

    #[tokio::test]
    async fn step_connecting_to_reading() {
        let (mut reader, requests, _rx) =
            make_test_read_stream(vec![Ok(vec![])], None, test_retry_config(5));

        let initial_retry = RetryState::new(true).set_attempt_count(2_u32);
        reader.state = ReadStreamState::Connecting {
            offset: 42,
            retry_state: initial_retry,
        };
        reader.step().await;

        let ReadStreamState::Reading {
            offset,
            retry_state: Some(retry_state),
            ..
        } = &reader.state
        else {
            panic!("expected Reading state with Some(retry_state)");
        };
        assert_eq!(*offset, 42);
        assert_eq!(retry_state.attempt_count, 2);
        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].offset, 42);
    }

    #[tokio::test]
    async fn step_connecting_retryable_error() {
        let (mut reader, _requests, _rx) =
            make_test_read_stream(vec![Err(transient_error())], None, test_retry_config(5));

        let initial_retry = RetryState::new(true).set_attempt_count(1_u32);
        reader.state = ReadStreamState::Connecting {
            offset: 10,
            retry_state: initial_retry,
        };
        let before = Instant::now();
        reader.step().await;

        let ReadStreamState::Backoff {
            offset,
            retry_state,
            deadline,
        } = &reader.state
        else {
            panic!("expected Backoff state");
        };
        assert_eq!(*offset, 10);
        assert_eq!(retry_state.attempt_count, 2);
        assert!(*deadline >= before);
    }

    #[tokio::test]
    async fn step_connecting_permanent_error() {
        let (mut reader, _requests, mut rx) =
            make_test_read_stream(vec![Err(permanent_error())], None, test_retry_config(5));

        reader.state = ReadStreamState::Connecting {
            offset: 0,
            retry_state: RetryState::new(true),
        };
        reader.step().await;

        assert!(matches!(
            reader.state,
            ReadStreamState::Terminated(TerminationReason::Finished)
        ));
        let err = rx.recv().await.unwrap().unwrap_err();
        let BigQueryError::Grpc(gax_err) = err else {
            panic!("expected Grpc error, got: {err:?}");
        };
        assert_eq!(
            gax_err.status().map(|s| s.code),
            Some(Code::PermissionDenied)
        );
    }

    #[tokio::test]
    async fn step_connecting_deadline_exhausted() {
        use google_cloud_gax::retry_throttler::{CircuitBreaker, RetryThrottlerArg};

        #[derive(Debug)]
        struct FixedBackoff(Duration);
        impl BackoffPolicy for FixedBackoff {
            fn on_failure(&self, _state: &RetryState) -> Duration {
                self.0
            }
        }

        let retry_config = StreamRetryConfig::new(
            Some(Arc::new(
                RetryableErrors.with_time_limit(Duration::from_secs(1)),
            )),
            Some(Arc::new(FixedBackoff(Duration::from_secs(60)))),
            RetryThrottlerArg::from(CircuitBreaker::default()).into(),
        );

        let (mut reader, _requests, mut rx) =
            make_test_read_stream(vec![Err(transient_error())], None, retry_config);

        reader.state = ReadStreamState::Connecting {
            offset: 0,
            retry_state: RetryState::new(true),
        };
        reader.step().await;

        assert!(matches!(
            reader.state,
            ReadStreamState::Terminated(TerminationReason::Finished)
        ));
        let err = rx.recv().await.unwrap().unwrap_err();
        let BigQueryError::Grpc(gax_err) = err else {
            panic!("expected Grpc error, got: {err:?}");
        };
        assert!(gax_err.is_exhausted(), "{gax_err:?}");
    }

    #[tokio::test]
    async fn step_backoff_to_connecting() {
        let (mut reader, requests, _rx) = make_test_read_stream(vec![], None, test_retry_config(5));

        reader.state = ReadStreamState::Backoff {
            offset: 42,
            retry_state: RetryState::new(true).set_attempt_count(2_u32),
            deadline: Instant::now(),
        };
        reader.step().await;

        let ReadStreamState::Connecting {
            offset,
            retry_state,
        } = &reader.state
        else {
            panic!("expected Connecting state");
        };
        assert_eq!(*offset, 42);
        assert_eq!(retry_state.attempt_count, 2);
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn step_first_message_clears_retry_state() {
        let (schema_bytes, resp) = create_test_arrow_payload(5);
        let schema = parse_arrow_schema(&schema_bytes).unwrap();
        let (mut reader, _requests, mut rx) =
            make_test_read_stream(vec![], Some(schema), test_retry_config(5));

        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(1);
        stream_tx.send(Ok(resp)).await.unwrap();
        let stream = ResponseStream::from(stream_rx);

        reader.state = ReadStreamState::Reading {
            offset: 10,
            retry_state: Some(RetryState::new(true).set_attempt_count(3_u32)),
            stream,
        };
        reader.step().await;

        let ReadStreamState::Reading {
            offset,
            retry_state,
            ..
        } = &reader.state
        else {
            panic!("expected Reading state");
        };
        assert_eq!(*offset, 15);
        assert!(
            retry_state.is_none(),
            "expected retry_state to be cleared to None after first message, got: {retry_state:?}"
        );
        let batch = rx.recv().await.expect("expected decoded batch").unwrap();
        assert_eq!(batch.num_rows(), 5);
    }

    #[tokio::test]
    async fn step_first_message_retryable_error_retains_retry_state() {
        let (mut reader, _requests, _rx) =
            make_test_read_stream(vec![], None, test_retry_config(5));

        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(1);
        stream_tx.send(Err(transient_error())).await.unwrap();
        let stream = ResponseStream::from(stream_rx);

        let prior_retry = RetryState::new(true).set_attempt_count(2_u32);
        reader.state = ReadStreamState::Reading {
            offset: 7,
            retry_state: Some(prior_retry),
            stream,
        };
        reader.step().await;

        let ReadStreamState::Backoff {
            offset,
            retry_state,
            ..
        } = &reader.state
        else {
            panic!("expected Backoff state");
        };
        assert_eq!(*offset, 7);
        assert_eq!(retry_state.attempt_count, 3);
    }

    #[tokio::test]
    async fn step_reading_to_reading() {
        let (schema_bytes, resp) = create_test_arrow_payload(3);
        let schema = parse_arrow_schema(&schema_bytes).unwrap();
        let (mut reader, _requests, mut rx) =
            make_test_read_stream(vec![], Some(schema), test_retry_config(5));

        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(1);
        stream_tx.send(Ok(resp)).await.unwrap();
        let stream = ResponseStream::from(stream_rx);

        reader.state = ReadStreamState::Reading {
            offset: 5,
            retry_state: None,
            stream,
        };
        reader.step().await;

        let ReadStreamState::Reading {
            offset,
            retry_state,
            ..
        } = &reader.state
        else {
            panic!("expected Reading state");
        };
        assert_eq!(*offset, 8);
        assert!(retry_state.is_none());
        let batch = rx.recv().await.expect("expected decoded batch").unwrap();
        assert_eq!(batch.num_rows(), 3);
    }

    #[tokio::test]
    async fn step_reading_retryable_error_resets_retry_state() {
        let (mut reader, _requests, _rx) =
            make_test_read_stream(vec![], None, test_retry_config(5));

        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(1);
        stream_tx.send(Err(transient_error())).await.unwrap();
        let stream = ResponseStream::from(stream_rx);

        reader.state = ReadStreamState::Reading {
            offset: 12,
            retry_state: None,
            stream,
        };
        reader.step().await;

        let ReadStreamState::Backoff {
            offset,
            retry_state,
            ..
        } = &reader.state
        else {
            panic!("expected Backoff state");
        };
        assert_eq!(*offset, 12);
        assert_eq!(retry_state.attempt_count, 1);
    }

    #[tokio::test]
    async fn step_reading_permanent_error_to_terminated() {
        let (mut reader, _requests, mut rx) =
            make_test_read_stream(vec![], None, test_retry_config(5));

        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(1);
        stream_tx.send(Err(permanent_error())).await.unwrap();
        let stream = ResponseStream::from(stream_rx);

        reader.state = ReadStreamState::Reading {
            offset: 0,
            retry_state: None,
            stream,
        };
        reader.step().await;

        assert!(matches!(
            reader.state,
            ReadStreamState::Terminated(TerminationReason::Finished)
        ));
        let err = rx.recv().await.unwrap().unwrap_err();
        let BigQueryError::Grpc(gax_err) = err else {
            panic!("expected Grpc error, got: {err:?}");
        };
        assert_eq!(
            gax_err.status().map(|s| s.code),
            Some(Code::PermissionDenied)
        );
    }

    #[tokio::test]
    async fn step_reading_eof_to_terminated() {
        let (mut reader, _requests, _rx) =
            make_test_read_stream(vec![], None, test_retry_config(5));

        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(1);
        drop(stream_tx);
        let stream = ResponseStream::from(stream_rx);

        reader.state = ReadStreamState::Reading {
            offset: 0,
            retry_state: None,
            stream,
        };
        reader.step().await;

        assert!(matches!(
            reader.state,
            ReadStreamState::Terminated(TerminationReason::Finished)
        ));
    }

    #[tokio::test]
    async fn step_terminated_stays_terminated() {
        let (mut reader, requests, _rx) = make_test_read_stream(vec![], None, test_retry_config(5));

        reader.state = ReadStreamState::Terminated(TerminationReason::Finished);
        reader.step().await;
        assert!(matches!(
            reader.state,
            ReadStreamState::Terminated(TerminationReason::Finished)
        ));

        reader.state = ReadStreamState::Terminated(TerminationReason::SwitchToReadSession);
        reader.step().await;
        assert!(matches!(
            reader.state,
            ReadStreamState::Terminated(TerminationReason::SwitchToReadSession)
        ));
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn retry_throttler_stops_retries_when_exhausted() {
        use google_cloud_gax::retry_throttler::{RetryThrottler, RetryThrottlerArg};

        #[derive(Debug)]
        struct AlwaysThrottle;
        impl RetryThrottler for AlwaysThrottle {
            fn throttle_retry_attempt(&self) -> bool {
                true
            }
            fn on_retry_failure(&mut self, _flow: &RetryResult) {}
            fn on_success(&mut self) {}
        }

        // With attempt_limit(2):
        // - handle_error increments attempt_count to 1 -> on_error returns Continue
        // - throttler throttles attempt -> attempt_count incremented to 2 -> on_throttle returns Exhausted
        let retry_config = StreamRetryConfig::new(
            Some(Arc::new(RetryableErrors.with_attempt_limit(2))),
            Some(Arc::new(NoBackoff)),
            RetryThrottlerArg::from(AlwaysThrottle).into(),
        );

        let (mut reader, requests, mut rx) =
            make_test_read_stream(vec![Err(transient_error())], None, retry_config);

        reader.step().await;
        assert!(matches!(
            reader.state,
            ReadStreamState::Terminated(TerminationReason::Finished)
        ));
        assert_eq!(requests.lock().unwrap().len(), 1);
        let err = rx.recv().await.unwrap().unwrap_err();
        let BigQueryError::Grpc(gax_err) = err else {
            panic!("expected Grpc error, got: {err:?}");
        };
        assert_eq!(gax_err.status().map(|s| s.code), Some(Code::Unavailable));
    }
}
