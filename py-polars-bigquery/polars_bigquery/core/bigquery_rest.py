"""Utilities for running a query in BigQuery and getting the results as a table.

TODO(tswast): Implement this in Rust using jobs.query and the
JOB_CREATION_OPTIONAL parameter to improve latency in small query results.
"""

from __future__ import annotations

import contextlib
import random
import time
import uuid
from collections.abc import Callable
from typing import Any

import arrow_bigquery.api.resources
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import polars_bigquery.exceptions

_BIGQUERY_ENDPOINT = "https://bigquery.googleapis.com/bigquery/v2"
# Keep timeout at least as long as the server-side timeout defined at
# https://github.com/googleapis/googleapis/blob/4bcbf04e688ffddb8ec1a20a2349df32b9fcbd9c/google/cloud/bigquery/v2/bigquery_grpc_service_config.json#L170-L181
_CONNECT_TIMEOUT = 60.0
_READ_TIMEOUT = 250.0
DEFAULT_TIMEOUT: tuple[float, float] = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
MAX_POLL_SECONDS: float = 21600.0
_RATE_LIMIT_REASONS = frozenset(
    {"rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded"}
)


def create_resilient_session() -> requests.Session:
    """Create a requests.Session with connection pooling and exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        backoff_jitter=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _is_rate_limit_error(response: requests.Response) -> bool:
    """Check if an HTTP error response represents a retryable BigQuery rate limit."""
    if response.status_code == 429:
        raw = getattr(response, "raw", None)
        retries = getattr(raw, "retries", None)
        history = getattr(retries, "history", None)
        # If urllib3 already exhausted its retry budget for this 429 response, do not retry again.
        return not (isinstance(history, (tuple, list)) and len(history) > 0)
    if response.status_code == 403:
        with contextlib.suppress(ValueError, KeyError, TypeError):
            err_json = response.json()
            if isinstance(err_json, dict) and "error" in err_json:
                err_obj = err_json["error"]
                if isinstance(err_obj, dict):
                    for err_item in err_obj.get("errors", []):
                        if (
                            isinstance(err_item, dict)
                            and err_item.get("reason") in _RATE_LIMIT_REASONS
                        ):
                            return True
                    if err_obj.get("status") == "RESOURCE_EXHAUSTED":
                        return True
    return False


def _request_with_retry(
    request_fn: Callable[..., requests.Response],
    *args: Any,
    max_retries: int = 5,
    base_backoff: float = 0.5,
    **kwargs: Any,
) -> requests.Response:
    """Execute an HTTP request with exponential backoff and jitter for BigQuery rate limits."""
    attempt = 0
    while True:
        try:
            response = request_fn(*args, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise polars_bigquery.exceptions.BigQueryError(str(exc)) from exc
        if not _is_rate_limit_error(response) or attempt >= max_retries:
            return response
        attempt += 1
        sleep_seconds = base_backoff * (2 ** (attempt - 1)) + random.uniform(0.0, 0.2)
        time.sleep(sleep_seconds)


def _raise_for_bigquery_error(response: requests.Response) -> None:
    """Raise BigQueryError with structured message on HTTP error."""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        message = str(exc)
        with contextlib.suppress(ValueError, KeyError, TypeError):
            err_json = response.json()
            if isinstance(err_json, dict) and "error" in err_json:
                err_obj = err_json["error"]
                if isinstance(err_obj, dict) and "message" in err_obj:
                    message = err_obj["message"]
        raise polars_bigquery.exceptions.BigQueryError(message) from exc


def _parse_json_response(response: requests.Response) -> dict[str, Any]:
    """Safely parse a JSON dict response from the BigQuery API."""
    try:
        data = response.json()
    except (ValueError, TypeError) as exc:
        raise polars_bigquery.exceptions.BigQueryError(
            "Invalid JSON response from BigQuery API"
        ) from exc
    if not isinstance(data, dict):
        raise polars_bigquery.exceptions.BigQueryError(
            "Expected JSON object from BigQuery API"
        )
    return data


def _extract_destination_table(job: dict[str, Any]) -> str:
    """Extract destination table reference from a completed job resource."""
    dest = job.get("configuration", {}).get("query", {}).get("destinationTable")
    if (
        not isinstance(dest, dict)
        or not isinstance(dest.get("projectId"), str)
        or not isinstance(dest.get("datasetId"), str)
        or not isinstance(dest.get("tableId"), str)
    ):
        raise polars_bigquery.exceptions.BigQueryError(
            "BigQuery query job completed without a destination table."
        )
    return f"{dest['projectId']}.{dest['datasetId']}.{dest['tableId']}"


def _get_table_metadata_url(
    *, table_ref: arrow_bigquery.api.resources.BigQueryTableId
) -> str:
    return (
        f"{_BIGQUERY_ENDPOINT}/projects/{table_ref.project_id}"
        f"/datasets/{table_ref.dataset_id}/tables/{table_ref.table_id}"
        "?fields=schema,timePartitioning"
    )


def _get_jobs_insert_url(quota_project_id: str) -> str:
    return f"{_BIGQUERY_ENDPOINT}/projects/{quota_project_id}/jobs"


def _get_jobs_insert_body(query: str, quota_project_id: str) -> dict:
    return {
        "jobReference": {
            "projectId": quota_project_id,
            "jobId": f"polars_bq_{uuid.uuid4().hex}",
        },
        "configuration": {
            "query": {
                "query": query,
                "useLegacySql": False,
            }
        },
    }


def _get_jobs_get_url(
    job_id: str, quota_project_id: str, location: str | None = None
) -> str:
    url = f"{_BIGQUERY_ENDPOINT}/projects/{quota_project_id}/jobs/{job_id}"
    if location:
        url += f"?location={location}"
    return url


def _get_query_results_url(
    job_id: str, quota_project_id: str, location: str | None = None
) -> str:
    # Set maxResults=0 and timeoutMs=10000 to use this endpoint to wait for job completion.
    url = (
        f"{_BIGQUERY_ENDPOINT}/projects/{quota_project_id}/queries/{job_id}/results"
        "?maxResults=0&timeoutMs=10000"
    )
    if location:
        url += f"&location={location}"
    return url


def _wait_for_job(
    job_id: str,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP,
    user_agent: str,
    *,
    location: str | None = None,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_poll_seconds: float = MAX_POLL_SECONDS,
) -> dict:
    http = session if session is not None else requests
    start_time = time.monotonic()
    while True:
        if time.monotonic() - start_time > max_poll_seconds:
            raise polars_bigquery.exceptions.BigQueryError(
                f"Timed out waiting for BigQuery job {job_id} after {max_poll_seconds}s"
            )
        headers = _get_headers(
            quota_project_id=quota_project_id,
            credentials_provider=credentials_provider,
            user_agent=user_agent,
        )
        response = _request_with_retry(
            http.get,
            _get_jobs_get_url(job_id, quota_project_id, location=location),
            headers=headers,
            timeout=timeout,
        )
        _raise_for_bigquery_error(response)
        job = _parse_json_response(response)

        status = job.get("status")
        if isinstance(status, dict) and status.get("state") == "DONE":
            if "errorResult" in status and isinstance(status["errorResult"], dict):
                raise polars_bigquery.exceptions.BigQueryError(
                    status["errorResult"].get("message", "BigQuery job failed")
                )
            return job

        # jobs.getQueryResults waits about 10s or until the query finishes.
        poll_start = time.monotonic()
        poll_response = _request_with_retry(
            http.get,
            _get_query_results_url(job_id, quota_project_id, location=location),
            headers=headers,
            timeout=timeout,
        )
        _raise_for_bigquery_error(poll_response)
        poll_data = _parse_json_response(poll_response)
        if poll_data.get("jobComplete") is True:
            errors = poll_data.get("errors")
            if isinstance(errors, list) and errors and isinstance(errors[0], dict):
                err_msg = errors[0].get("message", "BigQuery job failed")
                raise polars_bigquery.exceptions.BigQueryError(err_msg)
            if "destinationTable" in job.get("configuration", {}).get("query", {}):
                return job

        # Avoid busy-spinning if the polling endpoint returns without blocking
        elapsed = time.monotonic() - poll_start
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)


def _get_headers(
    *,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP,
    user_agent: str,
) -> dict:
    token_data, _ = credentials_provider()
    token = token_data["bearer_token"]
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": user_agent,
        "Content-Type": "application/json",
        "x-goog-user-project": quota_project_id,
    }


def run_query(
    query: str,
    *,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP,
    user_agent: str,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> str:
    """Run a query and return the destination table from the job resource."""
    http = session if session is not None else requests
    headers = _get_headers(
        quota_project_id=quota_project_id,
        credentials_provider=credentials_provider,
        user_agent=user_agent,
    )

    # 1. Insert the job
    insert_url = _get_jobs_insert_url(quota_project_id)
    body = _get_jobs_insert_body(query, quota_project_id)
    job_id = body["jobReference"]["jobId"]
    response = _request_with_retry(
        http.post, insert_url, headers=headers, json=body, timeout=timeout
    )
    if response.status_code == 409:
        # Job already exists from an earlier retry attempt; proceed to wait for completion.
        location = None
    else:
        _raise_for_bigquery_error(response)
        job_resource = _parse_json_response(response)

        status = job_resource.get("status")
        if isinstance(status, dict) and status.get("state") == "DONE":
            if "errorResult" in status and isinstance(status["errorResult"], dict):
                raise polars_bigquery.exceptions.BigQueryError(
                    status["errorResult"].get("message", "BigQuery job failed")
                )
            return _extract_destination_table(job_resource)

        job_ref = job_resource.get("jobReference")
        if isinstance(job_ref, dict):
            if isinstance(job_ref.get("jobId"), str):
                job_id = job_ref["jobId"]
            location = job_ref.get("location")
        else:
            location = None

    # 2. Wait for the job to complete (refreshing credentials on each poll)
    job = _wait_for_job(
        job_id,
        quota_project_id,
        credentials_provider,
        user_agent,
        location=location,
        session=session,
        timeout=timeout,
    )

    # 3. Return the destination table ID
    return _extract_destination_table(job)


def get_table_metadata(
    table_ref: arrow_bigquery.api.resources.BigQueryTableId,
    *,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP,
    user_agent: str,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> dict:
    http = session if session is not None else requests
    headers = _get_headers(
        quota_project_id=quota_project_id,
        credentials_provider=credentials_provider,
        user_agent=user_agent,
    )
    table_metadata_url = _get_table_metadata_url(table_ref=table_ref)
    response = _request_with_retry(
        http.get, table_metadata_url, headers=headers, timeout=timeout
    )
    _raise_for_bigquery_error(response)
    return _parse_json_response(response)


class BigQueryRestClient:
    """Stateful BigQuery REST client with connection pooling."""

    def __init__(
        self,
        *,
        quota_project_id: str,
        credentials_provider: pl.CredentialProviderGCP,
        user_agent: str,
        session: requests.Session | None = None,
    ) -> None:
        self._quota_project_id = quota_project_id
        self._credentials_provider = credentials_provider
        self._user_agent = user_agent
        self._owns_session = session is None
        self._session = session if session is not None else create_resilient_session()

    @property
    def session(self) -> requests.Session:
        return self._session

    def close(self) -> None:
        """Close the underlying HTTP session if owned by this client."""
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> BigQueryRestClient:  # noqa: PYI034
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def run_query(self, query: str) -> str:
        return run_query(
            query,
            quota_project_id=self._quota_project_id,
            credentials_provider=self._credentials_provider,
            user_agent=self._user_agent,
            session=self._session,
        )

    def get_table_metadata(
        self,
        table_ref: arrow_bigquery.api.resources.BigQueryTableId,
    ) -> dict[str, Any]:
        return get_table_metadata(
            table_ref,
            quota_project_id=self._quota_project_id,
            credentials_provider=self._credentials_provider,
            user_agent=self._user_agent,
            session=self._session,
        )
