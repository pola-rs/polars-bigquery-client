"""Utilities for running a query in BigQuery and getting the results as a table.

TODO(tswast): Implement this in Rust using jobs.query and the
JOB_CREATION_OPTIONAL parameter to improve latency in small query results.
"""

from __future__ import annotations

import contextlib
import time
import uuid
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


def create_resilient_session() -> requests.Session:
    """Create a requests.Session with connection pooling and exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
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
        response = http.get(
            _get_jobs_get_url(job_id, quota_project_id, location=location),
            headers=headers,
            timeout=timeout,
        )
        _raise_for_bigquery_error(response)
        job = response.json()

        if job["status"]["state"] == "DONE":
            if "errorResult" in job["status"]:
                raise polars_bigquery.exceptions.BigQueryError(
                    job["status"]["errorResult"]["message"]
                )
            return job

        # jobs.getQueryResults waits about 10s or until the query finishes.
        poll_start = time.monotonic()
        poll_response = http.get(
            _get_query_results_url(job_id, quota_project_id, location=location),
            headers=headers,
            timeout=timeout,
        )
        _raise_for_bigquery_error(poll_response)
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
    response = http.post(insert_url, headers=headers, json=body, timeout=timeout)
    _raise_for_bigquery_error(response)
    job_resource = response.json()
    job_ref = job_resource.get("jobReference", {})
    job_id = job_ref["jobId"]
    location = job_ref.get("location")

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
    dest = job["configuration"]["query"]["destinationTable"]
    return f"{dest['projectId']}.{dest['datasetId']}.{dest['tableId']}"


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
    response = http.get(table_metadata_url, headers=headers, timeout=timeout)
    _raise_for_bigquery_error(response)
    return response.json()


class BigQueryRestClient:
    """Stateful BigQuery REST client with connection pooling and metadata caching."""

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
