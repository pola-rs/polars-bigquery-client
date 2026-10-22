"""Utilities for fetching table metadata from the BigQuery REST API."""

from __future__ import annotations

import contextlib
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
