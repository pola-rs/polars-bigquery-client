"""Utilities for fetching table metadata from the BigQuery REST API."""

from __future__ import annotations

import contextlib
import inspect
import urllib.parse
from typing import Any

import arrow_bigquery.api.resources
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import polars_bigquery.exceptions

_BIGQUERY_ENDPOINT = "https://bigquery.googleapis.com/bigquery/v2"
# Keep query timeout at least as long as the server-side timeout defined at
# https://github.com/googleapis/googleapis/blob/4bcbf04e688ffddb8ec1a20a2349df32b9fcbd9c/google/cloud/bigquery/v2/bigquery_grpc_service_config.json#L170-L181
_CONNECT_TIMEOUT = 60.0
_READ_TIMEOUT = 250.0
DEFAULT_TIMEOUT: tuple[float, float] = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
# Lightweight timeout for `tables.get` schema metadata lookups so a stalled
# endpoint fails fast rather than blocking for 30+ minutes across retries.
METADATA_TIMEOUT: tuple[float, float] = (10.0, 30.0)
# Upper bound (in seconds) on a single `Retry-After` sleep so a `429`/`503` response
# (or clock-skewed proxy) cannot hang callers for hours (`urllib3` 2.7+ defaults
# `retry_after_max` to 6 hours; older versions impose no cap at all).
MAX_RETRY_AFTER_SECONDS: int = 60
_RETRY_SUPPORTS_BACKOFF_JITTER = "backoff_jitter" in inspect.signature(Retry).parameters
_RETRY_SUPPORTS_RETRY_AFTER_MAX = (
    "retry_after_max" in inspect.signature(Retry).parameters
)


class _BoundedRetry(Retry):
    """`urllib3.util.retry.Retry` that caps parsed `Retry-After` sleeps."""

    def parse_retry_after(self, retry_after: str) -> float:
        return min(
            float(super().parse_retry_after(retry_after)),
            float(MAX_RETRY_AFTER_SECONDS),
        )


def create_resilient_session() -> requests.Session:
    """Create a requests.Session with connection pooling and jittered exponential backoff."""
    session = requests.Session()
    retry_kwargs: dict[str, Any] = {
        "total": 5,
        "backoff_factor": 0.5,
        "status_forcelist": (429, 500, 502, 503, 504),
        # Restrict automatic retries to safe, idempotent GET requests so non-idempotent
        # mutations (e.g. POST datasets.insert / tables.insert) are never duplicated.
        "allowed_methods": frozenset(["GET"]),
        "respect_retry_after_header": True,
    }
    if _RETRY_SUPPORTS_BACKOFF_JITTER:
        retry_kwargs["backoff_jitter"] = 0.5
    if _RETRY_SUPPORTS_RETRY_AFTER_MAX:
        retry_kwargs["retry_after_max"] = MAX_RETRY_AFTER_SECONDS
    retry_strategy = _BoundedRetry(**retry_kwargs)
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


def _quote_path_segment(segment: str, *, name: str) -> str:
    """Percent-encode a single BigQuery REST URL path segment."""
    if not isinstance(segment, str):
        raise TypeError(f"Expected {name} to be a str, got {type(segment).__name__}")
    if segment in ("", ".", ".."):
        raise ValueError(f"Invalid {name} for URL path segment: {segment!r}")
    return urllib.parse.quote(segment, safe="")


def _get_table_metadata_url(
    *, table_ref: arrow_bigquery.api.resources.BigQueryTableId
) -> str:
    project_id = _quote_path_segment(table_ref.project_id, name="project_id")
    dataset_id = _quote_path_segment(table_ref.dataset_id, name="dataset_id")
    table_id = _quote_path_segment(table_ref.table_id, name="table_id")
    return (
        f"{_BIGQUERY_ENDPOINT}/projects/{project_id}"
        f"/datasets/{dataset_id}/tables/{table_id}"
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
    timeout: tuple[float, float] = METADATA_TIMEOUT,
) -> dict:
    http = session if session is not None else requests
    table_metadata_url = _get_table_metadata_url(table_ref=table_ref)

    def _send_get() -> requests.Response:
        try:
            headers = _get_headers(
                quota_project_id=quota_project_id,
                credentials_provider=credentials_provider,
                user_agent=user_agent,
            )
        except Exception as exc:
            msg = (
                f"Failed to obtain credentials for fetching table metadata for "
                f"'{table_ref}': {exc}"
            )
            raise polars_bigquery.exceptions.BigQueryError(msg) from exc
        try:
            return http.get(table_metadata_url, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            msg = (
                f"Failed to fetch table metadata for '{table_ref}' from BigQuery "
                f"REST API: {exc}"
            )
            raise polars_bigquery.exceptions.BigQueryError(msg) from exc

    response = _send_get()
    try:
        # Recover once from a stale/near-expiry or revoked bearer token:
        # `urllib3.Retry` does not refresh the Authorization header on 401, and
        # Polars `CachingCredentialProvider` caches credentials until expiry, so
        # invalidate the cache before retrying.
        if response.status_code == 401:
            response.close()
            clear_cache = getattr(
                credentials_provider, "clear_cached_credentials", None
            )
            if callable(clear_cache):
                clear_cache()
            response = _send_get()

        _raise_for_bigquery_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            msg = f"BigQuery REST API returned invalid JSON for table '{table_ref}': {exc}"
            raise polars_bigquery.exceptions.BigQueryError(msg) from exc
        if not isinstance(payload, dict):
            msg = (
                f"BigQuery REST API returned unexpected metadata payload type "
                f"{type(payload).__name__} for table '{table_ref}'"
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        return payload
    finally:
        response.close()


class BigQueryRestClient:
    """Stateful BigQuery REST client with connection pooling and retry handling."""

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
