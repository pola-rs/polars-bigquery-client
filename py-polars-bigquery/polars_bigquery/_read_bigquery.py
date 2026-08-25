from __future__ import annotations

from typing import Any, Dict

import polars as pl

import polars_bigquery.core.version
import arrow_bigquery
from .core.run_query import run_query


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"polars-bigquery/{polars_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


def _parse_table_id(table_id: Any) -> str:
    if not isinstance(table_id, str):
        if (
            hasattr(table_id, "project")
            and hasattr(table_id, "dataset_id")
            and hasattr(table_id, "table_id")
        ):
            return f"{table_id.project}.{table_id.dataset_id}.{table_id.table_id}"
        raise TypeError(f"Expected table_id to be a string, got {type(table_id)}")

    parts = table_id.split(".")
    if len(parts) < 3:
        raise ValueError("Invalid table ID")
    if len(parts) > 3 and not any(":" in part for part in parts[:-2]):
        raise TypeError("BigLake tables are not supported yet")

    # Let's just follow the rust regex logic:
    # it must have at least two dots, and the last two parts must not have dots.
    if len(parts) >= 3:
        return table_id

    raise ValueError("Invalid table ID")


class Client:
    """Client for reading data from BigQuery into Polars.

    Wraps an arrow_bigquery Client to keep connections open and reuse credentials across operations.
    """

    def __init__(
        self,
        *,
        credentials_provider: pl.CredentialProviderGCP | None = None,
        user_agent: str | None = None,
    ) -> None:
        if credentials_provider is None:
            credentials_provider = pl.CredentialProviderGCP()
        self._credentials_provider = credentials_provider
        self._user_agent = user_agent
        self._arrow_client = arrow_bigquery.Client(
            credentials_provider=credentials_provider,
            user_agent=_get_user_agent(user_agent),
        )

    @property
    def credentials_provider(self) -> pl.CredentialProviderGCP:
        return self._credentials_provider

    def read_bigquery_table(
        self,
        table: Any,
        *,
        quota_project_id: str,
        maintain_order: bool = False,
        user_agent: str | None = None,
    ) -> pl.DataFrame:
        user_agent = _get_user_agent(user_agent or self._user_agent)
        table_ref = _parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_bigquery_table(
            table_ref,
            quota_project_id=quota_project_id,
            maintain_order=maintain_order,
            user_agent=user_agent,
        )
        return pl.DataFrame(arrow_stream_exporter)

    read_table = read_bigquery_table

    def read_bigquery_query(
        self,
        query: str,
        *,
        quota_project_id: str,
        maintain_order: bool = False,
        user_agent: str | None = None,
    ) -> pl.DataFrame:
        user_agent = _get_user_agent(user_agent or self._user_agent)
        table = run_query(
            query,
            quota_project_id,
            self._credentials_provider,
            user_agent=user_agent,
        )
        table_ref = _parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_bigquery_table(
            table_ref,
            quota_project_id=quota_project_id,
            maintain_order=maintain_order,
            user_agent=user_agent,
        )
        return pl.DataFrame(arrow_stream_exporter)

    read_query = read_bigquery_query

    def scan_bigquery_table(
        self,
        table: Any,
        *,
        quota_project_id: str,
        user_agent: str | None = None,
    ) -> pl.LazyFrame:
        user_agent = _get_user_agent(user_agent or self._user_agent)
        table_ref = _parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_bigquery_table(
            table_ref,
            quota_project_id=quota_project_id,
            maintain_order=False,
            user_agent=user_agent,
        )
        return pl.scan_arrow_c_stream(arrow_stream_exporter)

    scan_table = scan_bigquery_table


def read_bigquery_table(
    table: Any,
    *,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP | None = None,
    maintain_order: bool = False,
    user_agent: str | None = None,
) -> pl.DataFrame:
    client = Client(credentials_provider=credentials_provider, user_agent=user_agent)
    return client.read_bigquery_table(
        table,
        quota_project_id=quota_project_id,
        maintain_order=maintain_order,
        user_agent=user_agent,
    )


def read_bigquery_query(
    query: str,
    *,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP | None = None,
    maintain_order: bool = False,
    user_agent: str | None = None,
) -> pl.DataFrame:
    client = Client(credentials_provider=credentials_provider, user_agent=user_agent)
    return client.read_bigquery_query(
        query,
        quota_project_id=quota_project_id,
        maintain_order=maintain_order,
        user_agent=user_agent,
    )


def scan_bigquery_table(
    table: Any,
    *,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP | None = None,
    user_agent: str | None = None,
) -> pl.LazyFrame:
    client = Client(credentials_provider=credentials_provider, user_agent=user_agent)
    return client.scan_bigquery_table(
        table,
        quota_project_id=quota_project_id,
        user_agent=user_agent,
    )
