from __future__ import annotations

from typing import Any

import arrow_bigquery
import arrow_bigquery.api.resources
import polars as pl

import polars_bigquery.core.version

from .core.run_query import run_query


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"polars-bigquery/{polars_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


class Client:
    """Client for reading data from BigQuery into Polars.

    Wraps an arrow_bigquery Client to keep connections open and reuse credentials across operations.
    """

    def __init__(
        self,
        *,
        quota_project_id: str,
        credentials_provider: pl.CredentialProviderGCP | None = None,
        user_agent: str | None = None,
    ) -> None:
        if credentials_provider is None:
            credentials_provider = pl.CredentialProviderGCP(
                quota_project_id=quota_project_id
            )
        self._credentials_provider = credentials_provider
        self._user_agent = _get_user_agent(user_agent)
        self._quota_project_id = quota_project_id
        self._arrow_client = arrow_bigquery.Client(
            quota_project_id=quota_project_id,
            credentials_provider=credentials_provider,
            user_agent=self._user_agent,
        )

    @property
    def credentials_provider(self) -> pl.CredentialProviderGCP:
        return self._credentials_provider

    @property
    def quota_project_id(self) -> str:
        return self._quota_project_id

    def read_table(
        self,
        table: Any,
        *,
        maintain_order: bool = False,
    ) -> pl.DataFrame:
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_table(
            table_ref,
            maintain_order=maintain_order,
        )
        return pl.DataFrame(arrow_stream_exporter)

    def read_query(
        self,
        query: str,
        *,
        maintain_order: bool = False,
    ) -> pl.DataFrame:
        table = run_query(
            query,
            self._quota_project_id,
            self._credentials_provider,
            user_agent=self._user_agent,
        )
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_table(
            table_ref,
            maintain_order=maintain_order,
        )
        return pl.DataFrame(arrow_stream_exporter)

    def scan_table(
        self,
        table: Any,
    ) -> pl.LazyFrame:
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_table(
            table_ref,
            maintain_order=False,
        )
        return pl.scan_arrow_c_stream(arrow_stream_exporter)
